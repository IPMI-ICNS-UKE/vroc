from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from aim import Image
from torch.optim import Adam
from torch.utils.data import DataLoader

from vroc.blocks import SpatialTransformer
from vroc.dataset import Lung4DArtifactBoostingDataset
from vroc.helper import (
    dict_collate,
    get_bounding_box,
    pad_bounding_box_to_pow_2,
    rescale_range,
)
from vroc.logger import init_fancy_logging
from vroc.loss import TRELoss, WarpedMSELoss
from vroc.models import FlexUNet
from vroc.trainer import BaseTrainer, MetricType


class VectorFieldArtifactBoosterTrainer(BaseTrainer):
    METRICS = {
        "loss": MetricType.SMALLER_IS_BETTER,
        "tre_loss_without_artifact": MetricType.SMALLER_IS_BETTER,
        "tre_loss_with_artifact": MetricType.SMALLER_IS_BETTER,
        "tre_loss_with_artifact_boost": MetricType.SMALLER_IS_BETTER,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.spatial_transformer = SpatialTransformer(shape=None).to(DEVICE)
        self.tre_loss = TRELoss(apply_sqrt=True, reduction=None)
        self.tre_metric = TRELoss(apply_sqrt=True, reduction="mean")
        self.mse_loss = WarpedMSELoss(shape=None, edge_weighting=0.0).to(self.device)

        self.test_plot = None

        self.i_batch = 0
        self.accumulate_n_batches = 1

    def train_on_batch(self, data: dict) -> dict:
        vector_field_with_artifact = torch.as_tensor(
            data["vector_field_with_artifact"],
            dtype=torch.float32,
            device=self.device,
        )
        vector_field_without_artifact = torch.as_tensor(
            data["vector_field_without_artifact"],
            dtype=torch.float32,
            device=self.device,
        )

        fixed_artifact_mask = torch.as_tensor(
            data["fixed_artifact_mask"][None],
            dtype=torch.bool,
            device=self.device,
        )
        fixed_artifact_bbox = data["fixed_artifact_bbox"]

        moving_image = torch.as_tensor(
            data["moving_image"][None], dtype=torch.float32, device=self.device
        )
        moving_image = rescale_range(
            moving_image, input_range=(-1024, 3071), output_range=(0, 1)
        )
        fixed_image = torch.as_tensor(
            data["fixed_image"][None], dtype=torch.float32, device=self.device
        )
        fixed_image = rescale_range(
            fixed_image, input_range=(-1024, 3071), output_range=(0, 1)
        )
        fixed_image_masked_artifact = fixed_image.clone()
        fixed_image_masked_artifact[fixed_artifact_mask] = 0.0

        moving_mask = torch.as_tensor(
            data["moving_mask"][None], dtype=torch.bool, device=self.device
        )
        fixed_mask = torch.as_tensor(
            data["fixed_mask"][None], dtype=torch.bool, device=self.device
        )
        union_mask = moving_mask | fixed_mask
        moving_keypoints = torch.as_tensor(
            data["moving_keypoints"],
            dtype=torch.float32,
            device=self.device,
        )
        fixed_keypoints = torch.as_tensor(
            data["fixed_keypoints"],
            dtype=torch.float32,
            device=self.device,
        )

        roi_bbox = get_bounding_box(union_mask)
        # skip batch and channels
        roi_bbox = roi_bbox[2:]
        roi_bbox = pad_bounding_box_to_pow_2(
            roi_bbox, reference_shape=union_mask.shape[2:]
        )
        roi_slicing = (..., *roi_bbox)

        # roi_bbox = data["padded_fixed_bounding_box"]
        # roi_slicing = (..., *roi_bbox[0])
        image_spacing = torch.as_tensor([0.9765625, 0.9765625, 2.0], device=self.device)

        artifact_slicing = (..., *fixed_artifact_bbox[0])

        artifact_slicing_padding = 10
        extended_artifact_slicing = (
            ...,
            artifact_slicing[1],
            artifact_slicing[2],
            slice(
                max(artifact_slicing[3].start - artifact_slicing_padding, 0),
                artifact_slicing[3].stop + artifact_slicing_padding,
            ),
        )

        extended_artifact_mask = torch.zeros_like(fixed_artifact_mask)
        extended_artifact_mask[extended_artifact_slicing] = True

        modified_fixed_artifact_mask = torch.zeros_like(fixed_artifact_mask)
        modified_fixed_artifact_mask[extended_artifact_slicing] = True
        modified_fixed_artifact_mask[~fixed_artifact_mask] = False

        # boost the vector field
        with torch.autocast(device_type="cuda", enabled=True):
            # x_mean = vector_field_with_artifact[:, 0:1][union_mask].mean()
            # y_mean = vector_field_with_artifact[:, 1:2][union_mask].mean()
            # z_mean = vector_field_with_artifact[:, 2:3][union_mask].mean()
            #
            # vector_field_mean = torch.as_tensor(
            #     [x_mean, y_mean, z_mean], device=self.device
            # )
            # vector_field_mean = vector_field_mean.resize(1, 3, 1, 1, 1)
            # x_std = vector_field_with_artifact[:, 0:1][union_mask].std()
            # y_std = vector_field_with_artifact[:, 1:2][union_mask].std()
            # z_std = vector_field_with_artifact[:, 2:3][union_mask].std()
            #
            # vector_field_std = torch.as_tensor(
            #     [x_std, y_std, z_std], device=self.device
            # )
            # vector_field_std = vector_field_std.resize(1, 3, 1, 1, 1)
            #
            # # normed_vector_field_with_artifact = (
            # #     vector_field_with_artifact - vector_field_mean
            # # ) / vector_field_std

            warped_moving_image = self.spatial_transformer(
                moving_image, vector_field_with_artifact
            )

            inputs = torch.concat(
                (
                    # normed_vector_field_with_artifact,
                    # moving_image,  # moving = artifact-free
                    fixed_image_masked_artifact,
                    warped_moving_image,
                    fixed_artifact_mask.to(torch.float32),
                ),
                dim=1,
            )

            vector_field_boost = self.model(inputs[roi_slicing])
            if not torch.isfinite(vector_field_boost).all():
                raise RuntimeError()

            c = 25
            vector_field_boost = c * F.softsign(vector_field_boost)

            boosted_vector_field = vector_field_with_artifact.clone()
            boosted_vector_field[
                roi_slicing
            ] = self.spatial_transformer.compose_vector_fields(
                vector_field_1=vector_field_with_artifact[roi_slicing],
                vector_field_2=vector_field_boost,
            )

            # # restrict boost to -c, +c maximum correction
            # c = 3
            # vector_field_boost = c * F.softsign(vector_field_boost)
            # vector_field_boost = (
            #     vector_field_boost * vector_field_std + vector_field_mean
            # )
            #
            # # boosted_vector_field = vector_field_with_artifact.clone()
            # # boosted_vector_field[roi_slicing] += vector_field_boost
            #
            # full_vector_field_boost = torch.zeros_like(vector_field_with_artifact)
            # full_vector_field_boost[roi_slicing] = vector_field_boost
            #
            # # additive
            # # boosted_vector_field = vector_field_with_artifact.clone()
            # # boosted_vector_field[artifact_slicing] += full_vector_field_boost[
            # #     artifact_slicing
            # # ]
            #
            # boosted_vector_field = vector_field_with_artifact.clone()
            # boosted_vector_field[extended_artifact_slicing] = full_vector_field_boost[
            #     extended_artifact_slicing
            # ]

        #     mse_loss_total = self.mse_loss(
        #         moving_image, boosted_vector_field, fixed_image, fixed_mask
        #     )
        #
        #
        #
        #
        #     mse_loss_artifact = self.mse_loss(
        #         moving_image,
        #         boosted_vector_field,
        #         fixed_image,
        #         modified_fixed_artifact_mask,
        #     )
        #
        #     mse_loss_vector_field = F.mse_loss(
        #         boosted_vector_field[extended_artifact_slicing],
        #         vector_field_without_artifact[extended_artifact_slicing],
        #     )
        #
        #     tre_loss_without_artifact = self.tre_loss(
        #         vector_field_without_artifact,
        #         moving_keypoints,
        #         fixed_keypoints,
        #         image_spacing,
        #     )
        #     tre_loss_with_artifact = self.tre_loss(
        #         vector_field_with_artifact,
        #         moving_keypoints,
        #         fixed_keypoints,
        #         image_spacing,
        #     )
        #     tre_loss_with_artifact_boost = self.tre_loss(
        #         boosted_vector_field,
        #         moving_keypoints,
        #         fixed_keypoints,
        #         image_spacing,
        #     )
        #
        #     _, keypoint_mask = mask_keypoints(
        #         fixed_keypoints, extended_artifact_slicing[1:]
        #     )
        #
        #     tre_loss_with_artifact_roi = self.tre_loss(
        #         vector_field_with_artifact,
        #         moving_keypoints[0, keypoint_mask],
        #         fixed_keypoints[0, keypoint_mask],
        #         image_spacing,
        #     )
        #
        #     tre_loss_with_artifact_boost_roi = self.tre_loss(
        #         boosted_vector_field,
        #         moving_keypoints[0, keypoint_mask],
        #         fixed_keypoints[0, keypoint_mask],
        #         image_spacing,
        #     )
        #
        # tre_loss_total = (tre_loss_with_artifact_boost - tre_loss_with_artifact).mean()
        # tre_loss_artifact = (
        #     tre_loss_with_artifact_boost_roi - tre_loss_with_artifact_roi
        # ).mean()
        #
        # # loss = tre_loss + mse_loss
        # loss = (
        #     0.2 * mse_loss_total
        #     + 0.8 * mse_loss_artifact
        #     + mse_loss_vector_field * 0.01
        #     + tre_loss_total * 0.001
        #     + tre_loss_artifact * 0.005
        # )

        mse_image_artifact_before = self.mse_loss(
            moving_image,
            vector_field_with_artifact,
            fixed_image,
            modified_fixed_artifact_mask,
        )
        mse_image_artifact_after = self.mse_loss(
            moving_image,
            boosted_vector_field,
            fixed_image,
            modified_fixed_artifact_mask,
        )

        mse_vector_field_artifact_before = F.mse_loss(
            vector_field_with_artifact[..., extended_artifact_mask[0, 0]],
            vector_field_without_artifact[..., extended_artifact_mask[0, 0]],
        )
        mse_vector_field_artifact_after = F.mse_loss(
            boosted_vector_field[..., extended_artifact_mask[0, 0]],
            vector_field_without_artifact[..., extended_artifact_mask[0, 0]],
        )

        mse_vector_field_non_artifact_before = F.mse_loss(
            vector_field_with_artifact[..., ~extended_artifact_mask[0, 0]],
            vector_field_without_artifact[..., ~extended_artifact_mask[0, 0]],
        )
        mse_vector_field_non_artifact_after = F.mse_loss(
            boosted_vector_field[..., ~extended_artifact_mask[0, 0]],
            vector_field_without_artifact[..., ~extended_artifact_mask[0, 0]],
        )

        mse_image_before = self.mse_loss(
            moving_image, vector_field_with_artifact, fixed_image, fixed_mask
        )
        mse_image_after = self.mse_loss(
            moving_image, boosted_vector_field, fixed_image, fixed_mask
        )

        # mse_vector_field_before = F.mse_loss(
        #     vector_field_with_artifact[extended_artifact_slicing],
        #     vector_field_without_artifact[extended_artifact_slicing],
        # )
        # mse_vector_field_after = F.mse_loss(
        #     boosted_vector_field[extended_artifact_slicing],
        #     vector_field_without_artifact[extended_artifact_slicing],
        # )
        #
        tre_before = self.tre_loss(
            vector_field_with_artifact,
            moving_keypoints,
            fixed_keypoints,
            image_spacing,
        )
        tre_after = self.tre_loss(
            boosted_vector_field,
            moving_keypoints,
            fixed_keypoints,
            image_spacing,
        )
        #
        # mse_image_loss = mse_image_after / mse_image_before
        # mse_vector_field_loss = mse_vector_field_after / mse_vector_field_before
        tre_loss = (tre_after / tre_before).mean()

        mse_image_artifact = mse_image_artifact_after / mse_image_artifact_before
        mse_image_non_artifact = mse_image_after / mse_image_before

        mse_image_loss = 0.8 * mse_image_artifact + 0.2 * mse_image_non_artifact
        # mse_vector_field_loss = mse_vector_field_after / mse_vector_field_before

        mse_vector_field_artifact_loss = (
            mse_vector_field_artifact_after / mse_vector_field_artifact_before
        )
        mse_vector_field_non_artifact_loss = (
            mse_vector_field_non_artifact_after / mse_vector_field_non_artifact_before
        )

        # loss = (
        #     0.45 * mse_image_loss +
        #     0.45 * mse_vector_field_artifact_loss +
        #     0.10 * mse_vector_field_non_artifact_loss
        # )
        mse_image_loss = 0.8 * mse_image_artifact + 0.2 * mse_image_non_artifact
        mse_vector_field_loss = (
            mse_vector_field_artifact_after / mse_vector_field_artifact_before
        )
        tre_loss = (tre_after / tre_before).mean()

        loss = mse_vector_field_loss  # + tre_loss
        # loss = loss * data["artifact_size"][0]

        self.scaler.scale(loss / self.accumulate_n_batches).backward()
        self.i_batch += 1

        self.log_info(
            f"artifact size = {float(data['artifact_size'][0])}", context="TRAIN"
        )

        if self.i_batch % self.accumulate_n_batches == 0:

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.log_info("Perform optimizer step", context="TRAIN")

            self.optimizer.zero_grad()
            self.i_batch = 0

        return {
            "loss": float(loss),
            "mse_vector_field_artifact_loss": float(mse_vector_field_artifact_loss),
            "mse_vector_field_non_artifact_loss": float(
                mse_vector_field_non_artifact_loss
            ),
            "mse_image_loss": float(mse_image_loss),
            "mse_image_artifact": float(mse_image_artifact),
            "mse_image_non_artifact": float(mse_image_non_artifact),
            # "mse_vector_field_loss": float(mse_vector_field_loss),
            "tre_loss": float(tre_loss),
            # "tre_loss_total": float(tre_loss_total),
            # "tre_loss_artifact": float(tre_loss_artifact),
            # "mse_loss_total": float(mse_loss_total),
            # "mse_loss_artifact": float(mse_loss_artifact),
            # "tre_loss_without_artifact": float(tre_loss_without_artifact.mean()),
            # "tre_loss_with_artifact": float(tre_loss_with_artifact.mean()),
            # "tre_loss_with_artifact_boost": float(tre_loss_with_artifact_boost.mean()),
        }

    def validate_on_batch(self, data: dict) -> dict:
        vector_field_with_artifact = torch.as_tensor(
            data["vector_field_with_artifact"],
            dtype=torch.float32,
            device=self.device,
        )
        vector_field_without_artifact = torch.as_tensor(
            data["vector_field_without_artifact"],
            dtype=torch.float32,
            device=self.device,
        )

        fixed_artifact_mask = torch.as_tensor(
            data["fixed_artifact_mask"][None],
            dtype=torch.bool,
            device=self.device,
        )
        fixed_artifact_bbox = data["fixed_artifact_bbox"]

        moving_image = torch.as_tensor(
            data["moving_image"][None], dtype=torch.float32, device=self.device
        )
        moving_image = rescale_range(
            moving_image, input_range=(-1024, 3071), output_range=(0, 1)
        )
        fixed_image = torch.as_tensor(
            data["fixed_image"][None], dtype=torch.float32, device=self.device
        )
        fixed_image = rescale_range(
            fixed_image, input_range=(-1024, 3071), output_range=(0, 1)
        )
        fixed_image_masked_artifact = fixed_image.clone()
        fixed_image_masked_artifact[fixed_artifact_mask] = 0.0

        moving_mask = torch.as_tensor(
            data["moving_mask"][None], dtype=torch.bool, device=self.device
        )
        fixed_mask = torch.as_tensor(
            data["fixed_mask"][None], dtype=torch.bool, device=self.device
        )
        union_mask = moving_mask | fixed_mask
        moving_keypoints = torch.as_tensor(
            data["moving_keypoints"],
            dtype=torch.float32,
            device=self.device,
        )
        fixed_keypoints = torch.as_tensor(
            data["fixed_keypoints"],
            dtype=torch.float32,
            device=self.device,
        )

        roi_bbox = get_bounding_box(union_mask)
        # skip batch and channels
        roi_bbox = roi_bbox[2:]
        roi_bbox = pad_bounding_box_to_pow_2(
            roi_bbox, reference_shape=union_mask.shape[2:]
        )
        roi_slicing = (..., *roi_bbox)

        # roi_bbox = data["padded_fixed_bounding_box"]
        # roi_slicing = (..., *roi_bbox[0])
        image_spacing = torch.as_tensor([0.9765625, 0.9765625, 2.0], device=self.device)

        artifact_slicing = (..., *fixed_artifact_bbox[0])

        artifact_slicing_padding = 10
        extended_artifact_slicing = (
            ...,
            artifact_slicing[1],
            artifact_slicing[2],
            slice(
                max(artifact_slicing[3].start - artifact_slicing_padding, 0),
                artifact_slicing[3].stop + artifact_slicing_padding,
            ),
        )

        extended_artifact_mask = torch.zeros_like(fixed_artifact_mask)
        extended_artifact_mask[extended_artifact_slicing] = True

        modified_fixed_artifact_mask = torch.zeros_like(fixed_artifact_mask)
        modified_fixed_artifact_mask[extended_artifact_slicing] = True
        modified_fixed_artifact_mask[~fixed_artifact_mask] = False

        # boost the vector field
        with torch.autocast(device_type="cuda", enabled=False), torch.inference_mode():
            # x_mean = vector_field_with_artifact[:, 0:1][union_mask].mean()
            # y_mean = vector_field_with_artifact[:, 1:2][union_mask].mean()
            # z_mean = vector_field_with_artifact[:, 2:3][union_mask].mean()
            #
            # vector_field_mean = torch.as_tensor(
            #     [x_mean, y_mean, z_mean], device=self.device
            # )
            # vector_field_mean = vector_field_mean.resize(1, 3, 1, 1, 1)
            # x_std = vector_field_with_artifact[:, 0:1][union_mask].std()
            # y_std = vector_field_with_artifact[:, 1:2][union_mask].std()
            # z_std = vector_field_with_artifact[:, 2:3][union_mask].std()
            #
            # vector_field_std = torch.as_tensor(
            #     [x_std, y_std, z_std], device=self.device
            # )
            # vector_field_std = vector_field_std.resize(1, 3, 1, 1, 1)
            #
            # normed_vector_field_with_artifact = (
            #     vector_field_with_artifact - vector_field_mean
            # ) / vector_field_std

            warped_moving_image = self.spatial_transformer(
                moving_image, vector_field_with_artifact
            )

            inputs = torch.concat(
                (
                    # normed_vector_field_with_artifact,
                    # moving_image,  # moving = artifact-free
                    fixed_image_masked_artifact,
                    warped_moving_image,
                    fixed_artifact_mask.to(torch.float32),
                ),
                dim=1,
            )

            vector_field_boost = self.model(inputs[roi_slicing])
            if not torch.isfinite(vector_field_boost).all():
                raise RuntimeError()

            c = 25
            vector_field_boost = c * F.softsign(vector_field_boost)

            boosted_vector_field = vector_field_with_artifact.clone()
            boosted_vector_field[
                roi_slicing
            ] = self.spatial_transformer.compose_vector_fields(
                vector_field_1=vector_field_with_artifact[roi_slicing],
                vector_field_2=vector_field_boost,
            )

            # # restrict boost to -c, +c maximum correction
            # c = 3
            # vector_field_boost = c * F.softsign(vector_field_boost)
            # vector_field_boost = (
            #     vector_field_boost * vector_field_std + vector_field_mean
            # )
            #
            # # boosted_vector_field = vector_field_with_artifact.clone()
            # # boosted_vector_field[roi_slicing] += vector_field_boost
            #
            # full_vector_field_boost = torch.zeros_like(vector_field_with_artifact)
            # full_vector_field_boost[roi_slicing] = vector_field_boost
            #
            # # additive
            # # boosted_vector_field = vector_field_with_artifact.clone()
            # # boosted_vector_field[artifact_slicing] += full_vector_field_boost[
            # #     artifact_slicing
            # # ]
            #
            # boosted_vector_field = vector_field_with_artifact.clone()
            # boosted_vector_field[extended_artifact_slicing] = full_vector_field_boost[
            #     extended_artifact_slicing
            # ]

        #     mse_loss_total = self.mse_loss(
        #         moving_image, boosted_vector_field, fixed_image, fixed_mask
        #     )
        #     mse_loss_artifact = self.mse_loss(
        #         moving_image,
        #         boosted_vector_field,
        #         fixed_image,
        #         modified_fixed_artifact_mask,
        #     )
        #
        #     mse_loss_vector_field = F.mse_loss(
        #         boosted_vector_field[extended_artifact_slicing],
        #         vector_field_without_artifact[extended_artifact_slicing],
        #     )
        #
        #     tre_loss_without_artifact = self.tre_loss(
        #         vector_field_without_artifact,
        #         moving_keypoints,
        #         fixed_keypoints,
        #         image_spacing,
        #     )
        #     tre_loss_with_artifact = self.tre_loss(
        #         vector_field_with_artifact,
        #         moving_keypoints,
        #         fixed_keypoints,
        #         image_spacing,
        #     )
        #     tre_loss_with_artifact_boost = self.tre_loss(
        #         boosted_vector_field,
        #         moving_keypoints,
        #         fixed_keypoints,
        #         image_spacing,
        #     )
        #
        #     _, keypoint_mask = mask_keypoints(
        #         fixed_keypoints, extended_artifact_slicing[1:]
        #     )
        #
        #     tre_loss_with_artifact_roi = self.tre_loss(
        #         vector_field_with_artifact,
        #         moving_keypoints[0, keypoint_mask],
        #         fixed_keypoints[0, keypoint_mask],
        #         image_spacing,
        #     )
        #
        #     tre_loss_with_artifact_boost_roi = self.tre_loss(
        #         boosted_vector_field,
        #         moving_keypoints[0, keypoint_mask],
        #         fixed_keypoints[0, keypoint_mask],
        #         image_spacing,
        #     )
        #
        # tre_loss_total = (tre_loss_with_artifact_boost - tre_loss_with_artifact).mean()
        # tre_loss_artifact = (
        #     tre_loss_with_artifact_boost_roi - tre_loss_with_artifact_roi
        # ).mean()
        #
        # # loss = tre_loss + mse_loss
        # loss = (
        #     0.2 * mse_loss_total
        #     + 0.8 * mse_loss_artifact
        #     + mse_loss_vector_field * 0.01
        #     + tre_loss_total * 0.001
        #     + tre_loss_artifact * 0.005
        # )

        mse_image_artifact_before = self.mse_loss(
            moving_image,
            vector_field_with_artifact,
            fixed_image,
            modified_fixed_artifact_mask,
        )
        mse_image_artifact_after = self.mse_loss(
            moving_image,
            boosted_vector_field,
            fixed_image,
            modified_fixed_artifact_mask,
        )

        mse_image_before = self.mse_loss(
            moving_image, vector_field_with_artifact, fixed_image, fixed_mask
        )
        mse_image_after = self.mse_loss(
            moving_image, boosted_vector_field, fixed_image, fixed_mask
        )

        mse_vector_field_artifact_before = F.mse_loss(
            vector_field_with_artifact[..., extended_artifact_mask[0, 0]],
            vector_field_without_artifact[..., extended_artifact_mask[0, 0]],
        )
        mse_vector_field_artifact_after = F.mse_loss(
            boosted_vector_field[..., extended_artifact_mask[0, 0]],
            vector_field_without_artifact[..., extended_artifact_mask[0, 0]],
        )

        mse_vector_field_non_artifact_before = F.mse_loss(
            vector_field_with_artifact[..., ~extended_artifact_mask[0, 0]],
            vector_field_without_artifact[..., ~extended_artifact_mask[0, 0]],
        )
        mse_vector_field_non_artifact_after = F.mse_loss(
            boosted_vector_field[..., ~extended_artifact_mask[0, 0]],
            vector_field_without_artifact[..., ~extended_artifact_mask[0, 0]],
        )

        tre_before = self.tre_loss(
            vector_field_with_artifact,
            moving_keypoints,
            fixed_keypoints,
            image_spacing,
        )
        tre_after = self.tre_loss(
            boosted_vector_field,
            moving_keypoints,
            fixed_keypoints,
            image_spacing,
        )

        mse_image_artifact = mse_image_artifact_after / mse_image_artifact_before
        mse_image_non_artifact = mse_image_after / mse_image_before

        mse_image_loss = 0.8 * mse_image_artifact + 0.2 * mse_image_non_artifact
        mse_vector_field_loss = (
            mse_vector_field_artifact_after / mse_vector_field_artifact_before
        )
        tre_loss = (tre_after / tre_before).mean()

        loss = mse_vector_field_loss  # + tre_loss

        # loss = 1.0 * mse_image_loss + 1.0 * tre_loss
        # loss = loss / 2.0

        self.log_info(
            f"artifact size = {float(data['artifact_size'][0])}", context="VAL"
        )

        # plot validation data
        image_spacing = (0.9765625, 0.9765625, 2.0)
        aspect = image_spacing[0] / image_spacing[2]
        vector_field_kwargs = {"cmap": "seismic", "clim": (-20, 20), "aspect": aspect}
        image_kwargs = {"cmap": "gray", "clim": (0, 0.5), "aspect": aspect}
        diff_kwargs = {"cmap": "seismic", "clim": (-0.5, 0.5), "aspect": aspect}
        mid_y_slice = (roi_bbox[1].start + roi_bbox[1].stop) // 2

        _warped_image_boosted = self.spatial_transformer(
            moving_image, boosted_vector_field
        )
        _warped_image = self.spatial_transformer(
            moving_image, vector_field_with_artifact
        )

        _fixed_image = fixed_image[0, 0, :, mid_y_slice, :].detach().cpu().numpy()
        _warped_image_boosted = (
            _warped_image_boosted[0, 0, :, mid_y_slice, :].detach().cpu().numpy()
        )
        _warped_image = _warped_image[0, 0, :, mid_y_slice, :].detach().cpu().numpy()

        _fixed_artifact_mask = (
            fixed_artifact_mask[0, 0, :, mid_y_slice, :].detach().cpu().numpy()
        )
        _vector_field_with_artifact = (
            vector_field_with_artifact[0, 2, :, mid_y_slice, :].detach().cpu().numpy()
        )
        _vector_field_without_artifact = (
            vector_field_without_artifact[0, 2, :, mid_y_slice, :]
            .detach()
            .cpu()
            .numpy()
        )
        _boosted_vector_field = (
            boosted_vector_field[0, 2, :, mid_y_slice, :].detach().cpu().numpy()
        )

        _vector_field_with_artifact_diff = (
            _vector_field_with_artifact - _vector_field_without_artifact
        )

        _boosted_vector_field_diff = (
            _boosted_vector_field - _vector_field_without_artifact
        )

        with plt.ioff():
            if not self.test_plot:
                fig, ax = plt.subplots(
                    4, 3, sharex=True, sharey=True, figsize=(12, 9), dpi=300
                )
                self.test_plot = (fig, ax)
            else:
                fig, ax = self.test_plot

            ax[0, 0].imshow(_vector_field_with_artifact, **vector_field_kwargs)
            ax[0, 1].imshow(_vector_field_without_artifact, **vector_field_kwargs)
            ax[0, 2].imshow(_boosted_vector_field, **vector_field_kwargs)
            ax[1, 0].imshow(_vector_field_with_artifact_diff, **vector_field_kwargs)
            ax[1, 1].imshow(_fixed_artifact_mask, aspect=aspect)
            ax[1, 2].imshow(_boosted_vector_field_diff, **vector_field_kwargs)

            ax[2, 0].imshow(_warped_image, **image_kwargs)
            ax[2, 1].imshow(_fixed_image, **image_kwargs)
            ax[2, 2].imshow(_warped_image_boosted, **image_kwargs)

            ax[3, 0].imshow(_warped_image - _fixed_image, **diff_kwargs)
            ax[3, 1].imshow(_fixed_image - _fixed_image, **diff_kwargs)
            ax[3, 2].imshow(_warped_image_boosted - _fixed_image, **diff_kwargs)

            ax[0, 0].set_title(f"w/ artifact (size: {float(data['artifact_size'][0])})")
            ax[0, 1].set_title("w/o artifact")
            ax[0, 2].set_title("prediction")

            for _ax in fig.axes:
                _ax.grid()

            losses = {
                "loss": float(loss),
                "mse_image_loss": float(mse_image_loss),
                "mse_image_artifact": float(mse_image_artifact),
                "mse_image_non_artifact": float(mse_image_non_artifact),
                "tre_loss": float(tre_loss),
            }

            plot = Image(
                fig, caption=Path(data["patient"][0]).name + f", losses: {losses}"
            )
            # plt.close(fig)

        losses = {
            "loss": float(loss),
            "mse_image_loss": float(mse_image_loss),
            "mse_image_artifact": float(mse_image_artifact),
            "mse_image_non_artifact": float(mse_image_non_artifact),
            "tre_loss": float(tre_loss),
        }

        return {
            **losses,
            # "tre_loss_total": float(tre_loss_total),
            # "tre_loss_artifact": float(tre_loss_artifact),
            # "mse_loss_total": float(mse_loss_total),
            # "mse_loss_artifact": float(mse_loss_artifact),
            # "tre_loss_without_artifact": float(tre_loss_without_artifact.mean()),
            # "tre_loss_with_artifact": float(tre_loss_with_artifact.mean()),
            # "tre_loss_with_artifact_boost": float(tre_loss_with_artifact_boost.mean()),
            "prediction": plot,
        }


if __name__ == "__main__":
    import time

    import torch.nn as nn

    init_fancy_logging()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger("vroc").setLevel(logging.INFO)

    DEVICE = "cuda:0"

    patients = sorted(list(Path("/datalake2/vroc_artifact_boosting/v2").glob("*")))

    train_dataset = Lung4DArtifactBoostingDataset(
        patient_folders=patients,
        train_size=0.80,
        is_train=True,
    )
    test_dataset = Lung4DArtifactBoostingDataset(
        patient_folders=patients,
        train_size=0.80,
        is_train=False,
    )

    print(f"length train/test: {len(train_dataset)}/{len(test_dataset)}")

    # for overfitting test
    # train_dataset.filepaths = train_dataset.filepaths[:1]

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        collate_fn=partial(
            dict_collate,
            noop_keys=[
                "fixed_bounding_box",
                "padded_fixed_bounding_box",
                "fixed_artifact_bbox",
                "registration_parameters",
            ],
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=partial(
            dict_collate,
            noop_keys=[
                "fixed_bounding_box",
                "padded_fixed_bounding_box",
                "fixed_artifact_bbox",
                "registration_parameters",
            ],
        ),
    )

    convolution_kwargs = {
        "kernel_size": 3,
        "padding": "same",
        "bias": False,
    }

    n_filters_init = 32
    encoder_n_filters = (32, 32, 32, 32)
    decoder_n_filters = (32, 32, 32, 32)
    n_filters_final = 32
    model = FlexUNet(
        n_channels=3,
        n_classes=3,
        n_levels=4,
        n_filters=(
            n_filters_init,
            *encoder_n_filters,
            *decoder_n_filters,
            n_filters_final,
        ),
        norm_layer=None,  # nn.InstanceNorm3d,
        skip_connections=True,
        convolution_kwargs=convolution_kwargs,
        return_bottleneck=False,
    )

    # init with zeros
    model.final_conv.weight.data.fill_(0.0)
    # model.final_conv.bias.data.fill_(0.0)

    optimizer = Adam(model.parameters(), lr=1e-5)

    # load states
    state = torch.load(
        "/datalake/learn2reg/artifact_runs/models_48bddfef744f4e6bbfe6b841/validation/step_3000.pth",
        map_location="cuda",
    )
    model.load_state_dict(state["model"])

    trainer = VectorFieldArtifactBoosterTrainer(
        model=model,
        loss_function=None,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=test_loader,
        run_folder="/datalake/learn2reg/artifact_runs",
        experiment_name="vroc_artifact_boosting",
        device=DEVICE,
    )
    trainer.logger.setLevel(logging.DEBUG)
    trainer.run(steps=100_000, save_interval=500)
