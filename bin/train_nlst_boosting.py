from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset

from vroc.blocks import DemonForces, DynamicRegularization3d, SpatialTransformer
from vroc.logger import LoggerMixin, RegistrationLogEntry, init_fancy_logging
from vroc.loss import TRELoss, WarpedMSELoss, smooth_vector_field_loss
from vroc.models import FlexUNet


class BoostingDataset(Dataset):
    def __init__(self, registration_results_filepaths: Sequence[Path]):
        super().__init__()

        self.registration_results = []
        for registration_results_filepath in registration_results_filepaths:
            with open(registration_results_filepath, "rb") as f:
                registration_result = pickle.load(f)
            self.registration_results.append(registration_result)

    def __getitem__(self, item):
        result = self.registration_results[item]
        return {
            "moving_image": result.moving_image,
            "fixed_image": result.fixed_image,
            "moving_mask": result.moving_mask,
            "fixed_mask": result.fixed_mask,
            "composed_vector_field": result.composed_vector_field,
            "moving_keypoints": result.moving_keypoints,
            "fixed_keypoints": result.fixed_keypoints,
        }


class DemonsVectorFieldBooster(nn.Module, LoggerMixin):
    def __init__(
        self,
        n_iterations: int = 10,
        filter_base: int = 16,
        gradient_type: Literal["active", "passive", "dual"] = "dual",
    ):
        super().__init__()

        self.n_iterations = n_iterations
        self.filter_base = filter_base
        self.forces = DemonForces(method=gradient_type)

        # self.regularization = TrainableRegularization3d(n_levels=4, filter_base=16)
        self.regularization = DynamicRegularization3d(filter_base=16)
        self.spatial_transformer = SpatialTransformer()

        self.factors = (0.125, 0.25, 0.5, 1.0)
        self.n_levels = len(self.factors)
        self.weighting_net = FlexUNet(
            n_channels=2,
            n_levels=4,
            n_classes=self.n_levels + 3,
            filter_base=2,
            norm_layer=nn.InstanceNorm3d,
            return_bottleneck=False,
            skip_connections=True,
        )

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
        n_iterations: int | None = None,
    ) -> torch.Tensor:

        spatial_image_shape = moving_image.shape[2:]
        vector_field_boost = torch.zeros(
            (1, 3) + spatial_image_shape, device=moving_image.device
        )

        _n_iterations = n_iterations or self.n_iterations
        for _ in range(_n_iterations):
            composed_vector_field = self.spatial_transformer.compose_vector_fields(
                vector_field, vector_field_boost
            )

            # warp image with boosted vector field
            warped_moving_image = self.spatial_transformer(
                moving_image, composed_vector_field
            )

            # diff = (warped_moving_image - fixed_image) / (fixed_image + 1e-6)
            # diff = F.softsign(diff)

            forces = self.forces(
                warped_moving_image,
                fixed_image,
                moving_mask,
                fixed_mask,
                image_spacing,
            )

            images = torch.concat((moving_image, fixed_image), dim=1)

            output = self.weighting_net(images)
            weights = output[:, : self.n_levels]
            weights = torch.softmax(weights, dim=1)
            taus = output[:, self.n_levels :]
            taus = 10 * torch.sigmoid(taus)
            # print(
            #     f'mean tau x/y/z: {taus[:, 0].mean():.2f}, {taus[:, 1].mean():.2f}, {taus[:, 2].mean():.2f}')

            vector_field_boost = vector_field_boost + taus * forces
            vector_field_boost = self.regularization(
                vector_field=vector_field_boost,
                moving_image=warped_moving_image,
                fixed_image=fixed_image,
                weights=weights,
            )

            # # plot weights and tau
            # m = warped_moving_image.detach().cpu().numpy()
            # f = fixed_image.detach().cpu().numpy()
            # diff = (warped_moving_image - fixed_image)
            # diff = diff.detach().cpu().numpy()
            # w = weights.detach().cpu().numpy()
            #
            # m, f = diff, diff
            # clim = (-1, 1)
            # cmap = 'seismic'
            # mid_slice = w.shape[-2] // 2
            # fig, ax = plt.subplots(1, self.n_levels + 2, sharex=True, sharey=True)
            # ax[0].imshow(f[0, 0, :, mid_slice, :], clim=clim, cmap=cmap)
            # ax[1].imshow(m[0, 0, :, mid_slice, :], clim=clim, cmap=cmap)
            # for i in range(self.n_levels):
            #     ax[i + 2].imshow(w[0, i, :, mid_slice, :])
            #
            # t = taus.detach().cpu().numpy()
            # mid_slice = t.shape[-2] // 2
            # fig, ax = plt.subplots(1, 3 + 2, sharex=True, sharey=True)
            # ax[0].imshow(f[0, 0, :, mid_slice, :], clim=clim, cmap=cmap)
            # ax[1].imshow(m[0, 0, :, mid_slice, :], clim=clim, cmap=cmap)
            # for i in range(3):
            #     ax[i + 2].imshow(t[0, i, :, mid_slice, :])

        return vector_field_boost


if __name__ == "__main__":

    init_fancy_logging()

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)
    logging.getLogger("vroc").setLevel(logging.DEBUG)
    logging.getLogger("vroc.affine").setLevel(logging.DEBUG)
    logging.getLogger("vroc.models.VarReg").setLevel(logging.DEBUG)

    DEVICE = "cuda:0"
    model = DemonsVectorFieldBooster(n_iterations=1).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-4)
    gradient_scaler = torch.cuda.amp.GradScaler()

    filepaths = sorted(
        list(Path("/datalake/learn2reg/NLST_Validation/boosting_samples").glob("*"))
    )
    train_filepaths = filepaths[:8]
    test_filepaths = filepaths[8:]

    train_dataset = BoostingDataset(registration_results_filepaths=train_filepaths)
    test_dataset = BoostingDataset(registration_results_filepaths=test_filepaths)

    valid_value_range = (-1000, 3071)
    keypoint_loss_weight = 1.0
    smoothness_loss_weight = 1.0
    image_loss_weight = 1.0

    image_loss = WarpedMSELoss()
    tre_loss = TRELoss(apply_sqrt=False, reduction=None)
    n_iterations = 16

    train_losses = []
    test_losses = []

    for i_epoch in range(10_000):
        train_loss = []
        for data in train_dataset:
            moving_image = data["moving_image"].to(DEVICE)
            fixed_image = data["fixed_image"].to(DEVICE)
            moving_mask = data["moving_mask"].to(DEVICE)
            fixed_mask = data["fixed_mask"].to(DEVICE)
            composed_vector_field = data["composed_vector_field"].to(DEVICE)
            moving_keypoints = data["moving_keypoints"].to(DEVICE)
            fixed_keypoints = data["fixed_keypoints"].to(DEVICE)

            # transform and clip images to value range [0, 1]
            moving_image = (moving_image - valid_value_range[0]) / (
                valid_value_range[1] - valid_value_range[0]
            )
            fixed_image = (fixed_image - valid_value_range[0]) / (
                valid_value_range[1] - valid_value_range[0]
            )

            moving_image = torch.clip(moving_image, 0, 1)
            fixed_image = torch.clip(fixed_image, 0, 1)

            image_spacing = torch.as_tensor((1.5, 1.5, 1.5), device=DEVICE)

            # init loss layers and values (even if not used)
            # setup and initialize losses and loss weights
            requested_losses = set()
            loss_weights = {}
            tre_loss_before_boosting = None
            smoothness_before_boosting = None
            image_loss_before_boosting = None
            tre_loss_after_boosting = None
            smoothness_after_boosting = None
            image_loss_after_boosting = None
            if (
                keypoint_loss_weight != 0
                and moving_keypoints is not None
                and fixed_keypoints is not None
            ):
                # we have to compute keypoint loss
                requested_losses.add("keypoint")
                loss_weights["keypoint"] = keypoint_loss_weight
                # calculate loss before boosting
                tre_loss_before_boosting = tre_loss(
                    composed_vector_field,
                    moving_keypoints,
                    fixed_keypoints,
                    image_spacing,
                )
                tre_metric_before_boosting = tre_loss_before_boosting.sqrt().mean()

            if smoothness_loss_weight != 0:
                # we have to compute smoothness loss
                requested_losses.add("smoothness")
                loss_weights["smoothness"] = smoothness_loss_weight
                # calculate loss before boosting
                smoothness_before_boosting = smooth_vector_field_loss(
                    vector_field=composed_vector_field,
                    mask=fixed_mask,
                    l2r_variant=True,
                )
            if image_loss_weight != 0:
                requested_losses.add("image")
                loss_weights["image"] = image_loss_weight
                # calculate loss before boosting
                image_loss_before_boosting = image_loss(
                    moving_image=moving_image,
                    vector_field=composed_vector_field,
                    fixed_image=fixed_image,
                    fixed_mask=fixed_mask,
                )

            # initial values in case of 0 iterations
            composed_boosted_vector_field = composed_vector_field
            vector_field_boost = torch.zeros_like(composed_boosted_vector_field)
            print("***")
            for i_iteration in range(n_iterations):
                # reset after 20 iterations
                # if i_iteration % 20 == 0:
                #     print(f'reset at {i_iteration = }')
                #     composed_boosted_vector_field = composed_vector_field

                # composed_boosted_vector_field = composed_boosted_vector_field.detach()
                with torch.autocast(device_type="cuda", enabled=True):

                    vector_field_boost = model(
                        moving_image,
                        fixed_image,
                        moving_mask,
                        fixed_mask,
                        composed_boosted_vector_field,
                        image_spacing,
                        n_iterations=None,
                    )

                    # # start from scratch
                    # composed_boosted_vector_field = (
                    #     vector_field_boost
                    #     + model.spatial_transformer(
                    #         composed_vector_field, vector_field_boost
                    #     )
                    # )

                    # continue with boosted result
                    composed_boosted_vector_field = (
                        vector_field_boost
                        + model.spatial_transformer(
                            composed_boosted_vector_field, vector_field_boost
                        )
                    )

                    # initialize log entry
                    log = RegistrationLogEntry(
                        stage="boosting",
                        iteration=i_iteration,
                    )
                    log.epoch = i_epoch

                    # compute specified losses (keypoints, smoothness and labels)
                    losses = {}
                    if "keypoint" in requested_losses:
                        tre_loss_after_boosting = tre_loss(
                            composed_boosted_vector_field,
                            moving_keypoints,
                            fixed_keypoints,
                            image_spacing,
                        )

                        tre_metric_after_boosting = (
                            tre_loss_after_boosting.sqrt().mean()
                        )
                        tre_ratio_loss = (
                            tre_loss_after_boosting.sqrt().mean()
                            / tre_loss_before_boosting.sqrt().mean()
                        )

                        losses["keypoint"] = tre_ratio_loss

                        # add to log
                        log.tre_metric_before_boosting = tre_metric_before_boosting
                        log.tre_metric_after_boosting = tre_metric_after_boosting

                    if "smoothness" in requested_losses:
                        smoothness_after_boosting = smooth_vector_field_loss(
                            vector_field=composed_boosted_vector_field,
                            mask=fixed_mask,
                            l2r_variant=True,
                        )

                        smoothness_ratio_loss = (
                            smoothness_after_boosting / smoothness_before_boosting
                        )

                        losses["smoothness"] = smoothness_ratio_loss

                        # add to log
                        log.smoothness_before_boosting = smoothness_before_boosting
                        log.smoothness_after_boosting = smoothness_after_boosting

                        # ratio < 1: better, ratio > 1 worse
                        # only penalize worsening of smoothness
                        # if smoothness_ratio_loss < 1:
                        #     smoothness_ratio_loss = 0.5 * smoothness_ratio_loss + 0.5
                        # smoothness_ratio_loss = torch.maximum(
                        #     smoothness_ratio_loss, torch.as_tensor(1.0)
                        # )

                    if "image" in requested_losses:
                        image_loss_after_boosting = image_loss(
                            moving_image=moving_image,
                            vector_field=composed_boosted_vector_field,
                            fixed_image=fixed_image,
                            fixed_mask=fixed_mask,
                        )

                        image_ratio_loss = (
                            image_loss_after_boosting / image_loss_before_boosting
                        )

                        losses["image"] = image_ratio_loss

                        # add to log
                        log.image_loss_before_boosting = image_loss_before_boosting
                        log.image_loss_after_boosting = image_loss_after_boosting

                    # reduce losses to scalar
                    loss = 0.0
                    weight_sum = 0.0
                    for loss_name, loss_value in losses.items():
                        loss += loss_weights[loss_name] * loss_value
                        weight_sum += loss_weights[loss_name]
                    loss /= weight_sum
                    loss = loss / n_iterations

                    train_loss.append(float(loss * n_iterations))
                    # add loss info to log
                    log.loss = loss
                    log.loss_unscaled = loss * n_iterations
                    log.losses = losses
                    log.loss_weights = loss_weights
                    logger.debug(log)

                    gradient_scaler.scale(loss).backward()

                if tre_loss_after_boosting is not None:
                    tre_loss_before_boosting = tre_loss_after_boosting.detach()
                    tre_metric_before_boosting = tre_metric_after_boosting.detach()
                if smoothness_after_boosting is not None:
                    smoothness_before_boosting = smoothness_after_boosting.detach()
                if image_loss_after_boosting is not None:
                    image_loss_before_boosting = image_loss_after_boosting.detach()

                composed_boosted_vector_field = composed_boosted_vector_field.detach()

            print("Update")
            train_losses.append(float(loss))
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            optimizer.zero_grad()

        train_losses.append(np.mean(train_loss))

        # apply:
        n_iterations = 16
        test_loss = []
        for data in test_dataset:
            moving_imge = data["moving_image"].to(DEVICE)
            fixed_image = data["fixed_image"].to(DEVICE)
            moving_mask = data["moving_mask"].to(DEVICE)
            fixed_maska = data["fixed_mask"].to(DEVICE)
            composed_vector_field = data["composed_vector_field"].to(DEVICE)
            moving_keypoints = data["moving_keypoints"].to(DEVICE)
            fixed_keypoints = data["fixed_keypoints"].to(DEVICE)

            # transform and clip images to value range [0, 1]
            moving_image = (moving_image - valid_value_range[0]) / (
                valid_value_range[1] - valid_value_range[0]
            )
            fixed_image = (fixed_image - valid_value_range[0]) / (
                valid_value_range[1] - valid_value_range[0]
            )

            moving_image = torch.clip(moving_image, 0, 1)
            fixed_image = torch.clip(fixed_image, 0, 1)

            image_spacing = torch.as_tensor((1.5, 1.5, 1.5), device=DEVICE)

            # init loss layers and values (even if not used)
            # setup and initialize losses and loss weights
            requested_losses = set()
            loss_weights = {}
            tre_loss_before_boosting = None
            smoothness_before_boosting = None
            image_loss_before_boosting = None
            tre_loss_after_boosting = None
            smoothness_after_boosting = None
            image_loss_after_boosting = None
            if (
                keypoint_loss_weight != 0
                and moving_keypoints is not None
                and fixed_keypoints is not None
            ):
                # we have to compute keypoint loss
                requested_losses.add("keypoint")
                loss_weights["keypoint"] = keypoint_loss_weight
                # calculate loss before boosting
                tre_loss_before_boosting = tre_loss(
                    composed_vector_field,
                    moving_keypoints,
                    fixed_keypoints,
                    image_spacing,
                )
                tre_metric_before_boosting = tre_loss_before_boosting.sqrt().mean()

            if smoothness_loss_weight != 0:
                # we have to compute smoothness loss
                requested_losses.add("smoothness")
                loss_weights["smoothness"] = smoothness_loss_weight
                # calculate loss before boosting
                smoothness_before_boosting = smooth_vector_field_loss(
                    vector_field=composed_vector_field,
                    mask=fixed_mask,
                    l2r_variant=True,
                )
            if image_loss_weight != 0:
                requested_losses.add("image")
                loss_weights["image"] = image_loss_weight
                # calculate loss before boosting
                image_loss_before_boosting = image_loss(
                    moving_image=moving_image,
                    vector_field=composed_vector_field,
                    fixed_image=fixed_image,
                    fixed_mask=fixed_mask,
                )

            # initial values in case of 0 iterations
            composed_boosted_vector_field = composed_vector_field
            vector_field_boost = torch.zeros_like(composed_boosted_vector_field)
            print("***")
            for i_iteration in range(n_iterations):
                with torch.autocast(device_type="cuda", enabled=True), torch.no_grad():

                    vector_field_boost = model(
                        moving_image,
                        fixed_image,
                        moving_mask,
                        fixed_mask,
                        composed_boosted_vector_field,
                        image_spacing,
                        n_iterations=None,
                    )

                    # # start from scratch
                    # composed_boosted_vector_field = (
                    #     vector_field_boost
                    #     + model.spatial_transformer(
                    #         composed_vector_field, vector_field_boost
                    #     )
                    # )

                    # continue with boosted result
                    composed_boosted_vector_field = (
                        vector_field_boost
                        + model.spatial_transformer(
                            composed_boosted_vector_field, vector_field_boost
                        )
                    )

                    # initialize log entry
                    log = RegistrationLogEntry(
                        stage="boosting",
                        iteration=i_iteration,
                    )

                    # compute specified losses (keypoints, smoothness and labels)
                    losses = {}
                    if "keypoint" in requested_losses:
                        tre_loss_after_boosting = tre_loss(
                            composed_boosted_vector_field,
                            moving_keypoints,
                            fixed_keypoints,
                            image_spacing,
                        )

                        tre_metric_after_boosting = (
                            tre_loss_after_boosting.sqrt().mean()
                        )
                        tre_ratio_loss = (
                            tre_loss_after_boosting.mean()
                            / tre_loss_before_boosting.mean()
                        )

                        losses["keypoint"] = tre_ratio_loss

                        # add to log
                        log.tre_metric_before_boosting = tre_metric_before_boosting
                        log.tre_metric_after_boosting = tre_metric_after_boosting

                    if "smoothness" in requested_losses:
                        smoothness_after_boosting = smooth_vector_field_loss(
                            vector_field=composed_boosted_vector_field,
                            mask=fixed_mask,
                            l2r_variant=True,
                        )

                        smoothness_ratio_loss = (
                            smoothness_after_boosting / smoothness_before_boosting
                        )

                        losses["smoothness"] = smoothness_ratio_loss

                        # add to log
                        log.smoothness_before_boosting = smoothness_before_boosting
                        log.smoothness_after_boosting = smoothness_after_boosting

                        # ratio < 1: better, ratio > 1 worse
                        # only penalize worsening of smoothness
                        # if smoothness_ratio_loss < 1:
                        #     smoothness_ratio_loss = 0.5 * smoothness_ratio_loss + 0.5
                        # smoothness_ratio_loss = torch.maximum(
                        #     smoothness_ratio_loss, torch.as_tensor(1.0)
                        # )

                    if "image" in requested_losses:
                        image_loss_after_boosting = image_loss(
                            moving_image=moving_image,
                            vector_field=composed_boosted_vector_field,
                            fixed_image=fixed_image,
                            fixed_mask=fixed_mask,
                        )

                        image_ratio_loss = (
                            image_loss_after_boosting / image_loss_before_boosting
                        )

                        losses["image"] = image_ratio_loss

                        # add to log
                        log.image_loss_before_boosting = image_loss_before_boosting
                        log.image_loss_after_boosting = image_loss_after_boosting

                    # reduce losses to scalar
                    loss = 0.0
                    weight_sum = 0.0
                    for loss_name, loss_value in losses.items():
                        loss += loss_weights[loss_name] * loss_value
                        weight_sum += loss_weights[loss_name]
                    loss /= weight_sum

                    test_loss.append(float(loss))
                    # add loss info to log
                    log.loss = loss
                    log.losses = losses
                    log.loss_weights = loss_weights
                    if i_iteration == n_iterations - 1 or True:
                        logger.debug(log)

                if tre_loss_after_boosting is not None:
                    tre_loss_before_boosting = tre_loss_after_boosting.detach()
                    tre_metric_before_boosting = tre_metric_after_boosting.detach()
                if smoothness_after_boosting is not None:
                    smoothness_before_boosting = smoothness_after_boosting.detach()
                if image_loss_after_boosting is not None:
                    image_loss_before_boosting = image_loss_after_boosting.detach()

                composed_boosted_vector_field = composed_boosted_vector_field.detach()

        test_losses.append(np.mean(test_loss))

        print(f"last train / test loss: {train_losses[-1]:.3f} / {test_losses[-1]:.3f}")
