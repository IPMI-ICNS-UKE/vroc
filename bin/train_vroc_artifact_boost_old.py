from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from vroc.blocks import SpatialTransformer
from vroc.dataset import Lung4DRegistrationDataset
from vroc.helper import (
    get_bounding_box,
    get_robust_bounding_box_3d,
    mask_keypoints,
    pad_bounding_box_to_pow_2,
    rescale_range,
)
from vroc.logger import LogFormatter
from vroc.loss import TRELoss, WarpedMSELoss, ncc_loss
from vroc.models import DemonsVectorFieldArtifactBooster
from vroc.registration import VrocRegistration
from vroc.trainer import BaseTrainer, MetricType


class VectorFieldArtifactBoosterTrainer(BaseTrainer):
    METRICS = {
        "loss": MetricType.SMALLER_IS_BETTER,
        "tre_loss_rel_change": MetricType.SMALLER_IS_BETTER,
        "tre_metric_abs_change": MetricType.SMALLER_IS_BETTER,
        "mse_loss_rel_change": MetricType.SMALLER_IS_BETTER,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.spatial_transformer = SpatialTransformer(shape=None).to(DEVICE)
        self.tre_loss = TRELoss(apply_sqrt=True, reduction=None)
        self.tre_metric = TRELoss(apply_sqrt=True, reduction="mean")
        self.mse_loss = WarpedMSELoss(shape=None).to(self.device)

    @staticmethod
    def generate_artifact_mask(
        roi_mask: np.ndarray, roi_z_range: Tuple[float, float], artifact_size: int = 1
    ) -> np.ndarray:
        # calculate valid range along z axis
        roi_bbox = get_robust_bounding_box_3d(roi_mask)

        n_slices = roi_bbox[-1].stop - roi_bbox[-1].start
        start = roi_bbox[-1].start + roi_z_range[0] * n_slices
        stop = roi_bbox[-1].start + roi_z_range[1] * n_slices

        start_slice_range = (
            start,
            max(stop - artifact_size, start + 1),  # stop is at least start + 1
        )

        random_start_slice = np.random.randint(*start_slice_range)

        artifact_mask = np.zeros_like(roi_mask, dtype=bool)
        artifact_mask[
            ..., random_start_slice : random_start_slice + artifact_size
        ] = True

        return artifact_mask

    def train_on_batch(self, data: dict) -> dict:

        registration = VrocRegistration(
            roi_segmenter=None,
            feature_extractor=None,
            parameter_guesser=None,
            device=self.device,
        )

        # exclude artifact region in fixed mask (fixed image is artifact affected image)
        # random aritfact size; low (inclusive) to high (exclusive)
        artifact_size = np.random.randint(5, 21)
        _fixed_mask = data["fixed_mask"][0, 0].detach().cpu().numpy()

        # calculate pow 2 ROI bbox
        roi_bbox = get_robust_bounding_box_3d(_fixed_mask)
        padded_roi_bbox = pad_bounding_box_to_pow_2(roi_bbox)

        fixed_artifact_mask = self.generate_artifact_mask(
            roi_mask=_fixed_mask,  # we need 3D input
            roi_z_range=(0.0, 0.3),
            artifact_size=artifact_size,
        )
        fixed_artifact_mask = fixed_artifact_mask[None, None]  # add color channel again

        moving_image = torch.as_tensor(data["moving_image"], device=self.device)
        fixed_image = torch.as_tensor(data["fixed_image"], device=self.device)
        moving_mask = torch.as_tensor(
            data["moving_mask"], device=self.device, dtype=torch.bool
        )
        fixed_mask = torch.as_tensor(
            data["fixed_mask"], device=self.device, dtype=torch.bool
        )
        fixed_artifact_mask = torch.as_tensor(
            fixed_artifact_mask, device=self.device, dtype=torch.bool
        )
        fixed_artifact_bbox = get_bounding_box(fixed_artifact_mask)
        # exclude artifact region from image registration
        fixed_mask = fixed_mask & (~fixed_artifact_mask)

        params = {
            "iterations": 200,
            "tau": 2.25,
            "tau_level_decay": 0.0,
            "tau_iteration_decay": 0.0,
            "sigma_x": 1.25,
            "sigma_y": 1.25,
            "sigma_z": 1.25,
            "sigma_level_decay": 0.0,
            "sigma_iteration_decay": 0.0,
            "n_levels": 3,
        }

        registration_result = registration.register(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            register_affine=True,
            affine_loss_function=ncc_loss,
            force_type="demons",
            gradient_type="active",
            valid_value_range=(-1024, 3071),
            early_stopping_delta=0.00001,
            early_stopping_window=None,
            debug=False,
            default_parameters=params,
            return_as_tensor=True,
        )

        composed_vector_field = registration_result.composed_vector_field.clone()
        warped_affine_moving_image = (
            registration_result.warped_affine_moving_image.clone()
        )
        warped_affine_moving_mask = (
            registration_result.warped_affine_moving_mask.clone()
        )
        varreg_vector_field = registration_result.vector_fields[-1].clone()

        moving_keypoints = torch.as_tensor(data["moving_keypoints"], device=self.device)
        fixed_keypoints = torch.as_tensor(data["fixed_keypoints"], device=self.device)
        image_spacing = torch.as_tensor(data["image_spacing"][0], device=self.device)

        roi_slicing = (..., *padded_roi_bbox)

        # boost the vector field
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            _vector_field_boost = self.model(
                warped_affine_moving_image[roi_slicing],
                fixed_image[roi_slicing],
                warped_affine_moving_mask[roi_slicing],
                fixed_mask[roi_slicing],
                fixed_artifact_mask[roi_slicing],
                varreg_vector_field[roi_slicing],
                image_spacing,
            )
            vector_field_boost = torch.zeros_like(varreg_vector_field)
            vector_field_boost[roi_slicing] = _vector_field_boost

            composed_boosted_vector_field = (
                vector_field_boost
                + self.spatial_transformer(composed_vector_field, vector_field_boost)
            )

            tre_loss_after_varreg = self.tre_loss(
                composed_vector_field,
                moving_keypoints,
                fixed_keypoints,
                image_spacing,
            )

            tre_loss_after_boosting = self.tre_loss(
                composed_boosted_vector_field,
                moving_keypoints,
                fixed_keypoints,
                image_spacing,
            )

            tre_metric_after_varreg = self.tre_metric(
                composed_vector_field,
                moving_keypoints,
                fixed_keypoints,
                image_spacing,
            )

            tre_metric_after_boosting = self.tre_metric(
                composed_boosted_vector_field,
                moving_keypoints,
                fixed_keypoints,
                image_spacing,
            )

            self.log_info(
                f"{tre_metric_after_varreg=:.3f} / {tre_metric_after_boosting=:.3f}",
                context="TRAIN",
            )

            tre_metric_abs_change = tre_metric_after_boosting - tre_metric_after_varreg

            mse_loss_after_varreg = self.mse_loss(
                moving_image, composed_vector_field, fixed_image, fixed_mask
            )
            mse_loss_after_boosting = self.mse_loss(
                moving_image, composed_boosted_vector_field, fixed_image, fixed_mask
            )

            tre_loss_rel_change = 1 + (
                tre_loss_after_boosting - tre_loss_after_varreg
            ) / (tre_loss_after_varreg + 1e-6)
            mse_loss_rel_change = 1 + (
                mse_loss_after_boosting - mse_loss_after_varreg
            ) / (mse_loss_after_varreg + 1e-6)

            loss = tre_loss_after_boosting - tre_loss_after_varreg

            # compute weighted mean
            _, keypoint_mask = mask_keypoints(
                keypoints=fixed_keypoints,
                bounding_box=fixed_artifact_bbox[-3:],  # just spatial bbox
            )

            tre_artifact_loss_after_varreg = float(
                tre_loss_after_varreg[keypoint_mask].mean()
            )
            tre_artifact_loss_after_boosting = float(
                tre_loss_after_boosting[keypoint_mask].mean()
            )
            self.logger.info(
                f"{tre_artifact_loss_after_varreg=:.3f} / {tre_artifact_loss_after_boosting=:.3f}"
            )

            keypoint_weighting = torch.ones_like(loss)
            keypoint_weighting[keypoint_mask] = 10.0
            loss = (loss @ keypoint_weighting) / keypoint_weighting.sum()

            # penalize worsening of TRE more
            if loss > 0:
                loss = loss**2

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        logger.info("Perform optimizer step")

        return {
            "loss": float(loss),
            "tre_loss_rel_change": float(tre_loss_rel_change.mean()),
            "tre_metric_abs_change": float(tre_metric_abs_change.mean()),
            "mse_loss_rel_change": float(mse_loss_rel_change.mean()),
        }

    def validate_on_batch(self, data: dict) -> dict:

        moving_image = rescale_range(
            data["moving_image"], input_range=(-1024, 3071), output_range=(0, 1)
        )
        fixed_image = rescale_range(
            data["fixed_image"], input_range=(-1024, 3071), output_range=(0, 1)
        )

        moving_image = torch.as_tensor(moving_image, device=self.device)
        fixed_image = torch.as_tensor(fixed_image, device=self.device)
        moving_mask = torch.as_tensor(data["moving_mask"], device=self.device)
        fixed_mask = torch.as_tensor(
            data["fixed_mask"], device=self.device, dtype=torch.bool
        )
        moving_keypoints = torch.as_tensor(data["moving_keypoints"], device=self.device)
        fixed_keypoints = torch.as_tensor(data["fixed_keypoints"], device=self.device)
        image_spacing = torch.as_tensor(data["image_spacing"][0], device=self.device)

        precomputed = data["precomputed_vector_field"][0]

        affine_vector_field = torch.as_tensor(
            precomputed["affine_vector_field"][None], device=self.device
        )

        varreg_vector_field = torch.as_tensor(
            precomputed["varreg_vector_field"][None], device=self.device
        )

        with torch.inference_mode():
            composed_varreg_vector_field = (
                varreg_vector_field
                + self.spatial_transformer(affine_vector_field, varreg_vector_field)
            )

            vector_field_boost = self.model(
                moving_image,
                fixed_image,
                moving_mask,
                fixed_mask,
                composed_varreg_vector_field,
                image_spacing,
            )

            composed_boosted_vector_field = (
                vector_field_boost
                + self.spatial_transformer(
                    composed_varreg_vector_field, vector_field_boost
                )
            )

            tre_loss_after_varreg = self.tre_loss(
                composed_varreg_vector_field,
                moving_keypoints,
                fixed_keypoints,
                image_spacing,
            )

            tre_loss_after_boosting = self.tre_loss(
                composed_boosted_vector_field,
                moving_keypoints,
                fixed_keypoints,
                image_spacing,
            )

            tre_metric_after_varreg = self.tre_metric(
                composed_varreg_vector_field,
                moving_keypoints,
                fixed_keypoints,
                image_spacing,
            )

            tre_metric_after_boosting = self.tre_metric(
                composed_boosted_vector_field,
                moving_keypoints,
                fixed_keypoints,
                image_spacing,
            )

            self.log_info(
                f"{tre_metric_after_varreg=:.3f} / {tre_metric_after_boosting=:.3f}",
                context="VAL",
            )

            tre_metric_abs_change = tre_metric_after_boosting - tre_metric_after_varreg

            mse_loss_after_varreg = self.mse_loss(
                moving_image, composed_varreg_vector_field, fixed_image, fixed_mask
            )
            mse_loss_after_boosting = self.mse_loss(
                moving_image, composed_boosted_vector_field, fixed_image, fixed_mask
            )

            tre_loss_rel_change = 1 + (
                tre_loss_after_boosting - tre_loss_after_varreg
            ) / (tre_loss_after_varreg + 1e-6)
            mse_loss_rel_change = 1 + (
                mse_loss_after_boosting - mse_loss_after_varreg
            ) / (mse_loss_after_varreg + 1e-6)

            loss = tre_loss_after_boosting - tre_loss_after_varreg
            loss = loss.mean()
            # penalize worsening of TRE more
            if loss > 0:
                loss = loss**2

        return {
            "loss": float(loss),
            "tre_loss_rel_change": float(tre_loss_rel_change.mean()),
            "tre_metric_abs_change": float(tre_metric_abs_change.mean()),
            "mse_loss_rel_change": float(mse_loss_rel_change.mean()),
        }


if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(LogFormatter())
    logging.basicConfig(handlers=[handler])

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger("vroc").setLevel(logging.INFO)

    DEVICE = "cuda:0"

    def filter_by_metadata(patient_folder: Path, filter_func):
        try:
            with open(patient_folder / "metadata.yml", "rt") as f:
                meta = yaml.safe_load(f)
            filter_result = filter_func(meta)
        except FileNotFoundError:
            logger.warning(f"No metadata.yml found in patient folder {patient_folder}")
            filter_result = False

        return filter_result

    def has_no_artifacts(meta: dict) -> bool:
        if (
            meta["artifactness_interpolation"] == 0
            and meta["artifactness_double_structure"] == 0
        ):
            return True
        else:
            return False

    patients = sorted(
        list(
            Path("/datalake_fast/xxx_4DCT_organ_type_status_nii").glob(
                "*_Lunge_amplitudebased_complete"
            )
        )
    )

    patients_no_artifacts = [
        patient
        for patient in patients
        if filter_by_metadata(patient, filter_func=has_no_artifacts)
    ]

    train_dataset = Lung4DRegistrationDataset(
        patient_folders=patients_no_artifacts,
        train_size=0.80,
        is_train=True,
    )
    test_dataset = Lung4DRegistrationDataset(
        patient_folders=patients_no_artifacts,
        train_size=0.80,
        is_train=False,
    )
    # for overfitting test
    # train_dataset.filepaths = train_dataset.filepaths[:1]

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        # num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )

    model = DemonsVectorFieldArtifactBooster(shape=None).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3)
    # optimizer = SGD(lr=1e-3)

    trainer = VectorFieldArtifactBoosterTrainer(
        model=model,
        loss_function=None,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=test_loader,
        run_folder="/datalake/learn2reg/artifact_runs",
        experiment_name="vector_field_artifact_boosting",
        device=DEVICE,
    )
    trainer.logger.setLevel(logging.DEBUG)
    trainer.run(steps=200_000, validation_interval=1000)
