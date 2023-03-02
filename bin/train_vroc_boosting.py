from __future__ import annotations

import logging
from functools import partial

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from vroc.blocks import SpatialTransformer
from vroc.dataset import NLSTDataset
from vroc.helper import dict_collate, rescale_range
from vroc.logger import LogFormatter
from vroc.loss import TRELoss, WarpedMSELoss
from vroc.models import DemonsVectorFieldBooster
from vroc.trainer import BaseTrainer, MetricType


class VectorFieldBoosterTrainer(BaseTrainer):
    METRICS = {
        "loss": MetricType.SMALLER_IS_BETTER,
        "tre_loss_rel_change": MetricType.SMALLER_IS_BETTER,
        "tre_metric_abs_change": MetricType.SMALLER_IS_BETTER,
        "mse_loss_rel_change": MetricType.SMALLER_IS_BETTER,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.spatial_transformer = SpatialTransformer(shape=(224, 192, 224)).to(DEVICE)
        self.tre_loss = TRELoss(apply_sqrt=True, reduction=None)
        self.tre_metric = TRELoss(apply_sqrt=True, reduction="mean")
        self.mse_loss = WarpedMSELoss(shape=(224, 192, 224)).to(DEVICE)

    def train_on_batch(self, data: dict) -> dict:
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

        # boost the vector field
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
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
                context="TRAIN",
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
    logger.setLevel(logging.INFO)
    logging.getLogger("vroc").setLevel(logging.INFO)

    DEVICE = "cuda:0"

    train_dataset = NLSTDataset(
        "/datalake/learn2reg/NLST",
        i_worker=None,
        n_worker=None,
        train_size=0.90,
        is_train=True,
        dilate_masks=1,
        unroll_vector_fields=True,
    )
    train_dataset.filepaths = train_dataset.filepaths[:1]

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=partial(dict_collate, noop_keys=("precomputed_vector_fields")),
        shuffle=True,
        # num_workers=4,
    )

    test_dataset = NLSTDataset(
        "/datalake/learn2reg/NLST",
        i_worker=None,
        n_worker=None,
        train_size=0.90,
        is_train=False,
        dilate_masks=1,
        unroll_vector_fields=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=partial(dict_collate, noop_keys=("precomputed_vector_fields")),
        shuffle=False,
    )

    model = DemonsVectorFieldBooster(shape=(224, 192, 224), n_iterations=4).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3)
    # optimizer = SGD(lr=1e-3)

    trainer = VectorFieldBoosterTrainer(
        model=model,
        loss_function=None,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=test_loader,
        run_folder="/datalake/learn2reg/runs",
        experiment_name="vector_field_boosting",
        device=DEVICE,
    )
    trainer.logger.setLevel(logging.DEBUG)
    trainer.run(steps=200_000, validation_interval=1000)
