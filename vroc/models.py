from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vroc.blocks import SpatialTransformer, DemonForces3d, GaussianSmoothing3d


class TrainableVarRegBlock(nn.Module):
    def __init__(
        self,
        patch_shape,
        iterations: Tuple[int, ...] = (100, 100),
        tau: float = 1.0,
        regularization_sigma=(1.0, 1.0, 1.0),
        disable_correction: bool = False,
        scale_factors: Tuple[float, ...] = (0.5, 1.0),
        early_stopping_delta: float = 0.0,
        early_stopping_window: int = 10,
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.iterations = iterations
        self.tau = tau
        self.regularization_sigma = regularization_sigma
        self.disable_correction = disable_correction
        self.scale_factors = scale_factors

        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_window = early_stopping_window
        self._metrics = []

        self._full_size_spatial_transformer = SpatialTransformer(shape=patch_shape)

        for i_level, scale_factor in enumerate(scale_factors):
            scaled_patch_shape = tuple(int(s * scale_factor) for s in self.patch_shape)

            self.add_module(
                name=f"spatial_transformer_level_{i_level}",
                module=SpatialTransformer(shape=scaled_patch_shape),
            )

        self._demon_forces_layer = DemonForces3d()

        self._regularization_layer = GaussianSmoothing3d(
            sigma=self.regularization_sigma, sigma_cutoff=2
        )

        self._vector_field_updater = nn.Sequential(
            nn.Conv3d(
                # 3 DVF channels, 2x1 image channels
                in_channels=5,
                out_channels=32,
                kernel_size=(3, 3, 3),
                padding="same",
                bias=False,
            ),
            nn.InstanceNorm3d(num_features=32),
            nn.Mish(),
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3, 3),
                padding="same",
                bias=False,
            ),
            nn.InstanceNorm3d(num_features=64),
            nn.Mish(),
            nn.Conv3d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 3, 3),
                padding="same",
                bias=False,
            ),
            nn.InstanceNorm3d(num_features=32),
            nn.Mish(),
            nn.Conv3d(
                in_channels=32,
                out_channels=3,
                kernel_size=(3, 3, 3),
                padding="same",
                bias=True,
            ),
        )

    @property
    def config(self):
        return dict(
            iterations=self.iterations,
            tau=self.tau,
            regularization_sigma=self.regularization_sigma,
        )

    def _check_early_stopping(self, metrics: List[float]):
        if (
            self.early_stopping_delta == 0
            or len(metrics) < self.early_stopping_window + 1
        ):
            return False

        rel_change = (metrics[-self.early_stopping_window] - metrics[-1]) / (
            metrics[-self.early_stopping_window] + 1e-9
        )
        # print(f'Rel. change: {rel_change}')

        if rel_change < self.early_stopping_delta:
            return True
        return False

    def _check_early_stopping_average_improvement(self, metrics: List[float]):
        if (
            self.early_stopping_delta == 0
            or len(metrics) < self.early_stopping_window + 1
        ):
            return False

        window = np.array(metrics[-self.early_stopping_window :])
        window_rel_changes = 1 - window[1:] / window[:-1]

        if window_rel_changes.mean() < self.early_stopping_delta:
            # print(f'Rel. change: {window_rel_changes}')
            return True
        return False

    def _perform_scaling(self, image, mask, conditional, scale_factor: float = 1.0):
        image = F.interpolate(
            image, scale_factor=scale_factor, mode="trilinear", align_corners=False
        )
        mask = F.interpolate(mask, scale_factor=scale_factor, mode="nearest")
        conditional = F.interpolate(
            conditional,
            scale_factor=scale_factor,
            mode="trilinear",
            align_corners=False,
        )

        return image, mask, conditional

    def _match_vector_field(self, vector_field, image):
        if vector_field.shape[2:] == image.shape[2:]:
            return vector_field

        vector_field_shape = vector_field.shape

        vector_field = F.interpolate(
            vector_field, size=image.shape[2:], mode="trilinear", align_corners=False
        )

        scale_factor = torch.ones((1, 3, 1, 1, 1), device=vector_field.device)
        scale_factor[0, :, 0, 0, 0] = torch.tensor(
            [s1 / s2 for (s1, s2) in zip(image.shape[2:], vector_field_shape[2:])]
        )

        return vector_field * scale_factor

    def warp_conditional(self, image, mask, conditional):
        # register conditional image onto image
        full_size_conditional = conditional

        vector_field = None
        with torch.no_grad():

            for i_level, (scale_factor, iterations) in enumerate(
                zip(self.scale_factors, self.iterations)
            ):
                metrics = []

                (scaled_image, scaled_mask, scaled_conditional) = self._perform_scaling(
                    image, mask, conditional, scale_factor=scale_factor
                )

                if vector_field is None:
                    vector_field = torch.zeros(
                        scaled_image.shape[:1] + (3,) + scaled_image.shape[2:],
                        device=conditional.device,
                    )
                elif vector_field.shape[2:] != scaled_image.shape[2:]:
                    vector_field = self._match_vector_field(vector_field, scaled_image)

                spatial_transformer = self.get_submodule(
                    f"spatial_transformer_level_{i_level}",
                )
                for i in range(iterations):
                    # print(f'*** ITERATION {i + 1} ***')

                    warped_conditional = spatial_transformer(
                        scaled_conditional, vector_field
                    )

                    metrics.append(float(F.mse_loss(scaled_image, warped_conditional)))

                    if self._check_early_stopping_average_improvement(metrics):
                        # print(f'Early stopping at iter {i}')
                        break

                    forces = self._demon_forces_layer(warped_conditional, scaled_image)
                    # mask forces with artifact mask (artifact = 0, valid = 1)

                    vector_field -= self.tau * (forces * scaled_mask)
                    vector_field = self._regularization_layer(vector_field)

                # print(f'ITERATION {i + 1}: Metric: {metrics[-1]}')
        if self.disable_correction:
            corrected_vector_field = vector_field
        else:
            # update/correct DVF at artifact region
            masked_image = image * mask
            stacked = torch.cat((vector_field, masked_image, warped_conditional), dim=1)

            vector_field_correction = self._vector_field_updater(stacked)
            corrected_vector_field = vector_field + vector_field_correction

            if not corrected_vector_field.isfinite().all():
                raise ValueError()

            corrected_vector_field = self._regularization_layer(corrected_vector_field)

        vector_field = self._match_vector_field(vector_field, full_size_conditional)
        corrected_vector_field = self._match_vector_field(
            corrected_vector_field, full_size_conditional
        )

        corrected_warped_conditional = self._full_size_spatial_transformer(
            full_size_conditional, corrected_vector_field
        )
        warped_conditional = self._full_size_spatial_transformer(
            full_size_conditional, vector_field
        )

        return (
            corrected_warped_conditional,
            warped_conditional,
            corrected_vector_field,
            vector_field,
        )

    def forward(self, image, mask, conditional):
        (
            corrected_warped_conditional,
            warped_conditional,
            corrected_vector_field,
            vector_field,
        ) = self.warp_conditional(image, mask, conditional)

        return (
            corrected_warped_conditional,
            warped_conditional,
            corrected_vector_field,
            vector_field,
        )
