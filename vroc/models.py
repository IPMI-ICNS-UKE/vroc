import time
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from vroc.blocks import SpatialTransformer, DemonForces3d, GaussianSmoothing3d
from helper import rescale_range


class TrainableVarRegBlock(nn.Module):
    def __init__(
            self,
            patch_shape,
            iterations: Tuple[int, ...] = (100, 100),
            tau: Tuple[float, ...] = (1.0, 1.0),
            demon_forces='active',
            regularization_sigma=(1.0, 1.0, 1.0),
            disable_correction: bool = False,
            scale_factors: Tuple[float, ...] = (0.5, 1.0),
            early_stopping_fn: Tuple[str, ...] = ('no_impr', 'lstsq'),
            early_stopping_delta: Tuple[float, ...] = (0.0, 0.0),
            early_stopping_window: Tuple[int, ...] = (10, 10),
            radius: Tuple[int, ...] = None
    ):
        super().__init__()
        self.patch_shape = patch_shape
        self.iterations = iterations
        self.tau = tau
        self.regularization_sigma = regularization_sigma
        self.disable_correction = disable_correction
        self.scale_factors = scale_factors
        self.demon_forces = demon_forces

        self.early_stopping_fn = early_stopping_fn
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_window = early_stopping_window
        self._metrics = []
        self._counter = 0

        self._full_size_spatial_transformer = SpatialTransformer(shape=patch_shape)

        for i_level, scale_factor in enumerate(scale_factors):
            scaled_patch_shape = tuple(int(s * scale_factor) for s in self.patch_shape)

            self.add_module(
                name=f"spatial_transformer_level_{i_level}",
                module=SpatialTransformer(shape=scaled_patch_shape),
            )

        self._demon_forces_layer = DemonForces3d()

        for i_level, sigma in enumerate(regularization_sigma):
            self.add_module(
                name=f"regularization_layer_level_{i_level}",
                module=GaussianSmoothing3d(sigma=sigma, sigma_cutoff=(3.0, 3.0, 3.0), same_size=True, radius=radius[i_level]),
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

    def _check_early_stopping(self, metrics: List[float], i_level):
        if (
                self.early_stopping_delta[i_level] == 0
                or len(metrics) < self.early_stopping_window[i_level] + 1
        ):
            return False

        rel_change = (metrics[-self.early_stopping_window[i_level]] - metrics[-1]) / (
                metrics[-self.early_stopping_window[i_level]] + 1e-9
        )
        # print(f'Rel. change: {rel_change}')

        if rel_change < self.early_stopping_delta[i_level]:
            return True
        return False

    def _check_early_stopping_average_improvement(self, metrics: List[float], i_level):
        if (
                self.early_stopping_delta[i_level] == 0
                or len(metrics) < self.early_stopping_window[i_level] + 1
        ):
            return False

        window = np.array(metrics[-self.early_stopping_window[i_level]:])
        window_rel_changes = 1 - window[1:] / window[:-1]

        if window_rel_changes.mean() < self.early_stopping_delta[i_level]:
            # print(f'Rel. change: {window_rel_changes}')
            return True
        return False

    def _check_early_stopping_improvement(self, metrics: List[float], i_level):
        if (
                self.early_stopping_delta[i_level] == 0
                or len(metrics) < self.early_stopping_window[i_level] + 1
        ):
            return False

        window = np.array(metrics[-self.early_stopping_window[i_level]:])
        window_rel_changes = 1 - window[1:] / window[:-1]

        if window_rel_changes.mean() < self.early_stopping_delta[i_level]:
            # print(f'Rel. change: {window_rel_changes}')
            return True
        return False

    def _check_early_stopping_increase_count(self, metrics: List[float], i_level):
        # if len(metrics) < self.early_stopping_window[i_level] + 1:
        #     return False

        window = np.array(metrics)
        if np.argmin(window) == (len(window) - 1):
            self._counter = 0
        else:
            self._counter += 1
        if self._counter == self.early_stopping_delta[i_level]:
            return True
        return False

    def _check_early_stopping_lstsq(self, metrics: List[float], i_level):
        if (
                self.early_stopping_delta[i_level] == 0
                or len(metrics) < self.early_stopping_window[i_level] + 1
        ):
            return False
        window = np.array(metrics[-self.early_stopping_window[i_level]:])
        scaled_window = rescale_range(window, (np.min(metrics), np.max(metrics)), (0, 1))
        lstsq_result = stats.linregress(np.arange(self.early_stopping_window[i_level]), scaled_window)
        if lstsq_result.slope > -self.early_stopping_delta[i_level]:
            return True
        return False

    def _perform_scaling(self, image, mask, moving, scale_factor: float = 1.0):
        if len(image.shape) == 4:
            mode = 'bilinear'
        elif len(image.shape) == 5:
            mode = 'trilinear'
        image = F.interpolate(
            image, scale_factor=scale_factor, mode=mode, align_corners=False
        )
        mask = F.interpolate(mask, scale_factor=scale_factor, mode="nearest")
        moving = F.interpolate(
            moving,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False,
        )

        return image, mask, moving

    def _match_vector_field(self, vector_field, image):
        if len(image.shape) == 4:
            mode = 'bilinear'
        elif len(image.shape) == 5:
            mode = 'trilinear'
        if vector_field.shape[2:] == image.shape[2:]:
            return vector_field

        vector_field_shape = vector_field.shape

        vector_field = F.interpolate(
            vector_field, size=image.shape[2:], mode=mode, align_corners=False
        )

        if len(image.shape) == 4:
            scale_factor = torch.ones((1, 2, 1, 1), device=vector_field.device)
            scale_factor[0, :, 0, 0] = torch.tensor(
                [s1 / s2 for (s1, s2) in zip(image.shape[2:], vector_field_shape[2:])]
            )
        if len(image.shape) == 5:
            scale_factor = torch.ones((1, 3, 1, 1, 1), device=vector_field.device)
            scale_factor[0, :, 0, 0, 0] = torch.tensor(
                [s1 / s2 for (s1, s2) in zip(image.shape[2:], vector_field_shape[2:])]
            )

        return vector_field * scale_factor

    def warp_moving(self, image, mask, moving, original_image_spacing):
        # register moving image onto fixed image
        if len(image.shape) == 4:
            dim_vf = 2
        elif len(image.shape) == 5:
            dim_vf = 3
        full_size_moving = moving

        metrics_all_level = []
        vector_field = None
        with torch.no_grad():

            for i_level, (scale_factor, iterations) in enumerate(
                    zip(self.scale_factors, self.iterations)
            ):
                metrics = []
                self._counter = 0

                (scaled_image, scaled_mask, scaled_moving) = self._perform_scaling(
                    image, mask, moving, scale_factor=scale_factor
                )

                if vector_field is None:
                    vector_field = torch.zeros(
                        scaled_image.shape[:1] + (dim_vf,) + scaled_image.shape[2:],
                        device=moving.device,
                    )
                elif vector_field.shape[2:] != scaled_image.shape[2:]:
                    vector_field = self._match_vector_field(vector_field, scaled_image)

                spatial_transformer = self.get_submodule(
                    f"spatial_transformer_level_{i_level}",
                )
                for i in range(iterations):
                    if (i % 100) == 0:
                        print(f'*** LEVEL {i_level + 1} *** ITERATION {i} ***')

                    warped_moving = spatial_transformer(
                        scaled_moving, vector_field
                    )

                    metrics.append(float(F.mse_loss(scaled_image[scaled_mask == 1], warped_moving[scaled_mask == 1])))

                    if self.early_stopping_fn[i_level] == 'lstsq':
                        if self._check_early_stopping_lstsq(metrics, i_level):
                            break
                    elif self.early_stopping_fn[i_level] == 'no_impr':
                        if self._check_early_stopping_increase_count(metrics, i_level):
                            break
                    elif self.early_stopping_fn[i_level] == 'no_average_impr':
                        if self._check_early_stopping_average_improvement(metrics, i_level):
                            break
                    elif self.early_stopping_fn[i_level] == 'none':
                        pass
                    else:
                        raise Exception(f'Early stopping method {self.early_stopping_fn[i_level]} is not implemented')

                    forces = self._demon_forces_layer(warped_moving, scaled_image, self.demon_forces,
                                                      original_image_spacing)
                    # mask forces with artifact mask (artifact = 0, valid = 1)

                    vector_field += self.tau[i_level] * (forces * scaled_mask)
                    _regularization_layer = self.get_submodule(
                        f"regularization_layer_level_{i_level}",
                    )
                    vector_field = _regularization_layer(vector_field)

                print(f'LEVEL {i_level + 1} stopped at ITERATION {i + 1}: Metric: {metrics[-1]}')
                metrics_all_level.append(metrics)

        if self.disable_correction:
            corrected_vector_field = vector_field
        else:
            # update/correct DVF at artifact region
            masked_image = image * mask
            stacked = torch.cat((vector_field, masked_image, warped_moving), dim=1)

            vector_field_correction = self._vector_field_updater(stacked)
            corrected_vector_field = vector_field + vector_field_correction

            if not corrected_vector_field.isfinite().all():
                raise ValueError()

            _regularization_layer = self.get_submodule(
                f"regularization_layer_level_{i_level}",
            )
            corrected_vector_field = _regularization_layer(corrected_vector_field)

        vector_field = self._match_vector_field(vector_field, full_size_moving)
        corrected_vector_field = self._match_vector_field(
            corrected_vector_field, full_size_moving
        )

        corrected_warped_moving = self._full_size_spatial_transformer(
            full_size_moving, corrected_vector_field
        )
        warped_moving = self._full_size_spatial_transformer(
            full_size_moving, vector_field
        )

        return (
            corrected_warped_moving,
            warped_moving,
            corrected_vector_field,
            vector_field,
            metrics_all_level
        )

    def forward(self, image, mask, moving, original_image_spacing):
        (
            corrected_warped_moving,
            warped_moving,
            corrected_vector_field,
            vector_field,
            metrics_all_level
        ) = self.warp_moving(image, mask, moving, original_image_spacing)

        return (
            corrected_warped_moving,
            warped_moving,
            corrected_vector_field,
            vector_field,
            metrics_all_level
        )
