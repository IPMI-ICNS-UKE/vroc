import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.optim import Adam

from vroc.blocks import (
    ConvBlock,
    DemonForces3d,
    DownBlock,
    GaussianSmoothing3d,
    SpatialTransformer,
    UpBlock,
)
from vroc.helper import rescale_range


class UNet(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 1, bilinear: bool = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = ConvBlock(n_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024 // factor)

        self.up1 = UpBlock(1024, 512 // factor, bilinear)
        self.up2 = UpBlock(512, 256 // factor, bilinear)
        self.up3 = UpBlock(256, 128 // factor, bilinear)
        self.up4 = UpBlock(128, 64, bilinear)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out(x)


class ParamNet(nn.Module):
    def __init__(
        self,
        params: dict,
        n_channels: int = 3,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.params = params
        self.n_params = len(self.params)

        self.conv_1 = DownBlock(
            in_channels=self.n_channels,
            out_channels=8,
            dimensions=1,
            norm_type="InstanceNorm",
        )
        self.conv_2 = DownBlock(
            in_channels=8, out_channels=16, dimensions=1, norm_type="InstanceNorm"
        )
        self.conv_3 = DownBlock(
            in_channels=16, out_channels=8, dimensions=1, norm_type="InstanceNorm"
        )
        self.conv_4 = DownBlock(
            in_channels=8, out_channels=4, dimensions=1, norm_type="InstanceNorm"
        )
        self.conv_5 = DownBlock(
            in_channels=4, out_channels=self.n_params, dimensions=1, norm_type=None
        )

    def forward(self, features):
        out = self.conv_1(features)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = torch.sigmoid(out)
        out = torch.squeeze(out)

        out_dict = {}
        # scale params according to min/max range
        for i_param, param_name in enumerate(self.params.keys()):
            out_min, out_max = (
                self.params[param_name]["min"],
                self.params[param_name]["max"],
            )
            out_dict[param_name] = (
                out[i_param] * (out_max - out_min)
            ) + out_min  # .to(self.params[param_name]['dtype'])

        return out_dict


class TrainableVarRegBlock(nn.Module):
    def __init__(
        self,
        iterations: Tuple[int, ...] = (100, 100),
        tau: Tuple[float, ...] = (1.0, 1.0),
        demon_forces="active",
        regularization_sigma=(1.0, 1.0, 1.0),
        original_image_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        disable_correction: bool = False,
        use_image_spacing: bool = False,
        scale_factors: Tuple[float, ...] = (0.5, 1.0),
        early_stopping_method: Optional[Tuple[str, ...]] = None,
        early_stopping_delta: Tuple[float, ...] = (0.0, 0.0),
        early_stopping_window: Tuple[int, ...] = (10, 10),
        radius: Tuple[int, ...] = None,
    ):
        super().__init__()
        self.iterations = iterations
        self.tau = tau
        self.regularization_sigma = regularization_sigma
        self.original_image_spacing = original_image_spacing
        self.disable_correction = disable_correction
        self.use_image_spacing = use_image_spacing
        self.scale_factors = scale_factors
        self.demon_forces = demon_forces

        self.early_stopping_fn = early_stopping_method
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_window = early_stopping_window
        self._metrics = []
        self._counter = 0

        self._image_shape = None
        self._full_size_spatial_transformer = None

        self._demon_forces_layer = DemonForces3d()

        for i_level, sigma in enumerate(regularization_sigma):
            if radius:
                self.add_module(
                    name=f"regularization_layer_level_{i_level}",
                    module=GaussianSmoothing3d(
                        sigma=sigma,
                        sigma_cutoff=(3.0, 3.0, 3.0),
                        same_size=True,
                        radius=radius[i_level],
                        spacing=self.original_image_spacing,
                        use_image_spacing=self.use_image_spacing,
                    ),
                )
            else:
                self.add_module(
                    name=f"regularization_layer_level_{i_level}",
                    module=GaussianSmoothing3d(
                        sigma=sigma,
                        sigma_cutoff=(3.0, 3.0, 3.0),
                        same_size=True,
                        spacing=self.original_image_spacing,
                        use_image_spacing=self.use_image_spacing,
                    ),
                )

    @property
    def config(self):
        return dict(
            iterations=self.iterations,
            tau=self.tau,
            regularization_sigma=self.regularization_sigma,
        )

    def _create_spatial_transformers(self, image_shape: Tuple[int, ...], device):
        if not image_shape == self._image_shape:
            self._image_shape = image_shape
            self._full_size_spatial_transformer = SpatialTransformer(
                shape=image_shape
            ).to(device)

            for i_level, scale_factor in enumerate(self.scale_factors):
                scaled_image_shape = tuple(int(s * scale_factor) for s in image_shape)

                try:
                    module = getattr(self, f"spatial_transformer_level_{i_level}")
                    del module
                except AttributeError:
                    pass
                self.add_module(
                    name=f"spatial_transformer_level_{i_level}",
                    module=SpatialTransformer(shape=scaled_image_shape).to(device),
                )

    # TODO: rename
    def _check_early_stopping_condition(self, metrics: List[float], i_level):
        if (
            self.early_stopping_delta[i_level] == 0
            or len(metrics) < self.early_stopping_window[i_level] + 1
        ):
            return False

    def _perform_scaling(self, image, mask, moving, scale_factor: float = 1.0):
        if len(image.shape) == 3:
            mode = "linear"
        elif len(image.shape) == 4:
            mode = "bilinear"
        elif len(image.shape) == 5:
            mode = "trilinear"
        else:
            mode = "nearest"
        image = F.interpolate(
            image, scale_factor=scale_factor, mode=mode, align_corners=False
        )
        # interpolation is not implemented for dtype bool
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = F.interpolate(mask, scale_factor=scale_factor, mode="nearest")
        mask = torch.as_tensor(mask, dtype=torch.bool)
        moving = F.interpolate(
            moving,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False,
        )

        return image, mask, moving

    def _match_vector_field(self, vector_field, image):
        vector_field_shape = vector_field.shape
        image_shape = image.shape

        if len(image_shape) == 4:
            mode = "bilinear"
        elif len(image_shape) == 5:
            mode = "trilinear"
        else:
            raise ValueError(f"Dimension {image_shape} not supported")

        if vector_field_shape[2:] == image_shape[2:]:
            return vector_field

        vector_field = F.interpolate(
            vector_field, size=image_shape[2:], mode=mode, align_corners=False
        )

        scale_factor = torch.tensor(
            [s1 / s2 for (s1, s2) in zip(image_shape[2:], vector_field_shape[2:])]
        )
        # 5D shape: (1, 3, 1, 1, 1), 4D shape: (1, 2, 1, 1)
        scale_factor = torch.reshape(
            scale_factor, (1, -1) + (1,) * (len(image_shape) - 2)
        )

        return vector_field * scale_factor.to(vector_field)

    def _calculate_features(self, fixed_image, mask, moving_image, bins=32) -> dict:
        mask = torch.as_tensor(mask, dtype=torch.bool)
        l2_diff = F.mse_loss(fixed_image, moving_image, reduction="none")

        masked_fixed_image = fixed_image[mask]
        masked_moving_image = moving_image[mask]
        masked_l2_diff = l2_diff[mask]

        def to_numpy(tensor: torch.Tensor) -> np.ndarray:
            t = tensor.detach().cpu().numpy()
            if t.ndim == 0:
                t = float(t)

            return t

        def calculate_image_features(image) -> dict:
            lower_percentile = torch.quantile(image, 0.05, interpolation="linear")
            upper_percentile = torch.quantile(image, 0.95, interpolation="linear")
            image_histogram = torch.histc(image, bins=bins)

            return {
                "histogram": to_numpy(image_histogram),
                "normalized_histogram": to_numpy(
                    image_histogram / image_histogram.sum()
                ),
                "percentile_5": to_numpy(lower_percentile),
                "percentile_95": to_numpy(upper_percentile),
                "min": to_numpy(image.min()),
                "max": to_numpy(image.max()),
                "mean": to_numpy(image.mean()),
                "median": to_numpy(image.median()),
            }

        return {
            "fixed_image": calculate_image_features(masked_fixed_image),
            "moving_image": calculate_image_features(masked_moving_image),
            "l2_difference": calculate_image_features(masked_l2_diff),
        }

    def run_registration(self, fixed_image, mask, moving_image, original_image_spacing):
        # create new spatial transformers if needed (skip batch and color dimension)
        self._create_spatial_transformers(
            fixed_image.shape[2:], device=fixed_image.device
        )

        # register moving image onto fixed image
        if len(fixed_image.shape) == 4:
            dim_vf = 2
        elif len(fixed_image.shape) == 5:
            dim_vf = 3
        full_size_moving = moving_image
        features = []
        metrics = []
        vector_field = None
        mask = torch.as_tensor(mask, dtype=torch.bool)

        for i_level, (scale_factor, iterations) in enumerate(
            zip(self.scale_factors, self.iterations)
        ):

            self._counter = 0

            (
                scaled_fixed_image,
                scaled_mask,
                scaled_moving_image,
            ) = self._perform_scaling(
                fixed_image, mask, moving_image, scale_factor=scale_factor
            )

            if vector_field is None:
                vector_field = torch.zeros(
                    scaled_fixed_image.shape[:1]
                    + (dim_vf,)
                    + scaled_fixed_image.shape[2:],
                    device=moving_image.device,
                )
            elif vector_field.shape[2:] != scaled_fixed_image.shape[2:]:
                vector_field = self._match_vector_field(
                    vector_field, scaled_fixed_image
                )

            spatial_transformer = self.get_submodule(
                f"spatial_transformer_level_{i_level}",
            )

            warped_scaled_moving_image = spatial_transformer(
                scaled_moving_image, vector_field
            )

            # calculate features of fixed/moving image at current level
            level_features = self._calculate_features(
                scaled_fixed_image, scaled_mask, warped_scaled_moving_image
            )
            level_features["current_level"] = i_level
            level_features["scale_factors"] = self.scale_factors
            level_metrics = []

            for i in range(iterations):
                # if (i % 100) == 0:
                #     print(f"*** LEVEL {i_level + 1} *** ITERATION {i} ***")

                warped_moving = spatial_transformer(scaled_moving_image, vector_field)

                level_metrics.append(
                    float(
                        F.mse_loss(
                            scaled_fixed_image[scaled_mask],
                            warped_moving[scaled_mask],
                        )
                    )
                )

                # if self.early_stopping_fn[i_level] == "lstsq":
                #     if self._check_early_stopping_lstsq(metrics, i_level):
                #         break
                # elif self.early_stopping_fn[i_level] == "no_impr":
                #     if self._check_early_stopping_increase_count(metrics, i_level):
                #         break
                # elif self.early_stopping_fn[i_level] == "no_average_impr":
                #     if self._check_early_stopping_average_improvement(metrics, i_level):
                #         break
                # elif self.early_stopping_fn[i_level] == "none":
                #     pass
                # else:
                #     raise Exception(
                #         f"Early stopping method {self.early_stopping_fn[i_level]} "
                #         f"is not implemented"
                #     )

                forces = self._demon_forces_layer(
                    warped_moving,
                    scaled_fixed_image,
                    self.demon_forces,
                    original_image_spacing,
                )
                # mask forces with artifact mask (artifact = 0, valid = 1)

                vector_field += self.tau[i_level] * (forces * scaled_mask)
                # vector_field += predicted_params["tau"] * (forces * scaled_mask)
                _regularization_layer = self.get_submodule(
                    f"regularization_layer_level_{i_level}",
                )
                vector_field = _regularization_layer(vector_field)

            # print(f'LEVEL {i_level + 1} stopped at ITERATION {i + 1}: Metric: {metrics[-1]}')

            metrics.append(level_metrics)
            features.append(level_features)

        vector_field = self._match_vector_field(vector_field, full_size_moving)
        warped_moving_image = self._full_size_spatial_transformer(
            full_size_moving, vector_field
        )

        misc = {"features": features, "metrics": metrics}

        return warped_moving_image, vector_field, misc

    def forward(self, fixed_image, mask, moving_image, original_image_spacing):
        return self.run_registration(
            fixed_image=fixed_image,
            mask=mask,
            moving_image=moving_image,
            original_image_spacing=original_image_spacing,
        )
