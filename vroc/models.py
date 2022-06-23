import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.optim import Adam

from vroc.blocks import (
    ConvBlock,
    DecoderBlock,
    DemonForces3d,
    DownBlock,
    EncoderBlock,
    GaussianSmoothing3d,
    SpatialTransformer,
    UpBlock,
)
from vroc.checks import are_of_same_length, is_tuple, is_tuple_of_tuples
from vroc.common_types import FloatTuple, FloatTuple3D, IntTuple, IntTuple3D
from vroc.helper import get_bounding_box, rescale_range
from vroc.logger import LoggerMixin


class FlexUNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        n_levels: int = 7,
        filter_base: int = 32,
        convolution_layer=nn.Conv3d,
        downsampling_layer=nn.MaxPool3d,
        upsampling_layer=nn.Upsample,
        norm_layer=nn.BatchNorm3d,
        skip_connections=False,
        convolution_kwargs=None,
        downsampling_kwargs=None,
        upsampling_kwargs=None,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.filter_base = filter_base

        self.convolution_layer = convolution_layer
        self.downsampling_layer = downsampling_layer
        self.upsampling_layer = upsampling_layer
        self.norm_layer = norm_layer
        self.skip_connections = skip_connections

        self.convolution_kwargs = convolution_kwargs or {
            "kernel_size": 3,
            "padding": "same",
            "bias": False,
        }
        self.downsampling_kwargs = downsampling_kwargs or {"kernel_size": 2}
        self.upsampling_kwargs = upsampling_kwargs or {"scale_factor": 2}

        self._build_layers()

    @property
    def encoder_block(self):
        return EncoderBlock

    @property
    def decoder_block(self):
        return DecoderBlock

    def _build_layers(self):
        enc_out_channels = []

        self.init_conv = self.convolution_layer(
            in_channels=self.n_channels,
            out_channels=self.filter_base,
            **self.convolution_kwargs,
        )

        self.final_conv = self.convolution_layer(
            in_channels=self.filter_base,
            out_channels=self.n_classes,
            **self.convolution_kwargs,
        )

        enc_out_channels.append(self.filter_base)
        previous_out_channels = self.filter_base

        for i_level in range(self.n_levels):
            out_channels = self.filter_base * 2**i_level
            enc_out_channels.append(out_channels)
            self.add_module(
                f"enc_{i_level}",
                self.encoder_block(
                    in_channels=previous_out_channels,
                    out_channels=out_channels,
                    n_convolutions=2,
                    convolution_layer=self.convolution_layer,
                    downsampling_layer=self.downsampling_layer,
                    norm_layer=self.norm_layer,
                    convolution_kwargs=self.convolution_kwargs,
                    downsampling_kwargs=self.downsampling_kwargs,
                ),
            )
            previous_out_channels = out_channels

        for i_level in reversed(range(self.n_levels)):

            out_channels = self.filter_base * 2**i_level
            if i_level > 0:  # deeper levels
                if self.skip_connections:
                    in_channels = previous_out_channels + enc_out_channels[i_level]
                else:
                    in_channels = previous_out_channels
            else:
                if self.skip_connections:
                    in_channels = previous_out_channels + self.filter_base
                else:
                    in_channels = previous_out_channels

            self.add_module(
                f"dec_{i_level}",
                self.decoder_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_convolutions=2,
                    convolution_layer=self.convolution_layer,
                    upsampling_layer=self.upsampling_layer,
                    norm_layer=self.norm_layer,
                    convolution_kwargs=self.convolution_kwargs,
                    upsampling_kwargs=self.upsampling_kwargs,
                ),
            )
            previous_out_channels = out_channels

    def forward(self, *inputs, **kwargs):
        outputs = []
        inputs = self.init_conv(*inputs)
        outputs.append(inputs)
        for i_level in range(self.n_levels):
            inputs = self.get_submodule(f"enc_{i_level}")(inputs)
            outputs.append(inputs)

        for i_level in reversed(range(self.n_levels)):
            inputs = self.get_submodule(f"dec_{i_level}")(inputs, outputs[i_level])

        inputs = self.final_conv(inputs)

        return inputs, outputs[-1]


class AutoEncoder(FlexUNet):
    def forward(self, *inputs, **kwargs):
        outputs = []
        inputs = self.init_conv(*inputs)
        outputs.append(inputs)
        for i_level in range(self.n_levels):
            inputs = self.get_submodule(f"enc_{i_level}")(inputs)
            outputs.append(inputs)

        encoded_size = outputs[-1].size()
        embedded = F.avg_pool3d(outputs[-1], kernel_size=encoded_size[2:]).view(
            encoded_size[0], -1
        )
        inputs = embedded[(...,) + (None,) * len(encoded_size[2:])].repeat(
            (1, 1) + encoded_size[2:]
        )

        for i_level in reversed(range(self.n_levels)):
            inputs = self.get_submodule(f"dec_{i_level}")(inputs, None)

        inputs = self.final_conv(inputs)

        return inputs, embedded


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


class TrainableVarRegBlock(nn.Module, LoggerMixin):
    def __init__(
        self,
        scale_factors: Union[FloatTuple, float] = (1.0,),
        iterations: Union[IntTuple, int] = 100,
        tau: Union[FloatTuple, float] = 1.0,
        demon_forces: str = "active",
        regularization_sigma: Union[FloatTuple3D, Tuple[FloatTuple3D, ...]] = (
            1.0,
            1.0,
            1.0,
        ),
        regularization_radius: Union[IntTuple3D, Tuple[IntTuple3D, ...]] = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        use_image_spacing: bool = False,
        early_stopping_method: Optional[Tuple[str, ...]] = None,
        early_stopping_delta: Optional[Tuple[float, ...]] = None,
        early_stopping_window: Optional[Tuple[int, ...]] = None,
        restrict_to_mask: bool = False,
    ):
        super().__init__()

        if not is_tuple(scale_factors, min_length=1):
            scale_factors = (scale_factors,)
        self.scale_factors = scale_factors

        self.scale_factors = scale_factors  # this also defines "n_levels"
        self.iterations = TrainableVarRegBlock._expand_to_level_tuple(
            iterations, n_levels=self.n_levels
        )

        self.tau = TrainableVarRegBlock._expand_to_level_tuple(
            tau, n_levels=self.n_levels
        )
        self.regularization_sigma = TrainableVarRegBlock._expand_to_level_tuple(
            regularization_sigma, n_levels=self.n_levels, is_tuple=True
        )
        self.regularization_sigma = TrainableVarRegBlock._expand_to_level_tuple(
            regularization_sigma, n_levels=self.n_levels, is_tuple=True
        )
        self.regularization_radius = TrainableVarRegBlock._expand_to_level_tuple(
            regularization_radius, n_levels=self.n_levels, is_tuple=True
        )
        self.original_image_spacing = original_image_spacing
        self.use_image_spacing = use_image_spacing

        self.demon_forces = demon_forces

        self.early_stopping_fn = early_stopping_method
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_window = early_stopping_window

        # TODO: Implement restriction of reg to mask bbox
        self.restrict_to_mask = restrict_to_mask

        # check if params are passed with/converted to consistent length
        # (== self.n_levels)
        if not are_of_same_length(
            self.scale_factors,
            self.iterations,
            self.tau,
            self.regularization_sigma,
        ):
            raise ValueError("Inconsistent lengths of passed parameters")

        self._metrics = []
        self._counter = 0

        self._image_shape = None
        self._full_size_spatial_transformer = None

        self._demon_forces_layer = DemonForces3d()

        for i_level, sigma in enumerate(self.regularization_sigma):
            if self.regularization_radius:
                self.add_module(
                    name=f"regularization_layer_level_{i_level}",
                    module=GaussianSmoothing3d(
                        sigma=sigma,
                        sigma_cutoff=None,
                        radius=self.regularization_radius[i_level],
                        spacing=self.original_image_spacing,
                        use_image_spacing=self.use_image_spacing,
                    ),
                )
            else:
                self.add_module(
                    name=f"regularization_layer_level_{i_level}",
                    module=GaussianSmoothing3d(
                        sigma=sigma,
                        sigma_cutoff=(2.0, 2.0, 2.0),
                        force_same_size=True,
                        spacing=self.original_image_spacing,
                        use_image_spacing=self.use_image_spacing,
                    ),
                )

    @staticmethod
    def _expand_to_level_tuple(
        value: Any, n_levels: int, is_tuple: bool = False, skip_none: bool = True
    ) -> Optional[Tuple]:
        if skip_none and value is None:
            return value
        else:
            if isinstance(value, (int, float)):
                return (value,) * n_levels
            elif is_tuple and not is_tuple_of_tuples(value):
                return (value,) * n_levels
            else:
                return value

    @property
    def n_levels(self):
        return len(self.scale_factors)

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

    def _calculate_features(self, fixed_image, mask, moving_image, bins=128) -> dict:
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
                "std": to_numpy(image.std()),
            }

        return {
            "fixed_image": calculate_image_features(masked_fixed_image),
            "moving_image": calculate_image_features(masked_moving_image),
            "l2_difference": calculate_image_features(masked_l2_diff),
        }

    def _calculate_metric(self, fixed_image, mask, moving_image) -> float:
        return float(
            F.mse_loss(
                fixed_image[mask],
                moving_image[mask],
            )
        )

    def run_registration(self, fixed_image, mask, moving_image, original_image_spacing):
        if self.restrict_to_mask and mask is not None:
            bbox = get_bounding_box(mask, padding=5)
            self.logger.debug(f"Restricting registration to bounding box {bbox}")
            fixed_image = fixed_image[bbox]
            mask = mask[bbox]
            moving_image = moving_image[bbox]

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

        runtimes = []

        metric_before = self._calculate_metric(fixed_image, mask, moving_image)

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

            t_start = time.time()
            for i in range(iterations):
                t_step_start = time.time()
                warped_moving = spatial_transformer(scaled_moving_image, vector_field)

                level_metrics.append(
                    self._calculate_metric(
                        scaled_fixed_image, scaled_mask, warped_moving
                    )
                )
                log = {"level": i_level, "iteration": i, "metric": level_metrics[-1]}

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
                vector_field += self.tau[i_level] * (forces * scaled_mask)

                _regularization_layer = self.get_submodule(
                    f"regularization_layer_level_{i_level}",
                )
                vector_field = _regularization_layer(vector_field)

                t_step_end = time.time()
                log["step_runtime"] = t_step_end - t_step_start
                self.logger.debug(log)

            t_end = time.time()
            runtimes.append(t_end - t_start)
            metrics.append(level_metrics)
            features.append(level_features)

        vector_field = self._match_vector_field(vector_field, full_size_moving)
        warped_moving_image = self._full_size_spatial_transformer(
            full_size_moving, vector_field
        )

        metric_after = self._calculate_metric(fixed_image, mask, warped_moving_image)

        misc = {
            "features": features,
            "metric_before": metric_before,
            "metric_after": metric_after,
            "level_metrics": metrics,
            "runtimes": runtimes,
        }

        return warped_moving_image, vector_field, misc

    def forward(self, fixed_image, mask, moving_image, original_image_spacing):
        return self.run_registration(
            fixed_image=fixed_image,
            mask=mask,
            moving_image=moving_image,
            original_image_spacing=original_image_spacing,
        )


if __name__ == "__main__":
    model = TrainableVarRegBlock(
        scale_factors=1.0,
        iterations=200,
        demon_forces="symmetric",
        tau=2.0,
        regularization_sigma=(4.0, 2.0, 2.0),
        early_stopping_method=None,
    )
