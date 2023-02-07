from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vroc.blocks import (
    ConvBlock,
    DecoderBlock,
    DemonForces,
    DownBlock,
    EncoderBlock,
    GaussianSmoothing2d,
    GaussianSmoothing3d,
    NCCForces,
    NGFForces,
    SpatialTransformer,
    UpBlock,
)
from vroc.checks import are_of_same_length, is_tuple, is_tuple_of_tuples
from vroc.common_types import (
    FloatTuple2D,
    FloatTuple3D,
    IntTuple2D,
    IntTuple3D,
    MaybeSequence,
)
from vroc.decay import exponential_decay
from vroc.decorators import timing
from vroc.helper import get_bounding_box
from vroc.interpolation import rescale, resize
from vroc.logger import LoggerMixin


# TODO: channel bug with skip_connections=False
class FlexUNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        n_levels: int = 6,
        filter_base: int | None = None,
        n_filters: Sequence[int] | None = None,
        convolution_layer=nn.Conv3d,
        downsampling_layer=nn.MaxPool3d,
        upsampling_layer=nn.Upsample,
        norm_layer=nn.BatchNorm3d,
        skip_connections=False,
        convolution_kwargs=None,
        downsampling_kwargs=None,
        upsampling_kwargs=None,
        return_bottleneck: bool = True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_levels = n_levels

        # either filter_base or n_filters must be set
        self.filter_base = filter_base
        self.n_filters = n_filters

        if not any((filter_base, n_filters)) or all((filter_base, n_filters)):
            raise ValueError("Please set either filter_base or n_filters")

        self.convolution_layer = convolution_layer
        self.downsampling_layer = downsampling_layer
        self.upsampling_layer = upsampling_layer
        self.norm_layer = norm_layer
        self.skip_connections = skip_connections

        self.convolution_kwargs = convolution_kwargs or {
            "kernel_size": 3,
            "padding": "same",
            "bias": True,
        }
        self.downsampling_kwargs = downsampling_kwargs or {"kernel_size": 2}
        self.upsampling_kwargs = upsampling_kwargs or {"scale_factor": 2}

        self.return_bottleneck = return_bottleneck

        self._build_layers()

    @property
    def encoder_block(self):
        return EncoderBlock

    @property
    def decoder_block(self):
        return DecoderBlock

    def _build_layers(self):

        if self.filter_base:
            n_filters = {
                "init": self.filter_base,
                "enc": [
                    self.filter_base * 2**i_level for i_level in range(self.n_levels)
                ],
                "dec": [
                    self.filter_base * 2**i_level
                    for i_level in reversed(range(self.n_levels))
                ],
                "final": self.filter_base,
            }
        else:
            n_filters = {
                "init": self.n_filters[0],
                "enc": self.n_filters[1 : self.n_levels + 1],
                "dec": self.n_filters[self.n_levels + 1 : -1],
                "final": self.n_filters[-1],
            }

        enc_out_channels = []

        self.init_conv = self.convolution_layer(
            in_channels=self.n_channels,
            out_channels=n_filters["init"],
            **self.convolution_kwargs,
        )

        self.final_conv = self.convolution_layer(
            in_channels=n_filters["final"],
            out_channels=self.n_classes,
            **self.convolution_kwargs,
        )

        enc_out_channels.append(n_filters["init"])
        previous_out_channels = n_filters["init"]

        for i_level in range(self.n_levels):
            out_channels = n_filters["enc"][i_level]
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

        for i, i_level in enumerate(reversed(range(self.n_levels))):

            out_channels = n_filters["dec"][i]

            if i_level > 0:  # deeper levels
                if self.skip_connections:
                    in_channels = previous_out_channels + enc_out_channels[i_level]
                else:
                    in_channels = previous_out_channels
            else:
                if self.skip_connections:
                    in_channels = previous_out_channels + n_filters["init"]
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

        if self.return_bottleneck:
            return inputs, outputs[-1]
        else:
            return inputs


class Unet3d(FlexUNet):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        n_levels: int = 6,
        filter_base: int | None = None,
        n_filters: Sequence[int] | None = None,
    ):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            n_levels=n_levels,
            filter_base=filter_base,
            n_filters=n_filters,
            convolution_layer=nn.Conv3d,
            downsampling_layer=nn.MaxPool3d,
            upsampling_layer=nn.Upsample,
            norm_layer=nn.InstanceNorm3d,
            skip_connections=True,
            convolution_kwargs=None,
            downsampling_kwargs=None,
            upsampling_kwargs=None,
        )

    def forward(self, *inputs, **kwargs):
        prediction, _ = super().forward(*inputs, **kwargs)

        return prediction


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


class VarReg(nn.Module, LoggerMixin):
    _INTERPOLATION_MODES = {
        2: "bilinear",
        3: "trilinear",
    }

    _GAUSSIAN_SMOOTHING = {2: GaussianSmoothing2d, 3: GaussianSmoothing3d}

    def __init__(
        self,
        scale_factors: MaybeSequence[float] = (1.0,),
        use_masks: MaybeSequence[bool] = True,
        iterations: MaybeSequence[int] = 100,
        tau: MaybeSequence[float] = 1.0,
        tau_level_decay: float = 0.0,
        tau_iteration_decay: float = 0.0,
        force_type: Literal["demons", "ncc", "ngf"] = "demons",
        gradient_type: Literal["active", "passive", "dual"] = "dual",
        regularization_sigma: MaybeSequence[FloatTuple2D | FloatTuple3D] = (
            1.0,
            1.0,
            1.0,
        ),
        regularization_radius: MaybeSequence[IntTuple2D | IntTuple3D] | None = None,
        sigma_level_decay: float = 0.0,
        sigma_iteration_decay: float = 0.0,
        original_image_spacing: FloatTuple2D | FloatTuple3D = (1.0, 1.0, 1.0),
        use_image_spacing: bool = False,
        restrict_to_mask_bbox: bool = False,
        early_stopping_delta: float = 0.0,
        early_stopping_window: int | None = 20,
        boosting_model: nn.Module | None = None,
        debug: bool = False,
        debug_output_folder: Path | None = None,
        debug_step_size: int = 10,
    ):
        super().__init__()

        n_spatial_dims = len(original_image_spacing)

        if not is_tuple(scale_factors, min_length=1):
            scale_factors = (scale_factors,)
        self.scale_factors = scale_factors

        self.scale_factors = scale_factors  # this also defines "n_levels"
        self.use_masks = VarReg._expand_to_level_tuple(
            use_masks, n_levels=self.n_levels
        )
        self.iterations = VarReg._expand_to_level_tuple(
            iterations, n_levels=self.n_levels
        )

        self.tau = VarReg._expand_to_level_tuple(tau, n_levels=self.n_levels)
        self.tau_level_decay = tau_level_decay
        self.tau_iteration_decay = tau_iteration_decay

        self.regularization_sigma = VarReg._expand_to_level_tuple(
            regularization_sigma, n_levels=self.n_levels, is_tuple=True
        )
        self.regularization_radius = VarReg._expand_to_level_tuple(
            regularization_radius, n_levels=self.n_levels, is_tuple=True
        )
        self.sigma_level_decay = sigma_level_decay
        self.sigma_iteration_decay = sigma_iteration_decay
        self.original_image_spacing = original_image_spacing
        self.use_image_spacing = use_image_spacing

        self.forces = gradient_type

        self.restrict_to_mask_bbox = restrict_to_mask_bbox
        self.early_stopping_delta = VarReg._expand_to_level_tuple(
            early_stopping_delta, n_levels=self.n_levels
        )
        self.early_stopping_window = VarReg._expand_to_level_tuple(
            early_stopping_window, n_levels=self.n_levels, expand_none=True
        )
        self.boosting_model = boosting_model
        self.debug = debug
        self.debug_output_folder = debug_output_folder
        self.debug_step_size = debug_step_size

        if self.debug:
            from vroc.plot import RegistrationProgressPlotter

            self._plotter = RegistrationProgressPlotter(
                output_folder=self.debug_output_folder
            )
        else:
            self._plotter = None

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

        if force_type == "demons":
            self._forces_layer = DemonForces(method=self.forces)
        elif force_type == "ncc":
            self._forces_layer = NCCForces(method=self.forces)
        elif force_type == "ngf":
            self._forces_layer = NGFForces()
        else:
            raise NotImplementedError(
                f"Registration variant {force_type} is not implemented"
            )

        # gaussian_smoothing = self._GAUSSIAN_SMOOTHING[n_spatial_dims]
        #
        # for i_level, sigma in enumerate(self.regularization_sigma):
        #     if self.regularization_radius:
        #         # gaussian smoothing has fixed kernel radius
        #         self.add_module(
        #             name=f"regularization_layer_level_{i_level}",
        #             module=gaussian_smoothing(
        #                 sigma=sigma,
        #                 sigma_cutoff=None,
        #                 radius=self.regularization_radius[i_level],
        #                 spacing=self.original_image_spacing,
        #                 use_image_spacing=self.use_image_spacing,
        #             ),
        #         )
        #     else:
        #         # gaussian smoothing kernel radius is defined by
        #         # sigma and sigma cutoff (here 2)
        #         sigma_cutoff = (2, 2, 2)[:n_spatial_dims]
        #         self.add_module(
        #             name=f"regularization_layer_level_{i_level}",
        #             module=gaussian_smoothing(
        #                 sigma=sigma,
        #                 sigma_cutoff=sigma_cutoff,
        #                 force_same_size=True,
        #                 spacing=self.original_image_spacing,
        #                 use_image_spacing=self.use_image_spacing,
        #             ),
        #         )

    @staticmethod
    def _expand_to_level_tuple(
        value: Any, n_levels: int, is_tuple: bool = False, expand_none: bool = False
    ) -> Optional[Tuple]:
        if not expand_none and value is None:
            return value
        else:
            if not is_tuple:
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
                scaled_image_shape = tuple(
                    int(round(s * scale_factor)) for s in image_shape
                )

                try:
                    module = getattr(self, f"spatial_transformer_level_{i_level}")
                    del module
                except AttributeError:
                    pass
                self.add_module(
                    name=f"spatial_transformer_level_{i_level}",
                    module=SpatialTransformer(shape=scaled_image_shape).to(device),
                )

    def _perform_scaling(
        self, *images, scale_factor: float = 1.0
    ) -> List[torch.Tensor]:

        scaled = []
        for image in images:
            if image is not None:
                is_mask = image.dtype == torch.bool
                order = 0 if is_mask else 1

                image = rescale(image, factor=scale_factor, order=1)

                # if image.dtype == torch.bool:
                #     # image is mask
                #     # interpolation is not implemented for dtype bool
                #     # also use NN interpolation
                #     image = torch.as_tensor(image, dtype=torch.uint8)
                #     image = F.interpolate(
                #         image, scale_factor=scale_factor, mode="nearest"
                #     )
                #     image = torch.as_tensor(image, dtype=torch.bool)
                # else:
                #     # normal image (moving or fixed)
                #     mode = VarReg._INTERPOLATION_MODES[image.ndim - 2]
                #     image = F.interpolate(
                #         image, scale_factor=scale_factor, mode=mode, align_corners=True
                #     )

            scaled.append(image)

        return scaled

    def _match_vector_field(
        self, vector_field: torch.Tensor, image: torch.Tensor
    ) -> torch.Tensor:
        vector_field_shape = vector_field.shape
        image_shape = image.shape

        if vector_field_shape[2:] == image_shape[2:]:
            # vector field and image are already the same size
            return vector_field

        mode = VarReg._INTERPOLATION_MODES[image.ndim - 2]
        # vector_field = F.interpolate(
        #     vector_field, size=image_shape[2:], mode=mode, align_corners=True
        # )
        vector_field = resize(vector_field, output_shape=image_shape[2:], order=1)

        # scale factor to scale the vector field values
        scale_factor = torch.tensor(
            [s1 / s2 for (s1, s2) in zip(image_shape[2:], vector_field_shape[2:])]
        )
        # 5D shape: (1, 3, 1, 1, 1), 4D shape: (1, 2, 1, 1)
        scale_factor = torch.reshape(
            scale_factor, (1, -1) + (1,) * (len(image_shape) - 2)
        )

        return vector_field * scale_factor.to(vector_field)

    def _calculate_metric(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        fixed_mask: torch.Tensor | None = None,
    ) -> float:
        if fixed_mask is not None:
            fixed_image = fixed_image[fixed_mask]
            moving_image = moving_image[fixed_mask]

        return float(F.mse_loss(fixed_image, moving_image))

    def _check_early_stopping(self, metrics: List[float], i_level: int) -> bool:
        early_stop = False
        if (
            self.early_stopping_delta[i_level]
            and self.early_stopping_window[i_level]
            and len(metrics) >= self.early_stopping_window[i_level]
        ):
            window = np.array(metrics[-self.early_stopping_window[i_level] :])
            window_rel_changes = 1 - window[1:] / window[:-1]

            mean_rel_change = window_rel_changes.mean()

            self.logger.debug(
                f"Mean relative metric change over window of "
                f"{self.early_stopping_window[i_level]} steps is {mean_rel_change:.6f}"
            )

            if mean_rel_change < self.early_stopping_delta[i_level]:
                early_stop = True
                self.logger.debug(
                    f"Early stopping triggered for level {i_level} after iteration "
                    f"{len(metrics)}: {mean_rel_change:.6f} < "
                    f"{self.early_stopping_delta[i_level]:.6f}"
                )

        return early_stop

    def run_registration(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
    ):
        device = moving_image.device
        if moving_image.ndim != fixed_image.ndim:
            raise RuntimeError("Dimension mismatch betwen moving and fixed image")
        # define dimensionalities and shapes
        n_image_dimensions = moving_image.ndim
        n_spatial_dims = n_image_dimensions - 2  # -1 batch dim, -1 color dim

        full_uncropped_shape = tuple(fixed_image.shape)

        original_moving_image = moving_image
        original_moving_mask = moving_mask
        original_initial_vector_field = initial_vector_field

        if self.restrict_to_mask_bbox and (
            moving_mask is not None or fixed_mask is not None
        ):
            masks = [m for m in (moving_mask, fixed_mask) if m is not None]
            if len(masks) == 2:
                # we compute the union of both masks to get the overall bounding box
                union_mask = torch.logical_or(*masks)
            else:
                union_mask = masks[0]

            bbox = get_bounding_box(union_mask, padding=5)
            self.logger.debug(f"Restricting registration to bounding box {bbox}")

            moving_image = moving_image[bbox]
            fixed_image = fixed_image[bbox]
            if moving_mask is not None:
                moving_mask = moving_mask[bbox]
            if fixed_mask is not None:
                fixed_mask = fixed_mask[bbox]
            if initial_vector_field is not None:
                initial_vector_field = initial_vector_field[(..., *bbox[2:])]
        else:
            bbox = ...

        if moving_mask is not None:
            moving_mask = torch.as_tensor(moving_mask, dtype=torch.bool)
        if fixed_mask is not None:
            fixed_mask = torch.as_tensor(fixed_mask, dtype=torch.bool)

        # create new spatial transformers if needed (skip batch and color dimension)
        self._create_spatial_transformers(
            fixed_image.shape[2:], device=fixed_image.device
        )

        full_size_moving = moving_image
        full_cropped_shape = tuple(fixed_image.shape)
        full_cropped_spatial_shape = full_cropped_shape[2:]

        # for tracking level-wise metrics (used for early stopping, if enabled)
        metrics = []
        vector_field = None

        metric_before = self._calculate_metric(
            moving_image=moving_image, fixed_image=fixed_image, fixed_mask=fixed_mask
        )

        for i_level, (scale_factor, iterations) in enumerate(
            zip(self.scale_factors, self.iterations)
        ):
            self.logger.info(
                f"Start level {i_level + 1}/{self.n_levels} with {scale_factor=} and {iterations=}"
            )
            self._counter = 0

            (
                scaled_moving_image,
                scaled_fixed_image,
                scaled_moving_mask,
                scaled_fixed_mask,
            ) = self._perform_scaling(
                moving_image,
                fixed_image,
                moving_mask,
                fixed_mask,
                scale_factor=scale_factor,
            )

            if vector_field is None:
                # create an empty (all zero) vector field
                vector_field = torch.zeros(
                    scaled_fixed_image.shape[:1]
                    + (n_spatial_dims,)
                    + scaled_fixed_image.shape[2:],
                    device=moving_image.device,
                )
            else:
                # this will also scale the initial (full scale) vector field in the
                # first iteration
                vector_field = self._match_vector_field(
                    vector_field, scaled_fixed_image
                )

            if initial_vector_field is not None:
                scaled_initial_vector_field = self._match_vector_field(
                    initial_vector_field, scaled_fixed_image
                )

            spatial_transformer = self.get_submodule(
                f"spatial_transformer_level_{i_level}",
            )

            level_metrics = []

            for i_iteration in range(iterations):
                t_step_start = time.time()
                if initial_vector_field is not None:
                    # if we have an initial vector field: compose both vector fields.
                    # Here: resolution defined by given registration level
                    composed_vector_field = vector_field + spatial_transformer(
                        scaled_initial_vector_field, vector_field
                    )
                else:
                    composed_vector_field = vector_field

                warped_moving = spatial_transformer(
                    scaled_moving_image,
                    composed_vector_field,
                )
                warped_scaled_moving_mask = spatial_transformer(
                    scaled_moving_mask, composed_vector_field
                )
                # warped_scaled_moving_mask = spatial_transformer(
                #     scaled_moving_mask, scaled_initial_vector_field
                # )

                level_metrics.append(
                    self._calculate_metric(
                        moving_image=warped_moving,
                        fixed_image=scaled_fixed_image,
                        fixed_mask=scaled_fixed_mask,
                    )
                )

                forces = self._forces_layer(
                    warped_moving,
                    scaled_fixed_image,
                    warped_scaled_moving_mask,
                    scaled_fixed_mask,
                    original_image_spacing,
                    use_masks=self.use_masks[i_level],
                )
                decayed_tau = exponential_decay(
                    initial_value=self.tau[i_level],
                    i_level=i_level,
                    i_iteration=i_iteration,
                    level_lambda=self.tau_level_decay,
                    iteration_lambda=self.tau_iteration_decay,
                )

                vector_field += decayed_tau * forces

                # _regularization_layer = self.get_submodule(
                #     f"regularization_layer_level_{i_level}",
                # )

                decayed_sigma = tuple(
                    exponential_decay(
                        initial_value=s,
                        i_level=i_level,
                        i_iteration=i_iteration,
                        level_lambda=self.sigma_level_decay,
                        iteration_lambda=self.sigma_iteration_decay,
                    )
                    for s in self.regularization_sigma[i_level]
                )
                sigma_cutoff = (2.0, 2.0, 2.0)[:n_spatial_dims]
                gaussian_smoothing = self._GAUSSIAN_SMOOTHING[n_spatial_dims]
                _regularization_layer = gaussian_smoothing(
                    sigma=decayed_sigma,
                    sigma_cutoff=sigma_cutoff,
                    force_same_size=True,
                    spacing=self.original_image_spacing,
                    use_image_spacing=self.use_image_spacing,
                ).to(device)

                vector_field = _regularization_layer(vector_field)

                # check early stopping
                if self._check_early_stopping(metrics=level_metrics, i_level=i_level):
                    break

                log = {
                    "level": i_level,
                    "iteration": i_iteration,
                    "tau": decayed_tau,
                    "metric": level_metrics[-1],
                }

                t_step_end = time.time()
                log["step_runtime"] = t_step_end - t_step_start
                self.logger.debug(log)

                # here we do the debug stuff
                if self.debug and (
                    i_iteration % self.debug_step_size == 0 or i_iteration < 10
                ):
                    debug_metrics = {
                        "tau": decayed_tau,
                        "metric": level_metrics[-1],
                        "level_image_shape": tuple(scaled_moving_image.shape),
                        "vector_field": {},
                    }

                    dim_names = ("x", "y", "z")
                    for i_dim in range(n_spatial_dims):
                        debug_metrics["vector_field"][dim_names[i_dim]] = {
                            "min": float(torch.min(vector_field[:, i_dim])),
                            # "Q0.05": float(
                            #     torch.quantile(vector_field[:, i_dim], 0.05)
                            # ),
                            "mean": float(torch.mean(vector_field[:, i_dim])),
                            # "Q0.95": float(
                            #     torch.quantile(vector_field[:, i_dim], 0.95)
                            # ),
                            "max": float(torch.max(vector_field[:, i_dim])),
                        }
                    self._plotter.save_snapshot(
                        moving_image=scaled_moving_image,
                        fixed_image=scaled_fixed_image,
                        warped_image=warped_moving,
                        forces=forces,
                        vector_field=vector_field,
                        moving_mask=scaled_moving_mask,
                        fixed_mask=scaled_fixed_mask,
                        warped_mask=warped_scaled_moving_mask,
                        full_spatial_shape=full_cropped_spatial_shape,
                        stage="vroc",
                        level=i_level,
                        scale_factor=scale_factor,
                        iteration=i_iteration,
                        metrics=debug_metrics,
                        output_folder=self.debug_output_folder,
                    )
                    self.logger.debug(
                        f"Created snapshot of registration at level={i_level}, iteration={i_iteration}"
                    )

            metrics.append(level_metrics)

        vector_field = self._match_vector_field(vector_field, full_size_moving)

        if self.restrict_to_mask_bbox:
            # undo restriction to mask, i.e. insert results into full size data
            _vector_field = torch.zeros(
                vector_field.shape[:2] + original_moving_image.shape[2:],
                device=vector_field.device,
            )
            _vector_field[(...,) + bbox[2:]] = vector_field
            vector_field = _vector_field

        spatial_transformer = SpatialTransformer(shape=full_uncropped_shape[2:]).to(
            fixed_image.device
        )

        result = {}

        if initial_vector_field is not None:
            # if we have an initial vector field: compose both vector fields.
            # Here: at full resolution without cropping/bbox
            composed_vector_field = vector_field + spatial_transformer(
                original_initial_vector_field, vector_field
            )
            result["vector_fields"] = [original_initial_vector_field, vector_field]
        else:
            composed_vector_field = vector_field
            result["vector_fields"] = [vector_field]

        result["composed_vector_field"] = composed_vector_field
        warped_moving_image = spatial_transformer(
            original_moving_image, composed_vector_field
        )
        warped_moving_image = spatial_transformer(
            original_moving_image, composed_vector_field
        )
        if original_moving_mask is not None:
            warped_moving_mask = spatial_transformer(
                original_moving_mask, composed_vector_field
            )
        else:
            warped_moving_mask = None

        if initial_vector_field is not None:
            warped_affine_moving_image = spatial_transformer(
                original_moving_image, original_initial_vector_field
            )

            if original_moving_mask is not None:
                warped_affine_moving_mask = spatial_transformer(
                    original_moving_mask, original_initial_vector_field
                )
            else:
                warped_affine_moving_mask = None
        else:
            warped_affine_moving_image = None
            warped_affine_moving_mask = None

        metric_after = self._calculate_metric(
            moving_image=warped_moving_image[bbox],
            fixed_image=fixed_image,
            fixed_mask=fixed_mask,
        )

        if self.debug:
            debug_info = {
                "metric_before": metric_before,
                "metric_after": metric_after,
                "level_metrics": metrics,
            }

        else:
            debug_info = None
        result["debug_info"] = debug_info
        result["warped_moving_image"] = warped_moving_image
        result["warped_moving_mask"] = warped_moving_mask
        result["warped_affine_moving_mask"] = warped_affine_moving_mask
        result["warped_affine_moving_image"] = warped_affine_moving_image

        return result

    @timing()
    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        original_image_spacing: FloatTuple3D,
    ):
        return self.run_registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            original_image_spacing=original_image_spacing,
        )


class DemonsVectorFieldBooster(nn.Module, LoggerMixin):
    def __init__(
        self,
        n_iterations: int = 10,
        gradient_type: Literal["active", "passive", "dual"] = "dual",
    ):
        super().__init__()

        self.n_iterations = n_iterations
        self.forces = DemonForces(method=gradient_type)

        regularization_1 = [
            nn.Conv3d(
                in_channels=4,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]
        regularization_2 = [
            nn.Conv3d(
                in_channels=4,
                out_channels=16,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        fuse = [
            nn.Conv3d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        self.regularization_1 = nn.Sequential(*regularization_1)
        self.regularization_2 = nn.Sequential(*regularization_2)
        self.fuse = nn.Sequential(*fuse)
        self.spatial_transformer = SpatialTransformer()

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        spatial_image_shape = moving_image.shape[2:]
        vector_field_boost = torch.zeros(
            (1, 3) + spatial_image_shape, device=moving_image.device
        )

        for _ in range(self.n_iterations):
            composed_vector_field = vector_field_boost + self.spatial_transformer(
                vector_field, vector_field_boost
            )
            warped_moving_image = self.spatial_transformer(
                moving_image, composed_vector_field
            )

            diff = (warped_moving_image - fixed_image) / (fixed_image + 1e-6)
            diff = F.softsign(diff)

            forces = self.forces(
                moving_image,
                fixed_image,
                moving_mask,
                fixed_mask,
                image_spacing,
            )

            updated_vector_field_boost = vector_field_boost + forces

            updated_vector_field_boost_1 = self.regularization_1(
                torch.concat((updated_vector_field_boost, diff), dim=1)
            )
            updated_vector_field_boost_2 = self.regularization_2(
                torch.concat((updated_vector_field_boost, diff), dim=1)
            )
            vector_field_boost = vector_field_boost + self.fuse(
                torch.concat(
                    (updated_vector_field_boost_1, updated_vector_field_boost_2),
                    dim=1,
                )
            )

        return vector_field_boost


# class DemonsVectorFieldBooster(nn.Module, LoggerMixin):
#     def __init__(
#         self,
#         n_iterations: int = 10,
#         filter_base: int = 16,
#         gradient_type: Literal["active", "passive", "dual"] = "dual",
#     ):
#         super().__init__()
#
#         self.n_iterations = n_iterations
#         self.filter_base = filter_base
#         self.forces = DemonForces(method=gradient_type)
#
#         # self.regularization = TrainableRegularization3d(n_levels=4, filter_base=16)
#         self.regularization = DynamicRegularization3d(filter_base=16)
#         self.spatial_transformer = SpatialTransformer()
#
#
#         self.factors = (0.125, 0.25, 0.5, 1.0)
#         self.n_levels = len(self.factors)
#         self.weighting_net = FlexUNet(
#             n_channels=2, n_levels=4, n_classes=self.n_levels + 3, filter_base=4, norm_layer=nn.InstanceNorm3d, return_bottleneck=False, skip_connections=True
#         )
#
#
#
#     def forward(
#         self,
#         moving_image: torch.Tensor,
#         fixed_image: torch.Tensor,
#         moving_mask: torch.Tensor,
#         fixed_mask: torch.Tensor,
#         vector_field: torch.Tensor,
#         image_spacing: torch.Tensor,
#         n_iterations: int | None = None,
#     ) -> torch.Tensor:
#
#         spatial_image_shape = moving_image.shape[2:]
#         vector_field_boost = torch.zeros(
#             (1, 3) + spatial_image_shape, device=moving_image.device
#         )
#
#         _n_iterations = n_iterations or self.n_iterations
#         for _ in range(_n_iterations):
#             composed_vector_field = self.spatial_transformer.compose_vector_fields(
#                 vector_field, vector_field_boost
#             )
#
#             # warp image with boosted vector field
#             warped_moving_image = self.spatial_transformer(
#                 moving_image, composed_vector_field
#             )
#
#             # diff = (warped_moving_image - fixed_image) / (fixed_image + 1e-6)
#             # diff = F.softsign(diff)
#
#             forces = self.forces(
#                 warped_moving_image,
#                 fixed_image,
#                 moving_mask,
#                 fixed_mask,
#                 image_spacing,
#             )
#
#             images = torch.concat((moving_image, fixed_image), dim=1)
#
#             output = self.weighting_net(images)
#             weights = output[:, :self.n_levels]
#             weights = torch.softmax(weights, dim=1)
#             taus = output[:, self.n_levels:]
#             taus = 5 * torch.sigmoid(taus)
#             print(f'mean tau x/y/z: {taus[:, 0].mean():.2f}, {taus[:, 1].mean():.2f}, {taus[:, 2].mean():.2f}')
#
#             vector_field_boost = vector_field_boost + taus * forces
#             vector_field_boost = self.regularization(
#                 vector_field=vector_field_boost, moving_image=warped_moving_image, fixed_image=fixed_image, weights=weights
#             )
#
#             # plot weights and tau
#             m = warped_moving_image.detach().cpu().numpy()
#             f = fixed_image.detach().cpu().numpy()
#             diff = (warped_moving_image - fixed_image)
#             diff = diff.detach().cpu().numpy()
#             w = weights.detach().cpu().numpy()
#
#             m, f = diff, diff
#             clim = (-1, 1)
#             cmap = 'seismic'
#             mid_slice = w.shape[-2] // 2
#             fig, ax = plt.subplots(1, self.n_levels + 2, sharex=True, sharey=True)
#             ax[0].imshow(f[0, 0, :, mid_slice, :], clim=clim, cmap=cmap)
#             ax[1].imshow(m[0, 0, :, mid_slice, :], clim=clim, cmap=cmap)
#             for i in range(self.n_levels):
#                 ax[i + 2].imshow(w[0, i, :, mid_slice, :])
#
#             t = taus.detach().cpu().numpy()
#             mid_slice = t.shape[-2] // 2
#             fig, ax = plt.subplots(1, 3 + 2, sharex=True, sharey=True)
#             ax[0].imshow(f[0, 0, :, mid_slice, :], clim=clim, cmap=cmap)
#             ax[1].imshow(m[0, 0, :, mid_slice, :], clim=clim, cmap=cmap)
#             for i in range(3):
#                 ax[i + 2].imshow(t[0, i, :, mid_slice, :])
#
#         return vector_field_boost

# def forward(
#     self,
#     moving_image: torch.Tensor,
#     fixed_image: torch.Tensor,
#     moving_mask: torch.Tensor,
#     fixed_mask: torch.Tensor,
#     vector_field: torch.Tensor,
#     image_spacing: torch.Tensor,
#     n_iterations: int | None = None,
# ) -> torch.Tensor:
#
#     spatial_image_shape = moving_image.shape[2:]
#     vector_field_boost = torch.zeros(
#         (1, 3) + spatial_image_shape, device=moving_image.device
#     )
#
#     _n_iterations = n_iterations or self.n_iterations
#     for _ in range(_n_iterations):
#         composed_vector_field = self.spatial_transformer.compose_vector_fields(
#             vector_field, vector_field_boost
#         )
#
#         # warp image with boosted vector field
#         warped_moving_image = self.spatial_transformer(
#             moving_image, composed_vector_field
#         )
#
#         diff = (warped_moving_image - fixed_image) / (fixed_image + 1e-6)
#         diff = F.softsign(diff)
#
#         forces = self.forces(
#             warped_moving_image,
#             fixed_image,
#             moving_mask,
#             fixed_mask,
#             image_spacing,
#         )
#
#         updated_vector_field_boost = vector_field_boost + forces
#
#         updated_vector_field_boost_1 = self.regularization_1(
#             torch.concat((updated_vector_field_boost, diff), dim=1)
#         )
#         updated_vector_field_boost_2 = self.regularization_2(
#             torch.concat((updated_vector_field_boost, diff), dim=1)
#         )
#         vector_field_boost = vector_field_boost + self.fuse(
#             torch.concat(
#                 (updated_vector_field_boost_1, updated_vector_field_boost_2),
#                 dim=1,
#             )
#         )
#
#     return vector_field_boost


class DemonsVectorFieldArtifactBooster(nn.Module):
    def __init__(self, shape: Tuple[int, int, int]):
        super().__init__()
        self.shape = shape

        boost_layers_1 = [
            nn.Conv3d(
                in_channels=4,
                out_channels=16,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        boost_layers_2 = [
            nn.Conv3d(
                in_channels=4,
                out_channels=16,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        fuse_layers = [
            nn.Conv3d(
                in_channels=2 * 32,
                out_channels=16,
                kernel_size=3,
                dilation=2,
                padding="same",
                bias=True,
            ),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                dilation=1,
                padding="same",
                bias=True,
            ),
        ]

        self.boost_1 = nn.Sequential(*boost_layers_1)
        self.boost_2 = nn.Sequential(*boost_layers_2)
        self.fuse = nn.Sequential(*fuse_layers)
        self.spatial_transformer = SpatialTransformer(shape=self.shape)

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        fixed_artifact_mask: torch.Tensor,
        vector_field: torch.Tensor,
        image_spacing: torch.Tensor,
    ) -> torch.Tensor:
        spatial_image_shape = moving_image.shape[-3:]
        vector_field_boost = torch.zeros(
            (1, 3) + spatial_image_shape, device=moving_image.device
        )

        # composed_vector_field = vector_field_boost + self.spatial_transformer(
        #     vector_field, vector_field_boost
        # )
        # moving_image = self.spatial_transformer(moving_image, composed_vector_field)
        #
        # diff = (moving_image - fixed_image) / (fixed_image + 1e-6)
        # diff = F.softsign(diff)

        input_images = torch.concat((vector_field, fixed_artifact_mask), dim=1)

        vector_field_boost_1 = self.boost_1(input_images)
        vector_field_boost_2 = self.boost_2(input_images)
        vector_field_boost = self.fuse(
            torch.concat((vector_field_boost_1, vector_field_boost_2), dim=1)
        )

        return vector_field_boost
