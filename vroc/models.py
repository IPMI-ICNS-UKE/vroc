from __future__ import annotations

import time
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from vroc.decorators import timing
from vroc.helper import get_bounding_box
from vroc.logger import LoggerMixin


class FlexUNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        n_levels: int = 6,
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
            "bias": True,
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


class LungCTSegmentationUnet3d(FlexUNet):
    def __init__(
        self,
        n_channels: int = 1,
        n_classes: int = 1,
        n_levels: int = 6,
        filter_base: int = 32,
    ):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            n_levels=n_levels,
            filter_base=filter_base,
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


class VarReg3d(nn.Module, LoggerMixin):
    _INTERPOLATION_MODES = {
        3: "linear",
        4: "bilinear",
        5: "trilinear",
    }

    def __init__(
        self,
        scale_factors: FloatTuple | float = (1.0,),
        iterations: IntTuple | int = 100,
        tau: FloatTuple | float = 1.0,
        demon_forces: Literal["active", "passive", "dual"] = "dual",
        regularization_sigma: FloatTuple3D
        | Tuple[FloatTuple3D, ...] = (
            1.0,
            1.0,
            1.0,
        ),
        regularization_radius: IntTuple3D | Tuple[IntTuple3D, ...] | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        use_image_spacing: bool = False,
        restrict_to_mask_bbox: bool = False,
    ):
        super().__init__()

        if not is_tuple(scale_factors, min_length=1):
            scale_factors = (scale_factors,)
        self.scale_factors = scale_factors

        self.scale_factors = scale_factors  # this also defines "n_levels"
        self.iterations = VarReg3d._expand_to_level_tuple(
            iterations, n_levels=self.n_levels
        )

        self.tau = VarReg3d._expand_to_level_tuple(tau, n_levels=self.n_levels)
        self.regularization_sigma = VarReg3d._expand_to_level_tuple(
            regularization_sigma, n_levels=self.n_levels, is_tuple=True
        )
        self.regularization_sigma = VarReg3d._expand_to_level_tuple(
            regularization_sigma, n_levels=self.n_levels, is_tuple=True
        )
        self.regularization_radius = VarReg3d._expand_to_level_tuple(
            regularization_radius, n_levels=self.n_levels, is_tuple=True
        )
        self.original_image_spacing = original_image_spacing
        self.use_image_spacing = use_image_spacing

        self.demon_forces = demon_forces

        self.restrict_to_mask_bbox = restrict_to_mask_bbox

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

        self._demon_forces_layer = DemonForces3d(method=self.demon_forces)

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

    def _perform_scaling(
        self, *images, scale_factor: float = 1.0
    ) -> List[torch.Tensor]:

        scaled = []
        for image in images:
            if image is not None:
                if image.dtype == torch.bool:
                    # image is mask
                    # interpolation is not implemented for dtype bool
                    # also use NN interpolation
                    image = torch.as_tensor(image, dtype=torch.uint8)
                    image = F.interpolate(
                        image, scale_factor=scale_factor, mode="nearest"
                    )
                    image = torch.as_tensor(image, dtype=torch.bool)
                else:
                    # normal image (moving or fixed)
                    mode = VarReg3d._INTERPOLATION_MODES[image.ndim]
                    image = F.interpolate(
                        image, scale_factor=scale_factor, mode=mode, align_corners=True
                    )

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

        mode = VarReg3d._INTERPOLATION_MODES[image.ndim]
        vector_field = F.interpolate(
            vector_field, size=image_shape[2:], mode=mode, align_corners=True
        )

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

    def run_registration(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        initial_vector_field: torch.Tensor | None = None,
    ):
        if moving_image.ndim != fixed_image.ndim:
            raise RuntimeError("Dimension mismatch betwen moving and fixed image")
        # define dimensionalities
        n_image_dimensions = moving_image.ndim
        n_vector_field_components = n_image_dimensions - 2  # -1 batch dim, -1 color dim

        if self.restrict_to_mask_bbox and (
            moving_mask is not None or fixed_mask is not None
        ):
            masks = [m for m in (moving_mask, fixed_mask) if m is not None]
            if len(masks) == 2:
                # we compute the union of both masks to get the overall bounding box
                union_mask = torch.logical_or(*masks)
            else:
                union_mask = masks[0]

            original_moving_image = moving_image
            bbox = get_bounding_box(union_mask, padding=5)
            self.logger.debug(f"Restricting registration to bounding box {bbox}")

            moving_image = moving_image[bbox]
            fixed_image = fixed_image[bbox]
            if moving_mask is not None:
                moving_mask = moving_mask[bbox]
            if fixed_mask is not None:
                fixed_mask = fixed_mask[bbox]
            if initial_vector_field is not None:
                original_initial_vector_field = initial_vector_field
                initial_vector_field = initial_vector_field[(..., *bbox[-3:])]

        if moving_mask is not None:
            moving_mask = torch.as_tensor(moving_mask, dtype=torch.bool)
        if fixed_mask is not None:
            fixed_mask = torch.as_tensor(fixed_mask, dtype=torch.bool)

        # create new spatial transformers if needed (skip batch and color dimension)
        self._create_spatial_transformers(
            fixed_image.shape[2:], device=fixed_image.device
        )

        full_size_moving = moving_image

        metrics = []
        # set initial vector field (if given)
        vector_field = None  # if initial_vector_field is None else initial_vector_field

        runtimes = []

        metric_before = self._calculate_metric(
            moving_image=moving_image, fixed_image=fixed_image, fixed_mask=fixed_mask
        )

        for i_level, (scale_factor, iterations) in enumerate(
            zip(self.scale_factors, self.iterations)
        ):
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
                    + (n_vector_field_components,)
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

            t_start = time.time()
            for i in range(iterations):
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
                    scaled_moving_image, composed_vector_field
                )

                level_metrics.append(
                    self._calculate_metric(
                        moving_image=warped_moving,
                        fixed_image=scaled_fixed_image,
                        fixed_mask=scaled_fixed_mask,
                    )
                )
                log = {"level": i_level, "iteration": i, "metric": level_metrics[-1]}

                forces = self._demon_forces_layer(
                    warped_moving,
                    scaled_fixed_image,
                    scaled_moving_mask,
                    scaled_fixed_mask,
                    original_image_spacing,
                )
                vector_field += self.tau[i_level] * forces

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

        vector_field = self._match_vector_field(vector_field, full_size_moving)

        if self.restrict_to_mask_bbox:
            # undo restriction to mask
            _vector_field = torch.zeros(
                vector_field.shape[:-3] + original_moving_image.shape[-3:],
                device=vector_field.device,
            )
            _vector_field[(...,) + bbox[2:]] = vector_field
            vector_field = _vector_field

        spatial_transformer = SpatialTransformer(
            shape=original_moving_image.shape[2:]
        ).to(original_moving_image.device)

        if initial_vector_field is not None:
            # if we have an initial vector field: compose both vector fields.
            # Here: at full resolution without cropping/bbox
            composed_vector_field = vector_field + spatial_transformer(
                original_initial_vector_field, vector_field
            )
        else:
            composed_vector_field = vector_field

        warped_moving_image = spatial_transformer(
            original_moving_image, composed_vector_field
        )

        metric_after = self._calculate_metric(
            moving_image=warped_moving_image[bbox],
            fixed_image=fixed_image,
            fixed_mask=fixed_mask,
        )

        misc = {
            "metric_before": metric_before,
            "metric_after": metric_after,
            "level_metrics": metrics,
            "runtimes": runtimes,
        }

        return warped_moving_image, composed_vector_field, misc

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


class VectorFieldBoosting(nn.Module):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    model = VarReg3d(
        scale_factors=1.0,
        iterations=200,
        demon_forces="symmetric",
        tau=2.0,
        regularization_sigma=(4.0, 2.0, 2.0),
    )
