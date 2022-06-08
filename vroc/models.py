import time
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from vroc.blocks import (
    SpatialTransformer,
    DemonForces3d,
    GaussianSmoothing3d,
    DownBlock,
    UpBlock,
    ConvBlock,
)
from vroc.helper import rescale_range
from torch.optim import Adam


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

        self.conv_1 = DownBlock(in_channels=self.n_channels, out_channels=8, dimensions=1, norm_type='InstanceNorm')
        self.conv_2 = DownBlock(in_channels=8, out_channels=16, dimensions=1, norm_type='InstanceNorm')
        self.conv_3 = DownBlock(in_channels=16, out_channels=8, dimensions=1, norm_type='InstanceNorm')
        self.conv_4 = DownBlock(in_channels=8, out_channels=4, dimensions=1, norm_type='InstanceNorm')
        self.conv_5 = DownBlock(in_channels=4, out_channels=self.n_params, dimensions=1, norm_type='InstanceNorm')

    def forward(self, features):
        out = self.conv_1(features)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = torch.sigmoid(out)
        out = torch.squeeze(out, dim=-1)

        out_dict = {}
        # scale params according to min/max range
        for i_param, param_name in enumerate(self.params.keys()):
            out_min, out_max = self.params[param_name]['min'], self.params[param_name]['max']
            out[:, i_param] = (out[:, i_param] * (out_max - out_min)) + out_min
            out_dict[param_name] = out[:, i_param]

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
            early_stopping_fn: Tuple[str, ...] = ("no_impr", "lstsq"),
            early_stopping_delta: Tuple[float, ...] = (0.0, 0.0),
            early_stopping_window: Tuple[int, ...] = (10, 10),
            radius: Tuple[int, ...] = None,
            param_net_buffer_size: int = 16
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

        self.early_stopping_fn = early_stopping_fn
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_window = early_stopping_window
        self._metrics = []
        self._counter = 0

        # stuff for param net
        self._feature_buffer = []
        self._metrics_buffer = []
        self._param_net_buffer_size = param_net_buffer_size
        self._param_net = ParamNet(
            params={
                'iterations': {'min': 0, 'max': 1000, 'dtype': int},
                'tau': {'min': 0.0, 'max': 10.0, 'dtype': float}
            },
            n_channels=3
        )
        self._param_net_optimizer = Adam(self._param_net.parameters())
        self._param_net_optimizer.zero_grad()

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

    def _create_spatial_transformers(
            self,
            image_shape: Tuple[int, ...],
            device
    ):
        if not image_shape == self._image_shape:
            self._image_shape = image_shape
            self._full_size_spatial_transformer = SpatialTransformer(shape=image_shape).to(device)

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

    def _check_early_stopping(self, metrics: List[float], i_level):
        if self._check_early_stopping_condition(metrics, i_level):
            rel_change = (metrics[-self.early_stopping_window[i_level]] - metrics[-1]) / (
                    metrics[-self.early_stopping_window[i_level]] + 1e-9
            )

            if rel_change < self.early_stopping_delta[i_level]:
                return True
        return False

    def _check_early_stopping_average_improvement(self, metrics: List[float], i_level):
        if self._check_early_stopping_condition(metrics, i_level):

            window = np.array(metrics[-self.early_stopping_window[i_level]:])
            window_rel_changes = 1 - window[1:] / window[:-1]

            if window_rel_changes.mean() < self.early_stopping_delta[i_level]:
                return True
        return False

    def _check_early_stopping_increase_count(self, metrics: List[float], i_level):
        window = np.array(metrics)
        if np.argmin(window) == (len(window) - 1):
            self._counter = 0
        else:
            self._counter += 1
        if self._counter == self.early_stopping_delta[i_level]:
            return True
        return False

    def _check_early_stopping_lstsq(self, metrics: List[float], i_level):
        if self._check_early_stopping_condition(metrics, i_level):

            window = np.array(metrics[-self.early_stopping_window[i_level]:])
            scaled_window = rescale_range(
                window, (np.min(metrics), np.max(metrics)), (0, 1)
            )
            lstsq_result = stats.linregress(
                np.arange(self.early_stopping_window[i_level]), scaled_window
            )
            if lstsq_result.slope > -self.early_stopping_delta[i_level]:
                return True
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
        mask = F.interpolate(mask, scale_factor=scale_factor, mode="nearest")
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
            raise ValueError(f'Dimension {image_shape} not supported')

        if vector_field_shape[2:] == image_shape[2:]:
            return vector_field

        vector_field = F.interpolate(
            vector_field, size=image_shape[2:], mode=mode, align_corners=False
        )

        scale_factor = torch.tensor(
            [s1 / s2 for (s1, s2) in zip(image_shape[2:], vector_field_shape[2:])]
        )
        # 5D shape: (1, 3, 1, 1, 1), 4D shape: (1, 2, 1, 1)
        scale_factor = torch.reshape(scale_factor, (1, -1) + (1,) * (len(image_shape) - 2))

        return vector_field * scale_factor.to(vector_field)

    def generate_param_net_features(self, fixed_image, mask, moving_image, bins=32):
        fixed_image = fixed_image.detach()
        mask = mask.detach()
        moving_image = moving_image.detach()

        valid_mask = mask == 1
        l2_image = F.mse_loss(fixed_image, moving_image, reduction="none")
        lower_quantile = torch.quantile(
            l2_image[valid_mask], 0.05, interpolation="linear"
        ).detach()
        upper_quantile = torch.quantile(
            l2_image[valid_mask], 0.95, interpolation="linear"
        ).detach()

        fixed_image_hist = torch.histc(fixed_image[valid_mask], bins=bins)
        moving_image_hist = torch.histc(moving_image[valid_mask], bins=bins)
        diff_hist = torch.histc(
            l2_image[valid_mask], bins=bins, min=lower_quantile, max=upper_quantile
        )

        histograms = torch.vstack((fixed_image_hist, moving_image_hist, diff_hist))

        # transform to density histograms
        histograms = histograms / histograms.sum(dim=1)[:, None]

        return histograms

    def train_param_net(self):
        if len(self._feature_buffer) >= self._param_net_buffer_size:
            features = torch.vstack([f[None] for f in self._feature_buffer])

            losses = []
            for metrics in self._metrics_buffer:
                loss = (metrics[-1] - metrics[0]) / metrics[-1]
                losses.append(loss)

            losses = torch.hstack(losses)

    def _predict_params(self, fixed_image, mask, moving_image):
        features = self.generate_param_net_features(
            fixed_image=fixed_image,
            mask=mask,
            moving_image=moving_image
        )
        predicted_params = self._param_net(features[None])

        return predicted_params


    def warp_moving(self, image, mask, moving, original_image_spacing):
        # register moving image onto fixed image
        if len(image.shape) == 4:
            dim_vf = 2
        elif len(image.shape) == 5:
            dim_vf = 3
        full_size_moving = moving
        features = []
        metrics_all_level = []
        vector_field = None
        # with torch.no_grad():

        predicted_params = self._predict_params(
            fixed_image=image,
            mask=mask,
            moving_image=moving
        )

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

            warped_moving = spatial_transformer(scaled_moving, vector_field)
            feature_vector = self.generate_feature_vector(
                scaled_image, scaled_mask, warped_moving, bins=20
            )

            # TODO: Test if e.g. minip is better as feature

            for i in range(iterations):
                # if (i % 100) == 0:
                #     print(f"*** LEVEL {i_level + 1} *** ITERATION {i} ***")

                warped_moving = spatial_transformer(scaled_moving, vector_field)

                metrics.append(
                    F.mse_loss(
                        scaled_image[scaled_mask == 1],
                        warped_moving[scaled_mask == 1],
                    )
                )

                if self.early_stopping_fn[i_level] == "lstsq":
                    if self._check_early_stopping_lstsq(metrics, i_level):
                        break
                elif self.early_stopping_fn[i_level] == "no_impr":
                    if self._check_early_stopping_increase_count(metrics, i_level):
                        break
                elif self.early_stopping_fn[i_level] == "no_average_impr":
                    if self._check_early_stopping_average_improvement(
                            metrics, i_level
                    ):
                        break
                elif self.early_stopping_fn[i_level] == "none":
                    pass
                else:
                    raise Exception(
                        f"Early stopping method {self.early_stopping_fn[i_level]} is not implemented"
                    )

                forces = self._demon_forces_layer(
                    warped_moving,
                    scaled_image,
                    self.demon_forces,
                    original_image_spacing,
                )
                # mask forces with artifact mask (artifact = 0, valid = 1)

                vector_field += self.tau[i_level] * (forces * scaled_mask)
                _regularization_layer = self.get_submodule(
                    f"regularization_layer_level_{i_level}",
                )
                vector_field = _regularization_layer(vector_field)

            # print(f'LEVEL {i_level + 1} stopped at ITERATION {i + 1}: Metric: {metrics[-1]}')
            metrics_all_level.append(metrics)
            features.append(feature_vector)

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
            metrics_all_level,
            features,
        )

    def forward(self, image, mask, moving, original_image_spacing):
        # create new spatial transformers if needed (skip batch and color dimension)
        self._create_spatial_transformers(image.shape[2:], device=image.device)

        (
            corrected_warped_moving,
            warped_moving,
            corrected_vector_field,
            vector_field,
            metrics_all_level,
            features_all_level,
        ) = self.warp_moving(image, mask, moving, original_image_spacing)

        self._feature_buffer.extend(features_all_level)
        self._metrics_buffer.extend(metrics_all_level)

        self.train_param_net()

        return (
            corrected_warped_moving,
            warped_moving,
            corrected_vector_field,
            vector_field,
            metrics_all_level,
            features_all_level,
        )


if __name__ == '__main__':
    net = ParamNet(
        params={
            'iterations': {'min': 0, 'max': 1000, 'dtype': int},
            'tau': {'min': 0.0, 'max': 10.0, 'dtype': float}
        },
        n_channels=3)

    features = torch.ones((16, 3, 32))
    out = net(features)
