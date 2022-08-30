from __future__ import annotations

import math
from abc import ABC
from typing import Literal, Optional, Tuple, Type, Union

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vroc.common_types import FloatTuple, FloatTuple3D, IntTuple, IntTuple3D
from vroc.helper import binary_dilation
from vroc.kernels import gradient_kernel_3d


class EncoderBlock(nn.Module):
    def __init__(
        self,
        convolution_layer: Type[nn.Module],
        downsampling_layer: Type[nn.Module],
        norm_layer: Type[nn.Module],
        in_channels,
        out_channels,
        n_convolutions: int = 1,
        convolution_kwargs: Optional[dict] = None,
        downsampling_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        if not convolution_kwargs:
            convolution_kwargs = {}
        if not downsampling_kwargs:
            downsampling_kwargs = {}

        self.down = downsampling_layer(**downsampling_kwargs)

        layers = []
        for i_conv in range(n_convolutions):
            layers.append(
                convolution_layer(
                    in_channels=in_channels if i_conv == 0 else out_channels,
                    out_channels=out_channels,
                    **convolution_kwargs,
                )
            )
            if norm_layer:
                layers.append(norm_layer(out_channels))
            layers.append(nn.LeakyReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

    def forward(self, *inputs):
        x = self.down(*inputs)
        return self.convs(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        convolution_layer: Type[nn.Module],
        upsampling_layer: Type[nn.Module],
        norm_layer: Type[nn.Module],
        in_channels,
        out_channels,
        n_convolutions: int = 1,
        convolution_kwargs: Optional[dict] = None,
        upsampling_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        if not convolution_kwargs:
            convolution_kwargs = {}
        if not upsampling_kwargs:
            upsampling_kwargs = {}

        self.up = upsampling_layer(**upsampling_kwargs)

        layers = []
        for i_conv in range(n_convolutions):
            layers.append(
                convolution_layer(
                    in_channels=in_channels if i_conv == 0 else out_channels,
                    out_channels=out_channels,
                    **convolution_kwargs,
                )
            )
            if norm_layer:
                layers.append(norm_layer(out_channels))
            layers.append(nn.LeakyReLU(inplace=True))
        self.convs = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.convs(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        dimensions: int = 2,
        norm_type: Optional[str] = "BatchNorm",
    ):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        conv = getattr(nn, f"Conv{dimensions}d")
        if norm_type:
            norm = getattr(nn, f"{norm_type}{dimensions}d")
            layers = [
                conv(in_channels, mid_channels, kernel_size=3, padding=1),
                norm(mid_channels),
                nn.ReLU(inplace=True),
                conv(mid_channels, out_channels, kernel_size=3, padding=1),
                norm(out_channels),
                nn.ReLU(inplace=True),
            ]
        else:
            layers = [
                conv(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                conv(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimensions: int = 2,
        pooling: Union[int, Tuple[int, ...]] = 2,
        norm_type: Optional[str] = "BatchNorm",
    ):
        super().__init__()

        if dimensions == 1:
            pool = nn.MaxPool1d
        elif dimensions == 2:
            pool = nn.MaxPool2d
        elif dimensions == 3:
            pool = nn.MaxPool3d
        else:
            raise ValueError(f"Cannot handle {dimensions=}")

        self.maxpool_conv = nn.Sequential(
            pool(pooling),
            ConvBlock(
                in_channels, out_channels, dimensions=dimensions, norm_type=norm_type
            ),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diff_y = x2.size()[2] - x1.size()[2]
            diff_x = x2.size()[3] - x1.size()[3]

            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class BaseGaussianSmoothing(ABC, nn.Module):
    @staticmethod
    def get_kernel_radius(sigma: float, sigma_cutoff: float):
        # make the radius of the filter equal to truncate standard deviations
        # at least radius of 1
        return max(int(sigma_cutoff * sigma + 0.5), 1)

    @staticmethod
    def _make_gaussian_kernel_1d(
        sigma: float, sigma_cutoff: float = None, radius: int = None
    ):
        if (sigma_cutoff is not None and radius is not None) or not (
            sigma_cutoff or radius
        ):
            raise ValueError("Either pass sigma_cutoff or radius")

        if not radius:
            radius = BaseGaussianSmoothing.get_kernel_radius(sigma, sigma_cutoff)

        sigma2 = sigma * sigma
        x = torch.arange(-radius, radius + 1)
        phi_x = torch.exp(-0.5 / sigma2 * x**2)
        phi_x = phi_x / phi_x.sum()

        return torch.as_tensor(phi_x, dtype=torch.float32)

    @staticmethod
    def make_gaussian_kernel(
        sigma: FloatTuple,
        sigma_cutoff: FloatTuple,
        force_same_size: bool = False,
        radius: Optional[IntTuple] = None,
    ):
        if sigma_cutoff and len(sigma) != len(sigma_cutoff):
            raise ValueError("sigma and sigma_cutoff has to be same length")

        n_dim = len(sigma)

        if force_same_size:
            # this forces same size of the kernels for more efficient convolution
            if not radius:
                max_radius = max(
                    BaseGaussianSmoothing.get_kernel_radius(s, c)
                    for s, c in zip(sigma, sigma_cutoff)
                )
            else:
                max_radius = max(radius)

            radius = (max_radius,) * n_dim
            sigma_cutoff = None  # we don't need cutoff anymore, we use max radius

        kernels = []
        for i in range(n_dim):
            if radius:
                kernel_1d = BaseGaussianSmoothing._make_gaussian_kernel_1d(
                    sigma=sigma[i], radius=radius[i]
                )
            else:
                kernel_1d = BaseGaussianSmoothing._make_gaussian_kernel_1d(
                    sigma=sigma[i], sigma_cutoff=sigma_cutoff[i]
                )

            if n_dim == 2:
                kernel = torch.einsum("i,j->ij", kernel_1d, kernel_1d)
            elif n_dim == 3:
                kernel = torch.einsum("i,j,k->ijk", kernel_1d, kernel_1d, kernel_1d)
            else:
                raise RuntimeError(f"Dimension {n_dim} not supported")

            kernel = kernel / kernel.sum()
            kernels.append(kernel)

        return kernels


class GaussianSmoothing3d(BaseGaussianSmoothing):
    def __init__(
        self,
        sigma: FloatTuple3D = (1.0, 1.0, 1.0),
        sigma_cutoff: Optional[FloatTuple3D] = (1.0, 1.0, 1.0),
        radius: Optional[IntTuple3D] = None,
        force_same_size: bool = False,
        spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        use_image_spacing: bool = False,
    ):
        super().__init__()

        if sigma_cutoff and not (len(sigma) == len(sigma_cutoff) == 3):
            raise ValueError("Length of sigma and sigma_cutoff has to be 3")

        if sigma_cutoff and radius:
            raise RuntimeError("Please set either sigma_cutoff or radius")

        self.sigma = sigma
        self.sigma_cutoff = sigma_cutoff
        self.force_same_size = force_same_size
        self.use_image_spacing = use_image_spacing
        self.spacing = spacing

        if self.use_image_spacing:
            self.sigma = tuple(
                elem_1 * elem_2 for elem_1, elem_2 in zip(self.sigma, self.spacing)
            )
        kernel_x, kernel_y, kernel_z = GaussianSmoothing3d.make_gaussian_kernel(
            sigma=self.sigma,
            sigma_cutoff=self.sigma_cutoff,
            force_same_size=self.force_same_size,
            radius=radius,
        )
        self.kernel_size = (kernel_x.shape[-1], kernel_y.shape[-1], kernel_z.shape[-1])

        kernel_x, kernel_y, kernel_z = (
            kernel_x[None, None],
            kernel_y[None, None],
            kernel_z[None, None],
        )

        self.same_size = kernel_x.shape == kernel_y.shape == kernel_z.shape

        if self.same_size:
            self.kernel = torch.cat((kernel_x, kernel_y, kernel_z), dim=0)
            self.register_buffer("weight", self.kernel)

        else:
            self.register_buffer("weight_x", kernel_x)
            self.register_buffer("weight_y", kernel_y)
            self.register_buffer("weight_z", kernel_z)

    def forward(self, image):
        if self.same_size:
            image = F.conv3d(image, self.weight, groups=3, stride=1, padding="same")
        else:
            image_x = F.conv3d(image[:, 0:1], self.weight_x, stride=1, padding="same")
            image_y = F.conv3d(image[:, 1:2], self.weight_y, stride=1, padding="same")
            image_z = F.conv3d(image[:, 2:3], self.weight_z, stride=1, padding="same")
            image = torch.cat((image_x, image_y, image_z), dim=1)

        return image


class SpatialTransformer(nn.Module):
    """N-D Spatial Transformer."""

    def __init__(
        self,
        shape: IntTuple,
        mode: str = "bilinear",
    ):
        super().__init__()

        self.mode = mode

        # create sampling grid
        identity_grid = self.create_identity_grid(shape)
        self.register_buffer("identity_grid", identity_grid, persistent=False)

    @staticmethod
    def create_identity_grid(shape, device: torch.device = None):
        vectors = [
            torch.arange(0, s, dtype=torch.float32, device=device) for s in shape
        ]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)

        return grid

    def _warp(self, image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        if is_mask := ((image_dtype := image.dtype) in (torch.bool, torch.uint8)):
            image = torch.as_tensor(image, dtype=torch.float32)
            mode = "nearest"
        else:
            mode = self.mode

        warped = F.grid_sample(image, grid, align_corners=True, mode=mode)

        if is_mask:
            warped = torch.as_tensor(warped, dtype=image_dtype)

        return warped

    def forward(self, image, transformation: torch.Tensor) -> torch.Tensor:
        # transformation can be a affine 4x4 matrix or a dense vector field:
        # shape affine matrix: (batch, 4, 4)
        # shape dense vector field: (batch, 3, x_size, y_size, z_size)

        image_spatial_shape = image.shape[2:]
        image_n_dim = len(image_spatial_shape)

        if transformation.shape[-2:] == (4, 4):
            # transformation is affine matrix
            # discard last row of 4x4 matrix to get 3x4 matrix
            # (last low of affine matrix is not used by PyTorch)
            grid = F.affine_grid(
                transformation[:, :3], size=image.shape, align_corners=True
            )
        else:
            # transformation is dense vector field
            grid = self.identity_grid + transformation
            # need to normalize grid values to [-1, 1] for PyTorch's grid_sample
            for i in range(image_n_dim):
                grid[:, i, ...] = 2 * (
                    grid[:, i, ...] / (image_spatial_shape[i] - 1) - 0.5
                )

            # move channels dim to last position and reverse channels
            if image_n_dim == 2:
                grid = grid.permute(0, 2, 3, 1)
                grid = grid[..., [1, 0]]
            elif image_n_dim == 3:
                grid = grid.permute(0, 2, 3, 4, 1)
                grid = grid[..., [2, 1, 0]]

        return self._warp(image=image, grid=grid)


class BaseForces3d(nn.Module):
    def __init__(self, method: Literal["active", "passive", "dual"] = "dual"):
        super().__init__()
        self.method = method
        self._fixed_image_gradient: torch.Tensor | None = None

    def clear_cache(self):
        self._fixed_image_gradient = None

    @staticmethod
    def _calc_image_gradient_3d(image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError(f"Expected 5D tensor, got tensor with shape {image.shape}")
        if image.shape[1] != 1:
            raise NotImplementedError(
                "Currently, only single channel images are supported"
            )
        # this is list of x, y, z grad
        grad = torch.gradient(image, dim=(2, 3, 4))
        # stack x, y, z grad back to one single tensor of
        # shape (batch_size, 3, x_size, y_size, z_size)
        return torch.cat(grad, dim=1)

    # first if we need fixed image gradient; if yes calculate or get from cache
    def _get_fixed_image_gradient(self, fixed_image: torch.Tensor) -> torch.Tensor:
        recalculate = False
        if self._fixed_image_gradient is None:
            # no cached gradient, do calculation
            recalculate = True
        elif self._fixed_image_gradient.shape[-3:] != fixed_image.shape[-3:]:
            # spatial size mismatch, most likely due to multi level registration
            recalculate = True

        if recalculate:
            grad_fixed = self._calc_image_gradient_3d(fixed_image)
            self._fixed_image_gradient = grad_fixed

        return self._fixed_image_gradient

    def _compute_total_grad(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
    ):
        if self.method in ("passive", "dual"):
            grad_fixed = self._get_fixed_image_gradient(fixed_image)
        else:
            grad_fixed = None

        if self.method == "active":
            grad_moving = self._calc_image_gradient_3d(moving_image)
            # gradient is defined in moving image domain -> use moving mask if given
            if moving_mask is not None:
                grad_moving = grad_moving * moving_mask
            total_grad = grad_moving

        elif self.method == "passive":
            # gradient is defined in fixed image domain -> use fixed mask if given
            if fixed_mask is not None:
                grad_fixed = grad_fixed * fixed_mask
            total_grad = grad_fixed

        elif self.method == "dual":
            # we need both gradient of moving and fixed image
            grad_moving = self._calc_image_gradient_3d(moving_image)

            # mask both gradients as above; also check that we have both masks
            masks_available = (m is not None for m in (moving_mask, fixed_mask))
            if not all(masks_available) and any(masks_available):
                # we only have one mask
                raise RuntimeError("Dual forces need both moving and fixed mask")

            if moving_mask is not None:
                grad_moving = grad_moving * moving_mask
            if fixed_mask is not None:
                grad_fixed = grad_fixed * fixed_mask
            total_grad = grad_moving + grad_fixed

        else:
            raise NotImplementedError(f"Demon forces {self.method} are not implemented")

        return total_grad


class DemonForces3d(BaseForces3d):
    def _calculate_demon_forces_3d(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        method: Literal["active", "passive", "dual"] = "dual",
        fixed_image_gradient: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
    ):
        # if method == 'dual' use union of moving and fixed mask
        if self.method == "dual":
            union_mask = torch.logical_or(moving_mask, fixed_mask)
            moving_mask, fixed_mask = union_mask, union_mask

        # grad tensors have the shape of (batch_size, 3, x_size, y_size, z_size)
        # if moving_mask is not None:
        #     moving_mask = binary_dilation(
        #         torch.as_tensor(moving_mask, dtype=torch.float32),
        #         kernel_size=(3, 3, 3)
        #     )
        # if fixed_mask is not None:
        #     fixed_mask = binary_dilation(
        #         torch.as_tensor(fixed_mask, dtype=torch.float32),
        #         kernel_size=(3, 3, 3)
        #     )

        total_grad = self._compute_total_grad(
            moving_image, fixed_image, moving_mask, fixed_mask
        )
        # gamma = 1 / (
        #     (sum(i**2 for i in original_image_spacing)) / len(original_image_spacing)
        # )
        gamma = 1
        # calculcate squared L2 of grad, i.e., || grad ||^2
        l2_grad = total_grad.pow(2).sum(dim=1, keepdim=True)
        epsilon = 1e-6  # to prevent division by zero
        norm = (fixed_image - moving_image) / (
            epsilon + l2_grad + gamma * (fixed_image - moving_image) ** 2
        )

        return norm * total_grad

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        original_image_spacing: FloatTuple3D,
    ):
        return self._calculate_demon_forces_3d(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            method=self.method,
            original_image_spacing=original_image_spacing,
        )


class NCCForces3d(BaseForces3d):
    def _calculate_ncc_forces_3d(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        method: Literal["active", "passive", "dual"] = "dual",
        fixed_image_gradient: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        epsilon: float = 1e-10,
        radius: Tuple[int, ...] = (5, 5, 5),
    ):
        # normalizer = sum(i * i for i in original_image_spacing) / len(original_image_spacing)

        filter = torch.ones((1, 1) + radius).to(moving_image)
        npixel_filter = torch.prod(torch.tensor(radius))
        stride = (1, 1, 1)

        # TODO: compute fixed params only once per level in base class init
        mm = moving_image * moving_image
        ff = fixed_image * fixed_image
        mf = moving_image * fixed_image

        sum_m = F.conv3d(moving_image, filter, stride=stride, padding="same")
        sum_f = F.conv3d(fixed_image, filter, stride=stride, padding="same")
        sum_mm = F.conv3d(mm, filter, stride=stride, padding="same")
        sum_ff = F.conv3d(ff, filter, stride=stride, padding="same")
        sum_mf = F.conv3d(mf, filter, stride=stride, padding="same")

        moving_mean = sum_m / npixel_filter
        fixed_mean = sum_f / npixel_filter

        var_m = (
            sum_mm - 2 * moving_mean * sum_m + npixel_filter * moving_mean * moving_mean
        )
        var_f = (
            sum_ff - 2 * fixed_mean * sum_f + npixel_filter * fixed_mean * fixed_mean
        )
        var_mf = var_m * var_f

        cross = (
            sum_mf
            - fixed_mean * sum_m
            - moving_mean * sum_f
            + npixel_filter * moving_mean * fixed_mean
        )

        cross_correlation = torch.ones_like(cross)
        cross_correlation[var_mf > epsilon] = (
            cross[var_mf > epsilon] * cross[var_mf > epsilon] / var_mf[var_mf > epsilon]
        )

        total_grad = self._compute_total_grad(
            moving_image, fixed_image, moving_mask, fixed_mask
        )
        if self.method == "dual":
            total_grad = total_grad * 0.5

        moving_mean_centered = moving_image - moving_mean
        fixed_mean_centered = fixed_image - fixed_mean

        mask = (
            (var_mf > epsilon)
            & (var_f > epsilon)
            & (fixed_mean_centered != 0.0)
            & (moving_mean_centered != 0.0)
        )

        factor = torch.zeros_like(cross)
        factor[mask] = (2.0 * cross[mask] / var_mf[mask]) * (
            moving_mean_centered[mask]
            - cross[mask] / (var_f[mask] * fixed_mean_centered[mask])
        )
        metric = 1 - torch.mean(cross_correlation[fixed_mask])
        return (-1) * factor * total_grad

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        original_image_spacing: FloatTuple3D,
    ):
        return self._calculate_ncc_forces_3d(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            method=self.method,
            original_image_spacing=original_image_spacing,
        )


class NGFForces3d(BaseForces3d):
    def _grad_param(self, method, axis):
        kernel = gradient_kernel_3d(method, axis)

        kernel = kernel.reshape(1, 1, *kernel.shape)
        return Parameter(torch.Tensor(kernel).float())

    def _calculate_ngf_forces_3d(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        epsilon: float = None,
    ):
        # # reshape
        # b, c = moving_image.shape[:2]
        # spatial_shape = moving_image.shape[2:]
        #
        # moving_image = moving_image.view(b * c, 1, *spatial_shape)
        # fixed_image = fixed_image.view(b * c, 1, *spatial_shape)

        grad_u_kernel = self._grad_param("finite_diff", axis=0).to("cuda")
        grad_v_kernel = self._grad_param("finite_diff", axis=1).to("cuda")
        grad_w_kernel = self._grad_param("finite_diff", axis=2).to("cuda")
        # gradient
        moving_grad_u = (
            F.conv3d(moving_image, grad_u_kernel, padding="same")
            * original_image_spacing[0]
        )
        moving_grad_v = (
            F.conv3d(moving_image, grad_v_kernel, padding="same")
            * original_image_spacing[1]
        )
        moving_grad_w = (
            F.conv3d(moving_image, grad_w_kernel, padding="same")
            * original_image_spacing[2]
        )

        fixed_grad_u = (
            F.conv3d(fixed_image, grad_u_kernel, padding="same")
            * original_image_spacing[0]
        )
        fixed_grad_v = (
            F.conv3d(fixed_image, grad_v_kernel, padding="same")
            * original_image_spacing[1]
        )
        fixed_grad_w = (
            F.conv3d(fixed_image, grad_w_kernel, padding="same")
            * original_image_spacing[2]
        )

        if epsilon is None:
            with torch.no_grad():
                epsilon = torch.mean(
                    torch.abs(moving_grad_u)
                    + torch.abs(moving_grad_v)
                    + torch.abs(moving_grad_w)
                )

        # gradient norm
        moving_grad_norm = (
            moving_grad_u**2 + moving_grad_v**2 + moving_grad_w**2 + epsilon**2
        )
        fixed_grad_norm = (
            fixed_grad_u**2 + fixed_grad_v**2 + fixed_grad_w**2 + epsilon**2
        )

        # nominator
        moving_grad = torch.concat([moving_grad_u, moving_grad_v, moving_grad_w], dim=1)
        fixed_grad = torch.concat([fixed_grad_u, fixed_grad_v, fixed_grad_w], dim=1)
        grad_product = moving_grad * fixed_grad

        # denominator
        norm_product = moving_grad_norm * fixed_grad_norm

        # integrator
        ngf = (grad_product**2 / norm_product) * fixed_mask

        metric = torch.mean(ngf)

        return (-1) * ngf

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        original_image_spacing: FloatTuple3D,
    ):
        return self._calculate_ngf_forces_3d(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            original_image_spacing=original_image_spacing,
        )


if __name__ == "__main__":
    import time

    d1 = GaussianSmoothing3d(
        sigma=(1.0, 2.0, 3.0),
        # radius=(1, 2, 3),
        sigma_cutoff=(2.0, 2.0, 2.0),
        force_same_size=True,
    )

    d2 = GaussianSmoothing3d(
        sigma=(1.0, 2.0, 3.0), radius=(1, 2, 3), sigma_cutoff=None, force_same_size=True
    )
    t = time.time()
    d3 = GaussianSmoothing3d(
        sigma=(1.0, 2.0, 3.0), radius=(1, 2, 3), sigma_cutoff=None, force_same_size=True
    )
    print(time.time() - t)

    print(d1.kernel_size)
    print(d2.kernel_size)
    print(d3.kernel_size)
