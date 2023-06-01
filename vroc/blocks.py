from __future__ import annotations

from abc import ABC
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vroc.common_types import (
    FloatTuple,
    FloatTuple2D,
    FloatTuple3D,
    IntTuple,
    IntTuple2D,
    IntTuple3D,
    Number,
)


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
        sigma2 = torch.as_tensor(sigma2)
        x = torch.arange(-radius, radius + 1, device=sigma2.device)
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


class GaussianSmoothing2d(BaseGaussianSmoothing):
    def __init__(
        self,
        sigma: FloatTuple3D = (1.0, 1.0),
        sigma_cutoff: FloatTuple2D | None = (1.0, 1.0, 1.0),
        radius: IntTuple2D | None = None,
        force_same_size: bool = False,
        spacing: FloatTuple2D = (1.0, 1.0),
        use_image_spacing: bool = False,
    ):
        super().__init__()

        if sigma_cutoff and not (len(sigma) == len(sigma_cutoff) == 2):
            raise ValueError("Length of sigma and sigma_cutoff has to be 2")

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
        kernel_x, kernel_y = BaseGaussianSmoothing.make_gaussian_kernel(
            sigma=self.sigma,
            sigma_cutoff=self.sigma_cutoff,
            force_same_size=self.force_same_size,
            radius=radius,
        )
        self.kernel_size = (kernel_x.shape[-1], kernel_y.shape[-1])

        kernel_x, kernel_y = (
            kernel_x[None, None],
            kernel_y[None, None],
        )

        self.same_size = kernel_x.shape == kernel_y.shape

        if self.same_size:
            self.kernel = torch.cat((kernel_x, kernel_y), dim=0)
            self.register_buffer("weight", self.kernel)

        else:
            self.register_buffer("weight_x", kernel_x)
            self.register_buffer("weight_y", kernel_y)

    def forward(self, image):
        if self.same_size:
            image = F.conv2d(image, self.weight, groups=2, stride=1, padding="same")
        else:
            image_x = F.conv3d(image[:, 0:1], self.weight_x, stride=1, padding="same")
            image_y = F.conv3d(image[:, 1:2], self.weight_y, stride=1, padding="same")
            image = torch.cat((image_x, image_y), dim=1)

        return image


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
        kernel_x, kernel_y, kernel_z = BaseGaussianSmoothing.make_gaussian_kernel(
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
        shape: IntTuple | None = None,
        default_value: Number = 0,
    ):
        super().__init__()

        self.shape = shape
        self.default_value = default_value

        # create sampling grid if shape is given
        if self.shape:
            identity_grid = self.create_identity_grid(shape)
            self.register_buffer("identity_grid", identity_grid, persistent=False)
        else:
            self.identity_grid = None

    @staticmethod
    def create_identity_grid(shape, device: torch.device = None):
        vectors = [
            torch.arange(0, s, dtype=torch.float32, device=device) for s in shape
        ]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)

        return grid

    def _warp(
        self,
        image: torch.Tensor,
        grid: torch.Tensor,
        default_value: Number | None = None,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        # initial default value can be overwritten by warp method
        default_value = default_value or self.default_value
        # save initial dtype
        image_dtype = image.dtype
        if not torch.is_floating_point(image):
            # everything like bool, uint8, ...
            mode = "nearest"
            # warning: No gradients with nearest interpolation

        # convert to float32 as needed for grid_sample
        image = torch.as_tensor(image, dtype=torch.float32)

        if default_value:  # we can skip this if default_value == 0
            # is default_value is given, we set the minimum value to 1, since
            # PyTorch uses 0 for out-of-bounds voxels
            shift = image.min() - 1
            image = image - shift  # valid image values are now >= 1

        warped = F.grid_sample(
            image, grid, align_corners=True, mode=mode, padding_mode="zeros"
        )

        if default_value:
            out_of_bounds = warped == 0
            warped[out_of_bounds] = default_value
            # undo value shift
            warped[~out_of_bounds] = warped[~out_of_bounds] + shift

        # convert back to initial dtype
        warped = torch.as_tensor(warped, dtype=image_dtype)

        return warped

    def forward(
        self,
        image,
        transformation: torch.Tensor,
        default_value: Number | None = None,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        # transformation can be an affine matrix or a dense vector field:
        # n_spatial_dims = 2 or 3
        # shape affine matrix: (batch, n_spatial_dims + 1, n_spatial_dims + 1)
        # shape dense vector field: (batch, n_spatial_dims, x_size, y_size[, z_size)]

        spatial_image_shape = image.shape[2:]
        n_spatial_dims = len(spatial_image_shape)

        if n_spatial_dims not in {2, 3}:
            raise NotImplementedError(
                f"SpatialTransformer for {n_spatial_dims}D images is currently "
                f"not supported"
            )

        # check for dense transformation (i.e. matching spatial dimensions)
        is_dense_transformation = (
            spatial_image_shape == transformation.shape[-n_spatial_dims:]
        )

        if self.identity_grid is None:
            # create identity grid dynamically
            identity_grid = self.create_identity_grid(
                spatial_image_shape, device=image.device
            )
            self.identity_grid = identity_grid
        elif self.identity_grid.shape[1:] != spatial_image_shape:
            # mismatch: create new identity grid
            identity_grid = self.create_identity_grid(
                spatial_image_shape, device=image.device
            )
            self.identity_grid = identity_grid
        else:
            identity_grid = self.identity_grid

        if not is_dense_transformation:
            # transformation is affine matrix
            # discard last row of 4x4/3x3 matrix
            # (last low of affine matrix is not used by PyTorch)
            grid = F.affine_grid(
                transformation[:, :n_spatial_dims], size=image.shape, align_corners=True
            )
        else:
            # transformation is dense vector field
            grid = identity_grid + transformation

            grid = SpatialTransformer.scale_grid(
                grid, spatial_image_shape=spatial_image_shape
            )

            # move channels dim to last position and reverse channels
            if n_spatial_dims == 2:
                grid = grid.permute(0, 2, 3, 1)
                grid = grid[..., [1, 0]]
            elif n_spatial_dims == 3:
                grid = grid.permute(0, 2, 3, 4, 1)
                grid = grid[..., [2, 1, 0]]

        return self._warp(
            image=image, grid=grid, default_value=default_value, mode=mode
        )

    @staticmethod
    def scale_grid(
        grid: torch.Tensor, spatial_image_shape: IntTuple | torch.Tensor | torch.Size
    ) -> torch.Tensor:
        spatial_image_shape = torch.as_tensor(spatial_image_shape, device=grid.device)
        if (n_spatial_dims := len(spatial_image_shape)) == 2:
            spatial_image_shape = spatial_image_shape[None, :, None, None]
        elif n_spatial_dims == 3:
            spatial_image_shape = spatial_image_shape[None, :, None, None, None]
        else:
            raise NotImplementedError(
                f"SpatialTransformer for {n_spatial_dims}D images is currently "
                f"not supported"
            )
        # scale grid values to [-1, 1] for PyTorch's grid_sample
        grid = 2 * (grid / (spatial_image_shape - 1) - 0.5)

        return grid

    def compose_vector_fields(
        self, vector_field_1: torch.Tensor, vector_field_2: torch.Tensor
    ) -> torch.Tensor:
        if vector_field_1.shape != vector_field_2.shape:
            raise RuntimeError(
                f"Shape mismatch between vector fields: "
                f"{vector_field_1.shape} vs. {vector_field_2.shape}"
            )

        return vector_field_2 + self(vector_field_1, vector_field_2)


class BaseForces(nn.Module):
    def __init__(self, method: Literal["active", "passive", "dual"] = "dual"):
        super().__init__()
        self.method = method
        self._fixed_image_gradient: torch.Tensor | None = None

    def clear_cache(self):
        self._fixed_image_gradient = None

    @staticmethod
    def _calc_image_gradient(image: torch.Tensor) -> torch.Tensor:
        if image.ndim not in {4, 5}:
            raise ValueError(
                f"Expected 4D or 5D tensor, got tensor with shape {image.shape}"
            )
        if image.shape[1] != 1:
            raise NotImplementedError(
                "Currently, only single channel images are supported"
            )

        dims = tuple(range(2, image.ndim))
        # grad is a list of x, y(, z) gradients
        grad = torch.gradient(image, dim=dims)
        # stack x, y(, z) gradients back to one single tensor of
        # shape (batch_size, 2 or 3, x_size, y_size[, z_size])
        return torch.cat(grad, dim=1)

    def _get_fixed_image_gradient(self, fixed_image: torch.Tensor) -> torch.Tensor:
        recalculate = False
        if self._fixed_image_gradient is None:
            # no cached gradient, do calculation
            recalculate = True
        elif self._fixed_image_gradient.shape[2:] != fixed_image.shape[2:]:
            # spatial size mismatch, most likely due to multi level registration
            recalculate = True

        if recalculate:
            grad_fixed = self._calc_image_gradient(fixed_image)
            self._fixed_image_gradient = grad_fixed

        return self._fixed_image_gradient

    def _compute_total_grad(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        use_masks: bool = True,
    ):
        if self.method in ("passive", "dual"):
            grad_fixed = self._get_fixed_image_gradient(fixed_image)
        else:
            grad_fixed = None

        if self.method == "active":
            grad_moving = self._calc_image_gradient(moving_image)
            # gradient is defined in moving image domain -> use moving mask if given
            if use_masks and moving_mask is not None:
                grad_moving = grad_moving * moving_mask
            total_grad = grad_moving

        elif self.method == "passive":
            # gradient is defined in fixed image domain -> use fixed mask if given
            if use_masks and fixed_mask is not None:
                grad_fixed = grad_fixed * fixed_mask
            total_grad = grad_fixed

        elif self.method == "dual":
            # we need both gradient of moving and fixed image
            grad_moving = self._calc_image_gradient(moving_image)

            if use_masks:
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


class DemonForces(BaseForces):
    def _calculate_demon_forces(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_mask: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
        method: Literal["active", "passive", "dual"] = "dual",
        fixed_image_gradient: torch.Tensor | None = None,
        original_image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        use_masks: bool = True,
    ):
        # if method == 'dual' use union of moving and fixed mask
        if self.method == "dual":
            union_mask = torch.logical_or(moving_mask, fixed_mask)
            moving_mask, fixed_mask = union_mask, union_mask

        # grad tensors have the shape of (batch_size, 2 or 3, x_size, y_size[, z_size])
        total_grad = self._compute_total_grad(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            use_masks=use_masks,
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
        use_masks: bool = True,
    ):
        return self._calculate_demon_forces(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
            method=self.method,
            original_image_spacing=original_image_spacing,
            use_masks=use_masks,
        )
