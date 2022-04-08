from abc import ABC
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        sigma: Tuple[float, ...],
        sigma_cutoff: Tuple[float, ...],
        same_size: bool = False,
    ):
        if len(sigma) != len(sigma_cutoff):
            raise ValueError("sigma and sigma_cutoff has to be same length")

        n_dim = len(sigma)

        max_kernel_radius = max(
            BaseGaussianSmoothing.get_kernel_radius(s, c)
            for s, c in zip(sigma, sigma_cutoff)
        )

        kernels = []
        for i in range(n_dim):
            if same_size:
                kernel_1d = BaseGaussianSmoothing._make_gaussian_kernel_1d(
                    sigma=sigma[i], radius=max_kernel_radius
                )
            else:
                kernel_1d = BaseGaussianSmoothing._make_gaussian_kernel_1d(
                    sigma=sigma[i], sigma_cutoff=sigma_cutoff[i]
                )
            if n_dim == 3:
                kernel = torch.einsum("i,j,k->ijk", kernel_1d, kernel_1d, kernel_1d)
            elif n_dim == 2:
                kernel = torch.einsum("i,j->ij", kernel_1d, kernel_1d)
            kernel = kernel / kernel.sum()
            kernels.append(kernel)

        return kernels


class GaussianSmoothing2d(BaseGaussianSmoothing):
    def __init__(
        self,
        sigma: Tuple[float, float] = (1.0, 1.0),
        sigma_cutoff: Tuple[float, float] = (2.0, 2.0),
        same_size: bool = False,
    ):
        super().__init__()

        if not (len(sigma) == len(sigma_cutoff) == 2):
            raise ValueError("Length of sigma and sigma_cutoff has to be 2")

        self.sigma = sigma
        self.sigma_cutoff = sigma_cutoff
        self.same_size = same_size

        kernel_x, kernel_y = GaussianSmoothing2d.make_gaussian_kernel(
            sigma=self.sigma, sigma_cutoff=self.sigma_cutoff
        )
        self.kernel_size = (kernel_x.shape[-1], kernel_y.shape[-1])

        kernel_x, kernel_y = (
            kernel_x[None, None],
            kernel_y[None, None],
        )
        if self.same_size:
            kernel = torch.cat((kernel_x, kernel_y), dim=0)
            self.register_buffer("weight", kernel)

        else:
            self.register_buffer("weight_x", kernel_x)
            self.register_buffer("weight_y", kernel_y)

    def forward(self, image):
        if self.same_size:
            image = F.conv2d(image, self.weight, groups=2, stride=1, padding="same")
        else:
            image_x = F.conv2d(image[:, 0:1], self.weight_x, stride=1, padding="same")
            image_y = F.conv2d(image[:, 1:2], self.weight_y, stride=1, padding="same")
            image = torch.cat((image_x, image_y), dim=1)

        return image


class GaussianSmoothing3d(BaseGaussianSmoothing):
    def __init__(
        self,
        sigma: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        sigma_cutoff: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        same_size: bool = False,
    ):
        super().__init__()

        if not (len(sigma) == len(sigma_cutoff) == 3):
            raise ValueError("Length of sigma and sigma_cutoff has to be 3")

        self.sigma = sigma
        self.sigma_cutoff = sigma_cutoff
        self.same_size = same_size

        kernel_x, kernel_y, kernel_z = GaussianSmoothing3d.make_gaussian_kernel(
            sigma=self.sigma, sigma_cutoff=self.sigma_cutoff
        )
        self.kernel_size = (kernel_x.shape[-1], kernel_y.shape[-1], kernel_z.shape[-1])

        kernel_x, kernel_y, kernel_z = (
            kernel_x[None, None],
            kernel_y[None, None],
            kernel_z[None, None],
        )
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
    """
    N-D Spatial Transformer
    """

    def __init__(self, shape, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        grid = self._create_grid(shape)
        self.register_buffer("grid", grid, persistent=False)

    def _create_grid(self, shape):
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        return grid

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class DemonForces3d(nn.Module):
    @staticmethod
    def _calculate_demon_forces_3d(
        image: torch.tensor,
        reference_image: torch.tensor,
        method: str,
        original_image_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        epsilon: float = 1e-6,
    ):
        if method == "active":
            x_grad, y_grad, z_grad = torch.gradient(image, dim=(2, 3, 4))
        elif method == "passive":
            x_grad, y_grad, z_grad = torch.gradient(reference_image, dim=(2, 3, 4))
            # TODO: Has to be only calculated once per level,
            #  as reference image does not change -> To be implemented
        elif method == "symmetric":
            x_grad, y_grad, z_grad = torch.stack(
                torch.gradient(image, dim=(2, 3, 4))
            ) + torch.stack(torch.gradient(reference_image, dim=(2, 3, 4)))
            # TODO: Has to be only calculated once per level,
            #  as reference image does not change -> To be implemented
        else:
            raise Exception("Specified demon forces not implemented")
        gamma = 1 / (
            (sum(i * i for i in original_image_spacing)) / len(original_image_spacing)
        )
        l2_grad = (
            x_grad**2 + y_grad**2 + z_grad**2
        )  # TODO: Same as above, if method == passive
        norm = (reference_image - image) / (
            epsilon + l2_grad + gamma * (reference_image - image) ** 2
        )

        return norm * torch.cat((x_grad, y_grad, z_grad), dim=1)

    def forward(
        self,
        image: torch.tensor,
        reference_image: torch.tensor,
        method: str,
        original_image_spacing: Tuple[float, ...],
    ):
        return DemonForces3d._calculate_demon_forces_3d(
            image=image,
            reference_image=reference_image,
            method=method,
            original_image_spacing=original_image_spacing,
        )
