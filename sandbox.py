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


if __name__ == "__main__":
    import time

    image = torch.ones((1, 2, 100, 100), device="cuda")

    slow1 = GaussianSmoothing2d(sigma=(1, 1), sigma_cutoff=(2, 2), same_size=False).to(
        image
    )
    slow2 = GaussianSmoothing2d(sigma=(1, 1), sigma_cutoff=(2, 2), same_size=False).to(
        image
    )

    fast1 = GaussianSmoothing2d(sigma=(1, 1), sigma_cutoff=(2, 2), same_size=True).to(
        image
    )
    fast2 = GaussianSmoothing2d(sigma=(1, 1), sigma_cutoff=(2, 2), same_size=True).to(
        image
    )

    t = time.time()
    fast1(image)
    print("fast1", time.time() - t)

    t = time.time()
    fast2(image)
    print("fast2", time.time() - t)

    t = time.time()
    slow1(image)
    print("slow1", time.time() - t)

    t = time.time()
    slow2(image)
    print("slow2", time.time() - t)
