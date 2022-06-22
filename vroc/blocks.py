from abc import ABC
from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            layers.extend([norm_layer(out_channels), nn.LeakyReLU(inplace=True)])
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
            layers.extend([norm_layer(out_channels), nn.LeakyReLU(inplace=True)])
        self.convs = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2:
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
        if x2:
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
        sigma: Tuple[float, ...],
        sigma_cutoff: Tuple[float, ...],
        same_size: bool = False,
        radius: int = None,
    ):
        if len(sigma) != len(sigma_cutoff):
            raise ValueError("sigma and sigma_cutoff has to be same length")

        n_dim = len(sigma)

        if not radius:
            radius = max(
                BaseGaussianSmoothing.get_kernel_radius(s, c)
                for s, c in zip(sigma, sigma_cutoff)
            )

        kernels = []
        for i in range(n_dim):
            if same_size:
                kernel_1d = BaseGaussianSmoothing._make_gaussian_kernel_1d(
                    sigma=sigma[i], radius=radius
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
        sigma_cutoff: Tuple[float, float] = (1.0, 1.0),
        same_size: bool = False,
        radius: int = None,
    ):
        super().__init__()

        if not (len(sigma) == len(sigma_cutoff) == 2):
            raise ValueError("Length of sigma and sigma_cutoff has to be 2")

        self.sigma = sigma
        self.sigma_cutoff = sigma_cutoff
        self.same_size = same_size

        kernel_x, kernel_y = GaussianSmoothing2d.make_gaussian_kernel(
            sigma=self.sigma,
            sigma_cutoff=self.sigma_cutoff,
            same_size=same_size,
            radius=radius,
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
        sigma_cutoff: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        same_size: bool = False,
        radius: int = None,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        use_image_spacing: bool = False,
    ):
        super().__init__()

        if not (len(sigma) == len(sigma_cutoff) == 3):
            raise ValueError("Length of sigma and sigma_cutoff has to be 3")

        self.sigma = sigma
        self.sigma_cutoff = sigma_cutoff
        self.same_size = same_size
        self.use_image_spacing = use_image_spacing
        self.spacing = spacing

        if self.use_image_spacing:
            self.sigma = tuple(
                elem_1 * elem_2 for elem_1, elem_2 in zip(self.sigma, self.spacing)
            )
        kernel_x, kernel_y, kernel_z = GaussianSmoothing3d.make_gaussian_kernel(
            sigma=self.sigma,
            sigma_cutoff=self.sigma_cutoff,
            same_size=same_size,
            radius=radius,
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
    """N-D Spatial Transformer."""

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
