import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers


class GaussianSmoothing3d(nn.Module):
    def __init__(self, sigma=(1.0, 1.0, 1.0), sigma_cutoff: float = 2.0):
        super().__init__()

        self.sigma = sigma
        self.sigma_cutoff = sigma_cutoff

        kernel_x = GaussianSmoothing3d._make_gaussian_kernel(
            sigma=self.sigma[0], sigma_cutoff=self.sigma_cutoff
        )
        kernel_y = GaussianSmoothing3d._make_gaussian_kernel(
            sigma=self.sigma[1], sigma_cutoff=self.sigma_cutoff
        )
        kernel_z = GaussianSmoothing3d._make_gaussian_kernel(
            sigma=self.sigma[2], sigma_cutoff=self.sigma_cutoff
        )
        self.kernel_size = (len(kernel_x), len(kernel_y), len(kernel_z))

        kernel_x = torch.einsum("i,j,k->ijk", kernel_x, kernel_x, kernel_x)
        kernel_x = kernel_x / kernel_x.sum()
        kernel_x = kernel_x[None, None, ...]
        kernel_y = torch.einsum("i,j,k->ijk", kernel_y, kernel_y, kernel_y)
        kernel_y = kernel_y / kernel_y.sum()
        kernel_y = kernel_y[None, None, ...]
        kernel_z = torch.einsum("i,j,k->ijk", kernel_z, kernel_z, kernel_z)
        kernel_z = kernel_z / kernel_z.sum()
        kernel_z = kernel_z[None, None, ...]
        self.register_buffer("weight_x", kernel_x)
        self.register_buffer("weight_y", kernel_y)
        self.register_buffer("weight_z", kernel_z)

    @staticmethod
    def _make_gaussian_kernel(sigma, sigma_cutoff: float = 2.0):
        sd = float(sigma)
        # make the radius of the filter equal to truncate standard deviations
        radius = int(sigma_cutoff * sd + 0.5)
        if radius == 0:
            radius = 1

        sigma2 = sigma * sigma
        x = torch.arange(-radius, radius + 1)
        phi_x = torch.exp(-0.5 / sigma2 * x ** 2)
        phi_x = phi_x / phi_x.sum()

        return phi_x

    def forward(self, input):
        padded_x = F.pad(
            input[:, 0:1], (self.kernel_size[0] // 2,) * 6, mode="constant"
        )
        padded_y = F.pad(
            input[:, 1:2], (self.kernel_size[1] // 2,) * 6, mode="constant"
        )
        padded_z = F.pad(
            input[:, 2:3], (self.kernel_size[2] // 2,) * 6, mode="constant"
        )

        input[:, 0:1] = F.conv3d(padded_x, self.weight_x, stride=1)
        input[:, 1:2] = F.conv3d(padded_y, self.weight_y, stride=1)
        input[:, 2:3] = F.conv3d(padded_z, self.weight_z, stride=1)

        return input
#
#
# class GaussianSmoothing(nn.Module):
#     """
#     Apply gaussian smoothing on a
#     1d, 2d or 3d tensor. Filtering is performed seperately for each channel
#     in the input using a depthwise convolution.
#     Arguments:
#         channels (int, sequence): Number of channels of the input tensors. Output will
#             have this number of channels as well.
#         kernel_size (int, sequence): Size of the gaussian kernel.
#         sigma (float, sequence): Standard deviation of the gaussian kernel.
#         dim (int, optional): The number of dimensions of the data.
#             Default value is 2 (spatial).
#     """
#
#     def __init__(self, channels, kernel_size, sigma, dim=2):
#         super(GaussianSmoothing, self).__init__()
#         if isinstance(kernel_size, numbers.Number):
#             kernel_size = [kernel_size] * dim
#         if isinstance(sigma, numbers.Number):
#             sigma = [sigma] * dim
#
#         # The gaussian kernel is the product of the
#         # gaussian function of each dimension.
#         kernel = 1
#         meshgrids = torch.meshgrid(
#             [
#                 torch.arange(size, dtype=torch.float32)
#                 for size in kernel_size
#             ]
#         )
#         for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
#             mean = (size - 1) / 2
#             kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
#                       torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
#
#         # Make sure sum of values in gaussian kernel equals 1.
#         kernel = kernel / torch.sum(kernel)
#
#         # Reshape to depthwise convolutional weight
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
#
#         self.register_buffer('weight', kernel)
#         self.groups = channels
#
#         if dim == 1:
#             self.conv = F.conv1d
#         elif dim == 2:
#             self.conv = F.conv2d
#         elif dim == 3:
#             self.conv = F.conv3d
#         else:
#             raise RuntimeError(
#                 'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
#             )
#
#     def forward(self, input):
#         """
#         Apply gaussian filter to input.
#         Arguments:
#             input (torch.Tensor): Input to apply gaussian filter on.
#         Returns:
#             filtered (torch.Tensor): Filtered output.
#         """
#         return self.conv(input, weight=self.weight, groups=self.groups)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, shape, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        grid = self._create_grid(shape)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
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

        # s_t = time.time()
        out = F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        # print(time.time()-s_t)
        return out


class DemonForces3d(nn.Module):
    @staticmethod
    def _calculate_demon_forces_3d(
            image: torch.tensor,
            reference_image: torch.tensor,
            method: str,
            gamma: float = 1.0,
            epsilon: float = 1e-6,
    ):
        if method == 'active':
            x_grad, y_grad, z_grad = torch.gradient(image, dim=(2, 3, 4))
        elif method == 'passive':
            x_grad, y_grad, z_grad = torch.gradient(reference_image, dim=(2, 3, 4))
            # TODO: Has to be only calculated once per level, as reference image does not change -> To be implemented
        elif method == 'symmetric':
            x_grad, y_grad, z_grad = torch.stack(torch.gradient(image, dim=(2, 3, 4))) + \
                                     torch.stack(torch.gradient(reference_image, dim=(2, 3, 4)))
            # TODO: Has to be only calculated once per level, as reference image does not change -> To be implemented
        else:
            raise Exception('Specified demon forces not implemented')
        l2_grad = x_grad ** 2 + y_grad ** 2 + z_grad ** 2  # TODO: Same as above, if method == passive
        norm = (reference_image - image) / (
                epsilon + l2_grad + gamma * (reference_image - image) ** 2
        )

        return norm * torch.cat((x_grad, y_grad, z_grad), dim=1)

    def forward(self, image: torch.tensor, reference_image: torch.tensor, method: str):
        return DemonForces3d._calculate_demon_forces_3d(
            image=image, reference_image=reference_image, method=method,
        )
