import torch
import torch.nn as nn
import torch.nn.functional as F

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

        sigma2 = sigma * sigma
        x = torch.arange(-radius, radius + 1)
        phi_x = torch.exp(-0.5 / sigma2 * x**2)
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
        if len(input.shape) == 5:
            input[:, 2:3] = F.conv3d(padded_z, self.weight_z, stride=1)

        return input


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

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode, padding_mode='reflection')


class DemonForces3d(nn.Module):
    @staticmethod
    def _calculate_active_demon_forces_3d(
        image: torch.tensor,
        reference_image: torch.tensor,
        gamma: float = 1.0,
        epsilon: float = 1e-6,
    ):
        if len(image.shape) == 5:
            x_grad, y_grad, z_grad = torch.gradient(image, dim=(2, 3, 4))
            l2_grad = x_grad**2 + y_grad**2 + z_grad**2
            norm = (reference_image - image) / (
                epsilon + l2_grad + gamma * (reference_image - image) ** 2
            )
            out = norm * torch.cat((x_grad, y_grad, z_grad), dim=1)
        elif len(image.shape) == 4:
            x_grad, y_grad = torch.gradient(image, dim=(2, 3))
            l2_grad = x_grad ** 2 + y_grad ** 2
            norm = (reference_image - image) / (
                    epsilon + l2_grad + gamma * (reference_image - image) ** 2
            )
            out = norm * torch.cat((x_grad, y_grad), dim=1)
        return out

    def forward(self, image: torch.tensor, reference_image: torch.tensor):
        return DemonForces3d._calculate_active_demon_forces_3d(
            image=image, reference_image=reference_image
        )
