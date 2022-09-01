from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vroc.blocks import SpatialTransformer
from vroc.common_types import FloatTuple3D, IntTuple3D


class TRELoss(nn.Module):
    def __init__(
        self,
        apply_sqrt: bool = False,
        reduction: Literal["mean", "sum", "none", "quantile_0.95"] | None = "mean",
    ):
        super().__init__()
        self.apply_sqrt = apply_sqrt
        # convert PyTorch's unpythonic string "none"
        self.reduction = reduction if reduction != "none" else None

        if self.reduction and self.reduction.startswith("quantile"):
            self.reduction, self.quantile = self.reduction.split("_")
            self.quantile = float(self.quantile)

    def forward(
        self,
        vector_field: torch.Tensor,
        moving_landmarks: torch.Tensor,
        fixed_landmarks: torch.Tensor,
        image_spacing: torch.Tensor,
    ):
        # vector_field: shape of (1, 3, x_dim, y_dim, z_dim), values are in voxel
        # displacement (i.e. not torch grid_sample convention [-1, 1])
        # {moving,fixed}_landmarks: shape of (1, n_landmarks, 3)

        # currently only implemented for batch size of 1
        # get rid of batch dimension
        vector_field = torch.as_tensor(vector_field[0], dtype=torch.float32)
        moving_landmarks = moving_landmarks[0]
        fixed_landmarks = fixed_landmarks[0]

        # round fixed_landmarks coordinates if dtype is float,
        # i.e. NN interpolation of vector_field
        if torch.is_floating_point(fixed_landmarks):
            fixed_landmarks = fixed_landmarks.round().to(torch.long)
        # get voxel of vector_field for each fixed landmark
        x_coordinates = fixed_landmarks[..., 0]
        y_coordinates = fixed_landmarks[..., 1]
        z_coordinates = fixed_landmarks[..., 2]
        # displacements is of shape (3, n_landmarks) after transposing
        displacements = vector_field[:, x_coordinates, y_coordinates, z_coordinates].T

        # warp fixed_landmarks and compare to moving_landmarks (euclidean distance)
        # distances will be float32 as displacements is float32
        distances = (fixed_landmarks + displacements) - moving_landmarks
        # scale x, x, z distance component with image spacing
        distances = distances * image_spacing
        distances = distances.pow(2).sum(dim=-1)

        if self.apply_sqrt:
            distances = distances.sqrt()

        if self.reduction == "mean":
            distances = distances.mean()
        elif self.reduction == "sum":
            distances = distances.sum()
        elif self.reduction == "quantile":
            distances = torch.quantile(distances, q=self.quantile)
        elif not self.reduction:
            # do nothing; this also covers falsy values like None, False, 0
            pass
        else:
            raise RuntimeError(f"Unsupported reduction {self._reduction}")

        return distances


class WarpedMSELoss(nn.Module):
    def __init__(self, shape: IntTuple3D):
        super().__init__()
        self.spatial_transformer = SpatialTransformer(shape=shape)

    def forward(
        self,
        moving_image: torch.Tensor,
        vector_field: torch.Tensor,
        fixed_image: torch.Tensor,
        fixed_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        warped_image = self.spatial_transformer(moving_image, vector_field)

        return F.mse_loss(warped_image[fixed_image_mask], fixed_image[fixed_image_mask])


def mse_loss(
    moving_image: torch.Tensor, fixed_image: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(moving_image[mask], fixed_image[mask])


def ncc_loss(
    moving_image: torch.Tensor, fixed_image: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    moving_image = torch.masked_select(moving_image, mask)
    fixed_image = torch.masked_select(fixed_image, mask)
    value = (
        -1.0
        * torch.sum(
            (fixed_image - torch.mean(fixed_image))
            * (moving_image - torch.mean(moving_image))
        )
        / torch.sqrt(
            torch.sum((fixed_image - torch.mean(fixed_image)) ** 2)
            * torch.sum((moving_image - torch.mean(moving_image)) ** 2)
            + 1e-10
        )
    )

    return value


def ngf_loss(
    moving_image: torch.Tensor,
    fixed_image: torch.Tensor,
    mask: torch.tensor,
    epsilon=1e-5,
) -> torch.Tensor:
    dx_f = fixed_image[..., 1:, 1:, 1:] - fixed_image[..., :-1, 1:, 1:]
    dy_f = fixed_image[..., 1:, 1:, 1:] - fixed_image[..., 1:, :-1, 1:]
    dz_f = fixed_image[..., 1:, 1:, 1:] - fixed_image[..., 1:, 1:, :-1]

    if epsilon is None:
        with torch.no_grad():
            epsilon = torch.mean(torch.abs(dx_f) + torch.abs(dy_f) + torch.abs(dz_f))

    norm = torch.sqrt(dx_f.pow(2) + dy_f.pow(2) + dz_f.pow(2) + epsilon**2)

    ngf_fixed_image = F.pad(
        torch.cat((dx_f, dy_f, dz_f), dim=1) / norm, (0, 1, 0, 1, 0, 1)
    )

    dx_m = moving_image[..., 1:, 1:, 1:] - moving_image[..., :-1, 1:, 1:]
    dy_m = moving_image[..., 1:, 1:, 1:] - moving_image[..., 1:, :-1, 1:]
    dz_m = moving_image[..., 1:, 1:, 1:] - moving_image[..., 1:, 1:, :-1]

    norm = torch.sqrt(dx_m.pow(2) + dy_m.pow(2) + dz_m.pow(2) + epsilon**2)

    ngf_moving_image = F.pad(
        torch.cat((dx_m, dy_m, dz_m), dim=1) / norm, (0, 1, 0, 1, 0, 1)
    )

    value = 0
    for dim in range(3):
        value = value + ngf_moving_image[:, dim, ...] * ngf_fixed_image[:, dim, ...]

    value = 0.5 * torch.masked_select(-value.pow(2), mask)

    return value.mean()


def jacobian_determinant(vector_field: torch.Tensor) -> torch.Tensor:
    # vector field has shape (1, 3, x, y, z)
    dx, dy, dz = torch.gradient(vector_field, dim=(2, 3, 4))

    # add identity matrix: det(dT/dx) = det(I + du/dx)
    dx[:, 0] = dx[:, 0] + 1
    dy[:, 1] = dy[:, 1] + 1
    dz[:, 2] = dz[:, 2] + 1

    # Straightforward application of rule of sarrus yields the following lines.

    # sarrus_plus_1 = dx[:, 0] * dy[:, 1] * dz[:, 2]
    # sarrus_plus_2 = dy[:, 0] * dz[:, 1] * dx[:, 2]
    # sarrus_plus_3 = dz[:, 0] * dx[:, 1] * dy[:, 2]
    #
    # sarrus_minus_1 = dx[:, 2] * dy[:, 1] * dz[:, 0]
    # sarrus_minus_2 = dy[:, 2] * dz[:, 1] * dx[:, 0]
    # sarrus_minus_3 = dz[:, 2] * dx[:, 1] * dy[:, 0]
    #
    # det_j = (sarrus_plus_1 + sarrus_plus_2 + sarrus_plus_3) - (
    #     sarrus_minus_1 + sarrus_minus_2 + sarrus_minus_3
    # )

    # However, we factor out ∂VFx/∂x, ∂VFx/∂y, ∂VFx/∂z to save a few FLOPS:

    det_j = (
        dx[:, 0] * (dy[:, 1] * dz[:, 2] - dy[:, 2] * dz[:, 1])
        + dy[:, 0] * (dz[:, 1] * dx[:, 2] - dz[:, 2] * dx[:, 1])
        + dz[:, 0] * (dx[:, 1] * dy[:, 2] - dx[:, 2] * dy[:, 1])
    )

    return det_j[:, None]


def smooth_vector_field_loss(
    vector_field: torch.Tensor, mask: torch.Tensor, l2r_variant: bool = False
) -> torch.Tensor:
    det_j = jacobian_determinant(vector_field)

    if l2r_variant:
        det_j = det_j + 3
        det_j = torch.clip(det_j, 1e-9, 1e9)
        det_j = torch.log(det_j)

    return det_j[mask].std()
