from __future__ import annotations

import warnings
from typing import Callable, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from vroc.blocks import SpatialTransformer
from vroc.common_types import FloatTuple3D, IntTuple3D
from vroc.helper import to_one_hot


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

    @staticmethod
    def _warped_fixed_landmarks(
        vector_field: torch.Tensor,
        fixed_landmarks: torch.Tensor,
    ) -> torch.Tensor:
        # vector_field: shape of (1, 3, x_dim, y_dim, z_dim), values are in voxel
        # displacement (i.e. not torch grid_sample convention [-1, 1])
        # {moving,fixed}_landmarks: shape of (1, n_landmarks, 3)

        # currently only implemented for batch size of 1
        # get rid of batch dimension
        vector_field = torch.as_tensor(vector_field[0], dtype=torch.float32)

        # round fixed_landmarks coordinates if dtype is float,
        # i.e. NN interpolation of vector_field
        if torch.is_floating_point(fixed_landmarks):
            fixed_landmarks = fixed_landmarks.round().to(torch.long)
        # get voxel of vector_field for each fixed landmark
        x_coordinates = fixed_landmarks[..., 0]
        y_coordinates = fixed_landmarks[..., 1]
        z_coordinates = fixed_landmarks[..., 2]
        # displacement is of shape (3, n_landmarks) after transposing
        displacements = vector_field[:, x_coordinates, y_coordinates, z_coordinates].T

        return fixed_landmarks + displacements

    def forward(
        self,
        vector_field: torch.Tensor,
        moving_landmarks: torch.Tensor,
        fixed_landmarks: torch.Tensor,
        image_spacing: torch.Tensor | FloatTuple3D,
    ):
        fixed_landmarks = fixed_landmarks[0]
        moving_landmarks = moving_landmarks[0]
        # warp fixed_landmarks and compare to moving_landmarks (euclidean distance)
        # distances will be float32 as displacements is float32

        distances = (
            TRELoss._warped_fixed_landmarks(
                fixed_landmarks=fixed_landmarks, vector_field=vector_field
            )
            - moving_landmarks
        )
        # scale x, x, z distance component with image spacing
        image_spacing = torch.as_tensor(
            image_spacing, dtype=torch.float32, device=distances.device
        )
        distances = distances * image_spacing
        distances = distances.pow(2).sum(dim=-1)

        if self.apply_sqrt:
            distances = distances.sqrt()

        if self.reduction == "mean":
            distances = distances.mean()
        elif self.reduction == "median":
            distances = torch.median(distances)
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


def calculate_gradient_l2(image: torch.tensor, eps: float = 1e-6) -> torch.tensor:
    x_grad, y_grad, z_grad = torch.gradient(image, dim=(2, 3, 4))
    l2_grad = torch.sqrt((eps + x_grad**2 + y_grad**2 + z_grad**2))

    return l2_grad


class WarpedMSELoss(nn.Module):
    def __init__(self, shape: IntTuple3D | None = None, edge_weighting: float = 0.0):
        super().__init__()
        self.spatial_transformer = SpatialTransformer(shape=shape)
        self.edge_weighting = edge_weighting

    def forward(
        self,
        moving_image: torch.Tensor,
        vector_field: torch.Tensor,
        fixed_image: torch.Tensor,
        fixed_mask: torch.Tensor,
    ) -> torch.Tensor:
        warped_image = self.spatial_transformer(
            image=moving_image, transformation=vector_field, mode="bilinear"
        )

        loss = F.mse_loss(warped_image[fixed_mask], fixed_image[fixed_mask])

        if self.edge_weighting > 0.0:
            warped_image_l2_grad = calculate_gradient_l2(warped_image)[fixed_mask]
            fixed_image_l2_grad = calculate_gradient_l2(fixed_image)[fixed_mask]

            loss = loss + self.edge_weighting * F.l1_loss(
                warped_image_l2_grad, fixed_image_l2_grad
            )

        return loss


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
