from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from vroc.common_types import FloatTuple3D


class TRELoss(nn.Module):
    def __init__(
        self,
        image_spacing: FloatTuple3D = (1.0, 1.0, 1.0),
        apply_sqrt: bool = False,
        reduction: Literal["mean", "sum", "none"] | None = "mean",
    ):
        super().__init__()
        self.apply_sqrt = apply_sqrt
        # convert PyTorch's unpythonic string "none"
        self.reduction = reduction if reduction != "none" else None
        self.register_buffer("image_spacing", torch.as_tensor(image_spacing))

    def forward(
        self,
        vector_field: torch.Tensor,
        moving_landmarks: torch.Tensor,
        fixed_landmarks: torch.Tensor,
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
        distances = distances * self.image_spacing
        distances = distances.pow(2).sum(dim=-1)

        if self.apply_sqrt:
            distances = distances.sqrt()

        if self.reduction == "mean":
            distances = distances.mean()
        elif self.reduction == "sum":
            distances = distances.sum()
        elif not self._reduction:
            # do nothing; this also covers falsy values like None, False, 0
            pass
        else:
            raise RuntimeError(f"Unsupported reduction {self._reduction}")

        return distances
