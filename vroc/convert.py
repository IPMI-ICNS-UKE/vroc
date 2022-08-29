from __future__ import annotations

import numpy as np
import torch


def as_tensor(
    image: np.ndaray | torch.Tensor,
    n_dim: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    current_n_dims = image.ndim
    if n_dim < current_n_dims:
        raise RuntimeError("Dimension mismatch")

    if n_dims_to_pad := n_dim - current_n_dims:
        image = image[(None,) * n_dims_to_pad]

    return torch.as_tensor(image, dtype=dtype, device=device)
