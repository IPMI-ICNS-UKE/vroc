from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

from vroc.common_types import ArrayOrTensor, FloatTuple3D
from vroc.convert import as_tensor


def _resize_torch(
    image: torch.Tensor, output_shape: Tuple[int, ...], order: int = 1
) -> torch.Tensor:
    if order not in (0, 1):
        raise NotImplementedError(
            "Currently only nearest and linear interpolation is implemented"
        )

    LINEAR_INTERPOLATION_MODES = {
        3: "linear",
        4: "bilinear",
        5: "trilinear",
    }

    mode = LINEAR_INTERPOLATION_MODES[image.ndim] if order == 1 else "nearest"

    # pytorch can only resample float32
    if (dtype := image.dtype) != torch.float32:
        image = torch.as_tensor(image, dtype=torch.float32)

    # almost equivalent to numpy with align_corners=False
    # align_corners cannot be set if mode == nearest
    resized = F.interpolate(
        input=image,
        size=output_shape,
        mode=mode,
        align_corners=None if mode == "nearest" else False,
    )

    return resized


def _resize_numpy(
    image: np.ndarray, output_shape: Tuple[int, ...], order: int = 1
) -> np.ndarray:
    factors = np.asarray(image.shape, dtype=np.float32) / np.asarray(
        output_shape, dtype=np.float32
    )

    coord_arrays = [
        factors[i] * (np.arange(d) + 0.5) - 0.5 for i, d in enumerate(output_shape)
    ]

    coord_map = np.stack(np.meshgrid(*coord_arrays, sparse=False, indexing="ij"))
    image = image.astype(np.float32)
    out = ndi.map_coordinates(image, coord_map, order=order, mode="nearest")

    return out


def resize(
    image: np.ndarray | torch.Tensor, output_shape: Tuple[int, ...], order: int = 1
) -> np.ndarray | torch.Tensor:
    dtype = image.dtype
    is_mask = dtype in (bool, torch.bool)
    is_numpy = isinstance(image, np.ndarray)

    if is_numpy:
        resize_function = _resize_numpy
    else:
        resize_function = _resize_torch

    image = resize_function(image=image, output_shape=output_shape, order=order)

    if is_mask:
        image = image > 0.5

    if is_numpy:
        image = image.astype(dtype)
    else:
        image = image.to(dtype)

    return image


def rescale(image: ArrayOrTensor, factor: float, order: int = 1):
    is_numpy = isinstance(image, np.ndarray)

    if is_numpy:
        spatial_image_shape = image.shape
    else:
        spatial_image_shape = image.shape[2:]

    rescaled_shape = tuple(int(round(s * factor)) for s in spatial_image_shape)
    return resize(image=image, output_shape=rescaled_shape, order=order)


def resize_spacing(
    image: np.ndarray | torch.Tensor,
    input_image_spacing: Tuple[float, ...],
    output_image_spacing: Tuple[float, ...],
    order: int = 1,
) -> np.ndarray | torch.Tensor:
    if not (n_dim := len(input_image_spacing)) == len(output_image_spacing):
        raise ValueError(
            f"Dimension mismatch between "
            f"{input_image_spacing=} and {output_image_spacing=}"
        )

    output_shape = tuple(
        int(round(sh * in_sp / out_sp))
        for (in_sp, out_sp, sh) in zip(
            input_image_spacing, output_image_spacing, image.shape[-n_dim:]
        )
    )
    return resize(image=image, output_shape=output_shape, order=order)


if __name__ == "__main__":
    a_np = (np.random.random((50, 50, 50)) * 255).astype(np.uint8)
    aa_np = resize(image=a_np, output_shape=(99, 99, 99))

    a_th = torch.as_tensor(a_np[None, None])
    aa_th = resize(image=a_th, output_shape=(99, 99, 99))
    aa_th = aa_th.cpu().numpy().squeeze()
