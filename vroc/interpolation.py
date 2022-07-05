from typing import Tuple

import numpy as np
import scipy.ndimage as ndi


def resize(image: np.ndarray, output_shape: Tuple[int, ...], order: int = 1):
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


def downscale(image: np.ndarray, factor: float, order: int = 1):
    downscaled_shape = tuple(s / factor for s in image.shape)
    return resize(image=image, output_shape=downscaled_shape, order=order)


def resize_spacing(
    image: np.ndarray,
    input_image_spacing: Tuple[float, ...],
    output_image_spacing: Tuple[float, ...],
    order: int = 1,
):
    output_shape = tuple(
        int(round(sh * in_sp / out_sp))
        for (in_sp, out_sp, sh) in zip(
            input_image_spacing, output_image_spacing, image.shape
        )
    )
    return resize(image=image, output_shape=output_shape, order=order)
