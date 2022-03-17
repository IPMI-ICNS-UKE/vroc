from typing import Optional

import numpy as np
import scipy.ndimage as ndi


def warp_image(
    image: np.ndarray,
    vector_field: np.ndarray,
    identity_grid: Optional[np.ndarray] = None,
    order: int = 3,
    mode: str = "-",
) -> np.ndarray:
    if identity_grid is None:
        identity_grid = np.mgrid[tuple(slice(None, s) for s in image.shape)]
    elif identity_grid.shape != image.shape:
        raise RuntimeError("Shape mismatch between identity grid and image")

    if mode == "-":
        coordinates = identity_grid - vector_field
    elif mode == "+":
        coordinates = identity_grid + vector_field
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    return ndi.map_coordinates(image, coordinates=coordinates, order=order)


class ImageWarper:
    def __init__(self, mode: str = "-"):
        self._identity_grid = None
        self._mode = mode

    def warp_image(
        self, image: np.ndarray, vector_field: np.ndarray, order: int = 3
    ) -> np.ndarray:
        if self._identity_grid is None:
            self._identity_grid = np.mgrid[tuple(slice(None, s) for s in image.shape)]

        return warp_image(
            image=image,
            vector_field=vector_field,
            identity_grid=self._identity_grid,
            order=order,
            mode=self._mode,
        )
