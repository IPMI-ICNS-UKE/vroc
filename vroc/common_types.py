import os
from typing import Any, Callable, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch

# generic
PathLike = Union[os.PathLike, str]
Function = Callable[..., Any]

# numbers
Number = Union[int, float]
PositiveNumber = Number

# tuples
IntTuple = Tuple[int, ...]
FloatTuple = Tuple[float, ...]

IntTuple3D = Tuple[int, int, int]
FloatTuple3D = Tuple[float, float, float]

SlicingTuple3D = Tuple[slice, slice, slice]

# images
Image = Union[np.ndarray, sitk.Image]

# torch
TorchDevice = Union[str, torch.device]
