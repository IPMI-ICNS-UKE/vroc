import os
from typing import Any, Callable, Tuple, Union

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
