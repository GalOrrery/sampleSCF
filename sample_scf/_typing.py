# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Custom typing."""

# STDLIB
from typing import Union

# THIRD PARTY
from numpy import floating
from numpy.random import Generator, RandomState
from numpy.typing import NDArray

__all__ = ["RandomGenerator", "RandomLike", "NDArrayF", "FArrayLike"]

RandomGenerator = Union[RandomState, Generator]
RandomLike = Union[None, int, RandomGenerator]
NDArrayF = NDArray[floating]

# float array-like
FArrayLike = Union[float, NDArrayF]
