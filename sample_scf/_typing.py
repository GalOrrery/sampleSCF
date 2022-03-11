# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Custom typing."""

# BUILT-IN
from typing import Union

# THIRD PARTY
import numpy as np
from numpy.typing import ArrayLike, NDArray

RandomGenerator = Union[np.random.RandomState, np.random.Generator]
RandomLike = Union[None, int, RandomGenerator]
NDArrayF = NDArray[np.floating]
