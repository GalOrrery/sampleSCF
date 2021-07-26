# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# BUILT-IN
import typing as T

# THIRD PARTY
import numpy as np
import numpy.typing as npt

RandomLike = T.Union[None, int, np.random.RandomState]
NDArray64 = npt.NDArray[np.float64]
