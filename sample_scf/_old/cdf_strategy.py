# -*- coding: utf-8 -*-

"""
Deal with non-monotonic CDFs.
The problem can arise if the PDF (density field) ever dips negative because of
an incorrect solution to the SCF coefficients. E.g. when solving for the
coefficients from an analytic density profile.

"""

# __all__ = []


##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Type, Union, cast

# THIRD PARTY
import numpy as np
from astropy.utils.state import ScienceState

# LOCAL
from sample_scf._typing import NDArrayF

__all__ = ["get_strategy", "CDFStrategy", "NoCDFStrategy", "LinearInterpolateCDFStrategy"]


##############################################################################
# PARAMETERS

CDF_STRATEGIES: Dict[Union[str, None], Type[CDFStrategy]] = {}


StrategyLike = Union[str, None, "CDFStrategy"]
"""Type variable describing."""

##############################################################################
# CODE
##############################################################################


def get_strategy(key: StrategyLike, /) -> CDFStrategy:
    item: CDFStrategy
    if isinstance(key, CDFStrategy):
        item = key
    elif key in CDF_STRATEGIES:
        item = CDF_STRATEGIES[key]()
    else:
        raise ValueError

    return item


# ============================================================================


class CDFStrategy(metaclass=ABCMeta):
    def __init_subclass__(cls, key: str, **kwargs: Any) -> None:
        super().__init_subclass__()

        CDF_STRATEGIES[key] = cls

    @classmethod
    @abstractmethod
    def apply(cls, cdf: NDArrayF, **kw: Any) -> NDArrayF:
        """Apply CDF strategy.

        .. warning::
            operates in-place on numpy arrays

        Parameters
        ----------
        cdf : array[float]
        **kw : Any
            Not used.

        Returns
        -------
        cdf : array[float]
            Modified in-place.
        """


class NoCDFStrategy(CDFStrategy, key=None):
    @classmethod
    def apply(cls, cdf: NDArrayF, **kw: Any) -> NDArrayF:
        """

        .. warning::
            operates in-place on numpy arrays

        """
        # find where cdf breaks monotonicity
        notreal = np.where(np.diff(cdf) <= 0)[0] + 1
        # raise error if any breaks
        if np.any(notreal):
            msg = "cdf contains unreal elements "
            msg += f"at index {kw['index']}" if "index" in kw else ""
            raise ValueError(msg)


class LinearInterpolateCDFStrategy(CDFStrategy, key="linear"):
    @classmethod
    def apply(cls, cdf: NDArrayF, **kw: Any) -> NDArrayF:
        """Apply linear interpolation.

        .. warning::

            operates in-place on numpy arrays

        Parameters
        ----------
        cdf : array[float]
        **kw : Any
            Not used.

        Returns
        -------
        cdf : array[float]
            Modified in-place.
        """
        # Find where cdf breaks monotonicity, and the startpoint of each break.
        notreal = np.where(np.diff(cdf) <= 0)[0] + 1
        breaks = np.where(np.diff(notreal) > 1)[0] + 1
        startnotreal = np.concatenate((notreal[:1], notreal[breaks]))

        # Loop over each start. Can't vectorize because they depend on each other.
        for i in startnotreal:
            i0 = i - 1  # before it dips negative
            i1 = i0 + np.argmax(cdf[i0:] - cdf[i0] > 0)  # start of net positive
            cdf[i0 : i1 + 1] = np.linspace(cdf[i0], cdf[i1], num=i1 - i0 + 1, endpoint=True)

        return cdf
