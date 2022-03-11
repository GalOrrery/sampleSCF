# -*- coding: utf-8 -*-

"""
Deal with non-monotonic CDFs.
The problem can arise if the PDF (density field) ever dips negative because of
an incorrect solution to the SCF coefficients. E.g. when solving for the
coefficients from an analytic density profile.

"""

# __all__ = [
#     # functions
#     "",
#     # other
#     "",
# ]


##############################################################################
# IMPORTS

# BUILT-IN
import abc
import inspect

# THIRD PARTY
import numpy as np
from astropy.utils.state import ScienceState

##############################################################################
# PARAMETERS

CDF_STRATEGIES = {}

##############################################################################
# CODE
##############################################################################


class default_cdf_strategy(ScienceState):

    _value = "error"
    _default_value = "error"

    @classmethod
    def validate(cls, value):
        if value is None:
            value = self._default_value

        if isinstance(value, str):
            if value not in CDF_STRATEGIES:
                raise ValueError
            return CDF_STRATEGIES[value]
        elif inspect.isclass(value) and issubclass(value, CDFStrategy):
            return value
        else:
            raise TypeError()


# =============================================================================


class CDFStrategy:
    def __init_subclass__(cls, key, **kwargs):
        CDF_STRATEGIES[key] = cls

    @classmethod
    @abc.abstractmethod
    def apply(cls, cdf, **kw):
        pass


# -------------------------------------------------------------------


class Error(CDFStrategy, key="error"):
    @classmethod
    def apply(cls, cdf, **kw):
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


# -------------------------------------------------------------------


class LinearInterpolate(CDFStrategy, key="linear"):
    @classmethod
    def apply(cls, cdf, **kw):
        """

        .. warning::
            operates in-place on numpy arrays

        """
        # find where cdf breaks monotonicity
        # and the startpoint of each break.
        notreal = np.where(np.diff(cdf) <= 0)[0] + 1
        startnotreal = np.concatenate((notreal[:1], notreal[np.where(np.diff(notreal) > 1)[0] + 1]))

        for i in startnotreal[:-1]:
            i0 = i - 1  # before it dips negative
            i1 = i0 + np.argmax(cdf[i0:] - cdf[i0] > 0)  # start of net positive
            cdf[i0 : i1 + 1] = np.linspace(cdf[i0], cdf[i1], num=i1 - i0 + 1, endpoint=True)

        return cdf
