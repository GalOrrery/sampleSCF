# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
from typing import Any

# THIRD PARTY
import numpy as np
from galpy.potential import SCFPotential
from numpy.typing import ArrayLike
from scipy.interpolate import InterpolatedUnivariateSpline

# LOCAL
from sample_scf._typing import NDArrayF
from sample_scf.base import r_distribution_base
from sample_scf.utils import r_of_zeta, zeta_of_r

__all__ = ["r_distribution"]


##############################################################################
# CODE
##############################################################################


class r_distribution(r_distribution_base):
    """Sample radial coordinate from an SCF potential.

    The potential must have a convergent mass function.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
    rgrid : ndarray
    **kw
        Passed to `scipy.stats.rv_continuous`
        "a", "b" are set to [0, inf]
    """

    def __init__(self, potential: SCFPotential, rgrid: NDArrayF, **kw: Any) -> None:
        kw["a"], kw["b"] = 0, np.nanmax(rgrid)  # allowed range of r
        super().__init__(potential, **kw)

        mgrid = np.array([potential._mass(x) for x in rgrid])  # :(
        # manual fixes for endpoints and normalization
        ind = np.where(np.isnan(mgrid))[0]
        mgrid[ind[rgrid[ind] == 0]] = 0
        mgrid = (mgrid - np.nanmin(mgrid)) / (np.nanmax(mgrid) - np.nanmin(mgrid))  # rescale
        infind = ind[rgrid[ind] == np.inf].squeeze()
        mgrid[infind] = 1
        if mgrid[infind - 1] == 1:  # munge the rescaling  TODO! do better
            mgrid[infind - 1] -= min(1e-8, np.diff(mgrid[slice(infind - 2, infind)]) / 2)

        # work in zeta, not r, since it is more numerically stable
        zeta = zeta_of_r(rgrid)
        # make splines for fast calculation
        self._spl_cdf = InterpolatedUnivariateSpline(
            zeta,
            mgrid,
            ext="raise",
            bbox=[-1, 1],
        )
        self._spl_ppf = InterpolatedUnivariateSpline(
            mgrid,
            zeta,
            ext="raise",
            bbox=[0, 1],
        )

        # TODO! make sure
        # # store endpoint values to ensure CDF normalized to [0, 1]
        # self._mi = self._spl_cdf(min(zeta))
        # self._mf = self._spl_cdf(max(zeta))

    def _cdf(self, r: ArrayLike, *args: Any, **kw: Any) -> NDArrayF:
        cdf: NDArrayF = self._spl_cdf(zeta_of_r(r))
        # (self._scfmass(zeta) - self._mi) / (self._mf - self._mi)
        # TODO! is this normalization even necessary?
        return cdf

    def _ppf(self, q: ArrayLike, *args: Any, **kw: Any) -> NDArrayF:
        return r_of_zeta(self._spl_ppf(q))
