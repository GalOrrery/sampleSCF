# -*- coding: utf-8 -*-

"""Radial sampling."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Any

# THIRD PARTY
import astropy.units as u
import numpy as np
from galpy.potential import SCFPotential
from numpy.typing import ArrayLike
from scipy.interpolate import InterpolatedUnivariateSpline

# LOCAL
from sample_scf._typing import NDArrayF
from sample_scf.base_univariate import r_distribution_base
from sample_scf.representation import FiniteSphericalRepresentation, zeta_of_r, r_of_zeta

__all__ = ["interpolated_r_distribution"]


##############################################################################
# CODE
##############################################################################

def calculate_mass_cdf(potential: SCFPotential, radii: Quantity) -> NDArrayF:

    rgalpy = radii.to_value(u.kpc) / potential._ro  # FIXME! wrong scaling
    mgrid = np.array([potential._mass(x) for x in rgalpy])  # :(
    # manual fixes for endpoints and normalization
    ind = np.where(np.isnan(mgrid))[0]
    mgrid[ind[radii[ind] == 0]] = 0
    mgrid = (mgrid - np.nanmin(mgrid)) / (np.nanmax(mgrid) - np.nanmin(mgrid))  # rescale
    infind = ind[radii[ind] == np.inf].squeeze()
    mgrid[infind] = 1
    if mgrid[infind - 1] == 1:  # munge the rescaling  TODO! do better
        mgrid[infind - 1] -= min(1e-8, np.diff(mgrid[slice(infind - 2, infind)]) / 2)

    return mgrid


class interpolated_r_distribution(r_distribution_base):
    """Sample radial coordinate from an SCF potential.

    The potential must have a convergent mass function.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
    radii : Quantity
        Radii at which to interpolate.
    **kw
        Passed to `scipy.stats.rv_continuous`
        "a", "b" are set to [0, inf]
    """

    _interp_in_zeta: bool

    def __init__(
        self, potential: SCFPotential, radii: Quantity, **kw: Any) -> None:
        kw["a"], kw["b"] = 0, np.nanmax(radii)  # allowed range of r
        super().__init__(potential, **kw)

        ### fraction of total mass grid ###
        # work in zeta, not r, since it is more numerically stable
        zetas_unsorted = zeta_of_r(radii, scale_radius=self.radial_scale_factor)  # (R,)
        rsort = np.argsort(zetas_unsorted)
        zetas = zetas_unsorted[rsort]

        mgrid = calculate_mass_cdf(potential, radii[rsort])

        ### splines ###
        # make splines for fast calculation
        self._spl_cdf = InterpolatedUnivariateSpline(
            zetas,
            mgrid,
            ext="raise",
            bbox=[-1, 1],
            k=1,
        )
        self._spl_ppf = InterpolatedUnivariateSpline(
            mgrid,
            zetas,
            ext="raise",
            bbox=[0, 1],
            k=1,
        )

    def cdf(self, radii: Quantity):  # TODO!
        return self._cdf(zeta_of_r(radii, self.radial_scale_factor))

    def _cdf(self, zeta: NDArrayF, *args: Any, **kw: Any) -> NDArrayF:
        cdf: NDArrayF = self._spl_cdf(zeta)
        # (self._scfmass(zeta) - self._mi) / (self._mf - self._mi)
        # TODO! is this normalization even necessary?
        return cdf

    def _ppf(self, q: ArrayLike, *args: Any, **kw: Any) -> NDArrayF:
        zeta = self._spl_ppf(q)
        return r_of_zeta(zeta, self.radial_scale_factor)  # TODO! not convert in private function
