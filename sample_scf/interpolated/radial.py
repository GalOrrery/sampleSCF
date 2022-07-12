# -*- coding: utf-8 -*-

"""Radial sampling."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Any, Tuple

# THIRD PARTY
import astropy.units as u
from astropy.units import Quantity
from galpy.potential import SCFPotential
from numpy import argsort, array, diff, inf, isnan, nanmax, nanmin, where
from numpy.typing import ArrayLike
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

# LOCAL
from sample_scf._typing import NDArrayF
from sample_scf.base_univariate import r_distribution_base
from sample_scf.representation import r_of_zeta, zeta_of_r

__all__ = ["interpolated_r_distribution"]


##############################################################################
# CODE
##############################################################################


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

    def __init__(self, potential: SCFPotential, radii: Quantity, **kw: Any) -> None:
        kw["a"], kw["b"] = 0, nanmax(radii)  # allowed range of r
        super().__init__(potential, **kw)

        # fraction of total mass grid
        # work in zeta, not r, since it is more numerically stable
        self._radii, self._zetas = self.order_radii(radii)
        self._mgrid = self.calculate_cumulative_mass(self._radii)

        # make splines for fast calculation
        self._spl_cdf = IUS(self._zetas, self._mgrid, ext="raise", bbox=[-1, 1], k=1)
        self._spl_ppf = IUS(self._mgrid, self._zetas, ext="raise", bbox=[0, 1], k=1)

    def order_radii(self, radii: Quantity) -> Tuple[Quantity, NDArrayF]:
        """Return ordered radii and zetas."""
        rsort = argsort(radii)  # same as zeta ordering
        radii = radii[rsort]
        zeta = zeta_of_r(radii, scale_radius=self.radial_scale_factor)
        return radii, zeta

    def calculate_cumulative_mass(self, radii: Quantity) -> NDArrayF:
        """Calculate cumulative mass function (ie the cdf).

        Parameters
        ----------
        radii : (R,) Quantity['length', float]

        Returns
        -------
        (R,) ndarray[float]
        """
        rgalpy = radii.to_value(u.kpc) / self.potential._ro
        mgrid = array([self.potential._mass(x) for x in rgalpy])  # :(
        # manual fixes for endpoints and normalization
        ind = where(isnan(mgrid))[0]
        mgrid[ind[radii[ind] == 0]] = 0
        mgrid = (mgrid - nanmin(mgrid)) / (nanmax(mgrid) - nanmin(mgrid))  # rescale
        infind = ind[radii[ind] == inf].squeeze()
        mgrid[infind] = 1
        if mgrid[infind - 1] == 1:  # munge the rescaling  TODO! do better
            mgrid[infind - 1] -= min(1e-8, diff(mgrid[slice(infind - 2, infind)]) / 2)

        return mgrid

    # ---------------------------------------------------------------

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
