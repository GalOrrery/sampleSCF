# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import itertools
import warnings
from typing import Any, Optional, Union, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from galpy.potential import SCFPotential
from numpy.typing import ArrayLike
from scipy.interpolate import RectBivariateSpline, splev, splrep

# LOCAL
from sample_scf._typing import NDArrayF, RandomLike
from sample_scf.base_univariate import theta_distribution_base
from sample_scf.exact.inclination import exact_theta_distribution_base
from sample_scf.representation import x_of_theta, zeta_of_r

__all__ = ["interpolated_theta_distribution"]


##############################################################################
# CODE
##############################################################################


class interpolated_theta_distribution(theta_distribution_base):
    """
    Sample inclination coordinate from an SCF potential.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`
    radii : (R,) |Quantity|
        Must be in correct sort order
    thetas : (T, ) ndarray ['radian'] or Quantity ['angle']
    intrp_step : float, optional
        Interpolation step.
    **kw
        Passed to `scipy.stats.rv_continuous`
        "a", "b" are set to [0, pi]
    """

    def __init__(
        self,
        potential: SCFPotential,
        radii: Quantity,
        thetas: Quantity,
        nintrp: float = 1e3,
        **kw: Any,
    ) -> None:
        Qls: NDArrayF = kw.pop("Qls", None)
        super().__init__(potential, **kw)  # allowed range of theta

        self._theta_interpolant = np.linspace(0, np.pi, num=int(nintrp)) << u.rad
        self._x_interpolant = x_of_theta(self._theta_interpolant)
        self._q_interpolant = np.linspace(0, 1, len(self._theta_interpolant))

        zetas_unsorted = zeta_of_r(radii, scale_radius=self.radial_scale_factor)  # (R,)
        rsort = np.argsort(zetas_unsorted)
        zetas = zetas_unsorted[rsort]

        # -------
        # build CDF in shells

        xs = np.sort(x_of_theta(thetas << u.rad))  # (T,)  sorted for interpolation
        Qls = Qls[rsort, :] if Qls is not None else self.calculate_Qls(radii)
        # check it's the right shape (R, L)
        if Qls.shape != (len(radii), self._lmax + 1):
            raise ValueError(f"Qls must be shape ({len(radii)}, {self._lmax + 1})")
        self._Qls: NDArrayF = Qls

        # calculate the CDFs exactly  # TODO! cleanup
        cdfs = exact_theta_distribution_base._cdf(self, xs, Qls)  # (R, T)

        # -------
        # interpolate
        # currently assumes a regular grid

        self._spl_cdf = RectBivariateSpline(  # (R, T)
            zetas,
            xs,
            cdfs,  # (R, T) is anti-theta ordered
            bbox=[-1, 1, -1, 1],  # [min(zeta), max(zeta), min(x), max(x)]
            kx=kw.get("kx", 2),
            ky=kw.get("ky", 2),
            s=kw.get("s", 0),
        )

        self._zetas = zetas  # FIXME!
        # return

        # ppf, one per r, supersampled
        # TODO! see if can use this to avoid resplining
        _cdfs = self._spl_cdf(zetas, self._x_interpolant[::-1], grid=True)
        spls = (  # work through the (R, T) is anti-theta ordered
            splrep(_cdfs[i, ::-1], self._theta_interpolant.value, s=0)
            for i in range(_cdfs.shape[0])
        )  # TODO! as generator
        ppfs = np.array([splev(self._q_interpolant, spl, ext=0) for spl in spls])

        self._spl_ppf = RectBivariateSpline(
            zetas,
            self._q_interpolant,
            ppfs,
            bbox=[-1, 1, 0, 1],  # [zetamin, zetamax, xmin, xmax]
            kx=kw.get("kx", 3),
            ky=kw.get("ky", 3),
            s=kw.get("s", 0),
        )

    def _cdf(self, x: ArrayLike, *args: Any, zeta: ArrayLike, **kw: Any) -> NDArrayF:
        cdf: NDArrayF = self._spl_cdf(zeta, x, grid=False)
        return cdf

    def cdf(self, theta: Quantity, r: ArrayLike) -> NDArrayF:
        """Cumulative Distribution Function.

        Parameters
        ----------
        theta : (T,) Quantity['angle']
        r : (R,) Quantity['length']

        Returns
        -------
        cdf : ndarray[float]
        """
        x = x_of_theta(theta << u.rad)
        zeta = zeta_of_r(r, scale_radius=self.radial_scale_factor)
        cdf = self._cdf(x, zeta=zeta)
        return cdf

    def _ppf(self, q: ArrayLike, *, r: ArrayLike, **kw: Any) -> NDArrayF:
        """Percent-point function.

        Parameters
        ----------
        q : float or (N,) array-like[float]
        r : float or (N,) array-like[float]

        Returns
        -------
        float or (N,) array-like[float]
            Same shape as 'r', 'q'.
        """
        zeta = zeta_of_r(r, scale_radius=self.radial_scale_factor)
        ppf: NDArrayF = self._spl_ppf(zeta, q, grid=False)
        return ppf

    def _rvs(
        self,
        r: Quantity,
        *,
        size: Optional[int] = None,
        random_state: Union[np.random.RandomState, np.random.Generator],
        # return_thetas: bool = True,  # TODO!
    ) -> NDArrayF:
        """Random variate sampling.

        Parameters
        ----------
        r : (R,) Quantity['length', float]
        size : int or None (optional, keyword-only)
        random_state : int or None (optional, keyword-only)

        Returns
        -------
        (size,) array-like[float]
        """
        # Use inverse cdf algorithm for RV generation.
        U = random_state.uniform(size=size)
        Y = self._ppf(U, r=r, grid=False)
        return Y

    def rvs(  # type: ignore
        self,
        r: Quantity,
        *,
        size: Optional[int] = None,
        random_state: RandomLike = None,
    ) -> Quantity:
        """Random variate sampling.

        Parameters
        ----------
        r : (R,) Quantity['length', float]
        size : int or None (optional, keyword-only)
        random_state : int or None (optional, keyword-only)

        Returns
        -------
        (R, size) Quantity[float]
            Shape 'size'.
        """
        return super().rvs(r, size=size, random_state=random_state) << u.rad
