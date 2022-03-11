# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
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
from sample_scf.base import rv_potential
from sample_scf.utils import difPls, phiRSms, thetaQls, x_of_theta, zeta_of_r

__all__ = ["theta_distribution"]


##############################################################################
# CODE
##############################################################################


class theta_distribution(rv_potential):
    """
    Sample inclination coordinate from an SCF potential.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`
    rgrid, tgrid : ndarray
    **kw
        Passed to `scipy.stats.rv_continuous`
        "a", "b" are set to [-pi/2, pi/2]
    """

    def __init__(
        self,
        potential: SCFPotential,
        rgrid: NDArrayF,
        tgrid: NDArrayF,
        intrp_step: float = 0.01,
        **kw: Any,
    ) -> None:
        kw["a"], kw["b"] = -np.pi / 2, np.pi / 2
        Qls: NDArrayF = kw.pop("Qls", None)
        super().__init__(potential, **kw)  # allowed range of theta

        self._theta_interpolant = np.arange(-np.pi / 2, np.pi / 2, intrp_step)
        self._x_interpolant = x_of_theta(self._theta_interpolant)
        self._q_interpolant = np.linspace(0, 1, len(self._theta_interpolant))

        self._lrange = np.arange(0, self._lmax + 1)

        # -------
        # build CDF in shells
        # TODO: clean up shape stuff

        zetas = zeta_of_r(rgrid)  # (R,)
        xs = x_of_theta(tgrid)  # (T,)

        Qls = Qls if Qls is not None else thetaQls(potential, rgrid)
        # check it's the right shape (R, Lmax)
        if Qls.shape != (len(rgrid), self._lmax):
            raise ValueError(f"Qls must be shape ({len(rgrid)}, {self._lmax})")

        # l = 0 : spherical symmetry
        term0 = cast(NDArrayF, 0.5 * (xs + 1.0))  # (T,)
        # l = 1+ : non-symmetry
        factor = 1.0 / (2.0 * Qls[:, 0])  # (R,)
        term1p = np.sum(
            (Qls[None, :, 1:] * difPls(xs, self._lmax - 1).T[:, None, :]).T,
            axis=0,
        )

        cdfs = term0[None, :] + np.nan_to_num(factor[:, None] * term1p)  # (R, T)

        # -------
        # interpolate
        # currently assumes a regular grid

        self._spl_cdf = RectBivariateSpline(
            zetas,
            xs,
            cdfs,
            bbox=[-1, 1, -1, 1],  # [zetamin, zetamax, xmin, xmax]
            kx=kw.get("kx", 3),
            ky=kw.get("ky", 3),
            s=kw.get("s", 0),
        )

        # ppf, one per r
        # TODO! see if can use this to avoid resplining
        _cdfs = self._spl_cdf(zetas, self._x_interpolant)
        spls = [  # work through the rs
            splrep(_cdfs[i, :], self._theta_interpolant, s=0) for i in range(_cdfs.shape[0])
        ]
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

    def cdf(self, theta: ArrayLike, r: ArrayLike) -> NDArrayF:
        """Cumulative Distribution Function.

        Parameters
        ----------
        theta : array-like or Quantity-like
        r : array-like or Quantity-like

        Returns
        -------
        cdf : ndarray[float]
        """
        # TODO! make sure r, theta in right domain
        cdf = self._cdf(x_of_theta(u.Quantity(theta, u.rad)), zeta=zeta_of_r(r))
        return cdf

    def _ppf(
        self,
        q: ArrayLike,
        *,
        r: ArrayLike,
        **kw: Any,
    ) -> NDArrayF:
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
        ppf: NDArrayF = self._spl_ppf(zeta_of_r(r), q, grid=False)
        return ppf

    def _rvs(
        self,
        r: ArrayLike,
        *,
        random_state: Union[np.random.RandomState, np.random.Generator],
        size: Optional[int] = None,
    ) -> NDArrayF:
        """Random variate sampling.

        Parameters
        ----------
        r : float or (N,) array-like[float]
        size : int (optional, keyword-only)
        random_state : int or None (optional, keyword-only)

        Returns
        -------
        float or array-like[float]
            Shape 'size'.
        """
        # Use inverse cdf algorithm for RV generation.
        U = random_state.uniform(size=size)
        Y = self._ppf(U, r=r, grid=False)
        return Y

    def rvs(  # type: ignore
        self,
        r: ArrayLike,
        *,
        size: Optional[int] = None,
        random_state: RandomLike = None,
    ) -> NDArrayF:
        """Random variate sampling.

        Parameters
        ----------
        r : float or (N,) array-like[float]
        size : int or None (optional, keyword-only)
        random_state : int or None (optional, keyword-only)

        Returns
        -------
        float or array-like[float]
            Shape 'size'.
        """
        return super().rvs(r, size=size, random_state=random_state)
