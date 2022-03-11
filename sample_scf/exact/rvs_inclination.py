# -*- coding: utf-8 -*-

"""Exact sampling of inclination coordinate."""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
import abc
from typing import Any, Optional, Union, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from galpy.potential import SCFPotential
from numpy.typing import ArrayLike

# LOCAL
from sample_scf._typing import NDArrayF, RandomLike
from sample_scf.base import rv_potential
from sample_scf.utils import difPls, theta_of_x, thetaQls, x_of_theta

__all__ = ["theta_fixed_distribution", "theta_distribution"]


##############################################################################
# CODE
##############################################################################


class theta_distribution_base(rv_potential):
    """Base class for sampling the inclination coordinate."""

    def __init__(self, potential: SCFPotential, **kw: Any) -> None:
        kw["a"], kw["b"] = -np.pi / 2, np.pi / 2  # allowed range of theta
        super().__init__(potential, **kw)
        self._lrange = np.arange(0, self._lmax + 1)  # lmax inclusive

    @abc.abstractmethod
    def _cdf(self, x: NDArrayF, Qls: NDArrayF) -> NDArrayF:
        xs = np.atleast_1d(x)
        Qls = np.atleast_2d(Qls)  # ({R}, L)

        # l = 0
        term0 = 0.5 * (xs + 1.0)  # (T,)
        # l = 1+ : non-symmetry
        factor = 1.0 / (2.0 * Qls[:, 0])  # (R,)
        term1p = np.sum(
            (Qls[:, 1:] * difPls(xs, self._lmax - 1).T).T,
            axis=0,
        )
        # difPls shape (L, T) -> (T, L)
        # term1p shape (R/T, L) -> (L, R/T) -> sum -> (R/T,)

        cdf = term0 + np.nan_to_num(factor * term1p)  # (R/T,)
        return cdf

    def _rvs(
        self,
        *args: Union[np.floating, ArrayLike],
        size: Optional[int] = None,
        random_state: RandomLike = None,
    ) -> NDArrayF:
        xs = super()._rvs(*args, size=size, random_state=random_state)
        rvs = theta_of_x(xs)
        return rvs

    def _ppf_to_solve(self, x: float, q: float, *args: Any) -> NDArrayF:
        ppf: NDArrayF = self._cdf(*(x,) + args) - q
        return ppf


class theta_fixed_distribution(theta_distribution_base):
    """
    Sample inclination coordinate from an SCF potential.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`
    r : float or None, optional
        If passed, these are the locations at which the theta CDF will be
        evaluated. If None (default), then the r coordinate must be given
        to the CDF and RVS functions.
    **kw:
        Not used.
    """

    def __init__(self, potential: SCFPotential, r: float, **kw: Any) -> None:
        super().__init__(potential)

        # points at which CDF is defined
        self._r = r
        self._Qlsatr = thetaQls(self._potential, r)

    def _cdf(self, x: ArrayLike, *args: Any) -> NDArrayF:
        cdf = super()._cdf(x, self._Qlsatr)
        return cdf

    def cdf(self, theta: ArrayLike) -> NDArrayF:
        """
        Cumulative distribution function of the given RV.

        Parameters
        ----------
        theta : quantity-like['angle']

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `theta`

        """
        return self._cdf(x_of_theta(u.Quantity(theta, u.rad).value))


class theta_distribution(theta_distribution_base):
    """
    Sample inclination coordinate from an SCF potential.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`

    """

    def _cdf(self, theta: NDArrayF, *args: Any, r: Optional[float] = None) -> NDArrayF:
        Qls = thetaQls(self._potential, cast(float, r))
        cdf = super()._cdf(theta, Qls)
        return cdf

    def cdf(self, theta: ArrayLike, *args: Any, r: float) -> NDArrayF:
        """
        Cumulative distribution function of the given RV.

        Parameters
        ----------
        theta : quantity-like['angle']
        *args
            Not used.
        r : array-like[float] (optional, keyword-only)

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `theta`

        """
        return self._cdf(x_of_theta(u.Quantity(theta, u.rad).value), *args, r=r)

    def rvs(
        self, r: ArrayLike, *, size: Optional[int] = None, random_state: RandomLike = None
    ) -> NDArrayF:
        # not thread safe!
        getattr(self._cdf, "__kwdefaults__", {})["r"] = r
        vals = super().rvs(size=size, random_state=random_state)
        getattr(self._cdf, "__kwdefaults__", {})["r"] = None
        return vals
