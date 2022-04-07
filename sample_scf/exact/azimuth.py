# -*- coding: utf-8 -*-

"""Exact sampling."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Any, Optional, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from galpy.potential import SCFPotential
from numpy.typing import ArrayLike

# LOCAL
from sample_scf._typing import NDArrayF, RandomLike
from sample_scf.base_univariate import phi_distribution_base

__all__ = ["exact_phi_fixed_distribution", "exact_phi_distribution"]


##############################################################################
# CODE
##############################################################################


class exact_phi_distribution_base(phi_distribution_base):
    """Sample Azimuthal Coordinate.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
    **kw
        Passed to `scipy.stats.rv_continuous`
        "a", "b" are set to [0, 2 pi]

    """

    def __init__(self, potential: SCFPotential, **kw: Any) -> None:
        kw["a"], kw["b"] = 0, 2 * np.pi
        super().__init__(potential, **kw)
        self._lrange = np.arange(0, self._lmax + 1)

        # for compatibility
        self._Sc: Optional[NDArrayF] = None
        self._Ss: Optional[NDArrayF] = None

    def _cdf(self, phi: NDArrayF, *args: Any, **kw: Any) -> NDArrayF:
        r"""Cumulative Distribution Function.

        Parameters
        ----------
        phi : float or ndarray[float] ['radian']
            Azimuthal coordinate in radians, :math:`\in [0, 2\pi]`.
        *args, **kw
            Not used.

        Returns
        -------
        cdf : float or ndarray[float]
            Shape (len(r), len(theta), len(phi)).
            :meth:`numpy.ndarray.squeeze` applied so scalar inputs has scalar
            output.

        """
        Rm, Sm = kw.get("Scs", (self._Sc, self._Ss))  # (R/T, L)

        Phis: NDArrayF = np.atleast_1d(phi)[:, None]  # (P, {L})

        # l = 0 : spherical symmetry
        term0: NDArrayF = Phis[..., 0] / (2 * np.pi)  # (1, P)

        # l = 1+ : non-symmetry
        factor = 1 / Rm[:, 0]  # R0  (R/T,)  # can be inf
        ms = np.arange(1, self._lmax)[None, :]  # ({R/T/P}, L)
        term1p = np.sum(
            (Rm[:, 1:] * np.sin(ms * Phis) + Sm[:, 1:] * (1 - np.cos(ms * Phis)))
            / (2 * np.pi * ms),
            axis=-1,
        )

        cdf: NDArrayF = term0 + np.nan_to_num(factor * term1p)  # (R/T/P,)
        # 'factor' can be inf and term1p 0 => inf * 0 = nan -> 0

        return cdf

    def _ppf_to_solve(self, phi: float, q: float, *args: Any) -> NDArrayF:
        # changed from .cdf() to ._cdf() to use default 'r', 'theta'
        return self._cdf(*(phi,) + args) - q


class exact_phi_fixed_distribution(exact_phi_distribution_base):
    """Sample Azimuthal Coordinate at fixed r, theta.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
    r, theta : float or ndarray[float]

    """

    def __init__(self, potential: SCFPotential, r: NDArrayF, theta: NDArrayF, **kw: Any) -> None:
        super().__init__(potential, **kw)

        # assign fixed r, theta
        self._r, self._theta = r, theta
        # and can compute the associated assymetry measures
        self._Sc, self._Ss = self.calculate_Scs(r, theta, grid=False, warn=False)

    def cdf(self, phi: NDArrayF, *args: Any, **kw: Any) -> NDArrayF:
        r"""Cumulative Distribution Function.

        Parameters
        ----------
        phi : float or ndarray[float] ['radian']
            Azimuthal coordinate in radians, :math:`\in [0, 2\pi]`.
        *args
        **kw

        Returns
        -------
        cdf : float or ndarray[float]
            Shape (len(r), len(theta), len(phi)).
            :meth:`numpy.ndarray.squeeze` applied so scalar inputs has scalar
            output.
        """
        return self._cdf(phi, *args, **kw)


class exact_phi_distribution(exact_phi_distribution_base):
    def _cdf(
        self,
        phi: ArrayLike,
        *args: Any,
        r: Optional[float] = None,
        theta: Optional[float] = None,
    ) -> NDArrayF:
        r"""Cumulative Distribution Function.

        Parameters
        ----------
        phi : float or ndarray[float] ['radian']
            Azimuthal coordinate in radians, :math:`\in [0, 2\pi]`.
        *args
        r : float or ndarray[float], keyword-only
            Radial coordinate at which to evaluate the CDF. Not optional.
        theta : float or ndarray[float], keyword-only
            Inclination coordinate at which to evaluate the CDF. Not optional.
            In [-pi/2, pi/2].

        Returns
        -------
        cdf : float or ndarray[float]
            Shape (len(r), len(theta), len(phi)).
            :meth:`numpy.ndarray.squeeze` applied so scalar inputs has scalar
            output.

        Raises
        ------
        ValueError
            If 'r' or 'theta' are None.
        """
        Scs = self.calculate_Scs(cast(float, r), cast(float, theta), grid=False, warn=False)
        cdf: NDArrayF = super()._cdf(phi, *args, Scs=Scs)
        return cdf

    def cdf(self, phi: ArrayLike, *args: Any, r: float, theta: float) -> NDArrayF:
        r"""Cumulative Distribution Function.

        Parameters
        ----------
        phi : quantity-like or array-like ['radian']
            Azimuthal angular coordinate, :math:`\in [0, 2\pi]`. If doesn't
            have units, must be in radians.
        *args
        r : float or ndarray[float], keyword-only
            Radial coordinate at which to evaluate the CDF. Not optional.
        theta : quantity-like or array-like ['radian'], keyword-only
            Inclination coordinate at which to evaluate the CDF. Not optional.
            In [-pi/2, pi/2]. If doesn't have units, must be in radians.

        Returns
        -------
        cdf : float or ndarray[float]
            Shape (len(r), len(theta), len(phi)).
            :meth:`numpy.ndarray.squeeze` applied so scalar inputs has scalar
            output.
        """
        phi = u.Quantity(phi, u.rad).value
        cdf: NDArrayF = self._cdf(phi, *args, r=r, theta=u.Quantity(theta, u.rad).value)
        return cdf

    def rvs(  # type: ignore
        self,
        r: float,
        theta: float,
        *,
        size: Optional[int] = None,
        random_state: RandomLike = None,
    ) -> NDArrayF:
        """Random Variate Sample.

        Parameters
        ----------
        r : float
        theta : float
        size : int or None (optional, keyword-only)
        random_state : int or `numpy.random.RandomState` or None (optional, keyword-only)

        Returns
        -------
        vals : ndarray[float]

        """
        getattr(self._cdf, "__kwdefaults__", {})["r"] = r
        getattr(self._cdf, "__kwdefaults__", {})["theta"] = theta
        vals = super().rvs(size=size, random_state=random_state)
        getattr(self._cdf, "__kwdefaults__", {})["r"] = None
        getattr(self._cdf, "__kwdefaults__", {})["theta"] = None
        return vals
