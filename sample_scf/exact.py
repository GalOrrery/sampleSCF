# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
import abc
import typing as T

# THIRD PARTY
import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import PhysicsSphericalRepresentation
from galpy.potential import SCFPotential

# LOCAL
from ._typing import NDArray64, RandomLike
from .base import SCFSamplerBase, rv_potential
from .utils import difPls, phiRSms, theta_of_x, thetaQls, x_of_theta

__all__: T.List[str] = [
    "SCFSampler",
    "SCFRSampler",
    "SCFThetaFixedSampler",
    "SCFThetaSampler",
    "SCFPhiFixedSampler",
    "SCFPhiSampler",
]


##############################################################################
# CODE
##############################################################################


class SCFSampler(SCFSamplerBase):
    """SCF sampler in spherical coordinates.

    The coordinate system is:
    - r : [0, infinity)
    - theta : [-pi/2, pi/2]  (positive at the North pole)
    - phi : [0, 2pi)

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`
    **kw
        Not used.

    """

    def __init__(self, potential: SCFPotential, **kw: T.Any) -> None:
        total_mass = kw.pop("total_mass", None)
        # make samplers
        self._rsampler = SCFRSampler(potential, total_mass=total_mass, **kw)
        self._thetasampler = SCFThetaSampler(potential, **kw)  # r=None
        self._phisampler = SCFPhiSampler(potential, **kw)  # r=None, theta=None

    # /def

    def rvs(
        self, *, size: T.Optional[int] = None, random_state: RandomLike = None
    ) -> PhysicsSphericalRepresentation:
        """Sample random variates.

        Parameters
        ----------
        size : int or None (optional, keyword-only)
            Defining number of random variates.
        random_state : int, `~numpy.random.RandomState`, or None (optional, keyword-only)
            If seed is None (or numpy.random), the `numpy.random.RandomState`
            singleton is used. If seed is an int, a new RandomState instance is
            used, seeded with seed. If seed is already a Generator or
            RandomState instance then that instance is used.

        Returns
        -------
        `~astropy.coordinates.PhysicsSphericalRepresentation`
        """
        return super().rvs(size=size, random_state=random_state, vectorized=False)

    # /def


# /class


# -------------------------------------------------------------------
# radial sampler


class SCFRSampler(rv_potential):
    """Sample radial coordinate from an SCF potential.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`
        A potential that can be used to calculate the enclosed mass.
    **kw
        Not used.
    """

    def __init__(self, potential: SCFPotential, total_mass=None, **kw: T.Any) -> None:
        # make sampler
        kw["a"], kw["b"] = 0, np.inf  # allowed range of r
        super().__init__(potential, **kw)

        # normalization for total mass
        if total_mass is None:
            total_mass = potential._mass(np.inf)
        if np.isnan(total_mass):
            raise ValueError(
                "Total mass is NaN. Need to pass kwarg " "`total_mass` with a non-NaN value.",
            )
        self._mtot = total_mass
        # vectorize mass function, which is scalar
        self._vec_cdf = np.vectorize(self._potential._mass)

    # /def

    def _cdf(self, r: npt.ArrayLike, *args: T.Any, **kw: T.Any) -> NDArray64:
        """Cumulative Distribution Function.

        Parameters
        ----------
        r : array-like
        *args
        **kwargs

        Returns
        -------
        mass : array-like
            Shape matches 'r'.
        """
        mass: NDArray64 = np.atleast_1d(self._vec_cdf(r)) / self._mtot
        mass[r == 0] = 0
        mass[r == np.inf] = 1
        return mass.item() if mass.shape == (1,) else mass

    cdf = _cdf
    # /def


# /class

##############################################################################
# Inclination sampler


class SCFThetaSamplerBase(rv_potential):
    """Base class for sampling the inclination coordinate."""

    def __init__(self, potential: SCFPotential, **kw: T.Any) -> None:
        kw["a"], kw["b"] = -np.pi / 2, np.pi / 2  # allowed range of theta
        super().__init__(potential, **kw)
        self._lrange = np.arange(0, self._lmax + 1)  # lmax inclusive

    # /def

    @abc.abstractmethod
    def _cdf(self, x: NDArray64, Qls: NDArray64) -> NDArray64:
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

    # /def

    def _rvs(
        self,
        *args: T.Union[float, npt.ArrayLike],
        size: T.Optional[int] = None,
        random_state: RandomLike = None,
    ) -> NDArray64:
        xs = super()._rvs(*args, size=size, random_state=random_state)
        rvs = theta_of_x(xs)
        return rvs

    # /def

    def _ppf_to_solve(self, x: float, q: float, *args: T.Any) -> NDArray64:
        ppf: NDArray64 = self._cdf(*(x,) + args) - q
        return ppf

    # /def


# /class


class SCFThetaFixedSampler(SCFThetaSamplerBase):
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

    def __init__(self, potential: SCFPotential, r: float, **kw: T.Any) -> None:
        super().__init__(potential)

        # points at which CDF is defined
        self._r = r
        self._Qlsatr = thetaQls(self._potential, r)

    # /def

    def _cdf(self, x: npt.ArrayLike, *args: T.Any) -> NDArray64:
        cdf = super()._cdf(x, self._Qlsatr)
        return cdf

    # /def

    def cdf(self, theta: npt.ArrayLike) -> NDArray64:
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

    # /def


# /class


class SCFThetaSampler(SCFThetaSamplerBase):
    """
    Sample inclination coordinate from an SCF potential.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`

    """

    def _cdf(self, theta: NDArray64, *args: T.Any, r: T.Optional[float] = None) -> NDArray64:
        Qls = thetaQls(self._potential, T.cast(float, r))
        cdf = super()._cdf(theta, Qls)
        return cdf

    # /def

    def cdf(self, theta: npt.ArrayLike, *args: T.Any, r: float) -> NDArray64:
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

    # /def

    def rvs(
        self, r: npt.ArrayLike, *, size: T.Optional[int] = None, random_state: RandomLike = None
    ) -> NDArray64:
        # not thread safe!
        getattr(self._cdf, "__kwdefaults__", {})["r"] = r
        vals = super().rvs(size=size, random_state=random_state)
        getattr(self._cdf, "__kwdefaults__", {})["r"] = None
        return vals

    # /def


# /class


###############################################################################
# Azimuth sampler


class SCFPhiSamplerBase(rv_potential):
    """Sample Azimuthal Coordinate.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
    **kw
        Passed to `scipy.stats.rv_continuous`
        "a", "b" are set to [0, 2 pi]

    """

    def __init__(self, potential: SCFPotential, **kw: T.Any) -> None:
        kw["a"], kw["b"] = 0, 2 * np.pi
        super().__init__(potential, **kw)
        self._lrange = np.arange(0, self._lmax + 1)

        # for compatibility
        self._Rm: T.Optional[NDArray64] = None
        self._Sm: T.Optional[NDArray64] = None

    # /def

    def _cdf(self, phi: NDArray64, *args: T.Any, **kw: T.Any) -> NDArray64:
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
        Rm, Sm = kw.get("RSms", (self._Rm, self._Sm))  # (R/T, L)

        Phis: NDArray64 = np.atleast_1d(phi)[:, None]  # (P, {L})

        # l = 0 : spherical symmetry
        term0: NDArray64 = Phis[..., 0] / (2 * np.pi)  # (1, P)

        # l = 1+ : non-symmetry
        factor = 1 / Rm[:, 0]  # R0  (R/T,)  # can be inf
        ms = np.arange(1, self._lmax)[None, :]  # ({R/T/P}, L)
        term1p = np.sum(
            (Rm[:, 1:] * np.sin(ms * Phis) + Sm[:, 1:] * (1 - np.cos(ms * Phis)))
            / (2 * np.pi * ms),
            axis=-1,
        )

        cdf: NDArray64 = term0 + np.nan_to_num(factor * term1p)  # (R/T/P,)
        # 'factor' can be inf and term1p 0 => inf * 0 = nan -> 0

        return cdf

    # /def

    def _ppf_to_solve(self, phi: float, q: float, *args: T.Any) -> NDArray64:
        # changed from .cdf() to ._cdf() to use default 'r', 'theta'
        return self._cdf(*(phi,) + args) - q

    # /def


# /class


class SCFPhiFixedSampler(SCFPhiSamplerBase):
    """Sample Azimuthal Coordinate at fixed r, theta.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
    r, theta : float or ndarray[float]

    """

    def __init__(
        self, potential: SCFPotential, r: NDArray64, theta: NDArray64, **kw: T.Any
    ) -> None:
        super().__init__(potential, **kw)

        # assign fixed r, theta
        self._r, self._theta = r, theta
        # and can compute the associated assymetry measures
        self._Rm, self._Sm = phiRSms(potential, r, theta, grid=False)

    # /def

    def cdf(self, phi: NDArray64, *args: T.Any, **kw: T.Any) -> NDArray64:
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

    # /def


# /class


class SCFPhiSampler(SCFPhiSamplerBase):
    def _cdf(
        self,
        phi: npt.ArrayLike,
        *args: T.Any,
        r: T.Optional[float] = None,
        theta: T.Optional[float] = None,
    ) -> NDArray64:
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
        RSms = phiRSms(self._potential, T.cast(float, r), T.cast(float, theta), grid=False)
        cdf: NDArray64 = super()._cdf(phi, *args, RSms=RSms)
        return cdf

    # /def

    def cdf(self, phi: npt.ArrayLike, *args: T.Any, r: float, theta: float) -> NDArray64:
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
        cdf: NDArray64 = self._cdf(phi, *args, r=r, theta=u.Quantity(theta, u.rad).value)
        return cdf

    # /def

    def rvs(  # type: ignore
        self,
        r: float,
        theta: float,
        *,
        size: T.Optional[int] = None,
        random_state: RandomLike = None,
    ) -> NDArray64:
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

    # /def


# /class

##############################################################################
# END
