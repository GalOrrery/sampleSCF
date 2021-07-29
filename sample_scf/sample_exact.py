# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.units as u
import numpy as np
import numpy.typing as npt
from galpy.potential import SCFPotential

# LOCAL
from ._typing import NDArray64, RandomLike
from .base import SCFSamplerBase, rv_potential
from .utils import difPls, phiRSms, thetaQls, x_of_theta

__all__: T.List[str] = ["SCFSampler", "SCFRSampler", "SCFThetaSampler", "SCFPhiSampler"]


##############################################################################
# PARAMETERS

TSCFPhi = T.TypeVar("TSCFPhi", bound="SCFPhiSampler")
TSCFThetaSamplerBase = T.TypeVar("TSCFThetaSamplerBase", bound="SCFThetaSamplerBase")

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

    def __init__(self, pot: SCFPotential, **kw: T.Any) -> None:
        self._rsampler = SCFRSampler(pot)
        # not fixed r, theta. slower!
        self._thetasampler = SCFThetaSampler_of_r(pot, r=None)
        self._phisampler = SCFPhiSampler_of_rtheta(pot, r=None, theta=None)

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

    def __init__(self, potential: SCFPotential, **kw: T.Any) -> None:
        kw["a"], kw["b"] = 0, np.inf  # allowed range of r
        super().__init__(potential, **kw)

    # /def

    def _cdf(self, r: npt.ArrayLike, *args: T.Any, **kw: T.Any) -> NDArray64:
        mass: NDArray64 = self._pot._mass(r)
        # (self._scfmass(zeta) - self._mi) / (self._mf - self._mi)
        # TODO! is this normalization even necessary?
        return mass

    # /def


# /class

# -------------------------------------------------------------------
# inclination sampler


class SCFThetaSamplerBase(rv_potential):
    def __new__(
        cls: T.Type[TSCFThetaSamplerBase],
        potential: SCFPotential,
        r: T.Optional[float] = None,
        **kw: T.Any
    ) -> TSCFThetaSamplerBase:
        if cls is SCFThetaSampler and r is None:
            cls = SCFThetaSampler_of_r

        self: TSCFThetaSamplerBase = super().__new__(cls)
        return self

    # /def

    def __init__(self, potential: SCFPotential, **kw: T.Any) -> None:
        kw["a"], kw["b"] = -np.pi / 2, np.pi / 2  # allowed range of theta
        super().__init__(potential, **kw)
        self._lrange = np.arange(0, self._lmax + 1)  # lmax inclusive

    # /def

    # @functools.lru_cache()
    def Qls(self, r: float) -> NDArray64:
        r"""
        :math:`Q_l(r) = \sum_{n=0}^{n_{\max}}A_{nl} \tilde{\rho}_{nl0}(r)`

        Parameters
        ----------
        r : float ['kpc']

        Returns
        -------
        Ql : ndarray

        """
        Qls: NDArray64 = thetaQls(self._pot, r)
        return Qls

    # /def

    def _ppf_to_solve(self, x: float, q: float, *args: T.Any) -> NDArray64:
        ppf: NDArray64 = self._cdf(*(x,) + args) - q  # FIXME? x or theta
        return ppf

    # /def


# /class


class SCFThetaSampler(SCFThetaSamplerBase):
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
        self._Qlsatr = self.Qls(r)

    # /def

    def _cdf(self, theta: npt.ArrayLike, *args: T.Any) -> NDArray64:
        """
        Cumulative distribution function of the given RV.

        Parameters
        ----------
        theta : array-like ['radian']
        r : array-like [float] (optional, keyword-only)

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `theta`

        """
        x = x_of_theta(theta)
        Qlsatr = self._Qlsatr

        # l = 0
        term0 = (1.0 + x) / 2.0
        # l = 1+
        factor = 1.0 / (2.0 * Qlsatr[0])
        term1p = np.sum((Qlsatr[None, 1:] * difPls(x, self._lmax - 1).T).T, axis=0)

        cdf: NDArray64 = term0 + factor * term1p
        return cdf

    # /def


# /class


class SCFThetaSampler_of_r(SCFThetaSamplerBase):
    def _cdf(self, theta: NDArray64, *args: T.Any, r: float) -> NDArray64:
        x = x_of_theta(theta)
        Qlsatr = self.Qls(r)

        # l = 0
        term0 = (1.0 + x) / 2.0
        # l = 1+
        factor = 1.0 / (2.0 * Qlsatr[0])
        term1p = np.sum((Qlsatr[None, 1:] * difPls(x, self._lmax - 1).T).T, axis=0)

        cdf: NDArray64 = term0 + factor * term1p
        return cdf

    # /def

    def cdf(self, theta: npt.ArrayLike, *args: T.Any, r: float) -> NDArray64:
        return self._cdf(u.Quantity(theta, u.rad).value, *args, r=r)

    # /def

    def rvs(  # type: ignore
        self, r: npt.ArrayLike, *, size: T.Optional[int] = None, random_state: RandomLike = None
    ) -> NDArray64:
        # not thread safe!
        getattr(self._cdf, "__kwdefaults__", {})["r"] = r
        vals = super().rvs(size=size, random_state=random_state)
        getattr(self._cdf, "__kwdefaults__", {})["r"] = None
        return vals

    # /def


# /class


# -------------------------------------------------------------------
# azimuth sampler


class SCFPhiSamplerBase(rv_potential):
    def __new__(
        cls: T.Type[TSCFPhi],
        potential: SCFPotential,
        r: T.Optional[float] = None,
        theta: T.Optional[float] = None,
        **kw: T.Any
    ) -> TSCFPhi:
        if cls is SCFPhiSampler and (r is None or theta is None):
            cls = SCFPhiSampler_of_rtheta

        self: TSCFPhi = super().__new__(cls)
        return self

    # /def

    def __init__(self, potential: SCFPotential, **kw: T.Any) -> None:
        kw["a"], kw["b"] = 0, 2 * np.pi
        super().__init__(potential, **kw)
        self._lrange = np.arange(0, self._lmax + 1)

    # /def

    # @functools.lru_cache()
    def RSms(self, r: float, theta: float) -> T.Tuple[NDArray64, NDArray64]:
        return phiRSms(self._pot, r, theta)

    # /def


# /class


class SCFPhiSampler(SCFPhiSamplerBase):
    """

    Parameters
    ----------
    pot
    r, theta : float, optional

    """

    def __init__(self, potential: SCFPotential, r: float, theta: float, **kw: T.Any) -> None:
        super().__init__(potential, **kw)

        self._r, self._theta = r, theta
        self._Rm, self._Sm = self.RSms(float(r), float(theta))

    # /def

    def _cdf(self, phi: NDArray64, *args: T.Any, **kw: T.Any) -> NDArray64:
        Rm, Sm = self._Rm, self._Sm

        # l = 0
        term0: NDArray64 = phi / (2 * np.pi)

        # l = 1+
        factor = 1 / Rm[0]  # R0
        ms = np.arange(1, Rm.shape[1])
        term1p = np.sum(
            (Rm[1:] * np.sin(ms * phi) + Sm[1:] * (1 - np.cos(ms * phi))) / (2 * np.pi * ms),
        )

        cdf: NDArray64 = term0 + factor * term1p
        return cdf

    def cdf(self, phi: NDArray64, *args: T.Any, **kw: T.Any) -> NDArray64:
        return self._cdf(phi, *args, **kw)

    # /def


# /class


class SCFPhiSampler_of_rtheta(SCFPhiSamplerBase):
    _Rm: T.Optional[NDArray64]
    _Sm: T.Optional[NDArray64]

    def _cdf(self, phi: npt.ArrayLike, *args: T.Any, r: float, theta: float) -> NDArray64:
        self._Rm, self._Sm = self.RSms(float(r), float(theta))
        cdf: NDArray64 = super()._cdf(phi, *args)
        self._Rm, self._Sm = None, None
        return cdf

    # /def

    def cdf(self, phi: npt.ArrayLike, *args: T.Any, r: float, theta: float) -> NDArray64:
        phi = u.Quantity(phi, u.rad).value
        cdf: NDArray64 = self._cdf(phi, *args, r=r, theta=theta)
        return cdf

    # /def

    def rvs(  # type: ignore
        self,
        r: float,
        theta: float,
        *,
        size: T.Optional[int] = None,
        random_state: RandomLike = None
    ) -> NDArray64:
        getattr(self._cdf, "__kwdefaults__", {})["r"] = r
        getattr(self._cdf, "__kwdefaults__", {})["theta"] = theta
        vals = super().rvs(size=size, random_state=random_state)
        getattr(self._cdf, "__kwdefaults__", {})["r"] = None
        getattr(self._cdf, "__kwdefaults__", {})["theta"] = None
        return vals

    # /def

    def _ppf_to_solve(self, x: float, q: float, *args: T.Any) -> NDArray64:
        # changed from .cdf() to ._cdf() to use default 'r'
        r: float = getattr(self._cdf, "__kwdefaults__", {})["r"]
        theta: float = getattr(self._cdf, "__kwdefaults__", {})["theta"]
        return self._cdf(*(x,) + args, r=r, theta=theta) - q

    # /def


# /class

##############################################################################
# END
