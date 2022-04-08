# -*- coding: utf-8 -*-

"""Base class for sampling from an SCF Potential."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from abc import ABCMeta
from typing import Any, List, Optional, Tuple, Type, TypeVar

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseRepresentation, PhysicsSphericalRepresentation
from astropy.utils.misc import NumpyRNGContext
from galpy.potential import SCFPotential

# LOCAL
from .base_univariate import (
    phi_distribution_base,
    r_distribution_base,
    rv_potential,
    theta_distribution_base,
)
from sample_scf._typing import NDArrayF, RandomGenerator, RandomLike

__all__: List[str] = ["SCFSamplerBase"]

##############################################################################
# PARAMETERS

RT = TypeVar("RT", bound=BaseRepresentation)

##############################################################################
# CODE
##############################################################################


class SCFSamplerBase(metaclass=ABCMeta):
    """Sample SCF in spherical coordinates.

    The coordinate system is:
    - r : [0, infinity)
    - theta : [-pi/2, pi/2]  (positive at the North pole)
    - phi : [0, 2pi)

    Parameters
    ----------
    pot : `galpy.potential.SCFPotential`
    """

    _potential: SCFPotential
    _r_distribution: r_distribution_base
    _theta_distribution: theta_distribution_base
    _phi_distribution: phi_distribution_base

    def __init__(self, potential: SCFPotential, **kwargs: Any) -> None:
        if not isinstance(potential, SCFPotential):
            msg = f"potential must be <galpy.potential.SCFPotential>, not {type(potential)}"
            raise TypeError(msg)

        potential.turn_physical_on()
        self._potential = potential

        # child classes set up the samplers
        # _r_distribution
        # _theta_distribution
        # _phi_distribution

    # -----------------------------------------------------

    @property
    def potential(self) -> SCFPotential:
        """The SCF Potential instance."""
        return self._potential

    @property
    def r_distribution(self) -> r_distribution_base:
        """Radial coordinate sampler."""
        return self._r_distribution

    @property
    def theta_distribution(self) -> theta_distribution_base:
        """Inclination coordinate sampler."""
        return self._theta_distribution

    @property
    def phi_distribution(self) -> phi_distribution_base:
        """Azimuthal coordinate sampler."""
        return self._phi_distribution

    @property
    def radial_scale_factor(self) -> Quantity:
        """Scale factor to convert dimensionful radii to a dimensionless form."""
        return self._r_distribution._radial_scale_factor

    @property
    def nmax(self) -> int:
        return self._r_distribution._nmax

    @property
    def lmax(self) -> int:
        return self._r_distribution._lmax

    # -----------------------------------------------------

    def calculate_rhoTilde(self, radii: Quantity) -> NDArrayF:
        """

        Parameters
        ----------
        radii : (R,) Quantity['length', float]

        returns
        -------
        (R, N, L) ndarray[float]
        """
        return rv_potential.calculate_rhoTilde(self, radii)

    def calculate_Qls(self, r: Quantity, rhoTilde=None) -> NDArrayF:
        r"""
        Radial sums for inclination weighting factors.
        The weighting factors measure perturbations from spherical symmetry.

        :math:`Q_l(r) = \sum_{n=0}^{n_{\max}}A_{nl} \tilde{\rho}_{nl0}(r)`

        Parameters
        ----------
        r : (R,) Quantity['kpc', float]
            Radii. Scalar or 1D array.

        Returns
        -------
        Ql : (R, L) array[float]
        """
        return theta_distribution_base.calculate_Qls(self, r, rhoTilde=rhoTilde)

    def calculate_Scs(
        self,
        r: Quantity,
        theta: Quantity,
        *,
        grid: bool = True,
        warn: bool = True,
    ) -> Tuple[NDArrayF, NDArrayF]:
        r"""Radial and inclination sums for azimuthal weighting factors.

        Parameters
        ----------
        pot : :class:`galpy.potential.SCFPotential`
            Has coefficient matrices Acos and Asin with shape (N, L, L).
        r : float or (R,) ndarray[float]
        theta : float or (T,) ndarray[float]
        grid : bool, optional keyword-only
        warn : bool, optional keyword-only

        Returns
        -------
        Rm, Sm : (R, T, L) ndarray[float]
            Azimuthal weighting factors.
        """
        return phi_distribution_base.calculate_Scs(self, r, theta, grid=grid, warn=warn)

    # -----------------------------------------------------

    def cdf(self, r: Quantity, theta: Quantity, phi: Quantity) -> NDArrayF:
        """Cumulative distribution Functions in r, theta(r), phi(r, theta).

        Parameters
        ----------
        r : (N,) Quantity ['length']
        theta : (N,) Quantity ['angle']
        phi : (N,) Quantity ['angle']

        Returns
        -------
        (N, 3) ndarray
        """
        R: NDArrayF = self.r_distribution.cdf(r)
        Theta: NDArrayF = self.theta_distribution.cdf(theta, r=r)
        Phi: NDArrayF = self.phi_distribution.cdf(phi, r=r, theta=theta)

        c: NDArrayF = np.c_[R, Theta, Phi].squeeze()
        return c

    def rvs(
        self,
        *,
        size: Optional[int] = None,
        random_state: RandomLike = None,
        # vectorized: bool = True,
        representation_type: Type[RT] = PhysicsSphericalRepresentation,
    ) -> RT:
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
        rs: Quantity
        thetas: Quantity
        phis: Quantity

        rs = self.r_distribution.rvs(size=size, random_state=random_state)
        thetas = self.theta_distribution.rvs(rs, size=size, random_state=random_state)
        phis = self.phi_distribution.rvs(rs, thetas, size=size, random_state=random_state)

        crd: RT
        crd = PhysicsSphericalRepresentation(r=rs, theta=thetas, phi=phis)
        crd = crd.represent_as(representation_type)

        return crd

    def __repr__(self) -> str:
        s: str = super().__repr__()
        s += f"\n  r_distribution: {self.r_distribution!r}"
        s += f"\n  theta_distribution: {self.theta_distribution!r}"
        s += f"\n  phi_distribution: {self.phi_distribution!r}"

        return s
