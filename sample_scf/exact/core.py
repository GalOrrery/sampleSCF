# -*- coding: utf-8 -*-

"""Exact sampling."""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
import abc
import typing as T
from typing import Any

# THIRD PARTY
from astropy.coordinates import PhysicsSphericalRepresentation
from galpy.potential import SCFPotential

# LOCAL
from .rvs_azimuth import phi_distribution
from .rvs_inclination import theta_distribution
from .rvs_radial import r_distribution
from sample_scf._typing import NDArrayF, RandomLike
from sample_scf.base import SCFSamplerBase

__all__ = ["SCFSampler"]


##############################################################################
# CODE
##############################################################################


class ExactSCFSampler(SCFSamplerBase):
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

    def __init__(self, potential: SCFPotential, **kw: Any) -> None:
        super().__init__(potential)

        # make samplers
        total_mass = kw.pop("total_mass", None)
        self._r_distribution = r_distribution(potential, total_mass=total_mass, **kw)
        self._theta_distribution = theta_distribution(potential, **kw)  # r=None
        self._phi_distribution = phi_distribution(potential, **kw)  # r=None, theta=None

    def rvs(
        self, *, size: Optional[int] = None, random_state: RandomLike = None
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