# -*- coding: utf-8 -*-

"""Exact sampling."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Any, Optional

# THIRD PARTY
from astropy.coordinates import PhysicsSphericalRepresentation
from galpy.potential import SCFPotential

# LOCAL
from .azimuth import exact_phi_distribution
from .inclination import exact_theta_distribution
from sample_scf._typing import RandomLike
from sample_scf.base_multivariate import SCFSamplerBase
from sample_scf.exact.radial import exact_r_distribution

__all__ = ["ExactSCFSampler"]


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
        self._r_distribution = exact_r_distribution(potential, total_mass=total_mass, **kw)
        self._theta_distribution = exact_theta_distribution(potential, **kw)  # r=None
        self._phi_distribution = exact_phi_distribution(potential, **kw)  # r=None, theta=None

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
        return super().rvs(size=size, random_state=random_state)
