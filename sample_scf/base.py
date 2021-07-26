# -*- coding: utf-8 -*-

"""Base class for sampling from an SCF Potential.

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
from astropy.coordinates import PhysicsSphericalRepresentation
from scipy._lib._util import check_random_state
from scipy.stats import rv_continuous

# LOCAL
from ._typing import NDArray64, RandomLike

__all__: T.List[str] = []


##############################################################################
# CODE
##############################################################################


class rv_continuous_modrvs(rv_continuous):
    """
    Modified :class:`scipy.stats.rv_continuous` to use custom rvs methods.
    Made by stripping down the original scipy implementation.
    See :class:`scipy.stats.rv_continuous` for details.
    """

    def rvs(
        self,
        *args: T.Union[float, npt.ArrayLike],
        size: T.Optional[int] = None,
        random_state: RandomLike = None
    ) -> NDArray64:
        """Random variate sampler.

        Parameters
        ----------
        *args
        size : int or None (optional, keyword-only)
            Size of random variates to generate.
        random_state : int, `~numpy.random.Generator`, `~numpy.random.RandomState`, or None (optional, keyword-only)
            If seed is None (or numpy.random), the `numpy.random.RandomState`
            singleton is used. If seed is an int, a new RandomState instance is
            used, seeded with seed. If seed is already a Generator or
            RandomState instance then that instance is used.

        Returns
        -------
        ndarray[float]
            Shape 'size'.
        """
        # extra gymnastics needed for a custom random_state
        rndm: np.random.RandomState
        if random_state is not None:
            random_state_saved = self._random_state
            rndm = check_random_state(random_state)
        else:
            rndm = self._random_state

        vals: NDArray64 = self._rvs(*args, size=size, random_state=rndm)

        # do not forget to restore the _random_state
        if random_state is not None:
            self._random_state: np.random.RandomState = random_state_saved

        return vals.squeeze()

    # /def


# /class


# -------------------------------------------------------------------


class SCFSamplerBase:
    """Sample SCF in spherical coordinates.

    The coordinate system is:
    - r : [0, infinity)
    - theta : [-pi/2, pi/2]  (positive at the North pole)
    - phi : [0, 2pi)

    Parameters
    ----------
    pot : `galpy.potential.SCFPotential`
    """

    _rsampler: rv_continuous_modrvs
    _thetasampler: rv_continuous_modrvs
    _phisampler: rv_continuous_modrvs

    @property
    def rsampler(self) -> rv_continuous_modrvs:
        """Radial coordinate sampler."""
        return self._rsampler

    # /def

    @property
    def thetasampler(self) -> rv_continuous_modrvs:
        """Inclination coordinate sampler."""
        return self._thetasampler

    # /def

    @property
    def phisampler(self) -> rv_continuous_modrvs:
        """Azimuthal coordinate sampler."""
        return self._phisampler

    # /def

    def cdf(
        self,
        r: npt.ArrayLike,
        theta: npt.ArrayLike,
        phi: npt.ArrayLike,
    ) -> NDArray64:
        """
        Cumulative Distribution Functions in r, theta(r), phi(r, theta)

        Parameters
        ----------
        r : (N,) array-like ['kpc']
        theta : (N,) array-like ['angle']
        phi : (N,) array-like ['angle']

        Returns
        -------
        (N, 3) ndarray
        """
        R: NDArray64 = self.rsampler.cdf(r)
        Theta: NDArray64 = self.thetasampler.cdf(theta=theta, r=r)
        Phi: NDArray64 = self.phisampler.cdf(phi, r=r, theta=theta)

        RTP: NDArray64 = np.c_[R, Theta, Phi]
        return RTP

    # /def

    def rvs(
        self, *, size: T.Optional[int] = None, random_state: RandomLike = None
    ) -> PhysicsSphericalRepresentation:
        """Sample random variates.

        Parameters
        ----------
        size : int or None (optional, keyword-only)
            Defining number of random variates.
        random_state : int, `~numpy.random.Generator`, `~numpy.random.RandomState`, or None (optional, keyword-only)
            If seed is None (or numpy.random), the `numpy.random.RandomState`
            singleton is used. If seed is an int, a new RandomState instance is
            used, seeded with seed. If seed is already a Generator or
            RandomState instance then that instance is used.

        Returns
        -------
        `~astropy.coordinates.PhysicsSphericalRepresentation`
        """
        rs = self.rsampler.rvs(size=size, random_state=random_state)
        thetas = self.thetasampler.rvs(rs, size=size, random_state=random_state)
        phis = self.phisampler.rvs(rs, thetas, size=size, random_state=random_state)

        crd = PhysicsSphericalRepresentation(
            r=rs,
            theta=(np.pi / 2 - thetas) * u.rad,
            phi=phis * u.rad,
        )
        return crd

    # /def


# /class

##############################################################################
# END