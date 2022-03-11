# -*- coding: utf-8 -*-

"""Base class for sampling from an SCF Potential."""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
from abc import ABCMeta
from typing import Optional, Tuple, Union

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import PhysicsSphericalRepresentation
from astropy.utils.misc import NumpyRNGContext
from galpy.potential import SCFPotential
from numpy.typing import ArrayLike
from scipy._lib._util import check_random_state
from scipy.stats import rv_continuous

# LOCAL
from sample_scf._typing import NDArrayF, RandomGenerator, RandomLike

__all__ = []


##############################################################################
# CODE
##############################################################################


class rv_potential(rv_continuous, metaclass=ABCMeta):
    """
    Modified :class:`scipy.stats.rv_continuous` to use custom ``rvs`` methods.
    Made by stripping down the original scipy implementation.
    See :class:`scipy.stats.rv_continuous` for details.

    Parameters
    ----------
    `rv_continuous` is a base class to construct specific distribution classes
    and instances for continuous random variables. It cannot be used
    directly as a distribution.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
        The potential from which to sample.
    momtype : int, optional
        The type of generic moment calculation to use: 0 for pdf, 1 (default)
        for ppf.
    a : float, optional
        Lower bound of the support of the distribution, default is minus
        infinity.
    b : float, optional
        Upper bound of the support of the distribution, default is plus
        infinity.
    xtol : float, optional
        The tolerance for fixed point calculation for generic ppf.
    badvalue : float, optional
        The value in a result arrays that indicates a value that for which
        some argument restriction is violated, default is np.nan.
    name : str, optional
        The name of the instance. This string is used to construct the default
        example for distributions.
    longname : str, optional
        This string is used as part of the first line of the docstring returned
        when a subclass has no docstring of its own. Note: `longname` exists
        for backwards compatibility, do not use for new subclasses.
    shapes : str, optional
        The shape of the distribution. For example ``"m, n"`` for a
        distribution that takes two integers as the two shape arguments for all
        its methods. If not provided, shape parameters will be inferred from
        the signature of the private methods, ``_pdf`` and ``_cdf`` of the
        instance.
    extradoc :  str, optional, deprecated
        This string is used as the last part of the docstring returned when a
        subclass has no docstring of its own. Note: `extradoc` exists for
        backwards compatibility, do not use for new subclasses.
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    """

    _random_state: RandomGenerator
    _potential: SCFPotential
    _nmax: int
    _lmax: int

    def __init__(
        self,
        potential: SCFPotential,
        momtype: int = 1,
        a: Optional[float] = None,
        b: Optional[float] = None,
        xtol: float = 1e-14,
        badvalue: Optional[float] = None,
        name: Optional[str] = None,
        longname: Optional[str] = None,
        shapes: Optional[Tuple[int, ...]] = None,
        extradoc: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            momtype=momtype,
            a=a,
            b=b,
            xtol=xtol,
            badvalue=badvalue,
            name=name,
            longname=longname,
            shapes=shapes,
            extradoc=extradoc,
            seed=seed,
        )

        if not isinstance(potential, SCFPotential):
            raise TypeError(
                f"potential must be <galpy.potential.SCFPotential>, not {type(potential)}",
            )
        self._potential = potential
        self._nmax, self._lmax = potential._Acos.shape[:2]

    @property
    def potential(self) -> SCFPotential:
        """The potential from which to sample"""
        return self._potential

    @property
    def nmax(self) -> int:
        return self._nmax

    @property
    def lmax(self) -> int:
        return self._lmax

    def rvs(
        self,
        *args: Union[np.floating, ArrayLike],
        size: Optional[int] = None,
        random_state: RandomLike = None,
    ) -> NDArrayF:
        """Random variate sampler.

        Parameters
        ----------
        *args
        size : int or None (optional, keyword-only)
            Size of random variates to generate.
        random_state : int, `~numpy.random.RandomState`, or None (optional, keyword-only)
            If seed is None (or numpy.random), the `numpy.random.RandomState`
            singleton is used. If seed is an int, a new RandomState instance is
            used, seeded with seed. If seed is already a Generator or
            RandomState instance then that instance is used.

        Returns
        -------
        ndarray[float]
            Shape 'size'.
        """
        # copied from `scipy`
        # extra gymnastics needed for a custom random_state
        rndm: RandomGenerator
        if random_state is not None:
            random_state_saved = self._random_state
            rndm = check_random_state(random_state)
        else:
            rndm = self._random_state

        # go directly to `_rvs`
        vals: NDArrayF = self._rvs(*args, size=size, random_state=rndm)

        # copied from `scipy`
        # do not forget to restore the _random_state
        if random_state is not None:
            self._random_state = random_state_saved

        return vals.squeeze()


# -------------------------------------------------------------------


class r_distribution_base(rv_potential):
    """Sample radial coordinate from an SCF potential.

    The potential must have a convergent mass function.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
    """

    pass


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
    _r_distribution: rv_potential
    _theta_distribution: rv_potential
    _phi_distribution: rv_potential

    def __init__(self, potential: SCFPotential):
        potential.turn_physical_on()
        self._potential = potential

        # child classes set up the samplers

    # -----------------------------------------------------

    @property
    def potential(self) -> SCFPotential:
        """The SCF Potential instance."""
        return self._potential

    @property
    def rsampler(self) -> rv_potential:
        """Radial coordinate sampler."""
        return self._r_distribution

    @property
    def thetasampler(self) -> rv_potential:
        """Inclination coordinate sampler."""
        return self._theta_distribution

    @property
    def phisampler(self) -> rv_potential:
        """Azimuthal coordinate sampler."""
        return self._phi_distribution

    # -----------------------------------------------------

    def cdf(
        self,
        r: ArrayLike,
        theta: ArrayLike,
        phi: ArrayLike,
    ) -> NDArrayF:
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
        # coordinates  # TODO! deprecate whan galpy can do ints
        r = np.asanyarray(r, dtype=float)
        theta = np.asanyarray(theta, dtype=float)
        phi = np.asanyarray(phi, dtype=float)

        R: NDArrayF = self.rsampler.cdf(r)
        Theta: NDArrayF = self.thetasampler.cdf(theta, r=r)
        Phi: NDArrayF = self.phisampler.cdf(phi, r=r, theta=theta)
        return np.c_[R, Theta, Phi].squeeze()

    def rvs(
        self,
        *,
        size: Optional[int] = None,
        random_state: RandomLike = None,
        vectorized=True,
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
        # TODO! fix that thetasampler is off by pi/2

        rs = self.rsampler.rvs(size=size, random_state=random_state)

        if vectorized:
            thetas = np.pi / 2 - self.thetasampler.rvs(rs, size=size, random_state=random_state)
            phis = self.phisampler.rvs(rs, thetas, size=size, random_state=random_state)

        else:  # TODO! speed up
            # sample from theta and phi. Note that each needs to be in a separate
            # NumpyRNGContext to ensure that the results match the vectorized
            # option, above.
            kw = dict(size=1, random_state=None)

            with NumpyRNGContext(random_state):
                rsd = np.atleast_1d(rs)
                thetas = np.pi / 2 - np.array([self.thetasampler.rvs(r, **kw) for r in rsd])

            with NumpyRNGContext(random_state):
                thd = np.atleast_1d(thetas)
                phis = np.array([self.phisampler.rvs(r, th, **kw) for r, th in zip(rsd, thd)])

        crd = PhysicsSphericalRepresentation(r=rs, theta=thetas << u.rad, phi=phis << u.rad)
        return crd
