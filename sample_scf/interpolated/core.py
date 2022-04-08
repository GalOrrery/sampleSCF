# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import warnings
from typing import Any

# THIRD PARTY
import astropy.units as u
import numpy as np
from galpy.potential import SCFPotential
from numpy import array, inf, isinf, nan_to_num, sum

# LOCAL
from .azimuth import interpolated_phi_distribution
from .inclination import interpolated_theta_distribution
from .radial import interpolated_r_distribution
from sample_scf._typing import NDArrayF
from sample_scf.base_multivariate import SCFSamplerBase
from sample_scf.representation import x_of_theta

__all__ = ["InterpolatedSCFSampler"]

##############################################################################
# CODE
##############################################################################


class InterpolatedSCFSampler(SCFSamplerBase):
    r"""Interpolated SCF Sampler.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`
    radii : array-like[float]
        The radial component of the interpolation grid.
    thetas : array-like[float]
        The inclination component of the interpolation grid.
        :math:`\theta \in [-\pi/2, \pi/2]`, from the South to North pole, so
        :math:`\theta = 0` is the equator.
    phis : array-like[float]
        The azimuthal component of the interpolation grid.
        :math:`phi \in [0, 2\pi)`.

    **kw:
        passed to :class:`~sample_scf.interpolated.interpolated_r_distribution`,
        :class:`~sample_scf.interpolated.interpolated_theta_distribution`,
        :class:`~sample_scf.interpolated.interpolated_phi_distribution`

    Examples
    --------
    For all examples we assume the following imports

        >>> import numpy as np
        >>> from galpy import potential

    For the SCF Potential we will use the simple example of a Hernquist sphere.

        >>> Acos = np.zeros((20, 24, 24))
        >>> Acos[0, 0, 0] = 1  # Hernquist potential
        >>> pot = potential.SCFPotential(Acos=Acos)

    Now we make the sampler, specifying the grid from which the interpolation
    will be built.

        >>> radii = np.geomspace(1e-1, 1e3, 100)
        >>> thetas = np.linspace(-np.pi / 2, np.pi / 2, 30)
        >>> phis = np.linspace(0, 2 * np.pi, 30)

        >>> sampler = SCFSampler(pot, radii=radii, thetas=thetas, phis=phis)

    Now we can evaluate the CDF

        >>> sampler.cdf(10.0, np.pi/3, np.pi)
        array([0.82666461, 0.9330127 , 0.5       ])

    And draw samples

        >>> sampler.rvs(size=5, random_state=3)
        <PhysicsSphericalRepresentation (phi, theta, r) in (rad, rad, )
            [(3.46076529, 1.46902493,  2.90496213),
             (4.44942399, 1.141429  ,  5.33343759),
             (1.82780838, 2.0022487 ,  1.19407968),
             (3.2096245 , 1.54913942,  2.53149096),
             (5.61055118, 0.66665592, 17.0125581 )]>
    """

    def __init__(
        self, potential: SCFPotential, radii: Quantity, thetas: Quantity, phis: Quantity, **kw: Any
    ) -> None:
        super().__init__(potential, **kw)

        # -------------------
        # Radial

        # sampler
        self._r_distribution = interpolated_r_distribution(potential, radii, **kw)
        radii = self._radii  # sorted

        # compute the r-dependent coefficient matrix.
        rhoT = self.calculate_rhoTilde(radii)

        # -------------------
        # Thetas

        # sampler
        self._theta_distribution = interpolated_theta_distribution(
            potential, radii, thetas, rhoTilde=rhoT, **kw
        )
        thetas, xs = self._thetas, self._xs  # sorted

        # -------------------
        # Phis

        self._phi_distribution = interpolated_phi_distribution(
            potential, radii, thetas, phis, rhoTilde=rhoT, **kw
        )

    @property
    def _radii(self) -> Quantity:
        return self._r_distribution._radii

    @property
    def _zetas(self) -> Quantity:
        return self._r_distribution._zetas

    @property
    def _thetas(self) -> Quantity:
        return self._theta_distribution._thetas

    @property
    def _xs(self) -> Quantity:
        return self._theta_distribution._xs

    @property
    def _Qls(self) -> NDArrayF:
        return self._theta_distribution._Qls

    @property
    def _phis(self) -> Quantity:
        self._phi_distribution._phis

    @property
    def _Scms(self) -> NDArrayF:
        return self._phi_distribution._Scms

    @property
    def _Ssms(self) -> NDArrayF:
        return self._phi_distribution._Ssms
