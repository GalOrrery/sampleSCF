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
from numpy import nan_to_num, inf, sum, isinf, array
from galpy.potential import SCFPotential

# LOCAL
from .azimuth import interpolated_phi_distribution
from .inclination import interpolated_theta_distribution
from .radial import interpolated_r_distribution
from sample_scf._typing import NDArrayF
from sample_scf.base_multivariate import SCFSamplerBase

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
        # coefficients
        Acos: np.ndarray = potential._Acos
        Asin: np.ndarray = potential._Asin

        rsort = np.argsort(radii)
        radii = radii[rsort]

        # Compute the r-dependent coefficient matrix.
        rhoT = self.calculate_rhoTilde(radii)

        # Compute the radial sums for inclination weighting factors.
        Qls = kw.pop("Qls", None)
        if Qls is None:
            Qls = self.calculate_Qls(radii, rhoTilde=rhoT)

        # ----------
        # phi Rm, Sm
        # radial and inclination sums

        Scs = kw.pop("Scs", None)
        if Scs is None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="(^invalid value)|(^overflow encountered)",
                )
                Scs = interpolated_phi_distribution._grid_Scs(
                    radii, rhoT, Acos=Acos, Asin=Asin, theta=thetas
                )

        # ----------
        # make samplers

        self._r_distribution = interpolated_r_distribution(potential, radii, **kw)
        self._theta_distribution = interpolated_theta_distribution(
            potential, radii, thetas, Qls=Qls, **kw
        )
        self._phi_distribution = interpolated_phi_distribution(
            potential, radii, thetas, phis, Scs=Scs, **kw
        )

    @property
    def _Qls(self) -> NDArrayF:
        return self._theta_distribution._Qls
