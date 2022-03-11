# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
import warnings
from typing import Any

# THIRD PARTY
import astropy.units as u
import numpy as np
from galpy.potential import SCFPotential

# LOCAL
from sample_scf._typing import NDArrayF
from sample_scf.base import SCFSamplerBase
from sample_scf.utils import _grid_phiRSms

from .rvs_azimuth import phi_distribution
from .rvs_inclination import theta_distribution
from .rvs_radial import r_distribution

__all__ = ["InterpolatedSCFSampler"]


##############################################################################
# CODE
##############################################################################


class InterpolatedSCFSampler(SCFSamplerBase):
    r"""Interpolated SCF Sampler.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`
    rgrid : array-like[float]
        The radial component of the interpolation grid.
    thetagrid : array-like[float]
        The inclination component of the interpolation grid.
        :math:`\theta \in [-\pi/2, \pi/2]`, from the South to North pole, so
        :math:`\theta = 0` is the equator.
    phigrid : array-like[float]
        The azimuthal component of the interpolation grid.
        :math:`phi \in [0, 2\pi)`.

    **kw:
        passed to :class:`~sample_scf.sample_interp.r_distribution`,
        :class:`~sample_scf.sample_interp.theta_distribution`,
        :class:`~sample_scf.sample_interp.phi_distribution`

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

        >>> rgrid = np.geomspace(1e-1, 1e3, 100)
        >>> thetagrid = np.linspace(-np.pi / 2, np.pi / 2, 30)
        >>> phigrid = np.linspace(0, 2 * np.pi, 30)

        >>> sampler = SCFSampler(pot, rgrid=rgrid, thetagrid=thetagrid, phigrid=phigrid)

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
        self,
        potential: SCFPotential,
        rgrid: NDArrayF,
        thetagrid: NDArrayF,
        phigrid: NDArrayF,
        **kw: Any,
    ) -> None:
        super().__init__(potential)

        # compute the r-dependent coefficient matrix $\tilde{\rho}$
        nmax, lmax = potential._Acos.shape[:2]
        rhoTilde = np.array([potential._rhoTilde(r, N=nmax, L=lmax) for r in rgrid])  # (R, N, L)
        # this matrix can have incorrect NaN values when rgrid=0, inf
        # and needs to be corrected
        ind = (rgrid == 0) | (rgrid == np.inf)
        rhoTilde[ind] = np.nan_to_num(
            rhoTilde[ind],
            copy=False,
            nan=0,
            posinf=np.inf,
            neginf=-np.inf,
        )

        # ----------
        # theta Qls
        # radial sums over $\cos$ portion of the density function
        # the $\sin$ part disappears in the integral.

        Qls = kw.pop("Qls", None)
        if Qls is None:
            Qls = np.sum(potential._Acos[None, :, :, 0] * rhoTilde, axis=1)  # ({R}, L)
            # this matrix can have incorrect NaN values when rgrid=0 because
            # rhoTilde will have +/- infs which when summed produce a NaN.
            # at r=0 this can be changed to 0.  # TODO! double confirm math
            ind0 = rgrid == 0
            if not np.sum(np.nan_to_num(rhoTilde[ind0, :, 0], posinf=1, neginf=-1)) == 0:
                # note: this if statement works even if ind0 is all False
                warnings.warn("Qls have non-cancelling infinities at r==0")
            else:
                Qls[ind0] = np.nan_to_num(Qls[ind0], copy=False)

        # ----------
        # phi Rm, Sm
        # radial and inclination sums

        RSms = kw.pop("RSms", None)
        if RSms is None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="(^invalid value)|(^overflow encountered)",
                )
                RSms = _grid_phiRSms(
                    rhoTilde,
                    Acos=potential._Acos,
                    Asin=potential._Asin,
                    r=rgrid,
                    theta=thetagrid,
                )

        # ----------
        # make samplers

        self._r_distribution = r_distribution(potential, rgrid, **kw)
        self._theta_distribution = theta_distribution(potential, rgrid, thetagrid, Qls=Qls, **kw)
        self._phi_distribution = phi_distribution(
            potential, rgrid, thetagrid, phigrid, RSms=RSms, **kw
        )
