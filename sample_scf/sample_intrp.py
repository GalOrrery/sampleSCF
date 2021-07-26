# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
import itertools
import typing as T
import warnings

# THIRD PARTY
import astropy.units as u
import numpy as np
import numpy.typing as npt
from galpy.potential import SCFPotential
from scipy.interpolate import (
    InterpolatedUnivariateSpline,
    RectBivariateSpline,
    RegularGridInterpolator,
    splev,
    splrep,
)
from scipy.stats import rv_continuous

# LOCAL
from ._typing import NDArray64, RandomLike
from .base import SCFSamplerBase, rv_continuous_modrvs
from .utils import (
    _phiRSms,
    _x_of_theta,
    difPls,
    phiRSms,
    r_of_zeta,
    thetaQls,
    x_of_theta,
    zeta_of_r,
)

__all__: T.List[str] = ["SCFSampler", "SCFRSampler", "SCFThetaSampler", "SCFPhiSampler"]


##############################################################################
# CODE
##############################################################################


class SCFSampler(SCFSamplerBase):
    r"""Interpolated SCF Sampler.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`
    rtgrid : array-like[float]
        The radial component of the interpolation grid.
    thetagrid : array-like[float]
        The inclination component of the interpolation grid.
        :math:`\theta \in [-\pi/2, \pi/2]`, from the South to North pole, so
        :math:`\theta = 0` is the equator.
    phigrid : array-like[float]
        The azimuthal component of the interpolation grid.
        :math:`phi \in [0, 2\pi)`.

    **kw:
        passed to :class:`~sample_scf.sample_interp.SCFRSampler`,
        :class:`~sample_scf.sample_interp.SCFThetaSampler`,
        :class:`~sample_scf.sample_interp.SCFPhiSampler`

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

        >>> sampler.cdf(10, np.pi/3, np.pi)
        array([[0.82644628, 0.9330127 , 0.5       ]])

    And draw samples

        >>> sampler.rvs(size=5, random_state=3)
        <PhysicsSphericalRepresentation (phi, theta, r) in (rad, rad, )
            [(3.46076529, 1.46902493,  2.8783381 ),
             (4.44942399, 1.141429  ,  5.30975319),
             (1.82780838, 2.0022487 ,  1.17087346),
             (3.2096245 , 1.54913942,  2.50535341),
             (5.61055118, 0.66665592, 17.16817722)]>

    """

    def __init__(
        self,
        pot: SCFPotential,
        rgrid: NDArray64,
        thetagrid: NDArray64,
        phigrid: NDArray64,
        **kw: T.Any,
    ) -> None:
        # compute the r-dependent coefficient matrix $\tilde{\rho}$
        nmax, lmax = pot._Acos.shape[:2]
        rhoTilde = np.array(
            [pot._rhoTilde(r, N=nmax, L=lmax) for r in rgrid],
        )  # (R, N, L)

        # ----------
        # theta Qls
        # radial sums over $\cos$ portion of the density function
        # the $\sin$ part disappears in the integral.
        Qls = np.sum(pot._Acos[None, :, :, 0] * rhoTilde, axis=1)  # ({R}, L)

        # ----------
        # phi Rm, Sm
        # radial and inclination sums

        Rm, Sm = _phiRSms(
            rhoTilde,
            Acos=pot._Acos,
            Asin=pot._Asin,
            r=rgrid,
            theta=thetagrid,
        )

        # ----------
        # make samplers

        self._rsampler = SCFRSampler(pot, rgrid, **kw)
        self._thetasampler = SCFThetaSampler(pot, rgrid, thetagrid, Qls=Qls, **kw)
        self._phisampler = SCFPhiSampler(pot, rgrid, thetagrid, phigrid, RSms=(Rm, Sm), **kw)

    # /def


# /class

# -------------------------------------------------------------------
# radial sampler


class SCFRSampler(rv_continuous_modrvs):
    """Sample radial coordinate from an SCF potential.

    The potential must have a convergent mass function.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential` or ndarray
        The mass enclosed in a spherical volume of radius(ii) 'rgrid', or the
        a potential that can be used to calculate the enclosed mass.
    rgrid : ndarray
    **kw
        not used
    """

    def __init__(self, pot: SCFPotential, rgrid: NDArray64, **kw: T.Any) -> None:
        super().__init__(a=0, b=np.inf)  # allowed range of r

        if isinstance(pot, np.ndarray):
            mgrid = pot
        elif isinstance(pot, SCFPotential):  # todo! generalize over potential
            mgrid = np.array([pot._mass(x) for x in rgrid])  # :(
            # manual fixes for endpoints
            ind = np.where(np.isnan(mgrid))[0]
            mgrid[ind[rgrid[ind] == 0]] = 0
            mgrid[ind[rgrid[ind] == np.inf]] = 1
        else:
            raise TypeError

        # work in zeta, not r, since it is more numerically stable
        zeta = zeta_of_r(rgrid)
        # make splines for fast calculation
        self._spl_cdf = InterpolatedUnivariateSpline(
            zeta,
            mgrid,
            ext="raise",
            bbox=[-1, 1],
        )
        self._spl_ppf = InterpolatedUnivariateSpline(
            mgrid,
            zeta,
            ext="raise",
            bbox=[0, 1],
        )

        # TODO! make sure
        # # store endpoint values to ensure CDF normalized to [0, 1]
        # self._mi = self._spl_cdf(min(zeta))
        # self._mf = self._spl_cdf(max(zeta))

    # /def

    def _cdf(self, r: npt.ArrayLike, *args: T.Any, **kw: T.Any) -> NDArray64:
        cdf: NDArray64 = self._spl_cdf(zeta_of_r(r))
        # (self._scfmass(zeta) - self._mi) / (self._mf - self._mi)
        # TODO! is this normalization even necessary?
        return cdf

    # /def

    def _ppf(self, q: npt.ArrayLike, *args: T.Any, **kw: T.Any) -> NDArray64:
        return r_of_zeta(self._spl_ppf(q))

    # /def


# /class

# -------------------------------------------------------------------
# inclination sampler


class SCFThetaSampler(rv_continuous_modrvs):
    """
    Sample inclination coordinate from an SCF potential.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`
    rgrid, tgrid : ndarray

    """

    def __init__(
        self,
        pot: SCFPotential,
        rgrid: NDArray64,
        tgrid: NDArray64,
        intrp_step: float = 0.01,
        **kw: T.Any,
    ) -> None:
        super().__init__(a=-np.pi / 2, b=np.pi / 2)  # allowed range of theta

        self._theta_interpolant = np.arange(-np.pi / 2, np.pi / 2, intrp_step)
        self._x_interpolant = _x_of_theta(self._theta_interpolant)
        self._q_interpolant = np.linspace(0, 1, len(self._theta_interpolant))

        # -------
        # parse from potential

        self._pot = pot
        self._nmax, self._lmax = (nmax, lmax) = pot._Acos.shape[:2]
        self._lrange = np.arange(0, self._lmax + 1)

        # -------
        # build CDF in shells
        # TODO: clean up shape stuff

        zetas = zeta_of_r(rgrid)  # (R,)
        xs = _x_of_theta(tgrid)  # (T,)

        if "Qls" in kw:
            Qls: NDArray64 = kw["Qls"]
        else:
            Qls = thetaQls(pot, rgrid)
        # check it's the right shape (R, Lmax)
        if Qls.shape != (len(rgrid), lmax):
            raise ValueError(f"Qls must be shape ({len(rgrid)}, {lmax})")

        # l = 0 : spherical symmetry
        term0 = T.cast(npt.NDArray, 0.5 * (xs + 1))  # (T,)
        # l = 1+ : non-symmetry
        factor = 1.0 / (2.0 * Qls[:, 0])  # (R,)
        term1p = np.sum(
            (Qls[None, :, 1:] * difPls(xs, lmax - 1).T[:, None, :]).T,
            axis=0,
        )

        cdfs = term0[None, :] + np.nan_to_num(factor[:, None] * term1p)  # (R, T)

        # -------
        # interpolate
        # currently assumes a regular grid

        self._spl_cdf = RectBivariateSpline(
            zetas,
            xs,
            cdfs,
            bbox=[-1, 1, -1, 1],  # [zetamin, zetamax, xmin, xmax]
            kx=kw.get("kx", 3),
            ky=kw.get("ky", 3),
            s=kw.get("s", 0),
        )

        # ppf, one per r
        # TODO! see if can use this to avoid resplining
        _cdfs = self._spl_cdf(zetas, self._x_interpolant)
        spls = [  # work through the rs
            splrep(_cdfs[i, :], self._theta_interpolant, s=0) for i in range(_cdfs.shape[0])
        ]
        ppfs = np.array([splev(self._q_interpolant, spl, ext=0) for spl in spls])
        self._spl_ppf = RectBivariateSpline(
            zetas,
            self._q_interpolant,
            ppfs,
            bbox=[-1, 1, 0, 1],  # [zetamin, zetamax, xmin, xmax]
            kx=kw.get("kx", 3),
            ky=kw.get("ky", 3),
            s=kw.get("s", 0),
        )

    # /def

    def _cdf(
        self,
        x: npt.ArrayLike,
        *args: T.Any,
        zeta: npt.ArrayLike,
        grid: bool = False,
    ) -> NDArray64:
        cdf: NDArray64 = self._spl_cdf(zeta, x, grid=grid)
        return cdf

    # /def

    def cdf(self, theta: npt.ArrayLike, r: npt.ArrayLike) -> NDArray64:
        """Cumulative Distribution Function.

        Parameters
        ----------
        theta : array-like or Quantity-like
        r : array-like or Quantity-like

        Returns
        -------
        cdf : ndarray[float]
        """
        # TODO! make sure r, theta in right domain
        cdf = self._cdf(x_of_theta(u.Quantity(theta, u.rad)), zeta=zeta_of_r(r))
        return cdf

    # /def

    def _ppf(
        self,
        q: npt.ArrayLike,
        *,
        r: npt.ArrayLike,
        grid: bool = False,
        **kw: T.Any,
    ) -> NDArray64:
        """Percent-point function.

        Parameters
        ----------
        q : float or (N,) array-like[float]
        r : float or (N,) array-like[float]

        Returns
        -------
        float or (N,) array-like[float]
            Same shape as 'r', 'q'.
        """
        ppf: NDArray64 = self._spl_ppf(zeta_of_r(r), q, grid=grid)
        return ppf

    # /def

    def _rvs(
        self,
        r: npt.ArrayLike,
        *,
        random_state: T.Union[np.random.RandomState, np.random.Generator],
        size: T.Optional[int] = None,
    ) -> NDArray64:
        """Random variate sampling.

        Parameters
        ----------
        r : float or (N,) array-like[float]
        size : int (optional, keyword-only)
        random_state : int or None (optional, keyword-only)

        Returns
        -------
        float or array-like[float]
            Shape 'size'.
        """
        # Use inverse cdf algorithm for RV generation.
        U = random_state.uniform(size=size)
        Y = self._ppf(U, r=r, grid=False)
        return Y

    # /def

    def rvs(  # type: ignore
        self,
        r: npt.ArrayLike,
        *,
        size: T.Optional[int] = None,
        random_state: RandomLike = None,
    ) -> NDArray64:
        """Random variate sampling.

        Parameters
        ----------
        r : float or (N,) array-like[float]
        size : int or None (optional, keyword-only)
        random_state : int or None (optional, keyword-only)

        Returns
        -------
        float or array-like[float]
            Shape 'size'.
        """
        return super().rvs(r, size=size, random_state=random_state)

    # /def


# -------------------------------------------------------------------
# Azimuth sampler


class SCFPhiSampler(rv_continuous_modrvs):
    """SCF phi sampler.

    .. todo::

        Make sure that stuff actually goes from 0 to 1.

    Parameters
    ----------
    pot : `galpy.potential.SCFPotential`
    rgrid : ndarray[float]
    tgrid : ndarray[float]
    pgrid : ndarray[float]
    intrp_step : float, optional
    **kw
        Not used
    """

    def __init__(
        self,
        pot: SCFPotential,
        rgrid: NDArray64,
        tgrid: NDArray64,
        pgrid: NDArray64,
        intrp_step: float = 0.01,
        **kw: T.Any,
    ) -> None:
        super().__init__(a=0, b=2 * np.pi)  # allowed range of r

        self._phi_interpolant = np.arange(0, 2 * np.pi, intrp_step)
        self._ninterpolant = len(self._phi_interpolant)
        self._q_interpolant = qarr = np.linspace(0, 1, self._ninterpolant)

        # -------
        # parse from potential

        self._pot = pot
        self._nmax, self._lmax = (nmax, lmax) = pot._Acos.shape[:2]

        # -------
        # build CDF

        zetas = zeta_of_r(rgrid)  # (R,)
        xs = _x_of_theta(tgrid)  # (T,)

        lR, lT, _ = len(rgrid), len(tgrid), len(pgrid)

        Phis = pgrid[None, None, :, None]  # ({R}, {T}, P, {L})

        if "RSms" in kw:
            (Rm, Sm) = kw["RSms"]
        else:
            (Rm, Sm) = phiRSms(pot, rgrid, tgrid)  # (R, T, L)
        # check it's the right shape
        if (Rm.shape != Sm.shape) or (Rm.shape != (lR, lT, lmax)):
            raise ValueError(f"Rm, Sm must be shape ({lR}, {lT}, {lmax})")

        # l = 0 : spherical symmetry
        term0 = pgrid[None, None, :] / (2 * np.pi)  # (1, 1, P)
        # l = 1+ : non-symmetry
        with warnings.catch_warnings():  # ignore true_divide RuntimeWarnings
            warnings.simplefilter("ignore")
            factor = 1 / Rm[:, :, :1]  # R0  (R, T, 1)  # can be inf

        ms = np.arange(1, lmax)[None, None, None, :]  # (1, 1, 1, L)
        term1p = np.sum(
            (Rm[:, :, None, 1:] * np.sin(ms * Phis) + Sm[:, :, None, 1:] * (1 - np.cos(ms * Phis)))
            / (2 * np.pi * ms),
            axis=-1,
        )

        cdfs = term0 + np.nan_to_num(factor * term1p)  # (R, T, P)
        # 'factor' can be inf and term1p 0 => inf * 0 = nan -> 0

        # interpolate
        # currently assumes a regular grid
        self._spl_cdf = RegularGridInterpolator((zetas, xs, pgrid), cdfs)

        # -------
        # ppf
        # start by supersampling
        Zetas, Xs, Phis = np.meshgrid(zetas, xs, self._phi_interpolant, indexing="ij")
        _cdfs = self._spl_cdf((Zetas.ravel(), Xs.ravel(), Phis.ravel())).reshape(
            lR,
            lT,
            len(self._phi_interpolant),
        )
        # build reverse spline
        ppfs = np.empty((lR, lT, self._ninterpolant), dtype=np.float64)
        for (i, j) in itertools.product(*map(range, ppfs.shape[:2])):
            spl = splrep(_cdfs[i, j, :], self._phi_interpolant, s=0)
            ppfs[i, j, :] = splev(qarr, spl, ext=0)
        # interpolate
        self._spl_ppf = RegularGridInterpolator(
            (zetas, xs, self._q_interpolant),
            ppfs,
            bounds_error=False,
        )

    # /def

    def _cdf(
        self,
        phi: npt.ArrayLike,
        *args: T.Any,
        zeta: npt.ArrayLike,
        x: npt.ArrayLike,
    ) -> NDArray64:
        cdf: NDArray64 = self._spl_cdf((zeta, x, phi))
        return cdf

    # /def

    def cdf(
        self,
        phi: npt.ArrayLike,
        r: npt.ArrayLike,
        theta: npt.ArrayLike,
    ) -> NDArray64:
        # TODO! make sure r, theta in right domain
        cdf = self._cdf(
            phi,
            zeta=zeta_of_r(r),
            x=x_of_theta(u.Quantity(theta, u.rad)),
        )
        return cdf

    # /def

    def _ppf(
        self,
        q: npt.ArrayLike,
        *args: T.Any,
        r: npt.ArrayLike,
        theta: NDArray64,
        grid: bool = False,
        **kw: T.Any,
    ) -> NDArray64:
        ppf: NDArray64 = self._spl_ppf((zeta_of_r(r), _x_of_theta(theta), q))
        return ppf

    # /def

    def _rvs(
        self,
        r: npt.ArrayLike,
        theta: NDArray64,
        *args: T.Any,
        random_state: np.random.RandomState,
        size: T.Optional[int] = None,
    ) -> NDArray64:
        # Use inverse cdf algorithm for RV generation.
        U = random_state.uniform(size=size)
        Y = self._ppf(U, *args, r=r, theta=theta)
        return Y

    # /def

    def rvs(  # type: ignore
        self,
        r: T.Union[float, npt.ArrayLike],
        theta: T.Union[float, npt.ArrayLike],
        *,
        size: T.Optional[int] = None,
        random_state: RandomLike = None,
    ) -> NDArray64:
        """Random variate sampler.

        Parameters
        ----------
        r, theta : array-like[float]
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
        return super().rvs(r, theta, size=size, random_state=random_state)

    # /def


# /class

##############################################################################
# END
