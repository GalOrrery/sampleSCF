# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import itertools
import warnings
from typing import Any, Optional, Tuple

# THIRD PARTY
import astropy.units as u
from astropy.units import Quantity
from galpy.potential import SCFPotential
from numpy import arange, argsort, column_stack, cos, empty, float64, inf, linspace, meshgrid
from numpy import nan_to_num, pi, random, sin, sum
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator, splev, splrep

# LOCAL
from .inclination import interpolated_theta_distribution
from .radial import interpolated_r_distribution
from sample_scf._typing import NDArrayF, RandomLike
from sample_scf.base_univariate import _grid_Scs, phi_distribution_base
from sample_scf.representation import x_of_theta, zeta_of_r

__all__ = ["interpolated_phi_distribution"]


##############################################################################
# PARAMETERS

_phi_filter = dict(category=RuntimeWarning, message="(^invalid value)|(^overflow encountered)")

##############################################################################
# CODE
##############################################################################


class interpolated_phi_distribution(phi_distribution_base):
    """SCF phi sampler.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
    radii : ndarray[float]
    thetas : ndarray[float]
    phis : ndarray[float]
    intrp_step : float, optional
    **kw
        Passed to `scipy.stats.rv_continuous`
        "a", "b" are set to [0, 2 pi]
    """

    def __init__(
        self,
        potential: SCFPotential,
        radii: Quantity,
        thetas: Quantity,
        phis: Quantity,
        nintrp: float = 1e3,
        **kw: Any,
    ) -> None:
        rhoTilde = kw.pop("rhoTilde", None)  # must be same sort order as
        super().__init__(potential, **kw)  # allowed range of r

        self._phi_interpolant = linspace(0, 2 * pi, int(nintrp)) << u.rad
        self._ninterpolant = len(self._phi_interpolant)
        self._q_interpolant = qarr = linspace(0, 1, self._ninterpolant)

        # -------
        # build CDF

        radii, zetas = interpolated_r_distribution.order_radii(self, radii)  # (R,)
        thetas, xs = interpolated_theta_distribution.order_thetas(thetas)  # (T,)
        phis = interpolated_phi_distribution.order_phis(phis)  # (P,)
        self._phis = phis

        lR, lT, _ = len(radii), len(thetas), len(phis)
        Phis = phis.to_value(u.rad)[None, None, :, None]  # ({R}, {T}, P, {L})

        # get Sc, Ss. We have defaults from above.
        if rhoTilde is None:
            rhoTilde = self.calculate_rhoTilde(radii)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", **_phi_filter)
            Sc, Ss = _grid_Scs(
                radii, thetas, rhoTilde=rhoTilde, Acos=potential._Acos, Asin=potential._Asin
            )  # (R, T, L)
        self._Scms, self._Ssms = Sc, Ss

        # l = 0 : spherical symmetry
        term0 = Phis[..., 0] / (2 * pi)  # (1, 1, P)
        # l = 1+ : non-symmetry
        with warnings.catch_warnings():  # ignore true_divide RuntimeWarnings
            warnings.simplefilter("ignore")
            factor = 1.0 / Sc[:, :, :1]  # R0  (R, T, 1)

        ms = arange(1, self._lmax + 1)[None, None, None, :]  # ({R}, {T}, {P}, L)
        term1p = sum(
            ((Sc[:, :, None, 1:] * sin(ms * Phis)) + (Ss[:, :, None, 1:] * (1 - cos(ms * Phis))))
            / (2 * pi * ms),
            axis=-1,
        )

        # cdfs = term0 + nan_to_num(factor * term1p)  # (R, T, P)
        cdfs = term0 + nan_to_num(factor * term1p, posinf=inf, neginf=-inf)  # (R, T, P)
        # 'factor' can be inf and term1p 0 => inf * 0 = nan -> 0

        # interpolate
        # currently assumes a regular grid
        self._spl_cdf = RegularGridInterpolator((zetas, xs, phis.to_value(u.rad)), cdfs)

        # -------
        # ppf
        # might need cdf strategy to enforce "reality"
        # cdfstrategy = get_strategy(cdf_strategy)

        # start by supersampling
        Zetas, Xs, Phis = meshgrid(zetas, xs, self._phi_interpolant.value, indexing="ij")
        _cdfs = self._spl_cdf((Zetas.ravel(), Xs.ravel(), Phis.ravel()))
        _cdfs.shape = (lR, lT, len(self._phi_interpolant))

        self._cdfs = _cdfs
        # return

        # build reverse spline
        # TODO! vectorize
        ppfs = empty((lR, lT, self._ninterpolant), dtype=float64)
        for (i, j) in itertools.product(*map(range, ppfs.shape[:2])):
            try:
                spl = splrep(_cdfs[i, j, :], self._phi_interpolant.value, s=0)
            except ValueError:  # CDF is non-real
                # STDLIB
                import pdb

                pdb.set_trace()
                raise

            ppfs[i, j, :] = splev(qarr, spl, ext=0)
        # interpolate
        self._spl_ppf = RegularGridInterpolator((zetas, xs, qarr), ppfs, bounds_error=False)

    @staticmethod
    def order_phis(phis: Quantity) -> Tuple[Quantity]:
        """Return ordered phis."""
        psort = argsort(phis)
        phis = phis[psort]
        return phis

    # ---------------------------------------------------------------

    def _cdf(self, phi: ArrayLike, *args: Any, zeta: ArrayLike, x: ArrayLike) -> NDArrayF:
        cdf: NDArrayF = self._spl_cdf((zeta, x, phi))
        return cdf

    def cdf(self, phi: Quantity, *, r: Quantity, theta: Quantity) -> NDArrayF:
        # TODO! make sure r, theta in right domain
        cdf = self._cdf(
            phi,
            zeta=zeta_of_r(r, self._radial_scale_factor),
            x=x_of_theta(theta << u.rad),
        )
        return cdf

    def _ppf(self, q: ArrayLike, *args: Any, r: ArrayLike, theta: NDArrayF, **kw: Any) -> NDArrayF:
        zeta = zeta_of_r(r, self._radial_scale_factor)
        x = x_of_theta(theta << u.rad)
        ppf: NDArrayF = self._spl_ppf(column_stack((zeta, x, q)))
        return ppf

    def _rvs(
        self,
        r: NDArrayF,
        theta: NDArrayF,
        *args: Any,
        random_state: random.RandomState,
        size: Optional[int] = None,
    ) -> NDArrayF:
        # Use inverse cdf algorithm for RV generation.
        U = random_state.uniform(size=size)
        Y = self._ppf(U, *args, r=r, theta=theta)
        return Y

    def rvs(  # type: ignore
        self,
        r: Quantity,
        theta: Quantity,
        *,
        size: Optional[int] = None,
        random_state: RandomLike = None,
    ) -> NDArrayF:
        """Random variate sampler.

        Parameters
        ----------
        r : Quantity['length', float]
        theta : Quantity['angle', float]
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
        return super().rvs(r, theta, size=size, random_state=random_state) << u.rad
