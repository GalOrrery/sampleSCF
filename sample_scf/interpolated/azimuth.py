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
from typing import Any, Optional, Union, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from galpy.potential import SCFPotential
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator, splev, splrep

# LOCAL
from sample_scf._typing import NDArrayF, RandomLike
from sample_scf.base_univariate import phi_distribution_base
from sample_scf.representation import x_of_theta, zeta_of_r

__all__ = ["interpolated_phi_distribution"]


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
        (Sc, Ss) = kw.pop("Scs", (None, None))
        super().__init__(potential, **kw)  # allowed range of r

        self._phi_interpolant = np.linspace(0, 2 * np.pi, int(nintrp)) << u.rad
        self._ninterpolant = len(self._phi_interpolant)
        self._q_interpolant = qarr = np.linspace(0, 1, self._ninterpolant)

        # -------
        # build CDF

        zetas = zeta_of_r(radii)  # (R,)

        xs_unsorted = x_of_theta(thetas << u.rad)  # (T,)
        xsort = np.argsort(xs_unsorted)
        xs = xs_unsorted[xsort]
        thetas = thetas[xsort]

        lR, lT, _ = len(radii), len(thetas), len(phis)

        Phis = phis[None, None, :, None]  # ({R}, {T}, P, {L})

        # get Sc, Ss. We have defaults from above.
        if Sc is None:
            print("WTF?")
            Sc, Ss = self.calculate_Scs(radii, thetas, grid=True, warn=False)  # (R, T, L)
        elif (Sc.shape != Ss.shape) or (Sc.shape != (lR, lT, self._lmax + 1)):
            # check the user-passed values are the right shape
            raise ValueError(f"Sc, Ss must be shape ({lR}, {lT}, {self._lmax + 1})")

        # l = 0 : spherical symmetry
        term0 = Phis[..., 0] / (2 * np.pi)  # (1, 1, P)
        # l = 1+ : non-symmetry
        with warnings.catch_warnings():  # ignore true_divide RuntimeWarnings
            warnings.simplefilter("ignore")
            factor = 1 / Sc[:, :, :1]  # R0  (R, T, 1)  # can be inf

        ms = np.arange(1, self._lmax + 1)[None, None, None, :]  # ({R}, {T}, {P}, L)
        term1p = np.sum(
            (
                (Sc[:, :, None, 1:] * np.sin(ms * Phis))
                + (Ss[:, :, None, 1:] * (1 - np.cos(ms * Phis)))
            )
            / (2 * np.pi * ms),
            axis=-1,
        )

        cdfs = term0 + np.nan_to_num(factor * term1p)  # (R, T, P)
        # 'factor' can be inf and term1p 0 => inf * 0 = nan -> 0

        # interpolate
        # currently assumes a regular grid
        self._spl_cdf = RegularGridInterpolator((zetas, xs, phis), cdfs)

        # -------
        # ppf
        # might need cdf strategy to enforce "reality"
        # cdfstrategy = get_strategy(cdf_strategy)

        # start by supersampling
        Zetas, Xs, Phis = np.meshgrid(zetas, xs, self._phi_interpolant.value, indexing="ij")
        _cdfs = self._spl_cdf((Zetas.ravel(), Xs.ravel(), Phis.ravel())).reshape(
            lR,
            lT,
            len(self._phi_interpolant),
        )
        # build reverse spline
        ppfs = np.empty((lR, lT, self._ninterpolant), dtype=np.float64)
        for (i, j) in itertools.product(*map(range, ppfs.shape[:2])):
            try:
                spl = splrep(_cdfs[i, j, :], self._phi_interpolant, s=0)
            except ValueError:  # CDF is non-real
                import pdb
            
                pdb.set_trace()
                raise

            ppfs[i, j, :] = splev(qarr, spl, ext=0)
        # interpolate
        self._spl_ppf = RegularGridInterpolator(
            (zetas, xs, qarr),
            ppfs,
            bounds_error=False,
        )

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
        ppf: NDArrayF = self._spl_ppf(np.c_[zeta, x, q])
        return ppf

    def _rvs(
        self,
        r: NDArrayF,
        theta: NDArrayF,
        *args: Any,
        random_state: np.random.RandomState,
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
