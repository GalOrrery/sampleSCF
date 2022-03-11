# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
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
from sample_scf.base import rv_potential
from sample_scf.cdf_strategy import default_cdf_strategy
from sample_scf.utils import phiRSms, x_of_theta, zeta_of_r

__all__ = ["phi_distribution"]


##############################################################################
# CODE
##############################################################################


class phi_distribution(rv_potential):
    """SCF phi sampler.

    .. todo::

        Make sure that stuff actually goes from 0 to 1.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
    rgrid : ndarray[float]
    tgrid : ndarray[float]
    pgrid : ndarray[float]
    intrp_step : float, optional
    **kw
        Passed to `scipy.stats.rv_continuous`
        "a", "b" are set to [0, 2 pi]
    """

    def __init__(
        self,
        potential: SCFPotential,
        rgrid: NDArrayF,
        tgrid: NDArrayF,
        pgrid: NDArrayF,
        intrp_step: float = 0.01,
        **kw: Any,
    ) -> None:
        kw["a"], kw["b"] = 0, 2 * np.pi
        (Rm, Sm) = kw.pop("RSms", (None, None))
        super().__init__(potential, **kw)  # allowed range of r

        self._phi_interpolant = np.arange(0, 2 * np.pi, intrp_step)
        self._ninterpolant = len(self._phi_interpolant)
        self._q_interpolant = qarr = np.linspace(0, 1, self._ninterpolant)

        # -------
        # build CDF

        zetas = zeta_of_r(rgrid)  # (R,)
        xs = x_of_theta(tgrid)  # (T,)

        lR, lT, _ = len(rgrid), len(tgrid), len(pgrid)

        Phis = pgrid[None, None, :, None]  # ({R}, {T}, P, {L})

        # get Rm, Sm. We have defaults from above.
        if Rm is None:
            print("WTF?")
            Rm, Sm = phiRSms(potential, rgrid, tgrid, grid=True, warn=False)  # (R, T, L)
        elif (Rm.shape != Sm.shape) or (Rm.shape != (lR, lT, self._lmax)):
            # check the user-passed values are the right shape
            raise ValueError(f"Rm, Sm must be shape ({lR}, {lT}, {self._lmax})")

        # l = 0 : spherical symmetry
        term0 = Phis[..., 0] / (2 * np.pi)  # (1, 1, P)
        # l = 1+ : non-symmetry
        with warnings.catch_warnings():  # ignore true_divide RuntimeWarnings
            warnings.simplefilter("ignore")
            factor = 1 / Rm[:, :, :1]  # R0  (R, T, 1)  # can be inf

        ms = np.arange(1, self._lmax)[None, None, None, :]  # ({R}, {T}, {P}, L)
        term1p = np.sum(
            (
                (Rm[:, :, None, 1:] * np.sin(ms * Phis))
                + (Sm[:, :, None, 1:] * (1 - np.cos(ms * Phis)))
            )
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
        # might need cdf strategy to enforce "reality"
        cdfstrategy = default_cdf_strategy.get()

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
            try:
                spl = splrep(_cdfs[i, j, :], self._phi_interpolant, s=0)
            except ValueError:  # CDF is non-real
                _cdf = cdfstrategy.apply(_cdfs[i, j, :], index=(i, j))
                spl = splrep(_cdf, self._phi_interpolant, s=0)

            ppfs[i, j, :] = splev(qarr, spl, ext=0)
        # interpolate
        self._spl_ppf = RegularGridInterpolator(
            (zetas, xs, qarr),
            ppfs,
            bounds_error=False,
        )

    def _cdf(
        self,
        phi: ArrayLike,
        *args: Any,
        zeta: ArrayLike,
        x: ArrayLike,
    ) -> NDArrayF:
        cdf: NDArrayF = self._spl_cdf((zeta, x, phi))
        return cdf

    def cdf(
        self,
        phi: ArrayLike,
        r: ArrayLike,
        theta: ArrayLike,
    ) -> NDArrayF:
        # TODO! make sure r, theta in right domain
        cdf = self._cdf(
            phi,
            zeta=zeta_of_r(r),
            x=x_of_theta(u.Quantity(theta, u.rad)),
        )
        return cdf

    def _ppf(
        self,
        q: ArrayLike,
        *args: Any,
        r: ArrayLike,
        theta: NDArrayF,
        **kw: Any,
    ) -> NDArrayF:
        ppf: NDArrayF = self._spl_ppf((zeta_of_r(r), x_of_theta(theta), q))
        return ppf

    def _rvs(
        self,
        r: ArrayLike,
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
        r: Union[np.floating, ArrayLike],
        theta: Union[np.floating, ArrayLike],
        *,
        size: Optional[int] = None,
        random_state: RandomLike = None,
    ) -> NDArrayF:
        """Random variate sampler.

        Parameters
        ----------
        r, theta : array-like[float]
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
        return super().rvs(r, theta, size=size, random_state=random_state)
