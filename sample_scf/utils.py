# -*- coding: utf-8 -*-

"""Utility functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
import functools
import warnings
from contextlib import nullcontext
from typing import Optional, Tuple, Union

# THIRD PARTY
import astropy.units as u
import numpy as np
import numpy.typing as npt
from galpy.potential import SCFPotential
from numpy import arange, arccos, array, atleast_1d, cos, divide, nan_to_num, pi, sqrt, stack, sum
from numpy.typing import ArrayLike
from scipy.special import legendre, lpmn

# LOCAL
from ._typing import NDArrayF

__all__ = [
    "zeta_of_r",
    "r_of_zeta",
    "x_of_theta",
    "difPls",
    "thetaQls",
    "phiRSms",
]

##############################################################################
# PARAMETERS

lpmn_vec = np.vectorize(lpmn, otypes=(object, object))

# # pre-compute the difPls
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # TODO! allow for lmax > 200.
    lrange = arange(0, 200 + 1)
    Pls = array([legendre(L) for L in lrange], dtype=object)
    # l=1+. l=0 is done separately
    _difPls = (Pls[2:] - Pls[:-2]) / (2 * lrange[1:-1] + 1)


def difPls(x: Union[float, NDArrayF], lmax: int) -> NDArrayF:
    # TODO? speed up
    # TODO! the lmax = 1 case
    return array([dPl(x) for dPl in _difPls[:lmax]])


##############################################################################
# CODE
##############################################################################


def zeta_of_r(r: Union[u.Quantity, NDArrayF]) -> NDArrayF:
    r""":math:`\zeta = \frac{r - 1}{r + 1}`

    Map the half-infinite domain [0, infinity) -> [-1, 1]

    Parameters
    ----------
    r : quantity-like ['length']
        'r' must be in [0, infinity).

    Returns
    -------
    zeta : ndarray
        With shape (len(r),)
    """
    ra: NDArrayF = np.array(r, dtype=np.float64, copy=False)
    zeta: NDArrayF = nan_to_num(divide(ra - 1, ra + 1), nan=1)
    return zeta


def r_of_zeta(
    zeta: ArrayLike,
    unit: Optional[u.UnitBase] = None,
) -> Union[u.Quantity, NDArrayF]:
    r""":math:`r = \frac{1 + \zeta}{1 - \zeta}`

    Map back to the half-infinite domain [0, infinity) <- [-1, 1]

    Parameters
    ----------
    zeta : array-like
    unit : `astropy.units.UnitBase` or None, optional

    Returns
    -------
    r: ndarray[float] or Quantity
        If Quantity, has units of 'units'.
    """
    z = array(zeta, subok=True)
    r = atleast_1d(divide(1 + z, 1 - z))
    r[r < 0] = 0  # correct small errors

    rq: Union[NDArrayF, u.Quantity]
    rq = r << unit if unit is not None else r

    return rq


# -------------------------------------------------------------------


@functools.singledispatch
def x_of_theta(theta: ArrayLike) -> NDArrayF:
    r""":math:`x = \cos{\theta}`.

    Parameters
    ----------
    theta : array-like ['radian']
        :math:`\theta \in [-\pi/2, \pi/2]`

    Returns
    -------
    x : ndarray[float]
        :math:`x \in [-1, 1]`
    """
    x: NDArrayF = cos(pi / 2 - np.asanyarray(theta))
    return x


@x_of_theta.register
def _(theta: u.Quantity) -> NDArrayF:
    r""":math:`x = \cos{\theta}`.

    Parameters
    ----------
    theta : quantity-like ['radian']

    Returns
    -------
    x : float or ndarray
    """
    x: NDArrayF = cos(pi / 2 - theta.to_value(u.rad))
    return x


def theta_of_x(
    x: ArrayLike,
    unit: Optional[u.UnitBase] = None,
) -> Union[NDArrayF, u.Quantity]:
    r""":math:`\theta = \cos^{-1}{x}`.

    Parameters
    ----------
    x : array-like
    unit : unit-like['angular'] or None, optional

    Returns
    -------
    theta : float or ndarray
    """
    th: NDArrayF = pi / 2 - arccos(x)

    theta: Union[NDArrayF, u.Quantity]
    if unit is not None:
        theta = u.Quantity(th, u.rad).to(unit)
    else:
        theta = th

    return theta


# -------------------------------------------------------------------


def thetaQls(pot: SCFPotential, r: Union[float, NDArrayF]) -> NDArrayF:
    r"""
    Radial sums for inclination weighting factors.
    The weighting factors measure perturbations from spherical symmetry.

    :math:`Q_l(r) = \sum_{n=0}^{n_{\max}}A_{nl} \tilde{\rho}_{nl0}(r)`

    Parameters
    ----------
    pot
    r : float ['kpc']

    Returns
    -------
    Ql : ndarray

    """
    # with warnings.catch_warnings():  # TODO! diagnose RuntimeWarning
    # warnings.filterwarnings("ignore", category=RuntimeWarning, message="(^invalid value)")

    # compute the r-dependent coefficient matrix $\tilde{\rho}$  # (R, N, L)
    nmax, lmax = pot._Acos.shape[:2]
    rs = atleast_1d(r)  # need r to be array.
    rhoTilde = nan_to_num(
        array([pot._rhoTilde(r, N=nmax, L=lmax) for r in rs]),
        posinf=np.inf,
        neginf=-np.inf,
    )

    # inclination weighting factors
    Qls = nan_to_num(sum(pot._Acos[None, :, :, 0] * rhoTilde, axis=1), nan=1)  # (R, N, L)

    # remove extra dimensions, e.g. scalar 'r'
    Ql: NDArrayF = Qls.squeeze()

    return Ql


# -------------------------------------------------------------------


def _pnts_phiRSms(
    rhoTilde: NDArrayF,
    Acos: NDArrayF,
    Asin: NDArrayF,
    r: ArrayLike,
    theta: ArrayLike,
) -> Tuple[NDArrayF, NDArrayF]:
    """Radial and inclination sums for azimuthal weighting factors.

    Parameters
    ----------
    rhoTilde: (R, N, L) ndarray
    Acos, Asin : (N, L, L) ndarray
    r, theta : float or ndarray[float]
        With shapes (R,), (T,), respectively.

    Returns
    -------
    Rm, Sm : (R, T, L) ndarray
        Azimuthal weighting factors.

    Warns
    -----
    RuntimeWarning
        For invalid values (inf addition -> Nan).
        For overflow encountered related to inf and 0 division.
    """
    # need r and theta to be arrays. Maintains units.
    tgrid: NDArrayF = atleast_1d(theta)

    # transform to correct shape for vectorized computation
    x = x_of_theta(tgrid)  # (R/T,)
    Xs = x[:, None, None, None]  # (R/T, {N}, {L}, {L})

    # compute the r-dependent coefficient matrix $\tilde{\rho}$
    nmax, lmax = Acos.shape[:2]
    RhoT = rhoTilde[:, :, :, None]  # (R/T, N, L, {L})

    # legendre polynomials
    lps = lpmn_vec(lmax - 1, lmax - 1, x)[0]  # drop deriv

    PP = np.stack(lps, axis=0).astype(float)[:, None, :, :]  # (R/T, {N}, L, L)

    # full R & S matrices
    RSnlm = RhoT * sqrt(1 - Xs ** 2) * PP  # (R/T, N, L, L)

    # n-sum  # (R/T, N, L, L) -> (R/T, L, L)
    Rlm = sum(Acos[None, :, :, :] * RSnlm, axis=1)
    Slm = sum(Asin[None, :, :, :] * RSnlm, axis=1)
    # fix adding +/- inf -> NaN. happens when r=0.
    idx = np.all(np.isnan(Rlm[:, 0, :]), axis=-1)
    Rlm[idx, 0, :] = nan_to_num(Rlm[idx, 0, :])
    Slm[idx, 0, :] = nan_to_num(Slm[idx, 0, :])

    # m-sum  # (R/T, L)
    sumidx = range(Rlm.shape[1])
    Rm = stack([sum(Rlm[:, m:, m], axis=1) for m in sumidx], axis=1)
    Sm = stack([sum(Slm[:, m:, m], axis=1) for m in sumidx], axis=1)

    return Rm, Sm


def _grid_phiRSms(
    rhoTilde: NDArrayF,
    Acos: NDArrayF,
    Asin: NDArrayF,
    r: ArrayLike,
    theta: ArrayLike,
) -> Tuple[NDArrayF, NDArrayF]:
    """Radial and inclination sums for azimuthal weighting factors.

    Parameters
    ----------
    rhoTilde: (R, N, L) ndarray
    Acos, Asin : (N, L, L) ndarray
    r, theta : float or ndarray[float]
        With shapes (R,), (T,), respectively.

    Returns
    -------
    Rm, Sm : (R, T, L) ndarray
        Azimuthal weighting factors.

    Warns
    -----
    RuntimeWarning
        For invalid values (inf addition -> Nan).
        For overflow encountered related to inf and 0 division.
    """
    # need r and theta to be arrays. Maintains units.
    tgrid: NDArrayF = atleast_1d(theta)
    rgrid: NDArrayF = atleast_1d(r)

    # transform to correct shape for vectorized computation
    x = x_of_theta(tgrid)  # (T,)
    Xs = x[None, :, None, None, None]  # ({R}, X, {N}, {L}, {L})

    # format the r-dependent coefficient matrix $\tilde{\rho}$
    nmax, lmax = Acos.shape[:2]
    RhoT = rhoTilde[:, None, :, :, None]  # (R, {X}, N, L, {L})

    # legendre polynomials  # raises RuntimeWarnings
    lps = lpmn_vec(lmax - 1, lmax - 1, x)[0]  # drop deriv

    PP = np.stack(lps, axis=0).astype(float)[None, :, None, :, :]
    # ({R}, X, {N}, L, L)

    # full R & S matrices  # raises RuntimeWarnings
    RSnlm = RhoT * sqrt(1 - Xs ** 2) * PP  # (R, X, N, L, L)
    # for r=0, rhoT can be +/- inf. If added / multiplied by 0 this will be NaN
    # we can safely set this to 0 if rhoT's infinities cancel
    i0 = rgrid == 0
    if not np.sum(np.nan_to_num(RhoT[i0, 0, :, 0, 0], posinf=1, neginf=-1)) == 0:
        # note: this if statement works even if ind0 is all False
        warnings.warn("RhoT have non-cancelling infinities at r==0")
    else:
        RSnlm[i0, 0, :, 0, :] = np.nan_to_num(RSnlm[i0, 0, :, 0, :], copy=False)

    # n-sum  # (R, X, L, L)  # raises RuntimeWarnings
    Rlm = sum(Acos[None, None, :, :, :] * RSnlm, axis=2)
    Slm = sum(Asin[None, None, :, :, :] * RSnlm, axis=2)
    # fix adding +/- inf -> NaN. happens when r=0.
    # the check for cancelling infinities is done above.
    # the [X]=0 case is handled above as well.
    Rlm[i0, 1:, 0, :] = nan_to_num(Rlm[i0, 1:, 0, :], copy=False)
    Slm[i0, 1:, 0, :] = nan_to_num(Slm[i0, 1:, 0, :], copy=False)

    # m-sum  # (R, X, L)
    sumidx = range(Rlm.shape[2])
    Rm = stack([sum(Rlm[:, :, m:, m], axis=2) for m in sumidx], axis=2)
    Sm = stack([sum(Slm[:, :, m:, m], axis=2) for m in sumidx], axis=2)

    return Rm, Sm


def phiRSms(
    pot: SCFPotential,
    r: ArrayLike,
    theta: ArrayLike,
    grid: bool = True,
    warn: bool = True,
) -> Tuple[NDArrayF, NDArrayF]:
    r"""Radial and inclination sums for azimuthal weighting factors.

    .. math::
        [R/S]_{l}^{m}(r,x)= \left(\sum_{n=0}^{n_{\max}} [A/B]_{nlm}
                                  \tilde{\rho}_{nlm}(r) \right) r \sqrt{1-x^2}
                                  P_{l}^{m}(x)

        [R/S]^{m}(r, x) = \sum_{l=m}^{l_{\max}} [R/S]_{l}^{m}(r,x)

    Parameters
    ----------
    pot : :class:`galpy.potential.SCFPotential`
        Has coefficient matrices Acos and Asin with shape (N, L, L).
    r : float or ndarray[float]
    theta : float or ndarray[float]

    Returns
    -------
    Rm, Sm : ndarray[float]
        Azimuthal weighting factors. Shape (len(r), len(theta), L).
    """
    # need r and theta to be arrays. The extra dimensions will be 'squeeze'd.
    rgrid: npt.NDArray = atleast_1d(r)
    tgrid: npt.NDArray = atleast_1d(theta)

    # compute the r-dependent coefficient matrix $\tilde{\rho}$  # (R, N, L)
    nmax: int
    lmax: int
    nmax, lmax = pot._Acos.shape[:2]
    rhoTilde = nan_to_num(
        array([pot._rhoTilde(r, N=nmax, L=lmax) for r in rgrid]),  # todo! vectorize
        nan=0,
        posinf=np.inf,
        neginf=-np.inf,
    )

    # pass to actual calculator, which takes the matrices and r, theta grids.
    with warnings.catch_warnings() if not warn else nullcontext():
        if not warn:
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="(^invalid value)|(^overflow encountered)",
            )
        if grid:
            Rm, Sm = _grid_phiRSms(rhoTilde, pot._Acos, pot._Asin, rgrid, tgrid)
        else:
            Rm, Sm = _pnts_phiRSms(rhoTilde, pot._Acos, pot._Asin, rgrid, tgrid)

    return Rm, Sm
