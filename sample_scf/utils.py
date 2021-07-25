# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
import typing as T
import warnings

# THIRD PARTY
import astropy.units as u
import numpy as np
import numpy.typing as npt
from galpy.potential import SCFPotential
from numpy import (
    arange,
    array,
    asanyarray,
    atleast_1d,
    cos,
    divide,
    nan_to_num,
    pi,
    sqrt,
    stack,
    sum,
)
from scipy.special import legendre, lpmn

# LOCAL
from ._typing import NDArray64

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

    # TODO! organize & allow for lmax > 200.
    lrange = arange(0, 200 + 1)
    Pls = array([legendre(L) for L in lrange], dtype=object)
    # l=1+. l=0 is done separately
    _difPls = (Pls[2:] - Pls[:-2]) / (2 * lrange[1:-1] + 1)


def difPls(x: T.Union[float, NDArray64], lmax: int) -> NDArray64:
    # TODO? speed up
    return array([dPl(x) for dPl in _difPls[:lmax]])


##############################################################################
# CODE
##############################################################################


def zeta_of_r(r: T.Union[u.Quantity, NDArray64]) -> NDArray64:
    r""":math:`\zeta = \frac{r - 1}{r + 1}`

    Map the half-infinite domain [0, infinity) -> [-1, 1]

    Parameters
    ----------
    r : quantity-like ['length']

    Returns
    -------
    zeta : ndarray
        With shape (len(r),)
    """
    ra: NDArray64 = r.value if isinstance(r, u.Quantity) else r
    zeta: NDArray64 = nan_to_num(divide(ra - 1, ra + 1), nan=1)
    return zeta


# /def


def r_of_zeta(
    zeta: npt.ArrayLike, unit: T.Optional[u.UnitBase] = None
) -> T.Union[u.Quantity, NDArray64]:
    r""":math:`r = \frac{1 + \zeta}{1 - \zeta}`

    Map back to the half-infinite domain [0, infinity) <- [-1, 1]

    Parameters
    ----------
    zeta : array-like
    unit : `astropy.units.UnitBase` or None, optional

    Returns
    -------
    r: ndarray
    """
    z = array(zeta, subok=True)
    r = atleast_1d(divide(1 + z, 1 - z))
    r[r < 0] = 0

    rq: T.Union[u.Quantity, NDArray64] = r * (unit or 1)
    return rq


# /def


# -------------------------------------------------------------------


def _x_of_theta(theta: NDArray64) -> NDArray64:
    r""":math:`x = \cos{\theta}`.

    Parameters
    ----------
    theta : array-like ['radian']

    Returns
    -------
    x : float or array-like
    """
    x: NDArray64 = cos(pi / 2 - theta)
    return x


# /def


# @u.quantity_input(theta=u.radian)
def x_of_theta(theta: u.Quantity) -> NDArray64:
    r""":math:`x = \cos{\theta}`.

    Parameters
    ----------
    theta : quantity-like ['radian']

    Returns
    -------
    x : float or ndarray
    """
    x = _x_of_theta(theta.to_value(u.rad))
    return x


# /def

# -------------------------------------------------------------------


def thetaQls(pot: SCFPotential, r: T.Union[float, NDArray64]) -> NDArray64:
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
    # compute the r-dependent coefficient matrix $\tilde{\rho}$  # (R, N, L)
    nmax, lmax = pot._Acos.shape[:2]
    rs = atleast_1d(r)  # need r to be array.
    rhoTilde = array([pot._rhoTilde(r, N=nmax, L=lmax) for r in rs])

    # inclination weighting factors
    Qls = sum(pot._Acos[None, :, :, 0] * rhoTilde, axis=1)  # (R, N, L)

    # remove extra dimensions, e.g. scalar 'r'
    Ql: NDArray64 = Qls.squeeze()
    return Ql


# /def

# -------------------------------------------------------------------


def _phiRSms(
    rhoTilde: NDArray64,
    Acos: NDArray64,
    Asin: NDArray64,
    r: npt.ArrayLike,
    theta: npt.ArrayLike,
) -> T.Tuple[NDArray64, NDArray64]:
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
    """
    # need r and theta to be arrays. Maintains units.
    tgrid: NDArray64 = atleast_1d(theta)

    # transform to correct shape for vectorized computation
    x = _x_of_theta(asanyarray(tgrid))  # (T,)
    Xs = x[None, :, None, None, None]  # ({R}, X, {N}, {L}, {L})

    # compute the r-dependent coefficient matrix $\tilde{\rho}$
    nmax, lmax = Acos.shape[:2]
    RhoT = rhoTilde[:, None, :, :, None]  # (R, {X}, N, L, {L})

    # legendre polynomials
    with warnings.catch_warnings():  # there's a RuntimeWarning to ignore
        warnings.simplefilter("ignore")
        lps = lpmn_vec(lmax - 1, lmax - 1, x)[0]  # drop deriv

    PP = np.stack(lps, axis=0).astype(float)[None, :, None, :, :]
    # ({R}, X, {N}, L, L)

    # full R & S matrices
    RSnlm = RhoT * sqrt(1 - Xs ** 2) * PP  # (R, X, N, L, L)

    # n-sum  # (R, X, L, L)
    Rlm = sum(Acos[None, None, :, :, :] * RSnlm, axis=2)
    Slm = sum(Asin[None, None, :, :, :] * RSnlm, axis=2)

    # m-sum  # (R, X, L)
    sumidx = range(Rlm.shape[2])
    Rm = stack([sum(Rlm[:, :, m:, m], axis=2) for m in sumidx], axis=2)
    Sm = stack([sum(Slm[:, :, m:, m], axis=2) for m in sumidx], axis=2)

    return Rm, Sm


# /def


def phiRSms(
    pot: SCFPotential, r: npt.ArrayLike, theta: npt.ArrayLike
) -> T.Tuple[NDArray64, NDArray64]:
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
        Azimuthal weighting factors. Will have shape (len(r), len(theta), L),
        with :meth:`numpy.ndarray.squeeze` applied to eliminate extraneous
        dimensions, e.g. scalar 'r'
    """
    # need r and theta to be arrays. The extra dimensions will be 'squeeze'd.
    rgrid = atleast_1d(r)
    tgrid = atleast_1d(theta)

    # compute the r-dependent coefficient matrix $\tilde{\rho}$  # (R, N, L)
    nmax: int
    lmax: int
    nmax, lmax = pot._Acos.shape[:2]
    rhoTilde = array([pot._rhoTilde(r, N=nmax, L=lmax) for r in rgrid])

    # pass to actual calculator, which takes the matrices and r, theta grids.
    Rm, Sm = _phiRSms(rhoTilde, pot._Acos, pot._Asin, rgrid, tgrid)

    # remove extra dimensions from azimuthal factors
    return Rm.squeeze(), Sm.squeeze()


# /def

##############################################################################
# END
