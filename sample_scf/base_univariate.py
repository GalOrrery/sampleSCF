# -*- coding: utf-8 -*-

"""Base class for sampling from an SCF Potential."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import warnings
from abc import ABCMeta
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

# THIRD PARTY
import astropy.units as u
from astropy.units import Quantity
from galpy.potential import SCFPotential
from numpy import arange, array, atleast_1d, floating, inf, isinf, nan_to_num, pi, result_type, sum
from numpy import tril_indices, zeros
from numpy.typing import ArrayLike
from scipy._lib._util import check_random_state
from scipy.special import lpmv
from scipy.stats import rv_continuous

# LOCAL
from sample_scf._typing import NDArrayF, RandomGenerator, RandomLike
from sample_scf.representation import x_of_theta

if TYPE_CHECKING:
    # LOCAL
    from .base_multivariate import SCFSamplerBase

__all__: List[str] = []  # nothing is publicly scoped

##############################################################################
# CODE
##############################################################################


def _calculate_rhoTilde(distr: Union["rv_potential", SCFSamplerBase], /, r: Quantity) -> NDArrayF:
    """Compute the r-dependent coefficient matrix.

    Parameters
    ----------
    distr : `rv_potential` or `SCFSamplerBase`
    r : (R,) Quantity['length', float]

    returns
    -------
    (R, N, L) ndarray[float]
    """
    # compute the r-dependent coefficient matrix $\tilde{\rho}$
    nmaxp1, lmaxp1 = distr._potential._Acos.shape[:2]
    gprs = atleast_1d(r.to_value(u.kpc)) / distr._potential._ro
    rhoT = array([distr._potential._rhoTilde(r, N=nmaxp1, L=lmaxp1) for r in gprs])  # (R, N, L)
    # this matrix can have incorrect NaN values when r=0, inf
    # and needs to be corrected
    ind = (r == 0) | isinf(r)
    rhoT[ind] = nan_to_num(rhoT[ind], copy=False, posinf=inf, neginf=-inf)

    return rhoT


def _calculate_Qls(
    distr: Union["rv_potential", SCFSamplerBase],
    /,
    r: Quantity,
    rhoTilde: Optional[NDArrayF] = None,
) -> NDArrayF:
    r"""
    Compute the radial sums for inclination weighting factors.
    The weighting factors measure perturbations from spherical symmetry.
    The sin component disappears in the integral.

    :math:`Q_l(r) = \sum_{n=0}^{n_{\max}}A_{nl} \tilde{\rho}_{nl0}(r)`

    Parameters
    ----------
    r : (R,) Quantity['kpc', float]
        Radii. Scalar or 1D array.
    rhoTilde : (R, N, L) array[float]

    Returns
    -------
    Ql : (R, L) array[float]
    """
    Acos = distr.potential._Acos  # (N, L, M)
    rhoT = distr.calculate_rhoTilde(r) if rhoTilde is None else rhoTilde

    # inclination weighting factors
    Qls: NDArrayF = sum(Acos[None, :, :, 0] * rhoT, axis=1)  # (R, L)
    # this matrix can have incorrect NaN values when radii=0 because
    # rhoTilde will have +/- infs which when summed produce a NaN.
    # at r=0 this can be changed to 0.  # TODO! double confirm math
    ind0 = r == 0
    if not sum(nan_to_num(rhoT[ind0, :, 0], posinf=1, neginf=-1)) == 0:
        # note: this if statement works even if ind0 is all False
        warnings.warn("Qls have non-cancelling infinities at r==0")
    else:
        Qls[ind0] = nan_to_num(Qls[ind0], copy=False)  # TODO! Nan-> 0 or 1?

    return Qls


def _pnts_Scs(
    radii: NDArrayF,
    theta: NDArrayF,
    rhoTilde: NDArrayF,
    Acos: NDArrayF,
    Asin: NDArrayF,
) -> Tuple[NDArrayF, NDArrayF]:
    """Radial and inclination sums for azimuthal weighting factors.

    Parameters
    ----------
    radii : (R/T,) ndarray[float]
    rhoTilde: (R/T, N, L) ndarray[float]
    Acos, Asin : (N, L, L) ndarray[float]
    theta : (R/T,) ndarray[float]

    Returns
    -------
    Scm, Ssm : (R, T, L) ndarray[float]
        Azimuthal weighting factors.
        Cosine and Sine, respectively.

    Warns
    -----
    RuntimeWarning
        For invalid values (inf addition -> Nan).
        For overflow encountered related to inf and 0 division.
    """
    T: int = len(theta)
    L = M = Acos.shape[1] - 1
    # N = Acos.shape[0] - 1

    # The r-dependent coefficient matrix $\tilde{\rho}$
    RhoT = rhoTilde[..., None]  # (R/T, N, L, {M})

    # need r and theta to be arrays. Maintains units.
    x: NDArrayF = x_of_theta(theta)  # (T,)
    xs = x[:, None, None, None]  # (R/T, {N}, {L}, {M})

    # legendre polynomials
    ls, ms = tril_indices(L + 1)  # index set I_(L, M)

    lps = zeros((T, L + 1, M + 1))  # (R/T, L, M)
    lps[:, ls, ms] = lpmv(ms[None, :], ls[None, :], xs[:, 0, 0, 0])
    Plm = lps[:, None, :, :]  # (R/T, {N}, L, M)

    # full S matrices (R/T, N, L, M)  # TODO! where's Nlm
    # n-sum  # (R/T, N, L, M) -> (R, T, L, M)
    Sclm = sum(Acos[None, :, :, :] * RhoT * Plm, axis=-3)
    Sslm = sum(Asin[None, :, :, :] * RhoT * Plm, axis=-3)

    # fix adding +/- inf -> NaN. happens when r=0.
    idx = radii == 0
    Sclm[idx] = nan_to_num(Sclm[idx], posinf=inf, neginf=-inf)
    Sslm[idx] = nan_to_num(Sslm[idx], posinf=inf, neginf=-inf)

    # l'-sum  # FIXME! confirm correct som
    Scm = sum(Sclm, axis=-2)
    Ssm = sum(Sslm, axis=-2)

    return Scm, Ssm


# TODO! it's possible to make the r, theta grids, flatten, use _pnts_Scs,
# then reshape or asssign by index to the grid. Then the Sc calc is only
# in one place.
def _grid_Scs(
    radii: NDArrayF,
    thetas: NDArrayF,
    rhoTilde: NDArrayF,
    Acos: NDArrayF,
    Asin: NDArrayF,
) -> Tuple[NDArrayF, NDArrayF]:
    """Radial and inclination sums for azimuthal weighting factors.

    Parameters
    ----------
    radii : (R,) ndarray[float]
    rhoTilde: (R, N, L) ndarray[float]
    Acos, Asin : (N, L, L) ndarray[float]
    thetas : (T,) ndarray[float]

    Returns
    -------
    Scm, Ssm : (R, T, L) ndarray[float]
        Azimuthal weighting factors.
        Cosine and Sine, respectively.

    Warns
    -----
    RuntimeWarning
        For invalid values (inf addition -> Nan).
        For overflow encountered related to inf and 0 division.
    """
    T: int = len(thetas)
    L = M = Acos.shape[1] - 1

    # The r-dependent coefficient matrix $\tilde{\rho}$
    RhoT = rhoTilde[:, None, :, :, None]  # (R, {T}, N, L, {M})

    # need r and theta to be arrays. Maintains units.
    x: NDArrayF = x_of_theta(thetas << u.rad)  # (T,)
    xs = x[None, :, None, None, None]  # ({R}, T, {N}, {L}, {M})

    # legendre polynomials
    ls, ms = tril_indices(L + 1)  # index set I_(L, M)

    lps = zeros((T, L + 1, M + 1))  # (T, L, M)
    lps[:, ls, ms] = lpmv(ms[None, ...], ls[None, ...], xs[0, :, 0, 0, 0, None])
    Plm = lps[None, :, None, :, :]  # ({R}, T, {N}, L, M)

    # full S matrices (R, T, N, L, M)
    # n-sum  # (R, T, N, L, M) -> (R, T, L, M)
    Sclm = sum(Acos[None, None, :, :, :] * RhoT * Plm, axis=-3)
    Sslm = sum(Asin[None, None, :, :, :] * RhoT * Plm, axis=-3)

    # fix adding +/- inf -> NaN. happens when r=0.
    idx = radii == 0
    Sclm[idx, ...] = nan_to_num(Sclm[idx, ...], posinf=inf, neginf=-inf)
    Sslm[idx, ...] = nan_to_num(Sslm[idx, ...], posinf=inf, neginf=-inf)

    # l'-sum
    Scm = sum(Sclm, axis=-2)
    Ssm = sum(Sslm, axis=-2)

    return Scm, Ssm


def _calculate_Scs(
    distr,
    r: Quantity,
    theta: Quantity,
    *,
    grid: bool = True,
    warn: bool = True,
) -> Tuple[NDArrayF, NDArrayF]:
    r"""Radial and inclination sums for azimuthal weighting factors.

    Parameters
    ----------
    r : float or (R,) ndarray[float]
    theta : float or (T,) ndarray[float]

    grid : bool, optional keyword-only
    warn : bool, optional keyword-only

    Returns
    -------
    Rm, Sm : (R, T, L) ndarray[float]
        Azimuthal weighting factors.
    """
    # need r and theta to be float arrays.
    rdtype = result_type(float, result_type(r))
    radii: NDArrayF = atleast_1d(r).astype(rdtype)  # (R,)
    thetas: NDArrayF = atleast_1d(theta) << u.rad  # (T,)

    if not grid and len(thetas) != len(radii):
        raise ValueError

    # compute the r-dependent coefficient matrix $\tilde{\rho}$  # (R, N, L)
    rhoTilde = _calculate_rhoTilde(distr, radii)

    # pass to actual calculator, which takes the matrices and r, theta grids.
    with warnings.catch_warnings() if not warn else nullcontext():
        if not warn:
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="(^invalid value)|(^overflow encountered)",
            )
        func = _grid_Scs if grid else _pnts_Scs
        Sc, Ss = func(
            radii,
            thetas,
            rhoTilde=rhoTilde,
            Acos=distr.potential._Acos,
            Asin=distr.potential._Asin,
        )

    return Sc, Ss


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
    momtype : int, optional keyword-only
        The type of generic moment calculation to use: 0 for pdf, 1 (default)
        for ppf.
    a : float, optional keyword-only
        Lower bound of the support of the distribution, default is minus
        infinity.
    b : float, optional keyword-only
        Upper bound of the support of the distribution, default is plus
        infinity.
    xtol : float, optional keyword-only
        The tolerance for fixed point calculation for generic ppf.
    badvalue : float, optional keyword-only
        The value in a result arrays that indicates a value that for which
        some argument restriction is violated, default is `~numpy.nan`.
    name : str, optional keyword-only
        The name of the instance. This string is used to construct the default
        example for distributions.
    longname : str, optional keyword-only
        This string is used as part of the first line of the docstring returned
        when a subclass has no docstring of its own. Note: `longname` exists
        for backwards compatibility, do not use for new subclasses.
    shapes : str, optional keyword-only
        The shape of the distribution. For example ``"m, n"`` for a
        distribution that takes two integers as the two shape arguments for all
        its methods. If not provided, shape parameters will be inferred from
        the signature of the private methods, ``_pdf`` and ``_cdf`` of the
        instance.
    extradoc :  str, optional keyword-only, deprecated
        This string is used as the last part of the docstring returned when a
        subclass has no docstring of its own. Note: `extradoc` exists for
        backwards compatibility, do not use for new subclasses.
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional keyword-only

        If `seed` is None (or `numpy.random`), the `numpy.random.RandomState`
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
    _radial_scale_factor: Quantity

    def __init__(
        self,
        potential: SCFPotential,
        *,
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
    ) -> None:
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
            msg = f"potential must be <galpy.potential.SCFPotential>, not {type(potential)}"
            raise TypeError(msg)

        self._potential = potential
        self._nmax = potential._Acos.shape[0] - 1  # 0 inclusive
        self._lmax = potential._Acos.shape[1] - 1  # 0 inclusive
        self._radial_scale_factor = (potential._a * potential._ro) << u.kpc

    @property
    def potential(self) -> SCFPotential:
        """The potential from which to sample"""
        return self._potential

    @property
    def radial_scale_factor(self) -> Quantity:
        """Scale factor to convert dimensionful radii to a dimensionless form."""
        return self._radial_scale_factor

    @property
    def nmax(self) -> int:
        return self._nmax

    @property
    def lmax(self) -> int:
        return self._lmax

    def calculate_rhoTilde(self, radii: Quantity) -> NDArrayF:
        """Compute the r-dependent coefficient matrix.

        Parameters
        ----------
        radii : (R,) Quantity['length', float]

        returns
        -------
        (R, N, L) ndarray[float]
        """
        return _calculate_rhoTilde(self, r=radii)

    # ---------------------------------------------------------------

    def rvs(
        self,
        *args: Union[floating, ArrayLike],
        size: Optional[int] = None,
        random_state: RandomLike = None,
        **kwargs,
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
        **kwargs

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
            random_state_saved = None

        # go directly to `_rvs`
        vals: NDArrayF = self._rvs(*args, size=size, random_state=rndm, **kwargs)

        # copied from `scipy`
        # do not forget to restore the _random_state
        if random_state is not None:
            self._random_state = random_state_saved

        return vals.squeeze()  # TODO? should it squeeze?


# -------------------------------------------------------------------


class r_distribution_base(rv_potential):
    """Sample radial coordinate from an SCF potential.

    The potential must have a convergent mass function.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
    """


class theta_distribution_base(rv_potential):
    """Sample inclination coordinate from an SCF potential.

    The potential must have a convergent mass function.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
    """

    def __init__(self, potential: SCFPotential, **kwargs) -> None:
        kwargs["a"], kwargs["b"] = 1, -1  # allowed range of x
        super().__init__(potential, **kwargs)

        self._lrange = arange(0, self._lmax + 1)  # lmax inclusive

    def rvs(
        self,
        *args: Union[floating, ArrayLike],
        size: Optional[int] = None,
        random_state: RandomLike = None,
    ) -> NDArrayF:
        return super().rvs(
            *args,
            size=size,
            random_state=random_state,
            # return_thetas=return_thetas
        )

    # ---------------------------------------------------------------

    def calculate_Qls(self, r: Quantity, rhoTilde: Optional[NDArrayF] = None) -> NDArrayF:
        r"""
        Compute the radial sums for inclination weighting factors.
        The weighting factors measure perturbations from spherical symmetry.
        The sin component disappears in the integral.

        :math:`Q_l(r) = \sum_{n=0}^{n_{\max}}A_{nl} \tilde{\rho}_{nl0}(r)`

        Parameters
        ----------
        r : (R,) Quantity['kpc', float]
            Radii. Scalar or 1D array.
        rhoTilde : (R, N, L) array[float]

        Returns
        -------
        Ql : (R, L) array[float]
        """
        return _calculate_Qls(self, r=r, rhoTilde=rhoTilde)


class phi_distribution_base(rv_potential):
    """Sample inclination coordinate from an SCF potential.

    The potential must have a convergent mass function.

    Parameters
    ----------
    potential : `galpy.potential.SCFPotential`
    """

    def __init__(self, potential: SCFPotential, **kwargs: Any) -> None:
        kwargs["a"], kwargs["b"] = 0, 2 * pi
        super().__init__(potential, **kwargs)

        self._lrange = arange(0, self._lmax + 1)

    def calculate_Scs(
        self,
        r: Quantity,
        theta: Quantity,
        *,
        grid: bool = True,
        warn: bool = True,
    ) -> Tuple[NDArrayF, NDArrayF]:
        r"""Radial and inclination sums for azimuthal weighting factors.

        Parameters
        ----------
        r : float or (R,) ndarray[float]
        theta : float or (T,) ndarray[float]

        grid : bool, optional keyword-only
        warn : bool, optional keyword-only

        Returns
        -------
        Rm, Sm : (R, T, L) ndarray[float]
            Azimuthal weighting factors.
        """
        return _calculate_Scs(self, r=r, theta=theta, grid=grid, warn=warn)
