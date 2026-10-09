# -*- coding: utf-8 -*-

"""Exact sampling of inclination coordinate."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from typing import Any, Optional, Union

# THIRD PARTY
import astropy.units as u
from astropy.units import Quantity
from galpy.potential import SCFPotential
from numpy import atleast_1d, atleast_2d, floating, nan_to_num, pad
from numpy.polynomial.legendre import legval
from numpy.typing import ArrayLike

# LOCAL
from sample_scf._typing import NDArrayF, RandomLike
from sample_scf.base_univariate import theta_distribution_base
from sample_scf.representation import theta_of_x, x_of_theta

__all__ = ["exact_theta_fixed_distribution", "exact_theta_distribution"]


##############################################################################
# CODE
##############################################################################


class exact_theta_distribution_base(theta_distribution_base):
    """Base class for sampling the inclination coordinate."""

    def _cdf(self, x: NDArrayF, Qls: NDArrayF) -> NDArrayF:
        """Cumulative Distribution Function.

        .. math::

            F_{\theta}(\theta; r) = \frac{1 + \\cos{\theta}}{2} +
                \frac{1}{2 Q_0(r)}\\sum_{\\ell=1}^{L_{\\max}}Q_{\\ell}(r)
                \frac{\\sin(\theta) P_{\\ell}^{1}(\\cos{\theta})}{\\ell(\\ell+1)}

            Where

            Q_{\\ell}(r) = \\sum_{n=0}^{N_{\\max}} N_{\\ell 0} A_{n\\ell 0}^{(\\cos)}
                \tilde{\rho}_{n\\ell}(r)

        Parameters
        ----------
        x : number or (T,) array[number]
            :math:`x = \\cos\theta`. Must be in the range [-1, 1]
        Qls : (R, L) array[float]
            Radially-dependent coefficients parameterizing the deviations from
            a uniform distribution on the inclination angle.

        Returns
        -------
        (R, T) array
        """
        xs = atleast_1d(x)  # (T,)
        Qls = atleast_2d(Qls)  # (R, L)

        # l = 0
        term0 = 0.5 * (1.0 - xs)  # (T,)
        # l = 1+ : non-symmetry
        factor = 1.0 / (2.0 * Qls[:, 0])  # (R,)

        wQls = Qls[:, 1:] / (2 * self._lrange[None, 1:] + 1)  # apply over (L,) dimension
        wQls_lp1 = pad(wQls, [[0, 0], [2, 0]])  # pad start of (L,) dimension

        sumPlp1 = legval(xs, wQls_lp1.T, tensor=True)  # (R, T)
        sumPlm1 = legval(xs, wQls.T, tensor=True)  # (R, T)

        cdf = term0 + nan_to_num((factor * (sumPlm1 - sumPlp1).T).T)  # (R, T)
        return cdf  # TODO! get rid of sf function

    #     @abc.abstractmethod
    #     def _cdf(self, x: NDArrayF, Qls: NDArrayF) -> NDArrayF:
    #         """Cumulative Distribution Function.
    #
    #         .. math::
    #
    #             F_{\theta}(\theta; r) = \frac{1 + \cos{\theta}}{2} +
    #                 \frac{1}{2 Q_0(r)}\sum_{\ell=1}^{L_{\max}}Q_{\ell}(r)
    #                 \frac{\sin(\theta) P_{\ell}^{1}(\cos{\theta})}{\ell(\ell+1)}
    #
    #             Where
    #
    #             Q_{\ell}(r) = \sum_{n=0}^{N_{\max}} N_{\ell 0} A_{n\ell 0}^{(\cos)}
    #                 \tilde{\rho}_{n\ell}(r)
    #
    #         Parameters
    #         ----------
    #         x : number or (T,) array[number]
    #             :math:`x = \cos\theta`. Must be in the range [-1, 1]
    #         Qls : (R, L) array[float]
    #             Radially-dependent coefficients parameterizing the deviations from
    #             a uniform distribution on the inclination angle.
    #
    #         Returns
    #         -------
    #         (R, T) array
    #         """
    #         sf = self._sf(x, Qls)
    #         return 1.0 - sf

    def _rvs(
        self,
        *args: Union[floating, ArrayLike],
        size: Optional[int] = None,
        random_state: RandomLike = None,
        # return_thetas: bool = True
    ) -> NDArrayF:
        xs = super()._rvs(*args, size=size, random_state=random_state)
        # ths = theta_of_x(xs) if return_thetas else xs
        ths = theta_of_x(xs)
        return ths

    def _ppf_to_solve(self, x: float, q: float, *args: Any) -> NDArrayF:
        ppf: NDArrayF = self._cdf(*(x,) + args) - q
        return ppf


class exact_theta_fixed_distribution(exact_theta_distribution_base):
    """
    Sample inclination coordinate from an SCF potential.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`
    r : Quantity or None, optional
        If passed, these are the locations at which the theta CDF will be
        evaluated. If None (default), then the r coordinate must be given
        to the CDF and RVS functions.
    **kw:
        Not used.
    """

    def __init__(self, potential: SCFPotential, r: Quantity, **kw: Any) -> None:
        super().__init__(potential)

        # points at which CDF is defined
        self._r = r
        self._Qlsatr = self.calculate_Qls(r)

    @property
    def fixed_radius(self) -> Quantity:
        return self._r

    def _cdf(self, x: ArrayLike, *args: Any) -> NDArrayF:
        cdf: NDArrayF = super()._cdf(x, self._Qlsatr)
        return cdf

    def cdf(self, theta: Quantity) -> NDArrayF:
        """Cumulative distribution function of the given RV.

        Parameters
        ----------
        theta : Quantity['angle']

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `theta`
        """
        return self._cdf(x_of_theta(theta << u.rad))

    def rvs(self, size: Optional[int] = None, random_state: RandomLike = None) -> NDArrayF:
        pts = super().rvs(self._r, size=size, random_state=random_state)
        return pts


class exact_theta_distribution(exact_theta_distribution_base):
    """
    Sample inclination coordinate from an SCF potential.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`

    """

    def _cdf(self, x: NDArrayF, r: float) -> NDArrayF:
        Qls = self.calculate_Qls(r)
        cdf = super()._cdf(x, Qls)
        return cdf

    def cdf(self, theta: Quantity, *args: Any, r: Quantity) -> NDArrayF:
        """Cumulative distribution function of the given RV.

        Parameters
        ----------
        theta : Quantity['angle']
        *args
            Not used.
        r : Quantity['length', float] (optional, keyword-only)

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `theta`
        """
        return self._cdf(x_of_theta(theta), *args, r=r)

    def rvs(
        self, r: Quantity, *, size: Optional[int] = None, random_state: RandomLike = None
    ) -> NDArrayF:
        pts = super().rvs(r, size=size, random_state=random_state)
        return pts
