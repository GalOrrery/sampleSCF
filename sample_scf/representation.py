# -*- coding: utf-8 -*-

"""Utility functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from functools import singledispatch
from inspect import isclass
from typing import Dict, Optional, Type, Union, overload

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import Angle, BaseDifferential, BaseRepresentation
from astropy.coordinates import CartesianRepresentation, Distance, PhysicsSphericalRepresentation
from astropy.coordinates import SphericalRepresentation, UnitSphericalRepresentation
from astropy.units import Quantity, UnitConversionError
from erfa import ufunc as erfa_ufunc
from numpy import abs, all, any, arccos, arctan2, atleast_1d, cos, divide, floating, hypot
from numpy import isfinite, less, nan_to_num, ndarray, sin, isnan
from numpy.typing import ArrayLike

# LOCAL
from ._typing import NDArrayF

__all__ = ["FiniteSphericalRepresentation"]

##############################################################################
# CODE
##############################################################################


@singledispatch
def _zeta_of_r(
    r: Union[ArrayLike, Quantity], /, scale_radius: Union[NDArrayF, Quantity, None] = None
) -> NDArrayF:
    # Default implementation, unless there's a registered specific method.
    # --------------
    # Checks: r must be non-negative, and the scale radius must be None or positive
    if any(less(r, 0)):
        raise ValueError("r must be >= 0")
    elif scale_radius is None:
        scale_radius = 1
    elif not all(isfinite(scale_radius)) or scale_radius <= 0:
        raise ValueError("scale_radius must be a finite number > 0")

    # Calculation
    r_a: Quantity = divide(r, scale_radius)  # can be inf
    zeta: NDArrayF = nan_to_num(divide(r_a - 1, r_a + 1), nan=1.0)
    return zeta


@overload
@_zeta_of_r.register
def zeta_of_r(r: Quantity, /, scale_radius=None) -> NDArrayF:  # type: ignore
    # Checks: r must be a non-negative length-type quantity, and the scale
    # radius must be None or a positive length-type quantity.
    if not isinstance(r, Quantity) or r.unit.physical_type != "length":
        raise UnitConversionError("r must be a Quantity with units of 'length'")
    elif any(isnan(r)) or any(r < 0):
        raise ValueError("r must be >= 0")
    elif scale_radius is not None:
        if not isinstance(scale_radius, Quantity) or scale_radius.unit.physical_type != "length":
            raise TypeError("scale_radius must be a Quantity with units of 'length'")
        elif not isfinite(scale_radius) or scale_radius <= 0:
            raise ValueError("scale_radius must be a finite number > 0")
    else:
        scale_radius = 1 * r.unit

    r_a: Quantity = r / scale_radius  # can be inf
    zeta: NDArrayF = nan_to_num(divide(r_a - 1, r_a + 1), nan=1.0)
    return zeta.value


def zeta_of_r(
    r: Union[NDArrayF, Quantity], /, scale_radius: Union[NDArrayF, Quantity, None] = None
) -> NDArrayF:
    r""":math:`\zeta(r) = \frac{r/a - 1}{r/a + 1}`.

    Map the half-infinite domain [0, infinity) -> [-1, 1].

    Parameters
    ----------
    r : (R,) Quantity['length'], position-only
    scale_radius : Quantity['length'] or None, optional
        If None (default), taken to be 1 in the units of `r`.

    Returns
    -------
    (R,) array[floating]

    Raises
    ------
    TypeError
        If `r` is a Quantity and scale radius is not a Quantity.
        If `r` is not a Quantity and scale radius is a Quantity.
    UnitConversionError
        If `r` is a Quantity but does not have units of length.
        If `r` is a Quantity and `scale_radius` is not None and does not have
        units of length.
    ValueError
        If `r` is less than 0.
        If `scale_radius` is not None and is less than or equal to 0.
    """
    return _zeta_of_r(r, scale_radius=scale_radius)


zeta_of_r.__wrapped__ = _zeta_of_r  # For easier access.


# -------------------------------------------------------------------


def r_of_zeta(
    zeta: ndarray, /, scale_radius: Union[float, floating, Quantity, None] = None
) -> Union[NDArrayF, Quantity]:
    r""":math:`r = \frac{1 + \zeta}{1 - \zeta}`.

    Map back to the half-infinite domain [0, infinity) <- [-1, 1].

    Parameters
    ----------
    zeta : (R,) array[floating] or (R,) Quantity['dimensionless'], position-only
    scale_radius : Quantity['length'] or None, optional

    Returns
    -------
    (R,) ndarray[float] or (R,) Quantity['length']
        A |Quantity| if scale_radius is not None, else a `numpy.ndarray`.

    Raises
    ------
    UnitConversionError
        If `scale_radius` is a |Quantity|, but does not have units of length.
    ValueError
        If `zeta` is not in [-1, 1].
        If `scale_radius` not in (0, `numpy.inf`).

    Warnings
    --------
    RuntimeWarning
        If zeta is 1 (r is `numpy.inf`). Don't worry, it's not a problem.
    """
    if any(zeta < -1) or any(zeta > 1):
        raise ValueError("zeta must be in [-1, 1].")
    elif scale_radius is None:
        scale_radius = 1
    elif scale_radius <= 0 or not isfinite(scale_radius):
        raise ValueError("scale_radius must be in (0, inf).")
    elif (
        isinstance(scale_radius, Quantity)
        and scale_radius.unit.physical_type != "length"  # type: ignore
    ):
        raise UnitConversionError("scale_radius must have units of length")

    r: NDArrayF = atleast_1d(divide(1 + zeta, 1 - zeta))
    r[r < 0] = 0  # correct small errors
    rq: Union[NDArrayF, Quantity]
    rq = scale_radius * r
    return rq


# -------------------------------------------------------------------


def x_of_theta(theta: Union[ndarray, Quantity["angle"]]) -> NDArrayF:  # type: ignore
    r""":math:`x = \cos{\theta}`.

    Parameters
    ----------
    theta : (T,) Quantity['angle'] or array['radian']

    Returns
    -------
    float or (T,) ndarray[floating]
    """
    x: NDArrayF = cos(theta)
    xval = x if not isinstance(x, Quantity) else x.value
    return xval


# -------------------------------------------------------------------


def theta_of_x(x: ArrayLike, unit=u.rad) -> Quantity:
    r""":math:`\theta = \cos^{-1}{x}`.

    Parameters
    ----------
    x : array-like
    unit : unit-like['angular'], optional
        Output units.

    Returns
    -------
    theta : float or ndarray
    """
    th: NDArrayF = arccos(x) << u.rad
    theta = th << unit
    return theta


###########################################################################


class FiniteSphericalRepresentation(BaseRepresentation):
    r"""
    Representation of points in 3D spherical coordinates (using the physics
    convention for azimuth and inclination from the pole) where the radius and
    inclination are rescaled to be on [-1, 1].

    .. math::

        \zeta = \frac{1 - r / a}{1 + r/a} x = \cos(\theta)
    """

    _phi: Quantity
    _x: NDArrayF
    _zeta: NDArrayF
    _scale_radius: Union[NDArrayF, Quantity]

    attr_classes: Dict[str, Type[Quantity]] = {"phi": Angle, "x": Quantity, "zeta": Quantity}

    def __init__(
        self,
        phi: Quantity,
        x: Union[NDArrayF, Quantity, None] = None,
        zeta: Union[NDArrayF, Quantity, None] = None,
        scale_radius: Optional[Quantity] = None,
        differentials: Union[BaseDifferential, Dict[str, BaseDifferential]] = None,
        copy: bool = True,
    ):
        # Adjustments if passing unitful quantities
        if isinstance(x, Quantity) and x.unit.physical_type == "angle":  # type: ignore
            x = x_of_theta(x)
        if isinstance(zeta, Quantity) and zeta.unit.physical_type == "length":  # type: ignore
            if scale_radius is None:
                scale_radius = 1 * zeta.unit  # type: ignore
            zeta = zeta_of_r(zeta, scale_radius=scale_radius)
        elif scale_radius is None:
            raise ValueError("if zeta is not a length, a scale_radius must given")

        super().__init__(phi, x, zeta, copy=copy, differentials=differentials)
        self._scale_radius = scale_radius

        # Wrap/validate phi/theta
        # Note that _phi already holds our own copy if copy=True.
        self._phi.wrap_at(360 * u.deg, inplace=True)

        if any(self._x < -1) or any(self._x > 1):
            raise ValueError(f"inclination angle(s) must be within -1 <= angle <= 1, got {x}")

        if any(self._zeta < -1) or any(self._zeta > 1):
            raise ValueError(f"distances must be within -1 <= zeta <= 1, got {zeta}")

    @property
    def phi(self) -> Quantity:
        """The azimuth of the point(s)."""
        return self._phi

    @property
    def x(self) -> Quantity:
        """The elevation of the point(s)."""
        return self._x

    @property
    def zeta(self) -> Quantity:
        """The distance from the origin to the point(s)."""
        return self._zeta

    @property
    def scale_radius(self) -> Union[NDArrayF, Quantity]:
        return self._scale_radius

    # -----------------------------------------------------
    # corresponding PhysicsSpherical coordinates

    @property
    def theta(self) -> Quantity:
        """The elevation of the point(s)."""
        return self.calculate_theta_of_x(self._x)

    @property
    def r(self) -> Union[NDArrayF, Quantity]:
        """The distance from the origin to the point(s)."""
        return Distance(self.calculate_r_of_zeta(self._zeta), copy=False)

    # -----------------------------------------------------
    # conversion functions

    def calculate_zeta_of_r(self, r: Union[NDArrayF, Quantity], /) -> NDArrayF:
        r""":math:`\zeta(r) = \frac{r/a - 1}{r/a + 1}`.

        Map the half-infinite domain [0, infinity) -> [-1, 1].

        Parameters
        ----------
        r : (R,) Quantity['length'], position-only

        Returns
        -------
        (R,) array[floating]

        See Also
        --------
        sample_scf.representation.zeta_of_r
        """
        return zeta_of_r(r, scale_radius=self.scale_radius)

    def calculate_r_of_zeta(self, zeta: ndarray, /) -> Union[NDArrayF, Quantity]:
        r""":math:`r = \frac{1 + \zeta}{1 - \zeta}`.

        Map back to the half-infinite domain [0, infinity) <- [-1, 1].

        Parameters
        ----------
        zeta : (R,) array[floating] or (R,) Quantity['dimensionless'], position-only

        Returns
        -------
        (R,) ndarray[float] or (R,) Quantity['length']
            A |Quantity| if scale_radius is not None, else a `numpy.ndarray`.

        See Also
        --------
        sample_scf.representation.r_of_zeta
        """
        return r_of_zeta(zeta, scale_radius=self.scale_radius)

    def calculate_x_of_theta(self, theta: Quantity) -> NDArrayF:
        r""":math:`x = \cos{\theta}`.

        Parameters
        ----------
        theta : (T,) Quantity['angle'] or array['radian']

        Returns
        -------
        float or (T,) ndarray[floating]
        """
        return x_of_theta(theta)

    def calculate_theta_of_x(self, x: ArrayLike) -> Quantity:
        r""":math:`\theta = \cos^{-1}{x}`.

        Parameters
        ----------
        x : array-like
        unit : unit-like['angular'] or None, optional
            Output units.

        Returns
        -------
        theta : float or ndarray
        """
        return theta_of_x(x)

    # -----------------------------------------------------

    # def unit_vectors(self):
    #     sinphi, cosphi = sin(self.phi), cos(self.phi)
    #     sintheta, x = sin(self.theta), self.x
    #     return {
    #         "phi": CartesianRepresentation(-sinphi, cosphi, 0.0, copy=False),
    #         "theta": CartesianRepresentation(x * cosphi, x * sinphi, -sintheta, copy=False),
    #         "r": CartesianRepresentation(sintheta * cosphi, sintheta * sinphi, x, copy=False),
    #     }

    # TODO!
    #     def scale_factors(self):
    #         r = self.r / u.radian
    #         sintheta = sin(self.theta)
    #         l = broadcast_to(1.*u.one, self.shape, subok=True)
    #         return {'phi': r * sintheta,
    #                 'theta': r,
    #                 'r': l}

    def represent_as(self, other_class, differential_class=None):
        # Take a short cut if the other class is a spherical representation

        if isclass(other_class):
            if issubclass(other_class, PhysicsSphericalRepresentation):
                diffs = self._re_represent_differentials(other_class, differential_class)
                return other_class(
                    phi=self.phi, theta=self.theta, r=self.r, differentials=diffs, copy=False
                )
            elif issubclass(other_class, SphericalRepresentation):
                diffs = self._re_represent_differentials(other_class, differential_class)
                return other_class(
                    lon=self.phi,
                    lat=90 * u.deg - self.theta,
                    distance=self.r,
                    differentials=diffs,
                    copy=False,
                )
            elif issubclass(other_class, UnitSphericalRepresentation):
                diffs = self._re_represent_differentials(other_class, differential_class)
                return other_class(
                    lon=self.phi, lat=90 * u.deg - self.theta, differentials=diffs, copy=False
                )

        return super().represent_as(other_class, differential_class)

    def to_cartesian(self):
        """
        Converts spherical polar coordinates to 3D rectangular cartesian
        coordinates.
        """
        # We need to convert Distance to Quantity to allow negative values.
        d = self.r.view(Quantity)

        x = d * sin(self.theta) * cos(self.phi)
        y = d * sin(self.theta) * sin(self.phi)
        z = d * cos(self.theta)

        return CartesianRepresentation(x=x, y=y, z=z, copy=False)

    @classmethod
    def from_cartesian(cls, cart, scale_radius: Optional[Quantity] = None):
        """
        Converts 3D rectangular cartesian coordinates to spherical polar
        coordinates.
        """
        s = hypot(cart.x, cart.y)
        r = hypot(s, cart.z)

        phi = arctan2(cart.y, cart.x) << u.rad
        theta = arctan2(s, cart.z) << u.rad

        return cls(phi=phi, x=theta, zeta=r, scale_radius=scale_radius, copy=False)

    @classmethod
    def from_physicsspherical(
        cls, psphere: PhysicsSphericalRepresentation, scale_radius: Optional[Quantity] = None
    ):
        """
        Converts spherical polar coordinates.
        """
        return cls(
            phi=psphere.phi, x=psphere.theta, zeta=psphere.r, scale_radius=scale_radius, copy=False
        )

    def transform(self, matrix, scale_radius: Optional[Quantity] = None):
        """Transform the spherical coordinates using a 3x3 matrix.

        This returns a new representation and does not modify the original one.
        Any differentials attached to this representation will also be
        transformed.

        Parameters
        ----------
        matrix : (3,3) array-like
            A 3x3 matrix, such as a rotation matrix (or a stack of matrices).
        """
        if self.differentials:
            # TODO! shortcut if there are differentials.
            # Currently just super, which uses Cartesian backend.
            rep = super().transform(matrix)

        else:
            # apply transformation in unit-spherical coordinates
            xyz = erfa_ufunc.s2c(self.phi, 90 * u.deg - self.theta)
            p = erfa_ufunc.rxp(matrix, xyz)
            lon, lat, ur = erfa_ufunc.p2s(p)  # `ur` is transformed unit-`r`
            theta = 90 * u.deg - lat

            # create transformed physics-spherical representation,
            # reapplying the distance scaling
            rep = self.__class__(phi=lon, x=theta, zeta=self.r * ur, scale_radius=scale_radius)

        return rep

    def norm(self):
        """Vector norm.

        The norm is the standard Frobenius norm, i.e., the square root of the
        sum of the squares of all components with non-angular units.  For
        spherical coordinates, this is just the absolute value of the radius.

        Returns
        -------
        norm : `astropy.units.Quantity`
            Vector norm, with the same shape as the representation.
        """
        return abs(self.zeta)
