# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.representation`."""


##############################################################################
# IMPORTS

# STDLIB
import contextlib
import re

# THIRD PARTY
import astropy.units as u
import pytest
from astropy.coordinates import CartesianRepresentation, Distance, PhysicsSphericalRepresentation
from astropy.coordinates import SphericalRepresentation, UnitSphericalRepresentation
from astropy.units import Quantity, UnitConversionError, allclose
from numpy import eye, pi, sin, cos, ndarray, inf, array, atleast_1d

# LOCAL
from sample_scf.representation import FiniteSphericalRepresentation, r_of_zeta, theta_of_x
from sample_scf.representation import x_of_theta, zeta_of_r

##############################################################################
# TESTS
##############################################################################


class Test_FiniteSphericalRepresentation:
    """Test :class:`sample_scf.FiniteSphericalRepresentation`."""

    def setup_class(self):
        """Setup class for testing."""

        self.phi = 0 * u.rad
        self.x = 0
        self.zeta = 0

    @pytest.fixture
    def rep_cls(self):
        return FiniteSphericalRepresentation

    @pytest.fixture
    def differentials(self):
        return None  # TODO!  maybe as a class-level parametrize

    @pytest.fixture
    def scale_radius(self):
        return 8 * u.kpc

    @pytest.fixture
    def rep(self, rep_cls, scale_radius, differentials):
        return rep_cls(
            self.phi,
            x=self.x,
            zeta=self.zeta,
            scale_radius=scale_radius,
            copy=False,
            differentials=differentials,
        )

    # ===============================================================
    # Method Tests

    def test_init_simple(self, rep_cls):
        """
        Test initializing an FiniteSphericalRepresentation.
        This is actually mostly tested by the pytest fixtures, which will fail
        if bad input is given.
        """
        rep = rep_cls(phi=1 * u.deg, x=-1, zeta=0, scale_radius=8 * u.kpc)

        assert isinstance(rep, rep_cls)
        assert (rep.phi, rep.x, rep.zeta) == (1 * u.deg, -1, 0)
        assert rep.scale_radius == 8 * u.kpc

    def test_init_dimensionless_radius(self, rep_cls):
        """Test initialization when scale radius is unit-less."""
        rep = rep_cls(phi=1 * u.deg, x=-1, zeta=0, scale_radius=8)

        assert isinstance(rep, rep_cls)
        assert (rep.phi, rep.x, rep.zeta) == (1 * u.deg, -1, 0)
        assert rep.scale_radius == 8

    def test_init_x_is_theta(self, rep_cls):
        """Test initialization when x has angular units."""
        rep = rep_cls(phi=1 * u.deg, x=90 * u.deg, zeta=0, scale_radius=8 * u.kpc)

        assert isinstance(rep, rep_cls)
        assert rep.phi == 1 * u.deg
        assert allclose(rep.x, 0, atol=1e-16)
        assert rep.zeta == 0
        assert rep.scale_radius == 8 * u.kpc

    def test_init_zeta_is_r(self, rep_cls):
        """Test initialization when zeta has units of length."""
        # When scale_radius is None
        rep = rep_cls(phi=1 * u.deg, x=-1, zeta=8 * u.kpc)
        assert isinstance(rep, rep_cls)
        assert (rep.phi, rep.x, rep.zeta) == (1 * u.deg, -1, 7 / 9)
        assert rep.scale_radius == 1 * u.kpc

        # When scale_radius is not None
        rep = rep_cls(phi=1 * u.deg, x=-1, zeta=8 * u.kpc, scale_radius=8 * u.kpc)
        assert isinstance(rep, rep_cls)
        assert (rep.phi, rep.x, rep.zeta) == (1 * u.deg, -1, 0)
        assert rep.scale_radius == 8 * u.kpc

        # Scale radius must match the units of zeta
        with pytest.raises(TypeError, match="scale_radius must be a Quantity"):
            rep_cls(phi=1 * u.deg, x=-1, zeta=8 * u.kpc, scale_radius=8)

    def test_init_needs_scale_radius(self, rep_cls):
        """
        Test initialization when zeta is correctly unit-less, but no scale
        radius was given.
        """
        with pytest.raises(ValueError, match="if zeta is not a length"):
            rep_cls(phi=1 * u.deg, x=-1, zeta=0)

    def test_init_x_out_of_bounds(self, rep_cls):
        """
        Test initialization when transformed inclination angle is out of bounds.
        """
        with pytest.raises(ValueError, match=re.escape("inclination angle(s) must be within")):
            rep_cls(phi=1 * u.deg, x=-2, zeta=1 * u.kpc)

        with pytest.raises(ValueError, match=re.escape("inclination angle(s) must be within")):
            rep_cls(phi=1 * u.deg, x=2, zeta=1 * u.kpc)

    def test_init_zeta_out_of_bounds(self, rep_cls):
        """Test initialization when transformed distance is out of bounds."""
        with pytest.raises(ValueError, match="distances must be within"):
            rep_cls(phi=1 * u.deg, x=0, zeta=-2, scale_radius=1)

        with pytest.raises(ValueError, match="distances must be within"):
            rep_cls(phi=1 * u.deg, x=0, zeta=2, scale_radius=1)

    # -------------------------------------------

    def test_phi(self, rep_cls, rep):
        """Test :attr:`sample_scf.FiniteSphericalRepresentation.phi`."""
        # class
        assert isinstance(rep_cls.phi, property)

        # instance
        assert rep.phi is rep._phi
        assert isinstance(rep.phi, Quantity)
        assert rep.phi.unit.physical_type == "angle"

    def test_x(self, rep_cls, rep):
        """Test :attr:`sample_scf.FiniteSphericalRepresentation.x`."""
        # class
        assert isinstance(rep_cls.x, property)

        # instance
        assert rep.x is rep._x
        assert isinstance(rep.x, Quantity)
        assert rep.x.unit.physical_type == "dimensionless"

    def test_zeta(self, rep_cls, rep):
        """Test :attr:`sample_scf.FiniteSphericalRepresentation.zeta`."""
        # class
        assert isinstance(rep_cls.zeta, property)

        # instance
        assert rep.zeta is rep._zeta
        assert isinstance(rep.zeta, Quantity)
        assert rep.zeta.unit.physical_type == "dimensionless"

    def test_scale_radius(self, rep_cls, rep):
        """Test :attr:`sample_scf.FiniteSphericalRepresentation.scale_radius`."""
        # class
        assert isinstance(rep_cls.scale_radius, property)

        # instance
        assert rep.scale_radius is rep._scale_radius
        assert isinstance(rep.scale_radius, Quantity)
        assert rep.scale_radius.unit.physical_type == "length"

    # -----------------------------------------------------
    # corresponding PhysicsSpherical coordinates

    def test_theta(self, rep_cls, rep):
        """Test :attr:`sample_scf.FiniteSphericalRepresentation.theta`."""
        # class
        assert isinstance(rep_cls.theta, property)

        # instance
        assert rep.theta == rep.calculate_theta_of_x(rep.x)
        assert isinstance(rep.theta, Quantity)
        assert rep.theta.unit.physical_type == "angle"

    def test_r(self, rep_cls, rep):
        """Test :attr:`sample_scf.FiniteSphericalRepresentation.r`."""
        # class
        assert isinstance(rep_cls.r, property)

        # instance
        assert rep.r == rep.calculate_r_of_zeta(rep.zeta)
        assert isinstance(rep.r, Distance)
        assert rep.r.unit == rep.scale_radius.unit
        assert rep.r.unit.physical_type == "length"

    # -----------------------------------------------------
    # conversion functions
    # TODO! from below tests

    @pytest.mark.skip("TODO!")
    def test_calculate_zeta_of_r(self, rep):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.calculate_zeta_of_r`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_calculate_r_of_zeta(self, rep):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.calculate_r_of_zeta`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_calculate_x_of_theta(self, rep):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.calculate_x_of_theta`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_calculate_theta_of_x(self, rep):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.calculate_theta_of_x`."""
        assert False

    # -----------------------------------------------------

    @pytest.mark.skip("TODO!")
    def test_unit_vectors(self, rep):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.unit_vectors`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_scale_factors(self, rep):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.scale_factors`."""
        assert False

    # --------------------------------------------

    def test_represent_as_PhysicsSphericalRepresentation(self, rep):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.represent_as`."""
        r = rep.represent_as(PhysicsSphericalRepresentation)
        assert allclose(r.phi, rep.phi)
        assert allclose(r.theta, rep.theta)
        assert allclose(r.r, rep.r)

    def test_represent_as_SphericalRepresentation(self, rep):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.represent_as`."""
        r = rep.represent_as(SphericalRepresentation)
        assert allclose(r.lon, rep.phi)
        assert allclose(r.lat, 90 * u.deg - rep.theta)
        assert allclose(r.distance, rep.r)

    def test_represent_as_UnitSphericalRepresentation(self, rep):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.represent_as`."""
        r = rep.represent_as(UnitSphericalRepresentation)
        assert allclose(r.lon, rep.phi)
        assert allclose(r.lat, 90 * u.deg - rep.theta)

    def test_represent_as_CartesianRepresentation(self, rep):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.represent_as`."""
        assert rep.represent_as(CartesianRepresentation) == rep.to_cartesian()

    # --------------------------------------------

    def test_to_cartesian(self, rep):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.to_cartesian`."""
        r = rep.to_cartesian()

        x = rep.r * sin(rep.theta) * cos(rep.phi)
        y = rep.r * sin(rep.theta) * sin(rep.phi)
        z = rep.r * cos(rep.theta)

        assert allclose(r.x, x)
        assert allclose(r.y, y)
        assert allclose(r.z, z)

    def test_from_cartesian(self, rep_cls, rep, scale_radius):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.from_cartesian`."""
        cart = rep.to_cartesian()

        # Not passing a scale radius
        r = rep_cls.from_cartesian(cart)
        assert rep != r

        r = rep_cls.from_cartesian(cart, scale_radius=scale_radius)
        assert allclose(rep.phi, r.phi)
        assert allclose(rep.theta, r.theta)
        assert allclose(rep.zeta, r.zeta)

    def test_from_physicsspherical(self, rep_cls, rep, scale_radius):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.from_physicsspherical`."""
        psphere = rep.represent_as(PhysicsSphericalRepresentation)

        # Not passing a scale radius
        r = rep_cls.from_physicsspherical(psphere)
        assert rep != r

        r = rep_cls.from_physicsspherical(psphere, scale_radius=scale_radius)
        assert allclose(rep.phi, r.phi)
        assert allclose(rep.theta, r.theta)
        assert allclose(rep.zeta, r.zeta)

    def test_transform(self, rep, scale_radius):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.transform`."""
        # Identity
        matrix = eye(3)
        r = rep.transform(matrix, scale_radius)
        assert allclose(rep.phi, r.phi)
        assert allclose(rep.theta, r.theta)
        assert allclose(rep.zeta, r.zeta)

        # alternating coordinates
        matrix = array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        r = rep.transform(matrix, scale_radius)
        assert allclose(rep.phi, r.phi - pi / 2 * u.rad)
        assert allclose(rep.theta, r.theta)
        assert allclose(rep.zeta, r.zeta)

    def test_norm(self, rep):
        """Test :meth:`sample_scf.FiniteSphericalRepresentation.norm`."""
        assert rep.norm() == abs(rep.zeta)


##############################################################################


def test_zeta_of_r_fail():
    """Test :func:`sample_scf.representation.r_of_zeta` with wrong r type."""
    # Negative value
    with pytest.raises(ValueError, match="r must be >= 0"):
        zeta_of_r(-1)

    # Type mismatch
    with pytest.raises(TypeError, match="scale radius cannot be a Quantity"):
        zeta_of_r(1, scale_radius=8 * u.kpc)

    # Negative value
    with pytest.raises(ValueError, match="scale_radius must be > 0"):
        zeta_of_r(1, scale_radius=-1)


@pytest.mark.parametrize(
    "r, scale_radius, expected, warns",
    [
        (0, None, -1.0, False),
        (1, None, 0.0, False),
        (inf, None, 1.0, RuntimeWarning),  # edge case
        (10, None, 9 / 11, False),
        ([0, 1, inf], None, [-1.0, 0.0, 1.0], False),
        ([0, 1, inf], None, [-1.0, 0.0, 1.0], False),
    ],
)
def test_zeta_of_r_ArrayLike(r, scale_radius, expected, warns):
    """Test :func:`sample_scf.representation.r_of_zeta` with wrong r type."""
    with pytest.warns(warns) if warns is not False else contextlib.nullcontext():
        zeta = zeta_of_r(r, scale_radius=scale_radius)  # TODO! scale radius

        assert allclose(zeta, expected)
        assert not isinstance(zeta, Quantity)


def test_zeta_of_r_Quantity_fail():
    """Test :func:`sample_scf.representation.r_of_zeta`: r=Quantity, with errors."""
    # Wrong units
    with pytest.raises(UnitConversionError, match="r must have units of length"):
        zeta_of_r(8 * u.s)

    # Negative value
    with pytest.raises(ValueError, match="r must be >= 0"):
        zeta_of_r(-1 * u.kpc)

    # Type mismatch
    with pytest.raises(TypeError, match="scale_radius must be a Quantity"):
        zeta_of_r(8 * u.kpc, scale_radius=1)

    # Wrong units
    with pytest.raises(UnitConversionError, match="scale_radius must have units of length"):
        zeta_of_r(8 * u.kpc, scale_radius=1 * u.s)

    # Non-positive value
    with pytest.raises(ValueError, match="scale_radius must be > 0"):
        zeta_of_r(1 * u.kpc, scale_radius=-1 * u.kpc)


@pytest.mark.parametrize(
    "r, scale_radius, expected, warns",
    [
        (0 * u.kpc, None, -1.0, False),
        (1 * u.kpc, None, 0.0, False),
        (inf * u.kpc, None, 1.0, RuntimeWarning),  # edge case
        (10 * u.km, None, 9 / 11, False),
        ([0, 1, inf] * u.kpc, None, [-1.0, 0.0, 1.0], False),
        ([0, 1, inf] * u.km, None, [-1.0, 0.0, 1.0], False),
    ],
)
def test_zeta_of_r_Quantity(r, scale_radius, expected, warns):
    """Test :func:`sample_scf.representation.r_of_zeta` with wrong r type."""
    with pytest.warns(warns) if warns is not False else contextlib.nullcontext():
        zeta = zeta_of_r(r, scale_radius=scale_radius)  # TODO! scale radius

        assert allclose(zeta, expected)
        assert isinstance(zeta, Quantity)
        assert zeta.unit.physical_type == "dimensionless"


@pytest.mark.parametrize("r", [0 * u.kpc, 1 * u.kpc, inf * u.kpc, [0, 1, inf] * u.kpc])
def test_zeta_of_r_roundtrip(r):
    """Test zeta and r round trip. Note that Quantities don't round trip."""
    assert allclose(r_of_zeta(zeta_of_r(r, None), 1), r.value)
    # TODO! scale radius


# -----------------------------------------------------


@pytest.mark.parametrize(
    "zeta, expected, warns",
    [
        (-1.0, 0, False),
        (0.0, 1, False),
        (1.0, inf, RuntimeWarning),  # edge case
        (array([-1.0, 0.0, 1.0]), [0, 1, inf], False),
    ],
)
def test_r_of_zeta(zeta, expected, warns):
    """Test :func:`sample_scf.representation.r_of_zeta`."""
    with pytest.warns(warns) if warns is not False else contextlib.nullcontext():
        r = r_of_zeta(zeta, 1)

        assert allclose(r, expected)  # TODO! scale_radius
        assert isinstance(r, ndarray)


def test_r_of_zeta_fail():
    """Test when the input is bad."""
    # Under lower bound
    with pytest.raises(ValueError, match="zeta must be in"):
        r_of_zeta(-2)

    # Above upper bound
    with pytest.raises(ValueError, match="zeta must be in"):
        r_of_zeta(2)


@pytest.mark.parametrize(
    "zeta, scale_radius, expected",
    [
        (0, 1 * u.pc, 1 * u.pc),
    ],
)
def test_r_of_zeta_unit_input(zeta, expected, scale_radius):
    """Test when input units."""
    assert allclose(r_of_zeta(zeta, scale_radius), expected)


@pytest.mark.skip("TODO!")
@pytest.mark.parametrize("zeta", [-1, 0, 1, [-1, 0, 1]])
def test_r_of_zeta_roundtrip(zeta):
    """Test zeta and r round trip. Note that Quantities don't round trip."""
    assert allclose(zeta_of_r(r_of_zeta(zeta, None), None), zeta)


# -----------------------------------------------------


@pytest.mark.parametrize(
    "theta, expected",
    [
        (0, 1),
        (pi / 2, 0),
        (pi, -1),
        ([0, pi / 2, pi], [1, 0, -1]),  # array
        # with units
        (0 << u.rad, 1),
        (pi / 2 << u.rad, 0),
        (pi << u.rad, -1),
        ([pi, pi / 2, 0] << u.rad, [-1, 0, 1]),  # array
    ],
)
def test_x_of_theta(theta, expected):
    """Test :func:`sample_scf.representation.x_of_theta`."""
    assert allclose(x_of_theta(theta), expected, atol=1e-16)


@pytest.mark.parametrize("theta", [0, pi / 2, pi, [0, pi / 2, pi]])  # TODO! units
def test_theta_of_x_roundtrip(theta):
    """Test theta and x round trip. Note that Quantities don't round trip."""
    assert allclose(theta_of_x(x_of_theta(theta)), theta << u.rad)


# -----------------------------------------------------


@pytest.mark.parametrize(
    "x, expected",
    [
        (-1, pi),
        (0, pi / 2),
        (1, 0),
        ([-1, 0, 1], [pi, pi / 2, 0]),  # array
    ],
)
def test_theta_of_x(x, expected):
    """Test :func:`sample_scf.representation.theta_of_x`."""
    assert allclose(theta_of_x(x), expected << u.rad)  # TODO! units


@pytest.mark.parametrize("x", [-1, 0, 1, [-1, 0, 1]])
def test_roundtrip(x):
    """Test x and theta round trip. Note that Quantities don't round trip."""
    assert allclose(x_of_theta(theta_of_x(x)), x, atol=1e-16)  # TODO! units
