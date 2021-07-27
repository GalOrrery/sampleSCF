# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.utils`."""


##############################################################################
# IMPORTS

# BUILT-IN
import contextlib

# THIRD PARTY
import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

# LOCAL
from sample_scf.utils import phiRSms, r_of_zeta, theta_of_x, thetaQls, x_of_theta, zeta_of_r

##############################################################################
# TESTS
##############################################################################


class Test_zeta_of_r:
    """Testing :func:`sample_scf.utils.zeta_of_r`."""

    # ===============================================================
    # Usage Tests

    @pytest.mark.parametrize(
        "r, expected, warns",
        [
            (0, -1.0, False),  # int -> float
            (1, 0.0, False),
            (0.0, -1.0, False),  # float -> float
            (1.0, 0.0, False),
            (np.inf, 1.0, RuntimeWarning),  # edge case
            (u.Quantity(10, u.km), 9 / 11, False),
            (u.Quantity(8, u.s), 7 / 9, False),  # Note the unit doesn't matter
        ],
    )
    def test_scalar_input(self, r, expected, warns):
        """Test when input scalar."""
        with pytest.warns(warns) if warns is not False else contextlib.nullcontext():
            assert_allclose(zeta_of_r(r), expected)

    # /def

    @pytest.mark.parametrize(
        "r, expected",
        [
            ([0, 1, np.inf], [-1.0, 0.0, 1.0]),
            (u.Quantity([0, 1, np.inf], u.km), [-1.0, 0.0, 1.0]),
        ],
    )
    def test_array_input(self, r, expected):
        """Test when input array."""
        with pytest.warns(RuntimeWarning):
            assert_allclose(zeta_of_r(r), expected)

    # /def

    @pytest.mark.parametrize("r", [0, 1, np.inf, [0, 1, np.inf]])
    def test_roundtrip(self, r):
        """Test zeta and r round trip. Note that Quantities don't round trip."""
        assert_allclose(r_of_zeta(zeta_of_r(r)), r)

    # /def


# /class


# -------------------------------------------------------------------


class Test_r_of_zeta:
    """Testing :func:`sample_scf.utils.r_of_zeta`."""

    # ===============================================================
    # Usage Tests

    @pytest.mark.parametrize(
        "zeta, expected, warns",
        [
            (-1.0, 0, False),  # int -> float
            (0.0, 1, False),
            (-1.0, 0.0, False),  # float -> float
            (0.0, 1.0, False),
            (1.0, np.inf, RuntimeWarning),  # edge case
            (2.0, 0, False),  # out of bounds
            (-2.0, 0, False),  # out of bounds
        ],
    )
    def test_scalar_input(self, zeta, expected, warns):
        """Test when input scalar."""
        with pytest.warns(warns) if warns is not False else contextlib.nullcontext():
            assert_allclose(r_of_zeta(zeta), expected)

    # /def

    @pytest.mark.parametrize(
        "zeta, expected, warns",
        [
            ([-1.0, 0.0, 1.0], [0, 1, np.inf], RuntimeWarning),
        ],
    )
    def test_array_input(self, zeta, expected, warns):
        """Test when input array."""
        with pytest.warns(warns) if warns is not False else contextlib.nullcontext():
            assert_allclose(r_of_zeta(zeta), expected)

    # /def

    @pytest.mark.parametrize(
        "zeta, expected, unit",
        [
            (0, 1, None),
            (0, 1 * u.pc, u.pc),
            (0, 1 * u.Hz, u.Hz),
        ],
    )
    def test_unit_input(self, zeta, expected, unit):
        """Test when input units."""
        assert_allclose(r_of_zeta(zeta, unit=unit), expected)

    # /def

    @pytest.mark.parametrize("zeta", [-1, 0, 1, [-1, 0, 1]])
    def test_roundtrip(self, zeta):
        """Test zeta and r round trip. Note that Quantities don't round trip."""
        assert_allclose(zeta_of_r(r_of_zeta(zeta)), zeta)

    # /def


# /class


# -------------------------------------------------------------------


class Test_x_of_theta:
    """Test `sample_scf.utils.x_of_theta`."""

    @pytest.mark.parametrize(
        "theta, expected",
        [
            (-np.pi / 2, -1),
            (0, 0),
            (np.pi / 2, 1),
            ([-np.pi / 2, 0, np.pi / 2], [-1, 0, 1]),  # array
            # with units
            (-np.pi / 2 << u.rad, -1),
            (0 << u.deg, 0),
            (np.pi / 2 << u.rad, 1),
            ([-np.pi / 2, 0, np.pi / 2] << u.rad, [-1, 0, 1]),  # array
        ],
    )
    def test_x_of_theta(self, theta, expected):
        assert_allclose(x_of_theta(theta), expected, atol=1e-16)

    # /def

    @pytest.mark.parametrize("theta", [-np.pi / 2, 0, np.pi / 2, [-np.pi / 2, 0, np.pi / 2]])
    def test_roundtrip(self, theta):
        """Test theta and x round trip. Note that Quantities don't round trip."""
        assert_allclose(theta_of_x(x_of_theta(theta << u.rad)), theta)

    # /def


# /class

# -------------------------------------------------------------------


class Test_theta_of_x:
    """Test `sample_scf.utils.theta_of_x`."""

    @pytest.mark.parametrize(
        "x, expected",
        [
            (-1, -np.pi / 2),
            (0, 0),
            (1, np.pi / 2),
            ([-1, 0, 1], [-np.pi / 2, 0, np.pi / 2]),  # array
        ],
    )
    def test_theta_of_x(self, x, expected):
        assert_allclose(theta_of_x(x), expected)

    # /def

    @pytest.mark.parametrize(
        "x, expected, unit",
        [
            (-1, -np.pi / 2, None),
            (0, 0 * u.deg, u.deg),
            (1, np.pi / 2 * u.rad, u.rad),
        ],
    )
    def test_unit_input(self, x, expected, unit):
        """Test when input units."""
        assert_allclose(theta_of_x(x, unit=unit), expected)

    # /def

    @pytest.mark.parametrize("x", [-1, 0, 1, [-1, 0, 1]])
    def test_roundtrip(self, x):
        """Test x and theta round trip. Note that Quantities don't round trip."""
        assert_allclose(x_of_theta(theta_of_x(x)), x, atol=1e-16)

    # /def


# -------------------------------------------------------------------


class Test_thetaQls:
    """Test `sample_scf.utils.x_of_theta`."""

    # ===============================================================
    # Usage Tests

    @pytest.mark.parametrize("r, expected", [(0, 1), (1, 0.01989437), (np.inf, 0)])
    def test_hernquist(self, hernquist_scf_potential, r, expected):
        Qls = thetaQls(hernquist_scf_potential, r=r)
        # shape should be L (see setup_class)
        assert len(Qls) == 6
        # only 1st index is non-zero
        assert np.isclose(Qls[0], expected)
        assert_allclose(Qls[1:], 0)

    # /def

    @pytest.mark.skip("TODO!")
    def test_nfw(self, nfw_scf_potential):
        assert False

    # /def


# /class

# -------------------------------------------------------------------


class Test_phiRSms:
    """Test `sample_scf.utils.x_of_theta`."""

    # ===============================================================
    # Tests

    # @pytest.mark.skip("TODO!")
    @pytest.mark.parametrize(
        "r, theta, expected",
        [
            # show it doesn't depend on theta
            (0, -np.pi / 2, (np.zeros(5), np.zeros(5))),
            (0, 0, (np.zeros(5), np.zeros(5))),  # special case when x=0 is 0
            (0, np.pi / 6, (np.zeros(5), np.zeros(5))),
            (0, np.pi / 2, (np.zeros(5), np.zeros(5))),
            # nor on r
            (1, -np.pi / 2, (np.zeros(5), np.zeros(5))),
            (10, -np.pi / 4, (np.zeros(5), np.zeros(5))),
            (100, np.pi / 6, (np.zeros(5), np.zeros(5))),
            (1000, np.pi / 2, (np.zeros(5), np.zeros(5))),
            # Legendre[n=0, l=0, z=z] = 1 is a special case
            (1, 0, (np.zeros(5), np.zeros(5))),
            (10, 0, (np.zeros(5), np.zeros(5))),
            (100, 0, (np.zeros(5), np.zeros(5))),
            (1000, 0, (np.zeros(5), np.zeros(5))),
        ],
    )
    def test_phiRSms_hernquist(self, hernquist_scf_potential, r, theta, expected):
        Rm, Sm = phiRSms(hernquist_scf_potential, r, theta)
        assert_allclose(Rm[1:], expected[0], atol=1e-16)
        assert_allclose(Sm[1:], expected[1], atol=1e-16)

        if theta == 0 and r != 0:
            assert Rm[0] != 0
            assert Sm[0] == 0

    # /def


# /class

##############################################################################
# END
