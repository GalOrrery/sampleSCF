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
from galpy.potential import SCFPotential

# LOCAL
from sample_scf.utils import _x_of_theta, r_of_zeta, thetaQls, x_of_theta, zeta_of_r

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
            assert np.allclose(zeta_of_r(r), expected)

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
            assert np.allclose(zeta_of_r(r), expected)

    # /def

    @pytest.mark.parametrize("r", [0, 1, np.inf, [0, 1, np.inf]])
    def test_roundtrip(self, r):
        """Test zeta and r round trip. Note that Quantities don't round trip."""
        assert np.allclose(r_of_zeta(zeta_of_r(r)), r)

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
            assert np.allclose(r_of_zeta(zeta), expected)

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
            assert np.allclose(r_of_zeta(zeta), expected)

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
        assert np.allclose(r_of_zeta(zeta, unit=unit), expected)

    # /def

    @pytest.mark.parametrize("zeta", [-1, 0, 1, [-1, 0, 1]])
    def test_roundtrip(self, zeta):
        """Test zeta and r round trip. Note that Quantities don't round trip."""
        assert np.allclose(zeta_of_r(r_of_zeta(zeta)), zeta)

    # /def


# /class


# -------------------------------------------------------------------


class Test_x_of_theta:
    """Test `sample_scf.utils.x_of_theta`."""

    @pytest.mark.parametrize(
        "theta, expected",
        [(-np.pi / 2, -1), (0, 0), (np.pi / 2, 1), ([-np.pi / 2, 0, np.pi / 2], [-1, 0, 1])],
    )
    def test__x_of_theta(self, theta, expected):
        assert np.allclose(_x_of_theta(theta), expected)

    # /def

    @pytest.mark.parametrize(
        "theta, expected",
        [(-np.pi / 2, -1), (0, 0), (np.pi / 2, 1), ([-np.pi / 2, 0, np.pi / 2], [-1, 0, 1])],
    )
    def test_x_of_theta(self, theta, expected):
        assert np.allclose(x_of_theta(theta << u.rad), expected)

    # /def


# /class


# -------------------------------------------------------------------


class Test_thetaQls:
    """Test `sample_scf.utils.x_of_theta`."""

    def setup_class(self):
        """Set up class."""
        Acos = np.zeros((5, 6, 6))

        Acos_hern = Acos.copy()
        Acos_hern[0, 0, 0] = 1
        self.hernquist_pot = SCFPotential(Acos=Acos_hern)

    # /def

    # ===============================================================
    # Usage Tests

    @pytest.mark.parametrize("r, expected", [(0, 1), (1, 0.01989437), (np.inf, 0)])
    def test_hernquist(self, r, expected):
        Qls = thetaQls(self.hernquist_pot, r=r)
        assert len(Qls) == 6
        assert np.isclose(Qls[0], expected)
        assert np.allclose(Qls[1:], 0)

    # /def

    @pytest.mark.skip("TODO!")
    def test_triaxialnfw(self):
        assert False

    # /def


# /class

# -------------------------------------------------------------------


class Test_phiRSms:
    """Test `sample_scf.utils.x_of_theta`."""

    @pytest.mark.skip("TODO!")
    def test__phiRSms(self):
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test_phiRSms(self):
        assert False

    # /def


# /class

##############################################################################
# END
