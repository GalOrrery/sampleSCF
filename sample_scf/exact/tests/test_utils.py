# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.interpolated.utils`."""


##############################################################################
# IMPORTS

# STDLIB
import contextlib

# THIRD PARTY
import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

# LOCAL
from sample_scf.interpolated.utils import 

##############################################################################
# TESTS
##############################################################################


class Test_Qls:
    """Test `sample_scf.base_univariate.Qls`."""

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

    @pytest.mark.skip("TODO!")
    def test_nfw(self, nfw_scf_potential):
        assert False


# -------------------------------------------------------------------


class Test_phiScs:

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
    def test_phiScs_hernquist(self, hernquist_scf_potential, r, theta, expected):
        Rm, Sm = phiScs(hernquist_scf_potential, r, theta, warn=False)
        assert Rm.shape == Sm.shape
        assert Rm.shape == (1, 1, 6)
        assert_allclose(Rm[0, 0, 1:], expected[0], atol=1e-16)
        assert_allclose(Sm[0, 0, 1:], expected[1], atol=1e-16)

        if theta == 0 and r != 0:
            assert Rm[0, 0, 0] != 0
            assert Sm[0, 0, 0] == 0
