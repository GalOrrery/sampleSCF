# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.interpolated.utils`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import pytest
from numpy import inf, isclose, pi, zeros
from numpy.testing import assert_allclose

# LOCAL
from sample_scf.base_univariate import _calculate_Qls, _calculate_Scs

##############################################################################
# TESTS
##############################################################################


class Test_Qls:
    """Test `sample_scf.base_univariate.Qls`."""

    # ===============================================================
    # Usage Tests

    @pytest.mark.parametrize("r, expected", [(0, 1), (1, 0.01989437), (inf, 0)])
    def test_hernquist(self, hernquist_scf_potential, r, expected):
        Qls = _calculate_Qls(hernquist_scf_potential, r=r)
        # shape should be L (see setup_class)
        assert len(Qls) == 6
        # only 1st index is non-zero
        assert isclose(Qls[0], expected)
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
            (0, -pi / 2, (zeros(5), zeros(5))),
            (0, 0, (zeros(5), zeros(5))),  # special case when x=0 is 0
            (0, pi / 6, (zeros(5), zeros(5))),
            (0, pi / 2, (zeros(5), zeros(5))),
            # nor on r
            (1, -pi / 2, (zeros(5), zeros(5))),
            (10, -pi / 4, (zeros(5), zeros(5))),
            (100, pi / 6, (zeros(5), zeros(5))),
            (1000, pi / 2, (zeros(5), zeros(5))),
            # Legendre[n=0, l=0, z=z] = 1 is a special case
            (1, 0, (zeros(5), zeros(5))),
            (10, 0, (zeros(5), zeros(5))),
            (100, 0, (zeros(5), zeros(5))),
            (1000, 0, (zeros(5), zeros(5))),
        ],
    )
    def test_phiScs_hernquist(self, hernquist_scf_potential, r, theta, expected):
        Rm, Sm = _calculate_Scs(hernquist_scf_potential, r, theta, warn=False)
        assert Rm.shape == Sm.shape
        assert Rm.shape == (1, 1, 6)
        assert_allclose(Rm[0, 0, 1:], expected[0], atol=1e-16)
        assert_allclose(Sm[0, 0, 1:], expected[1], atol=1e-16)

        if theta == 0 and r != 0:
            assert Rm[0, 0, 0] != 0
            assert Sm[0, 0, 0] == 0
