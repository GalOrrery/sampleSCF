# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.sample_intrp`."""


##############################################################################
# IMPORTS

# THIRD PARTY
# import astropy.units as u
# import numpy as np
import pytest

# LOCAL
from .test_base import Test_rv_continuous_modrvs, Test_SCFSamplerBase
from sample_scf import sample_intrp

# from galpy.potential import SCFPotential


##############################################################################
# TESTS
##############################################################################


@pytest.mark.skip("TODO!")
class Test_SCFSampler(Test_SCFSamplerBase):
    """Test :class:`sample_scf.sample_intrp.SCFSampler`."""

    _cls = sample_intrp.SCFSampler


# /class

# -------------------------------------------------------------------


class Test_SCFRSampler(Test_rv_continuous_modrvs):
    """Test :class:`sample_scf.`"""

    # ===============================================================
    # Method Tests

    @pytest.mark.skip("TODO!")
    def test___init__(self):
        """Test :meth:`sample_scf.sample_intrp.SCFRSampler._cdf`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test__cdf(self):
        """Test :meth:`sample_scf.sample_intrp.SCFRSampler._cdf`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test__ppf(self):
        """Test :meth:`sample_scf.sample_intrp.SCFRSampler._ppf`."""
        assert False

    # /def

    # ===============================================================
    # Usage Tests


# /class

# -------------------------------------------------------------------


class Test_SCFThetaSampler(Test_rv_continuous_modrvs):
    """Test :class:`sample_scf.sample_intrp.SCFThetaSampler`."""

    # ===============================================================
    # Method Tests

    @pytest.mark.skip("TODO!")
    def test___init__(self):
        """Test :meth:`sample_scf.sample_intrp.SCFThetaSampler._cdf`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test__cdf(self):
        """Test :meth:`sample_scf.sample_intrp.SCFThetaSampler._cdf`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test_cdf(self):
        """Test :meth:`sample_scf.sample_intrp.SCFThetaSampler.cdf`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test__ppf(self):
        """Test :meth:`sample_scf.sample_intrp.SCFThetaSampler._ppf`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test__rvs(self):
        """Test :meth:`sample_scf.sample_intrp.SCFThetaSampler._rvs`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test_rvs(self):
        """Test :meth:`sample_scf.sample_intrp.SCFThetaSampler.rvs`."""
        assert False

    # /def

    # ===============================================================
    # Usage Tests


# /class

# -------------------------------------------------------------------


class Test_SCFPhiSampler(Test_rv_continuous_modrvs):
    """Test :class:`sample_scf.sample_intrp.SCFPhiSampler`."""

    # ===============================================================
    # Method Tests

    @pytest.mark.skip("TODO!")
    def test___init__(self):
        """Test :meth:`sample_scf.sample_intrp.SCFPhiSampler._cdf`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test__cdf(self):
        """Test :meth:`sample_scf.sample_intrp.SCFPhiSampler._cdf`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test_cdf(self):
        """Test :meth:`sample_scf.sample_intrp.SCFPhiSampler.cdf`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test__ppf(self):
        """Test :meth:`sample_scf.sample_intrp.SCFPhiSampler._ppf`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test__rvs(self):
        """Test :meth:`sample_scf.sample_intrp.SCFPhiSampler._rvs`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test_rvs(self):
        """Test :meth:`sample_scf.sample_intrp.SCFPhiSampler.rvs`."""
        assert False

    # /def

    # ===============================================================
    # Usage Tests


# /class

##############################################################################
# END
