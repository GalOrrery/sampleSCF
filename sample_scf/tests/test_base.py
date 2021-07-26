# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.base`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.units as u
import numpy as np
import pytest
from galpy.potential import SCFPotential

# LOCAL
from sample_scf import base

##############################################################################
# TESTS
##############################################################################


class Test_rv_continuous_modrvs:
    """Test `sample_scf.base.rv_continuous_modrvs`."""

    @pytest.mark.skip("TODO!")
    def test_rvs(self):
        """Test :meth:`sample_scf.base.rv_continuous_modrvs.rvs`."""
        assert False

    # /def


# /class


# -------------------------------------------------------------------


class Test_SCFSamplerBase:
    """Test :class:`sample_scf.base.SCFSamplerBase`."""

    _cls = base.SCFSamplerBase

    @pytest.mark.skip("TODO!")
    def test_rsampler(self):
        """Test :meth:`sample_scf.base.SCFSamplerBase.rsampler`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test_thetasampler(self):
        """Test :meth:`sample_scf.base.SCFSamplerBase.thetasampler`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test_phisampler(self):
        """Test :meth:`sample_scf.base.SCFSamplerBase.phisampler`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test_cdf(self):
        """Test :meth:`sample_scf.base.SCFSamplerBase.cdf`."""
        assert False

    # /def

    @pytest.mark.skip("TODO!")
    def test_rvs(self):
        """Test :meth:`sample_scf.base.SCFSamplerBase.rvs`."""
        assert False

    # /def


# /class


##############################################################################
# END
