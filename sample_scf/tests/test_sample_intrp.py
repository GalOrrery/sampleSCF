# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.sample_intrp`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import numpy as np
import pytest
from numpy.testing import assert_allclose

# LOCAL
from .test_base import SCFSamplerTestBase
from .test_base import Test_RVContinuousModRVS as RVContinuousModRVSTest
from sample_scf import sample_intrp

##############################################################################
# PARAMETERS

rgrid = np.geomspace(1e-1, 1e3, 100)
tgrid = np.linspace(-np.pi / 2, np.pi / 2, 30)
pgrid = np.linspace(0, 2 * np.pi, 30)


##############################################################################
# TESTS
##############################################################################


class Test_SCFSampler(SCFSamplerTestBase):
    """Test :class:`sample_scf.sample_intrp.SCFSampler`."""

    def setup_class(self):
        super().setup_class(self)

        self.cls = sample_intrp.SCFSampler
        self.cls_args = (rgrid, tgrid, pgrid)

    # /def

    @pytest.mark.skip("TODO!")
    def test_cdf(self, sampler, r, theta, phi, expected):
        """Test :meth:`sample_scf.base.SCFSamplerBase.cdf`."""
        assert np.allclose(sampler.cdf(r, theta, phi), expected, atol=1e-16)

    # /def

    @pytest.mark.skip("TODO!")
    def test_rvs(self, sampler, id, size, random):
        """Test :meth:`sample_scf.base.SCFSamplerBase.rvs`."""
        samples = sampler.rvs(size=size, random_state=random)
        sce = coord.PhysicsSphericalRepresentation(**self.expected_rvs[id])

        assert_allclose(samples.r, sce.r, atol=1e-16)
        assert_allclose(samples.theta.value, sce.theta.value, atol=1e-16)
        assert_allclose(samples.phi.value, sce.phi.value, atol=1e-16)

    # /def


# /class

# -------------------------------------------------------------------


class Test_SCFRSampler(RVContinuousModRVSTest):
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

    @pytest.mark.skip("TODO!")
    def test_rvs(self):
        """Test :meth:`sample_scf.sample_intrp.SCFRSampler.rvs`."""
        assert False

    # /def

    # ===============================================================
    # Usage Tests


# /class

# -------------------------------------------------------------------


class Test_SCFThetaSampler(RVContinuousModRVSTest):
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


class Test_SCFPhiSampler(RVContinuousModRVSTest):
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
