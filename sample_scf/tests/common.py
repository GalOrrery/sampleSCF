# -*- coding: utf-8 -*-

"""Common Test Codes."""


##############################################################################
# IMPORTS

# BUILT-IN
import time

# THIRD PARTY
import numpy as np
import pytest
from numpy.testing import assert_allclose

# LOCAL
from .test_base import Test_RVPotential as RVPotentialTest
from sample_scf import conftest

##############################################################################
# CODE
##############################################################################


class SCFRSamplerTestBase(RVPotentialTest):
    def test__cdf(self, sampler, r):
        """Test :meth:`sample_scf.interpolated.SCFRSampler._cdf`."""
        # args and kwargs don't matter
        assert_allclose(sampler._cdf(r), sampler._cdf(r, 10, test=14))

    # /def

    def test__cdf_edge(self, sampler):
        """Test :meth:`sample_scf.interpolated.SCFRSampler._cdf`."""
        assert np.isclose(sampler._cdf(0.0), 0.0, 1e-20)
        assert np.isclose(sampler._cdf(np.inf), 1.0, 1e-20)

    # /def


# /class


class SCFThetaSamplerTestBase(RVPotentialTest):
    def setup_class(self):
        self.cls = None
        self.cls_args = ()
        self.cls_kwargs = {}
        self.cls_pot_kw = {}

        self.cdf_args = ()
        self.cdf_kwargs = {"r": 10}

        self.rvs_args = ()
        self.rvs_kwargs = {"r": 10}

        self.cdf_time_scale = 0
        self.rvs_time_scale = 0

        self.theory = dict(
            hernquist=conftest.hernquist_df,
        )

    # /def

    # ===============================================================
    # Method Tests

    # ===============================================================
    # Time Scaling Tests

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_cdf_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        x = np.linspace(-np.pi / 2, np.pi / 2, size)
        tic = time.perf_counter()
        sampler.cdf(x, *self.cdf_args, **self.cdf_kwargs)
        toc = time.perf_counter()

        assert (toc - tic) < self.cdf_time_scale * size  # linear scaling

    # /def

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_rvs_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        tic = time.perf_counter()
        sampler.rvs(size=size, *self.cdf_args, **self.cdf_kwargs)
        toc = time.perf_counter()

        assert (toc - tic) < self.rvs_time_scale * size  # linear scaling

    # /def


# /class


class SCFPhiSamplerTestBase(RVPotentialTest):
    def setup_class(self):
        self.cls = None
        self.cls_args = ()
        self.cls_kwargs = {}
        self.cls_pot_kw = {}

        self.cdf_args = ()
        self.cdf_kwargs = {"r": 10, "theta": np.pi / 6}

        self.rvs_args = ()
        self.rvs_kwargs = {"r": 10, "theta": np.pi / 6}

        self.cdf_time_scale = 0
        self.rvs_time_scale = 0

        self.theory = dict(
            hernquist=conftest.hernquist_df,
        )

    # /def

    # ===============================================================
    # Method Tests

    # ===============================================================
    # Time Scaling Tests

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_cdf_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        x = np.linspace(0, 2 * np.pi, size)
        tic = time.perf_counter()
        sampler.cdf(x, *self.cdf_args, **self.cdf_kwargs)
        toc = time.perf_counter()

        assert (toc - tic) < self.cdf_time_scale * size  # linear scaling

    # /def

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_rvs_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        tic = time.perf_counter()
        sampler.rvs(size=size, *self.cdf_args, **self.cdf_kwargs)
        toc = time.perf_counter()

        assert (toc - tic) < self.rvs_time_scale * size  # linear scaling

    # /def


# /class


##############################################################################
# END
