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
from .test_base import RVPotentialTest
from sample_scf import conftest

##############################################################################
# CODE
##############################################################################


class r_distributionTestBase(RVPotentialTest):
    def test__cdf(self, sampler, r):
        """Test :meth:`sample_scf.interpolated.r_distribution._cdf`."""
        # args and kwargs don't matter
        assert_allclose(sampler._cdf(r), sampler._cdf(r, 10, test=14))

    def test__cdf_edge(self, sampler):
        """Test :meth:`sample_scf.interpolated.r_distribution._cdf`."""
        assert np.isclose(sampler._cdf(0.0), 0.0, 1e-20)
        assert np.isclose(sampler._cdf(np.inf), 1.0, 1e-20)


class theta_distributionTestBase(RVPotentialTest):
    def setup_class(self):
        super().setup_class(self)

        self.cls = None
        self.cdf_kwargs = {"r": 10}
        self.rvs_kwargs = {"r": 10}

        # time-scale tests
        self.cdf_time_arr = lambda self, size: np.linspace(-np.pi / 2, np.pi / 2, size)
        self.cdf_time_scale = 0
        self.rvs_time_scale = 0


class phi_distributionTestBase(RVPotentialTest):
    def setup_class(self):
        super().setup_class(self)

        self.cls = None
        self.cdf_kwargs = {"r": 10, "theta": np.pi / 6}
        self.rvs_kwargs = {"r": 10, "theta": np.pi / 6}

        # time-scale tests
        self.cdf_time_arr = lambda self, size: np.linspace(0, 2 * np.pi, size)
        self.cdf_time_scale = 0
        self.rvs_time_scale = 0
