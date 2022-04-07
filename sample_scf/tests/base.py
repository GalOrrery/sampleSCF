# -*- coding: utf-8 -*-


##############################################################################
# IMPORTS

# STDLIB
from abc import ABCMeta, abstractmethod
import inspect
import time

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext
from galpy.potential import KeplerPotential
from numpy.testing import assert_allclose
from scipy.stats import rv_continuous

# LOCAL
from sample_scf.conftest import _hernquist_scf_potential
from sample_scf.base_univariate import rv_potential

##############################################################################
# TESTS
##############################################################################


class BaseTest_Sampler(metaclass=ABCMeta):

    @pytest.fixture(
        scope="class",
        params=[
            "hernquist_scf_potential",
            # "nfw_scf_potential",  # TODO! turn on
        ],
    )
    def potential(self, request):
        if request.param in ("hernquist_scf_potential"):
            potential = _hernquist_scf_potential
        elif request.param == "nfw_scf_potential":
            # potential = nfw_scf_potential.__wrapped__()
            pass
        yield potential

    @pytest.fixture(scope="class")
    @abstractmethod
    def rv_cls(self):
        """Sample class."""
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def rv_cls_args(self):
        return ()

    @pytest.fixture(scope="class")
    def rv_cls_kw(self):
        return {}

    @pytest.fixture(scope="class")
    def cls_pot_kw(self):
        return {}

    @pytest.fixture(scope="class")
    def full_rv_cls_kw(self, rv_cls_kw, cls_pot_kw, potential):
        return {**rv_cls_kw, **cls_pot_kw.get(potential, {})}

    @pytest.fixture(scope="class")
    def sampler(self, rv_cls, potential, rv_cls_args, full_rv_cls_kw):
        """Set up r, theta, or phi sampler."""
        sampler = rv_cls(potential, *rv_cls_args, **full_rv_cls_kw)
        return sampler

    # cdf tests

    @pytest.fixture(scope="class")
    def cdf_args(self):
        return ()

    @pytest.fixture(scope="class")
    def cdf_kw(self):
        return {}

    # rvs tests

    @pytest.fixture(scope="class")
    def rvs_args(self):
        return ()

    @pytest.fixture(scope="class")
    def rvs_kw(self):
        return {}

    # time-scale tests

    def cdf_time_arr(self, size):
        return np.linspace(0, 1e4, size)

    @pytest.fixture(scope="class")
    def cdf_time_scale(self):
        return 0

    @pytest.fixture(scope="class")
    def rvs_time_scale(self):
        return 0

    # ===============================================================
    # Method Tests
    
    def test_init_wrong_potential(self, rv_cls, rv_cls_args, rv_cls_kw):
        """Test initialization when the potential is wrong."""
        # bad value
        with pytest.raises(TypeError, match="<galpy.potential.SCFPotential>"):
            rv_cls(KeplerPotential(), *rv_cls_args, **rv_cls_kw)

    # ---------------------------------------------------------------

    def test_potential_property(self, sampler):
        # Identity
        assert sampler.potential is sampler._potential

    # ===============================================================
    # Time Scaling Tests

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_cdf_time_scaling(self, sampler, size, cdf_args, cdf_kw, cdf_time_scale):
        """Test that the time scales as X * size"""
        x = self.cdf_time_arr(size)
        tic = time.perf_counter()
        sampler.cdf(x, *cdf_args, **cdf_kw)
        toc = time.perf_counter()

        assert (toc - tic) < cdf_time_scale * size  # linear scaling

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_rvs_time_scaling(self, sampler, size, rvs_args, rvs_kw, rvs_time_scale):
        """Test that the time scales as X * size"""
        tic = time.perf_counter()
        sampler.rvs(size=size, *rvs_args, **rvs_kw)
        toc = time.perf_counter()

        assert (toc - tic) < rvs_time_scale * size  # linear scaling
