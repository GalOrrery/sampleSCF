# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.base_univariate`."""


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
from sample_scf.base_univariate import rv_potential, r_distribution_base

from .data import results
from .base import BaseTest_Sampler


##############################################################################
# PARAMETERS

radii = np.concatenate(([0], np.geomspace(1e-1, 1e3, 28), [np.inf]))  # same shape as ↓
thetas = np.linspace(0, np.pi, 30)
phis = np.linspace(0, 2 * np.pi, 30)


##############################################################################
# TESTS
##############################################################################


class BaseTest_rv_potential(BaseTest_Sampler):
    """Test subclasses of `sample_scf.base_univariate.rv_potential`."""

    # ===============================================================
    # Method Tests

    def test_init_signature(self, sampler):
        """Test signature is compatible with `scipy.stats.rv_continuous`.

        The subclasses pass to parent by kwargs, so can't check the full
        suite of parameters.
        """
        sig = inspect.signature(sampler.__init__)
        params = sig.parameters

        assert "potential" in params

    def test_init_attrs(self, sampler, rv_cls_args, rv_cls_kw):
        """Test attributes set at initialization."""
        # check it has the expected attributes
        assert hasattr(sampler, "_potential")
        assert hasattr(sampler, "_nmax")
        assert hasattr(sampler, "_lmax")
        assert hasattr(sampler, "_radial_scale_factor")

        # TODO! expected parameters from scipy rv_continuous

    # ---------------------------------------------------------------

    def test_radial_scale_factor_property(self, sampler):
        # Identity
        assert sampler.radial_scale_factor is sampler._radial_scale_factor
    
    def test_nmax_property(self, sampler):
        # Identity
        assert sampler.nmax is sampler._nmax
    
    def test_lmax_property(self, sampler):
        # Identity
        assert sampler.lmax is sampler._lmax

    # ---------------------------------------------------------------

    @abstractmethod
    def test_cdf(self, sampler, position, expected):
        """Test cdf method."""
        assert_allclose(sampler.cdf(size=size, *position), expected, atol=1e-16)

    @abstractmethod
    def test_rvs(self, sampler, size, random, expected):
        """Test rvs method.
    
        The ``NumpyRNGContext`` is to control the random generator used to make
        the RandomState. For ``random != None``, this doesn't matter.
    
        Each child class will need to define the set of expected results.
        """
        with NumpyRNGContext(4):
            assert_allclose(sampler.rvs(size=size, random_state=random), expected, atol=1e-16)


##############################################################################


class rvtestsampler(rv_potential):
    """A sampler for testing the modified ``rv_continuous`` base class."""

    def _cdf(self, x, *args, **kwargs):
        return x

    cdf = _cdf

    def _rvs(self, *args, size=None, random_state=None):
        if random_state is None:
            random_state = np.random

        return np.atleast_1d(random_state.uniform(size=size))


class Test_rv_potential(BaseTest_rv_potential):
    """Test :class:`sample_scf.base_univariate.rv_potential`."""

    @pytest.fixture(scope="class")
    def rv_cls(self):
        return rvtestsampler

    @pytest.fixture(scope="class")
    def cdf_time_scale(self):
        return 4e-6

    @pytest.fixture(scope="class")
    def rvs_time_scale(self):
        return 1e-4

    # ===============================================================
    # Method Tests

    def test_init_signature(self, sampler):
        """Test signature is compatible with `scipy.stats.rv_continuous`."""
        sig = inspect.signature(sampler.__init__)
        params = sig.parameters

        scipyps = inspect.signature(rv_continuous.__init__).parameters

        assert params["momtype"].default == scipyps["momtype"].default
        assert params["a"].default == scipyps["a"].default
        assert params["b"].default == scipyps["b"].default
        assert params["xtol"].default == scipyps["xtol"].default
        assert params["badvalue"].default == scipyps["badvalue"].default
        assert params["name"].default == scipyps["name"].default
        assert params["longname"].default == scipyps["longname"].default
        assert params["shapes"].default == scipyps["shapes"].default
        assert params["extradoc"].default == scipyps["extradoc"].default
        assert params["seed"].default == scipyps["seed"].default

    @pytest.mark.parametrize(  # TODO! instead by request.param lookup and index
        list(results["Test_rv_potential"]["rvs"].keys()),
        zip(*results["Test_rv_potential"]["rvs"].values()),
    )
    def test_rvs(self, sampler, size, random, expected):
        super().test_rvs(sampler, size, random, expected)


##############################################################################


class BaseTest_r_distribution_base(BaseTest_rv_potential):
    """Test :class:`sample_scf.base_multivariate.r_distribution_base`."""

    @pytest.fixture(scope="class")
    @abstractmethod
    def rv_cls(self):
        """Sample class."""
        return r_distribution_base

    # ===============================================================
    # Method Tests


##############################################################################


class BaseTest_theta_distribution_base(BaseTest_rv_potential):
    """Test :class:`sample_scf.base_multivariate.theta_distribution_base`."""

    @pytest.fixture(scope="class")
    @abstractmethod
    def rv_cls(self):
        return theta_distribution_base

    def cdf_time_arr(self, size: int):
        return np.linspace(0, np.pi, size)

    # ===============================================================
    # Method Tests

    def test_init_attrs(self, sampler):
        """Test attributes set at initialization."""
        super().test_init_attrs(sampler)

        assert hasattr(sampler, "_lrange")
        assert min(sampler._lrange) == 0
        assert max(sampler._lrange) == sampler.lmax + 1

    @pytest.mark.skip("TODO!")
    def test_calculate_Qls(self, potential, sampler):
        """Test :meth:`sample_scf.base_univariate.theta_distribution_base.calculate_Qls`."""
        assert False


##############################################################################


class BaseTest_phi_distribution_base(BaseTest_rv_potential):
    """Test :class:`sample_scf.base_multivariate.phi_distribution_base`."""

    @pytest.fixture(scope="class")
    @abstractmethod
    def rv_cls(self):
        return theta_distribution_base

    def cdf_time_arr(self, size: int):
        return np.linspace(0, 2*np.pi, size)

    # ===============================================================
    # Method Tests

    def test_init_attrs(self, sampler):
        """Test attributes set at initialization."""
        super().test_init_attrs(sampler)

        # l-range
        assert hasattr(sampler, "_lrange")
        assert min(sampler._lrange) == 0
        assert max(sampler._lrange) == sampler.lmax + 1

    @pytest.mark.skip("TODO!")
    def test_pnts_Scs(self, sampler):
        """Test :class:`sample_scf.base_multivariate.phi_distribution_base._pnts_Scs`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_pnts_Scs(self, sampler):
        """Test :class:`sample_scf.base_multivariate.phi_distribution_base._grid_Scs`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_calculate_Scs(self, sampler):
        """Test :class:`sample_scf.base_multivariate.phi_distribution_base.calculate_Scs`."""
        assert False
