# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.base_multivariate`."""


##############################################################################
# IMPORTS

# STDLIB
import inspect
import time
from abc import ABCMeta, abstractmethod

# THIRD PARTY
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import BaseRepresentation
from galpy.potential import SCFPotential
from numpy.testing import assert_allclose
from astropy.coordinates import PhysicsSphericalRepresentation


# LOCAL
from sample_scf import conftest
from sample_scf.base_univariate import r_distribution_base, theta_distribution_base, phi_distribution_base

from .base import BaseTest_Sampler
from .test_base_univariate import rvtestsampler, radii, thetas, phis


##############################################################################
# TESTS
##############################################################################


class BaseTest_SCFSamplerBase(BaseTest_Sampler):
    """Test :class:`sample_scf.base_multivariate.SCFSamplerBase`."""

    @pytest.fixture(scope="class")
    @abstractmethod
    def rv_cls(self):
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def r_distribution_cls(self):
        return r_distribution_base

    @pytest.fixture(scope="class")
    def theta_distribution_cls(self):
        return theta_distribution_base

    @pytest.fixture(scope="class")
    def phi_distribution_cls(self):
        return phi_distribution_base

    def setup_class(self):
        self.expected_rvs = {
            0: dict(r=0.548813503927, theta=1.021982822867 * u.rad, phi=0.548813503927 * u.rad),
            1: dict(r=0.548813503927, theta=1.021982822867 * u.rad, phi=0.548813503927 * u.rad),
            2: dict(
                r=[0.9670298390136, 0.5472322491757, 0.9726843599648, 0.7148159936743],
                theta=[0.603766487781, 1.023564077619, 0.598111966830, 0.855980333120] * u.rad,
                phi=[0.9670298390136, 0.547232249175, 0.9726843599648, 0.7148159936743] * u.rad,
            ),
        }

    # ===============================================================
    # Method Tests

    def test_init_attrs(self, sampler):
        assert hasattr(sampler, "_potential")
        assert hasattr(sampler, "_r_distribution")
        assert hasattr(sampler, "_theta_distribution")
        assert hasattr(sampler, "_phi_distribution")

    # ---------------------------------------------------------------

    def test_potential_property(self, sampler, potential):
        """Test :meth:`sample_scf.base_multivariate.SCFSamplerBase.potential`."""
        # Identity
        assert sampler.potential is sampler._potential
        # Properties
        assert isinstance(sampler.potential, SCFPotential)

    def test_r_distribution_property(self, sampler, r_distribution_cls):
        """Test :meth:`sample_scf.base_multivariate.SCFSamplerBase.r_distribution`."""
        # Identity
        assert sampler.r_distribution is sampler._r_distribution
        # Properties
        assert isinstance(sampler.r_distribution, r_distribution_cls)

    def test_theta_distribution_property(self, sampler, theta_distribution_cls):
        """Test :meth:`sample_scf.base_multivariate.SCFSamplerBase.theta_distribution`."""
        # Identity
        assert sampler.theta_distribution is sampler._theta_distribution
        # Properties
        assert isinstance(sampler.theta_distribution, theta_distribution_cls)

    def test_phi_distribution_property(self, sampler, phi_distribution_cls):
        """Test :meth:`sample_scf.base_multivariate.SCFSamplerBase.phi_distribution`."""
        # Identity
        assert sampler.phi_distribution is sampler._phi_distribution
        # Properties
        assert isinstance(sampler.phi_distribution, phi_distribution_cls)

    def test_radial_scale_factor_property(self, sampler):
        # Identity
        assert sampler.radial_scale_factor is sampler.r_distribution.radial_scale_factor

    def test_nmax_property(self, sampler):
        # Identity
        assert sampler.nmax is sampler.r_distribution.nmax

    def test_lmax_property(self, sampler):
        # Identity
        assert sampler.lmax is sampler.r_distribution.lmax

    # ---------------------------------------------------------------
    
    @abstractmethod
    def test_cdf(self, sampler, position, expected):
        """Test cdf method."""
        cdf = sampler.cdf(size=size, *position)

        assert isinstance(cdf, np.ndarray)
        assert False

        assert_allclose(cdf, expected, atol=1e-16)
    
    @abstractmethod
    def test_rvs(self, sampler, size, random, expected):
        """Test rvs method.
    
        The ``NumpyRNGContext`` is to control the random generator used to make
        the RandomState. For ``random != None``, this doesn't matter.
    
        Each child class will need to define the set of expected results.
        """
        with NumpyRNGContext(4):
            rvs = sampler.rvs(size=size, random_state=random)

        assert isinstance(rvs, BaseRepresentation)

        r = rvs.represent_as(PhysicsSphericalRepresentation)
        assert_allclose(r.r, expected.r, atol=1e-16)
        assert_allclose(r.theta, expected.theta, atol=1e-16)
        assert_allclose(r.phi, expected.phi, atol=1e-16)

    # ---------------------------------------------------------------

    def test_repr(self):
        """Test :meth:`sample_scf.base_multivariate.SCFSamplerBase.__repr__`."""
        assert False


##############################################################################


class Test_SCFSamplerBase(BaseTest_SCFSamplerBase):

    @pytest.fixture(scope="class")
    def rv_cls(self):
        return SCFSamplerBase

    @pytest.fixture()
    def sampler(self, potential):
        """Set up r, theta, phi sampler."""
        super().sampler(potential)

        sampler._r_distribution = rvtestsampler(potentials)
        sampler._theta_distribution = rvtestsampler(potentials)
        sampler._phi_distribution = rvtestsampler(potentials)

        return sampler

    # ===============================================================
    # Method Tests

    @pytest.mark.parametrize(
        "r, theta, phi, expected",
        [
            (0, 0, 0, [0, 0, 0]),
            (1, 0, 0, [1, 0, 0]),
            ([0, 1], [0, 0], [0, 0], [[0, 0, 0], [1, 0, 0]]),
        ],
    )
    def test_cdf(self, sampler, r, theta, phi, expected):
        """Test :meth:`sample_scf.base_multivariate.SCFSamplerBase.cdf`."""
        cdf = sampler.cdf(r, theta, phi)
        assert np.allclose(cdf, expected, atol=1e-16)
    
        # also test shape
        assert tuple(np.atleast_1d(np.squeeze((*np.shape(r), 3)))) == cdf.shape
    
    @pytest.mark.parametrize(
        "id, size, random, vectorized",
        [
            (0, None, 0, True),
            (0, None, 0, False),
            (1, 1, 0, True),
            (1, 1, 0, False),
            (2, 4, 4, True),
            (2, 4, 4, False),
        ],
    )
    def test_rvs(self, sampler, id, size, random, vectorized):
        """Test :meth:`sample_scf.base_multivariate.SCFSamplerBase.rvs`."""
        samples = sampler.rvs(size=size, random_state=random, vectorized=vectorized)
        sce = PhysicsSphericalRepresentation(**self.expected_rvs[id])
    
        assert_allclose(samples.r, sce.r, atol=1e-16)
        assert_allclose(samples.theta.value, sce.theta.value, atol=1e-16)
        assert_allclose(samples.phi.value, sce.phi.value, atol=1e-16)
