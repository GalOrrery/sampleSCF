# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.base`."""


##############################################################################
# IMPORTS

# BUILT-IN
import abc
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
from sample_scf import base, conftest

##############################################################################
# TESTS
##############################################################################


class rvtestsampler(base.rv_potential):
    """A sampler for testing the modified ``rv_continuous`` base class."""

    def _cdf(self, x, *args, **kwargs):
        return x

    cdf = _cdf

    def _rvs(self, *args, size=None, random_state=None):
        if random_state is None:
            random_state = np.random

        return np.atleast_1d(random_state.uniform(size=size))


class Test_RVPotential(metaclass=abc.ABCMeta):
    """Test `sample_scf.base.rv_potential`."""

    def setup_class(self):
        # sampler initialization
        self.cls = rvtestsampler
        self.cls_args = ()
        self.cls_kwargs = {}
        self.cls_pot_kw = {}

        # (kw)args into cdf()
        self.cdf_args = ()
        self.cdf_kwargs = {}

        # (kw)args into rvs()
        self.rvs_args = ()
        self.rvs_kwargs = {}

        # time-scale tests
        self.cdf_time_arr = lambda self, size: np.linspace(0, 1e4, size)
        self.cdf_time_scale = 4e-6
        self.rvs_time_scale = 1e-4

    @pytest.fixture()
    def sampler(self, potentials):
        """Set up r, theta, or phi sampler."""
        kw = {**self.cls_kwargs, **self.cls_pot_kw.get(potentials, {})}
        sampler = self.cls(potentials, *self.cls_args, **kw)

        return sampler

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

    def test_init(self, sampler):
        """Test initialization."""
        # check it has the expected attributes
        assert hasattr(sampler, "_potential")
        assert hasattr(sampler, "_nmax")
        assert hasattr(sampler, "_lmax")

        # bad value
        with pytest.raises(TypeError, match="<galpy.potential.SCFPotential>"):
            sampler.__class__(KeplerPotential(), *self.cls_args, **self.cls_kwargs)

    def test_cdf(self, sampler):
        """Test :meth:`sample_scf.base.rv_potential.cdf`."""
        assert sampler.cdf(0.0, *self.cdf_args, **self.cdf_kwargs) == 0.0

    @pytest.mark.parametrize(
        "size, random, expected",
        [
            (None, 0, 0.5488135039273248),
            (1, 2, 0.43599490214200376),
            ((3, 1), 4, (0.9670298390136767, 0.5472322491757223, 0.9726843599648843)),
            ((3, 1), None, (0.9670298390136767, 0.5472322491757223, 0.9726843599648843)),
        ],
    )
    def test_rvs(self, sampler, size, random, expected):
        """Test :meth:`sample_scf.base.rv_potential.rvs`.

        The ``NumpyRNGContext`` is to control the random generator used to make
        the RandomState. For ``random != None``, this doesn't matter.
        """
        with NumpyRNGContext(4):
            assert_allclose(sampler.rvs(size=size, random_state=random), expected, atol=1e-16)

    # ===============================================================
    # Time Scaling Tests

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_cdf_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        x = self.cdf_time_arr(size)
        tic = time.perf_counter()
        sampler.cdf(x, *self.cdf_args, **self.cdf_kwargs)
        toc = time.perf_counter()

        assert (toc - tic) < self.cdf_time_scale * size  # linear scaling

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_rvs_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        tic = time.perf_counter()
        sampler.rvs(size=size, *self.rvs_args, **self.rvs_kwargs)
        toc = time.perf_counter()

        assert (toc - tic) < self.rvs_time_scale * size  # linear scaling


class RVPotentialTest(Test_RVPotential):
    """Test rv_potential subclasses."""

    def setup_class(self):
        super().setup_class(self)

        # self.cls_pot_kw = conftest.cls_pot_kw
        self.theory = conftest.theory

    def test_init_signature(self, sampler):
        """Test signature is compatible with `scipy.stats.rv_continuous`."""
        sig = inspect.signature(sampler.__init__)
        params = sig.parameters

        assert "potential" in params


##############################################################################


class Test_SCFSamplerBase:
    """Test :class:`sample_scf.base.SCFSamplerBase`."""

    def setup_class(self):
        # sampler initialization
        self.cls = base.SCFSamplerBase
        self.cls_args = ()
        self.cls_kwargs = {}
        self.cls_pot_kw = {}

        self.expected_rvs = {
            0: dict(r=0.548813503927, theta=1.021982822867 * u.rad, phi=0.548813503927 * u.rad),
            1: dict(r=0.548813503927, theta=1.021982822867 * u.rad, phi=0.548813503927 * u.rad),
            2: dict(
                r=[0.9670298390136, 0.5472322491757, 0.9726843599648, 0.7148159936743],
                theta=[0.603766487781, 1.023564077619, 0.598111966830, 0.855980333120] * u.rad,
                phi=[0.9670298390136, 0.547232249175, 0.9726843599648, 0.7148159936743] * u.rad,
            ),
        }

    @pytest.fixture()
    def sampler(self, potentials):
        """Set up r, theta, & phi sampler."""

        sampler = self.cls(potentials, *self.cls_args, **self.cls_kwargs)
        sampler._r_distribution = rvtestsampler(potentials)
        sampler._theta_distribution = rvtestsampler(potentials)
        sampler._phi_distribution = rvtestsampler(potentials)

        return sampler

    # ===============================================================
    # Method Tests

    def test_init(self, sampler, potentials):
        assert sampler._potential is potentials

    def test_r_distribution_property(self, sampler):
        """Test :meth:`sample_scf.base.SCFSamplerBase.rsampler`."""
        assert isinstance(sampler.rsampler, base.rv_potential)

    def test_theta_distribution_property(self, sampler):
        """Test :meth:`sample_scf.base.SCFSamplerBase.thetasampler`."""
        assert isinstance(sampler.thetasampler, base.rv_potential)

    def test_phi_distribution_property(self, sampler):
        """Test :meth:`sample_scf.base.SCFSamplerBase.phisampler`."""
        assert isinstance(sampler.phisampler, base.rv_potential)

    @pytest.mark.parametrize(
        "r, theta, phi, expected",
        [
            (0, 0, 0, [0, 0, 0]),
            (1, 0, 0, [1, 0, 0]),
            ([0, 1], [0, 0], [0, 0], [[0, 0, 0], [1, 0, 0]]),
        ],
    )
    def test_cdf(self, sampler, r, theta, phi, expected):
        """Test :meth:`sample_scf.base.SCFSamplerBase.cdf`."""
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
        """Test :meth:`sample_scf.base.SCFSamplerBase.rvs`."""
        samples = sampler.rvs(size=size, random_state=random, vectorized=vectorized)
        sce = coord.PhysicsSphericalRepresentation(**self.expected_rvs[id])

        assert_allclose(samples.r, sce.r, atol=1e-16)
        assert_allclose(samples.theta.value, sce.theta.value, atol=1e-16)
        assert_allclose(samples.phi.value, sce.phi.value, atol=1e-16)


class SCFSamplerTestBase(Test_SCFSamplerBase, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def setup_class(self):
        pass

    @pytest.fixture()
    def sampler(self, potentials):
        """Set up r, theta, phi sampler."""
        kw = {**self.cls_kwargs, **self.cls_pot_kw.get(potentials, {})}
        sampler = self.cls(potentials, *self.cls_args, **kw)

        return sampler
