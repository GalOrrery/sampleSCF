# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.base`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext
from numpy.testing import assert_allclose

# LOCAL
from sample_scf import base

##############################################################################
# TESTS
##############################################################################


class testrvsampler(base.rv_continuous_modrvs):
    def _cdf(self, x, *args, **kwargs):
        return x

    # /def

    cdf = _cdf

    def _rvs(self, *args, size=None, random_state=None):
        if random_state is None:
            random_state = np.random

        return np.atleast_1d(random_state.uniform(size=size))

    # /def


# /class


class Test_RVContinuousModRVS:
    """Test `sample_scf.base.rv_continuous_modrvs`."""

    def setup_class(self):
        self.sampler = testrvsampler()

    # /def

    # ===============================================================

    @pytest.mark.parametrize(
        "size, random, expected",
        [
            (None, 0, 0.5488135039273248),
            (1, 2, 0.43599490214200376),
            ((3, 1), 4, (0.9670298390136767, 0.5472322491757223, 0.9726843599648843)),
            ((3, 1), None, (0.9670298390136767, 0.5472322491757223, 0.9726843599648843)),
        ],
    )
    def test_rvs(self, size, random, expected):
        """Test :meth:`sample_scf.base.rv_continuous_modrvs.rvs`.

        The ``NumpyRNGContext`` is to control the random generator used to make
        the RandomState. For ``random != None``, this doesn't matter.
        """
        with NumpyRNGContext(4):
            assert_allclose(self.sampler.rvs(size=size, random_state=random), expected, atol=1e-16)

    # /def


# /class


# -------------------------------------------------------------------


class Test_SCFSamplerBase:
    """Test :class:`sample_scf.base.SCFSamplerBase`."""

    def setup_class(self):
        self.cls = base.SCFSamplerBase
        self.cls_args = ()

        self.expected_rvs = {
            0: dict(r=0.548813503927, theta=1.021982822867 * u.rad, phi=0.548813503927 * u.rad),
            1: dict(r=0.548813503927, theta=1.021982822867 * u.rad, phi=0.548813503927 * u.rad),
            2: dict(
                r=[0.9670298390136, 0.5472322491757, 0.9726843599648, 0.7148159936743],
                theta=[0.603766487781, 1.023564077619, 0.598111966830, 0.855980333120] * u.rad,
                phi=[0.9670298390136, 0.547232249175, 0.9726843599648, 0.7148159936743] * u.rad,
            ),
        }

    # /def

    @pytest.fixture(autouse=True, scope="class")
    def sampler(self, potentials):
        """Set up r, theta, phi sampler."""
        sampler = self.cls(potentials, *self.cls_args)
        sampler._rsampler = testrvsampler()
        sampler._thetasampler = testrvsampler()
        sampler._phisampler = testrvsampler()

        return sampler

    # /def

    # ===============================================================

    def test_rsampler(self, sampler):
        """Test :meth:`sample_scf.base.SCFSamplerBase.rsampler`."""
        assert isinstance(sampler.rsampler, base.rv_continuous_modrvs)

    # /def

    def test_thetasampler(self, sampler):
        """Test :meth:`sample_scf.base.SCFSamplerBase.thetasampler`."""
        assert isinstance(sampler.thetasampler, base.rv_continuous_modrvs)

    # /def

    def test_phisampler(self, sampler):
        """Test :meth:`sample_scf.base.SCFSamplerBase.phisampler`."""
        assert isinstance(sampler.phisampler, base.rv_continuous_modrvs)

    # /def

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
        assert np.allclose(sampler.cdf(r, theta, phi), expected, atol=1e-16)

    # /def

    @pytest.mark.parametrize(
        "id, size, random",
        [
            (0, None, 0),
            (1, 1, 0),
            (2, 4, 4),
        ],
    )
    def test_rvs(self, sampler, id, size, random):
        """Test :meth:`sample_scf.base.SCFSamplerBase.rvs`."""
        samples = sampler.rvs(size=size, random_state=random)
        sce = coord.PhysicsSphericalRepresentation(**self.expected_rvs[id])

        assert_allclose(samples.r, sce.r, atol=1e-16)
        assert_allclose(samples.theta.value, sce.theta.value, atol=1e-16)
        assert_allclose(samples.phi.value, sce.phi.value, atol=1e-16)

    # /def

    # ===============================================================
    # Time Scaling Tests

    # ===============================================================
    # Image tests


# /class


class SCFSamplerTestBase(Test_SCFSamplerBase):
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

    # /def

    @pytest.fixture(autouse=True, scope="class")
    def sampler(self, potentials):
        """Set up r, theta, phi sampler."""
        sampler = self.cls(potentials, *self.cls_args)

        return sampler

    # /def


# /class

##############################################################################
# END
