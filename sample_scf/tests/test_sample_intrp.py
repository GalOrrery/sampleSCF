# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.sample_intrp`."""


##############################################################################
# IMPORTS

# BUILT-IN
import time

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

# LOCAL
from .test_base import SCFSamplerTestBase
from .test_base import Test_RVPotential as RVPotentialTest
from sample_scf import sample_intrp
from sample_scf.utils import phiRSms, r_of_zeta, thetaQls, x_of_theta, zeta_of_r

##############################################################################
# PARAMETERS

rgrid = np.concatenate(([0], np.geomspace(1e-1, 1e3, 100)))
tgrid = np.linspace(-np.pi / 2, np.pi / 2, 30)
pgrid = np.linspace(0, 2 * np.pi, 30)


##############################################################################
# TESTS
##############################################################################


class Test_SCFSampler(SCFSamplerTestBase):
    """Test :class:`sample_scf.sample_intrp.SCFSampler`."""

    def setup_class(self):

        self.cls = sample_intrp.SCFSampler
        self.cls_args = (rgrid, tgrid, pgrid)
        self.cls_kwargs = {}

        self.expected_rvs = {
            0: dict(r=2.8583146808697, theta=1.473013568997 * u.rad, phi=3.4482969442579 * u.rad),
            1: dict(r=2.8583146808697, theta=1.473013568997 * u.rad, phi=3.4482969442579 * u.rad),
            2: dict(
                r=[59.15672032022, 2.842480998054, 71.71466505664, 5.471148006362],
                theta=[0.36517953566424, 1.4761907683040, 0.33207251545636, 1.1267111320704]
                * u.rad,
                phi=[6.076027676095, 3.438361627636, 6.11155607905, 4.491321348792] * u.rad,
            ),
        }

    # /def
    
    # ===============================================================
    # Method Tests

    # TODO! make sure these are correct
    @pytest.mark.parametrize(
        "r, theta, phi, expected",
        [
            (0, 0, 0, [0, 0.5, 0]),
            (1, 0, 0, [0.25, 0.5, 0]),
            ([0, 1], [0, 0], [0, 0], [[0, 0.5, 0], [0.25, 0.5, 0]]),
        ],
    )
    def test_cdf(self, sampler, r, theta, phi, expected):
        """Test :meth:`sample_scf.base.SCFSamplerBase.cdf`."""
        assert np.allclose(sampler.cdf(r, theta, phi), expected, atol=1e-16)

    # /def


# /class

##############################################################################


class InterpRVPotentialTest(RVPotentialTest):
    def test___init__(self, sampler):
        """Test initialization."""
        potential = sampler._potential

        assert hasattr(sampler, "_spl_cdf")
        assert hasattr(sampler, "_spl_ppf")

        # good
        newsampler = self.cls(potential, *self.cls_args)

        cdfk = sampler._spl_cdf.get_knots()
        ncdfk = newsampler._spl_cdf.get_knots()
        if isinstance(cdfk, np.ndarray):  # 1D splines
            assert_allclose(ncdfk, cdfk, atol=1e-16)
        else:  # 2D and 3D splines
            for k, nk in zip(cdfk, ncdfk):
                assert_allclose(k, nk, atol=1e-16)

        ppfk = sampler._spl_ppf.get_knots()
        nppfk = newsampler._spl_ppf.get_knots()
        if isinstance(ppfk, np.ndarray):  # 1D splines
            assert_allclose(nppfk, ppfk, atol=1e-16)
        else:  # 2D and 3D splines
            for k, nk in zip(ppfk, nppfk):
                assert_allclose(k, nk, atol=1e-16)

        # bad
        with pytest.raises(TypeError, match="SCFPotential"):
            self.cls(None, *self.cls_args)

    # /def


# /class


# ----------------------------------------------------------------------------


class Test_SCFRSampler(InterpRVPotentialTest):
    """Test :class:`sample_scf.`"""

    def setup_class(self):
        self.cls = sample_intrp.SCFRSampler
        self.cls_args = (rgrid,)

        self.cdf_time_scale = 6e-4  # milliseconds
        self.rvs_time_scale = 2e-4  # milliseconds

    # /def

    # ===============================================================
    # Method Tests

    def test___init__(self, sampler):
        """Test initialization."""
        super().test___init__(sampler)

        # TODO! test mgrid endpoints, cdf, and ppf

    # /def

    # TODO! use hypothesis
    @pytest.mark.parametrize("r", np.random.default_rng(0).uniform(0, 1e4, 10))
    def test__cdf(self, sampler, r):
        """Test :meth:`sample_scf.sample_intrp.SCFRSampler._cdf`."""
        # expected
        assert_allclose(sampler._cdf(r), sampler._spl_cdf(zeta_of_r(r)))

        # args and kwargs don't matter
        assert_allclose(sampler._cdf(r), sampler._cdf(r, 10, test=14))

    # /def

    def test__cdf_edge(self, sampler):
        """Test :meth:`sample_scf.sample_intrp.SCFRSampler._cdf`."""
        assert np.isclose(sampler._cdf(0), 0.0, 1e-20)
        assert np.isclose(sampler._cdf(np.inf), 1.0, 1e-20)

    # /def

    # TODO! use hypothesis
    @pytest.mark.parametrize("q", np.random.default_rng(0).uniform(0, 1, 10))
    def test__ppf(self, sampler, q):
        """Test :meth:`sample_scf.sample_intrp.SCFRSampler._ppf`."""
        # expected
        assert_allclose(sampler._ppf(q), r_of_zeta(sampler._spl_ppf(q)))

        # args and kwargs don't matter
        assert_allclose(sampler._ppf(q), sampler._ppf(q, 10, test=14))

    # /def

    @pytest.mark.parametrize(
        "size, random, expected",
        [
            (None, 0, 2.85831468),
            (1, 2, 1.94376617),
            ((3, 1), 4, (59.15672032, 2.842481, 71.71466506)),
            ((3, 1), None, (59.15672032, 2.842481, 71.71466506)),
        ],
    )
    def test_rvs(self, sampler, size, random, expected):
        """Test :meth:`sample_scf.sample_intrp.SCFRSampler.rvs`."""
        super().test_rvs(sampler, size, random, expected)

    # /def

    # ===============================================================
    # Time Scaling Tests

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_cdf_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        x = np.linspace(0, 1e4, size)
        tic = time.perf_counter()
        sampler.cdf(x)
        toc = time.perf_counter()

        assert (toc - tic) < self.cdf_time_scale * size  # linear scaling

    # /def

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_rvs_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        tic = time.perf_counter()
        sampler.rvs(size=size)
        toc = time.perf_counter()

        assert (toc - tic) < self.rvs_time_scale * size  # linear scaling

    # /def

    # ===============================================================
    # Usage Tests


# /class

# ----------------------------------------------------------------------------


class Test_SCFThetaSampler(InterpRVPotentialTest):
    """Test :class:`sample_scf.sample_intrp.SCFThetaSampler`."""

    def setup_class(self):
        self.cls = sample_intrp.SCFThetaSampler
        self.cls_args = (rgrid, tgrid)

        self.cdf_time_scale = 3e-4
        self.rvs_time_scale = 6e-4

    # /def

    # ===============================================================
    # Method Tests

    def test___init__(self, sampler):
        """Test initialization."""
        super().test___init__(sampler)

        # a shape mismatch
        Qls = thetaQls(sampler._potential, rgrid[1:-1])
        with pytest.raises(ValueError, match="Qls must be shape"):
            sampler.__class__(sampler._potential, rgrid, tgrid, Qls=Qls)

    # /def

    # TODO! use hypothesis
    @pytest.mark.parametrize(
        "x, zeta",
        [
            *zip(
                np.random.default_rng(0).uniform(-1, 1, 10),
                np.random.default_rng(1).uniform(-1, 1, 10),
            ),
        ],
    )
    def test__cdf(self, sampler, x, zeta):
        """Test :meth:`sample_scf.sample_intrp.SCFThetaSampler._cdf`."""
        # expected
        assert_allclose(sampler._cdf(x, zeta=zeta), sampler._spl_cdf(zeta, x, grid=False))

        # args and kwargs don't matter
        assert_allclose(sampler._cdf(x, zeta=zeta), sampler._cdf(x, 10, zeta=zeta, test=14))

    # /def

    @pytest.mark.parametrize("zeta", np.random.default_rng(0).uniform(-1, 1, 10))
    def test__cdf_edge(self, sampler, zeta):
        """Test :meth:`sample_scf.sample_intrp.SCFRSampler._cdf`."""
        assert np.isclose(sampler._cdf(-1, zeta=zeta), 0.0, atol=1e-16)
        assert np.isclose(sampler._cdf(1, zeta=zeta), 1.0, atol=1e-16)

    # /def

    @pytest.mark.parametrize(
        "theta, r",
        [
            *zip(
                np.random.default_rng(0).uniform(-np.pi / 2, np.pi / 2, 10),
                np.random.default_rng(1).uniform(0, 1e4, 10),
            ),
        ],
    )
    def test_cdf(self, sampler, theta, r):
        """Test :meth:`sample_scf.sample_intrp.SCFThetaSampler.cdf`."""
        assert_allclose(
            sampler.cdf(theta, r),
            sampler._spl_cdf(zeta_of_r(r), x_of_theta(u.Quantity(theta, u.rad)), grid=False),
        )

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
    # Time Scaling Tests

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_cdf_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        x = np.linspace(-np.pi / 2, np.pi / 2, size)
        tic = time.perf_counter()
        sampler.cdf(x, r=10)
        toc = time.perf_counter()

        assert (toc - tic) < self.cdf_time_scale * size  # linear scaling

    # /def

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_rvs_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        tic = time.perf_counter()
        sampler.rvs(size=size, r=10)
        toc = time.perf_counter()

        assert (toc - tic) < self.rvs_time_scale * size  # linear scaling

    # /def

    # ===============================================================
    # Usage Tests


# /class

# ----------------------------------------------------------------------------


class Test_SCFPhiSampler(InterpRVPotentialTest):
    """Test :class:`sample_scf.sample_intrp.SCFPhiSampler`."""

    def setup_class(self):
        self.cls = sample_intrp.SCFPhiSampler
        self.cls_args = (rgrid, tgrid, pgrid)

        self.cdf_time_scale = 12e-4
        self.rvs_time_scale = 12e-4

    # /def

    # ===============================================================
    # Method Tests

    def test___init__(self, sampler):
        """Test :meth:`sample_scf.sample_intrp.SCFPhiSampler._cdf`."""
        # super().test___init__(sampler)  # doesn't work  TODO!

        # a shape mismatch
        RSms = phiRSms(sampler._potential, rgrid[1:-1], tgrid[1:-1])
        with pytest.raises(ValueError, match="Rm, Sm must be shape"):
            sampler.__class__(sampler._potential, rgrid, tgrid, pgrid, RSms=RSms)

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
    # Time Scaling Tests

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_cdf_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        x = np.linspace(0, 2 * np.pi, size)
        tic = time.perf_counter()
        sampler.cdf(x, r=10, theta=np.pi / 6)
        toc = time.perf_counter()

        assert (toc - tic) < self.cdf_time_scale * size  # linear scaling

    # /def

    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_rvs_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        tic = time.perf_counter()
        sampler.rvs(size=size, r=10, theta=np.pi / 6)
        toc = time.perf_counter()

        assert (toc - tic) < self.rvs_time_scale * size  # linear scaling

    # /def

    # ===============================================================
    # Usage Tests


# /class

##############################################################################
# END
