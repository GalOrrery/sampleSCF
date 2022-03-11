# -*- coding: utf-8 -*-

"""Tests for :mod:`sample_scf.exact`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext
from numpy.testing import assert_allclose

# LOCAL
from .common import phi_distributionTestBase, r_distributionTestBase, theta_distributionTestBase
from .test_base import SCFSamplerTestBase
from sample_scf import conftest, exact
from sample_scf.utils import difPls, r_of_zeta, thetaQls, x_of_theta

##############################################################################
# PARAMETERS

rgrid = np.concatenate(([0], np.geomspace(1e-1, 1e3, 29)))  # same shape as ↓
tgrid = np.linspace(-np.pi / 2, np.pi / 2, 30)
pgrid = np.linspace(0, 2 * np.pi, 30)


##############################################################################
# CODE
##############################################################################


class Test_SCFSampler(SCFSamplerTestBase):
    """Test :class:`sample_scf.exact.SCFSampler`."""

    def setup_class(self):
        super().setup_class(self)

        # sampler initialization
        self.cls = exact.SCFSampler
        self.cls_args = ()
        self.cls_kwargs = {}
        self.cls_pot_kw = conftest.cls_pot_kw

        # TODO! make sure these are right!
        self.expected_rvs = {
            0: dict(r=2.85831468, theta=1.473013568997 * u.rad, phi=4.49366731864 * u.rad),
            1: dict(r=2.85831468, theta=1.473013568997 * u.rad, phi=4.49366731864 * u.rad),
            2: dict(
                r=[59.156720319468995, 2.8424809956410684, 71.71466505619023, 5.471148006577435],
                theta=[0.365179487932, 1.476190768288, 0.3320725403573, 1.126711132015] * u.rad,
                phi=[4.383959499105, 1.3577303436664, 6.134113310024, 0.039145847961457] * u.rad,
            ),
        }

    # ===============================================================
    # Method Tests

    def test_init(self, potentials):
        kw = {**self.cls_kwargs, **self.cls_pot_kw.get(potentials, {})}
        instance = self.cls(potentials, *self.cls_args, **kw)

        assert isinstance(instance.rsampler, exact.r_distribution)
        assert isinstance(instance.thetasampler, exact.theta_distribution)
        assert isinstance(instance.phisampler, exact.phi_distribution)

    def test_rvs(self, sampler):
        """Test Random Variates Sampler."""

    # TODO! make sure these are correct
    @pytest.mark.parametrize(
        "r, theta, phi, expected",
        [
            (0.0, 0.0, 0.0, [0, 0.5, 0]),
            (1.0, 0.0, 0.0, [0.25, 0.5, 0]),
            # ([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [[0, 0.5, 0], [0.25, 0.5, 0]]),
        ],
    )
    def test_cdf(self, sampler, r, theta, phi, expected):
        """Test :meth:`sample_scf.base.SCFSamplerBase.cdf`."""
        assert np.allclose(sampler.cdf(r, theta, phi), expected, atol=1e-16)

    # ===============================================================
    # Plot Tests

    def test_exact_cdf_plot(self, sampler):
        """Plot cdf."""
        kw = dict(marker="o", ms=5, c="k", zorder=5, label="CDF")
        cdf = sampler.cdf(rgrid, tgrid, pgrid)

        fig = plt.figure(figsize=(15, 3))

        # r
        ax = fig.add_subplot(
            131,
            title=r"$m(\leq r) / m_{tot}$",
            xlabel="r",
            ylabel=r"$m(\leq r) / m_{tot}$",
        )
        ax.semilogx(rgrid, cdf[:, 0], **kw)

        # theta
        ax = fig.add_subplot(
            132,
            title=r"CDF($\theta$)",
            xlabel=r"$\theta$",
            ylabel=r"CDF($\theta$)",
        )
        ax.plot(tgrid, cdf[:, 1], **kw)

        # phi
        ax = fig.add_subplot(
            133,
            title=r"CDF($\phi$)",
            xlabel=r"$\phi$",
            ylabel=r"CDF($\phi$)",
        )
        ax.plot(pgrid, cdf[:, 2], **kw)

        return fig

    def test_exact_sampling_plot(self, sampler):
        """Plot sampling."""
        samples = sampler.rvs(size=int(1e3), random_state=3)

        fig = plt.figure(figsize=(15, 4))

        ax = fig.add_subplot(
            131,
            title=r"$m(\leq r) / m_{tot}$",
            xlabel="r",
            ylabel=r"$m(\leq r) / m_{tot}$",
        )
        ax.hist(samples.r.value[samples.r < 5e3], log=True, bins=50, density=True)

        ax = fig.add_subplot(
            132,
            title=r"CDF($\theta$)",
            xlabel=r"$\theta$",
            ylabel=r"CDF($\theta$)",
        )
        ax.hist(samples.theta.value, bins=50, density=True)

        ax = fig.add_subplot(133, title=r"CDF($\phi$)", xlabel=r"$\phi$", ylabel=r"CDF($\phi$)")
        ax.hist(samples.phi.value, bins=50)

        return fig


# ============================================================================


class Test_r_distribution(r_distributionTestBase):
    """Test :class:`sample_scf.exact.r_distribution`"""

    def setup_class(self):
        super().setup_class(self)

        # sampler initialization
        self.cls = exact.r_distribution
        self.cls_args = ()
        self.cls_kwargs = {}
        self.cls_pot_kw = conftest.cls_pot_kw

        # time-scale tests
        self.cdf_time_scale = 1e-2  # milliseconds
        self.rvs_time_scale = 1e-2  # milliseconds

    @pytest.fixture()
    def sampler(self, potentials):
        """Set up r, theta, or phi sampler."""
        kw = {**self.cls_kwargs, **self.cls_pot_kw.get(potentials, {})}
        sampler = self.cls(potentials, *self.cls_args, **kw)

        return sampler

    # ===============================================================
    # Method Tests

    @pytest.mark.skip("TODO!")
    def test_init(self):
        assert False
        # test if mgrid is SCFPotential

    # TODO! use hypothesis
    @pytest.mark.parametrize("r", np.random.default_rng(0).uniform(0, 1e4, 10))
    def test__cdf(self, sampler, r):
        """Test :meth:`sample_scf.exact.r_distribution._cdf`."""
        super().test__cdf(sampler, r)

        # expected
        mass = np.atleast_1d(sampler._potential._mass(r)) / sampler._mtot
        assert_allclose(sampler._cdf(r), mass)

    @pytest.mark.parametrize(
        "size, random, expected",
        [
            (None, 0, 2.85831468026),
            (1, 2, 1.9437661234293),
            ((3, 1), 4, [59.156720319468, 2.8424809956410, 71.71466505619]),
            ((3, 1), None, [59.156720319468, 2.8424809956410, 71.71466505619]),
        ],
    )
    def test_rvs(self, sampler, size, random, expected):
        """Test :meth:`sample_scf.exact.r_distribution.rvs`."""
        super().test_rvs(sampler, size, random, expected)

    # ===============================================================
    # Time Scaling Tests

    # TODO! generalize for subclasses
    @pytest.mark.parametrize("size", [1, 10, 100, 1000])  # rm 1e4
    def test_rvs_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        super().test_rvs_time_scaling(sampler, size)

    # ===============================================================
    # Image Tests

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline_images",
        # hash_library="baseline_images/path_to_file.json",  # TODO!
    )
    def test_exact_r_cdf_plot(self, sampler):
        fig = plt.figure(figsize=(10, 3))

        ax = fig.add_subplot(
            111,
            title=r"$m(\leq r) / m_{tot}$",
            xlabel="r",
            ylabel=r"$m(\leq r) / m_{tot}$",
        )
        kw = dict(marker="o", ms=5, c="k", zorder=5, label="CDF")
        ax.semilogx(rgrid, sampler.cdf(rgrid), **kw)
        ax.axvline(0.0, c="tab:blue")
        ax.axhline(sampler.cdf(0.0), c="tab:blue", label="r=0")
        ax.axvline(1.0, c="tab:green")
        ax.axhline(sampler.cdf(1.0), c="tab:green", label="r=1")
        ax.axvline(1e2, c="tab:red")
        ax.axhline(sampler.cdf(1e2), c="tab:red", label="r=100")

        ax.set_xlim((1e-1, None))
        ax.legend(loc="lower right")

        fig.tight_layout()
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline_images",
        # hash_library="baseline_images/path_to_file.json",
    )
    def test_exact_r_sampling_plot(self, sampler):
        """Test sampling."""
        with NumpyRNGContext(0):  # control the random numbers
            sample = sampler.rvs(size=int(1e3))
            sample = sample[sample < 1e4]

            theory = self.theory[sampler._potential].sample(n=int(1e6)).r()
            theory = theory[theory < 1e4 * u.kpc]

        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(121, title="SCF vs theory sampling", xlabel="r", ylabel="frequency")
        _, bins, *_ = ax.hist(sample, bins=30, log=True, alpha=0.5, label="SCF sample")
        # Comparing to expected
        ax.hist(
            theory.to_value(u.kpc),
            bins=bins,
            log=True,
            alpha=0.5,
            label="Hernquist theoretical",
        )
        ax.legend()
        fig.tight_layout()

        return fig


# ----------------------------------------------------------------------------


class Test_theta_distribution(theta_distributionTestBase):
    """Test :class:`sample_scf.exact.theta_distribution`."""

    def setup_class(self):
        super().setup_class(self)

        self.cls = exact.theta_distribution

        self.cdf_time_scale = 1e-3
        self.rvs_time_scale = 7e-2

    # ===============================================================
    # Method Tests

    # TODO! use hypothesis

    @pytest.mark.parametrize(
        "x, r",
        [
            *zip(
                np.random.default_rng(1).uniform(-1, 1, 10),
                r_of_zeta(np.random.default_rng(1).uniform(-1, 1, 10)),
            ),
        ],
    )
    def test__cdf(self, sampler, x, r):
        """Test :meth:`sample_scf.exact.theta_distribution._cdf`."""
        Qls = np.atleast_2d(thetaQls(sampler._potential, r))

        # basically a test it's Hernquist, only the first term matters
        if np.allclose(Qls[:, 1:], 0.0):
            assert_allclose(sampler._cdf(x, r=r), 0.5 * (x + 1.0))

        else:
            # TODO! a more robust test

            # l = 0
            term0 = 0.5 * (x + 1.0)  # (T,)
            # l = 1+ : non-symmetry
            factor = 1.0 / (2.0 * Qls[:, 0])  # (R,)
            term1p = np.sum(
                (Qls[None, :, 1:] * difPls(x, self._lmax - 1).T[:, None, :]).T,
                axis=0,
            )
            cdf = term0[None, :] + np.nan_to_num(factor[:, None] * term1p)  # (R, T)

            assert_allclose(sampler._cdf(x, r=r), cdf)

    @pytest.mark.parametrize("r", r_of_zeta(np.random.default_rng(0).uniform(-1, 1, 10)))
    def test__cdf_edge(self, sampler, r):
        """Test :meth:`sample_scf.exact.r_distribution._cdf`."""
        assert np.isclose(sampler._cdf(-1, r=r), 0.0, atol=1e-16)
        assert np.isclose(sampler._cdf(1, r=r), 1.0, atol=1e-16)

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
        """Test :meth:`sample_scf.exact.theta_distribution.cdf`."""
        self.test__cdf(sampler, x_of_theta(theta), r)

    @pytest.mark.skip("TODO!")
    def test__rvs(self):
        """Test :meth:`sample_scf.exact.theta_distribution._rvs`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_rvs(self):
        """Test :meth:`sample_scf.exact.theta_distribution.rvs`."""
        assert False

    # ===============================================================
    # Time Scaling Tests

    # TODO! generalize for subclasses
    @pytest.mark.parametrize("size", [1, 10, 100, 1000])  # rm 1e4
    def test_rvs_time_scaling(self, sampler, size):
        """Test that the time scales as X * size"""
        super().test_rvs_time_scaling(sampler, size)

    # ===============================================================
    # Image Tests

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline_images",
        # hash_library="baseline_images/path_to_file.json",  # TODO!
    )
    def test_exact_theta_cdf_plot(self, sampler):
        fig = plt.figure(figsize=(10, 3))

        # plot 1
        ax = fig.add_subplot(
            121,
            title=r"CDF($\theta$)",
            xlabel=r"$\theta$",
            ylabel=r"CDF($\theta$)",
        )
        kw = dict(marker="o", ms=5, c="k", zorder=5, label="CDF")
        ax.plot(tgrid, sampler.cdf(tgrid, r=10), **kw)
        ax.axvline(-np.pi / 2, c="tab:blue")
        ax.axhline(sampler.cdf(-np.pi / 2, r=10), c="tab:blue", label=r"$\theta=-\frac{\pi}{2}$")
        ax.axvline(0, c="tab:green")
        ax.axhline(sampler.cdf(0, r=10), c="tab:green", label=r"$\theta=0$")
        ax.axvline(np.pi / 2, c="tab:red")
        ax.axhline(sampler.cdf(np.pi / 2, r=10), c="tab:red", label=r"$\theta=\frac{\pi}{2}$")
        ax.legend(loc="lower right")

        # plot 2
        ax = fig.add_subplot(
            122,
            title=r"CDF($x$)",
            xlabel=r"x$",
            ylabel=r"CDF($x$)",
        )
        ax.plot(x_of_theta(tgrid), sampler.cdf(tgrid, r=10), **kw)
        ax.axvline(x_of_theta(-1), c="tab:blue")
        ax.axhline(sampler.cdf(-1, r=10), c="tab:blue", label=r"$\theta=-\frac{\pi}{2}$")
        ax.axvline(x_of_theta(0), c="tab:green")
        ax.axhline(sampler.cdf(0, r=10), c="tab:green", label=r"$\theta=0$")
        ax.axvline(x_of_theta(1), c="tab:red")
        ax.axhline(sampler.cdf(1, r=10), c="tab:red", label=r"$\theta=\frac{\pi}{2}$")
        ax.legend(loc="upper left")

        fig.tight_layout()
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline_images",
        # hash_library="baseline_images/path_to_file.json",
    )
    def test_exact_theta_sampling_plot(self, sampler):
        """Test sampling."""
        with NumpyRNGContext(0):  # control the random numbers
            sample = sampler.rvs(size=int(1e3), r=10)
            sample = sample[sample < 1e4]

            theory = self.theory[sampler._potential].sample(n=int(1e6)).theta()
            theory -= np.pi / 2 * u.rad

        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(
            121,
            title="SCF vs theory sampling",
            xlabel=r"$\theta$",
            ylabel="frequency",
        )
        _, bins, *_ = ax.hist(sample, bins=30, log=True, alpha=0.5, label="SCF sample")
        # Comparing to expected
        ax.hist(
            theory.to_value(u.rad),
            bins=bins,
            log=True,
            alpha=0.5,
            label="Hernquist theoretical",
        )
        ax.legend()
        fig.tight_layout()

        return fig


###############################################################################


class Test_phi_distribution(phi_distributionTestBase):
    """Test :class:`sample_scf.exact.phi_distribution`."""

    def setup_class(self):
        super().setup_class(self)

        self.cls = exact.phi_distribution

        self.cdf_time_scale = 3e-3
        self.rvs_time_scale = 3e-3

    # ===============================================================
    # Method Tests

    @pytest.mark.skip("TODO!")
    def test__cdf(self):
        """Test :meth:`sample_scf.exactolated.phi_distribution._cdf`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_cdf(self):
        """Test :meth:`sample_scf.exactolated.phi_distribution.cdf`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test__rvs(self):
        """Test :meth:`sample_scf.exactolated.phi_distribution._rvs`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_rvs(self):
        """Test :meth:`sample_scf.exactolated.phi_distribution.rvs`."""
        assert False

    # ===============================================================
    # Image Tests

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline_images",
        # hash_library="baseline_images/path_to_file.json",  # TODO!
    )
    def test_exact_phi_cdf_plot(self, sampler):
        fig = plt.figure(figsize=(5, 3))

        ax = fig.add_subplot(
            111,
            title=r"CDF($\phi$)",
            xlabel=r"$\phi$",
            ylabel=r"CDF($\phi$)",
        )
        kw = dict(marker="o", ms=5, c="k", zorder=5, label="CDF")
        ax.plot(pgrid, sampler.cdf(pgrid, r=10, theta=np.pi / 6), **kw)
        ax.axvline(0, c="tab:blue")
        ax.axhline(sampler.cdf(0, r=10, theta=np.pi / 6), c="tab:blue", label=r"$\phi=0$")
        ax.axvline(np.pi, c="tab:green")
        ax.axhline(sampler.cdf(np.pi, r=10, theta=np.pi / 6), c="tab:green", label=r"$\phi=\pi$")
        ax.axvline(2 * np.pi, c="tab:red")
        ax.axhline(sampler.cdf(2 * np.pi, r=10, theta=np.pi / 6), c="tab:red", label=r"$\phi=2\pi$")
        ax.legend(loc="lower right")

        fig.tight_layout()
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline_images",
        # hash_library="baseline_images/path_to_file.json",
    )
    def test_exact_phi_sampling_plot(self, sampler):
        """Test sampling."""
        with NumpyRNGContext(0):  # control the random numbers
            sample = sampler.rvs(size=int(1e3), r=10, theta=np.pi / 6)
            sample = sample[sample < 1e4]

            theory = self.theory[sampler._potential].sample(n=int(1e3)).phi()

        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(
            121,
            title="SCF vs theory sampling",
            xlabel=r"$\phi$",
            ylabel="frequency",
        )
        _, bins, *_ = ax.hist(sample, bins=30, log=True, alpha=0.5, label="SCF sample")
        # Comparing to expected
        ax.hist(
            theory.to_value(u.rad),
            bins=bins,
            log=True,
            alpha=0.5,
            label="Hernquist theoretical",
        )
        ax.legend()
        fig.tight_layout()

        return fig
