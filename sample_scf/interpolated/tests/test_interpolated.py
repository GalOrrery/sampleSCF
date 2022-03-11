# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.interpolated`."""


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
from .test_base import RVPotentialTest, SCFSamplerTestBase
from sample_scf import conftest, interpolated
from sample_scf.utils import phiRSms, r_of_zeta, thetaQls, x_of_theta, zeta_of_r

##############################################################################
# PARAMETERS

rgrid = np.concatenate(([0], np.geomspace(1e-1, 1e3, 100), [np.inf]))
tgrid = np.linspace(-np.pi / 2, np.pi / 2, 30)
pgrid = np.linspace(0, 2 * np.pi, 30)


##############################################################################
# TESTS
##############################################################################


class Test_SCFSampler(SCFSamplerTestBase):
    """Test :class:`sample_scf.interpolated.SCFSampler`."""

    def setup_class(self):
        super().setup_class(self)

        self.cls = interpolated.SCFSampler
        self.cls_args = (rgrid, tgrid, pgrid)
        self.cls_kwargs = {}
        self.cls_pot_kw = {}

        # TODO! make sure these are right!
        self.expected_rvs = {
            0: dict(r=2.8473287899985, theta=1.473013568997 * u.rad, phi=3.4482969442579 * u.rad),
            1: dict(r=2.8473287899985, theta=1.473013568997 * u.rad, phi=3.4482969442579 * u.rad),
            2: dict(
                r=[55.79997672576021, 2.831600636133138, 66.85343958872159, 5.435971037191061],
                theta=[0.3651795356642, 1.476190768304, 0.3320725154563, 1.126711132070] * u.rad,
                phi=[6.076027676095, 3.438361627636, 6.11155607905, 4.491321348792] * u.rad,
            ),
        }

    # ===============================================================
    # Method Tests

    # TODO! make sure these are correct
    @pytest.mark.parametrize(
        "r, theta, phi, expected",
        [
            (0, 0, 0, [0, 0.5, 0]),
            (1, 0, 0, [0.2505, 0.5, 0]),
            ([0, 1], [0, 0], [0, 0], [[0, 0.5, 0], [0.2505, 0.5, 0]]),
        ],
    )
    def test_cdf(self, sampler, r, theta, phi, expected):
        """Test :meth:`sample_scf.base.SCFSamplerBase.cdf`."""
        assert np.allclose(sampler.cdf(r, theta, phi), expected, atol=1e-16)

    # ===============================================================
    # Plot Tests

    @pytest.mark.skip("TODO!")
    def test_interp_cdf_plot(self):
        assert False

    @pytest.mark.skip("TODO!")
    def test_interp_sampling_plot(self):
        assert False


##############################################################################


class InterpRVPotentialTest(RVPotentialTest):
    def test_init(self, sampler):
        """Test initialization."""
        potential = sampler._potential

        assert hasattr(sampler, "_spl_cdf")
        assert hasattr(sampler, "_spl_ppf")

        # good
        newsampler = self.cls(potential, *self.cls_args)

        # compare that the knots are the same when initializing a second time
        # ie that the splines are stable
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


# ----------------------------------------------------------------------------


class Test_r_distribution(r_distributionTestBase, InterpRVPotentialTest):
    """Test :class:`sample_scf.sample_interp.r_distribution`"""

    def setup_class(self):
        super().setup_class(self)

        self.cls = interpolated.r_distribution
        self.cls_args = (rgrid,)
        self.cls_kwargs = {}
        self.cls_pot_kw = {}

        self.cdf_time_scale = 6e-4  # milliseconds
        self.rvs_time_scale = 2e-4  # milliseconds

    # ===============================================================
    # Method Tests

    def test_init(self, sampler):
        """Test initialization."""
        super().test_init(sampler)

        # TODO! test mgrid endpoints, cdf, and ppf

    # TODO! use hypothesis
    @pytest.mark.parametrize("r", np.random.default_rng(0).uniform(0, 1e4, 10))
    def test__cdf(self, sampler, r):
        """Test :meth:`sample_scf.interpolated.r_distribution._cdf`."""
        super().test__cdf(sampler, r)

        # expected
        assert_allclose(sampler._cdf(r), sampler._spl_cdf(zeta_of_r(r)))

    # TODO! use hypothesis
    @pytest.mark.parametrize("q", np.random.default_rng(0).uniform(0, 1, 10))
    def test__ppf(self, sampler, q):
        """Test :meth:`sample_scf.interpolated.r_distribution._ppf`."""
        # expected
        assert_allclose(sampler._ppf(q), r_of_zeta(sampler._spl_ppf(q)))

        # args and kwargs don't matter
        assert_allclose(sampler._ppf(q), sampler._ppf(q, 10, test=14))

    @pytest.mark.parametrize(
        "size, random, expected",
        [
            (None, 0, 2.84732879),
            (1, 2, 1.938060987),
            ((3, 1), 4, (55.79997672, 2.831600636, 66.85343958)),
            ((3, 1), None, (55.79997672, 2.831600636, 66.85343958)),
        ],
    )
    def test_rvs(self, sampler, size, random, expected):
        """Test :meth:`sample_scf.interpolated.r_distribution.rvs`."""
        super().test_rvs(sampler, size, random, expected)

    # ===============================================================
    # Image Tests

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline_images",
        # hash_library="baseline_images/path_to_file.json",  # TODO!
    )
    def test_interp_r_cdf_plot(self, sampler):
        fig = plt.figure(figsize=(10, 3))

        ax = fig.add_subplot(
            121,
            title=r"$m(\leq r) / m_{tot}$",
            xlabel="r",
            ylabel=r"$m(\leq r) / m_{tot}$",
        )
        kw = dict(marker="o", ms=5, c="k", zorder=5, label="CDF")
        ax.semilogx(rgrid, sampler.cdf(rgrid), **kw)
        ax.axvline(0, c="tab:blue")
        ax.axhline(sampler.cdf(0), c="tab:blue", label="r=0")
        ax.axvline(1, c="tab:green")
        ax.axhline(sampler.cdf(1), c="tab:green", label="r=1")
        ax.axvline(1e2, c="tab:red")
        ax.axhline(sampler.cdf(1e2), c="tab:red", label="r=100")

        ax.set_xlim((1e-1, None))
        ax.legend(loc="lower right")

        ax = fig.add_subplot(
            122,
            title=r"$m(\leq \zeta) / m_{tot}$",
            xlabel=r"$\zeta$",
            ylabel=r"$m(\leq \zeta) / m_{tot}$",
        )
        ax.plot(zeta_of_r(rgrid), sampler.cdf(rgrid), **kw)
        ax.axvline(zeta_of_r(0), c="tab:blue")
        ax.axhline(sampler.cdf(0), c="tab:blue", label="r=0")
        ax.axvline(zeta_of_r(1), c="tab:green")
        ax.axhline(sampler.cdf(1), c="tab:green", label="r=1")
        ax.axvline(zeta_of_r(1e2), c="tab:red")
        ax.axhline(sampler.cdf(1e2), c="tab:red", label="r=100")
        ax.legend(loc="upper left")

        fig.tight_layout()
        return fig

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline_images",
        # hash_library="baseline_images/path_to_file.json",
    )
    def test_interp_r_sampling_plot(self, sampler):
        """Test sampling."""
        with NumpyRNGContext(0):  # control the random numbers
            sample = sampler.rvs(size=int(1e6))
            sample = sample[sample < 1e4]

            theory = self.theory[sampler._potential].sample(n=int(1e6)).r()
            theory = theory[theory < 1e4 * u.kpc]

        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(121, title="SCF vs theory sampling", xlabel="r", ylabel="frequency")
        _, bins, *_ = ax.hist(
            sample, bins=30, log=True, alpha=0.5, label="SCF sample", c="tab:blue"
        )
        # Comparing to expected
        ax.hist(
            theory.to_value(u.kpc),
            bins=bins,
            log=True,
            edgecolor="black",
            linewidth=1.2,
            fc=(1, 0, 0, 0.0),
            label="Theoretical",
        )
        ax.legend()
        fig.tight_layout()

        return fig


# ----------------------------------------------------------------------------


class Test_theta_distribution(theta_distributionTestBase, InterpRVPotentialTest):
    """Test :class:`sample_scf.interpolated.theta_distribution`."""

    def setup_class(self):
        super().setup_class(self)

        self.cls = interpolated.theta_distribution
        self.cls_args = (rgrid, tgrid)

        self.cdf_time_scale = 3e-4
        self.rvs_time_scale = 6e-4

    # ===============================================================
    # Method Tests

    def test_init(self, sampler):
        """Test initialization."""
        super().test_init(sampler)

        # a shape mismatch
        Qls = thetaQls(sampler._potential, rgrid[1:-1])
        with pytest.raises(ValueError, match="Qls must be shape"):
            sampler.__class__(sampler._potential, rgrid, tgrid, Qls=Qls)

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
        """Test :meth:`sample_scf.interpolated.theta_distribution._cdf`."""
        # expected
        assert_allclose(sampler._cdf(x, zeta=zeta), sampler._spl_cdf(zeta, x, grid=False))

        # args and kwargs don't matter
        assert_allclose(sampler._cdf(x, zeta=zeta), sampler._cdf(x, 10, zeta=zeta, test=14))

    @pytest.mark.parametrize("zeta", np.random.default_rng(0).uniform(-1, 1, 10))
    def test__cdf_edge(self, sampler, zeta):
        """Test :meth:`sample_scf.interpolated.r_distribution._cdf`."""
        assert np.isclose(sampler._cdf(-1, zeta=zeta), 0.0, atol=1e-16)
        assert np.isclose(sampler._cdf(1, zeta=zeta), 1.0, atol=1e-16)

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
        """Test :meth:`sample_scf.interpolated.theta_distribution.cdf`."""
        assert_allclose(
            sampler.cdf(theta, r),
            sampler._spl_cdf(zeta_of_r(r), x_of_theta(u.Quantity(theta, u.rad)), grid=False),
        )

    @pytest.mark.skip("TODO!")
    def test__ppf(self):
        """Test :meth:`sample_scf.interpolated.theta_distribution._ppf`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test__rvs(self):
        """Test :meth:`sample_scf.interpolated.theta_distribution._rvs`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_rvs(self):
        """Test :meth:`sample_scf.interpolated.theta_distribution.rvs`."""
        assert False

    # ===============================================================
    # Image Tests

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline_images",
        # hash_library="baseline_images/path_to_file.json",  # TODO!
    )
    def test_interp_theta_cdf_plot(self, sampler):
        fig = plt.figure(figsize=(10, 3))

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
    def test_interp_theta_sampling_plot(self, sampler):
        """Test sampling."""
        with NumpyRNGContext(0):  # control the random numbers
            sample = sampler.rvs(size=int(1e6), r=10)
            sample = sample[sample < 1e4]

            theory = self.theory[sampler._potential].sample(n=int(1e6)).theta()
            theory -= np.pi / 2 * u.rad  # adjust range back

        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(
            121,
            title="SCF vs theory sampling",
            xlabel=r"$\theta$",
            ylabel="frequency",
        )
        _, bins, *_ = ax.hist(
            sample, bins=30, log=True, label="SCF sample", color="tab:blue", alpha=0.5
        )
        # Comparing to expected
        ax.hist(
            theory.to_value(u.rad),
            bins=bins,
            log=True,
            edgecolor="black",
            linewidth=1.2,
            fc=(1, 0, 0, 0.0),
            label="Theoretical",
        )
        ax.legend()
        fig.tight_layout()

        return fig


# ----------------------------------------------------------------------------


class Test_phi_distribution(phi_distributionTestBase, InterpRVPotentialTest):
    """Test :class:`sample_scf.interpolated.phi_distribution`."""

    def setup_class(self):
        super().setup_class(self)

        self.cls = interpolated.phi_distribution
        self.cls_args = (rgrid, tgrid, pgrid)

        self.cdf_time_scale = 12e-4
        self.rvs_time_scale = 12e-4

    # ===============================================================
    # Method Tests

    def test_init(self, sampler):
        """Test :meth:`sample_scf.interpolated.phi_distribution._cdf`."""
        # super().test_init(sampler)  # doesn't work  TODO!

        # a shape mismatch
        RSms = phiRSms(sampler._potential, rgrid[1:-1], tgrid[1:-1], warn=False)
        with pytest.raises(ValueError, match="Rm, Sm must be shape"):
            sampler.__class__(sampler._potential, rgrid, tgrid, pgrid, RSms=RSms)

    @pytest.mark.skip("TODO!")
    def test__cdf(self):
        """Test :meth:`sample_scf.interpolated.phi_distribution._cdf`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_cdf(self):
        """Test :meth:`sample_scf.interpolated.phi_distribution.cdf`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test__ppf(self):
        """Test :meth:`sample_scf.interpolated.phi_distribution._ppf`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test__rvs(self):
        """Test :meth:`sample_scf.interpolated.phi_distribution._rvs`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_rvs(self):
        """Test :meth:`sample_scf.interpolated.phi_distribution.rvs`."""
        assert False

    # ===============================================================
    # Image Tests

    @pytest.mark.mpl_image_compare(
        baseline_dir="baseline_images",
        # hash_library="baseline_images/path_to_file.json",  # TODO!
    )
    def test_interp_phi_cdf_plot(self, sampler):
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
    def test_interp_phi_sampling_plot(self, sampler):
        """Test sampling."""
        with NumpyRNGContext(0):  # control the random numbers
            sample = sampler.rvs(size=int(1e6), r=10, theta=np.pi / 6)
            sample = sample[sample < 1e4]

            theory = self.theory[sampler._potential].sample(n=int(1e6)).phi()

        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(
            121,
            title="SCF vs theory sampling",
            xlabel=r"$\phi$",
            ylabel="frequency",
        )
        _, bins, *_ = ax.hist(
            sample, bins=30, log=True, alpha=0.5, c="tab:blue", label="SCF sample"
        )
        # Comparing to expected
        ax.hist(
            theory.to_value(u.rad),
            bins=bins,
            log=True,
            edgecolor="black",
            linewidth=1.2,
            fc=(1, 0, 0, 0.0),
            label="Theoretical",
        )
        ax.legend()
        fig.tight_layout()

        return fig
