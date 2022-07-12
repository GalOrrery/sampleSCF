# -*- coding: utf-8 -*-

"""Tests for :mod:`sample_scf.exact.core`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import pytest
from sampler_scf.base_multivariate import SCFSamplerBase

# LOCAL
from .test_base_multivariate import BaseTest_SCFSamplerBase
from sample_scf import ExactSCFSampler

##############################################################################
# CODE
##############################################################################


class Test_ExactSCFSampler(BaseTest_SCFSamplerBase):
    """Test :class:`sample_scf.exact.ExactSCFSampler`."""

    @pytest.fixture(scope="class")
    def rv_cls(self):
        return ExactSCFSampler

    def setup_class(self):
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

    def test_init_attrs(self, sampler):
        super().test_init_attrs(sampler)

        hasattr(sampler, "_sampler")
        assert sampler._sampler is None or isinstance(sampler._sampler, SCFSamplerBase)

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
        """Test :meth:`sample_scf.base_multivariate.SCFSamplerBase.cdf`."""
        super().test_cdf(sampler, r, theta, phi, expected)

    @pytest.mark.skip("TODO!")
    def test_rvs(self, sampler):
        """Test Random Variates Sampler."""

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
