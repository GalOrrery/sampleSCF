# -*- coding: utf-8 -*-

"""Tests for :mod:`sample_scf.sample_exact`."""

__all__ = [
    # "Test_SCFRSampler",
    # "Test_SCFThetaSampler",
    # "Test_SCFThetaSampler_of_r",
]


##############################################################################
# IMPORTS

# BUILT-IN
import pathlib
import time

# THIRD PARTY
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext
from galpy.df import isotropicHernquistdf
from galpy.potential import HernquistPotential, SCFPotential

# LOCAL
from sample_scf.sample_exact import SCFPhiSampler, SCFRSampler, SCFSampler, SCFThetaSampler
from sample_scf.utils import x_of_theta, zeta_of_r

##############################################################################
# PARAMETERS

# hernpot = TriaxialHernquistPotential(b=0.8, c=1.2)
hernpot = HernquistPotential()
coeffs = np.load(pathlib.Path(__file__).parent / "scf_coeffs.npz")
Acos, Asin = coeffs["Acos"], coeffs["Asin"]

pot = SCFPotential(Acos=Acos, Asin=Asin)
pot.turn_physical_off()

# r sampling
r = np.unique(np.concatenate([[0], np.geomspace(1e-7, 1e3, 100), [np.inf]]))
zeta = zeta_of_r(r)
m = [pot._mass(x) for x in r]
m[0] = 0
m[-1] = 1

# theta sampling
theta = np.linspace(-np.pi / 2, np.pi / 2, 30)

# phi sampling
phi = np.linspace(0, 2 * np.pi, 30)

##############################################################################
# CODE
##############################################################################


# class Test_SCFRSampler:
#     def setup_class(self):
#         self.sampler = SCFRSampler(m, r)
#         self.theory = isotropicHernquistdf(hernpot)
#
#     # /def
#
#     # ===============================================================
#     # Method Tests
#
#     def test___init__(self):
#         pass  # TODO!
#         # test if mgrid is SCFPotential
#
#     # ===============================================================
#     # Usage Tests
#
#     @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
#     def test_cdf_time_scaling(self, size):
#         """Test that the time scales as ~200 microseconds * size"""
#         tic = time.perf_counter()
#         self.sampler.cdf(np.linspace(0, 1e4, size))
#         toc = time.perf_counter()
#
#         assert (toc - tic) < 0.0003 * size  # 200 microseconds * linear scaling
#
#     # /def
#
#     @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
#     def test_rvs_time_scaling(self, size):
#         """Test that the time scales as ~300 microseconds * size"""
#         tic = time.perf_counter()
#         self.sampler.rvs(size=size)
#         toc = time.perf_counter()
#
#         assert (toc - tic) < 0.0004 * size  # 300 microseconds * linear scaling
#
#     # /def
#
#     # ----------------------------------------------------------------
#     # Image tests
#
#     @pytest.mark.mpl_image_compare(
#         baseline_dir="baseline_images",
#         hash_library="baseline_images/path_to_file.json",
#     )
#     def test_r_cdf_plot(self):
#         """Compare"""
#         fig = plt.figure(figsize=(10, 3))
#
#         ax = fig.add_subplot(
#             121,
#             title=r"$m(\leq r) / m_{tot}$",
#             xlabel="r",
#             ylabel=r"$m(\leq r) / m_{tot}$",
#         )
#         ax.semilogx(
#             r,
#             self.sampler.cdf(r),
#             marker="o",
#             ms=5,
#             c="k",
#             zorder=10,
#             label="CDF",
#         )
#         ax.axvline(0, c="tab:blue")
#         ax.axhline(self.sampler.cdf(0), c="tab:blue", label="r=0")
#         ax.axvline(1, c="tab:green")
#         ax.axhline(self.sampler.cdf(1), c="tab:green", label="r=1")
#         ax.axvline(1e2, c="tab:red")
#         ax.axhline(self.sampler.cdf(1e2), c="tab:red", label="r=100")
#
#         ax.set_xlim((1e-1, None))
#         ax.legend()
#
#         ax = fig.add_subplot(
#             122,
#             title=r"$m(\leq \zeta) / m_{tot}$",
#             xlabel=r"$\zeta$",
#             ylabel=r"$m(\leq \zeta) / m_{tot}$",
#         )
#         ax.plot(
#             zeta,
#             self.sampler.cdf(r),
#             marker="o",
#             ms=4,
#             c="k",
#             zorder=10,
#             label="CDF",
#         )
#         ax.axvline(zeta_of_r(0), c="tab:blue")
#         ax.axhline(self.sampler.cdf(0), c="tab:blue", label="r=0")
#         ax.axvline(zeta_of_r(1), c="tab:green")
#         ax.axhline(self.sampler.cdf(1), c="tab:green", label="r=1")
#         ax.axvline(zeta_of_r(1e2), c="tab:red")
#         ax.axhline(self.sampler.cdf(1e2), c="tab:red", label="r=100")
#
#         ax.legend()
#
#         fig.tight_layout()
#         return fig
#
#     # /def
#
#     @pytest.mark.mpl_image_compare(
#         baseline_dir="baseline_images",
#         hash_library="baseline_images/path_to_file.json",
#     )
#     def test_r_sampling_plot(self):
#         """Test sampling."""
#         with NumpyRNGContext(0):  # control the random numbers
#             sample = self.sampler.rvs(size=1000000)
#             sample = sample[sample < 1e4]
#
#             theory = self.theory.sample(n=1000000).r()
#             theory = theory[theory < 1e4]
#
#         fig = plt.figure(figsize=(10, 3))
#         ax = fig.add_subplot(121, title="SCF vs theory sampling", xlabel="r", ylabel="frequency")
#         _, bins, *_ = ax.hist(sample, bins=30, log=True, alpha=0.5, label="SCF sample")
#         # Comparing to expected
#         ax.hist(
#             theory,
#             bins=bins,
#             log=True,
#             alpha=0.5,
#             label="Hernquist theoretical",
#         )
#         ax.legend()
#         fig.tight_layout()
#
#         return fig
#
#     # /def
#
#
# # /class
#
#
# class Test_SCFThetaSampler:
#     def setup_class(self):
#         self.sampler = SCFThetaSampler(pot, r=1)
#         self.theory = isotropicHernquistdf(hernpot)
#
#     # /def
#
#     # ===============================================================
#     # Method Tests
#
#     def test___init__(self):
#         pass  # TODO!
#         # test if mgrid is SCFPotential
#
#     # ===============================================================
#     # Usage Tests
#
#     @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
#     def test_cdf_time_scaling(self, size):
#         """Test that the time scales as ~800 microseconds * size"""
#         tic = time.perf_counter()
#         self.sampler.cdf(np.linspace(0, np.pi, size))
#         toc = time.perf_counter()
#
#         assert (toc - tic) < 0.001 * size  # 800 microseconds * linear scaling
#
#     # /def
#
#     @pytest.mark.parametrize("size", [1, 10, 100])
#     def test_rvs_time_scaling(self, size):
#         """Test that the time scales as ~4 milliseconds * size"""
#         tic = time.perf_counter()
#         self.sampler.rvs(size=size)
#         toc = time.perf_counter()
#
#         assert (toc - tic) < 0.005 * size  # linear scaling
#
#     # /def
#
#     # ----------------------------------------------------------------
#     # Image tests
#
#     @pytest.mark.mpl_image_compare(
#         baseline_dir="baseline_images",
#         hash_library="baseline_images/path_to_file.json",
#     )
#     def test_theta_cdf_plot(self):
#         """Compare"""
#         fig = plt.figure(figsize=(10, 3))
#
#         ax = fig.add_subplot(
#             121,
#             title=r"$\Theta(\leq \theta; r=1)$",
#             xlabel=r"$\theta$",
#             ylabel=r"$\Theta(\leq \theta; r=1)$",
#         )
#         ax.plot(
#             theta,
#             self.sampler.cdf(theta),
#             marker="o",
#             ms=5,
#             c="k",
#             zorder=10,
#             label="CDF",
#         )
#         ax.legend(loc="lower right")
#
#         # Plotting CDF against x.
#         # This should be a straight line.
#         ax = fig.add_subplot(
#             122,
#             title=r"$\Theta(\leq \theta; r=1)$",
#             xlabel=r"$x=\cos\theta$",
#             ylabel=r"$\Theta(\leq \theta; r=1)$",
#         )
#         ax.plot(
#             x_of_theta(theta),
#             self.sampler.cdf(theta),
#             marker="o",
#             ms=4,
#             c="k",
#             zorder=10,
#             label="CDF",
#         )
#         ax.legend(loc="lower right")
#
#         fig.tight_layout()
#         return fig
#
#     # /def
#
#     @pytest.mark.mpl_image_compare(
#         baseline_dir="baseline_images",
#         hash_library="baseline_images/path_to_file.json",
#     )
#     def test_theta_sampling_plot(self):
#         """Test sampling."""
#         with NumpyRNGContext(0):  # control the random numbers
#             sample = self.sampler.rvs(size=1000)
#             sample = sample[sample < 1e4]
#
#             theory = self.theory.sample(n=1000).theta() - np.pi / 2
#             theory = theory[theory < 1e4]
#
#         fig = plt.figure(figsize=(7, 5))
#         ax = fig.add_subplot(
#             111,
#             title="SCF vs theory sampling",
#             xlabel="theta",
#             ylabel="frequency",
#         )
#         _, bins, *_ = ax.hist(sample, bins=30, log=True, alpha=0.5, label="SCF sample")
#         # Comparing to expected
#         ax.hist(
#             theory,
#             bins=bins,
#             log=True,
#             alpha=0.5,
#             label="Hernquist theoretical",
#         )
#         ax.legend(fontsize=12)
#         fig.tight_layout()
#
#         return fig
#
#     # /def
#
#
# # /class
#
#
# class Test_SCFThetaSampler_of_r:
#     def setup_class(self):
#         self.sampler = SCFThetaSampler(pot)
#         self.theory = isotropicHernquistdf(hernpot)
#
#     # /def
#
#     # ===============================================================
#     # Method Tests
#
#     def test___init__(self):
#         pass  # TODO!
#         # test if mgrid is SCFPotential
#
#     # ===============================================================
#     # Usage Tests
#
#     @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
#     def test_cdf_time_scaling(self, size):
#         """Test that the time scales as ~3 milliseconds * size"""
#         tic = time.perf_counter()
#         self.sampler.cdf(np.linspace(0, np.pi, size), r=1)
#         toc = time.perf_counter()
#
#         assert (toc - tic) < 0.003 * size  # 3 microseconds * linear scaling
#
#     # /def
#
#     @pytest.mark.parametrize("size", [1, 10, 100])
#     def test_rvs_time_scaling(self, size):
#         """Test that the time scales as ~4 milliseconds * size"""
#         tic = time.perf_counter()
#         self.sampler.rvs(size=size, r=1)
#         toc = time.perf_counter()
#
#         assert (toc - tic) < 0.04 * size  # linear scaling
#
#     # /def
#
#     def test_cdf_independent_of_r(self):
#         """The Hernquist potential CDF(theta) is r independent."""
#         expected = (x_of_theta(theta) + 1) / 2
#         # [-1, 1] -> [0, 1] with a
#
#         # assert np.allclose(self.sampler.cdf(theta, r=0), expected)  # FIXME!
#         assert np.allclose(self.sampler.cdf(theta, r=1), expected)
#         assert np.allclose(self.sampler.cdf(theta, r=2), expected)
#         # assert np.allclose(self.sampler.cdf(theta, r=np.inf), expected)  # FIXME!
#
#
# # ------------------------------------------------------------------------------
#
#
# class Test_SCFPhiSampler:
#     def setup_class(self):
#         self.sampler = SCFPhiSampler(pot, r=1, theta=np.pi / 3)
#         self.theory = isotropicHernquistdf(hernpot)
#
#     # /def
#
#     # ===============================================================
#     # Method Tests
#
#     def test___init__(self):
#         pass  # TODO!
#         # test if mgrid is SCFPotential
#
#     # ===============================================================
#     # Usage Tests
#
#     @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
#     def test_cdf_time_scaling(self, size):
#         """Test that the time scales as ~800 microseconds * size"""
#         tic = time.perf_counter()
#         self.sampler.cdf(np.linspace(0, 2 * np.pi, size))
#         toc = time.perf_counter()
#
#         assert (toc - tic) < 0.001 * size  # 800 microseconds * linear scaling
#
#     # /def
#
#     @pytest.mark.parametrize("size", [1, 10, 100])
#     def test_rvs_time_scaling(self, size):
#         """Test that the time scales as ~4 milliseconds * size"""
#         tic = time.perf_counter()
#         self.sampler.rvs(size=size)
#         toc = time.perf_counter()
#
#         assert (toc - tic) < 0.005 * size  # linear scaling
#
#     # /def
#
#     # ----------------------------------------------------------------
#     # Image tests
#
#     @pytest.mark.mpl_image_compare(
#         baseline_dir="baseline_images",
#         hash_library="baseline_images/path_to_file.json",
#     )
#     def test_phi_cdf_plot(self):
#         """Compare"""
#         fig = plt.figure(figsize=(10, 3))
#
#         ax = fig.add_subplot(
#             121,
#             title=r"$\Phi(\leq \phi; r=1)$",
#             xlabel=r"$\phi$",
#             ylabel=r"$\Phi(\leq \phi; r=1)$",
#         )
#         ax.plot(
#             phi,
#             self.sampler.cdf(phi),
#             marker="o",
#             ms=5,
#             c="k",
#             zorder=10,
#             label="CDF",
#         )
#         ax.legend(loc="lower right")
#
#         # Plotting CDF against x.
#         # This should be a straight line.
#         ax = fig.add_subplot(
#             122,
#             title=r"$\Phi(\leq \phi; r=1)$",
#             xlabel=r"$\phi/2\pi$",
#             ylabel=r"$\Phi(\leq \phi; r=1)$",
#         )
#         ax.plot(
#             phi / (2 * np.pi),
#             self.sampler.cdf(phi),
#             marker="o",
#             ms=4,
#             c="k",
#             zorder=10,
#             label="CDF",
#         )
#         ax.legend(loc="lower right")
#
#         fig.tight_layout()
#         return fig
#
#     # /def
#
#     @pytest.mark.mpl_image_compare(
#         baseline_dir="baseline_images",
#         hash_library="baseline_images/path_to_file.json",
#     )
#     def test_phi_sampling_plot(self):
#         """Test sampling."""
#         with NumpyRNGContext(0):  # control the random numbers
#             sample = self.sampler.rvs(size=1000)
#             sample = sample[sample < 1e4]
#
#             theory = self.theory.sample(n=1000).phi()
#             theory = theory[theory < 1e4]
#
#         fig = plt.figure(figsize=(7, 5))
#         ax = fig.add_subplot(
#             111,
#             title="SCF vs theory sampling",
#             xlabel="theta",
#             ylabel="frequency",
#         )
#         _, bins, *_ = ax.hist(sample, bins=30, log=True, alpha=0.5, label="SCF sample")
#         # Comparing to expected
#         ax.hist(
#             theory,
#             bins=bins,
#             log=True,
#             alpha=0.5,
#             label="Hernquist theoretical",
#         )
#         ax.legend(fontsize=12)
#         fig.tight_layout()
#
#         return fig
#
#     # /def
#
#
# # /class


##############################################################################
# END
