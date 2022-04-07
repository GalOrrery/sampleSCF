# -*- coding: utf-8 -*-
# mypy: ignore-errors

"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

# STDLIB
import copy
import os

# THIRD PARTY
import numpy as np
import pytest
from astropy.utils.data import get_pkg_data_filename, get_pkg_data_path
from galpy.df import isotropicHernquistdf, isotropicNFWdf, osipkovmerrittNFWdf
from galpy.potential import (
    HernquistPotential,
    NFWPotential,
    SCFPotential,
    TriaxialNFWPotential,
    scf_compute_coeffs_axi,
)

try:
    # THIRD PARTY
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False

# ============================================================================
# Configuration


def pytest_configure(config):
    """Configure Pytest with Astropy.

    Parameters
    ----------
    config : pytest configuration

    """
    if ASTROPY_HEADER:

        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the list of
        # packages for which version numbers are displayed when running the tests.
        PYTEST_HEADER_MODULES.pop("Pandas", None)
        PYTEST_HEADER_MODULES["scikit-image"] = "skimage"

        # LOCAL
        from . import __version__

        packagename = os.path.basename(os.path.dirname(__file__))
        TESTED_VERSIONS[packagename] = __version__


# ============================================================================
# Fixtures

# Hernquist
hernquist_potential = HernquistPotential()
hernquist_potential.turn_physical_on()
hernquist_df = isotropicHernquistdf(hernquist_potential)

Acos = np.zeros((5, 6, 6))
Acos[0, 0, 0] = 1
_hernquist_scf_potential = SCFPotential(Acos=Acos)
_hernquist_scf_potential.turn_physical_on()


# # NFW
# nfw_potential = NFWPotential(normalize=1)
# nfw_potential.turn_physical_on()
# nfw_df = isotropicNFWdf(nfw_potential, rmax=1e4)
# # FIXME! load this up as a test data file
# fpath = get_pkg_data_path("tests/data/nfw.npz", package="sample_scf")
# try:
#     data = np.load(fpath)
# except FileNotFoundError:
#     a_scf = 80
#     Acos, Asin = scf_compute_coeffs_axi(nfw_potential.dens, N=40, L=30, a=a_scf)
#     np.savez(fpath, Acos=Acos, Asin=Asin, a_scf=a_scf)
# else:
#     data = np.load(fpath, allow_pickle=True)
#     Acos = copy.deepcopy(data["Acos"])
#     Asin = None
#     a_scf = data["a_scf"]
# 
# _nfw_scf_potential = SCFPotential(Acos=Acos, Asin=None, a=a_scf, normalize=1.0)
# _nfw_scf_potential.turn_physical_on()


# Triaxial NFW
# tnfw_potential = TriaxialNFWPotential(normalize=1.0, c=1.4, a=1.0)
# tnfw_potential.turn_physical_on()
# tnfw_df = osipkovmerrittNFWdf(tnfw_potential, rmax=1e4)


# ------------------------
cls_pot_kw = {
    _hernquist_scf_potential: {"total_mass": 1.0},
    # _nfw_scf_potential: {"total_mass": 1.0},
}
theory = {
    _hernquist_scf_potential: hernquist_df,
    # _nfw_scf_potential: nfw_df,
}


@pytest.fixture(scope="session")
def hernquist_scf_potential():
    """Make a SCF of a Hernquist potential.

    This is tested for quality in ``test_conftest.py``
    """
    return _hernquist_scf_potential


# @pytest.fixture(scope="session")
# def nfw_scf_potential():
#     """Make a SCF of a triaxial NFW potential."""
#     return _nfw_scf_potential


@pytest.fixture(
    params=[
        "hernquist_scf_potential",
        # "nfw_scf_potential",  # TODO! turn on
    ],
)
def potentials(request):
    if request.param in ("hernquist_scf_potential"):
        potential = hernquist_scf_potential.__wrapped__()
    elif request.param == "nfw_scf_potential":
        potential = nfw_scf_potential.__wrapped__()

    yield potential
