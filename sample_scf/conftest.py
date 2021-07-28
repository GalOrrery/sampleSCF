# -*- coding: utf-8 -*-
# mypy: ignore-errors

"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

# BUILT-IN
import os

# THIRD PARTY
import numpy as np
import pytest
from galpy.potential import SCFPotential

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


# /def


# Uncomment the last two lines in this block to treat all DeprecationWarnings as
# exceptions. For Astropy v2.0 or later, there are 2 additional keywords,
# as follow (although default should work for most cases).
# To ignore some packages that produce deprecation warnings on import
# (in addition to 'compiler', 'scipy', 'pygments', 'ipykernel', and
# 'setuptools'), add:
#     modules_to_ignore_on_import=['module_1', 'module_2']
# To ignore some specific deprecation warning messages for Python version
# MAJOR.MINOR or later, add:
#     warnings_to_ignore_by_pyver={(MAJOR, MINOR): ['Message to ignore']}
# from astropy.tests.helper import enable_deprecations_as_exceptions  # noqa: F401
# enable_deprecations_as_exceptions()


# ============================================================================
# Fixtures


@pytest.fixture(autouse=True, scope="session")
def hernquist_scf_potential():
    """Make a SCF of a Hernquist potential."""
    Acos = np.zeros((5, 6, 6))

    Acos_hern = Acos.copy()
    Acos_hern[0, 0, 0] = 1

    hernpot = SCFPotential(Acos=Acos_hern)
    return hernpot


# /def


# @pytest.fixture(autouse=True, scope="session")
# def nfw_scf_potential():
#     """Make a SCF of a triaxial NFW potential."""
#     raise NotImplementedError("TODO")
#
#
# # /def


@pytest.fixture(
    # autouse=True,
    scope="session",
    params=[
        "hernquist_scf_potential",  # TODO! use hernquist_scf_potential
    ],
)
def potentials(request):
    if request.param == "hernquist_scf_potential":
        Acos = np.zeros((5, 6, 6))
        Acos_hern = Acos.copy()
        Acos_hern[0, 0, 0] = 1
        potential = SCFPotential(Acos=Acos_hern)

    yield potential
