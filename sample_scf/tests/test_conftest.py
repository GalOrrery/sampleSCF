# -*- coding: utf-8 -*-

"""Testing :mod:`~sample_scf.conftest`.

Even the test source should be tested.
In particular, the potential fixtures need confirmation that the SCF form
matches the theoretical, within tolerances.
"""

__all__ = [
    "Test_ClassName",
    "test_function",
]


##############################################################################
# IMPORTS

# BUILT-IN
import abc

# THIRD PARTY
import numpy as np
import pytest

# LOCAL
from sample_scf import conftest

##############################################################################
# PARAMETERS


##############################################################################
# TESTS
##############################################################################


class PytestPotential(metaclass=abc.ABCMeta):
    """Test a Pytest Potential."""

    @classmethod
    @abc.abstractmethod
    def setup_class(self):
        """Setup fixtures for testing."""
        self.R = np.linspace(0.0, 3.0, num=1001)
        self.atol = 1e-6
        self.restrict_ind = np.ones(1001, dtype=bool)

    @pytest.fixture(scope="class")
    @abc.abstractmethod
    def scf_potential(self):
        """The `galpy.potential.SCFPotential` from a `pytest.fixture`."""
        return

    def compare_to_theory(self, theory, scf, atol=1e-6):
        # test that where theory is finite they match and where it's infinite,
        # the scf is NaN
        fnt = ~np.isinf(theory)
        ind = self.restrict_ind & fnt

        assert np.allclose(theory[ind], scf[ind], atol=atol)
        assert np.all(np.isnan(scf[~fnt]))

    # ===============================================================
    # sanity checks

    def test_df(self):
        assert self.df._pot is self.theory

    # ===============================================================

    def test_density_along_Rz_equality(self, scf_potential):
        theory = self.theory.dens(self.R, self.R)
        scf = scf_potential.dens(self.R, self.R)
        self.compare_to_theory(theory, scf, atol=self.atol)

    @pytest.mark.parametrize("z", [0, 10, 15])
    def test_density_at_z(self, scf_potential, z):
        theory = self.theory.dens(self.R, z)
        scf = scf_potential.dens(self.R, z)
        self.compare_to_theory(theory, scf, atol=self.atol)


# -------------------------------------------------------------------


class Test_hernquist_scf_potential(PytestPotential):
    @classmethod
    def setup_class(self):
        """Setup fixtures for testing."""
        super().setup_class()

        self.theory = conftest.hernquist_potential
        self.df = conftest.hernquist_df

    @pytest.fixture(scope="class")
    def scf_potential(self, hernquist_scf_potential):
        """The `galpy.potential.SCFPotential` from a `pytest.fixture`."""
        return hernquist_scf_potential


# -------------------------------------------------------------------


class Test_nfw_scf_potential(PytestPotential):
    @classmethod
    def setup_class(self):
        """Setup fixtures for testing."""
        super().setup_class()

        self.theory = conftest.nfw_potential
        self.df = conftest.nfw_df

        self.atol = 1e-2
        self.restrict_ind[:18] = False  # skip some of the inner ones

    @pytest.fixture(scope="class")
    def scf_potential(self, nfw_scf_potential):
        """The `galpy.potential.SCFPotential` from a `pytest.fixture`."""
        return nfw_scf_potential
