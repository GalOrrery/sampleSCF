# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.core`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.units as u
import numpy as np
import pytest

# LOCAL
from sample_scf import SCFSampler
from sample_scf.exact import exact_r_distribution, exact_theta_distribution, exact_phi_distribution

from .test_base_multivariate import BaseTest_SCFSamplerBase

##############################################################################
# TESTS
##############################################################################


def test_MethodsMapping(potentials):
    """Test `sample_scf.core.MethodsMapping`."""
    # Good
    mm = MethodsMapping(
        r=exact_r_distribution(potentials, total_mass=1),
        theta=exact_theta_distribution(potentials, )
        phi=exact_phi_distribution(potentials)
    )

    assert False


##############################################################################


class Test_SCFSampler(BaseTest_SCFSamplerBase):
    """Test :class:`sample_scf.core.SCFSample`."""

    @pytest.fixture(scope="class")
    def rv_cls(self):
        return SCFSampler

    @pytest.fixture(scope="class")
    def rv_cls_args(self):
        return ("interp",)  # TODO! iterate over this

    @pytest.fixture(scope="class")
    def rv_cls_kw(self):
        # return dict(rgrid=rgrid, thetagrid=tgrid, phigrid=pgrid)
        return {}

    def setup_class(self):
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

    @pytest.mark.parametrize(
        "r, theta, phi, expected",
        [
            (0, 0, 0, [0, 0.5, 0]),
            (1, 0, 0, [0.2505, 0.5, 0]),
            ([0, 1], [0, 0], [0, 0], [[0, 0.5, 0], [0.2505, 0.5, 0]]),
        ],
    )
    def test_cdf(self, sampler, r, theta, phi, expected):
        """Test :meth:`sample_scf.base_multivariate.SCFSamplerBase.cdf`."""
        cdf = sampler.cdf(r, theta, phi)
        assert np.allclose(cdf, expected, atol=1e-16)

        # also test shape
        assert tuple(np.atleast_1d(np.squeeze((*np.shape(r), 3)))) == cdf.shape
