# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.core`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.units as u
import numpy as np
import pytest

# LOCAL
from .test_base import SCFSamplerTestBase
from .test_interpolated import pgrid, rgrid, tgrid
from sample_scf import conftest, core

##############################################################################
# TESTS
##############################################################################


class Test_SCFSampler(SCFSamplerTestBase):
    """Test :class:`sample_scf.core.SCFSample`."""

    @pytest.fixture()
    def sampler(self, potentials):
        """Set up r, theta, phi sampler."""
        kw = {**self.cls_kwargs, **self.cls_pot_kw.get(potentials, {})}
        sampler = self.cls(potentials, *self.cls_args, **kw)

        return sampler

    def setup_class(self):
        super().setup_class(self)

        self.cls = core.SCFSampler
        self.cls_args = ("interp",)  # TODO! iterate over this
        self.cls_kwargs = dict(rgrid=rgrid, thetagrid=tgrid, phigrid=pgrid)
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
        cdf = sampler.cdf(r, theta, phi)
        assert np.allclose(cdf, expected, atol=1e-16)

        # also test shape
        assert tuple(np.atleast_1d(np.squeeze((*np.shape(r), 3)))) == cdf.shape
