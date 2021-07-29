# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.core`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.units as u

# LOCAL
from .test_base import Test_SCFSamplerBase as SCFSamplerBaseTests
from .test_sample_intrp import pgrid, rgrid, tgrid
from sample_scf import core

##############################################################################
# TESTS
##############################################################################


class Test_SCFSampler(SCFSamplerBaseTests):
    """Test :class:`sample_scf.core.SCFSample`."""

    def setup_class(self):
        super().setup_class(self)

        self.cls = core.SCFSampler
        self.cls_args = ("interp",)  # TODO! iterate over this
        self.cls_kwargs = dict(rgrid=rgrid, thetagrid=tgrid, phigrid=pgrid)

        self.expected_rvs = {
            0: dict(r=0.548813503927, theta=1.021982822867 * u.rad, phi=0.548813503927 * u.rad),
            1: dict(r=0.548813503927, theta=1.021982822867 * u.rad, phi=0.548813503927 * u.rad),
            2: dict(
                r=[0.9670298390136, 0.5472322491757, 0.9726843599648, 0.7148159936743],
                theta=[0.603766487781, 1.023564077619, 0.598111966830, 0.855980333120] * u.rad,
                phi=[0.9670298390136, 0.547232249175, 0.9726843599648, 0.7148159936743] * u.rad,
            ),
        }

    # /def


# /class

##############################################################################
# END
