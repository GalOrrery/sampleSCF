# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.core`."""


##############################################################################
# IMPORTS

# LOCAL
from .test_base import Test_SCFSamplerBase
from sample_scf import core

##############################################################################
# TESTS
##############################################################################


class Test_SCFSampler(Test_SCFSamplerBase):
    """Test :class:`sample_scf.core.SCFSample`."""

    _cls = core.SCFSampler


# /class

##############################################################################
# END
