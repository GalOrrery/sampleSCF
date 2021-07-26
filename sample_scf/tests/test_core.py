# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.core`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import pytest

# LOCAL
from .test_base import Test_SCFSamplerBase
from sample_scf import core

##############################################################################
# TESTS
##############################################################################


@pytest.mark.skip("TODO!")
class Test_SCFSampler(Test_SCFSamplerBase):
    """Test :class:`sample_scf.core.SCFSample`."""

    _cls = core.SCFSampler


# /class

##############################################################################
# END
