# -*- coding: utf-8 -*-

"""Testing :mod:`scample_scf.core`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.units as u
import numpy as np
import pytest
from galpy.potential import SCFPotential

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
