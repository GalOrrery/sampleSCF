# -*- coding: utf-8 -*-

"""Some basic tests."""

##############################################################################
# IMPORTS

# BUILT-IN
import inspect

# LOCAL
import sample_scf
from sample_scf import core, exact, interpolated

##############################################################################
# TESTS
##############################################################################


def test_expected_imports():
    """Test can import expected modules and objects."""
    assert inspect.ismodule(sample_scf)
    assert inspect.ismodule(core)
    assert inspect.ismodule(exact)
    assert inspect.ismodule(interpolated)

    assert sample_scf.SCFSampler is core.SCFSampler
    assert sample_scf.ExactSCFSampler is exact.SCFSampler
    assert sample_scf.InterpolatedSCFSampler is interpolated.SCFSampler
