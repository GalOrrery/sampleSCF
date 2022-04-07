# -*- coding: utf-8 -*-

"""Some basic tests."""

##############################################################################
# IMPORTS

# STDLIB
import inspect

# LOCAL
import sample_scf
from sample_scf.core import SCFSampler
from sample_scf.exact import ExactSCFSampler
from sample_scf.interpolated import InterpolatedSCFSampler

##############################################################################
# TESTS
##############################################################################


def test_expected_imports():
    """Test can import expected modules and objects."""
    assert inspect.ismodule(sample_scf)

    assert sample_scf.SCFSampler is SCFSampler
    assert sample_scf.ExactSCFSampler is ExactSCFSampler
    assert sample_scf.InterpolatedSCFSampler is InterpolatedSCFSampler
