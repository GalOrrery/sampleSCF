# -*- coding: utf-8 -*-

"""Some basic tests."""

__all__ = [
    "test_expected_imports",
]


##############################################################################
# IMPORTS

# BUILT-IN
import inspect

##############################################################################
# TESTS
##############################################################################


def test_expected_imports():
    """Test can import expected modules and objects."""
    # LOCAL
    import sample_scf
    from sample_scf import core, exact, interpolated

    assert inspect.ismodule(sample_scf)
    assert inspect.ismodule(core)
    assert inspect.ismodule(exact)
    assert inspect.ismodule(interpolated)

    assert sample_scf.SCFSampler is core.SCFSampler
    assert sample_scf.SCFSamplerExact is exact.SCFSampler
    assert sample_scf.SCFSamplerInterp is interpolated.SCFSampler


# /def


# -------------------------------------------------------------------


##############################################################################
# END
