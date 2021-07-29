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
    from sample_scf import core, sample_exact, sample_intrp

    assert inspect.ismodule(sample_scf)
    assert inspect.ismodule(core)
    assert inspect.ismodule(sample_exact)
    assert inspect.ismodule(sample_intrp)

    assert sample_scf.SCFSampler is core.SCFSampler
    assert sample_scf.SCFSamplerExact is sample_exact.SCFSampler
    assert sample_scf.SCFSamplerInterp is sample_intrp.SCFSampler


# /def


# -------------------------------------------------------------------


##############################################################################
# END
