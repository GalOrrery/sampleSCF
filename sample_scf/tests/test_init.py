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

    assert inspect.ismodule(sample_scf)


# /def


# -------------------------------------------------------------------


##############################################################################
# END
