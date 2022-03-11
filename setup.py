#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# NOTE: The configuration for the package, including the name, version, and
# other information are set in the setup.cfg file.

# BUILT-IN
import os
import sys

# THIRD PARTY
from extension_helpers import get_extensions
from setuptools import setup

# from mypyc.build import mypycify

# First provide helpful messages if contributors try and run legacy commands
# for tests or docs.

TEST_HELP = """
Note: running tests is no longer done using 'python setup.py test'. Instead
you will need to run:

    tox -e test

If you don't already have tox installed, you can install it with:

    pip install tox

If you only want to run part of the test suite, you can also use pytest
directly with::

    pip install -e .[test]
    pytest

For more information, see:

  http://docs.astropy.org/en/latest/development/testguide.html#running-tests
"""

if "test" in sys.argv:
    print(TEST_HELP)
    sys.exit(1)

DOCS_HELP = """
Note: building the documentation is no longer done using
'python setup.py build_docs'. Instead you will need to run:

    tox -e build_docs

If you don't already have tox installed, you can install it with:

    pip install tox

You can also build the documentation with Sphinx directly using::

    pip install -e .[docs]
    cd docs
    make html

For more information, see:

  http://docs.astropy.org/en/latest/install.html#builddocs
"""

if "build_docs" in sys.argv or "build_sphinx" in sys.argv:
    print(DOCS_HELP)
    sys.exit(1)

VERSION_TEMPLATE = """
# Note that we need to fall back to the hard-coded version if either
# setuptools_scm can't be imported or setuptools_scm can't determine the
# version, so we catch the generic 'Exception'.
try:
    from setuptools_scm import get_version
    version = get_version(root='..', relative_to=__file__)
except Exception:
    version = '{version}'
""".lstrip()

# # TODO! model after https://github.com/python/mypy/blob/master/setup.py
# mypyc_targets = [
#     os.path.join("sample_scf", x)
#     for x in ("__init__.py", "base.py", "core.py", "utils.py", "interpolated.py",
#               "exact.py")
# ]
# # The targets come out of file system apis in an unspecified
# # order. Sort them so that the mypyc output is deterministic.
# mypyc_targets.sort()

setup(
    use_scm_version={
        "write_to": os.path.join("sample_scf", "version.py"),
        "write_to_template": VERSION_TEMPLATE,
    },
    ext_modules=get_extensions(),
    # name="sample_scf",
    # packages=["sample_scf"],
    # ext_modules=mypycify(["--disallow-untyped-defs", *mypyc_targets]),
)
