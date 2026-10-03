#!/usr/bin/env python
"""Utility functions used by the test suite. """


import os
import shutil
import tempfile
import contextlib


@contextlib.contextmanager
def tempdir():
    """Create and change into a temporary directory, and delete it afterwards.
    """
    testdir = tempfile.mkdtemp()
    prevdir = os.getcwd()
    try:
        os.chdir(testdir)
        yield testdir
    finally:
        os.chdir(prevdir)
        shutil.rmtree(testdir)
