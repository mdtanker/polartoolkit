from __future__ import annotations

import importlib.metadata

import polartoolkit


def test_version():
    assert (
        importlib.metadata.version("polartoolkit") == polartoolkit.__version__
    )
