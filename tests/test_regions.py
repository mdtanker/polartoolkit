# Copyright (c) 2024 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#
"""
Tests for regions module.
"""

# %%
from __future__ import annotations

import sys
from importlib import reload
from unittest.mock import MagicMock, patch

import pytest

from polartoolkit import regions

try:
    import ipyleaflet
except ImportError:
    ipyleaflet = None

try:
    from IPython.display import display
except ImportError:
    display = None

regions_dict = regions.get_regions()
names = list(regions_dict.keys())
region_values = list(regions_dict.values())


def check_region_valid(region: tuple[float, float, float, float]) -> None:
    assert isinstance(region, tuple)
    assert len(region) == 4
    assert all(isinstance(i, (int, float)) for i in region)
    assert region[0] < region[1]
    assert region[2] < region[3]
    assert region[0] != region[1]
    assert region[2] != region[3]


@pytest.mark.parametrize(("testname", "region"), zip(names, region_values))
def test_regions(testname, region):  # noqa: ARG001
    check_region_valid(region)


def test_combine_regions():
    reg1 = (1, 2, 3, 4)
    reg2 = (0, 1, 2, 3)

    assert regions.combine_regions(reg1, reg2) == (0, 2, 2, 4)


def test_draw_region_missing_ipyleaflet():
    """
    Check error raise after calling draw_region when ipyleaflet is missing
    """
    with patch.dict(sys.modules, {"ipyleaflet": None}):
        reload(sys.modules["polartoolkit.regions"])
        with pytest.raises(ImportError) as exception:
            regions.draw_region()
        assert "'ipyleaflet'" in str(exception.value)


def test_draw_region_missing_ipython():
    """
    Check error raise after calling draw_region when ipython is missing
    """
    sys.modules["ipyleaflet"] = MagicMock()
    with patch.dict(sys.modules, {"IPython.display": None}):
        reload(sys.modules["polartoolkit.regions"])
        with pytest.raises(ImportError) as exception:
            regions.draw_region()
        assert "'ipython'" in str(exception.value)
