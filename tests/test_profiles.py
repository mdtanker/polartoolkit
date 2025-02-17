# Copyright (c) 2024 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#
"""
Tests for profiles module.
"""

# This is only here to suppress the bug described in
# https://github.com/pydata/xarray/issues/7259
# We have to make sure that netcdf4 is imported before
# numpy is imported for the first time, e.g. also via
# importing xarray
import netCDF4  # noqa: F401
import numpy as np
import pytest

from polartoolkit import profiles


def test_old_profile_module():
    """
    Check error raise after import old profile module
    """
    with pytest.raises(ImportError) as exception:
        from polartoolkit import profile  # noqa: F401
    assert "'profile'" in str(exception.value)


default_layers_test = [
    (
        {
            "version": "bedmap2",
            "hemisphere": "south",
        },
        [  # mean values of grids
            602.32306,
            11.617859,
            -2127.135,
        ],
    ),
    (
        {
            "version": "bedmachine",
            "hemisphere": "south",
        },
        [  # mean values of grids
            595.5868,
            7.055721,
            -2522.419,
        ],
    ),
    (
        {
            "version": "bedmachine",
            "hemisphere": "north",
        },
        [  # mean values of grids
            934.1983,
            231.09322,
            -312.9952,
        ],
    ),
]


@pytest.mark.fetch
@pytest.mark.earthdata
@pytest.mark.parametrize(("test_input", "expected"), default_layers_test)
def test_default_layers(test_input, expected):
    """
    Check default layers
    """
    layers = profiles.default_layers(**test_input)
    names = [v["name"] for v in layers.values()]
    colors = [v["color"] for v in layers.values()]

    assert names == ["ice", "water", "earth"]
    assert colors == ["lightskyblue", "darkblue", "lightbrown"]
    assert [np.nanmean(v["grid"]) for v in layers.values()] == pytest.approx(
        expected, 0.1
    )
