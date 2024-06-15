# Copyright (c) 2024 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#
# Copyright (c) 2022 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#
"""
Tests for utils module.
"""

# %%
from __future__ import annotations

import deprecation
import numpy as np
import pandas as pd
import pytest
import verde as vd
import xarray as xr

from polartoolkit import regions, utils


@deprecation.fail_if_not_removed
def test_alter_region():
    utils.alter_region(regions.ross_ice_shelf, zoom=10e3)


def dummy_grid() -> xr.Dataset:
    (x, y, z) = vd.grid_coordinates(
        region=(-100, 100, 200, 400),
        spacing=100,
        extra_coords=20,
    )

    # create topographic features
    misfit = y**2

    return vd.make_xarray_grid(
        (x, y),
        (misfit, z),
        data_names=("misfit", "upward"),
        dims=("northing", "easting"),
    )


# %%
def test_subset_grid():
    "test the subset grid function"
    grid = dummy_grid()

    region = (0, 100, 200, 300)
    subset = utils.subset_grid(grid.misfit, region)

    reg = utils.get_grid_info(subset)[1]

    assert reg == region


def test_subset_grid_bigger():
    "test the subset grid function with a region bigger than the grid"
    grid = dummy_grid()

    region = (-200, 100, 200, 500)
    subset = utils.subset_grid(grid.misfit, region)

    reg = utils.get_grid_info(subset)[1]

    assert reg == (-100, 100, 200, 400)


def test_rmse():
    """
    test the RMSE function
    """
    # create some dummy data
    data = np.array([1, 2, 3])
    # calculate the RMSE
    rmse = utils.rmse(data)
    # test that the RMSE is correct
    assert rmse == pytest.approx(2.160247, rel=0.0001)


def test_rmse_median():
    """
    test the RMedianSE function
    """
    # create some dummy data
    data = np.array([1, 2, 3])
    # calculate the RMedianSE
    rmse = utils.rmse(data, as_median=True)
    # test that the RMSE is correct
    assert rmse == 2


def test_get_grid_info():
    """
    test the get_grid_info function
    """
    grid = dummy_grid()

    info = utils.get_grid_info(grid.misfit)

    assert info == (100.0, (-100.0, 100.0, 200.0, 400.0), 40000.0, 160000.0, "g")


def test_dd2dms():
    """
    test the dd2dms function
    """

    dd = 130.25

    dms = utils.dd2dms(dd)

    assert dms == "130:15:0.0"


def test_region_to_df():
    """
    test the region_to_df function
    """

    reg = regions.ross_ice_shelf

    df = utils.region_to_df(reg)

    expected = pd.DataFrame(
        {
            "x": [
                -680000.0,
                470000.0,
                -680000.0,
                470000.0,
            ],
            "y": [
                -1420000.0,
                -1420000.0,
                -310000.0,
                -310000.0,
            ],
        }
    )

    pd.testing.assert_frame_equal(df, expected)

    reg2 = utils.region_to_df(df, reverse=True)

    assert reg2 == reg


def test_region_xy_to_ll_north():
    """
    test the GMT_xy_to_ll function
    """

    reg_xy = regions.north_greenland

    reg_ll = utils.region_xy_to_ll(reg_xy, hemisphere="north", dms=True)

    assert reg_ll == (
        "-82:34:6.931303778954316",
        "-2:17:26.19615349869855",
        "77:39:39.3112071675132",
        "82:26:25.183721040550154",
    )

    reg_ll = utils.region_xy_to_ll(reg_xy, hemisphere="north")

    assert reg_ll == (
        -82.56859202882748,
        -2.2906100426385274,
        77.66091977976876,
        82.44032881140015,
    )


def test_region_xy_to_ll_south():
    """
    test the GMT_xy_to_ll function
    """

    reg_xy = regions.ross_ice_shelf

    reg_ll = utils.region_xy_to_ll(reg_xy, hemisphere="south", dms=True)

    assert reg_ll == (
        "-154:24:41.35269126086496",
        "161:41:10.261402247124352",
        "-84:49:17.300473876937758",
        "-75:34:58.96941344602965",
    )

    reg_ll = utils.region_xy_to_ll(reg_xy, hemisphere="south")

    assert reg_ll == (
        -154.41148685868356,
        161.6861837228464,
        -84.8214723538547,
        -75.58304705929056,
    )


def test_region_to_bounding_box():
    """
    test the region_to_bounding_box function
    """

    reg = regions.ross_ice_shelf

    box = utils.region_to_bounding_box(reg)

    assert box == (-680000.0, -1420000.0, 470000.0, -310000.0)


def test_latlon_to_epsg3031():
    """
    test the latlon_to_epsg3031 function
    """

    df_ll = pd.DataFrame(
        {
            "lat": [-75.583047, -76.296586, -83.129754, -84.82147],
            "lon": [-154.411487, 161.686184, -114.507405, 123.407825],
        }
    )

    df_xy = utils.latlon_to_epsg3031(df_ll)

    expected = pd.DataFrame(
        {
            "x": [
                -680000.0,
                470000.0,
                -680000.0,
                470000.0,
            ],
            "y": [
                -1420000.0,
                -1420000.0,
                -310000.0,
                -310000.0,
            ],
        }
    )
    pd.testing.assert_frame_equal(df_xy[["x", "y"]], expected)


def test_latlon_to_epsg3031_region():
    """
    test the latlon_to_epsg3031 function output a region
    """

    df_ll = pd.DataFrame(
        {
            "lat": [-75.583047, -76.296586, -83.129754, -84.82147],
            "lon": [-154.411487, 161.686184, -114.507405, 123.407825],
        }
    )

    reg = utils.latlon_to_epsg3031(df_ll, reg=True)

    assert reg == pytest.approx(regions.ross_ice_shelf, abs=10)


def test_epsg3031_to_latlon():
    """
    test the epsg3031_to_latlon function
    """

    df_xy = utils.region_to_df(regions.ross_ice_shelf)

    df_ll = utils.epsg3031_to_latlon(df_xy)

    expected = pd.DataFrame(
        {
            "x": [
                -680000.0,
                470000.0,
                -680000.0,
                470000.0,
            ],
            "y": [
                -1420000.0,
                -1420000.0,
                -310000.0,
                -310000.0,
            ],
            "lat": [-75.583047, -76.296586, -83.129754, -84.82147],
            "lon": [-154.411487, 161.686184, -114.507405, 123.407825],
        }
    )
    pd.testing.assert_frame_equal(df_ll, expected)


def test_epsg3031_to_latlon_region():
    """
    test the epsg3031_to_latlon function output a region
    """

    df_xy = utils.region_to_df(regions.ross_ice_shelf)

    reg = utils.epsg3031_to_latlon(df_xy, reg=True)

    assert reg == pytest.approx((-154.41, 161.69, -84.82, -75.58), abs=0.01)


def test_latlon_to_epsg3413():
    """
    test the latlon_to_epsg3413 function
    """

    df_ll = pd.DataFrame(
        {
            "lat": [68.366181, 68.070993, 70.129560, 69.806264],
            "lon": [-35.776078, -31.773128, -34.930937, -30.586402],
        }
    )

    df_xy = utils.latlon_to_epsg3413(df_ll)

    expected = pd.DataFrame(
        {
            "x": [
                380000.0,
                550000.0,
                380000.0,
                550000.0,
            ],
            "y": [
                -2340000.0,
                -2340000.0,
                -2140000.0,
                -2140000.0,
            ],
        }
    )
    pd.testing.assert_frame_equal(df_xy[["x", "y"]], expected)


def test_latlon_to_epsg3413_region():
    """
    test the latlon_to_epsg3413 function output a region
    """

    df_ll = pd.DataFrame(
        {
            "lat": [68.366181, 68.070993, 70.129560, 69.806264],
            "lon": [-35.776078, -31.773128, -34.930937, -30.586402],
        }
    )

    reg = utils.latlon_to_epsg3413(df_ll, reg=True)

    assert reg == pytest.approx(regions.kangerlussuaq_glacier, abs=10)


def test_epsg3413_to_latlon():
    """
    test the epsg3413_to_latlon function
    """

    df_xy = utils.region_to_df(regions.kangerlussuaq_glacier)

    df_ll = utils.epsg3413_to_latlon(df_xy)

    expected = pd.DataFrame(
        {
            "x": [
                380000.0,
                550000.0,
                380000.0,
                550000.0,
            ],
            "y": [
                -2340000.0,
                -2340000.0,
                -2140000.0,
                -2140000.0,
            ],
            "lat": [68.366181, 68.070993, 70.129560, 69.806264],
            "lon": [-35.776078, -31.773128, -34.930937, -30.586402],
        }
    )
    pd.testing.assert_frame_equal(df_ll, expected)


def test_epsg3413_to_latlon_region():
    """
    test the epsg3413_to_latlon function output a region
    """

    df_xy = utils.region_to_df(regions.kangerlussuaq_glacier)

    reg = utils.epsg3413_to_latlon(df_xy, reg=True)

    assert reg == pytest.approx((-35.78, -30.59, 68.07, 70.13), abs=0.01)


def test_points_inside_region():
    """
    test the points_inside_region function
    """
    # first point is inside, second is outside
    df = pd.DataFrame(
        {
            "x": [-50e3, 0],
            "y": [-1000e3, 0],
        }
    )

    assert len(df) == 2

    reg = regions.ross_ice_shelf

    df_in = utils.points_inside_region(df, reg)

    assert len(df_in) == 1
    assert df_in.x.iloc[0] == -50e3

    df_out = utils.points_inside_region(df, reg, reverse=True)

    assert len(df_out) == 1
    assert df_out.x.iloc[0] == 0.0
