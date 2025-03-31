# Copyright (c) 2024 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#
# %%
"""
Tests for fetch module. Use pre-determined results of utils.get_grid_info() to verify
grids have been properly fetch. Also tests the `resample_grid()` function
Follow this format:
def test_():
    grid = fetch.()
    expected =
    assert utils.get_grid_info(grid) == expected
"""

from __future__ import annotations

import os

import deepdiff
import deprecation
import geopandas as gpd

# This is only here to suppress the bug described in
# https://github.com/pydata/xarray/issues/7259
# We have to make sure that netcdf4 is imported before
# numpy is imported for the first time, e.g. also via
# importing xarray
import netCDF4  # noqa: F401
import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv
from pandas.testing import assert_frame_equal

from polartoolkit import fetch, regions, utils

try:
    import pyogrio

    ENGINE = "pyogrio"
except ImportError:
    pyogrio = None
    ENGINE = "fiona"

load_dotenv()

earthdata_login = [
    os.environ.get("EARTHDATA_USERNAME", None),
    os.environ.get("EARTHDATA_PASSWORD", None),
]

# create skipif decorator for fetch calls which use NSIDC Earthdata logins
skip_earthdata = pytest.mark.skipif(
    earthdata_login == [None, None],
    reason="requires earthdata login credentials set as environment variables",
)


@pytest.mark.fetch
@pytest.mark.earthdata
@skip_earthdata
@deprecation.fail_if_not_removed
def test_measures_boundaries():
    fetch.measures_boundaries("Coastline")


# %% resample_grid
resample_test = [
    # no inputs
    (
        {},
        (
            10000.0,
            (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            -384.5,
            204.800003052,
            "g",
        ),
    ),
    # return original with given initials
    (
        {
            "initial_region": (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            "initial_spacing": 10e3,
            "initial_registration": "g",
        },
        (
            10000,
            (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            -384.5,
            204.800003052,
            "g",
        ),
    ),
    # give false initial values, return actual initial values
    (
        {
            "initial_region": (-2800000.0, 2800000.0, -2800000.0, 2800000.0),
            "initial_spacing": 8e3,
            "initial_registration": "p",
        },
        (
            10000,
            (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            -384.5,
            204.800003052,
            "g",
        ),
    ),
    # Only registration is different
    (
        {
            "registration": "p",
        },
        (
            10000,
            (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            -337.490234375,
            170.69921875,
            "p",
        ),
    ),
    # smaller spacing, uneven, reset region to keep exact spacing
    (
        {"spacing": 8212},
        (
            8212,
            (-3325860.0, 3325860.0, -3325860.0, 3325860.0),
            -374.47366333,
            182.33392334,
            "g",
        ),
    ),
    # larger spacing, uneven, reset region to keep exact spacing
    (
        {"spacing": 10119},
        (
            10119,
            (-3329151.0, 3329151.0, -3329151.0, 3329151.0),
            -318.772613525,
            177.986114502,
            "g",
        ),
    ),
    # uneven subregion, reset region to keep exact spacing
    (
        {"region": (210012.0, 390003.0, -1310217.0, -1121376.0)},
        (
            10000,
            (210000.0, 400000.0, -1320000.0, -1120000.0),
            -175.400009155,
            54.1000022888,
            "g",
        ),
    ),
    # uneven subregion with diff reg, reset region to keep exact spacing
    (
        {
            "region": (210012.0, 390003.0, -1310217.0, -1121376.0),
            "registration": "p",
        },
        (
            10000,
            (210000.0, 400000.0, -1320000.0, -1120000.0),
            -156.026565552,
            46.8070335388,
            "p",
        ),
    ),
    # uneven spacing (smaller) and uneven region, reset region to keep exact spacing
    (
        {
            "spacing": 8212,
            "region": (210012.0, 390003.0, -1310217.0, -1121376.0),
        },
        (
            8212,
            (205300.0, 402388.0, -1322132.0, -1116832.0),
            -170.436401367,
            47.9773178101,
            "g",
        ),
    ),
    # uneven spacing (larger) and uneven region, reset region to keep exact spacing
    (
        {
            "spacing": 10119,
            "region": (210012.0, 390003.0, -1310217.0, -1121376.0),
        },
        (
            10119,
            (212499.0, 384522.0, -1305351.0, -1123209.0),
            -173.363143921,
            50.2054672241,
            "g",
        ),
    ),
    # larger than initial region, return initial region
    (
        {"region": (-3400e3, 3400e3, -3400e3, 34030e3)},
        (
            10000,
            (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            -384.5,
            204.800003052,
            "g",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.parametrize(("test_input", "expected"), resample_test)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_resample_grid(test_input, expected):
    grid = fetch.gravity(version="antgg", anomaly_type="FA").free_air_anomaly
    resampled = fetch.resample_grid(grid, **test_input)
    # assert utils.get_grid_info(resampled) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(resampled),
        expected,
        ignore_order=True,
        significant_digits=0,
    )


# test_input = dict(
#     spacing=10119,
#     region=(-3400e3, 3400e3, -3400e3, 34030e3),
#     registration='p',
# )
# grid = fetch.gravity(version='antgg', anomaly_type='FA')
# resampled = fetch.resample_grid(grid, **test_input)
# utils.get_grid_info(resampled)


# %% ice_vel
@pytest.mark.fetch
@pytest.mark.earthdata
@skip_earthdata
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_ice_vel_lowres():
    grid = fetch.ice_vel(spacing=5e3, hemisphere="south")
    expected = (
        5000,
        (-2800000.0, 2795000.0, -2795000.0, 2800000.0),
        2.87566308543e-05,
        4201.68994141,
        "g",
    )
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid), expected, ignore_order=True, significant_digits=2
    )


@pytest.mark.fetch
@pytest.mark.earthdata
@skip_earthdata
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_ice_vel_highres():
    grid = fetch.ice_vel(spacing=450, hemisphere="south")
    expected = (
        450,
        (-2800000.0, 2799800.0, -2799800.0, 2800000.0),
        2.34232032881e-07,
        4218.26513672,
        "g",
    )
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )

    grid = fetch.ice_vel(spacing=250, hemisphere="north")
    expected = (
        250,
        (-645000.0, 859750.0, -3370000.0, -640250.0),
        0.00136277475394,
        12906.4941406,
        "g",
    )
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid), expected, ignore_order=True, significant_digits=2
    )


# grid = fetch.ice_vel(spacing=1e3)
# utils.get_grid_info(grid)

# %% modis


@pytest.mark.fetch
@pytest.mark.earthdata
@skip_earthdata
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_modis():
    grid = fetch.modis(version="750m", hemisphere="south")
    expected = (
        750,
        (-3174450.0, 2867550.0, -2816675.0, 2406325.0),
        0.0,
        42374.0,
        "p",
    )
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )

    grid = fetch.modis(version="500m", hemisphere="north")
    expected = (
        500,
        (-1200000.0, 900000.0, -3400000.0, -600000.0),
        0.0,
        35746.0,
        "p",
    )
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# version="125m"of MoA and "100m" of MoG not tested since too large

# grid = fetch.modis(version="750m")
# utils.get_grid_info(grid)


@pytest.mark.fetch
@pytest.mark.earthdata
@skip_earthdata
@deprecation.fail_if_not_removed
def test_modis_moa():
    fetch.modis_moa()


@pytest.mark.fetch
@pytest.mark.earthdata
@skip_earthdata
@deprecation.fail_if_not_removed
def test_modis_mog():
    fetch.modis_mog()


# %% imagery


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_imagery():
    grid = fetch.imagery()
    expected = (
        240.000000516,
        (-2668274.98913, 2813804.9199, -2294625.04002, 2362334.96998),
        1.79769313486e308,
        -1.79769313486e308,
        "p",
    )
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# grid = fetch.imagery()
# utils.get_grid_info(grid)


# %% sediment thickness


sediment_test = [
    (
        "ANTASed",
        (10000, (-2350000.0, 2490000.0, -1990000.0, 2090000.0), 0.0, 12730.0, "g"),
    ),
    (
        "tankersley-2022",
        (
            5000,
            (-3330000.0, 1900000.0, -3330000.0, 1850000.0),
            0.0,
            8002.51953125,
            "p",
        ),
    ),
    (
        "lindeque-2016",
        (5000, (-4600000.0, 1900000.0, -3900000.0, 1850000.0), 0.0, 8042.0, "g"),
    ),
    (
        "GlobSed",
        (
            1000,
            (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            -23.5582103729,
            14005.65625,
            "g",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected"), sediment_test)
def test_sediment_thickness(test_input, expected):
    grid = fetch.sediment_thickness(test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# grid = fetch.sediment_thickness(version='GlobSed')
# utils.get_grid_info(grid)


# %% GeoMap data

geomap_test = [
    ("faults", [750, 765]),
    ("units", [402, 371]),
]


@pytest.mark.fetch
@pytest.mark.parametrize(("test_input", "expected"), geomap_test)
def test_geomap(test_input, expected):
    # collect a few points
    data = fetch.geomap(
        version=test_input, region=regions.alter_region(regions.ross_island, zoom=25e3)
    )[0:2]

    # check if the first 2 object id's match the expected
    assert data.objectid.values.sort() == expected.sort()


# %% IBCSO coverage data


@pytest.mark.fetch
def test_ibcso_coverage():
    # collect a few points
    points, polygons = fetch.ibcso_coverage(
        region=regions.alter_region(regions.siple_coast, zoom=270e3)
    )
    points = points.drop(columns=["geometry"])

    # re-create the expected geodataframe
    expected = pd.DataFrame(
        {
            "dataset_name": [
                "RossSea_seismic_usedbyTinto2019_PS65.xyz",
                "RIGGS_7378_seismic_PS65.xyz",
                "RossSea_seismic_usedbyTinto2019_PS65.xyz",
                "RossSea_seismic_usedbyTinto2019_PS65.xyz",
            ],
            "dataset_tid": np.array([12, 12, 12, 12]).astype("int64"),
            "weight": np.array([10, 10, 10, 10]).astype("int64"),
            "easting": [-242248.738888, -306281.045379, -331166.112422, -245655.331481],
            "northing": [
                -806606.217740,
                -827640.753371,
                -762830.788535,
                -751230.207946,
            ],
        },
        index=[0, 1, 0, 0],
    )

    # check if they match
    assert_frame_equal(points, expected, check_like=True)
    # check that the polygon geodataframe is empty
    assert len(polygons) == 0


# %% IBCSO surface and bed elevations


ibcso_test = [
    (
        {
            "layer": "surface",
            "spacing": 500,
        },
        (
            500,
            (-2800000.0, 2800000.0, -2800000.0, 2800000.0),
            -6322.51318359,
            4790.52978516,
            "g",
        ),
    ),
    (
        {
            "layer": "surface",
            "spacing": 5e3,
        },
        (
            5000,
            (-2800000.0, 2800000.0, -2800000.0, 2800000.0),
            -6302.79589844,
            4546.14599609,
            "g",
        ),
    ),
    (
        {
            "layer": "bed",
            "spacing": 500,
        },
        (
            500,
            (-2800000.0, 2800000.0, -2800000.0, 2800000.0),
            -6322.51318359,
            4731.80712891,
            "g",
        ),
    ),
    (
        {
            "layer": "bed",
            "spacing": 5e3,
        },
        (
            5000,
            (-2800000.0, 2800000.0, -2800000.0, 2800000.0),
            -6302.79589844,
            4568.85058594,
            "g",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected"), ibcso_test)
def test_ibcso(test_input, expected):
    grid = fetch.ibcso(**test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# test_input = dict(
#     layer='surface',
#     spacing=500,
# )
# grid = fetch.ibcso(test_input)
# utils.get_grid_info(grid)


# %% bedmachine
# test for all layers, but only test reference models with 1 layer

bedmachine_test = [
    (
        "icebase",
        (
            500.0,
            (-3333000.0, 3333000.0, -3333000.0, 3333000.0),
            -3827.19604492,
            4818.15576172,
            "g",
        ),
        "south",
    ),
    (
        "surface",
        (
            500,
            (-3333000.0, 3333000.0, -3333000.0, 3333000.0),
            0.0,
            4818.15576172,
            "g",
        ),
        "south",
    ),
    (
        "thickness",
        (
            500,
            (-3333000.0, 3333000.0, -3333000.0, 3333000.0),
            0.0,
            4822.79492188,
            "g",
        ),
        "south",
    ),
    (
        "bed",
        (
            500,
            (-3333000.0, 3333000.0, -3333000.0, 3333000.0),
            -8166.31542969,
            4818.15576172,
            "g",
        ),
        "south",
    ),
    (
        "geoid",
        (500, (-3333000.0, 3333000.0, -3333000.0, 3333000.0), -66.0, 52.0, "g"),
        "south",
    ),
    (
        "icebase",
        (
            150,
            (-653000.0, 879700.0, -3384350.0, -632750.0),
            -1913.28369141,
            3673.34838867,
            "p",
        ),
        "north",
    ),
    (
        "surface",
        (
            150,
            (-653000.0, 879700.0, -3384350.0, -632750.0),
            0.0,
            3673.38549805,
            "p",
        ),
        "north",
    ),
    (
        "thickness",
        (
            150,
            (-653000.0, 879700.0, -3384350.0, -632750.0),
            0.0,
            3409.73779297,
            "p",
        ),
        "north",
    ),
    (
        "bed",
        (
            150,
            (-653000.0, 879700.0, -3384350.0, -632750.0),
            -5571.67285156,
            3673.34838867,
            "p",
        ),
        "north",
    ),
    (
        "geoid",
        (
            150,
            (-653000.0, 879700.0, -3384350.0, -632750.0),
            6.0,
            64.0,
            "p",
        ),
        "north",
    ),
]


@pytest.mark.fetch
@pytest.mark.earthdata
@skip_earthdata
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected", "hemisphere"), bedmachine_test)
def test_bedmachine(test_input, expected, hemisphere):
    grid = fetch.bedmachine(test_input, hemisphere=hemisphere)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


@pytest.mark.fetch
@pytest.mark.earthdata
@skip_earthdata
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_bedmachine_reference_south():
    # fetch variations of grids and reference models
    region = (-101e3, -100e3, -51e3, -50e3)
    eigen_6c4_grid = fetch.geoid(
        spacing=500,
        region=region,
        hemisphere="south",
    )
    BM_eigen_6c4_grid = fetch.bedmachine(
        layer="geoid",
        region=region,
        hemisphere="south",
    )
    surface_6c4_grid = fetch.bedmachine(
        layer="surface",
        reference="eigen-6c4",
        region=region,
        hemisphere="south",
    )
    surface_ellipsoid_grid = fetch.bedmachine(
        layer="surface",
        reference="ellipsoid",
        region=region,
        hemisphere="south",
    )

    # get mean values
    eigen_6c4 = np.nanmean(eigen_6c4_grid)
    BM_eigen_6c4 = np.nanmean(BM_eigen_6c4_grid)
    surface_6c4 = np.nanmean(surface_6c4_grid)
    surface_ellipsoid = np.nanmean(surface_ellipsoid_grid)

    assert surface_ellipsoid - BM_eigen_6c4 == pytest.approx(surface_6c4, rel=0.1)
    assert surface_6c4 + BM_eigen_6c4 == pytest.approx(surface_ellipsoid, rel=0.1)
    assert BM_eigen_6c4 == pytest.approx(eigen_6c4, rel=0.1)


@pytest.mark.fetch
@pytest.mark.earthdata
@skip_earthdata
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_bedmachine_reference_north():
    # fetch variations of grids and reference models
    region = (380e3, 382e3, -2340e3, -2338e3)
    eigen_6c4_grid = fetch.geoid(
        spacing=150,
        region=region,
        hemisphere="north",
    )
    BM_eigen_6c4_grid = fetch.bedmachine(
        layer="geoid",
        region=region,
        hemisphere="north",
    )
    surface_6c4_grid = fetch.bedmachine(
        layer="surface",
        reference="eigen-6c4",
        region=region,
        hemisphere="north",
    )
    surface_ellipsoid_grid = fetch.bedmachine(
        layer="surface",
        reference="ellipsoid",
        region=region,
        hemisphere="north",
    )

    # get mean values
    eigen_6c4 = np.nanmean(eigen_6c4_grid)
    BM_eigen_6c4 = np.nanmean(BM_eigen_6c4_grid)
    surface_6c4 = np.nanmean(surface_6c4_grid)
    surface_ellipsoid = np.nanmean(surface_ellipsoid_grid)

    assert surface_ellipsoid - BM_eigen_6c4 == pytest.approx(surface_6c4, rel=0.1)
    assert surface_6c4 + BM_eigen_6c4 == pytest.approx(surface_ellipsoid, rel=0.1)
    assert BM_eigen_6c4 == pytest.approx(eigen_6c4, rel=0.1)


# grid = fetch.bedmachine(layer='surface', reference="ellipsoid")
# utils.get_grid_info(grid)

# %% bedmap3
# test for all layers, but only test reference models with 1 layer and fill_nans with 1
# layer

bedmap3_test = [
    (
        {"layer": "surface"},
        (
            500,
            # (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            (-3333250.0, 3333250.0, -3333250.0, 3333250.0),
            1,
            4734,
            "g",
        ),
    ),
    (
        {"layer": "icebase"},
        (
            500,
            # (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            (-3333250.0, 3333250.0, -3333250.0, 3333250.0),
            -2973,
            4704,
            "g",
        ),
    ),
    (
        {"layer": "bed"},
        (
            500,
            # (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            (-3333250.0, 3333250.0, -3333250.0, 3333250.0),
            -7409,
            4704,
            "g",
        ),
    ),
    (
        {"layer": "ice_thickness"},
        (
            500,
            # (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            (-3333250.0, 3333250.0, -3333250.0, 3333250.0),
            0,
            4757,
            "g",
        ),
    ),
    (
        {"layer": "water_thickness"},
        (
            500,
            # (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            (-3333250.0, 3333250.0, -3333250.0, 3333250.0),
            0,
            2818,
            "g",
        ),
    ),
    (
        {"layer": "bed_uncertainty"},
        (
            500,
            # (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            (-3333250.0, 3333250.0, -3333250.0, 3333250.0),
            7,
            306,
            "g",
        ),
    ),
    (
        {"layer": "ice_thickness_uncertainty"},
        (
            500,
            # (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            (-3333250.0, 3333250.0, -3333250.0, 3333250.0),
            0,
            272,
            "g",
        ),
    ),
    (
        {"layer": "mask"},
        (
            500,
            # (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            (-3333250.0, 3333250.0, -3333250.0, 3333250.0),
            1,
            4,
            "g",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected"), bedmap3_test)
def test_bedmap3(test_input, expected):
    grid = fetch.bedmap3(**test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_bedmap3_reference():
    # fetch variations of grids and reference models
    region = (-101e3, -100e3, -51e3, -50e3)
    eigen_gl04c_grid = fetch.bedmap2(
        layer="gl04c_geiod_to_WGS84",
        region=region,
    )
    eigen_6c4_grid = fetch.geoid(
        region=region,
        spacing=1e3,
        hemisphere="south",
    )
    surface_6c4_grid = fetch.bedmap3(
        layer="surface",
        reference="eigen-6c4",
        region=region,
    )
    surface_ellipsoid_grid = fetch.bedmap3(
        layer="surface",
        reference="ellipsoid",
        region=region,
    )
    surface_gl04c_grid = fetch.bedmap3(
        layer="surface",
        reference="eigen-gl04c",
        region=region,
    )
    # get mean values
    eigen_gl04c = np.nanmean(eigen_gl04c_grid)
    eigen_6c4 = np.nanmean(eigen_6c4_grid)

    surface_6c4 = np.nanmean(surface_6c4_grid)
    surface_ellipsoid = np.nanmean(surface_ellipsoid_grid)
    surface_gl04c = np.nanmean(surface_gl04c_grid)

    assert surface_ellipsoid - eigen_6c4 == pytest.approx(surface_6c4, rel=0.1)
    assert surface_ellipsoid - eigen_gl04c == pytest.approx(surface_gl04c, rel=0.1)
    assert surface_gl04c + eigen_gl04c == pytest.approx(surface_ellipsoid, rel=0.1)
    assert surface_6c4 + eigen_6c4 == pytest.approx(surface_ellipsoid, rel=0.1)


bedmap3_fill_nans_test = [
    (
        {"layer": "surface"},
        [1962.4438, 601.9953],
    ),
    (
        {"layer": "ice_thickness"},
        [1943.3606, 596.1414],
    ),
    (
        {"layer": "icebase"},
        [19.08591, 5.8547554],
    ),
]


@pytest.mark.fetch
@pytest.mark.parametrize(("test_input", "expected"), bedmap3_fill_nans_test)
def test_bedmap3_fill_nans(test_input, expected):
    grid = fetch.bedmap3(**test_input)
    filled_grid = fetch.bedmap3(**test_input, fill_nans=True)
    assert np.nanmean(grid) == pytest.approx(expected[0], rel=0.1)
    assert np.nanmean(filled_grid) == pytest.approx(expected[1], rel=0.1)


# %% bedmap2
# test for all layers, but only test reference models with 1 layer and fill_nans with 1
# layer

bedmap2_test = [
    (
        {"layer": "bed"},
        (1000, (-3333000.0, 3333000.0, -3333000.0, 3333000.0), -7054.0, 3972.0, "g"),
    ),
    (
        {"layer": "coverage"},
        (1000, (-3333000.0, 3333000.0, -3333000.0, 3333000.0), 1.0, 1.0, "g"),
    ),
    (
        {"layer": "grounded_bed_uncertainty"},
        (1000, (-3333000.0, 3333000.0, -3333000.0, 3333000.0), 0.0, 65535.0, "g"),
    ),
    (
        {"layer": "icemask_grounded_and_shelves"},
        (1000, (-3333000.0, 3333000.0, -3333000.0, 3333000.0), 0.0, 1.0, "g"),
    ),
    (
        {"layer": "rockmask"},
        (1000, (-3333000.0, 3333000.0, -3333000.0, 3333000.0), 0.0, 0.0, "g"),
    ),
    (
        {"layer": "surface"},
        (1000, (-3333000.0, 3333000.0, -3333000.0, 3333000.0), 1.0, 4082.0, "g"),
    ),
    (
        {"layer": "thickness"},
        (1000, (-3333000.0, 3333000.0, -3333000.0, 3333000.0), 0.0, 4621.0, "g"),
    ),
    (
        {"layer": "icebase"},
        (1000, (-3333000.0, 3333000.0, -3333000.0, 3333000.0), -2736.0, 3972.0, "g"),
    ),
    (
        {"layer": "lakemask_vostok"},
        (1000, (1190000.0, 1470000.0, -402000.0, -291000.0), 1.0, 1.0, "g"),
    ),
    (
        {"layer": "thickness_uncertainty_5km"},
        (5000, (-3399000.0, 3401000.0, -3400000.0, 3400000.0), 0.0, 65535.0, "g"),
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected"), bedmap2_test)
def test_bedmap2(test_input, expected):
    grid = fetch.bedmap2(**test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_bedmap2_reference():
    # fetch variations of grids and reference models
    region = (-101e3, -100e3, -51e3, -50e3)
    eigen_gl04c_grid = fetch.bedmap2(
        layer="gl04c_geiod_to_WGS84",
        region=region,
    )
    eigen_6c4_grid = fetch.geoid(
        region=region,
        spacing=1e3,
        hemisphere="south",
    )
    surface_6c4_grid = fetch.bedmap2(
        layer="surface",
        reference="eigen-6c4",
        region=region,
    )
    surface_ellipsoid_grid = fetch.bedmap2(
        layer="surface",
        reference="ellipsoid",
        region=region,
    )
    surface_gl04c_grid = fetch.bedmap2(
        layer="surface",
        reference="eigen-gl04c",
        region=region,
    )
    # get mean values
    eigen_gl04c = np.nanmean(eigen_gl04c_grid)
    eigen_6c4 = np.nanmean(eigen_6c4_grid)

    surface_6c4 = np.nanmean(surface_6c4_grid)
    surface_ellipsoid = np.nanmean(surface_ellipsoid_grid)
    surface_gl04c = np.nanmean(surface_gl04c_grid)

    assert surface_ellipsoid - eigen_6c4 == pytest.approx(surface_6c4, rel=0.1)
    assert surface_ellipsoid - eigen_gl04c == pytest.approx(surface_gl04c, rel=0.1)
    assert surface_gl04c + eigen_gl04c == pytest.approx(surface_ellipsoid, rel=0.1)
    assert surface_6c4 + eigen_6c4 == pytest.approx(surface_ellipsoid, rel=0.1)


bedmap2_fill_nans_test = [
    (
        {"layer": "surface"},
        [1964.5349, 602.32306],
    ),
    (
        {"layer": "thickness"},
        [1926.642, 590.70514],
    ),
    (
        {"layer": "icebase"},
        [37.89277, 11.617859],
    ),
]


@pytest.mark.fetch
@pytest.mark.parametrize(("test_input", "expected"), bedmap2_fill_nans_test)
def test_bedmap2_fill_nans(test_input, expected):
    grid = fetch.bedmap2(**test_input)
    filled_grid = fetch.bedmap2(**test_input, fill_nans=True)
    assert np.nanmean(grid) == pytest.approx(expected[0], rel=0.1)
    assert np.nanmean(filled_grid) == pytest.approx(expected[1], rel=0.1)


# grid = fetch.bedmap2(layer="surface")
# utils.get_grid_info(grid)
# np.nanmean(grid)

# %% Bedmap points


@pytest.mark.fetch
def test_bedmap_points():
    df = fetch.bedmap_points(
        version="all",
        region=regions.minna_bluff,
    )
    expected = [
        166.26541073215904,
        -78.3583361446912,
        102.86136846208544,
        280.40327021647704,
        -54.59116960358044,
        828.8807757122288,
        300618.8269781072,
        -1232077.35794502,
    ]
    assert df.describe().iloc[1].tolist() == pytest.approx(
        expected,
        rel=0.1,
    )


# df = fetch.bedmap_points(version='bedmap1')
# df.describe().iloc[1].tolist()


# %% deepbedmap


@pytest.mark.fetch
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_deepbedmap():
    grid = fetch.deepbedmap()
    expected = (
        250,
        (-2700000.0, 2800000.0, -2199750.0, 2299750.0),
        -6156.0,
        4215.0,
        "p",
    )
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# grid = fetch.deepbedmap()
# utils.get_grid_info(grid)

# %% gravity
# only testing 1 anomaly type (FA) for each version

gravity_test = [
    (
        "antgg",
        (
            10000,
            (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            -384.5,
            204.800003052,
            "g",
        ),
        None,
    ),
    (
        "antgg-2021",
        (
            5000,
            (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            -250.750549316,
            308.477203369,
            "g",
        ),
        None,
    ),
    (
        "eigen",
        (
            5000.0,
            (-3500000.0, 3500000.0, -3500000.0, 3500000.0),
            977667.5625,
            980167.75,
            "g",
        ),
        "south",
    ),
    (
        "eigen",
        (
            5000.0,
            (-3500000.0, 3500000.0, -3500000.0, 3500000.0),
            977640.875,
            980201.9375,
            "g",
        ),
        "north",
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected", "hemisphere"), gravity_test)
def test_gravity(test_input, expected, hemisphere):
    grid = fetch.gravity(test_input, anomaly_type="FA", hemisphere=hemisphere)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    if test_input == "antgg":
        grid = grid.free_air_anomaly
    if test_input == "antgg-2021":
        grid = grid.free_air_anomaly
    if test_input == "eigen":
        grid = grid.gravity

    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# grid = fetch.gravity(version='eigen')
# utils.get_grid_info(grid)


# %% magnetics

magnetics_test = [
    (
        "admap1",
        (
            5000,
            (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            -936.875366211,
            1766.1373291,
            "g",
        ),
        None,
    ),
    (
        "admap2",
        (
            1500.0,
            (-3423000.0, 3426000.0, -3424500.0, 3426000.0),
            -2827.82373047,
            4851.73828125,
            "g",
        ),
        None,
    ),
    (
        "LCS-1",
        (
            10000.0,
            (-3500000.0, 3500000.0, -3500000.0, 3500000.0),
            -316.829925537,
            601.757263184,
            "g",
        ),
        "south",
    ),
    (
        "LCS-1",
        (
            10000.0,
            (-3500000.0, 3500000.0, -3500000.0, 3500000.0),
            -359.858154297,
            570.593200684,
            "g",
        ),
        "north",
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected", "hemisphere"), magnetics_test)
def test_magnetics(test_input, expected, hemisphere):
    grid = fetch.magnetics(test_input, hemisphere=hemisphere)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# grid = fetch.magnetics(version='admap1')
# utils.get_grid_info(grid)

# %% mass change

mass_change_test = [
    (
        "ais_dhdt_floating",
        (
            5001.18699879,
            (-2521652.10412, 2843360.03282, -2229531.47932, 2336552.25058),
            -9.33203697205,
            11.7978229523,
            "p",
        ),
    ),
    (
        "ais_dmdt_grounded",
        (
            5000.0,
            (-2526157.06916, 2648842.93084, -2124966.01441, 2180033.98559),
            -27.9888286591,
            1.0233386755,
            "p",
        ),
    ),
    (
        "gris_dhdt",
        (
            5000.0,
            (-626302.876984, 818697.123016, -3236440.11184, -681440.111839),
            -58.5592041016,
            1.15196788311,
            "p",
        ),
    ),
    (
        "gris_dmdt_filt",
        (
            5000.0,
            (-626302.876984, 818697.123016, -3236440.11184, -681440.111839),
            -36.8038787842,
            0.690221190453,
            "p",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected"), mass_change_test)
def test_mass_change(test_input, expected):
    grid = fetch.mass_change(version=test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# %% basal melt

basal_melt_test = [
    (
        "w_b",
        (
            500.0,
            (-2736000.0, 2734000.0, -2374000.0, 2740000.0),
            -19.8239116669,
            272.470550537,
            "g",
        ),
    ),
]


@pytest.mark.parametrize(("test_input", "expected"), basal_melt_test)
@pytest.mark.fetch
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_basal_melt(test_input, expected):
    grid = fetch.basal_melt(variable=test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# %% ghf

ghf_test = [
    (
        "an-2015",
        (
            5000,
            (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            26.5547351837,
            102.389022827,
            "g",
        ),
    ),
    (
        "martos-2017",
        (
            15000,
            (-2535000.0, 2715000.0, -2130000.0, 2220000.0),
            42.6263694763,
            240.510910034,
            "g",
        ),
    ),
    (
        "burton-johnson-2020",
        (
            17000,
            (-2543500.0, 2624500.0, -2121500.0, 2213500.0),
            42.2533454895,
            106.544433594,
            "p",
        ),
    ),
    (
        "losing-ebbing-2021",
        (
            5000,
            (-2990000.0, 2990000.0, -2990000.0, 2990000.0),
            24.609621048,
            144.53793335,
            "g",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected"), ghf_test)
def test_ghf(test_input, expected):
    grid = fetch.ghf(test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=0,
    )


@pytest.mark.fetch
def test_ghf_points():
    df = fetch.ghf(version="burton-johnson-2020", points=True)
    expected = [
        -56.5667,
        34.1833,
        "C11-44",
        0.0,
        11,
        300,
        0.77,
        229.0,
        -5372.0,
        "Anderson1977",
        "https://doi.org/10.1594/PANGAEA.796541",
        "S3",
        "Unconsolidated sediments",
        2098568.3517061966,
        3089886.43259545,
    ]

    assert df.iloc[0].dropna().tolist() == pytest.approx(expected, rel=0.1)


# df = fetch.ghf(version='burton-johnson-2020', points=True)
# df.iloc[0].dropna().tolist()

# grid = fetch.ghf(version='burton-johnson-2020')
# utils.get_grid_info(grid)

# %% gia

gia_test = [
    (
        "stal-2020",
        (
            10000,
            (-2800000.0, 2800000.0, -2800000.0, 2800000.0),
            -2953.8605957,
            3931.43554688,
            "p",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected"), gia_test)
def test_gia(test_input, expected):
    grid = fetch.gia(test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# grid = fetch.gia(version='stal-2020')
# utils.get_grid_info(grid)

# %% crustal_thickness

crust_test = [
    # (
    #     "shen-2018",
    #     (
    #         10000,
    #         (-2800000.0, 2800000.0, -2800000.0, 2800000.0),
    #         17216.1484375,
    #         57233.3320313,
    #         "g",
    #     ),
    # ),
    (
        "an-2015",
        (
            5000,
            (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            6264.58496094,
            65197.1328125,
            "g",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected"), crust_test)
def test_crustal_thickness(test_input, expected):
    grid = fetch.crustal_thickness(test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# grid = fetch.crustal_thickness(version='an-2015')
# utils.get_grid_info(grid)

# %% moho

moho_test = [
    (
        "shen-2018",
        (
            10000,
            (-2800000.0, 2800000.0, -2800000.0, 2800000.0),
            -57223.5273438,
            -17218.0996094,
            "g",
        ),
    ),
    (
        "an-2015",
        (
            5000,
            (-3330000.0, 3330000.0, -3330000.0, 3330000.0),
            -65197.1328125,
            -6264.58496094,
            "g",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected"), moho_test)
def test_moho(test_input, expected):
    grid = fetch.moho(test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# grid = fetch.moho(version='shen-2018')
# utils.get_grid_info(grid)

# %% geoid

geoid_test = [
    (
        "north",
        (
            5000,
            (-3500000.0, 3500000.0, -3500000.0, 3500000.0),
            -52.0835571289,
            68.4151992798,
            "g",
        ),
    ),
    (
        "south",
        (
            5000,
            (-3500000.0, 3500000.0, -3500000.0, 3500000.0),
            -66.1241073608,
            52.2200813293,
            "g",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected"), geoid_test)
def test_geoid(test_input, expected):
    grid = fetch.geoid(hemisphere=test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# grid = fetch.geoid()
# utils.get_grid_info(grid)

# %% etopo


etopo_test = [
    (
        "north",
        (
            5000,
            (-3500000.0, 3500000.0, -3500000.0, 3500000.0),
            -7265.75048828,
            3226.35986328,
            "g",
        ),
    ),
    (
        "south",
        (
            5000,
            (-3500000.0, 3500000.0, -3500000.0, 3500000.0),
            -7946.32617188,
            4089.17773438,
            "g",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected"), etopo_test)
def test_etopo(test_input, expected):
    grid = fetch.etopo(hemisphere=test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# grid = fetch.etopo()
# utils.get_grid_info(grid)

# %% REMA


rema_test = [
    (
        {"version": "500m"},
        (
            500,
            (-2700250.0, 2750250.0, -2500250.0, 3342250.0),
            -66.4453125,
            4702.28125,
            "g",
        ),
    ),
    (
        {"version": "1km"},
        (
            1000,
            (-2700500.0, 2750500.0, -2500500.0, 3342500.0),
            -66.4453125,
            4639.3125,
            "g",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("test_input", "expected"), rema_test)
def test_rema(test_input, expected):
    grid = fetch.rema(**test_input)
    # assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


# grid = fetch.rema(version="500m")
# utils.get_grid_info(grid)

# %% Buttressing


buttressing_test = [
    (
        {"version": "max"},
        (
            1000,
            (-2750000.0, 2750000.0, -2750000.0, 2750000.0),
            -781.639465332,
            717.836303711,
            "g",
        ),
    ),
    (
        {"version": "min"},
        (
            1000,
            (-2750000.0, 2750000.0, -2750000.0, 2750000.0),
            -1537.85681152,
            83.4083175659,
            "g",
        ),
    ),
    (
        {"version": "flow"},
        (
            1000,
            (-2750000.0, 2750000.0, -2750000.0, 2750000.0),
            -1433.67492676,
            102.327476501,
            "g",
        ),
    ),
    (
        {"version": "viscosity"},
        (
            1000,
            (-2750000.0, 2750000.0, -2750000.0, 2750000.0),
            6.90586421115e-06,
            384.891693115,
            "g",
        ),
    ),
]


@pytest.mark.fetch
@pytest.mark.parametrize(("test_input", "expected"), buttressing_test)
def test_buttressing(test_input, expected):
    grid = fetch.buttressing(**test_input)
    print("\n\n\n GRID INFO", utils.get_grid_info(grid))
    assert not deepdiff.DeepDiff(
        utils.get_grid_info(grid),
        expected,
        ignore_order=True,
        significant_digits=2,
    )


groundingline_test = [
    (
        "depoorter-2013",
        {
            "Id": 4,
            "Id_text": "Ice shelf",
            "Area_km2": 60302.95,
        },
    ),
    (
        "measures-v2",
        {
            "NAME": "Grounded",
            "TYPE": "GR",
        },
    ),
    (
        "measures-greenland",
        {
            "GM_LABEL": "SEACST",
            "GM_SOURCE_": "KMS",
        },
    ),
]


@pytest.mark.fetch
@pytest.mark.parametrize(("test_input", "expected"), groundingline_test)
def test_groundingline(test_input, expected):
    # download file and read first row
    df = gpd.read_file(
        fetch.groundingline(version=test_input),
        rows=1,
        engine=ENGINE,
    ).drop(columns=["geometry"])

    # df_expected = pd.DataFrame(expected)
    # assert_assert_frame_equal(df, df_expected)

    if test_input == "depoorter-2013":
        assert df.Id.iloc[0] == expected["Id"]
        assert df.Id_text.iloc[0] == expected["Id_text"]
        assert df.Area_km2.iloc[0] == expected["Area_km2"]
    elif test_input == "measures-v2":
        assert df.NAME.iloc[0] == expected["NAME"]
        assert df.TYPE.iloc[0] == expected["TYPE"]
    elif test_input == "measures-greenland":
        assert df.GM_LABEL.iloc[0] == expected["GM_LABEL"]
        assert df.GM_SOURCE_.iloc[0] == expected["GM_SOURCE_"]


antarctic_boundaries_test = [
    (
        "Coastline",
        {
            "NAME": "Coastline",
        },
    ),
    (
        "Basins_Antarctica",
        {
            "NAME": "LarsenE",
            "Regions": "Peninsula",
        },
    ),
]


@pytest.mark.fetch
@pytest.mark.parametrize(("test_input", "expected"), antarctic_boundaries_test)
def test_antarctic_boundaries(test_input, expected):
    # download file and read first row
    df = gpd.read_file(
        fetch.antarctic_boundaries(version=test_input),
        rows=1,
        engine=ENGINE,
    ).drop(columns=["geometry"])

    # df_expected = pd.DataFrame(expected)
    # assert_assert_frame_equal(df, df_expected)

    if test_input == "Coastline":
        assert df.NAME.iloc[0] == expected["NAME"]
    elif test_input == "Basins_Antarctica":
        assert df.NAME.iloc[0] == expected["NAME"]
        assert df.Regions.iloc[0] == expected["Regions"]


# %%
