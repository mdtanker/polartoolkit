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
import pandas as pd
import pandas.testing as pdt
import pytest
import verde as vd
import xarray as xr

from polartoolkit import profiles

################
################
# DUMMY FUNCTIONS
################
################


def dummy_grid() -> xr.Dataset:
    (x, y, z) = vd.grid_coordinates(
        region=[000, 200, 200, 400],
        spacing=100,
        extra_coords=20,
    )
    # create topographic features
    data = y**2 + x**2
    return vd.make_xarray_grid(
        (x, y),
        (data, z),
        data_names=("scalars", "upward"),
        dims=("northing", "easting"),
    )


################
################
# TESTS
################
################


def test_old_profile_module():
    """
    Check error raise after import old profile module
    """
    with pytest.raises(ImportError) as exception:
        from polartoolkit import profile  # noqa: F401 PLC0415
    assert "'profile'" in str(exception.value)


default_layers_test = [
    (
        {
            "version": "bedmap2",
            "hemisphere": "south",
            "spacing": 5e3,
        },
        [  # mean values of grids
            602.32306,
            11.617859,
            -2127.135,
        ],
    ),
    (
        {
            "version": "bedmap3",
            "hemisphere": "south",
            "spacing": 5e3,
        },
        [  # mean values of grids
            602.32306,
            5.658511,
            -2127.135,
        ],
    ),
    (
        {
            "version": "bedmachine",
            "hemisphere": "south",
            "spacing": 5e3,
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
            "spacing": 5e3,
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


def test_sample_grids_on_nodes():
    """
    Test if the sampled column contains valid values at grid nodes.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    df = pd.DataFrame({"easting": [0, 100, 200], "northing": [200, 300, 400]})
    result_df = profiles.sample_grids(df, grid, sampled_name=name)
    expected = pd.DataFrame(
        {
            "easting": [0, 100, 200],
            "northing": [200, 300, 400],
            name: [40000, 100000, 200000],
        },
    )
    pdt.assert_frame_equal(
        result_df,
        expected,
        check_dtype=False,
    )


def test_sample_grids_off_nodes():
    """
    Test if the sampled column contains valid values not on grid nodes.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    df = pd.DataFrame({"easting": [50, 101], "northing": [280, 355]})
    result_df = profiles.sample_grids(df, grid, sampled_name=name)
    expected = pd.DataFrame(
        {"easting": [50, 101], "northing": [280, 355], name: [83790.0, 138949.640109]}
    )
    pdt.assert_frame_equal(
        result_df,
        expected,
        check_dtype=False,
    )


def test_sample_grids_custom_coordinate_names():
    """
    Test if the function handles custom coordinate names correctly.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    df = pd.DataFrame({"lon": [0, 100, 200], "lat": [200, 300, 400]})
    # check function raises KeyError if coordinate names are not found in the grid
    with pytest.raises(KeyError):
        profiles.sample_grids(df, grid, sampled_name=name)
    # check function works if coordinate names are provided
    result_df = profiles.sample_grids(
        df, grid, sampled_name=name, coord_names=("lon", "lat")
    )
    expected = pd.DataFrame(
        {"lon": [0, 100, 200], "lat": [200, 300, 400], name: [40000, 100000, 200000]},
        dtype="int64",
    )
    pdt.assert_frame_equal(
        result_df,
        expected,
        check_dtype=False,
    )


def test_sample_grids_one_out_of_grid_coordinates():
    """
    Test if the function handles out-of-grid coordinates gracefully by setting NaN
    values.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    df = pd.DataFrame({"easting": [0, -1000, 200], "northing": [200, -1000, 400]})
    result_df = profiles.sample_grids(df, grid, sampled_name=name)
    assert np.isnan(result_df[name].iloc[1])


def test_sample_grids_first_out_of_grid_coordinates():
    """
    Test if the function handles out-of-grid coordinates gracefully by setting NaN
    values.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    df = pd.DataFrame({"easting": [-50, 150, 200], "northing": [500, 350, 400]})
    result_df = profiles.sample_grids(df, grid, sampled_name=name)
    assert np.isnan(result_df[name].iloc[0])


def test_sample_grids_last_out_of_grid_coordinates():
    """
    Test if the function handles out-of-grid coordinates gracefully by setting NaN
    values.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    df = pd.DataFrame({"easting": [200, 150, 0], "northing": [200, 350, 0]})
    result_df = profiles.sample_grids(df, grid, sampled_name=name)
    assert np.isnan(result_df[name].iloc[2])


def test_sample_grids_all_out_of_grid_coordinates_all():
    """
    Test if the function handles out-of-grid coordinates gracefully by setting NaN
    values.
    """
    grid = dummy_grid().scalars
    name = "sampled_data"
    points = pd.DataFrame({"easting": [-100, -200, -300], "northing": [500, 1000, 600]})
    result_df = profiles.sample_grids(points, grid, sampled_name=name)
    assert result_df[name].isna().all()  # All values should be NaN


def test_sample_bounding_surfaces_valid_values():
    """
    Ensure that correct values are sampled, including a NaN
    """
    lower_confining_layer = dummy_grid().scalars
    points = pd.DataFrame({"easting": [0, -100, 200], "northing": [200, -300, 400]})
    result_df = profiles.sample_grids(
        points,
        lower_confining_layer,
        sampled_name="sampled",
    )
    expected = pd.DataFrame(
        {
            "easting": [0, -100, 200],
            "northing": [200, -300, 400],
            "sampled": [40000, np.nan, 200000],
        }
    )
    pdt.assert_frame_equal(result_df, expected)
