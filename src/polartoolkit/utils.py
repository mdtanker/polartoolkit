# Copyright (c) 2024 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#
# pylint: disable=too-many-lines
from __future__ import annotations

import os
import typing
import warnings

import deprecation
import geopandas as gpd
import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
import xrft
from numpy.typing import NDArray
from pyproj import Transformer

import polartoolkit
from polartoolkit import fetch, logger, maps, regions

try:
    import pyogrio  # pylint: disable=unused-import

    ENGINE = "pyogrio"
except ImportError:
    pyogrio = None
    ENGINE = "fiona"


def default_hemisphere(hemisphere: str | None) -> str:
    """
    Returns the default hemisphere set in the users environment variables or raises a
    error.

    Parameters
    ----------
    hemisphere : str | None
        hemisphere to use, either "north" or "south", or None to use the default set in
        the users environment variables.

    Returns
    -------
    str
        hemisphere to use, either "north" or "south"
    """

    if hemisphere is None:
        try:
            return os.environ["POLARTOOLKIT_HEMISPHERE"]
        except KeyError as e:
            msg = (
                "hemisphere not set, either set it as a temp environment variable in "
                "python (os.environ['POLARTOOLKIT_HEMISPHERE']='north'), set it as a "
                "permanent operating system environment variable (i.e. for Unix, add "
                "'export POLARTOOLKIT_HEMISPHERE=south' to the end of your .bashrc "
                "file) or pass it as an argument (hemisphere='north')"
            )
            raise KeyError(msg) from e
    return hemisphere


def rmse(data: typing.Any, as_median: bool = False) -> float:
    """
    function to give the root mean/median squared error (RMSE) of data

    Parameters
    ----------
    data : numpy.ndarray[typing.Any, typing.Any]
        input data
    as_median : bool, optional
        choose to give root median squared error instead, by default False

    Returns
    -------
    float
        RMSE value
    """
    if as_median:
        value: float = np.sqrt(np.nanmedian(data**2).item())
    else:
        value = np.sqrt(np.nanmean(data**2).item())

    return value


def get_grid_info(
    grid: str | xr.DataArray,
    print_info: bool = False,
) -> tuple[
    float | None,
    tuple[float, float, float, float] | None,
    float | None,
    float | None,
    str | None,
]:
    """
    Returns information of the specified grid.

    Parameters
    ----------
    grid : str or xarray.DataArray
        Input grid to get info from. Filename string or loaded grid.
    print_info : bool, optional
        If true, prints out the grid info, by default False

    Returns
    -------
    tuple
        (string of grid spacing,
        array with the region boundary,
        data min,
        data max,
        grid registration)
    """

    # if isinstance(grid, str):
    # grid = xr.load_dataarray(grid)
    # try:
    # grid = xr.load_dataarray(grid).squeeze()
    # except ValueError:
    # print("loading grid as dataarray didn't work")
    # raise
    # pass
    # grid = xr.open_rasterio(grid)
    # grid = rioxarray.open_rasterio(grid)
    if isinstance(grid, xr.DataArray) and int(len(grid.dims)) > 2:
        grid = grid.squeeze()

    try:
        spacing: float | None = float(pygmt.grdinfo(grid, per_column="n", o=7)[:-1])
    except Exception as e:  # pylint: disable=broad-exception-caught
        # pygmt.exceptions.GMTInvalidInput:
        logger.exception(e)
        logger.warning("grid spacing can't be extracted")
        spacing = None

    try:
        region: typing.Any = tuple(
            float(pygmt.grdinfo(grid, per_column="n", o=i)[:-1]) for i in range(4)
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        # pygmt.exceptions.GMTInvalidInput:
        logger.exception(e)
        logger.warning("grid region can't be extracted")
        region = None

    try:
        zmin: float | None = float(pygmt.grdinfo(grid, per_column="n", o=4)[:-1])
    except Exception as e:  # pylint: disable=broad-exception-caught
        # pygmt.exceptions.GMTInvalidInput:
        logger.exception(e)
        logger.warning("grid zmin can't be extracted")
        zmin = None

    try:
        zmax = float(pygmt.grdinfo(grid, per_column="n", o=5)[:-1])
    except Exception as e:  # pylint: disable=broad-exception-caught
        # pygmt.exceptions.GMTInvalidInput:
        logger.exception(e)
        logger.warning("grid zmax can't be extracted")
        zmax = None

    try:
        reg = grid.gmt.registration  # type: ignore[union-attr]
        registration: str | None = "g" if reg == 0 else "p"
    except AttributeError:
        logger.warning(
            "grid registration not extracted, re-trying with file loaded as xarray grid"
        )
        # grid = xr.load_dataarray(grid)
        with xr.open_dataarray(grid) as da:
            try:
                reg = da.gmt.registration
                registration = "g" if reg == 0 else "p"
            except AttributeError:
                logger.warning("grid registration can't be extracted, setting to 'g'.")
                registration = "g"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(e)
        logger.warning("grid registration can't be extracted")
        registration = None

    if print_info:
        info = (
            f"grid spacing: {spacing} m\n"
            f"grid region: {region}\n"
            f"grid zmin: {zmin}\n"
            f"grid zmax: {zmax}\n"
            f"grid registration: {registration}"
        )
        print(info)  # noqa: T201

    return spacing, region, zmin, zmax, registration


def dd2dms(dd: float) -> str:
    """
    Convert decimal degrees to minutes, seconds. Modified from
    https://stackoverflow.com/a/10286690/18686384

    Parameters
    ----------
    dd : float
        input decimal degrees

    Returns
    -------
    str
        degrees in the format "DD:MM:SS"
    """
    is_positive = dd >= 0
    dd = abs(dd)
    minutes, seconds = divmod(dd * 3600, 60)
    degrees, minutes = divmod(minutes, 60)
    degrees = degrees if is_positive else -degrees
    return f"{int(degrees)}:{int(minutes)}:{seconds}"


def region_to_df(
    region: tuple[typing.Any, typing.Any, typing.Any, typing.Any] | pd.DataFrame,
    coord_names: tuple[str, str] = ("easting", "northing"),
    reverse: bool = False,
) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any] | pd.DataFrame:
    """
    Convert region bounds in format [xmin, xmax, ymin, ymax] to pandas dataframe with
    coordinates of region corners, or reverse this if `reverse` is True.

    Parameters
    ----------
    region : tuple[typing.Any, typing.Any, typing.Any, typing.Any] | pandas.DataFrame
        bounding region in format [xmin, xmax, ymin, ymax] or, if `reverse` is True,
        a DataFrame with coordinate columns with names set by `cood_names`
    coord_names : tuple[str, str], optional
        names of input or output coordinate columns, by default ("easting", "northing")
    reverse : bool, optional
        If True, convert from df to region tuple, else, convert from region tuple to df,
        by default False

    Returns
    -------
    tuple[typing.Any, typing.Any, typing.Any, typing.Any] | pandas.DataFrame
        Dataframe with easting and northing columns, and a row for each corner of the
        region, or, if `reverse` is True, an array in the format
        [xmin, xmax, ymin, ymax].
    """

    if reverse:
        return (
            region[coord_names[0]][0],  # type: ignore[call-overload]
            region[coord_names[0]][1],  # type: ignore[call-overload]
            region[coord_names[1]][0],  # type: ignore[call-overload]
            region[coord_names[1]][2],  # type: ignore[call-overload]
        )

    bl = (region[0], region[2])
    br = (region[1], region[2])
    tl = (region[0], region[3])
    tr = (region[1], region[3])
    return pd.DataFrame(data=[bl, br, tl, tr], columns=(coord_names[0], coord_names[1]))


def region_xy_to_ll(
    region: tuple[typing.Any, typing.Any, typing.Any, typing.Any],
    hemisphere: str | None = None,
    dms: bool = False,
) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
    """
    Convert region in format [xmin, xmax, ymin, ymax] in projected meters to lat / lon

    Parameters
    ----------
    hemisphere : str, optional,
        choose between the "north" or "south" hemispheres
    region : tuple[typing.Any, typing.Any, typing.Any, typing.Any]
        region boundaries in format [xmin, xmax, ymin, ymax] in meters
    dms: bool
        if True, will return results as deg:min:sec instead of decimal degrees, by
        default False

    Returns
    -------
    tuple[typing.Any, typing.Any, typing.Any, typing.Any]
        region boundaries in format [lon_min, lon_max, lat_min, lat_max]
    """
    hemisphere = default_hemisphere(hemisphere)

    df = region_to_df(region)
    if hemisphere == "north":
        df_proj = epsg3413_to_latlon(df, reg=True)
    elif hemisphere == "south":
        df_proj = epsg3031_to_latlon(df, reg=True)
    else:
        msg = "hemisphere must be 'north' or 'south'"
        raise ValueError(msg)

    return tuple([dd2dms(x) for x in df_proj] if dms is True else df_proj)


def region_ll_to_xy(
    region: tuple[float, float, float, float],
    hemisphere: str | None = None,
) -> tuple[float, float, float, float]:
    """
    Convert region in format [lon_min, lon_max, lat_min, lat_max] to projected meters in
    the north or south polar stereographic projections.

    Parameters
    ----------
    hemisphere : str, optional,
        choose between the "north" or "south" hemispheres
    region : tuple[float, float, float, float]
        region boundaries in format [xmin, xmax, ymin, ymax] in decimal degrees

    Returns
    -------
    tuple[float, float, float, float]
        region boundaries in format [x_min, x_max, y_min, y_max]
    """
    hemisphere = default_hemisphere(hemisphere)

    df = region_to_df(region, coord_names=("lon", "lat"))
    if hemisphere == "north":
        df_proj: tuple[float, float, float, float] = latlon_to_epsg3413(df, reg=True)
    elif hemisphere == "south":
        df_proj = latlon_to_epsg3031(df, reg=True)
    else:
        msg = "hemisphere must be 'north' or 'south'"
        raise ValueError(msg)
    return df_proj


def region_to_bounding_box(
    region: tuple[typing.Any, typing.Any, typing.Any, typing.Any],
) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
    """
    Convert region in format [xmin, xmax, ymin, ymax] to bounding box format used for
    icepyx:
    [
    lower left longitude,
    lower left latitude,
    upper right longitude,
    upper right latitude
    ]
    Same format as [xmin, ymin, xmax, ymax], used for `bbox` parameter of
    geopandas.read_file

    Parameters
    ----------
    region : tuple[typing.Any, typing.Any, typing.Any, typing.Any]
        region boundaries in format [xmin, xmax, ymin, ymax] in meters or degrees.

    Returns
    -------
    tuple[typing.Any, typing.Any, typing.Any, typing.Any]
        region boundaries in bounding box format.
    """
    return (region[0], region[2], region[1], region[3])


def epsg3031_to_latlon(
    df: pd.DataFrame | tuple[typing.Any],
    reg: bool = False,
    input_coord_names: tuple[str, str] | None = None,
    output_coord_names: tuple[str, str] = ("lon", "lat"),
) -> pd.DataFrame | tuple[typing.Any]:
    """
    Convert coordinates from EPSG:3031 Antarctic Polar Stereographic in meters to
    EPSG:4326 WGS84 in decimal degrees.

    Parameters
    ----------
    df : pandas.DataFrame or tuple[typing.Any]
        input dataframe with easting and northing columns, or tuple [x,y]
    reg : bool, optional
        if true, returns a GMT formatted region string, by default False
    input_coord_names : tuple | None, optional
        set names for input coordinate columns, by default ("x", "y") or
        ("easting", "northing")
    output_coord_names : tuple | None, optional
        set names for output coordinate columns, by default ("lon", "lat")

    Returns
    -------
    pandas.DataFrame or tuple[typing.Any]
        Updated dataframe with new latitude and longitude columns, numpy.ndarray in
        format [xmin, xmax, ymin, ymax], or tuple in format [lat, lon]
    """

    return reproject(
        df,
        "epsg:3031",
        "epsg:4326",
        reg=reg,
        input_coord_names=input_coord_names,
        output_coord_names=output_coord_names,
    )


def epsg3413_to_latlon(
    df: pd.DataFrame | tuple[typing.Any],
    reg: bool = False,
    input_coord_names: tuple[str, str] | None = None,
    output_coord_names: tuple[str, str] = ("lon", "lat"),
) -> pd.DataFrame | tuple[typing.Any]:
    """
    Convert coordinates from EPSG:3413 North Polar Stereographic in meters to
    EPSG:4326 WGS84 in decimal degrees.

    Parameters
    ----------
    df : pandas.DataFrame or tuple[typing.Any]
        input dataframe with easting and northing columns, or tuple [x,y]
    reg : bool, optional
        if true, returns a GMT formatted region string, by default False
    input_coord_names : tuple | None, optional
        set names for input coordinate columns, by default ("x", "y") or
        ("easting", "northing")
    output_coord_names : tuple | None, optional
        set names for output coordinate columns, by default ("lon", "lat")

    Returns
    -------
    pandas.DataFrame or tuple[typing.Any]
        Updated dataframe with new latitude and longitude columns, numpy.ndarray in
        format [xmin, xmax, ymin, ymax], or tuple in format [lat, lon]
    """

    return reproject(
        df,
        "epsg:3413",
        "epsg:4326",
        reg=reg,
        input_coord_names=input_coord_names,
        output_coord_names=output_coord_names,
    )


def latlon_to_epsg3031(
    df: pd.DataFrame | NDArray[typing.Any, typing.Any],
    reg: bool = False,
    input_coord_names: tuple[str, str] | None = None,
    output_coord_names: tuple[str, str] = ("easting", "northing"),
) -> pd.DataFrame | NDArray[typing.Any, typing.Any]:
    """
    Convert coordinates from EPSG:4326 WGS84 in decimal degrees to EPSG:3031 Antarctic
    Polar Stereographic in meters.

    Parameters
    ----------
    df : pandas.DataFrame or numpy.ndarray[typing.Any, typing.Any]
        input dataframe with latitude and longitude columns
    reg : bool, optional
        if true, returns a GMT formatted region string, by default False
    input_coord_names : tuple | None, optional
        set names for input coordinate columns, by default ("lon", "lat")
    output_coord_names : tuple | None, optional
        set names for output coordinate columns, by default ("easting", "northing")

    Returns
    -------
    pandas.DataFrame or numpy.ndarray[typing.Any, typing.Any]
        Updated dataframe with new easting and northing columns or numpy.ndarray in
        format [xmin, xmax, ymin, ymax]
    """

    return reproject(
        df,
        "epsg:4326",
        "epsg:3031",
        reg=reg,
        input_coord_names=input_coord_names,
        output_coord_names=output_coord_names,
    )


def latlon_to_epsg3413(
    df: pd.DataFrame | NDArray[typing.Any, typing.Any],
    reg: bool = False,
    input_coord_names: tuple[str, str] | None = None,
    output_coord_names: tuple[str, str] = ("easting", "northing"),
) -> pd.DataFrame | NDArray[typing.Any, typing.Any]:
    """
    Convert coordinates from EPSG:4326 WGS84 in decimal degrees to EPSG:3413 North Polar
    Stereographic in meters.

    Parameters
    ----------
    df : pandas.DataFrame or numpy.ndarray[typing.Any, typing.Any]
        input dataframe with latitude and longitude columns
    reg : bool, optional
        if true, returns a GMT formatted region string, by default False
    input_coord_names : tuple | None, optional
        set names for input coordinate columns, by default ("lon", "lat")
    output_coord_names : tuple | None, optional
        set names for output coordinate columns, by default ("easting", "northing")

    Returns
    -------
    pandas.DataFrame or numpy.ndarray[typing.Any, typing.Any]
        Updated dataframe with new easting and northing columns or numpy.ndarray in
        format [xmin, xmax, ymin, ymax]
    """

    return reproject(
        df,
        "epsg:4326",
        "epsg:3413",
        reg=reg,
        input_coord_names=input_coord_names,
        output_coord_names=output_coord_names,
    )


def reproject(
    df: pd.DataFrame | tuple[typing.Any],
    input_crs: str,
    output_crs: str,
    input_coord_names: tuple[str, str] | None = None,
    output_coord_names: tuple[str, str] | None = None,
    reg: bool = False,
) -> pd.DataFrame | tuple[typing.Any]:
    """
    Convert coordinates from input CRS to output CRS. Coordinates can be supplied as a
    dataframe with coordinate columns set by input_coord_names, or as a tuple of a list
    of x coordinates and a list of y coordinates.

    Parameters
    ----------
    df : pandas.DataFrame or tuple[typing.Any]
        input dataframe with easting/longitude and northing/latitude columns, or tuple
        [x,y]
    input_crs : str
        input CRS in EPSG format, e.g. "epsg:4326"
    output_crs : str
        output CRS in EPSG format, e.g. "epsg:3413"
    reg : bool, optional
        if true, returns a GMT formatted region string, by default False
    input_coord_names : tuple, optional
        set names for input coordinate columns, by default "x"/"y" or
        "easting"/"northing" if input_crs is "epsg:3413" or "epsg:3031", or if input_crs
        is "epsg_4326", "lon"/"lat"
    output_coord_names : tuple, optional
        set names for output coordinate columns, by default "x"/"y" if output_crs is
        "epsg:3413" or "epsg:3031", or if output_crs is "epsg_4326", "lon"/"lat".

    Returns
    -------
    pandas.DataFrame or tuple[typing.Any]
        Updated dataframe with new latitude and longitude columns, numpy.ndarray in
        format [xmin, xmax, ymin, ymax], or tuple in format [lat, lon]
    """

    transformer = Transformer.from_crs(
        input_crs,
        output_crs,
        always_xy=True,
    )

    if isinstance(df, pd.DataFrame):
        df = df.copy()
        # use sensible default coord names
        if input_crs == "epsg:4326":
            if input_coord_names is None:
                input_coord_names = ("lon", "lat")
        else:
            if input_coord_names is None:
                # check for coord column names
                if ("x" in df.columns) and ("y" in df.columns):
                    input_coord_names = ("x", "y")
                elif ("easting" in df.columns) and ("northing" in df.columns):
                    input_coord_names = ("easting", "northing")

        if output_crs == "epsg:4326":
            if output_coord_names is None:
                output_coord_names = ("lon", "lat")
        else:
            if output_coord_names is None:
                # check for coord column names
                if ("x" in df.columns) and ("y" in df.columns):
                    output_coord_names = ("x", "y")
                elif ("easting" in df.columns) and ("northing" in df.columns):
                    output_coord_names = ("easting", "northing")
            if output_coord_names is None:
                output_coord_names = ("lon", "lat")
        (  # pylint: disable=unpacking-non-sequence
            df[output_coord_names[0]],
            df[output_coord_names[1]],
        ) = transformer.transform(
            df[input_coord_names[0]].tolist(),  # type: ignore[index]
            df[input_coord_names[1]].tolist(),  # type: ignore[index]
        )
        if reg is True:
            df = (
                df[output_coord_names[0]].min(),
                df[output_coord_names[0]].max(),
                df[output_coord_names[1]].min(),
                df[output_coord_names[1]].max(),
            )
    else:
        df = tuple(transformer.transform(df[0], df[1]))  # type: ignore[misc]
    return df


def points_inside_region(
    df: pd.DataFrame,
    region: tuple[float, float, float, float],
    names: tuple[str, str] = ("x", "y"),
    reverse: bool = False,
) -> pd.DataFrame:
    """
    return a subset of a dataframe which is within a region

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe with coordinate columns to use for defining if within region
    region : tuple[float, float, float, float]
        bounding region in format [xmin, xmax, ymin, ymax] for bounds of new subset
        dataframe
    names : tuple[str, str], optional
        column names to use for x and y coordinates, by default ("x", "y") or
        ("easting", "northing")
    reverse : bool, optional
        if True, will return points outside the region, by default False

    Returns
    -------
    pandas.DataFrame
       returns a subset dataframe
    """
    # make a copy of the dataframe
    df1 = df.copy()

    # check for coord column names
    if ("x" in df1.columns) and ("y" in df1.columns):
        pass
    elif ("easting" in df1.columns) and ("northing" in df1.columns):
        names = ("easting", "northing")

    # make column of booleans for whether row is within the region
    df1["inside_tmp"] = vd.inside(
        coordinates=(df1[names[0]], df1[names[1]]), region=region
    )

    if reverse is True:
        # subset if False
        df_result = df1.loc[df1.inside_tmp == False].copy()  # noqa: E712 # pylint: disable=singleton-comparison

    else:
        # subset if True
        df_result = df1.loc[df1.inside_tmp == True].copy()  # noqa: E712 # pylint: disable=singleton-comparison

    # drop the column 'inside'
    return df_result.drop(columns="inside_tmp")


def block_reduce(
    df: pd.DataFrame,
    reduction: typing.Callable[..., float | int],
    input_coord_names: tuple[str, str] = ("x", "y"),
    input_data_names: typing.Any | None = None,
    **kwargs: typing.Any,
) -> pd.DataFrame:
    """
    perform a block reduction of a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        data to block reduce
    reduction : typing.Callable
        function to use in reduction, e.g. np.mean
    input_coord_names : tuple[str, str], optional
        strings of coordinate column names, by default ("x", "y") or
        ("easting", "northing")
    input_data_names : typing.Any | None, optional
        strings of data column names, by default None

    Returns
    -------
    pandas.DataFrame
        a block-reduced dataframe
    """
    # define verde reducer function
    reducer = vd.BlockReduce(reduction, **kwargs)

    # check for coord column names
    if ("x" in df.columns) and ("y" in df.columns):
        pass
    elif ("easting" in df.columns) and ("northing" in df.columns):
        input_coord_names = ("easting", "northing")

    # if no data names provided, use all columns
    if input_data_names is None:
        input_data_names = tuple(df.columns.drop(list(input_coord_names)))

    # get tuples of pd.Series
    input_coords = tuple([df[col] for col in input_coord_names])  # pylint: disable=consider-using-generator
    input_data = tuple([df[col] for col in input_data_names])  # pylint: disable=consider-using-generator

    # apply reduction
    coordinates, data = reducer.filter(
        coordinates=input_coords,
        data=input_data,
    )

    # add reduced coordinates to a dictionary
    coord_cols = dict(zip(input_coord_names, coordinates))

    # add reduced data to a dictionary
    if len(input_data_names) < 2:
        data_cols = {input_data_names[0]: data}
    else:
        data_cols = dict(zip(input_data_names, data))

    # merge dicts and create dataframe
    return pd.DataFrame(data=coord_cols | data_cols)


def nearest_grid_fill(
    grid: xr.DataArray,
    method: str = "verde",
    crs: str | None = None,
) -> xr.DataArray:
    """
    fill missing values in a grid with the nearest value.

    Parameters
    ----------
    grid : xarray.DataArray
        grid with missing values
    method : str, optional
        choose method of filling, by default "verde"
    crs : str | None, optional
        if method is 'rioxarray', provide the crs of the grid, in format 'epsg:xxxx',
        by default None
    Returns
    -------
    xarray.DataArray
        filled grid
    """

    # get coordinate names
    original_dims = tuple(grid.sizes.keys())

    # get original grid name
    original_name = grid.name

    if method == "rioxarray":
        filled: xr.DataArray = (
            grid.rio.write_crs(crs)
            .rio.set_spatial_dims(original_dims[1], original_dims[0])
            .rio.write_nodata(np.nan)
            .rio.interpolate_na(method="nearest")
            .rename(original_name)
        )
    elif method == "verde":
        df = vd.grid_to_table(grid)
        df_dropped = df[df[grid.name].notnull()]
        coords = (df_dropped[grid.dims[1]], df_dropped[grid.dims[0]])
        region = vd.get_region((df[grid.dims[1]], df[grid.dims[0]]))
        filled = (
            vd.KNeighbors()
            .fit(coords, df_dropped[grid.name])
            .grid(region=region, shape=grid.shape, data_names=original_name)[
                original_name
            ]
        )
    # elif method == "pygmt":
    #     filled = pygmt.grdfill(grid, mode="n", verbose="q").rename(original_name)
    else:
        msg = "method must be 'rioxarray', or 'verde'"
        raise ValueError(msg)

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        return filled.rename(
            {
                next(iter(filled.dims)): original_dims[0],
                list(filled.dims)[1]: original_dims[1],
            }
        )


def filter_grid(
    grid: xr.DataArray,
    filter_width: float | None = None,
    filt_type: str = "lowpass",
    pad_width_factor: int = 3,
    pad_mode: str = "linear_ramp",
    pad_constant: float | None = None,
    pad_end_values: float | None = None,
) -> xr.DataArray:
    """
    Apply a spatial filter to a grid.

    Parameters
    ----------
    grid : xarray.DataArray
        grid to filter the values of
    filt_type : str, optional
        type of filter to use from 'lowpass', 'highpass' 'up_deriv', 'easting_deriv',
        'northing_deriv', or 'total_gradient' by default "lowpass"
    filt_type : str, optional
        type of filter to use, by default "lowpass"
    pad_width_factor : int, optional
        factor of grid width to pad the grid by, by default 3, which equates to a pad
        with a width of 1/3 of the grid width.
    pad_mode : str, optional
        mode of padding, can be "linear", by default "linear_ramp"
    pad_constant : float | None, optional
        constant value to use for padding, by default None
    pad_end_values : float | None, optional
        value to use for end of padding if pad_mode is "linear_ramp", by default None

    Returns
    -------
    xarray.DataArray
        a filtered grid
    """
    # get coordinate names
    original_dims = tuple(grid.sizes.keys())

    # get original grid name
    original_name = grid.name

    # if there are nan's, fill them with nearest neighbor
    if grid.isnull().any():
        filled = nearest_grid_fill(grid, method="verde")
    else:
        filled = grid.copy()

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        filled = filled.rename(
            {
                next(iter(filled.dims)): original_dims[0],
                list(filled.dims)[1]: original_dims[1],
            }
        )

    # define width of padding in each direction
    pad_width = {
        original_dims[1]: grid[original_dims[1]].size // pad_width_factor,
        original_dims[0]: grid[original_dims[0]].size // pad_width_factor,
    }

    if pad_mode == "constant":
        if pad_constant is None:
            pad_constant = filled.median()
        pad_end_values = None

    if (pad_mode == "linear_ramp") and (pad_end_values is None):
        pad_end_values = filled.median()

    if pad_mode != "constant":
        pad_constant = (
            None  # needed until https://github.com/xgcm/xrft/issues/211 is fixed
        )

    # apply padding
    pad_kwargs = {
        **pad_width,
        "mode": pad_mode,
        "constant_values": pad_constant,
        "end_values": pad_end_values,
    }

    padded = xrft.pad(
        filled,
        **pad_kwargs,
    )

    if filt_type == "lowpass":
        filt = hm.gaussian_lowpass(padded, wavelength=filter_width).rename("filt")
    elif filt_type == "highpass":
        filt = hm.gaussian_highpass(padded, wavelength=filter_width).rename("filt")
    elif filt_type == "up_deriv":
        filt = hm.derivative_upward(padded).rename("filt")
    elif filt_type == "easting_deriv":
        filt = hm.derivative_easting(padded).rename("filt")
    elif filt_type == "northing_deriv":
        filt = hm.derivative_northing(padded).rename("filt")
    elif filt_type == "total_gradient":
        filt = hm.total_gradient_amplitude(padded).rename("filt")
    else:
        msg = (
            "filt_type must be 'lowpass', 'highpass' 'up_deriv', 'easting_deriv', "
            "'northing_deriv', or 'total_gradient'"
        )
        raise ValueError(msg)

    unpadded = xrft.unpad(filt, pad_width)

    # reset coordinate values to original (avoid rounding errors)
    unpadded = unpadded.assign_coords(
        {
            original_dims[0]: grid[original_dims[0]].to_numpy(),
            original_dims[1]: grid[original_dims[1]].to_numpy(),
        }
    )

    if grid.isnull().any():
        result: xr.DataArray = xr.where(grid.notnull(), unpadded, grid)
    else:
        result = unpadded.copy()

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        result = result.rename(
            {
                next(iter(result.dims)): original_dims[0],
                # list(result.dims)[0]: original_dims[0],
                list(result.dims)[1]: original_dims[1],
            }
        )

    return result.rename(original_name)


def points_inside_shp(
    points: pd.DataFrame | gpd.geodataframe.GeoDataFrame,
    shapefile: gpd.geodataframe.GeoDataFrame,
    crs: str | None = None,
    coord_names: tuple[str, str] | None = None,
    hemisphere: str | None = None,
) -> pd.DataFrame | gpd.geodataframe.GeoDataFrame:
    """
    Add a column to a dataframe indicating whether each point is inside a shapefile.

    Parameters
    ----------
    points : pd.DataFrame | gpd.geodataframe.GeoDataFrame
        dataframe with coordinate columns specified by coord_names to use for defining
        if within shapefile
    shapefile : gpd.geodataframe.GeoDataFrame
        shapefile to use for defining if point are within it or not
    crs : str | None, optional
        if points is not a geodataframe, crs to use to convert into a geodataframe, by
        default None
    coord_names : tuple[str, str] | None, optional
        names of coordinate columns, by default 'x' and 'y' or 'easting' and 'northing'
    hemisphere : str | None, optional
        hemisphere to use for automatically detecting crs, by default None

    Returns
    -------
    pd.DataFrame | gpd.geodataframe.GeoDataFrame
        Dataframe with a new column 'inside' which is True if the point is inside the
        shapefile
    """
    points = points.copy()

    if isinstance(points, pd.DataFrame):
        if crs is None:
            hemisphere = default_hemisphere(hemisphere)
            if hemisphere == "north":
                crs = "epsg:3413"
            elif hemisphere == "south":
                crs = "epsg:3031"
            else:
                msg = "provide 'crs' or set hemisphere to 'north' or 'south'"
                raise ValueError(msg)

        if coord_names is None:
            # check for coord column names
            if ("x" in points.columns) and ("y" in points.columns):
                coord_names = ("x", "y")
            elif ("easting" in points.columns) and ("northing" in points.columns):
                coord_names = ("easting", "northing")

        points = gpd.GeoDataFrame(
            points,
            geometry=gpd.points_from_xy(
                x=points[coord_names[0]],  # type: ignore[index]
                y=points[coord_names[1]],  # type: ignore[index]
            ),
            crs=crs,
        )

    points["inside"] = points.within(shapefile.geometry[0])

    return points


def mask_from_shp(
    shapefile: str | gpd.geodataframe.GeoDataFrame,
    hemisphere: str | None = None,
    invert: bool = True,
    xr_grid: xr.DataArray | None = None,
    grid_file: str | None = None,
    region: str | tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    masked: bool = False,
    pixel_register: bool = True,
    input_coord_names: tuple[str, str] = ("easting", "northign"),
) -> xr.DataArray:
    """
    Create a mask or a masked grid from area inside or outside of a closed shapefile.

    Parameters
    ----------
    shapefile : str or geopandas.GeoDataFrame
        either path to .shp filename, must by in same directory as accompanying files :
        .shx, .prj, .dbf, should be a closed polygon file, or shapefile which as already
        been loaded into a geodataframe.
    hemisphere : str, optional
        choose "north" for EPSG:3413 or "south" for EPSG:3031
    invert : bool, optional
        choose whether to mask data outside the shape (False) or inside the shape
        (True), by default True (masks inside of shape)
    xr_grid : xarray.DataArray, optional
        _xarray.DataArray; to use to define region, or to mask, by default None
    grid_file : str, optional
        path to a .nc or .tif file to use to define region or to mask, by default None
    region : str or tuple[float, float, float, float], optional
        bounding region in format [xmin, xmax, ymin, ymax] in meters to create a dummy
        grid if none are supplied, by default None
    spacing : str or int, optional
        grid spacing in meters to create a dummy grid if none are supplied, by default
        None
    masked : bool, optional
        choose whether to return the masked grid (True) or the mask itself (False), by
        default False
    pixel_register : bool, optional
        choose whether the grid is pixel registered (True) or grid registered (False),
        by default True
    input_coord_names : tuple[str, str], optional
        set names for input coordinate columns, by default ("easting", "northing")

    Returns
    -------
    xarray.DataArray
        Returns either a masked grid, or the mask grid itself.
    """
    hemisphere = default_hemisphere(hemisphere)

    shp = (
        gpd.read_file(shapefile, engine=ENGINE)
        if isinstance(shapefile, str)
        else shapefile
    )

    if hemisphere == "north":
        crs = "epsg:3413"
    elif hemisphere == "south":
        crs = "epsg:3031"
    else:
        msg = "hemisphere must be 'north' or 'south'"
        raise ValueError(msg)

    if xr_grid is None and grid_file is None:
        coords = vd.grid_coordinates(
            region=region,
            spacing=spacing,
            pixel_register=pixel_register,
        )
        ds = vd.make_xarray_grid(
            coords,
            np.ones_like(coords[0]),
            dims=input_coord_names[::-1],
            data_names="z",
        )
        xds = ds.z.rio.write_crs(crs)
    elif xr_grid is not None:
        # get coordinate names
        original_dims = tuple(xr_grid.sizes.keys())
        xds = xr_grid.rio.write_crs(crs).rio.set_spatial_dims(
            original_dims[1], original_dims[0]
        )
    elif grid_file is not None:
        grid = xr.load_dataarray(grid_file)
        # get coordinate names
        original_dims = tuple(grid.sizes.keys())
        xds = grid.rio.write_crs(crs).rio.set_spatial_dims(
            original_dims[1], original_dims[0]
        )
    else:
        msg = "can't supply both xr_grid and grid_file."
        raise ValueError(msg)

    # if single geometry, convert to list
    try:
        iter(shp.geometry)
    except TypeError:
        geom = [shp.geometry]
    else:
        geom = shp.geometry

    masked_grd = xds.rio.clip(
        geom,
        xds.rio.crs,
        drop=False,
        invert=invert,
    )
    mask_grd = np.isfinite(masked_grd)

    if masked is True:
        output = masked_grd
    elif masked is False:
        output = mask_grd

    try:
        output = output.drop_vars("spatial_ref")  # pylint: disable=used-before-assignment
    except ValueError as e:
        logger.exception(e)

    return typing.cast(xr.DataArray, output)


@deprecation.deprecated(
    deprecated_in="0.4.0",
    removed_in="1.0.0",
    current_version=polartoolkit.__version__,
    details="alter_region has been moved to the regions module, use that instead",
)
def alter_region(
    starting_region: tuple[float, float, float, float],
    **kwargs: typing.Any,
) -> tuple[float, float, float, float]:
    """deprecated function, use regions.alter_region instead"""
    return regions.alter_region(
        starting_region,
        **kwargs,
    )


def set_proj(
    region: tuple[float, float, float, float],
    hemisphere: str | None = None,
    fig_height: float = 15,
    fig_width: float | None = None,
) -> tuple[str, str | None, float, float]:
    """
    Gives GMT format projection string from region and figure height or width.
    Inspired from https://github.com/mrsiegfried/Venturelli2020-GRL.

    Parameters
    ----------
    region : tuple[float, float, float, float]
        region boundaries in format [xmin, xmax, ymin, ymax] in projected meters
    hemisphere : str, optional
        set whether to lat lon projection is for "north" hemisphere (EPSG:3413) or
        "south" hemisphere (EPSG:3031)
    fig_height : float
        desired figure height in cm
    fig_width : float
        instead of using figure height, set the projection based on figure width in cm,
        by default is None

    Returns
    -------
    tuple
        returns a tuple of the following variables: proj, proj_latlon, fig_width,
        fig_height
    """
    try:
        hemisphere = default_hemisphere(hemisphere)
    except KeyError:
        hemisphere = None

    xmin, xmax, ymin, ymax = region

    if fig_width is not None:
        fig_height = fig_width * (ymax - ymin) / (xmax - xmin)
        ratio = (xmax - xmin) / (fig_width / 100)
    else:
        fig_width = fig_height * (xmax - xmin) / (ymax - ymin)
        ratio = (ymax - ymin) / (fig_height / 100)

    proj = f"x1:{ratio}"

    if hemisphere == "north":
        proj_latlon = f"s-45/90/70/1:{ratio}"
    elif hemisphere == "south":
        proj_latlon = f"s0/-90/-71/1:{ratio}"
    else:
        proj_latlon = None

    return proj, proj_latlon, fig_width, fig_height


def grd_trend(
    da: xr.DataArray,
    coords: tuple[str, str, str] = ("x", "y", "z"),
    deg: int = 1,
    plot: bool = False,
    **kwargs: typing.Any,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Fit an arbitrary order trend to a grid and use it to detrend.

    Parameters
    ----------
    da : xarray.DataArray
        input grid
    coords : tuple[str, str, str], optional
        coordinate names of the supplied grid, by default ['x', 'y', 'z']
    deg : int, optional
        trend order to use, by default 1
    plot : bool, optional
        plot the results, by default False

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray]
        returns xarray.DataArrays of the fitted surface, and the detrended grid.
    """

    # convert grid to a dataframe
    df0 = vd.grid_to_table(da).astype("float64")
    df = df0.dropna().copy()

    # define a trend
    trend = vd.Trend(degree=deg).fit((df[coords[0]], df[coords[1]]), df[coords[2]])

    # fit a trend to the grid of degree: deg
    df["fit"] = trend.predict((df[coords[0]], df[coords[1]]))

    # remove the trend from the data
    df["detrend"] = df[coords[2]] - df.fit

    info = get_grid_info(da)
    spacing = info[0]
    region = info[1]
    registration = info[4]

    fit = pygmt.xyz2grd(
        data=df[[coords[0], coords[1], "fit"]],
        region=region,
        spacing=spacing,
        registration=registration,
    )

    detrend = pygmt.xyz2grd(
        data=df[[coords[0], coords[1], "detrend"]],
        region=region,
        spacing=spacing,
        registration=registration,
    )

    if plot is True:
        cmap: typing.Any = kwargs.get("cmap", "plasma")
        coast: typing.Any = kwargs.get("coast", True)
        inset: typing.Any = kwargs.get("inset", True)
        inset_position: typing.Any = kwargs.get("inset_position", "jTL+jTL+o0/0")
        inset_pos: typing.Any = kwargs.get("inset_pos")
        origin_shift: typing.Any = kwargs.get("origin_shift", "y")
        fit_label: typing.Any = kwargs.get("fit_label", f"fitted trend (order {deg})")
        input_label: typing.Any = kwargs.get("input_label", "input grid")
        title: typing.Any = kwargs.get("title", "Detrending a grid")
        detrended_label: typing.Any = kwargs.get("detrended_label", "detrended")

        fig = maps.plot_grd(
            da,
            cbar_label=input_label,
            title=title,
            cmap=cmap,
            # grd2cpt=True,
            inset=inset,
            inset_position=inset_position,
            inset_pos=inset_pos,
            coast=coast,
            hist=True,
            robust=True,
            **kwargs,
        )

        fig = maps.plot_grd(
            fit,
            fig=fig,
            cmap=cmap,
            # grd2cpt=True,
            cbar_label=fit_label,
            origin_shift=origin_shift,
            coast=coast,
            hist=True,
            robust=True,
            **kwargs,
        )

        fig = maps.plot_grd(
            detrend,
            fig=fig,
            cmap=cmap,
            # grd2cpt=True,
            cbar_label=detrended_label,
            origin_shift=origin_shift,
            coast=coast,
            hist=True,
            robust=True,
            **kwargs,
        )

        fig.show()

    return fit, detrend


def get_combined_min_max(
    values: tuple[xr.DataArray | pd.Series | NDArray],
    shapefile: str | gpd.geodataframe.GeoDataFrame | None = None,
    robust: bool = False,
    region: tuple[float, float, float, float] | None = None,
    hemisphere: str | None = None,
    absolute: bool = False,
    robust_percentiles: tuple[float, float] = (0.02, 0.98),
) -> tuple[float, float]:
    """
    Get a grids max and min values.

    Parameters
    ----------
    values : tuple[xarray.DataArray | pandas.Series | numpy.ndarray]
        values to get min and max for
    shapefile : Union[str or geopandas.GeoDataFrame], optional
        path or loaded shapefile to use for a mask, by default None
    robust: bool, optional
        choose whether to return the 2nd and 98th percentile values, instead of the
        min/max
    region : tuple[float, float, float, float], optional
        give a subset region to get min and max values from, in format
        [xmin, xmax, ymin, ymax], by default None
    hemisphere : str, optional
        set whether to lat lon projection is for "north" hemisphere (EPSG:3413) or
        "south" hemisphere (EPSG:3031)
    absolute : bool, optional
        choose whether to return the absolute min and max values, by default False
    robust_percentiles : tuple[float, float], optional
        percentiles to use for robust min and max values, by default (0.02, 0.98)

    Returns
    -------
    tuple[float, float]
        returns the min and max values.
    """
    try:
        hemisphere = default_hemisphere(hemisphere)
    except KeyError:
        hemisphere = None

    # get min max of each grid
    limits = []
    for v in values:
        limits.append(
            get_min_max(
                v,
                robust=robust,
                region=region,
                shapefile=shapefile,
                hemisphere=hemisphere,
                absolute=absolute,
                robust_percentiles=robust_percentiles,
            )
        )

    # get min of all mins and max of all maxes
    ar = np.array(limits)

    return np.min(ar[:, 0]), np.max(ar[:, 1])


def grd_compare(
    da1: xr.DataArray | str,
    da2: xr.DataArray | str,
    plot: bool = True,
    plot_type: typing.Any | None = None,
    robust: bool = False,
    **kwargs: typing.Any,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Find the difference between 2 grids and plot the results, if necessary resample and
    cut grids to match

    Parameters
    ----------
    da1 : xarray.DataArray or str
        first grid, loaded grid of filename
    da2 : xarray.DataArray or str
        second grid, loaded grid of filename
    plot : bool, optional
        plot the results, by default True
    plot_type : typing.Any or None, optional
        this argument has been deprecated and will default to 'pygmt'
    robust : bool, optional
        use xarray robust color lims instead of min and max, by default is False.

    Keyword Args
    ------------
    shp_mask : str
        shapefile filename to use to mask the grids for setting the color range.
    robust : bool
        use xarray robust color lims instead of min and max, by default is False.
    region : tuple[float, float, float, float]
        choose a specific region to compare, in format [xmin, xmax, ymin, ymax].
    rmse_in_title: bool
        add the RMSE to the title, by default is True.
    cpt_lims : tuple[float, float]
        set the colorbar limits for the two grids.
    diff_lims : tuple[float, float]
        set the colorbar limits for the difference grid.

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray, xarray.DataArray]
        three xarray.DataArrays: (diff, resampled grid1, resampled grid2)
    """
    if plot_type is not None:
        warnings.warn(
            "plot_type has been deprecated and will default to 'pygmt'",
            DeprecationWarning,
            stacklevel=2,
        )
    shp_mask = kwargs.get("shp_mask")
    region = kwargs.get("region")
    verbose = kwargs.get("verbose", "e")
    if isinstance(da1, str):
        da1 = xr.load_dataarray(da1)

    if isinstance(da2, str):
        da2 = xr.load_dataarray(da2)

    # first cut the grids to save time on the possible resampling below
    if region is not None:
        da1 = pygmt.grdcut(
            da1,
            region=region,
            verbose=verbose,
        )
        da2 = pygmt.grdcut(
            da2,
            region=region,
            verbose=verbose,
        )

    # extract grid info of both grids
    da1_info = get_grid_info(da1)
    da2_info = get_grid_info(da2)

    # extract spacing of both grids
    assert da1_info[0] is not None
    assert da2_info[0] is not None
    da1_spacing: float = da1_info[0]
    da2_spacing: float = da2_info[0]

    # extract regions of both grids
    da1_reg = da1_info[1]
    da1_reg = typing.cast(tuple[float, float, float, float], da1_reg)
    da2_reg = da2_info[1]
    da2_reg = typing.cast(tuple[float, float, float, float], da2_reg)
    # if spacing and region match, no resampling
    if (da1_spacing == da2_spacing) and (da1_reg == da2_reg):
        grid1 = da1
        grid2 = da2
    else:
        # get minimum grid spacing of both grids
        if da1_spacing != da2_spacing:
            spacing = min(da1_spacing, da2_spacing)
            logger.info(
                "grid spacings don't match, using smaller spacing (%s m).",
                spacing,
            )
        else:
            spacing = da1_spacing
        # get inside region of both grids
        if da1_reg != da2_reg:
            xmin = max(da1_reg[0], da2_reg[0])
            xmax = min(da1_reg[1], da2_reg[1])
            ymin = max(da1_reg[2], da2_reg[2])
            ymax = min(da1_reg[3], da2_reg[3])
            region = (xmin, xmax, ymin, ymax)
            logger.info("grid regions dont match, using inner region %s", region)
        else:
            region = da1_reg
        # use registration from first grid, or from kwarg
        if kwargs.get("registration") is None:
            registration = get_grid_info(da1)[4]
        else:
            registration = kwargs.get("registration")
        # resample grids
        grid1 = fetch.resample_grid(
            da1,
            spacing=spacing,
            region=region,
            registration=registration,
            verbose=verbose,
        )

        grid2 = fetch.resample_grid(
            da2,
            spacing=spacing,
            region=region,
            registration=registration,
            verbose=verbose,
        )

    grid1 = typing.cast(xr.DataArray, grid1)
    grid2 = typing.cast(xr.DataArray, grid2)

    dif = grid1 - grid2

    cpt_lims = kwargs.get("cpt_lims")
    if cpt_lims is not None:
        vmin, vmax = cpt_lims
    else:
        # get individual min/max values (and masked values if shapefile is provided)
        grid1_cpt_lims = get_min_max(
            grid1,
            shp_mask,
            robust=robust,
            hemisphere=kwargs.get("hemisphere"),
            robust_percentiles=kwargs.get("robust_percentiles", (0.02, 0.98)),
        )
        grid2_cpt_lims = get_min_max(
            grid2,
            shp_mask,
            robust=robust,
            hemisphere=kwargs.get("hemisphere"),
            robust_percentiles=kwargs.get("robust_percentiles", (0.02, 0.98)),
        )
        # get min and max of both grids together
        vmin = min((grid1_cpt_lims[0], grid2_cpt_lims[0]))
        vmax = max(grid1_cpt_lims[1], grid2_cpt_lims[1])

    if kwargs.get("diff_lims") is not None:
        diff_lims = kwargs.get("diff_lims")
    else:
        diff_lims = get_min_max(
            dif,
            shp_mask,
            robust=robust,
            robust_percentiles=kwargs.get("robust_percentiles", (0.02, 0.98)),
            hemisphere=kwargs.get("hemisphere"),
            absolute=kwargs.get("diff_maxabs", True),
        )

    if plot is True:
        title = kwargs.get("title", "Comparing Grids")
        if kwargs.get("rmse_in_title", True) is True:
            title += f", RMSE: {round(rmse(dif),kwargs.get('RMSE_decimals', 2))}"

        fig_height = kwargs.get("fig_height", 12)
        coast = kwargs.get("coast", False)
        origin_shift = kwargs.get("origin_shift", "x")
        cmap = kwargs.get("cmap", "viridis")
        subplot_labels = kwargs.get("subplot_labels", False)

        new_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in [
                "cmap",
                "region",
                "coast",
                "title",
                "cpt_lims",
                "fig_height",
                "inset",
                "inset_pos",
                "inset_position",
                "shp_mask",
            ]
        }
        diff_kwargs = {
            key: value
            for key, value in new_kwargs.items()
            if key not in ["reverse_cpt", "cbar_label", "shp_mask"]
        }
        fig = maps.plot_grd(
            grid1,
            cmap=cmap,
            region=region,
            coast=coast,
            title=kwargs.get("grid1_name", "grid 1"),
            cpt_lims=(vmin, vmax),
            fig_height=fig_height,
            **new_kwargs,
        )

        if subplot_labels is True:
            fig.text(
                position="TL",
                justify="BL",
                text="a)",
                font=kwargs.get("label_font", "18p,Helvetica,black"),
                offset=kwargs.get("label_offset", "j0/.3"),
                no_clip=True,
            )
        fig = maps.plot_grd(
            dif,
            cmap=kwargs.get("diff_cmap", "balance+h0"),
            region=region,
            coast=coast,
            origin_shift=origin_shift,
            cbar_label="difference",
            cpt_lims=diff_lims,
            fig=fig,
            title=title,
            inset=kwargs.get("inset", False),
            inset_position=kwargs.get("inset_position", "jTL+jTL+o0/0"),
            inset_pos=kwargs.get("inset_pos"),
            fig_height=fig_height,
            **diff_kwargs,
        )
        if subplot_labels is True:
            fig.text(
                position="TL",
                justify="BL",
                text="b)",
                font=kwargs.get("label_font", "20p,Helvetica,black"),
                offset=kwargs.get("label_offset", "j0/.3"),
                no_clip=True,
            )
        fig = maps.plot_grd(
            grid2,
            cmap=cmap,
            region=region,
            coast=coast,
            origin_shift=origin_shift,
            fig=fig,
            title=kwargs.get("grid2_name", "grid 2"),
            cpt_lims=(vmin, vmax),
            fig_height=fig_height,
            **new_kwargs,
        )
        if subplot_labels is True:
            fig.text(
                position="TL",
                justify="BL",
                text="c)",
                font=kwargs.get("label_font", "20p,Helvetica,black"),
                offset=kwargs.get("label_offset", "j0/.3"),
                no_clip=True,
            )

        fig.show()

    return (dif, grid1, grid2)


def make_grid(
    region: tuple[float, float, float, float],
    spacing: float,
    value: float,
    name: str,
) -> xr.DataArray:
    """
    Create a grid with 1 variable by defining a region, spacing, name and constant value

    Parameters
    ----------
    region : tuple[float, float, float, float]
        bounding region in format [xmin, xmax, ymin, ymax]
    spacing : float
        spacing for grid
    value : float
        constant value to use for variable
    name : str
        name for variable

    Returns
    -------
    xarray.DataArray
        Returns a xarray.DataArray with 1 variable of constant value.
    """
    coords = vd.grid_coordinates(region=region, spacing=spacing, pixel_register=True)
    data = np.ones_like(coords[0]) * value
    return typing.cast(
        xr.DataArray,
        vd.make_xarray_grid(coords, data, dims=["y", "x"], data_names=name),
    )


def square_subplots(n: int) -> tuple[int, int]:
    """
    From https://github.com/matplotlib/grid-strategy/blob/master/src/grid_strategy/strategies.py
    Calculate the number of rows and columns based on the total number of items (n) to
    make an arrangement as close to square as looks good.

    Parameters
    ----------
    n : int
        The number of total plots in the subplot

    Returns
    -------
    tuple[int, int]
        Returns a tuple in the format (number of rows, number of columns), so for
        example a 3 x 2 grid would be represented as ``(3, 3)``, because there are 2
        rows of length 3.
    """
    special_cases = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (2, 4),
        8: (2, 4),
        9: (3, 3),
        10: (3, 4),
    }
    if n in special_cases:
        return special_cases[n]

    # May not work for very large n
    n_sqrtf = np.sqrt(n)
    n_sqrt = int(np.ceil(n_sqrtf))

    if n_sqrtf == n_sqrt:
        # Perfect square, we're done
        x, y = n_sqrt, n_sqrt
    elif n <= n_sqrt * (n_sqrt - 1):
        # An n_sqrt x n_sqrt - 1 grid is close enough to look pretty
        # square, so if n is less than that value, will use that rather
        # than jumping all the way to a square grid.
        x, y = n_sqrt, n_sqrt - 1
    elif not (n_sqrt % 2) and n % 2:
        # If the square root is even and the number of axes is odd, in
        # order to keep the arrangement horizontally symmetrical, using a
        # grid of size (n_sqrt + 1 x n_sqrt - 1) looks best and guarantees
        # symmetry.
        x, y = (n_sqrt + 1, n_sqrt - 1)
    else:
        # It's not a perfect square, but a square grid is best
        x, y = n_sqrt, n_sqrt

    if n == x * y:
        # There are no deficient rows, so we can just return from here
        return x, y  # tuple(x for i in range(y))

    # If exactly one of these is odd, make it the rows
    if (x % 2) != (y % 2) and (x % 2):
        x, y = y, x

    return x, y


def random_color() -> str:
    """
    generate a random color in format R/G/B

    Returns
    -------
    str
        returns a random color string, for example '95/226/100'
    """
    rng = np.random.default_rng()

    numbers = rng.integers(low=0, high=256, size=3)
    return f"{numbers[0]}/{numbers[1]}/{numbers[2]}"


def subset_grid(
    grid: xr.DataArray,
    region: tuple[float, float, float, float],
) -> xr.DataArray:
    """
    Return a subset of a grid based on a region

    Parameters
    ----------
    grid : xarray.DataArray
        grid to be clipped
    region : tuple[float, float, float, float]
        region to clip to, in format [xmin, xmax, ymin, ymax]

    Returns
    -------
    xarray.DataArray
        clipped grid
    """
    ew = [region[0], region[1]]
    ns = [region[2], region[3]]

    return grid.sel(
        {
            list(grid.sizes.keys())[1]: slice(min(ew), max(ew)),
            list(grid.sizes.keys())[0]: slice(min(ns), max(ns)),  # noqa: RUF015
        }
    )


def get_min_max(
    values: xr.DataArray | pd.Series | NDArray,
    shapefile: str | gpd.geodataframe.GeoDataFrame | None = None,
    robust: bool = False,
    region: tuple[float, float, float, float] | None = None,
    hemisphere: str | None = None,
    absolute: bool = False,
    robust_percentiles: tuple[float, float] = (0.02, 0.98),
) -> tuple[float, float]:
    """
    Get a grids max and min values.

    Parameters
    ----------
    values : xarray.DataArray or pandas.Series or numpy.ndarray
        values to find min or max for
    shapefile : Union[str or geopandas.GeoDataFrame], optional
        path or loaded shapefile to use for a mask, by default None
    robust: bool, optional
        choose whether to return the 2nd and 98th percentile values, instead of the
        min/max
    region : tuple[float, float, float, float], optional
        give a subset region to get min and max values from, in format
        [xmin, xmax, ymin, ymax], by default None
    hemisphere : str, optional
        set whether to lat lon projection is for "north" hemisphere (EPSG:3413) or
        "south" hemisphere (EPSG:3031)
    absolute : bool, optional
        return the absolute min and max values, by default False
    robust_percentiles : tuple[float, float], optional
        decimal percentiles to use for robust min and max, by default (0.02, 0.98)

    Returns
    -------
    tuple[float, float]
        returns the min and max values.
    """
    try:
        hemisphere = default_hemisphere(hemisphere)
    except KeyError:
        hemisphere = None

    if region is not None:
        values = subset_grid(values, region)

    if shapefile is None:
        if robust:
            v_min, v_max = np.nanquantile(values, robust_percentiles)
        else:
            v_min, v_max = np.nanmin(values), np.nanmax(values)
    elif shapefile is not None:
        if isinstance(values, xr.DataArray):
            masked = mask_from_shp(
                shapefile,
                hemisphere=hemisphere,
                xr_grid=values,
                masked=True,
                invert=False,
            )
        else:
            msg = "values must be an xarray.DataArray to use shapefile masking"
            raise ValueError(msg)

        if robust is True:
            v_min, v_max = np.nanquantile(masked, robust_percentiles)
        elif robust is False:
            v_min, v_max = np.nanmin(masked), np.nanmax(masked)

    if absolute is True:
        v_min, v_max = -vd.maxabs([v_min, v_max]), vd.maxabs([v_min, v_max])  # pylint: disable=used-before-assignment

    assert v_min <= v_max, "min value should be less than or equal to max value"  # pylint: disable=possibly-used-before-assignment
    return (v_min, v_max)


def shapes_to_df(
    shapes: list[float],
    hemisphere: str | None = None,
) -> pd.DataFrame:
    """
    convert the output of `regions.draw_region` and `profiles.draw_lines` to a dataframe
    of easting and northing points

    Parameters
    ----------
    hemisphere : str, optional
        choose between the "north" or "south" hemispheres
    shapes : list
        list of vertices

    Returns
    -------
    pandas.DataFrame
        Dataframe with easting, northing, and shape_num.
    """
    hemisphere = default_hemisphere(hemisphere)

    df = pd.DataFrame()
    for i, j in enumerate(shapes):
        lon = [coord[0] for coord in j]  # type: ignore[attr-defined]
        lat = [coord[1] for coord in j]  # type: ignore[attr-defined]
        shape = pd.DataFrame({"lon": lon, "lat": lat, "shape_num": i})
        df = pd.concat((df, shape))

    if hemisphere == "north":
        df = latlon_to_epsg3413(df)
    elif hemisphere == "south":
        df = latlon_to_epsg3031(df)
    else:
        msg = "hemisphere must be 'north' or 'south'"
        raise ValueError(msg)

    return df


def polygon_to_region(
    polygon: list[float],
    hemisphere: str | None = None,
) -> tuple[float, float, float, float]:
    """
    convert the output of `regions.draw_region` to bounding region in EPSG:3031 for the
    south hemisphere and EPSG:3413 for the north hemisphere.

    Parameters
    ----------
    polyon : list
        list of polygon vertices
    hemisphere : str, optional
        choose between the "north" or "south" hemispheres

    Returns
    -------
    tuple[float, float, float, float]
        region in format in format [xmin, xmax, ymin, ymax]
    """
    hemisphere = default_hemisphere(hemisphere)

    df = shapes_to_df(shapes=polygon, hemisphere=hemisphere)

    if df.shape_num.max() > 0:
        logger.info(
            "supplied dataframe has multiple polygons, only using the first one."
        )
        df = df[df.shape_num == 0]

    reg: tuple[float, float, float, float] = vd.get_region((df.easting, df.northing))

    return reg


def mask_from_polygon(
    polygon: list[float],
    hemisphere: str | None = None,
    invert: bool = False,
    drop_nans: bool = False,
    grid: str | xr.DataArray | None = None,
    region: tuple[float, float, float, float] | None = None,
    spacing: int | None = None,
    **kwargs: typing.Any,
) -> xr.DataArray:
    """
    convert the output of `regions.draw_region` to a mask or use it to mask a grid

    Parameters
    ----------
    polygon : list
       list of polygon vertices
    hemisphere : str, optional
        choose between the "north" or "south" hemispheres
    invert : bool, optional
        reverse the sense of masking, by default False
    drop_nans : bool, optional
        drop nans after masking, by default False
    grid : Union[str, xarray.DataArray], optional
        grid to mask, by default None
    region : tuple[float, float, float, float], optional
        region to create a grid if none is supplied, in format [xmin, xmax, ymin, ymax],
        by default None
    spacing : int, optional
        spacing to create a grid if none is supplied, by default None

    Returns
    -------
    xarray.DataArray
        masked grid or mask grid with 1's inside the mask.
    """
    hemisphere = default_hemisphere(hemisphere)

    # convert drawn polygon into dataframe
    df = shapes_to_df(polygon, hemisphere=hemisphere)
    data_coords = (df.easting, df.northing)

    # remove additional polygons
    if df.shape_num.max() > 0:
        logger.info(
            "supplied dataframe has multiple polygons, only using the first one."
        )
        df = df[df.shape_num == 0]

    # if grid given as filename, load it
    if isinstance(grid, str):
        grid = xr.load_dataarray(grid)
        grid = typing.cast(xr.DataArray, grid)
        ds = grid.to_dataset()
    elif isinstance(grid, xr.DataArray):
        ds = grid.to_dataset()
    # if no grid given, make a dummy one with supplied region and spacing
    elif grid is None:
        coords = vd.grid_coordinates(
            region=region,
            spacing=spacing,
            pixel_register=kwargs.get("pixel_register", False),
        )
        ds = vd.make_xarray_grid(
            coords, np.ones_like(coords[0]), dims=("y", "x"), data_names="z"
        )
    else:
        msg = "grid must be a xr.DataArray, a filename, or None"
        raise ValueError(msg)

    masked = vd.convexhull_mask(
        data_coords,
        grid=ds,
    ).z

    # reverse the mask
    if invert is True:
        inverse = masked.isnull()
        inverse = inverse.where(inverse != 0)
        masked = inverse * ds.z

    # drop nans
    if drop_nans is True:
        masked = masked.where(masked.notnull() == 1, drop=True)

    return typing.cast(xr.DataArray, masked)


def change_reg(grid: xr.DataArray) -> xr.DataArray:
    """
    Use GMT grdedit to change the registration type in the metadata.

    Parameters
    ----------
    grid : xarray.DataArray
        input grid to change the reg for.

    Returns
    -------
    xarray.DataArray
        returns a xarray.DataArray with switched reg type.
    """
    with pygmt.clib.Session() as ses:  # noqa: SIM117
        # store the input grid in a virtual file so GMT can read it from a dataarray
        with ses.virtualfile_from_grid(grid) as f_in:
            # send the output to a file so that we can read it
            with pygmt.helpers.GMTTempFile(suffix=".nc") as tmpfile:
                args = f"{f_in} -T -G{tmpfile.name}"
                ses.call_module("grdedit", args)
                f_out: xr.DataArray = pygmt.load_dataarray(tmpfile.name)
    return f_out


def grd_blend(
    grid1: xr.DataArray,
    grid2: xr.DataArray,
) -> xr.DataArray:
    """
    Use GMT grdblend to blend 2 grids into 1.

    Parameters
    ----------
    grid1 : xarray.DataArray
        input grid to change the reg for.

    grid2 : xarray.DataArray
        input grid to change the reg for.

    Returns
    -------
    xarray.DataArray
        returns a blended grid.
    """
    with pygmt.clib.Session() as session:  # noqa: SIM117
        with pygmt.helpers.GMTTempFile(suffix=".nc") as tmpfile:
            # store the input grids in a virtual files so GMT can read it from
            # dataarrays
            file_context1 = session.virtualfile_from_grid(grid1)
            file_context2 = session.virtualfile_from_grid(grid2)
            with file_context1 as infile1, file_context2 as infile2:
                # if (outgrid := kwargs.get("G")) is None:
                #     kwargs["G"] = outgrid = tmpfile.name # output to tmpfile
                args = f"{infile1} {infile2} -Cf -G{tmpfile.name}"
                session.call_module(module="grdblend", args=args)
    return typing.cast(
        xr.DataArray, pygmt.load_dataarray(infile1)
    )  # if outgrid == tmpfile.name else None


def get_fig_width() -> float:
    """
    Get the width of the current PyGMT figure instance.

    Returns
    -------
    float
        width of the figure
    """
    with pygmt.clib.Session() as session:  # noqa: SIM117
        with pygmt.helpers.GMTTempFile() as tmpfile:
            session.call_module("mapproject", f"-Ww ->{tmpfile.name}")
            map_width = tmpfile.read().strip()
    return float(map_width)


def get_fig_height() -> float:
    """
    Get the height of the current PyGMT figure instance.

    Returns
    -------
    float
        height of the figure
    """
    with pygmt.clib.Session() as session:  # noqa: SIM117
        with pygmt.helpers.GMTTempFile() as tmpfile:
            session.call_module("mapproject", f"-Wh ->{tmpfile.name}")
            map_height = tmpfile.read().strip()
    return float(map_height)


def gmt_str_to_list(region: tuple[float, float, float, float]) -> str:
    """
    convert a tuple of floats representing the boundaries of a region into a GMT-style
    region string

    Parameters
    ----------
    region : tuple[float, float, float, float]
        bounding region in format [xmin, xmax, ymin, ymax]

    Returns
    -------
    str
        a GMT style region string
    """
    return "".join([str(x) + "/" for x in region])[:-1]
