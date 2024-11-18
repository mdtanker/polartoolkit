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
# pylint: disable=too-many-lines
from __future__ import annotations

import logging
import os
import random
import typing

import deprecation
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
import pyogrio
import verde as vd
import xarray as xr
from numpy.typing import NDArray
from pyproj import Transformer

import polartoolkit
from polartoolkit import fetch, maps, regions

if typing.TYPE_CHECKING:
    import geopandas as gpd

try:
    import seaborn as sns
except ImportError:
    sns = None


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
        logging.exception(e)
        logging.warning("grid spacing can't be extracted")
        spacing = None

    try:
        region: typing.Any = tuple(
            float(pygmt.grdinfo(grid, per_column="n", o=i)[:-1]) for i in range(4)
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        # pygmt.exceptions.GMTInvalidInput:
        logging.exception(e)
        logging.warning("grid region can't be extracted")
        region = None

    try:
        zmin: float | None = float(pygmt.grdinfo(grid, per_column="n", o=4)[:-1])
    except Exception as e:  # pylint: disable=broad-exception-caught
        # pygmt.exceptions.GMTInvalidInput:
        logging.exception(e)
        logging.warning("grid zmin can't be extracted")
        zmin = None

    try:
        zmax = float(pygmt.grdinfo(grid, per_column="n", o=5)[:-1])
    except Exception as e:  # pylint: disable=broad-exception-caught
        # pygmt.exceptions.GMTInvalidInput:
        logging.exception(e)
        logging.warning("grid zmax can't be extracted")
        zmax = None

    try:
        reg = grid.gmt.registration  # type: ignore[union-attr]
        registration: str | None = "g" if reg == 0 else "p"
    except AttributeError:
        logging.warning(
            "grid registration not extracted, re-trying with file loaded as xarray grid"
        )
        # grid = xr.load_dataarray(grid)
        with xr.open_dataarray(grid) as da:
            try:
                reg = da.gmt.registration
                registration = "g" if reg == 0 else "p"
            except AttributeError:
                logging.warning("grid registration can't be extracted, setting to 'g'.")
                registration = "g"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception(e)
        logging.warning("grid registration can't be extracted")
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
    coord_names: tuple[str, str] = ("x", "y"),
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
        names of input or output coordinate columns, by default ("x", "y")
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


def latlon_to_epsg3031(
    df: pd.DataFrame | NDArray[typing.Any, typing.Any],
    reg: bool = False,
    input_coord_names: tuple[str, str] = ("lon", "lat"),
    output_coord_names: tuple[str, str] = ("x", "y"),
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
    input_coord_names : list, optional
        set names for input coordinate columns, by default ["lon", "lat"]
    output_coord_names : list, optional
        set names for output coordinate columns, by default ["x", "y"]

    Returns
    -------
    pandas.DataFrame or numpy.ndarray[typing.Any, typing.Any]
        Updated dataframe with new easting and northing columns or numpy.ndarray in
        format [xmin, xmax, ymin, ymax]
    """
    transformer = Transformer.from_crs("epsg:4326", "epsg:3031")

    if isinstance(df, pd.DataFrame):
        df_new = df.copy()
        (  # pylint: disable=unpacking-non-sequence
            df_new[output_coord_names[0]],
            df_new[output_coord_names[1]],
        ) = transformer.transform(
            df_new[input_coord_names[1]].tolist(), df_new[input_coord_names[0]].tolist()
        )
    else:
        ll = df.copy()
        df_new = list(transformer.transform(ll[0], ll[1]))

    if reg is True:
        df_new = [
            df_new[output_coord_names[0]].min(),
            df_new[output_coord_names[0]].max(),
            df_new[output_coord_names[1]].min(),
            df_new[output_coord_names[1]].max(),
        ]

    return df_new


def epsg3031_to_latlon(
    df: pd.DataFrame | list[typing.Any],
    reg: bool = False,
    input_coord_names: tuple[str, str] = ("x", "y"),
    output_coord_names: tuple[str, str] = ("lon", "lat"),
) -> pd.DataFrame | list[typing.Any]:
    """
    Convert coordinates from EPSG:3031 Antarctic Polar Stereographic in meters to
    EPSG:4326 WGS84 in decimal degrees.

    Parameters
    ----------
    df : pandas.DataFrame or list[typing.Any]
        input dataframe with easting and northing columns, or list [x,y]
    reg : bool, optional
        if true, returns a GMT formatted region string, by default False
    input_coord_names : list, optional
        set names for input coordinate columns, by default ["x", "y"]
    output_coord_names : list, optional
        set names for output coordinate columns, by default ["lon", "lat"]

    Returns
    -------
    pandas.DataFrame or list[typing.Any]
        Updated dataframe with new latitude and longitude columns, numpy.ndarray in
        format [xmin, xmax, ymin, ymax], or list in format [lat, lon]
    """

    transformer = Transformer.from_crs("epsg:3031", "epsg:4326")

    df_out = df.copy()

    if isinstance(df, pd.DataFrame):
        (  # pylint: disable=unpacking-non-sequence
            df_out[output_coord_names[1]],  # type: ignore[call-overload]
            df_out[output_coord_names[0]],  # type: ignore[call-overload]
        ) = transformer.transform(
            df_out[input_coord_names[0]].tolist(),  # type: ignore[call-overload]
            df_out[input_coord_names[1]].tolist(),  # type: ignore[call-overload]
        )
        if reg is True:
            df_out = [
                df_out[output_coord_names[0]].min(),  # type: ignore[call-overload]
                df_out[output_coord_names[0]].max(),  # type: ignore[call-overload]
                df_out[output_coord_names[1]].min(),  # type: ignore[call-overload]
                df_out[output_coord_names[1]].max(),  # type: ignore[call-overload]
            ]
    else:
        df_out = list(transformer.transform(df_out[0], df_out[1]))
    return df_out


def latlon_to_epsg3413(
    df: pd.DataFrame | NDArray[typing.Any, typing.Any],
    reg: bool = False,
    input_coord_names: tuple[str, str] = ("lon", "lat"),
    output_coord_names: tuple[str, str] = ("x", "y"),
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
    input_coord_names : list, optional
        set names for input coordinate columns, by default ["lon", "lat"]
    output_coord_names : list, optional
        set names for output coordinate columns, by default ["x", "y"]

    Returns
    -------
    pandas.DataFrame or numpy.ndarray[typing.Any, typing.Any]
        Updated dataframe with new easting and northing columns or numpy.ndarray in
        format [xmin, xmax, ymin, ymax]
    """
    transformer = Transformer.from_crs("epsg:4326", "epsg:3413")

    if isinstance(df, pd.DataFrame):
        df_new = df.copy()
        (  # pylint: disable=unpacking-non-sequence
            df_new[output_coord_names[0]],
            df_new[output_coord_names[1]],
        ) = transformer.transform(
            df_new[input_coord_names[1]].tolist(), df_new[input_coord_names[0]].tolist()
        )
    else:
        ll = df.copy()
        df_new = list(transformer.transform(ll[0], ll[1]))

    if reg is True:
        df_new = [
            df_new[output_coord_names[0]].min(),
            df_new[output_coord_names[0]].max(),
            df_new[output_coord_names[1]].min(),
            df_new[output_coord_names[1]].max(),
        ]

    return df_new


def epsg3413_to_latlon(
    df: pd.DataFrame | list[typing.Any],
    reg: bool = False,
    input_coord_names: tuple[str, str] = ("x", "y"),
    output_coord_names: tuple[str, str] = ("lon", "lat"),
) -> pd.DataFrame | list[typing.Any]:
    """
    Convert coordinates from EPSG:3413 North Polar Stereographic in meters to
    EPSG:4326 WGS84 in decimal degrees.

    Parameters
    ----------
    df : pandas.DataFrame or list[typing.Any]
        input dataframe with easting and northing columns, or list [x,y]
    reg : bool, optional
        if true, returns a GMT formatted region string, by default False
    input_coord_names : list, optional
        set names for input coordinate columns, by default ["x", "y"]
    output_coord_names : list, optional
        set names for output coordinate columns, by default ["lon", "lat"]

    Returns
    -------
    pandas.DataFrame or list[typing.Any]
        Updated dataframe with new latitude and longitude columns, numpy.ndarray in
        format [xmin, xmax, ymin, ymax], or list in format [lat, lon]
    """

    transformer = Transformer.from_crs("epsg:3413", "epsg:4326")

    df_out = df.copy()

    if isinstance(df, pd.DataFrame):
        (  # pylint: disable=unpacking-non-sequence
            df_out[output_coord_names[1]],  # type: ignore[call-overload]
            df_out[output_coord_names[0]],  # type: ignore[call-overload]
        ) = transformer.transform(
            df_out[input_coord_names[0]].tolist(),  # type: ignore[call-overload]
            df_out[input_coord_names[1]].tolist(),  # type: ignore[call-overload]
        )
        if reg is True:
            df_out = [
                df_out[output_coord_names[0]].min(),  # type: ignore[call-overload]
                df_out[output_coord_names[0]].max(),  # type: ignore[call-overload]
                df_out[output_coord_names[1]].min(),  # type: ignore[call-overload]
                df_out[output_coord_names[1]].max(),  # type: ignore[call-overload]
            ]
    else:
        df_out = list(transformer.transform(df_out[0], df_out[1]))
    return df_out


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
        dataframe with columns 'x','y' to use for defining if within region
    region : tuple[float, float, float, float]
        bounding region in format [xmin, xmax, ymin, ymax] for bounds of new subset
        dataframe
    names : tuple[str, str], optional
        column names to use for x and y coordinates, by default ("x", "y")
    reverse : bool, optional
        if True, will return points outside the region, by default False

    Returns
    -------
    pandas.DataFrame
       returns a subset dataframe
    """
    # make a copy of the dataframe
    df1 = df.copy()

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
        strings of coordinate column names, by default ("x", "y")
    input_data_names : typing.Any | None, optional
        strings of data column names, by default None

    Returns
    -------
    pandas.DataFrame
        a block-reduced dataframe
    """
    # define verde reducer function
    reducer = vd.BlockReduce(reduction, **kwargs)

    # if no data names provided, use all columns
    if input_data_names is None:
        input_data_names = tuple(df.columns.drop(input_coord_names))

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
    input_coord_names: tuple[str, str] = ("x", "y"),
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

    Returns
    -------
    xarray.DataArray
        Returns either a masked grid, or the mask grid itself.
    """
    hemisphere = default_hemisphere(hemisphere)

    shp = pyogrio.read_dataframe(shapefile) if isinstance(shapefile, str) else shapefile

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
            dims=reversed(input_coord_names),
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

    masked_grd = xds.rio.clip(
        shp.geometry,
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
        logging.info(e)

    return typing.cast(xr.DataArray, output)


@deprecation.deprecated(
    deprecated_in="0.4.0",
    removed_in="0.8.0",
    current_version=polartoolkit.__version__,
    details="alter_region has been moved to the regions module, use that instead",
)
def alter_region(
    starting_region: tuple[float, float, float, float] | None = None,
    zoom: float = 0,
    n_shift: float = 0,
    w_shift: float = 0,
    buffer: float = 0,  # noqa: ARG001 # pylint: disable=unused-argument
    print_reg: bool = False,  # noqa: ARG001 # pylint: disable=unused-argument
) -> tuple[float, float, float, float]:
    """deprecated function, use regions.alter_region instead"""
    return regions.alter_region(starting_region, zoom, n_shift, w_shift)  # type: ignore[arg-type]


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
    plot_type: str = "pygmt",
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
    plot_type : str
        choose to plot results with pygmt or xarray, by default "pygmt"

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
        if plot_type == "xarray":
            fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 20))
            da.plot(
                ax=ax[0],
                robust=True,
                cmap="viridis",
                cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1.8),
                    "label": "test",
                },
            )
            ax[0].set_title("Input grid")
            fit.plot(
                ax=ax[1],
                robust=True,
                cmap="viridis",
                cbar_kwargs={"orientation": "horizontal", "anchor": (1, 1.8)},
            )
            ax[1].set_title(f"Trend order {deg}")
            detrend.plot(
                ax=ax[2],
                robust=True,
                cmap="viridis",
                cbar_kwargs={"orientation": "horizontal", "anchor": (1, 1.8)},
            )
            ax[2].set_title("Detrended")
            for a in ax:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_xlabel("")
                a.set_ylabel("")
                a.set_aspect("equal")

        elif plot_type == "pygmt":
            cmap: typing.Any = kwargs.get("cmap", "plasma")
            coast: typing.Any = kwargs.get("coast", True)
            inset: typing.Any = kwargs.get("inset", True)
            inset_pos: typing.Any = kwargs.get("inset_pos", "BL")
            origin_shift: typing.Any = kwargs.get("origin_shift", "y")
            fit_label: typing.Any = kwargs.get(
                "fit_label", f"fitted trend (order {deg})"
            )
            input_label: typing.Any = kwargs.get("input_label", "input grid")
            title: typing.Any = kwargs.get("title", "Detrending a grid")
            detrended_label: typing.Any = kwargs.get("detrended_label", "detrended")

            fig = maps.plot_grd(
                detrend,
                cmap=cmap,
                grd2cpt=True,
                coast=coast,
                cbar_label=detrended_label,
                **kwargs,
            )

            fig = maps.plot_grd(
                fit,
                fig=fig,
                cmap=cmap,
                grd2cpt=True,
                coast=coast,
                cbar_label=fit_label,
                inset=inset,
                inset_pos=inset_pos,
                origin_shift=origin_shift,
                **kwargs,
            )

            fig = maps.plot_grd(
                da,
                fig=fig,
                cmap=cmap,
                grd2cpt=True,
                coast=coast,
                cbar_label=input_label,
                title=title,
                origin_shift=origin_shift,
                **kwargs,
            )

            fig.show()

    return fit, detrend


def get_combined_min_max(
    grids: tuple[xr.DataArray],
    shapefile: str | gpd.geodataframe.GeoDataFrame | None = None,
    robust: bool = False,
    region: tuple[float, float, float, float] | None = None,
    hemisphere: str | None = None,
) -> tuple[float, float]:
    """
    Get a grids max and min values.

    Parameters
    ----------
    grids : tuple[xarray.DataArray]
        grids to get values for
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
    for g in grids:
        limits.append(
            get_min_max(
                g,
                robust=robust,
                region=region,
                shapefile=shapefile,
                hemisphere=hemisphere,
            )
        )

    # get min of all mins and max of all maxes
    ar = np.array(limits)

    return np.min(ar[:, 0]), np.max(ar[:, 1])


def grd_compare(
    da1: xr.DataArray | str,
    da2: xr.DataArray | str,
    plot: bool = False,
    plot_type: str = "pygmt",
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
        plot the results, by default False
    plot_type : str, optional
        choose the style of plot, by default is pygmt, can choose xarray for faster,
        simpler plots.
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

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray, xarray.DataArray]
        three xarray.DataArrays: (diff, resampled grid1, resampled grid2)
    """
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
            logging.info(
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
            logging.info("grid regions dont match, using inner region %s", region)
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

    # get individual grid min/max values (and masked values if shapefile is provided)
    grid1_cpt_lims = get_min_max(
        grid1,
        shp_mask,
        robust=robust,
        hemisphere=kwargs.get("hemisphere"),
    )
    grid2_cpt_lims = get_min_max(
        grid2,
        shp_mask,
        robust=robust,
        hemisphere=kwargs.get("hemisphere"),
    )

    diff_maxabs = kwargs.get("diff_maxabs", True)
    if diff_maxabs is False:
        diff_lims: typing.Any = get_min_max(
            dif,
            shp_mask,
            robust=robust,
            hemisphere=kwargs.get("hemisphere"),
        )
    else:
        diff_maxabs = vd.maxabs(
            get_min_max(
                dif,
                shp_mask,
                robust=robust,
                hemisphere=kwargs.get("hemisphere"),
            )
        )
        diff_lims = kwargs.get("diff_lims", (-diff_maxabs, diff_maxabs))

    # get min and max of both grids together
    vmin: typing.Any = min((grid1_cpt_lims[0], grid2_cpt_lims[0]))
    vmax: typing.Any = max(grid1_cpt_lims[1], grid2_cpt_lims[1])

    if plot is True:
        title = kwargs.get("title", "Comparing Grids")
        if kwargs.get("rmse_in_title", True) is True:
            title += f", RMSE: {round(rmse(dif),kwargs.get('RMSE_decimals', 2))}"

        if plot_type == "pygmt":
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
                inset=kwargs.get("inset", True),
                inset_pos=kwargs.get("inset_pos", "TL"),
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

        elif plot_type == "xarray":
            if robust:
                vmin, vmax = None, None
                diff_lims = (None, None)
            cmap = kwargs.get("cmap", "viridis")

            sub_width = 5
            nrows, ncols = 1, 3
            # setup subplot figure
            fig, ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(sub_width * ncols, sub_width * nrows),
            )

            grid1.plot(
                ax=ax[0],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                robust=True,
                cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1),
                    "fraction": 0.05,
                    "pad": 0.04,
                },
            )
            ax[0].set_title(kwargs.get("grid1_name", "grid 1"))

            dif.plot(
                ax=ax[1],
                vmin=diff_lims[0],
                vmax=diff_lims[1],
                robust=True,
                cmap=kwargs.get("diff_cmap", "RdBu_r"),
                cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1),
                    "fraction": 0.05,
                    "pad": 0.04,
                },
            )

            ax[1].set_title(title)

            grid2.plot(
                ax=ax[2],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                robust=True,
                cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1),
                    "fraction": 0.05,
                    "pad": 0.04,
                },
            )
            ax[2].set_title(kwargs.get("grid2_name", "grid 2"))

            for a in ax:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_xlabel("")
                a.set_ylabel("")
                a.set_aspect("equal")
                if kwargs.get("points") is not None:
                    a.plot(kwargs.get("points").x, kwargs.get("points").y, "k+")  # type: ignore[union-attr]
                show_region = kwargs.get("show_region")
                if show_region is not None:
                    a.add_patch(
                        mpl.patches.Rectangle(
                            xy=(show_region[0], show_region[2]),
                            width=(show_region[1] - show_region[0]),
                            height=(show_region[3] - show_region[2]),
                            linewidth=1,
                            fill=False,
                        )
                    )

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


def raps(
    data: pd.DataFrame | xr.DataArray | xr.Dataset,
    names: typing.Any,
    plot_type: str = "mpl",
    filter_str: str | None = None,
    **kwargs: typing.Any,
) -> None:
    """
    Compute and plot the Radially Averaged Power Spectrum input data.

    Parameters
    ----------
    data : Union[pandas.DataFrame, str, list, xarray.Dataset, xarray.DataArray]
        if dataframe: need with columns 'x', 'y', and other columns to calc RAPS for.
        if str: should be a .nc or .tif file.
        if list: list of grids or filenames.
    names : typing.Any
        names of pandas.dataframe columns, xarray.Dataset variables, xarray.DataArray
        variable, or files to calculate and plot RAPS for.
    plot_type : str, optional
        choose whether to plot with PyGMT or matplotlib, by default 'mpl'
    filter_str : str
        GMT string to use for pre-filtering data, ex. "c100e3+h" is a 100km low-pass
        cosine filter, by default is None.

    Keyword Args
    ------------
    region : str | tuple[float, float, float, float]
        grid region if input is not a grid, in format [xmin, xmax, ymin, ymax]
    spacing : float
        grid spacing if input is not a grid
    """
    # Check if seaborn is installed
    if sns is None:
        msg = "Missing optional dependency 'seaborn' required for plotting."
        raise ImportError(msg)

    region = kwargs.get("region")
    spacing = kwargs.get("spacing")

    if plot_type == "pygmt":
        spec = pygmt.Figure()
        spec.basemap(
            region="10/1000/.001/10000",
            projection="X-10cl/10cl",
            frame=[
                "WSne",
                'xa1f3p+l"Wavelength (km)"',
                'ya1f3p+l"Power (mGal@+2@+km)"',
            ],
        )
    elif plot_type == "mpl":
        plt.figure()
    for i, j in enumerate(names):
        if isinstance(data, pd.DataFrame):
            df = data
            grid = pygmt.xyz2grd(
                df[["x", "y", j]],
                registration="p",
                region=region,
                spacing=spacing,
            )
            pygmt.grdfill(grid, mode="n", outgrid="tmp_outputs/fft.nc")
            grid = "tmp_outputs/fft.nc"
        elif isinstance(data, str):
            grid = data
        elif isinstance(data, list):
            data[i].to_netcdf("tmp_outputs/fft.nc")
            pygmt.grdfill("tmp_outputs/fft.nc", mode="n", outgrid="tmp_outputs/fft.nc")
            grid = "tmp_outputs/fft.nc"
        elif isinstance(data, xr.Dataset):
            data[j].to_netcdf("tmp_outputs/fft.nc")
            pygmt.grdfill("tmp_outputs/fft.nc", mode="n", outgrid="tmp_outputs/fft.nc")
            grid = "tmp_outputs/fft.nc"
        elif isinstance(data, xr.DataArray):
            data.to_netcdf("tmp_outputs/fft.nc")
            pygmt.grdfill("tmp_outputs/fft.nc", mode="n", outgrid="tmp_outputs/fft.nc")
            grid = "tmp_outputs/fft.nc"
        if filter_str is not None:
            with pygmt.clib.Session() as session:
                fin = grid
                fout = "tmp_outputs/fft.nc"
                args = f"{fin} -F{filter_str} -D0 -G{fout}"
                session.call_module("grdfilter", args)
            grid = "tmp_outputs/fft.nc"
        with pygmt.clib.Session() as session:
            fin = grid
            fout = "tmp_outputs/raps.txt"
            args = f"{fin} -Er+wk -Na+d -G{fout}"
            session.call_module("grdfft", args)
        if plot_type == "mpl":
            raps_df = pd.read_csv(
                "tmp_outputs/raps.txt",
                header=None,
                delimiter="\t",
                names=("wavelength", "power", "stdev"),
            )
            ax = sns.lineplot(x=raps_df.wavelength, y=raps_df.power, label=j)
            ax = sns.scatterplot(x=raps_df.wavelength, y=raps_df.power)
            ax.set_xlabel("Wavelength (km)")
            ax.set_ylabel("Radially Averaged Power ($mGal^{2}km$)")
        elif plot_type == "pygmt":
            color = f"{random.randrange(255)}/{random.randrange(255)}/{random.randrange(255)}"  # noqa: E501
            spec.plot("tmp_outputs/raps.txt", pen=f"1p,{color}")
            spec.plot(
                "tmp_outputs/raps.txt",
                color=color,
                style="T5p",
                # error_bar='y+p0.5p',
                label=j,
            )
    if plot_type == "mpl":
        ax.invert_xaxis()
        ax.set_yscale("log")
        ax.set_xlim(200, 0)
        # ax.set_xscale('log')
    elif plot_type == "pygmt":
        spec.show()

    # plt.phase_spectrum(df_anomalies.ice_forward_grav, label='phase spectrum')
    # plt.psd(df_anomalies.ice_forward_grav, label='psd')
    # plt.legend()


def coherency(
    grids: tuple[xr.DataArray | str | pd.DataFrame, xr.DataArray | str | pd.DataFrame],
    label: str,
    **kwargs: typing.Any,
) -> typing.Any:
    """
    Compute and plot the Radially Averaged Power Spectrum input data.

    Parameters
    ----------
    grids : list
        list of 2 grids to calculate the cohereny between.
        grid format can be str (filename), xarray.DataArray, or pandas.DataFrame.
    label : str
        used to label line.

    Keyword Args
    ------------
    region : str | tuple[float, float, float, float]
        grid region if input is pandas.DataFrame, in format [xmin, xmax, ymin, ymax]
    spacing : float
        grid spacing if input is pandas.DataFrame
    """

    # Check if seaborn is installed
    if sns is None:
        msg = "Missing optional dependency 'seaborn' required for plotting."
        raise ImportError(msg)

    region = kwargs.get("region")
    spacing = kwargs.get("spacing")

    plt.figure()

    if isinstance(grids[0], (str, xr.DataArray)):
        pygmt.grdfill(grids[0], mode="n", outgrid="tmp_outputs/fft_1.nc")
        pygmt.grdfill(grids[1], mode="n", outgrid="tmp_outputs/fft_2.nc")
    elif isinstance(grids[0], pd.DataFrame):  # type: ignore[unreachable]
        grid1 = pygmt.xyz2grd(
            grids[0],
            registration="p",
            region=region,
            spacing=spacing,
        )
        grid2 = pygmt.xyz2grd(
            grids[1],
            registration="p",
            region=region,
            spacing=spacing,
        )
        pygmt.grdfill(grid1, mode="n", outgrid="tmp_outputs/fft_1.nc")
        pygmt.grdfill(grid2, mode="n", outgrid="tmp_outputs/fft_2.nc")

    with pygmt.clib.Session() as session:
        fin1 = "tmp_outputs/fft_1.nc"
        fin2 = "tmp_outputs/fft_2.nc"
        fout = "tmp_outputs/coherency.txt"
        args = f"{fin1} {fin2} -E+wk+n -Na+d -G{fout}"
        session.call_module("grdfft", args)

    df = pd.read_csv(
        "tmp_outputs/coherency.txt",
        header=None,
        delimiter="\t",
        names=(
            "Wavelength (km)",
            "Xpower",
            "stdev_xp",
            "Ypower",
            "stdev_yp",
            "coherent power",
            "stdev_cp",
            "noise power",
            "stdev_np",
            "phase",
            "stdev_p",
            "admittance",
            "stdev_a",
            "gain",
            "stdev_g",
            "coherency",
            "stdev_c",
        ),
    )
    ax = sns.lineplot(df["Wavelength (km)"], df.coherency, label=label)  # pylint: disable=too-many-function-args
    ax = sns.scatterplot(x=df["Wavelength (km)"], y=df.coherency)

    ax.invert_xaxis()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(2000, 10)

    return ax


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
    # special_cases = {
    #     1: (1, 1),
    #     2: (1, 2),
    #     3: (2, 2),
    #     4: (2, 2),
    #     5: (2, 3),
    #     6: (2, 3),
    #     7: (3, 3),
    #     8: (3, 3),
    #     9: (3, 3),
    # }
    special_cases = {3: (2, 1), 5: (2, 3)}
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
    grid: xr.DataArray,
    shapefile: str | gpd.geodataframe.GeoDataFrame | None = None,
    robust: bool = False,
    region: tuple[float, float, float, float] | None = None,
    hemisphere: str | None = None,
) -> tuple[float, float]:
    """
    Get a grids max and min values.

    Parameters
    ----------
    grid : xarray.DataArray
        grid to get values for
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
        grid = subset_grid(grid, region)

    if shapefile is None:
        if robust:
            v_min, v_max = np.nanquantile(grid, [0.02, 0.98])
        else:
            v_min, v_max = np.nanmin(grid), np.nanmax(grid)

    elif shapefile is not None:
        masked = mask_from_shp(
            shapefile,
            hemisphere=hemisphere,
            xr_grid=grid,
            masked=True,
            invert=False,
        )

        if robust is True:
            v_min, v_max = np.nanquantile(masked, [0.02, 0.98])
        elif robust is False:
            v_min, v_max = np.nanmin(masked), np.nanmax(masked)

    assert v_min <= v_max, "min value should be less than or equal to max value"  # pylint: disable=possibly-used-before-assignment
    return (v_min, v_max)


def shapes_to_df(
    shapes: list[float],
    hemisphere: str | None = None,
) -> pd.DataFrame:
    """
    convert the output of `regions.draw_region` and `profiles.draw_lines` to a dataframe
    of x and y points

    Parameters
    ----------
    hemisphere : str, optional
        choose between the "north" or "south" hemispheres
    shapes : list
        list of vertices

    Returns
    -------
    pandas.DataFrame
        Dataframe with x, y, and shape_num.
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
        logging.info(
            "supplied dataframe has multiple polygons, only using the first one."
        )
        df = df[df.shape_num == 0]

    reg: tuple[float, float, float, float] = vd.get_region((df.x, df.y))

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
    data_coords = (df.x, df.y)

    # remove additional polygons
    if df.shape_num.max() > 0:
        logging.info(
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
