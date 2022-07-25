# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#
import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
from pyproj import Transformer


def get_grid_info(grid):
    """
    Returns the spacing and region of an input grid.

    Parameters
    ----------
    grid : str or xarray.DataArray
        Input grid to get info from. Filename string or loaded grid.

    Returns
    -------
    tuple
        tuple, first item is a string of grid spacing, second item is
        an array with the region boundaries
    """

    spacing = pygmt.grdinfo(grid, per_column="n", o=7)[:-1]
    region = [int(pygmt.grdinfo(grid, per_column="n", o=i)[:-1]) for i in range(4)]
    return spacing, region


def dd2dms(dd: float):
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


def latlon_to_epsg3031(
    df,
    reg: bool = False,
    input=["lon", "lat"],
    output=["x", "y"],
):
    """
    Convert coordinates from EPSG:4326 WGS84 in decimal degrees to EPSG:3031 Antarctic
    Polar Stereographic in meters.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe with latitude and longitude columns
    reg : bool, optional
        if true, returns a GMT formatted region string, by default False
    input : list, optional
        set names for input columns, by default ["lon", "lat"]
    output : list, optional
        set names for output columns, by default ["x", "y"]

    Returns
    -------
    pd.DataFrame or np.ndarray
        Updated dataframe with new easting and northing columns or np.ndarray in format
        [e, w, n, s]
    """
    transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
    df[output[0]], df[output[1]] = transformer.transform(
        df[input[1]].tolist(), df[input[0]].tolist()
    )
    if reg is True:
        df = [
            df[output[0]].min(),
            df[output[0]].max(),
            df[output[1]].max(),
            df[output[1]].min(),
        ]
    return df


def epsg3031_to_latlon(df, reg: bool = False, input=["x", "y"], output=["lon", "lat"]):
    """
        Convert coordinates from EPSG:3031 Antarctic Polar Stereographic in meters to
        EPSG:4326 WGS84 in decimal degrees.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe with easting and northing columns
    reg : bool, optional
        if true, returns a GMT formatted region string, by default False
    input : list, optional
        set names for input columns, by default ["x", "y"]
    output : list, optional
        set names for output columns, by default ["lon", "lat"]

    Returns
    -------
    pd.DataFrame or np.ndarray
        Updated dataframe with new latitude and longitude columns or np.ndarray in
        format [e, w, n, s]
    """
    transformer = Transformer.from_crs("epsg:3031", "epsg:4326")
    df[output[1]], df[output[0]] = transformer.transform(
        df[input[1]].tolist(), df[input[0]].tolist()
    )
    if reg is True:
        df = [
            df[output[0]].min(),
            df[output[0]].max(),
            df[output[1]].min(),
            df[output[1]].max(),
        ]
    return df


def reg_str_to_df(input, names=["x", "y"]):
    """
    Convert GMT region string [e, w, n, s] to pandas dataframe with coordinates of
    region corners

    Parameters
    ----------
    input : np.ndarray
        Array of 4 strings in GMT format; [e, w, n, s]
    names : list, optional
        Names of names to use for easting and northing, by default ["x", "y"]

    Returns
    -------
    pd.DataFrame
        Dataframe with easting and northing columns, and a row for each corner of the
        region.
    """
    bl = (input[0], input[2])
    br = (input[1], input[2])
    tl = (input[0], input[3])
    tr = (input[1], input[3])
    df = pd.DataFrame(data=[bl, br, tl, tr], columns=(names[0], names[1]))
    return df


def GMT_reg_xy_to_ll(input):
    """
    Convert GMT region string [e, w, n, s] in EPSG:3031 to deg:min:sec

    Parameters
    ----------
    input : np.ndarray
        Array of 4 strings in GMT format; [e, w, n, s] in meters

    Returns
    -------
    np.ndarray
        Array of 4 strings in GMT format; [e, w, n, s] in lat, lon
    """
    df = reg_str_to_df(input)
    df_proj = epsg3031_to_latlon(df, reg=True)
    output = [dd2dms(x) for x in df_proj]
    return output


def mask_from_shp(
    shapefile: str,
    invert: bool = True,
    xr_grid=None,
    grid_file: str = None,
    region=None,
    spacing=None,
    masked: bool = False,
    crs: str = "epsg:3031",
):
    """
    Create a mask or a masked grid from area inside or outside of a closed shapefile.

    Parameters
    ----------
    shapefile : str
        path to .shp filename, must by in same directory as accompanying files : .shx,
        .prj, .dbf, should be a closed polygon file.
    invert : bool, optional
        choose whether to mask data outside the shape (False) or inside the shape
        (True), by default True (masks inside of shape)
    xr_grid : xarray.DataArray, optional
        _xarray.DataArray; to use to define region, or to mask, by default None
    grid_file : str, optional
        path to a .nc or .tif file to use to define region or to mask, by default None
    region : str or np.ndarray, optional
        GMT region string or 1x4 ndarray in meters to create a dummy grid if none are
        supplied, by default None
    spacing : str or int, optional
        grid spacing in meters to create a dummy grid if none are supplied, by default
        None
    masked : bool, optional
        choose whether to return the masked grid (True) or the mask itself (False), by
        default False
    crs : str, optional
        if grid is provided, rasterio needs to assign a coordinate reference system via
        an epsg code, by default "epsg:3031"

    Returns
    -------
    xarray.DataArray
        Returns either a masked grid, or the mask grid itself.
    """
    shp = gpd.read_file(shapefile).geometry
    if xr_grid is None and grid_file is None:
        coords = vd.grid_coordinates(
            region=region, spacing=spacing, pixel_register=True
        )
        ds = vd.make_xarray_grid(
            coords, np.ones_like(coords[0]), dims=("y", "x"), data_names="z"
        )
        xds = ds.z.rio.write_crs(crs)
    elif xr_grid is not None:
        xds = xr_grid.rio.write_crs(crs)
    elif grid_file is not None:
        xds = xr.load_dataarray(grid_file).rio.write_crs(crs)

    masked_grd = xds.rio.clip(shp.geometry, xds.rio.crs, drop=False, invert=invert)
    mask_grd = np.isfinite(masked_grd)

    if masked is True:
        output = masked_grd
    elif masked is False:
        output = mask_grd
    return output


# def plot_grd(
#     grid,
#     cmap: str,
#     cbar_label: str,
#     plot_region=None,
#     cmap_region=None,
#     coast=False,
#     grd2cpt_name=False,
#     origin_shift="initialize",
# ):
#     """
#     Function to automate PyGMT plotting

#     Parameters
#     ----------
#     grid : str or xarray.DataArray
#         grid to plot.
#     cmap : str
#         GMT colorscale to use.
#     cbar_label : str
#         label to add to colorbar.
#     plot_region : str or np.ndarray, optional
#         GMT region to set map extent to, by default is entire Antarctic region
#     cmap_region : str or np.ndarray, optional
#         GMT region to define the color scale limits, by default is equal to
#           plot_region
#     coast : bool, optional
#         choose to plot coastline and groundingline, by default False
#     grd2cpt_name : bool, optional
#         file name which will be given to a cpt create with pygmt.grd2cpt() and used in
#         the plot, by default False
#     origin_shift : str, optional
#         choose whether to start a new figure:'initialize', create a new subplot to the
#         right:'xshift', or create a new subplot above:'yshift', by default
#           "initialize"
#     """
#     import warnings

#     warnings.filterwarnings("ignore", message="pandas.Int64Index")
#     warnings.filterwarnings("ignore", message="pandas.Float64Index")

#     global fig, projection
#     if plot_region is None:
#         plot_region = (-3330000, 3330000, -3330000, 3330000)
#     if cmap_region is None:
#         cmap_region = plot_region

#     # initialize figure or shift for new subplot
#     if origin_shift == "initialize":
#         fig = pygmt.Figure()
#     elif origin_shift == "xshift":
#         fig.shift_origin(xshift=(fig_width + 2) / 10)
#     elif origin_shift == "yshift":
#         fig.shift_origin(yshift=(fig_height + 12) / 10)

#     # set cmap
#     if grd2cpt_name:
#         pygmt.grd2cpt(
#             cmap=cmap,
#             grid=grid,
#             region=cmap_region,
#             background=True,
#             continuous=True,
#             output=f"plotting/{grd2cpt_name}.cpt",
#         )
#         cmap = f"plotting/{grd2cpt_name}.cpt"

#     fig.grdimage(
#         grid=grid,
#         cmap=cmap,
#         projection=projection,
#         region=plot_region,
#         nan_transparent=True,
#         frame=["+gwhite"],
#     )

#     fig.colorbar(cmap=cmap, position="jBC+jTC+h", frame=f'x+l"{cbar_label}"')

#     if coast == True:
#         fig.plot(
#             projection=projection,
#             region=plot_region,
#             # data=gpd.read_file("plotting/GroundingLine_Antarctica_v02.shp"),
#             data=gpd.read_file(fetch.groundingline()),
#             pen="1.2p,black",
#             verbose="q",
#         )

#     if plot_region == buffer_reg:
#         fig.plot(
#             x=[inv_reg[0], inv_reg[0], inv_reg[1], inv_reg[1], inv_reg[0]],
#             y=[inv_reg[2], inv_reg[3], inv_reg[3], inv_reg[2], inv_reg[2]],
#             pen="2p,black",
#             projection=projection,
#             region=plot_region,
#         )
