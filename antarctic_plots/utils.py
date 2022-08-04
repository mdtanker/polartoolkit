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
from typing import Union
import warnings

from antarctic_plots import fetch

def get_grid_info(grid):
    """
    Returns information of the specified grid.

    Parameters
    ----------
    grid : str or xarray.DataArray
        Input grid to get info from. Filename string or loaded grid.

    Returns
    -------
    list
        (string of grid spacing, array with the region boundaries, data min, data max)
    """

    spacing = pygmt.grdinfo(grid, per_column="n", o=7)[:-1]
    region = [int(pygmt.grdinfo(grid, per_column="n", o=i)[:-1]) for i in range(4)]
    min = float(pygmt.grdinfo(grid, per_column="n",o=4)[:-1])
    max = float(pygmt.grdinfo(grid, per_column="n",o=5)[:-1])

    return spacing, region, min, max


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

def alter_region(
    starting_region: np.ndarray,
    zoom: float = 0,
    n_shift: float = 0,
    w_shift: float = 0,
    buffer: float = 0,):
 
    e = starting_region[0] +zoom + w_shift
    w = starting_region[1] -zoom + w_shift
    n = starting_region[2] +zoom - n_shift
    s = starting_region[3] -zoom - n_shift
    region = [e, w, n, s]

    e_buff, w_buff, n_buff, s_buff = int(e-buffer), int(w+buffer), int(n-buffer), int(s+buffer)

    buffer_region = [e_buff, w_buff, n_buff, s_buff]

    fig_height = 80
    fig_width = fig_height*(w-e)/(s-n)
    
    ratio = (s-n)/(fig_height/1000)

    proj = f"x1:{ratio}"

    print(f"region is {int((w-e)/1e3)} x {int((s-n)/1e3)} km")
    return region, buffer_region, proj

def set_proj(
    region : Union[str or np.ndarray],
    fig_height : float = 10,
) -> str:
    """
    Gives GMT format projection string from region and figure height.
    Inspired from https://github.com/mrsiegfried/Venturelli2020-GRL.

    Parameters
    ----------
    region : Union[str or np.ndarray]
        GMT-format region str or list (e, w, n, s) in meters EPSG:3031
    fig_height : float
        desired figure height in cm

    Returns
    -------
    str
        _description_
    """
    e, w, n, s = region
    fig_width = (fig_height*10)*(w-e)/(s-n)
    
    ratio = (s-n)/(fig_height/100)
    proj = f"x1:{ratio}"
    proj_latlon = f"s0/-90/-71/1:{ratio}"

    return proj, proj_latlon, fig_width


def plot_grd(
    grid : Union[str or xr.DataArray], 
    cmap : str = 'viridis',
    plot_region : Union[str or np.ndarray] = None, 
    coast : bool = False,
    origin_shift: str = 'initialize',
    **kwargs
    ):
    """
    Helps easily create PyGMT maps, individually or as subplots.

    Parameters
    ----------
    grid : Union[str or xr.DataArray]
        grid file to plot, either loaded xr.DataArray or string of a filename
    cmap : str, optional
        GMT color scale to use, by default 'viridis'
    plot_region : Union[str or np.ndarray], optional
        region to plot, by default is extent of input grid
    coast : bool, optional
        choose whether to plot Antarctic coastline and grounding line, by default False
    origin_shift : str, optional
        set to 'x_shift' to instead add plot to right of previous plot, or 'y_shift' to 
        add plot above previous plot, by default 'initialize'.

    Keyword Args
    ------------
    grd2cpt : bool
        use GMT module grd2cpt to set color scale from grid values, by default is False
    cmap_region : Union[str or np.ndarray]
        region to use to define color scale if grd2cpt is True, by default is plot_region
    cbar_label : str
        label to add to colorbar.
    points : pd.DataFrame
        points to plot on map, must contain columns 'x' and 'y'.
    square : np.ndarray
        GMT-format region to use to plot a square.
    cpt_lims : Union[str or tuple]
        limits to use for color scale max and min, by default is max and min of data.
    fig : pygmt.Figure()
        if adding subplots, set the first returned figure to a variable, and add that 
        variable as the kwargs 'fig' to subsequent calls to plot_grd.
    
    Returns
    -------
    PyGMT.Figure() 
        

    Example
    -------
    ::

        fig = utils.plot_grd('grid1.nc')

        utils.plot_grd(
            'grid2.nc',
            origin_shift='xshift',
            fig=fig
            )
       
    
    """

    warnings.filterwarnings('ignore', message="pandas.Int64Index")
    warnings.filterwarnings('ignore', message="pandas.Float64Index")

    if plot_region is None:
        plot_region = get_grid_info(grid)[1]

    cmap_region = kwargs.get('cmap_region', plot_region) 
    square = kwargs.get('square', None)
    cpt_lims = kwargs.get('cpt_lims', None)

    proj, proj_latlon, fig_width = set_proj(plot_region)

    # initialize figure or shift for new subplot
    if origin_shift=='initialize':
        fig = pygmt.Figure()   
    elif origin_shift=='xshift':
        fig=kwargs.get('fig')
        fig.shift_origin(xshift=(fig_width + 2)/10)
    elif origin_shift=='yshift':
        fig=kwargs.get('fig')
        fig.shift_origin(yshift=(fig_height + 12)/10)

    # set cmap
    if kwargs.get('grd2cpt', False) is True:
        pygmt.grd2cpt(
            cmap=cmap, 
            grid=grid, 
            region=kwargs.get('cmap_region'), 
            background=True, 
            continuous=True,
        )
    elif cpt_lims is not None:
        pygmt.makecpt(
            cmap=cmap, 
            background=True, 
            continuous=True,
            series=cpt_lims,
        )
    else:
        min, max = get_grid_info(grid)[2:]
        pygmt.makecpt(
            cmap=cmap, 
            background=True, 
            continuous=True,
            series=(min, max),
        )

    # display grid
    fig.grdimage(
        grid=grid,
        cmap=True,
        projection=proj, 
        region=plot_region,
        nan_transparent=True,
        frame=['+gwhite'],
    )

    # display colorbar
    fig.colorbar(
        cmap=True, 
        position='jBC+jTC+h', 
        frame=f"x+l{kwargs.get('cbar_label',' ')}",
    )

    # plot groundingline and coastlines    
    if coast==True:
        fig.plot(
            data=fetch.groundingline(),
            pen="1.2p,black",
        )

    # add datapoints
    if kwargs.get('points', None) is not None:
        fig.plot(
            x = points.x, 
            y = points.y, 
            style = 'c1.2p',
            color = 'black',
        )

    # add square
    if square is not None:
        fig.plot(
            x = [square[0], square[0], square[1], square[1], square[0]], 
            y = [square[2], square[3], square[3], square[2], square[2]], 
            pen = '2p,black', 
        )
    
    return fig