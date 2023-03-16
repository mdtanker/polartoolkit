# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#

from typing import TYPE_CHECKING, Callable, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
import pyogrio
import rioxarray  # noqa
import verde as vd
import xarray as xr
from pyproj import Transformer

from antarctic_plots import fetch, maps

if TYPE_CHECKING:
    import geopandas as gpd

try:
    import seaborn as sns
except ImportError:
    _has_seaborn = False
else:
    _has_seaborn = True


# function to give RMSE of data
def RMSE(data):
    return np.sqrt(np.nanmedian(data**2).item())
    # return np.sqrt(np.nanmean(data**2).item())


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
        (string of grid spacing,
        array with the region boundaries,
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
    if isinstance(grid, xr.DataArray):
        if int(len(grid.dims)) > 2:
            grid = grid.squeeze()

    try:
        spacing = float(pygmt.grdinfo(grid, per_column="n", o=7)[:-1])
    except Exception:  # pygmt.exceptions.GMTInvalidInput:
        print("grid spacing can't be extracted")
        spacing = None

    try:
        region = [
            float(pygmt.grdinfo(grid, per_column="n", o=i)[:-1]) for i in range(4)
        ]
    except Exception:  # pygmt.exceptions.GMTInvalidInput:
        print("grid region can't be extracted")
        region = None

    try:
        zmin = float(pygmt.grdinfo(grid, per_column="n", o=4)[:-1])
    except Exception:  # pygmt.exceptions.GMTInvalidInput:
        print("grid zmin can't be extracted")
        zmin = None

    try:
        zmax = float(pygmt.grdinfo(grid, per_column="n", o=5)[:-1])
    except Exception:  # pygmt.exceptions.GMTInvalidInput:
        print("grid zmax can't be extracted")
        zmax = None

    try:
        reg = grid.gmt.registration
        registration = "g" if reg == 0 else "p"
    except AttributeError:
        print(
            "grid registration not extracted, re-trying with file loaded as xarray grid"
        )  # noqa
        grid = xr.load_dataarray(grid)
        reg = grid.gmt.registration
        registration = "g" if reg == 0 else "p"
    except Exception:
        print("grid registration can't be extracted")
        registration = None

    return spacing, region, zmin, zmax, registration


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

    if isinstance(df, list):
        ll = df.copy()
        df = list(transformer.transform(ll[0], ll[1]))
    else:
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
    df : pd.DataFrame or list
        input dataframe with easting and northing columns, or list [x,y]
    reg : bool, optional
        if true, returns a GMT formatted region string, by default False
    input : list, optional
        set names for input columns, by default ["x", "y"]
    output : list, optional
        set names for output columns, by default ["lon", "lat"]

    Returns
    -------
    pd.DataFrame or np.ndarray
        Updated dataframe with new latitude and longitude columns, np.ndarray in
        format [e, w, n, s], or list in format [lat, lon]
    """

    transformer = Transformer.from_crs("epsg:3031", "epsg:4326")

    df_out = df.copy()

    if isinstance(df, list):
        df_out = list(transformer.transform(df_out[0], df_out[1]))
    else:
        df_out[output[1]], df_out[output[0]] = transformer.transform(
            df_out[input[0]].tolist(), df_out[input[1]].tolist()
        )
        if reg is True:
            df_out = [
                df_out[output[0]].min(),
                df_out[output[0]].max(),
                df_out[output[1]].min(),
                df_out[output[1]].max(),
            ]
    return df_out


def reg_str_to_df(input, names=["x", "y"], reverse=False):
    """
    Convert GMT region string [e, w, n, s] to pandas dataframe with coordinates of
    region corners

    Parameters
    ----------
    input : np.ndarray or pd.DataFrame
        Array of 4 strings in GMT format; [e, w, n, s], or, if `reverse` is True, a
        DataFrame with easting and northing columns with names set by `names`
    names : list, optional
        List of names to use for easting and northing, by default ["x", "y"]
    reverse : bool, False
        If True, convert from df to region string.
    Returns
    -------
    pd.DataFrame or np.ndarray
        Dataframe with easting and northing columns, and a row for each corner of the
        region, or, if `reverse` is True, a array in the format [e,w,n,s].
    """
    if reverse is False:
        bl = (input[0], input[2])
        br = (input[1], input[2])
        tl = (input[0], input[3])
        tr = (input[1], input[3])
        df = pd.DataFrame(data=[bl, br, tl, tr], columns=(names[0], names[1]))

    elif reverse is True:
        df = [
            input[names[0]][0],
            input[names[0]][1],
            input[names[1]][0],
            input[names[1]][2],
        ]

    return df


def GMT_reg_xy_to_ll(input, decimal_degree=False):
    """
    Convert GMT region string [e, w, n, s] in EPSG:3031 to deg:min:sec

    Parameters
    ----------
    input : np.ndarray
        Array of 4 strings in GMT format; [e, w, n, s] in meters
    decimal_degrees: bool, False
        if True, will return results as decimal degrees instead of deg:min:sec

    Returns
    -------
    np.ndarray
        Array of 4 strings in GMT format; [e, w, n, s] in lat, lon
    """
    df = reg_str_to_df(input)
    df_proj = epsg3031_to_latlon(df, reg=True)

    if decimal_degree is False:
        output = [dd2dms(x) for x in df_proj]
    elif decimal_degree is True:
        output = df_proj
    return output


def GMT_reg_to_bounding_box(input):
    """
    Convert GMT region string [e, w, n, s] to bounding box format used for icepyx:
    [ lower left longitude,
      lower left latitude,
      upper right longitude,
      uper right latitude
    ]
    Same format as [xmin, ymin, xmax, ymax], used for `bbox` parameter of
    geopandas.read_file

    Parameters
    ----------
    input : np.ndarray
        Array of 4 strings in GMT format; [e, w, n, s] in meters or degrees.

    Returns
    -------
    list
        Array of 4 strings in bounding box format.
    """
    return [input[0], input[2], input[1], input[3]]


def region_to_bounding_box(input):
    """
    Convert regions in format [e,w,n,s] in EPSG:3031 meters to format
    [ll lon, ll lat, ur lon, ur lat] to be used as a bounding box in icepyx.

    Parameters
    ----------
    input : np.ndarray
        Array of 4 strings in GMT format; [e, w, n, s] in meters

    Returns
    -------
    np.ndarray
        Array of 4 strings in bounding box format.
    """
    reg_ll = GMT_reg_xy_to_ll(input, decimal_degree=True)
    box = GMT_reg_to_bounding_box(reg_ll)
    return box


def points_inside_region(
    df: pd.DataFrame,
    region: list,
    names: list = ["x", "y"],
):
    """
    return a subset of a dataframe which is within a region

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with columns 'x','y' to use for defining if within region
    region : list
        GMT region string to use as bounds for new subset dataframe

    Returns
    -------
    pd.DataFrame
       returns a subset dataframe
    """
    # make a copy of the dataframe
    df1 = df.copy()

    # make column of booleans for whether row is within the region
    df1["inside"] = vd.inside(coordinates=(df1[names[0]], df1[names[1]]), region=region)

    # subset if True
    df_inside = df1.loc[df1.inside == True].copy()  # noqa

    # drop the column 'inside'
    df_inside.drop(columns="inside", inplace=True)

    return df_inside


def block_reduce(
    df: pd.DataFrame,
    reduction: Callable,
    input_coord_names: list = ["x", "y"],
    input_data_names: list = None,
    **kwargs,
):
    # define verde reducer function
    reducer = vd.BlockReduce(reduction, **kwargs)

    # if no data names provided, use all columns
    if input_data_names is None:
        input_data_names = list(df.columns.drop(input_coord_names))

    # get tuples of pd.Series
    input_coords = tuple([df[col] for col in input_coord_names])
    input_data = tuple([df[col] for col in input_data_names])

    # apply reduction
    coordinates, data = reducer.filter(
        coordinates=input_coords,
        data=input_data,
    )

    # add reduced coordinates to a dictionary
    coord_cols = {k: v for k, v in zip(input_coord_names, coordinates)}

    # add reduced data to a dictionary
    if len(input_data_names) < 2:
        data_cols = {input_data_names[0]: data}
    else:
        data_cols = {k: v for k, v in zip(input_data_names, data)}

    # merge dicts and create dataframe
    df_reduced = pd.DataFrame(data=coord_cols | data_cols)

    return df_reduced


def mask_from_shp(
    shapefile: Union[str or gpd.geodataframe.GeoDataFrame],
    invert: bool = True,
    xr_grid=None,
    grid_file: str = None,
    region=None,
    spacing=None,
    masked: bool = False,
    crs: str = "epsg:3031",
    pixel_register=True,
    input_coord_names=("x", "y"),
):
    """
    Create a mask or a masked grid from area inside or outside of a closed shapefile.

    Parameters
    ----------
    shapefile : Union[str or gpd.geodataframe.GeoDataFrame]
        either path to .shp filename, must by in same directory as accompanying files :
        .shx, .prj, .dbf, should be a closed polygon file, or shapefile which as already
         been loaded into a geodataframe.
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

    if isinstance(shapefile, str):
        shp = pyogrio.read_dataframe(shapefile)
    else:
        shp = shapefile

    if xr_grid is None and grid_file is None:
        coords = vd.grid_coordinates(
            region=region,
            spacing=spacing,
            pixel_register=pixel_register,
        )
        ds = vd.make_xarray_grid(
            coords, np.ones_like(coords[0]), dims=("y", "x"), data_names="z"
        )
        xds = ds.z.rio.write_crs(crs)
    elif xr_grid is not None:
        xds = xr_grid.rio.write_crs(crs).rio.set_spatial_dims(
            input_coord_names[0], input_coord_names[1]
        )
    elif grid_file is not None:
        xds = (
            xr.load_dataarray(grid_file)
            .rio.write_crs(crs)
            .rio.set_spatial_dims(input_coord_names[0], input_coord_names[1])
        )

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

    return output


def alter_region(
    starting_region: np.ndarray,
    zoom: float = 0,
    n_shift: float = 0,
    w_shift: float = 0,
    buffer: float = 0,
    print_reg: bool = False,
):
    """
    Change a region string by shifting the box east/west or north/south, zooming in or
    out, or adding a seperate buffer region.

    Parameters
    ----------
    starting_region : np.ndarray
        Initial GMT formatted region in meters, [e,w,n,s]
    zoom : float, optional
        zoom in or out, in meters, by default 0
    n_shift : float, optional
        shift north, or south if negative, in meters, by default 0
    w_shift : float, optional
        shift west, or eash if negative, in meters, by default 0
    buffer : float, optional
        create new region which is zoomed out in all direction, in meters, by default 0
    print_reg : bool, optional
        print out the dimensions of the altered region, by default False

    Returns
    -------
    list
        Returns of list of the following variables: region, buffer_region
    """
    E, W = starting_region[0], starting_region[1]
    N, S = starting_region[2], starting_region[3]

    e = E + zoom + w_shift
    w = W - zoom + w_shift

    n = N + zoom - n_shift
    s = S - zoom - n_shift

    region = [e, w, n, s]

    e_buff, w_buff, n_buff, s_buff = (
        int(e - buffer),
        int(w + buffer),
        int(n - buffer),
        int(s + buffer),
    )

    buffer_region = [e_buff, w_buff, n_buff, s_buff]

    if print_reg is True:
        print(f"inner region is {int((w-e)/1e3)} x {int((s-n)/1e3)} km")
    return region, buffer_region


def set_proj(
    region: Union[str or np.ndarray],
    fig_height: float = 15,
    fig_width: float = None,
) -> str:
    """
    Gives GMT format projection string from region and figure height or width.
    Inspired from https://github.com/mrsiegfried/Venturelli2020-GRL.

    Parameters
    ----------
    region : Union[str or np.ndarray]
        GMT-format region str or list (e, w, n, s) in meters EPSG:3031
    fig_height : float
        desired figure height in cm
    fig_width : float
        instead of using figure height, set the projection based on figure width in cm,
        by default is None

    Returns
    -------
    list
        returns a list of the following variables: (proj, proj_latlon, fig_width,
        fig_height)
    """
    e, w, n, s = region

    if fig_width is not None:
        fig_height = fig_width * (s - n) / (w - e)
        ratio = (w - e) / (fig_width / 100)
    else:
        fig_width = fig_height * (w - e) / (s - n)
        ratio = (s - n) / (fig_height / 100)

    proj = f"x1:{ratio}"
    proj_latlon = f"s0/-90/-71/1:{ratio}"

    return proj, proj_latlon, fig_width, fig_height


def grd_trend(
    da: xr.DataArray,
    coords: list = ["x", "y", "z"],
    deg: int = 1,
    plot: bool = False,
    plot_type="pygmt",
    **kwargs,
):
    """
    Fit an arbitrary order trend to a grid and use it to detrend.

    Parameters
    ----------
    da : xr.DataArray
        input grid
    coords : list, optional
        coordinate names of the supplied grid, by default ['x', 'y', 'z']
    deg : int, optional
        trend order to use, by default 1
    plot : bool, optional
        plot the results, by default False
    plot_type : str, by default "pygmt"
        choose to plot results with pygmt or xarray

    Returns
    -------
    tuple
        returns xr.DataArrays of the fitted surface, and the detrended grid.
    """

    # convert grid to a dataframe
    df = vd.grid_to_table(da).astype("float64")
    df.dropna(inplace=True)

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
            cmap = kwargs.get("cmap", "plasma")
            coast = kwargs.get("coast", True)
            inset = kwargs.get("inset", True)
            inset_pos = kwargs.get("inset_pos", "BL")
            origin_shift = kwargs.get("origin_shift", "yshift")
            fit_label = kwargs.get("fit_label", f"fitted trend (order {deg})")
            input_label = kwargs.get("input_label", "input grid")
            title = kwargs.get("title", "Detrending a grid")
            detrended_label = kwargs.get("detrended_label", "detrended")

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


def grd_compare(
    da1: Union[xr.DataArray, str],
    da2: Union[xr.DataArray, str],
    plot: bool = False,
    plot_type: str = "pygmt",
    **kwargs,
):
    """
    Find the difference between 2 grids and plot the results, if necessary resample and
    cut grids to match

    Parameters
    ----------
    da1 : xr.DataArray or str
        first grid, loaded grid of filename
    da2 : xr.DataArray or str
        second grid, loaded grid of filename
    plot : bool, optional
        plot the results, by default False
    plot_type : str, optional
        choose the style of plot, by default is pygmt, can choose xarray for faster,
        simplier plots.
    Keyword Args
    ------------
    shp_mask : str
        shapefile filename to use to mask the grids for setting the color range.
    robust : bool
        use xarray robust color lims instead of min and max, by default is False.
    region : Union[str, np.ndarray]
        choose a specific region to compare.
    rmse_in_title: bool
        add the RMSE to the title, by default is True.
    Returns
    -------
    list
        list of xr.DataArrays: (diff, resampled grid1, resampled grid2)
    """
    shp_mask = kwargs.get("shp_mask", None)
    region = kwargs.get("region", None)

    if isinstance(da1, str):
        da1 = xr.load_dataarray(da1)

    if isinstance(da2, str):
        da2 = xr.load_dataarray(da2)

    # first cut the grids to save time on the possible resampling below
    if region is not None:
        da1 = pygmt.grdcut(
            da1,
            region=region,
            verbose=kwargs.get("verbose", "e"),
        )
        da2 = pygmt.grdcut(
            da2,
            region=region,
            verbose=kwargs.get("verbose", "e"),
        )

    # get minimum grid spacing of both grids
    da1_spacing = float(get_grid_info(da1)[0])
    da2_spacing = float(get_grid_info(da2)[0])

    min_spacing = min(da1_spacing, da2_spacing)

    if da1_spacing != da2_spacing:
        print(
            "grid spacings don't match, using smaller spacing",
            f"({min_spacing}m).",
        )

    # get inside region of both grids
    da1_reg = get_grid_info(da1)[1]
    da2_reg = get_grid_info(da2)[1]

    e = max(da1_reg[0], da2_reg[0])
    w = min(da1_reg[1], da2_reg[1])
    n = max(da1_reg[2], da2_reg[2])
    s = min(da1_reg[3], da2_reg[3])

    sub_region = [e, w, n, s]

    if da1_reg != da2_reg:
        print(f"grid regions dont match, using inner region {sub_region}")

    # use registration from first grid, or from kwarg
    if kwargs.get("registration", None) is None:
        registration = get_grid_info(da1)[4]
    else:
        registration = kwargs.get("registration", None)

    grid1 = fetch.resample_grid(
        da1,
        spacing=min_spacing,
        region=sub_region,
        registration=registration,
        verbose=kwargs.get("verbose", "e"),
    )

    grid2 = fetch.resample_grid(
        da2,
        spacing=min_spacing,
        region=sub_region,
        registration=registration,
        verbose=kwargs.get("verbose", "e"),
    )

    dif = grid1 - grid2

    # get individual grid min/max values (and masked values if shapefile is provided)
    grid1_cpt_lims = get_min_max(grid1, shp_mask)
    grid2_cpt_lims = get_min_max(grid2, shp_mask)

    # if kwarg supplied, reset diff_maxabs
    diff_maxabs = kwargs.get("diff_maxabs", vd.maxabs(get_min_max(dif, shp_mask)))

    diff_lims = kwargs.get("diff_lims", (-diff_maxabs, diff_maxabs))

    # get min and max of both grids together
    vmin = min((grid1_cpt_lims[0], grid2_cpt_lims[0]))
    vmax = max(grid1_cpt_lims[1], grid2_cpt_lims[1])

    if plot is True:
        title = kwargs.get("title", "Comparing Grids")
        if kwargs.get("rmse_in_title", True) is True:
            title += f", RMSE: {round(RMSE(dif),kwargs.get('RMSE_decimals', 2))}"

        if plot_type == "pygmt":
            fig_height = kwargs.get("fig_height", 10)
            coast = kwargs.get("coast", True)
            origin_shift = kwargs.get("origin_shift", "xshift")
            cmap = kwargs.get("cmap", "viridis")
            subplot_labels = kwargs.get("subplot_labels", False)

            new_kwargs = {
                kw: kwargs[kw]
                for kw in kwargs
                if kw
                not in [
                    "cmap",
                    "region",
                    "coast",
                    "title",
                    "cpt_lims",
                    "fig_height",
                    "inset",
                    "inset_pos",
                ]
            }

            diff_kwargs = {
                kw: new_kwargs[kw]
                for kw in new_kwargs
                if kw
                not in [
                    "reverse_cpt",
                    "cbar_label",
                ]
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
                    font=kwargs.get("label_font", "26p,Helvetica,black"),
                    offset="j0/.3",
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
                    font=kwargs.get("label_font", "26p,Helvetica,black"),
                    offset="j0/.3",
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
                    font=kwargs.get("label_font", "26p,Helvetica,black"),
                    offset="j0/.3",
                    no_clip=True,
                )

            fig.show()

        elif plot_type == "xarray":
            if kwargs.get("robust", False) is True:
                vmin, vmax = None, None
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
                if kwargs.get("points", None) is not None:
                    a.plot(kwargs.get("points").x, kwargs.get("points").y, "k+")
                if kwargs.get("show_region", None) is not None:
                    show_region = kwargs.get("show_region", None)
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
    region: Union[str, np.ndarray],
    spacing: float,
    value: float,
    name: str,
):
    """
    Create a grid with 1 variable by defining a region, spacing, name and constant value

    Parameters
    ----------
    region : Union[str, np.ndarray]
        GMT format region for the inverion, by default is extent of gravity data,
    spacing : float
        spacing for grid
    value : float
        constant value to use for variable
    name : str
        name for variable

    Returns
    -------
    xr.DataArray
        Returns a xr.DataArray with 1 variable of constant value.
    """
    coords = vd.grid_coordinates(region=region, spacing=spacing, pixel_register=True)
    data = np.ones_like(coords[0]) * value
    grid = vd.make_xarray_grid(coords, data, dims=["y", "x"], data_names=name)
    return grid


def raps(
    data: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
    names: np.ndarray,
    plot_type: str = "mpl",
    filter: str = None,
    **kwargs,
):
    """
    Compute and plot the Radially Averaged Power Spectrum input data.

    Parameters
    ----------
    data : Union[pd.DataFrame, str, list, xr.Dataset, xr.Dataarray]
        if dataframe: need with columns 'x', 'y', and other columns to calc RAPS for.
        if str: should be a .nc or .tif file.
        if list: list of grids or filenames.
    names : np.ndarray
        names of pd.dataframe columns, xr.dataset variables, xr.dataarray variable, or
        files to calculate and plot RAPS for.
    plot_type : str, optional
        choose whether to plot with PyGMT or matplotlib, by default 'mpl'
    filter : str
        GMT string to use for pre-filtering data, ex. "c100e3+h" is a 100km low-pass
        cosine filter, by default is None.
    Keyword Args
    ------------
    region : Union[str, np.ndarray]
        grid region if input is not a grid
    spacing : float
        grid spacing if input is not a grid
    """
    if not _has_seaborn:
        raise ImportError("seaborn is required for this function.")

    region = kwargs.get("region", None)
    spacing = kwargs.get("spacing", None)

    if plot_type == "pygmt":
        import random

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
        if filter is not None:
            with pygmt.clib.Session() as session:
                fin = grid
                fout = "tmp_outputs/fft.nc"
                args = f"{fin} -F{filter} -D0 -G{fout}"
                session.call_module("grdfilter", args)
            grid = "tmp_outputs/fft.nc"
        with pygmt.clib.Session() as session:
            fin = grid
            fout = "tmp_outputs/raps.txt"
            args = f"{fin} -Er+wk -Na+d -G{fout}"
            session.call_module("grdfft", args)
        if plot_type == "mpl":
            raps = pd.read_csv(
                "tmp_outputs/raps.txt",
                header=None,
                delimiter="\t",
                names=("wavelength", "power", "stdev"),
            )
            ax = sns.lineplot(x=raps.wavelength, y=raps.power, label=j)
            ax = sns.scatterplot(x=raps.wavelength, y=raps.power)
            ax.set_xlabel("Wavelength (km)")
            ax.set_ylabel("Radially Averaged Power ($mGal^{2}km$)")
            pass
        elif plot_type == "pygmt":
            color = f"{random.randrange(255)}/{random.randrange(255)}/{random.randrange(255)}"  # noqa
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


def coherency(grids: list, label: str, **kwargs):
    """
    Compute and plot the Radially Averaged Power Spectrum input data.

    Parameters
    ----------
    grids : list
        list of 2 grids to calculate the cohereny between.
        grid format can be str (filename), xr.DataArray, or pd.DataFrame.
    label : str
        used to label line.
    Keyword Args
    ------------
    region : Union[str, np.ndarray]
        grid region if input is pd.DataFrame
    spacing : float
        grid spacing if input is pd.DataFrame
    """

    if not _has_seaborn:
        raise ImportError("seaborn is required for this function.")

    region = kwargs.get("region", None)
    spacing = kwargs.get("spacing", None)

    plt.figure()

    if isinstance(grids[0], (str, xr.DataArray)):
        pygmt.grdfill(grids[0], mode="n", outgrid="tmp_outputs/fft_1.nc")
        pygmt.grdfill(grids[1], mode="n", outgrid="tmp_outputs/fft_2.nc")

    elif isinstance(grids[0], pd.DataFrame):
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
    ax = sns.lineplot(df["Wavelength (km)"], df.coherency, label=label)
    ax = sns.scatterplot(x=df["Wavelength (km)"], y=df.coherency)

    ax.invert_xaxis()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(2000, 10)
    return ax
    """
    Examples:
    utils.coherency(
    grids = [
        iter_corrections[['x','y','iter_1_initial_top']],
        df_inversion[['x','y','Gobs']]],
        spacing=grav_spacing,
        region=inv_reg,
        label='0'
        )
    utils.coherency(
        grids = [
            iter_corrections[['x','y','iter_1_final_top']],
            df_inversion[['x','y','Gobs']]],
            spacing=grav_spacing,
            region=inv_reg,
            label='1'
            )
    utils.coherency(
        grids = [
            iter_corrections[['x','y','iter_2_final_top']],
            df_inversion[['x','y','Gobs']]],
            spacing=grav_spacing,
            region=inv_reg,
            label='2'
            )
    utils.coherency(
        grids = [
            iter_corrections[['x','y','iter_3_final_top']],
            df_inversion[['x','y','Gobs']]],
            spacing=grav_spacing,
            region=inv_reg,
            label='3'
            )
    """


def square_subplots(n: int):
    """
    From https://github.com/matplotlib/grid-strategy/blob/master/src/grid_strategy/strategies.py # noqa
    Calculate the number of rows and columns based on the total number of items (n) to
    make an arrangement as close to square as looks good.

    Parameters
    ----------
    n : int
        The number of total plots in the subplot

    Returns
    -------
    tuple
        Returns a tuple in the format (number of rows, number of columns), so for example a 3 x 2
        grid would be represented as ``(3, 3)``, because there are 2 rows
        of length 3.
    """
    SPECIAL_CASES = {
        1: (1, 1),
        2: (1, 2),
        3: (2, 2),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (3, 3),
        8: (3, 3),
        9: (3, 3),
    }
    if n in SPECIAL_CASES:
        return SPECIAL_CASES[n]

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
        return tuple(x for i in range(y))

    # If exactly one of these is odd, make it the rows
    if (x % 2) != (y % 2) and (x % 2):
        x, y = y, x

    return x, y


def random_color():
    """
    generate a random color in format R/G/B

    Returns
    -------
    str
        returns a random color string
    """
    color = (
        f"{int(np.random.random() * 256)}/{int(np.random.random() * 256)}"
        f"/{int(np.random.random() * 256)}"
    )
    return color


def get_min_max(
    grid: xr.DataArray,
    shapefile: Union[str or gpd.geodataframe.GeoDataFrame] = None,
):
    """
    Get a grids max and min values, optionally just for the region within a shapefile.

    Parameters
    ----------
    grid : xr.DataArray
        grid to get values for
    shapefile : Union[str or gpd.geodataframe.GeoDataFrame], optional
        path or loaded shapefile to use for a mask, by default None

    Returns
    -------
    tuple
        returns the min and max values.
    """

    if shapefile is None:
        v_min, v_max = np.nanmin(grid), np.nanmax(grid)

    elif shapefile is not None:
        masked = mask_from_shp(shapefile, xr_grid=grid, masked=True, invert=False)
        v_min, v_max = np.nanmin(masked), np.nanmax(masked)

    return (v_min, v_max)


def shapes_to_df(shapes: list):
    """
    convert the output of `regions.draw_region` and `profile.draw_lines` to a dataframe
    of x and y points

    Parameters
    ----------
    shapes : list
        list of vertices

    Returns
    -------
    pd.DataFrame
        Dataframe with x, y, and shape_num.
    """

    df = pd.DataFrame()
    for i, j in enumerate(shapes):
        lon = [coord[0] for coord in j]
        lat = [coord[1] for coord in j]
        shape = pd.DataFrame({"lon": lon, "lat": lat, "shape_num": i})
        df = pd.concat((df, shape))

    df_xy = latlon_to_epsg3031(df)

    return df_xy


def polygon_to_region(polygon: list):
    """
    convert the output of `regions.draw_region` to bounding region in EPSG:3031

    Parameters
    ----------
    polyon : list
        list of polygon vertices

    Returns
    -------
    list
        list in format [e,w,n,s]
    """

    df = shapes_to_df(polygon)

    if df.shape_num.max() > 0:
        print("supplied dataframe has multiple polygons, only using the first one.")
        df = df[df.shape_num == 0]

    region = vd.get_region((df.x, df.y))

    return region


def mask_from_polygon(
    polygon: list,
    invert: bool = False,
    drop_nans: bool = False,
    grid: Union[str, xr.DataArray] = None,
    region: list = None,
    spacing: int = None,
    **kwargs,
):
    """
    convert the output of `regions.draw_region` to a mask or use it to mask a grid

    Parameters
    ----------
    polygon : list
       list of polygon vertices
    invert : bool, optional
        reverse the sense of masking, by default False
    drop_nans : bool, optional
        drop nans after masking, by default False
    grid : Union[str, xr.DataArray], optional
        grid to mask, by default None
    region : list, optional
        region to create a grid if none is supplied, by default None
    spacing : int, optional
        spacing to create a grid if none is supplied, by default None

    Returns
    -------
    xr.Dataarray
        masked grid or mask grid with 1's inside the mask.
    """

    # convert drawn polygon into dataframe
    df = shapes_to_df(polygon)
    data_coords = (df.x, df.y)

    # remove additional polygons
    if df.shape_num.max() > 0:
        print("supplied dataframe has multiple polygons, only using the first one.")
        df = df[df.shape_num == 0]

    # if grid given as filename, load it
    if isinstance(grid, str):
        grid = xr.load_dataarray(grid)
        ds = grid.to_dataset()
    elif isinstance(grid, xr.DataArray):
        ds = grid.to_dataset()

    # if no grid given, make a dummy one with supplied region and spacing
    if grid is None:
        coords = vd.grid_coordinates(
            region=region,
            spacing=spacing,
            pixel_register=kwargs.get("pixel_register", False),
        )
        ds = vd.make_xarray_grid(
            coords, np.ones_like(coords[0]), dims=("y", "x"), data_names="z"
        )

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

    return masked


def change_reg(grid):
    """
    Use GMT grdedit to change the registration type in the metadata.

    Parameters
    ----------
    grid : xr.DataArray
        input grid to change the reg for.

    Returns
    -------
    xr.DataArray
        returns a dataarray with switch reg type.
    """
    with pygmt.clib.Session() as ses:
        # store the input grid in a virtual file so GMT can read it from a dataarray
        with ses.virtualfile_from_grid(grid) as f_in:
            # send the output to a file so that we can read it
            with pygmt.helpers.GMTTempFile(suffix=".nc") as tmpfile:
                args = f"{f_in} -T -G{tmpfile.name}"
                ses.call_module("grdedit", args)
                f_out = pygmt.load_dataarray(tmpfile.name)
    return f_out


def grd_blend(
    grid1: xr.DataArray,
    grid2: xr.DataArray,
    **kwargs,
):
    """
    Use GMT grdblend to blend 2 grids into 1.

    Parameters
    ----------
    grid1 : xr.DataArray
        input grid to change the reg for.

    grid2 : xr.DataArray
        input grid to change the reg for.

    Returns
    -------
    xr.DataArray
        returns a blended dataarray.
    """
    with pygmt.clib.Session() as session:
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
    return pygmt.load_dataarray(infile1)  # if outgrid == tmpfile.name else None


def get_fig_width(figure):
    with pygmt.clib.Session() as session:
        with pygmt.helpers.GMTTempFile() as tmpfile:
            session.call_module("mapproject", f"-Ww ->{tmpfile.name}")
            map_width = tmpfile.read().strip()
    return float(map_width)


def get_fig_height(figure):
    with pygmt.clib.Session() as session:
        with pygmt.helpers.GMTTempFile() as tmpfile:
            session.call_module("mapproject", f"-Wh ->{tmpfile.name}")
            map_height = tmpfile.read().strip()
    return float(map_height)


def GMT_str_to_list(region: list):
    return "".join([str(x) + "/" for x in region])[:-1]


def grd_mask(
    df,
    spacing,
    region,
    clobber="o",
    values="0/0/1",
    radius="0c",
):
    with pygmt.clib.Session() as ses:
        # store the input grid in a virtual file so GMT can read it from a dataarray
        with ses.virtualfile_from_data(x=df.x, y=df.y, z=df.z) as f_in:
            # send the output to a file so that we can read it
            with pygmt.helpers.GMTTempFile(suffix=".nc") as tmpfile:
                args = (
                    f"{f_in} -I{spacing} -R{str(GMT_str_to_list(region))}",
                    f"-C{clobber} -N{values} -S{radius} -G{tmpfile.name}",
                )
                ses.call_module("grdmask", args)
                f_out = pygmt.load_dataarray(tmpfile.name)
    return f_out
