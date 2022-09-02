# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#
import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
from pyproj import Transformer
import pyogrio

from antarctic_plots import fetch

# import seaborn as sns


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

    spacing = pygmt.grdinfo(grid, per_column="n", o=7)[:-1]
    region = [float(pygmt.grdinfo(grid, per_column="n", o=i)[:-1]) for i in range(4)]
    zmin = float(pygmt.grdinfo(grid, per_column="n", o=4)[:-1])
    zmax = float(pygmt.grdinfo(grid, per_column="n", o=5)[:-1])

    if isinstance(grid, str):
        grid = pygmt.load_dataarray(grid)

    reg = grid.gmt.registration

    registration = "g" if reg == 0 else "p"

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
    Convert coordinates from EPSG:3031 Antarctic Polar Stereographic in meters to EPSG:4326 WGS84 in decimal degrees.

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
    # shp = gpd.read_file(shapefile).geometry
    shp = pyogrio.read_dataframe(shapefile)
    
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
    buffer: float = 0,
    print_reg: bool = False,
):
    """
    Change a region string by shifting the box east/west or north/south, zooming in or out, or adding a seperate buffer region.

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
        create a new region which is zoomed out in all direction, in meters, by default 0
    print_reg : bool, optional
        print out the dimensions of the altered region, by default False

    Returns
    -------
    list
        Returns of list of the following variables (region, buffer_region, proj)
    """
    e = starting_region[0] + zoom + w_shift
    w = starting_region[1] - zoom + w_shift
    n = starting_region[2] + zoom - n_shift
    s = starting_region[3] - zoom - n_shift
    region = [e, w, n, s]

    e_buff, w_buff, n_buff, s_buff = (
        int(e - buffer),
        int(w + buffer),
        int(n - buffer),
        int(s + buffer),
    )

    buffer_region = [e_buff, w_buff, n_buff, s_buff]

    fig_height = 80
    fig_width = fig_height * (w - e) / (s - n)

    ratio = (s - n) / (fig_height / 1000)

    proj = f"x1:{ratio}"

    if print_reg is True:
        print(f"inner region is {int((w-e)/1e3)} x {int((s-n)/1e3)} km")
    return region, buffer_region, proj


def set_proj(
    region: Union[str or np.ndarray],
    fig_height: float = 10,
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
    list
        returns a list of the following variables: (proj, proj_latlon, fig_width, fig_height)
    """
    e, w, n, s = region
    fig_width = fig_height * (w - e) / (s - n)

    ratio = (s - n) / (fig_height / 100)
    proj = f"x1:{ratio}"
    proj_latlon = f"s0/-90/-71/1:{ratio}"

    return proj, proj_latlon, fig_width, fig_height


def grd_trend(
    da: xr.DataArray,
    coords: list = ["x", "y", "z"],
    deg: int = 1,
    plot_all: bool = False,
):
    """
    Fit an arbitrary order trend to a grid and use it to detrend.

    Parameters
    ----------
    da : xr.DataArray
        input grid
    coords : list, optional
        coordinate names of grid, by default ['x', 'y', 'z']
    deg : int, optional
        trend order to use, by default 1
    plot_all : bool, optional
        plot the results, by default False

    Returns
    -------
    tuple
        returns xr.DataArrays of the fitted surface, and the detrended grid.
    """

    df = vd.grid_to_table(da).astype("float64")
    trend = vd.Trend(degree=deg).fit((df[coords[0]], df[coords[1]]), df[coords[2]])
    df["fit"] = trend.predict((df[coords[0]], df[coords[1]]))
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

    if plot_all is True:
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
    return fit, detrend


def grd_compare(
    da1: Union[xr.DataArray, str],
    da2: Union[xr.DataArray, str],
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

    Keyword Args
    ------------
    shp_mask : str
        shapefile filename to use to mask the grids for setting the color range.
    robust : bool
        use xarray robust color lims instead of min and max, by default is False.
    Returns
    -------
    xr.DataArray
        the result of da1 - da2
    """
    shp_mask = kwargs.get("shp_mask", None)

    if isinstance(da1, str):
        da1 = xr.load_dataarray(da1)

    if isinstance(da2, str):
        da2 = xr.load_dataarray(da2)

    da1_spacing = get_grid_info(da1)[0]
    da2_spacing = get_grid_info(da2)[0]

    da1_reg = get_grid_info(da1)[1]
    da2_reg = get_grid_info(da2)[1]

    spacing = min(da1_spacing, da2_spacing)

    e = max(da1_reg[0], da2_reg[0])
    w = min(da1_reg[1], da2_reg[1])
    n = max(da1_reg[2], da2_reg[2])
    s = min(da1_reg[3], da2_reg[3])

    region = [e, w, n, s]

    if (da1_spacing != da2_spacing) and (da1_reg != da2_reg):
        print(
            "grid spacings and regions dont match, using smaller spacing",
            f"({spacing}m) and inner region.",
        )
        grid1 = pygmt.grdsample(da1, spacing=spacing, region=region, registration="p")
        grid2 = pygmt.grdsample(da2, spacing=spacing, region=region, registration="p")
    elif da1_spacing != da2_spacing:
        print("grid spacings dont match, using smaller spacing of supplied grids")
        grid1 = pygmt.grdsample(da1, spacing=spacing, registration="p")
        grid2 = pygmt.grdsample(da2, spacing=spacing, registration="p")
    elif da1_reg != da2_reg:
        print("grid regions dont match, using inner region of supplied grids")
        grid1 = pygmt.grdcut(da1, region=region, registration="p")
        grid2 = pygmt.grdcut(da2, region=region, registration="p")
    else:
        print("grid regions and spacing match")
        grid1 = da1
        grid2 = da2

    dif = grid1 - grid2

    vmin = min((np.nanmin(da1), np.nanmin(da2)))
    vmax = max((np.nanmax(da1), np.nanmax(da2)))

    if kwargs.get("robust", False) is True:
        vmin, vmax = None, None
    if shp_mask is not None:
        masked1 = mask_from_shp(shp_mask, xr_grid=grid1, masked=True, invert=False)
        masked2 = mask_from_shp(shp_mask, xr_grid=grid2, masked=True, invert=False)
        vmin = min((np.nanmin(masked1), np.nanmin(masked2)))
        vmax = max((np.nanmax(masked1), np.nanmax(masked2)))

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 20))
    grid1.plot(
        ax=ax[0],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        robust=True,
        cbar_kwargs={"orientation": "horizontal", "anchor": (1, 1.8)},
    )
    grid2.plot(
        ax=ax[1],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        robust=True,
        cbar_kwargs={"orientation": "horizontal", "anchor": (1, 1.8)},
    )
    dif.plot(
        ax=ax[2],
        robust=True,
        cbar_kwargs={"orientation": "horizontal", "anchor": (1, 1.8)},
    )
    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_xlabel("")
        a.set_ylabel("")
        a.set_aspect("equal")

    return dif


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
    data = np.ones_like(coords[0]) + value
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
        names of pd.dataframe columns, xr.dataset variables, xr.dataarray variable, or files to calculate and plot RAPS for.
    plot_type : str, optional
        choose whether to plot with PyGMT or matplotlib, by default 'mpl'
    filter : str
        GMT string to use for pre-filtering data, ex. "c100e3+h" is a 100km low-pass cosine filter, by default is None.
    Keyword Args
    ------------
    region : Union[str, np.ndarray]
        grid region if input is not a grid
    spacing : float
        grid spacing if input is not a grid
    """
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
            ax = sns.lineplot(raps.wavelength, raps.power, label=j, palette="viridis")
            ax = sns.scatterplot(x=raps.wavelength, y=raps.power)
            ax.set_xlabel("Wavelength (km)")
            ax.set_ylabel("Radially Averaged Power ($mGal^{2}km$)")
            pass
        elif plot_type == "pygmt":
            color = f"{random.randrange(255)}/{random.randrange(255)}/{random.randrange(255)}"
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
    region = kwargs.get("region", None)
    spacing = kwargs.get("spacing", None)

    plt.figure()

    if isinstance(grids[0], (str, xr.DataArray)):
        pygmt.grdfill(grids[0], mode="n", outgrid=f"tmp_outputs/fft_1.nc")
        pygmt.grdfill(grids[1], mode="n", outgrid=f"tmp_outputs/fft_2.nc")

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
        pygmt.grdfill(grid1, mode="n", outgrid=f"tmp_outputs/fft_1.nc")
        pygmt.grdfill(grid2, mode="n", outgrid=f"tmp_outputs/fft_2.nc")

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
    # ax = sns.lineplot(df['Wavelength (km)'], df.coherency, label=label)
    # ax = sns.scatterplot(x=df['Wavelength (km)'], y=df.coherency)

    ax.invert_xaxis()
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_xlim(2000, 10)
    # return ax
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
