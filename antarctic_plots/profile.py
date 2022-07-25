# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt
import verde as vd

from antarctic_plots import fetch


def create_profile(
    method: str,
    start: np.ndarray = None,
    stop: np.ndarray = None,
    shapefile: str = None,
    num=100,
    shp_num_points: int = None,
    **kwargs,
):
    """
    Create a pandas DataFrame of points along a line with multiple methods.

    Parameters
    ----------
    method : str
        Choose sampling method, either "points" or "shapefile"
    start :np.ndarray, optional
        Coordinates for starting point of profile, by default None
    stop : np.ndarray, optional
        Coordinates for eding point of profile, by default None
    num : int, optional
        Number of points between start and stop, by default 100
    shp_num_points : int, optional
        Number of points to resample shapefile to, by default None
    shapefile : str, optional
        shapefile file name to create points along, by default None

    Returns
    -------
    pd.Dataframe
        Dataframe with 'x', 'y', and 'dist' columns for points along line or shapefile
        path.
    """
    methods = ["points", "shapefile"]
    if method not in methods:
        raise ValueError(f"Invalid method type. Expected one of {methods}")
    if method == "points":
        if any(a is None for a in [start, stop]):
            raise ValueError(f"If method = {method}, 'start' and 'stop' must be set.")
        coordinates = pd.DataFrame(
            data=np.linspace(start=start, stop=stop, num=num), columns=["x", "y"]
        )
    elif method == "shapefile":
        if shapefile is None:
            raise ValueError(f"If method = {method}, need to provide a valid shapefile")
        shp = gpd.read_file(shapefile)
        df = pd.DataFrame()
        df["coords"] = shp.geometry[0].coords[:]
        coordinates = df.coords.apply(pd.Series, index=["x", "y"])
    coordinates["dist"] = np.sqrt(
        (coordinates.x - coordinates.x.iloc[0]) ** 2
        + (coordinates.y - coordinates.y.iloc[0]) ** 2
    )
    coordinates.sort_values(by=["dist"], inplace=True)

    if method == "shapefile":
        if shp_num_points is not None:
            df = coordinates.set_index("dist")
            dist_resampled = np.linspace(
                coordinates.dist.min(),
                coordinates.dist.max(),
                shp_num_points,
                dtype=float,
            )
            df1 = (
                df.reindex(df.index.union(dist_resampled))
                .interpolate("cubic")
                .reset_index()
            )
            df2 = df1[df1.dist.isin(dist_resampled)]
        else:
            df2 = coordinates
    else:
        df2 = coordinates

    df_out = df2[["x", "y", "dist"]].reset_index(drop=True)

    # region=vd.get_region((df_out.x, df_out.y))

    return df_out


def sample_grids(df, grid, name: str = None):
    """
    Sample data at every point along a line

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing columns 'x', 'y'
    grid : str or xr.DataArray
        Grid to sample, either file name or xr.DataArray
    name : str, optional
        Name for sampled column, by default is str(grid)

    Returns
    -------
    pd.DataFrame
        Dataframe with new column (name) of sample values from (grid)
    """
    if name is None:
        name = grid

    df[name] = (pygmt.grdtrack(points=df[["x", "y"]], grid=grid, newcolname=str(name)))[
        name
    ]
    return df


def fill_nans(
    df,
):
    """
    Fill NaN's in sampled layer with values from above layer.

    Parameters
    ----------
    df : pd.DataFrame
        First 3 columns as they are assumed to by x, y, dist.

    Returns
    -------
    pd.DataFrame
        Dataframe with NaN's of lower layers filled
    """
    cols = df.columns[3:].values
    for i, j in enumerate(cols):
        if i == 0:
            pass
        else:
            df[j] = df[j].fillna(df[cols[i - 1]])
    return df


def shorten(df, max_dist=None, min_dist=None, **kwargs):
    """
    Shorten a dataframe at either end based on distance column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to shorten and recalculate distance, must contain 'x', 'y', 'dist'
    max_dist : int, optional
        remove rows with dist>max_dist, by default None
    min_dist : int, optional
        remove rows with dist<min_dist, by default None

    Returns
    -------
    pd.DataFrame
        Shortened dataframe
    """
    if max_dist is None:
        max_dist = df.dist.max()
    if min_dist is None:
        min_dist = df.dist.min()
    shortened = df[(df.dist < max_dist) & (df.dist > min_dist)].copy()
    shortened["dist"] = np.sqrt(
        (shortened.x - shortened.x.iloc[0]) ** 2
        + (shortened.y - shortened.y.iloc[0]) ** 2
    )
    return shortened


def make_data_dict(names: list, grids: list, colors: list) -> dict:
    """
    Create nested dictionary of data and attributes

    Parameters
    ----------
    names : list[str]
        data names
    grids : list[str or xarray.DataArray]
        files or xarray.DataArray's
    colors : list[str]
        colors to plot data

    Returns
    -------
    dict[dict]
        Nested dictionaries of grids and attributes
    """
    data_dict = {
        j: {"name": names[i], "grid": grids[i], "color": colors[i]}
        for i, j in enumerate(names)
    }
    return data_dict


def default_layers() -> dict:
    """
    Fetch default Bedmap2 layers.

    Returns
    -------
    dict[dict]
        Nested dictionary of Bedmap2 layers and attributes
    """
    surface = fetch.bedmap2("surface")
    icebase = fetch.bedmap2("surface") - fetch.bedmap2("thickness")
    bed = fetch.bedmap2("bed")
    layer_names = [
        "surface",
        "icebase",
        "bed",
    ]
    layer_colors = [
        "lightskyblue",
        "darkblue",
        "lightbrown",
    ]
    layer_grids = [surface, icebase, bed]

    layers_dict = {
        j: {"name": layer_names[i], "grid": layer_grids[i], "color": layer_colors[i]}
        for i, j in enumerate(layer_names)
    }
    return layers_dict


def default_data(region=None) -> dict:
    """
    Fetch default gravity and magnetic datasets.

    Parameters
    ----------
    region : str or list[int], optional
       region of Antarctic to load, by default is entire Antarctic region.

    Returns
    -------
    dict[dict]
        Nested dictionary of data and attributes
    """
    if region is None:
        region = (-3330000, 3330000, -3330000, 3330000)
    mag = fetch.magnetics(version="admap1", region=region, spacing=10e3)
    FA_grav = fetch.gravity("FA", region=region, spacing=10e3)
    data_names = [
        "ADMAP-1 magnetics",
        "ANT-4d Free-air grav",
    ]
    data_grids = [
        mag,
        FA_grav,
    ]
    data_colors = [
        "red",
        "blue",
    ]
    data_dict = {
        j: {"name": data_names[i], "grid": data_grids[i], "color": data_colors[i]}
        for i, j in enumerate(data_names)
    }
    return data_dict


def plot_profile(
    method: str,
    layers_dict: dict = None,
    data_dict: dict = None,
    add_map: bool = False,
    **kwargs,
):
    """
    Show sampled layers and/or data on a cross section, with an optional location map.

    Parameters
    ----------
    method : str
        Choose the sample method, either 'points', or 'shapefile'.
    layers_dict : dict, optional
        nested dictionary of layers to include in cross-section, construct with
        `profile.make_data_dict`, by default is Bedmap2 layers.
    data_dict : dict, optional
        nested dictionary of data to include in option graph, construct with
        `profile.make_data_dict`, by default is gravity and magnetic anomalies.
    add_map : bool = False
        Choose whether to add a location map, by default is False.
    Other Parameters
    ----------------
    fillnans: bool
        Choose whether to fill nans in layers, defaults to True
    clip: bool
        Choose whether to clip the profile based on distance
    max_dist: int
        Clip all distances greater than.
    min_dist: int
        Clip all distances less than.
    **kwargs : dict
        map_background: str or xarray.DataArray
            Change the map background by passing a filename string or grid, by default
            is imagery.
        map_cmap: str
            Change the map colorscale by passing a valid GMT cmap string, by default is
            'earth'.
        map_buffer: float (0-1)
            Change map zoom as relative percentage of profile length, by default is 0.2
        layer_buffer: float (0-1)
            Change vertical white space within cross-section, by default is 0.1
        data_buffer: float (0-1)
            Change vertical white space within data graph, by default is 0.1
        save: bool
            Choose to save the image, by default is False
        path: str
            Filename for saving image, by default is None.
    """

    points = create_profile(method, **kwargs)

    data_region = vd.get_region((points.x, points.y))

    if layers_dict is None:
        layers_dict = default_layers()

    if data_dict == "default":
        data_dict = default_data(data_region)

    # sample cross-section layers from grids
    for k, v in layers_dict.items():
        df_layers = sample_grids(points, v["grid"], name=v["name"])

    # fill layers with above layer's values
    if kwargs.get("fillnans", True) is True:
        df_layers = fill_nans(df_layers)

    if data_dict is not None:
        points = points[["x", "y", "dist"]].copy()
        for k, v in data_dict.items():
            df_data = sample_grids(points, v["grid"], name=v["name"])

    # shorten profiles
    if kwargs.get("clip") is True:
        if (kwargs.get("max_dist") or kwargs.get("min_dist")) is None:
            raise ValueError(
                f"If clip = {kwargs.get('clip')}, max_dist and min_dist must be set."
            )
        df_layers = shorten(df_layers, **kwargs)
        if data_dict is not None:
            df_data = shorten(df_data, **kwargs)

    fig = pygmt.Figure()

    if add_map is True:
        # Automatic data extent + buffer as % of line length
        buffer = df_layers.dist.max() * kwargs.get("map_buffer", 0.2)
        e = df_layers.x.min() - buffer
        w = df_layers.x.max() + buffer
        n = df_layers.y.min() - buffer
        s = df_layers.y.max() + buffer

        # Set figure parameters
        fig_height = 90  # in mm
        fig_width = fig_height * (w - e) / (s - n)
        fig_ratio = (s - n) / (fig_height / 1000)
        fig_reg = [e, w, n, s]
        fig_proj = "x1:" + str(fig_ratio)
        fig_proj_ll = "s0/-90/-71/1:" + str(fig_ratio)

        fig.coast(
            region=fig_reg,
            projection=fig_proj_ll,
            land="grey",
            water="grey",
            frame=["nwse", "xf100000", "yf100000", "g0"],
            verbose="q",
        )

        fig.grdimage(
            projection=fig_proj,
            grid=kwargs.get("map_background", fetch.imagery()),
            cmap=kwargs.get("map_cmap", "earth"),
            verbose="q",
        )

        # plot groundingline and coastlines
        fig.plot(
            data=fetch.groundingline(),
            pen="1.2p,black",
        )

        # Plot graticules overtop, at 4d latitude and 30d longitude
        with pygmt.config(
            MAP_ANNOT_OFFSET_PRIMARY="-2p",
            MAP_FRAME_TYPE="inside",
            MAP_ANNOT_OBLIQUE=0,
            FONT_ANNOT_PRIMARY="6p,black,-=2p,white",
            MAP_GRID_PEN_PRIMARY="grey",
            MAP_TICK_LENGTH_PRIMARY="-10p",
            MAP_TICK_PEN_PRIMARY="thinnest,grey",
            FORMAT_GEO_MAP="dddF",
            MAP_POLAR_CAP="90/90",
        ):
            fig.basemap(
                projection=fig_proj_ll,
                region=fig_reg,
                frame=["NSWE", "xa30g15", "ya4g2"],
                verbose="q",
            )
            with pygmt.config(FONT_ANNOT_PRIMARY="6p,black"):
                fig.basemap(
                    projection=fig_proj_ll,
                    region=fig_reg,
                    frame=["NSWE", "xa30", "ya4"],
                    verbose="q",
                )

        # plot profile location, and endpoints on map
        fig.plot(
            projection=fig_proj,
            region=fig_reg,
            x=df_layers.x,
            y=df_layers.y,
            pen="2p,red",
        )
        fig.text(
            x=df_layers.loc[df_layers.dist.idxmin()].x,
            y=df_layers.loc[df_layers.dist.idxmin()].y,
            text="A",
            fill="white",
            font="12p,Helvetica,black",
            justify="CM",
            clearance="+tO",
        )
        fig.text(
            x=df_layers.loc[df_layers.dist.idxmax()].x,
            y=df_layers.loc[df_layers.dist.idxmax()].y,
            text="B",
            fill="white",
            font="12p,Helvetica,black",
            justify="CM",
            clearance="+tO",
        )
        # shift figure to the right to make space for x-section and profiles
        fig.shift_origin(xshift=(fig_width / 10) + 1)

    # PLOT CROSS SECTION AND PROFILES
    # add space above and below top and bottom of cross-section
    min = df_layers[df_layers.columns[3:]].min().min()
    max = df_layers[df_layers.columns[3:]].max().max()
    y_buffer = (max - min) * kwargs.get("layer_buffer", 0.1)
    # set region for x-section
    region_layers = [
        df_layers.dist.min(),
        df_layers.dist.max(),
        min - y_buffer,
        max + y_buffer,
    ]
    # if data for profiles is set, set region and plot them, if not,
    # make region for x-section fill space
    if data_dict is not None:
        try:
            for k, v in data_dict.items():
                # add space above and below top and bottom of cross-section
                min = df_data[k].min()
                max = df_data[k].max()
                y_buffer = (max - min) * kwargs.get("data_buffer", 0.1)
                region_data = [
                    df_data.dist.min(),
                    df_data.dist.max(),
                    min - y_buffer,
                    max + y_buffer,
                ]
                fig.plot(
                    region=region_data,
                    projection="X9c/2.5c",
                    frame=["nSew", "ag"],
                    x=df_data.dist,
                    y=df_data[k],
                    pen=f"2p,{v['color']}",
                    label=v["name"],
                )
            fig.legend(position="JBR+jBL+o0c", box=True)
            fig.shift_origin(yshift="h+.5c")
            fig.basemap(region=region_layers, projection="X9c/6c", frame=True)
        except:
            print("error plotting data profiles")
    else:
        fig.basemap(region=region_layers, projection="X9c/9c", frame=True)

    # plot colored df_layers
    for k, v in layers_dict.items():
        fig.plot(
            x=df_layers.dist,
            y=df_layers[k],
            close="+yb",
            color=v["color"],
            frame=["nSew", "a"],
        )

    # plot lines between df_layers
    for k, v in layers_dict.items():
        fig.plot(x=df_layers.dist, y=df_layers[k], pen="1p,black")

    # plot 'A','B' locations
    fig.text(
        x=region_layers[0],
        y=region_layers[3],
        text="A",
        font="20p,Helvetica,black",
        justify="CM",
        fill="white",
        no_clip=True,
    )
    fig.text(
        x=region_layers[1],
        y=region_layers[3],
        text="B",
        font="20p,Helvetica,black",
        justify="CM",
        fill="white",
        no_clip=True,
    )

    fig.show()

    if kwargs.get("save") is True:
        if kwargs.get("path") is None:
            raise ValueError(f"If save = {kwargs.get('save')}, 'path' must be set.")
        fig.savefig(kwargs.get("path"), dpi=300)
