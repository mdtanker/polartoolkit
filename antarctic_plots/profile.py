# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#

import numpy as np
import pandas as pd
import pygmt
import pyogrio
import verde as vd

from antarctic_plots import fetch, maps, regions, utils

try:
    import ipyleaflet
except ImportError:
    _has_ipyleaflet = False
else:
    _has_ipyleaflet = True

from contextlib import redirect_stderr, redirect_stdout
import os


def create_profile(
    method: str,
    start: np.ndarray = None,
    stop: np.ndarray = None,
    num: int = None,
    shapefile: str = None,
    polyline: pd.DataFrame = None,
    **kwargs,
):
    """
    Create a pandas DataFrame of points along a line with multiple methods.

    Parameters
    ----------
    method : str
        Choose sampling method, either "points", "shapefile", or "polyline"
    start :np.ndarray, optional
        Coordinates for starting point of profile, by default None
    stop : np.ndarray, optional
        Coordinates for eding point of profile, by default None
    num : int, optional
        Number of points to sample at, for "points" by default is 100, for other methods
        num by default is determined by shapefile or dataframe
    shapefile : str, optional
        shapefile file name to create points along, by default None
    polyline : pd.DataFrame, optional
        pandas dataframe with columns x and y as vertices of a polyline, by default None

    Returns
    -------
    pd.Dataframe
        Dataframe with 'x', 'y', and 'dist' columns for points along line or shapefile
        path.
    """
    methods = ["points", "shapefile", "polyline"]
    if method not in methods:
        raise ValueError(f"Invalid method type. Expected one of {methods}")
    if method == "points":
        if num is None:
            num = 100
        if any(a is None for a in [start, stop]):
            raise ValueError(f"If method = {method}, 'start' and 'stop' must be set.")
        coordinates = pd.DataFrame(
            data=np.linspace(start=start, stop=stop, num=num), columns=["x", "y"]
        )
        # for points, dist is from first point
        coordinates["dist"] = np.sqrt(
            (coordinates.x - coordinates.x.iloc[0]) ** 2
            + (coordinates.y - coordinates.y.iloc[0]) ** 2
        )

    elif method == "shapefile":
        if shapefile is None:
            raise ValueError(f"If method = {method}, need to provide a valid shapefile")
        shp = pyogrio.read_dataframe(shapefile)
        df = pd.DataFrame()
        df["coords"] = shp.geometry[0].coords[:]
        coordinates_rel = df.coords.apply(pd.Series, index=["x", "y"])
        # for shapefiles, dist is cumulative from previous points
        coordinates = cum_dist(coordinates_rel, **kwargs)

    elif method == "polyline":
        if polyline is None:
            raise ValueError(f"If method = {method}, need to provide a valid dataframe")
        # for shapefiles, dist is cumulative from previous points
        coordinates = cum_dist(polyline, **kwargs)

    coordinates.sort_values(by=["dist"], inplace=True)

    if method in ["shapefile", "polyline"]:
        try:
            if num is not None:
                df = coordinates.set_index("dist")
                # print(df.describe())
                dist_resampled = np.linspace(
                    coordinates.dist.min(),
                    coordinates.dist.max(),
                    num,
                    dtype=float,
                )
                # print(dist_resampled)
                df1 = (
                    df.reindex(df.index.union(dist_resampled))
                    .interpolate("cubic")  # cubic needs at least 4 points
                    .reset_index()
                )
                # print(df1.describe())
                df2 = df1[df1.dist.isin(dist_resampled)]
                # print(df2.describe())
            else:
                df2 = coordinates
        except ValueError:
            print(
                "Issue with resampling, possibly due to number of points, you must provide at least 4 points. Returning unsampled points"  # noqa
            )
            df2 = coordinates
    else:
        df2 = coordinates

    df_out = df2[["x", "y", "dist"]].reset_index(drop=True)

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


def fill_nans(df):
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


def default_layers(version, region=None) -> dict:
    """
    Fetch default Bedmachine layers.

    Parameters
    ----------
    version : str
        choose between 'bedmap2' and 'bedmachine' layers

    Returns
    -------
    dict[dict]
        Nested dictionary of Bedmachine layers and attributes
    """
    if region is None:
        region = regions.antarctica

    if version == "bedmap2":
        surface = fetch.bedmap2("surface", region=region, fill_nans=True)
        icebase = fetch.bedmap2("icebase", region=region, fill_nans=True)
        bed = fetch.bedmap2("bed", region=region)

    elif version == "bedmachine":
        surface = fetch.bedmachine("surface", region=region)
        icebase = fetch.bedmachine("icebase", region=region)
        bed = fetch.bedmachine("bed", region=region)

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
        region = regions.antarctica

    mag = fetch.magnetics(
        version="admap1",
        region=region,
        spacing=10e3,
    )
    FA_grav = fetch.gravity(
        version="antgg-update",
        anomaly_type="FA",
        region=region,
        spacing=10e3,
    )
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
    layers_version="bedmap2",
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
    layers_version : str, optional
        choose between using 'bedmap2' or 'bedmachine' layers, by default is 'bedmap2'

    Keyword Args
    ------------
    fillnans: bool
        Choose whether to fill nans in layers, defaults to True.
    clip: bool
        Choose whether to clip the profile based on distance.
    num: int
        Number of points to sample at along a line.
    max_dist: int
        Clip all distances greater than.
    min_dist: int
        Clip all distances less than.
    map_background: str or xarray.DataArray
        Change the map background by passing a filename string or grid, by default is
        imagery.
    map_cmap: str
        Change the map colorscale by passing a valid GMT cmap string, by default is
        'earth'.
    map_buffer: float (0-1)
        Change map zoom as relative percentage of profile length, by default is 0.3.
    layer_buffer: float (0-1)
        Change vertical white space within cross-section, by default is 0.1.
    data_buffer: float (0-1)
        Change vertical white space within data graph, by default is 0.1.
    legend_loc: str
        Change the legend location with a GMT position string, by default is
        "JBR+jBL+o0c" which puts the Bottom Left corner of the legend in the Bottom
        Right corner of the plot, with 0 offset.
    inset : bool
        choose to plot inset map showing figure location, by default is True
    inset_pos : str
        position for inset map; either 'TL', 'TR', BL', 'BR', by default is 'TL'
    save: bool
        Choose to save the image, by default is False.
    path: str
        Filename for saving image, by default is None.
    """
    inset = kwargs.get("inset", True)
    inset_pos = kwargs.get("inset_pos", "TL")
    points = create_profile(method, **kwargs)
    data_region = vd.get_region((points.x, points.y))
    subplot_orientation = kwargs.get("subplot_orientation", "horizontal")
    fig_height = kwargs.get('fig_height', 9)
    grid_lines = kwargs.get("grid_lines", True)
    map_points = kwargs.get("map_points", None)
    coast = kwargs.get("coast", True)

    if layers_dict is None:
        with redirect_stdout(None), redirect_stderr(None):
            layers_dict = default_layers(layers_version, region=data_region)

    if data_dict == "default":
        with redirect_stdout(None), redirect_stderr(None):
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

    # PLOT CROSS SECTION AND PROFILES
    # get max and min of all the layers
    layers_min = df_layers[df_layers.columns[3:]].min().min()
    layers_max = df_layers[df_layers.columns[3:]].max().max()
    # add space above and below top and bottom of cross-section
    y_buffer = (layers_max - layers_min) * kwargs.get("layer_buffer", 0.1)
    # set region for x-section
    fig_reg = [
        df_layers.dist.min(),
        df_layers.dist.max(),
        layers_min - y_buffer,
        layers_max + y_buffer,
    ]
    # if data for profiles is set, set region and plot them, if not,
    # make region for x-section fill space
    if data_dict is not None:
        # if using a shared y-axis for data, get overall max and min values
        if kwargs.get('share_yaxis', False) is True:
            data_min = df_data[df_data.columns[3:]].min().min()
            data_max = df_data[df_data.columns[3:]].max().max()
            # add space above and below top and bottom of graph
            y_buffer = (data_max - data_min) * kwargs.get("data_buffer", 0.1)
            # set region for data
            region_data = [
                df_data.dist.min(),
                df_data.dist.max(),
                data_min - y_buffer,
                data_max + y_buffer,
            ]
            # set frame
            frame=["neSW", "ag",]
        
            data_projection = f"X{fig_height}c/2.5c"
            layers_projection = f"X{fig_height}c/6c"

        try:
            for k, v in data_dict.items():
                if kwargs.get('share_yaxis', False) is False:
                    # if using individual y-axes for data, get individual max/mins
                    data_min = df_data[k].min()
                    data_max = df_data[k].max()
                    # add space above and below top and bottom of graph
                    y_buffer = (data_max - data_min) * kwargs.get("data_buffer", 0.1)
                    region_data = [
                        df_data.dist.min(),
                        df_data.dist.max(),
                        data_min - y_buffer,
                        data_max + y_buffer,
                    ]
                    # turn off frame tick labels
                    frame=["neSw", "xag",]

                fig.plot(
                    region=region_data,
                    projection=data_projection,
                    frame=frame,
                    x=df_data.dist,
                    y=df_data[k],
                    pen=f"2p,{v['color']}",
                    label=v["name"],
                )
            fig.legend(position=kwargs.get("legend_loc", "JBR+jBL+o0c"), box=True)
            fig.shift_origin(yshift="3c")
            fig.basemap(region=fig_reg, projection=layers_projection, frame=True)
        except Exception:
            print("error plotting data profiles")
    else:
        # if no data, make xsection fill space
        fig.basemap(region=fig_reg, projection=f"X{fig_height}c/{fig_height}c", frame=True)

    # plot colored df_layers
    for i, (k, v) in enumerate(layers_dict.items()):
        fig.plot(
            x=df_layers.dist,
            y=df_layers[k],
            # close the polygons, 
            close="+yb",
            color=v["color"],
            frame=["nSew", "a"],
        )

    # plot lines between df_layers
    for k, v in layers_dict.items():
        fig.plot(x=df_layers.dist, y=df_layers[k], pen="1p,black")

    # plot 'A','B' locations
    fig.text(
        x=fig_reg[0],
        y=fig_reg[3],
        text="A",
        font="20p,Helvetica,black",
        justify="CM",
        fill="white",
        no_clip=True,
    )
    fig.text(
        x=fig_reg[1],
        y=fig_reg[3],
        text="B",
        font="20p,Helvetica,black",
        justify="CM",
        fill="white",
        no_clip=True,
    )

    if add_map is True:

        # Automatic data extent + buffer as % of line length
        df_reg = vd.get_region((df_layers.x, df_layers.y))
        buffer = df_layers.dist.max() * kwargs.get("map_buffer", 0.3)
        fig_reg = utils.alter_region(df_reg, buffer=buffer)[1]

        # Set figure parameters
        if subplot_orientation == "horizontal":
            # if shifting horizontally, set map height to match graph height
            fig_proj, fig_proj_ll, fig_width, fig_height = utils.set_proj(
                fig_reg, fig_height=fig_height)
            # shift map to the right with a 1 cm margin
            fig.shift_origin(xshift=(-fig_width) - 1, yshift="-3c")
        elif subplot_orientation == "vertical":
           # if shifting vertically, set map width to match graph width
            fig_proj, fig_proj_ll, fig_width, fig_height = utils.set_proj(
                fig_reg, fig_width=fig_height)
            # shift map down with a 1 cm margin
            fig.shift_origin(yshift=(fig_height) + 3)
        else:
            raise ValueError("invalid subplot_orientation string")

        # plot imagery, or supplied grid as background
        fig.grdimage(
            region=fig_reg,
            projection=fig_proj,
            grid=kwargs.get("map_background", fetch.imagery()),
            cmap=kwargs.get("map_cmap", "earth"),
            verbose="q",
        )

        # plot groundingline and coastlines
        if coast is True:
            maps.add_coast(
                fig,
                fig_reg,
                fig_proj,
                pen=kwargs.get("coast_pen", "1.2p,black"),
                no_coast=kwargs.get("no_coast", False),
            )

        # add lat long grid lines
        if grid_lines is True:
            maps.add_gridlines(
                fig,
                fig_reg,
                fig_proj_ll,
                x_annots=kwargs.get("x_annots", 30),
                y_annots=kwargs.get("y_annots", 4),
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

        # add x,y points to plot
        if map_points is not None:
            fig.plot(
                x=map_points.x,
                y=map_points.y,
                style=kwargs.get("map_points_style", "x.15c"),
                pen=kwargs.get("map_points_pen", ".2p,blue"),
                color=kwargs.get("map_points_color", "blue"),
            )

        # add inset map
        if inset is True:
            maps.add_inset(
                fig,
                fig_reg,
                fig_width,
                inset_pos = kwargs.get("inset_pos", "TL"),
                inset_width = kwargs.get("inset_width", 0.25),
                inset_reg = kwargs.get("inset_reg", [-2800e3, 2800e3, -2800e3, 2800e3]),
                )

    fig.show()

    if kwargs.get("save") is True:
        if kwargs.get("path") is None:
            raise ValueError(f"If save = {kwargs.get('save')}, 'path' must be set.")
        fig.savefig(kwargs.get("path"), dpi=300)

def plot_data(
    method: str,
    data_dict: dict,
    **kwargs,
):
    """
    Sample and plot data along a path.

    Parameters
    ----------
    method : str
        Choose the sample method, either 'points', or 'shapefile'.
    data_dict : dict
        nested dictionary of data to include in option graph, construct with
        `profile.make_data_dict`.

    Keyword Args
    ------------
    fillnans: bool
        Choose whether to fill nans in layers, defaults to True.
    clip: bool
        Choose whether to clip the profile based on distance.
    num: int
        Number of points to sample at along a line.
    max_dist: int
        Clip all distances greater than.
    min_dist: int
        Clip all distances less than.
    data_buffer: float (0-1)
        Change vertical white space within data graph, by default is 0.1.
    legend_loc: str
        Change the legend location with a GMT position string, by default is
        "JBR+jBL+o0c" which puts the Bottom Left corner of the legend in the Bottom
        Right corner of the plot, with 0 offset.
    save: bool
        Choose to save the image, by default is False.
    path: str
        Filename for saving image, by default is None.
    """
    points = create_profile(method, **kwargs)
    data_region = vd.get_region((points.x, points.y))
    fig_height = kwargs.get('fig_height', 5)
    fig_width = kwargs.get('fig_width', 10)
    pen_width = kwargs.get('pen_width', "1.5p")

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

    # if using a shared y-axis for data, get overall max and min values
    if kwargs.get('share_yaxis', False) is True:
        data_min = df_data[df_data.columns[3:]].min().min()
        data_max = df_data[df_data.columns[3:]].max().max()
        # add space above and below top and bottom of graph
        y_buffer = (data_max - data_min) * kwargs.get("data_buffer", 0.1)
        # set region for data
        region_data = [
            df_data.dist.min(),
            df_data.dist.max(),
            data_min - y_buffer,
            data_max + y_buffer,
        ]
        # set frame
        frame=["neSW", "ag",]
    
    data_projection = f"X{fig_width}c/{fig_height}c"

    for k, v in data_dict.items():
        if kwargs.get('share_yaxis', False) is False:
            # if using individual y-axes for data, get individual max/mins
            data_min = df_data[k].min()
            data_max = df_data[k].max()
            # add space above and below top and bottom of graph
            y_buffer = (data_max - data_min) * kwargs.get("data_buffer", 0.1)
            region_data = [
                df_data.dist.min(),
                df_data.dist.max(),
                data_min - y_buffer,
                data_max + y_buffer,
            ]
            # turn off frame tick labels
            frame=["neSw", "xag",]

        fig.plot(
            region=region_data,
            projection=data_projection,
            frame=frame,
            x=df_data.dist,
            y=df_data[k],
            pen=f"{pen_width},{v['color']}",
            label=v["name"],
        )
    fig.legend(position=kwargs.get("legend_loc", "JBR+jBL+o0c"), box=True)

    fig.show()

    if kwargs.get("save") is True:
        if kwargs.get("path") is None:
            raise ValueError(f"If save = {kwargs.get('save')}, 'path' must be set.")
        fig.savefig(kwargs.get("path"), dpi=300)


def rel_dist(
    df: pd.DataFrame,
    reverse: bool = False,
):
    """
    calculate distance between x,y points in a dataframe, relative to the previous row.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing columns x and y in meters.
    reverse : bool, optional,
        choose whether to reverse the profile, by default is False

    Returns
    -------
    pd.DataFrame
        Returns original dataframe with additional column rel_dist
    """
    if reverse is True:
        df1 = df[::-1].reset_index(drop=True)
    elif reverse is False:
        df1 = df.copy()

    df1["rel_dist"] = 0

    for i in range(1, len(df1)):
        if i == 0:
            pass
        else:
            df1.loc[i, "rel_dist"] = np.sqrt(
                (df1.loc[i, "x"] - df1.loc[i - 1, "x"]) ** 2
                + (df1.loc[i, "y"] - df1.loc[i - 1, "y"]) ** 2
            )
    return df1


def cum_dist(df: pd.DataFrame, **kwargs):
    """
    calculate cumulative distance of points along a line.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing columns x, y, and rel_dist

    Returns
    -------
    pd.DataFrame
        Returns orignal dataframe with additional column dist
    """
    reverse = kwargs.get("reverse", False)
    df = rel_dist(df, reverse=reverse)
    df["dist"] = df.rel_dist.cumsum()
    return df


def draw_lines(**kwargs):
    """
    Plot an interactive map, and use the "Draw a Polyline" button to create vertices of
    a line. Verticles will be returned as the output of the function.

    Returns
    -------
    tuple
        Returns a tuple of list of vertices for each polyline in lat long.
    """

    m = maps.interactive_map(**kwargs, show=False)

    def clear_m():
        global lines
        lines = list()

    clear_m()

    myDrawControl = ipyleaflet.DrawControl(
        polyline={
            "shapeOptions": {
                "fillColor": "#fca45d",
                "color": "#fca45d",
                "fillOpacity": 1.0,
            }
        },
        rectangle={},
        circlemarker={},
        polygon={},
    )

    def handle_line_draw(self, action, geo_json):
        global lines
        shapes = []
        for coords in geo_json["geometry"]["coordinates"]:
            shapes.append(list(coords))
        shapes = list(shapes)
        if action == "created":
            lines.append(shapes)

    myDrawControl.on_draw(handle_line_draw)
    m.add_control(myDrawControl)

    clear_m()
    display(m)  # noqa

    return lines
