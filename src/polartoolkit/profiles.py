# Copyright (c) 2024 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#

# pylint: disable=too-many-lines
from __future__ import annotations

import typing

import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr

from polartoolkit import fetch, logger, maps, regions, utils

try:
    from IPython.display import display
except ImportError:
    display = None

try:
    import ipyleaflet
except ImportError:
    ipyleaflet = None
try:
    import pyogrio  # pylint: disable=unused-import

    ENGINE = "pyogrio"
except ImportError:
    pyogrio = None
    ENGINE = "fiona"


def create_profile(
    method: str,
    start: tuple[float, float] | None = None,
    stop: tuple[float, float] | None = None,
    num: int | None = None,
    shapefile: str | None = None,
    polyline: pd.DataFrame | None = None,
    **kwargs: typing.Any,
) -> pd.DataFrame:
    """
    Create a pandas DataFrame of points along a line with multiple methods.

    Parameters
    ----------
    method : str
        Choose sampling method, either "points", "shapefile", or "polyline"
    start : tuple[float, float], optional
        Coordinates for starting point of profile, by default None
    stop : tuple[float, float], optional
        Coordinates for ending point of profile, by default None
    num : int, optional
        Number of points to sample at, for "points" by default is 100, for other methods
        num by default is determined by shapefile or dataframe
    shapefile : str, optional
        shapefile file name to create points along, by default None
    polyline : pandas.DataFrame, optional
        pandas dataframe with columns x and y as vertices of a polyline, by default None

    Returns
    -------
    pandas.DataFrame
        Dataframe with 'easting', 'northing', and 'dist' columns for points along line
        or shapefile path.
    """
    methods = ["points", "shapefile", "polyline"]
    if method not in methods:
        msg = f"If method = {method}, 'start' and 'stop' must be set."
        raise ValueError(msg)
    if method == "points":
        if num is None:
            num = 1000
        if any(a is None for a in [start, stop]):
            msg = f"If method = {method}, 'start' and 'stop' must be set."
            raise ValueError(msg)
        start = typing.cast(tuple[float, float], start)
        stop = typing.cast(tuple[float, float], stop)
        coordinates = pd.DataFrame(
            data=np.linspace(start=start, stop=stop, num=num),
            columns=["easting", "northing"],
        )
        # for points, dist is from first point
        coordinates["dist"] = np.sqrt(
            (coordinates.easting - coordinates.easting.iloc[0]) ** 2
            + (coordinates.northing - coordinates.northing.iloc[0]) ** 2
        )

    elif method == "shapefile":
        if shapefile is None:
            msg = f"If method = {method}, need to provide a valid shapefile"
            raise ValueError(msg)
        shp = gpd.read_file(shapefile, engine=ENGINE)
        df = pd.DataFrame()
        df["coords"] = shp.geometry[0].coords[:]
        coordinates_rel = df.coords.apply(pd.Series, index=["easting", "northing"])
        # for shapefiles, dist is cumulative from previous points
        coordinates = cum_dist(coordinates_rel, **kwargs)

    elif method == "polyline":
        if polyline is None:
            msg = f"If method = {method}, need to provide a valid dataframe"
            raise ValueError(msg)
        # for shapefiles, dist is cumulative from previous points
        coordinates = cum_dist(polyline, **kwargs)

    coords = coordinates.sort_values(by=["dist"])

    if method in ["shapefile", "polyline"]:
        try:
            if num is not None:
                df = coords.set_index("dist")
                dist_resampled = np.linspace(
                    coords.dist.min(),
                    coords.dist.max(),
                    num,
                    dtype=float,
                )
                df1 = (
                    df.reindex(df.index.union(dist_resampled))
                    .interpolate("cubic")  # cubic needs at least 4 points
                    .reset_index()
                )
                df2 = df1[df1.dist.isin(dist_resampled)]
            else:
                df2 = coords
        except ValueError:
            logger.info(
                (
                    "Issue with resampling, possibly due to number of points, ",
                    "you must provide at least 4 points. Returning unsampled points",
                )
            )
            df2 = coords
    else:
        df2 = coords

    return df2[["easting", "northing", "dist"]].reset_index(drop=True)


def sample_grids(
    df: pd.DataFrame,
    grid: str | xr.DataArray,
    sampled_name: str,
    **kwargs: typing.Any,
) -> pd.DataFrame:
    """
    Sample data at every point along a line

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing columns 'easting', 'northing', or columns with names
        defined by kwarg "coord_names".
    grid : str or xarray.DataArray
        Grid to sample, either file name or xarray.DataArray
    sampled_name : str,
        Name for sampled column

    Returns
    -------
    pandas.DataFrame
        Dataframe with new column (sampled_name) of sample values from (grid)
    """

    # drop name column if it already exists
    try:
        df1 = df.drop(columns=sampled_name)
    except KeyError:
        df1 = df.copy()

    if "index" in df1.columns:
        msg = "index column must be removed or renamed before sampling"
        raise ValueError(msg)

    df2 = df1.copy()

    # reset the index
    df3 = df2.reset_index()

    # get x and y column names
    x, y = kwargs.get("coord_names", ("x", "y"))

    # check column names exist, if not, use other common names
    if (x in df3.columns) and (y in df3.columns):
        pass
    elif ("easting" in df3.columns) and ("northing" in df3.columns):
        x, y = ("easting", "northing")

    # get points to sample at
    points = df3[[x, y]].copy()

    # sample the grid at all x,y points
    sampled = pygmt.grdtrack(
        points=points,
        grid=grid,
        newcolname=sampled_name,
        # radius=kwargs.get("radius", None),
        no_skip=True,  # if false causes issues
        verbose=kwargs.get("verbose", "w"),
        interpolation=kwargs.get("interpolation", "c"),
    )

    df3[sampled_name] = sampled[sampled_name]

    # reset index to previous
    df4 = df3.set_index("index")

    # reset index name to be same as originals
    df4.index.name = df1.index.name

    # check that dataframe is identical to original except for new column
    pd.testing.assert_frame_equal(df4.drop(columns=sampled_name), df1)

    return df4


def fill_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN's in sampled layer with values from above layer.

    Parameters
    ----------
    df : pandas.DataFrame
        First 3 columns as they are assumed to by x, y, dist.

    Returns
    -------
    pandas.DataFrame
        Dataframe with NaN's of lower layers filled
    """
    cols = df.columns[3:].values
    for i, j in enumerate(cols):
        if i == 0:
            pass
        else:
            df[j] = df[j].fillna(df[cols[i - 1]])
    return df


def shorten(
    df: pd.DataFrame, max_dist: float | None = None, min_dist: float | None = None
) -> pd.DataFrame:
    """
    Shorten a dataframe at either end based on distance column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to shorten and recalculate distance, must contain 'easting',
        'northing', 'dist'
    max_dist : float, optional
        remove rows with dist>max_dist, by default None
    min_dist : float, optional
        remove rows with dist<min_dist, by default None

    Returns
    -------
    pandas.DataFrame
        Shortened dataframe
    """
    if max_dist is None:
        max_dist = df.dist.max()
    if min_dist is None:
        min_dist = df.dist.min()
    shortened = df[(df.dist < max_dist) & (df.dist > min_dist)].copy()
    shortened["dist"] = np.sqrt(
        (shortened.easting - shortened.easting.iloc[0]) ** 2
        + (shortened.northing - shortened.northing.iloc[0]) ** 2
    )
    return shortened


def make_data_dict(
    names: list[str],
    grids: list[xr.DataArray],
    colors: list[str],
    axes: list[int] | None = None,
) -> dict[typing.Any, typing.Any]:
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
    axes : list[int]
        y axes to use for each data. By default all data are on axis 0. Only 0 and 1 are
        used, if you supply values > 1, they will use the same axis as 1.

    Returns
    -------
    dict[dict]
        Nested dictionaries of grids and attributes
    """

    return {
        f"{i}": {
            "name": j,
            "grid": grids[i],
            "color": colors[i],
            "axis": axes[i] if axes is not None else 0,
        }
        for i, j in enumerate(names)
    }


def default_layers(
    version: str,
    hemisphere: str | None = None,
    reference: str | None = None,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
) -> dict[str, dict[str, str | xr.DataArray]]:
    """
    Fetch default ice surface, ice base, and bed layers.

    Parameters
    ----------
    version : str
        choose between 'bedmap2', 'bedmap3', and 'bedmachine' layers for Antarctica, and
        just 'bedmachine' Greenland
    hemisphere : str, optional
        choose between plotting in the "north" or "south" hemispheres, by default None
    reference : str, optional
        choose between 'ellipsoid', 'eigen-6c4' or 'eigen-gl04c' (only for bedmap),
        for an elevation reference frame, by default None
    region : tuple[float], optional
        region to subset grids by, in format [xmin, xmax, ymin, ymax], by default None
    spacing : float, optional
        grid spacing to resample the grids to, by default None

    Returns
    -------
    dict[str, dict[str, str | xarray.DataArray]]
        Nested dictionary of earth layers and attributes
    """
    hemisphere = utils.default_hemisphere(hemisphere)

    if (spacing is not None) or (reference is not None) or (region is not None):
        logger.warning(
            "Supplying any spacing, reference, or region to `default_layers` will "
            "result in resampling of the grids, which will likely take longer than "
            "just using the full-resolution defaults."
        )

    if version == "bedmap2":
        if hemisphere == "north":
            msg = "Bedmap2 is not available for the northern hemisphere."
            raise ValueError(msg)
        if reference is None:
            reference = "eigen-gl04c"
        surface = fetch.bedmap2(
            "surface",
            fill_nans=True,
            region=region,
            reference=reference,
            spacing=spacing,
        )
        icebase = fetch.bedmap2(
            "icebase",
            fill_nans=True,
            region=region,
            reference=reference,
            spacing=spacing,
        )
        bed = fetch.bedmap2(
            "bed",
            region=region,
            reference=reference,
            spacing=spacing,
        )
    elif version == "bedmap3":
        if hemisphere == "north":
            msg = "Bedmap3 is not available for the northern hemisphere."
            raise ValueError(msg)
        if reference is None:
            reference = "eigen-gl04c"
        surface = fetch.bedmap3(
            "surface",
            fill_nans=True,
            region=region,
            reference=reference,
            spacing=spacing,
        )
        icebase = fetch.bedmap3(
            "icebase",
            fill_nans=True,
            region=region,
            reference=reference,
            spacing=spacing,
        )
        bed = fetch.bedmap3(
            "bed",
            region=region,
            reference=reference,
            spacing=spacing,
        )
    elif version == "bedmachine":
        if reference is None:
            reference = "eigen-6c4"
        surface = fetch.bedmachine(
            "surface",
            hemisphere=hemisphere,
            region=region,
            reference=reference,
            spacing=spacing,
        )
        icebase = fetch.bedmachine(
            "icebase",
            hemisphere=hemisphere,
            region=region,
            reference=reference,
            spacing=spacing,
        )
        bed = fetch.bedmachine(
            "bed",
            hemisphere=hemisphere,
            region=region,
            reference=reference,
            spacing=spacing,
        )
    else:
        msg = "version must be either 'bedmap2', 'bedmap3' or 'bedmachine'"
        raise ValueError(msg)

    layer_names = [
        "ice",
        "water",
        "earth",
    ]
    layer_colors = [
        "lightskyblue",
        "darkblue",
        "lightbrown",
    ]
    layer_grids = [surface, icebase, bed]

    return {
        j: {"name": j, "grid": layer_grids[i], "color": layer_colors[i]}
        for i, j in enumerate(layer_names)
    }


def default_data(
    region: tuple[float, float, float, float] | None = None,
    hemisphere: str | None = None,
    verbose: str = "q",
) -> dict[typing.Any, typing.Any]:
    """
    Fetch default gravity and magnetic datasets.

    Parameters
    ----------
    region : tuple[float, float, float, float], optional
        region to subset grids by, in format [xmin, xmax, ymin, ymax], by default None
    hemisphere : str, optional
        choose between plotting in the "north" or "south" hemispheres, by default None

    Returns
    -------
    dict[typing.Any, typing.Any]
        Nested dictionary of data and attributes
    """
    hemisphere = utils.default_hemisphere(hemisphere)

    if hemisphere == "north":
        msg = "Default data is not yet available for the northern hemisphere."
        raise ValueError(msg)

    mag = fetch.magnetics(
        version="admap1",
        region=region,
        # spacing=10e3,
        verbose=verbose,
    )
    mag = typing.cast(xr.DataArray, mag)

    fa_grav = fetch.gravity(
        version="antgg-2021",
        anomaly_type="FA",
        region=region,
        # spacing=10e3,
        verbose=verbose,
    ).free_air_anomaly
    data_names = [
        "ADMAP-1 magnetics",
        "AntGG-2021 Free-air grav",
    ]
    data_grids = [
        mag,
        fa_grav,
    ]
    data_colors = [
        "red",
        "blue",
    ]

    return make_data_dict(data_names, data_grids, data_colors)


def plot_profile(
    method: str,
    layers_dict: dict[typing.Any, typing.Any] | None = None,
    data_dict: typing.Any | None = None,
    add_map: bool = False,
    layers_version: str | None = None,
    fig_height: float = 9,
    fig_width: float = 14,
    hemisphere: str | None = None,
    **kwargs: typing.Any,
) -> tuple[pygmt.Figure, pd.DataFrame, pd.DataFrame]:
    """
    Show sampled layers and/or data on a cross section, with an optional location map.

    Parameters
    ----------
    method : str
        Choose sampling method, either "points", "shapefile", or "polyline"
    layers_dict : dict, optional
        nested dictionary of layers to include in cross-section, construct with
        `profiles.make_data_dict`, by default is Bedmap2 layers.
    data_dict : dict, optional
        nested dictionary of data to include in option graph, construct with
        `profiles.make_data_dict`, by default is gravity and magnetic anomalies.
    add_map : bool
        Choose whether to add a location map, by default is False.
    layers_version : str, optional
        choose between using 'bedmap2' or 'bedmachine' layers, by default is Bedmap2,
        unless plotting for Greenland, then it is Bedmachine.
    fig_height : float, optional
        Set the height of the figure (excluding the map) in cm, by default is 9.
    fig_width : float, optional
        Set the width of the figure (excluding the map) in cm, by default is 14.
    hemisphere : str, optional
        choose between plotting in the "north" or "south" hemispheres, by default None
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
        'viridis'.
    map_buffer: float
        Change map zoom as relative percentage of profile length (0-1), by default is
        0.3.
    layer_buffer: float
        Change vertical white space within cross-section (0-1), by default is 0.1.
    data_buffer: float
        Change vertical white space within data graph (0-1), by default is 0.1.
    layers_legend_loc: str
        Change the legend location with a GMT position string, by default is
        "JBR+jBL+o0c".
    data_legend_loc: str
        Change the legend location with a GMT position string, by default is
        "JBR+jBL+o0c".
    layers_legend_columns: int
        Change the number of columns in the legend, by default is 1.
    inset : bool
        choose to plot inset map showing figure location, by default is True
    inset_position : str
        position for inset map with PyGMT syntax, by default is "jTL+jTL+o0/0"
    save: bool
        Choose to save the image, by default is False.
    path: str
        Filename for saving image, by default is None.
    """
    try:
        hemisphere = utils.default_hemisphere(hemisphere)
    except KeyError:
        hemisphere = None

    inset = kwargs.get("inset", True)
    subplot_orientation = kwargs.get("subplot_orientation", "horizontal")
    gridlines = kwargs.get("gridlines", True)
    map_points = kwargs.get("map_points")
    coast = kwargs.get("coast", True)

    # create dataframe of points
    points = create_profile(method, **kwargs)

    # if no layers supplied, use default
    if layers_dict is None:
        if hemisphere == "north":
            layers_version = "bedmachine"
        elif hemisphere == "south":
            layers_version = "bedmap3"
        # with redirect_stdout(None), redirect_stderr(None):
        layers_dict = default_layers(
            layers_version,  # type: ignore[arg-type]
            # region=vd.get_region((points.x, points.y)),
            reference=kwargs.get("default_layers_reference"),
            spacing=kwargs.get("default_layers_spacing"),
            hemisphere=hemisphere,
        )

    # create default data dictionary
    if data_dict == "default":
        # with redirect_stdout(None), redirect_stderr(None):
        data_dict = default_data(
            region=vd.get_region((points.easting, points.northing)),
            hemisphere=hemisphere,
        )

    # sample cross-section layers from grids
    df_layers = points.copy()
    for k, v in layers_dict.items():
        df_layers = sample_grids(df_layers, v["grid"], sampled_name=k)

    # fill layers with above layer's values
    if kwargs.get("fillnans", True) is True:
        df_layers = fill_nans(df_layers)

    # sample data grids
    df_data = points.copy()
    if data_dict is not None:
        points = points[["easting", "northing", "dist"]].copy()
        for k, v in data_dict.items():
            df_data = sample_grids(df_data, v["grid"], sampled_name=k)

    # shorten profiles
    if kwargs.get("clip") is True:
        if (kwargs.get("max_dist") or kwargs.get("min_dist")) is None:
            msg = f"If clip = {kwargs.get('clip')}, max_dist and min_dist must be set."
            raise ValueError(msg)
        df_layers = shorten(
            df_layers,
            max_dist=kwargs.get("max_dist"),
            min_dist=kwargs.get("min_dist"),
        )

        if data_dict is not None:
            df_data = shorten(
                df_data,
                max_dist=kwargs.get("max_dist"),
                min_dist=kwargs.get("min_dist"),
            )

    fig = pygmt.Figure()

    # PLOT CROSS SECTION AND DATA
    # get max and min of all the layers
    layers_min = df_layers[df_layers.columns[3:]].min().min()
    layers_max = df_layers[df_layers.columns[3:]].max().max()
    if layers_min == layers_max:
        layers_min -= 0.00001
        layers_max += 0.00001

    if kwargs.get("layers_ylims") is not None:
        layers_min = kwargs["layers_ylims"][0]
        layers_max = kwargs["layers_ylims"][1]
    # add space above and below top and bottom of cross-section
    y_buffer = (layers_max - layers_min) * kwargs.get("layer_buffer", 0.1)
    # set region for x-section
    layers_reg = [
        df_layers.dist.min(),
        df_layers.dist.max(),
        layers_min - y_buffer,
        layers_max + y_buffer,
    ]
    # if there is data to plot as profiles, set region and plot them, if not,
    # make region for x-section fill space
    if data_dict is not None:
        # height of data and layers, plus 0.5cm margin equals total figure height
        data_height = kwargs.get("data_height", 2.5)
        layers_height = fig_height - 0.5 - data_height

        data_projection = f"X{fig_width}c/{data_height}c"
        layers_projection = f"X{fig_width}c/{layers_height}c"

        # get axes from data dict
        axes = pd.Series([v["axis"] for k, v in data_dict.items()])

        # for each axis get overall max and min values
        ax0_min_max = []
        ax1_min_max = []
        for k, v in data_dict.items():
            if v["axis"] == axes.unique()[0]:
                ax0_min_max.append(utils.get_min_max(df_data[k]))
            else:
                ax1_min_max.append(utils.get_min_max(df_data[k]))

        frames = kwargs.get("data_frame")

        if isinstance(frames, (str, type(None))):  # noqa: SIM114
            frames = [frames]
        elif isinstance(frames, list) and isinstance(frames[0], str):
            frames = [frames]

        for i, (k, v) in enumerate(data_dict.items()):
            if v["axis"] == axes.unique()[0]:
                data_min = np.min([a for (a, b) in ax0_min_max])
                data_max = np.max([b for (a, b) in ax0_min_max])
                if data_min == data_max:
                    data_min -= 0.00001
                    data_max += 0.00001
                if frames[0] is None:
                    frame = [
                        "neSW",
                        f"xag+l{kwargs.get('data_x_label',' ')}",
                        f"yag+l{kwargs.get('data_y0_label',' ')}",
                    ]
                else:
                    frame = frames[0]
            else:
                data_min = next(np.min(a) for (a, b) in ax1_min_max)
                data_max = next(np.max(b) for (a, b) in ax1_min_max)
                if data_min == data_max:
                    data_min -= 0.00001
                    data_max += 0.00001
                try:
                    if frames[1] is None:
                        frame = [
                            "nEsw",
                            f"ya+l{kwargs.get('data_y1_label',' ')}",
                        ]
                    else:
                        frame = frames[1]
                except IndexError:
                    frame = [
                        "nEsw",
                        f"ya+l{kwargs.get('data_y1_label',' ')}",
                    ]

            if kwargs.get("data_ylims") is not None:
                data_min = kwargs["data_ylims"][0]
                data_max = kwargs["data_ylims"][1]

            # add space above and below top and bottom of graph
            y_buffer = (data_max - data_min) * kwargs.get("data_buffer", 0.1)

            # set region for data
            data_reg = [
                df_data.dist.min(),
                df_data.dist.max(),
                data_min - y_buffer,
                data_max + y_buffer,
            ]

            # plot data
            if kwargs.get("data_line_cmap") is None:
                # plot data as lines
                data_pen = kwargs.get("data_pen")
                if isinstance(data_pen, list):
                    data_pen = data_pen[i]
                if data_pen is not None:
                    pen = data_pen
                else:
                    thick = kwargs.get("data_pen_thickness", 1)
                    if isinstance(thick, (float, int)):
                        thick = [thick] * len(data_dict.items())

                    color = kwargs.get("data_pen_color")
                    if isinstance(color, list):
                        color = color[i]
                    if color is None:
                        color = v["color"]

                    style = kwargs.get("data_pen_style")
                    if isinstance(style, list):
                        style = style[i]
                    if style is None:
                        style = ""
                pen = f"{thick[i]}p,{color},{style}"

                data_line_style = kwargs.get("data_line_style")
                if isinstance(data_line_style, list):
                    data_line_style = data_line_style[i]

                fig.plot(
                    region=data_reg,
                    projection=data_projection,
                    frame=frame,
                    x=df_data.dist,
                    y=df_data[k],
                    pen=pen,
                    style=data_line_style,
                    label=v["name"],
                )
                # fig.plot(
                #     region=data_reg,
                #     projection=data_projection,
                #     frame=frame,
                #     x=df_data.dist,
                #     y=df_data[k],
                #     pen = f"{kwargs.get('data_pen', [1]*len(data_dict.items()))[i]}p,{v['color']}", # noqa: E501
                #     label = v["name"],
                # )
            else:
                pygmt.makecpt(
                    cmap=kwargs.get("data_line_cmap"),
                    series=[
                        np.min([v["color"] for k, v in data_dict.items()]),
                        np.max([v["color"] for k, v in data_dict.items()]),
                    ],
                )

                fig.plot(
                    region=data_reg,
                    projection=data_projection,
                    frame=frame,
                    x=df_data.dist,
                    y=df_data[k],
                    pen=f"{kwargs.get('data_pen', [1]*len(data_dict.items()))[i]}p,+z",
                    label=v["name"],
                    cmap=True,
                    zvalue=v["color"],
                )
        with pygmt.config(
            FONT_ANNOT_PRIMARY=kwargs.get("data_legend_font", "10p,Helvetica,black"),
        ):
            if kwargs.get("data_legend", True) is True:
                fig.legend(
                    position=kwargs.get("data_legend_loc", "JBR+jBL+o0c"),
                    box=kwargs.get("data_legend_box", False),
                    S=kwargs.get("data_legend_scale", 1),
                )

        if kwargs.get("data_line_cmap") is not None:
            fig.colorbar(
                cmap=True,
                frame=f"a+l{kwargs.get('data_line_cmap_label', ' ')}",
                position=f"JMR+o0.5c/0c+w{data_height*.8}c/{data_height*.16}c",
            )

        # shift origin up by height of the data profile plus 1/2 cm buffer
        fig.shift_origin(yshift=f"{data_height+0.5}c")
        # setup cross-section plot
        fig.basemap(
            region=layers_reg,
            projection=layers_projection,
            frame=kwargs.get("layers_frame", True),
        )
    else:
        # if no data, make xsection fill space
        layers_projection = f"X{fig_width}c/{fig_height}c"
        fig.basemap(
            region=layers_reg,
            projection=layers_projection,
            frame=kwargs.get("layers_frame", True),
        )
        layers_height = fig_height - 0.5

    # plot colored df_layers
    for i, (k, v) in enumerate(layers_dict.items()):
        # fill in layers and draw lines between
        if kwargs.get("fill_layers", True) is True:
            fig.plot(
                x=df_layers.dist,
                y=df_layers[k],
                close="+yb",  # close the polygons,
                fill=v["color"],
                frame=kwargs.get("layers_frame", ["nSew", "a"]),
                transparency=kwargs.get(
                    "layer_transparency", [0] * len(layers_dict.items())
                )[i],
            )
            # plot lines between df_layers
            layers_pen = kwargs.get("layers_pen")
            if isinstance(layers_pen, list):
                layers_pen = layers_pen[i]
            # if pen properties supplied, use then
            if layers_pen is not None:
                pen = layers_pen
            # if not, get pen properties from kwargs
            else:
                thick = kwargs.get("layers_pen_thickness", 1)
                if isinstance(thick, (float, int)):
                    thick = [thick] * len(layers_dict.items())

                color = kwargs.get("layers_pen_color")
                if isinstance(color, list):
                    color = color[i]
                if color is None:
                    color = "black"

                style = kwargs.get("layers_pen_style")
                if isinstance(style, list):
                    style = style[i]
                if style is None:
                    style = ""
                pen = f"{thick[i]}p,{color},{style}"

            layers_line_style = kwargs.get("layers_line_style")
            if isinstance(layers_line_style, list):
                layers_line_style = layers_line_style[i]
            # plot lines between df_layers
            fig.plot(
                x=df_layers.dist,
                y=df_layers[k],
                pen=pen,
                style=layers_line_style,
            )
            # plot transparent lines to get legend
            fig.plot(
                x=df_layers.dist,
                y=df_layers[k],
                pen=f"5p,{v['color']}",
                label=v["name"],
                transparency=100,
            )
        # dont fill layers, just draw lines
        else:
            if kwargs.get("layers_line_cmap") is None:
                # get pen properties
                layers_pen = kwargs.get("layers_pen")
                if isinstance(layers_pen, list):
                    layers_pen = layers_pen[i]
                if layers_pen is not None:
                    pen = layers_pen
                else:
                    thick = kwargs.get("layers_pen_thickness", 1)
                    if isinstance(thick, (float, int)):
                        thick = [thick] * len(layers_dict.items())

                    color = kwargs.get("layers_pen_color")
                    if isinstance(color, list):
                        color = color[i]
                    if color is None:
                        color = v["color"]

                    style = kwargs.get("layers_pen_style")
                    if isinstance(style, list):
                        style = style[i]
                    if style is None:
                        style = ""
                    pen = f"{thick[i]}p,{color},{style}"

                if i == 0:
                    label = f"{v['name']}+N{kwargs.get('layers_legend_columns',1)}"
                else:
                    label = v["name"]

                fig.plot(
                    x=df_layers.dist,
                    y=df_layers[k],
                    # pen = f"{kwargs.get('layer_pen', [1]*len(layers_dict.items()))[i]}p,{v['color']}", # noqa: E501
                    pen=pen,
                    frame=kwargs.get("layers_frame", ["nSew", "a"]),
                    label=label,
                )
            else:
                pygmt.makecpt(
                    cmap=kwargs.get("layers_line_cmap"),
                    series=[
                        np.min([v["color"] for k, v in layers_dict.items()]),
                        np.max([v["color"] for k, v in layers_dict.items()]),
                    ],
                )
                fig.plot(
                    x=df_layers.dist,
                    y=df_layers[k],
                    pen=f"{kwargs.get('layer_pen', [1]*len(layers_dict.items()))[i]}p,+z",  # noqa: E501
                    frame=kwargs.get("layers_frame", ["nSew", "a"]),
                    # label=v["name"],
                    cmap=True,
                    zvalue=v["color"],
                )

    if kwargs.get("layers_line_cmap") is not None:
        fig.colorbar(
            cmap=True,
            frame=f"a+l{kwargs.get('layers_line_cmap_label', ' ')}",
            position=f"JMR+o0.5c/0c+w{layers_height*.8}c/{layers_height*.16}c",
        )

    # add legend of layer names
    with pygmt.config(
        FONT_ANNOT_PRIMARY=kwargs.get("layers_legend_font", "10p,Helvetica,black"),
    ):
        if kwargs.get("layers_legend", True) is True:
            fig.legend(
                position=kwargs.get("layers_legend_loc", "JBR+jBL+o0c"),
                box=kwargs.get("layers_legend_box", False),
                S=kwargs.get("layers_legend_scale", 1),
            )

    # plot 'A','B' locations
    start_end_font = kwargs.get(
        "start_end_font",
        "18p,Helvetica,black",
    )
    start_end_fill = kwargs.get("start_end_fill", "white")
    start_end_pen = kwargs.get("start_end_pen", "1p,black")

    x1 = layers_reg[0]
    x2 = layers_reg[1]

    if kwargs.get("start_end_label_position", "T") == "T":
        y = layers_reg[3]
    elif kwargs.get("start_end_label_position", "B") == "B":
        y = layers_reg[2]
    else:
        msg = "invalid start_end_label_position string"
        raise ValueError(msg)

    fig.text(
        x=x1,
        y=y,
        # position="n0/1",
        # position = kwargs.get("start_label_position", "TL"),
        text=kwargs.get("start_label", "A"),
        font=start_end_font,
        justify=kwargs.get("start_label_justify", "BR"),
        pen=start_end_pen,
        fill=start_end_fill,
        no_clip=True,
        offset=kwargs.get("start_label_offset", "-0.1c/0.1c"),
    )
    fig.text(
        x=x2,
        y=y,
        # position="n1/1",
        # position = kwargs.get("end_label_position", "TR"),
        text=kwargs.get("end_label", "B"),
        font=start_end_font,
        justify=kwargs.get("end_label_justify", "BL"),
        pen=start_end_pen,
        fill=start_end_fill,
        no_clip=True,
        offset=kwargs.get("end_label_offset", "0.1c/0.1c"),
    )

    if add_map is True:
        # Automatic data extent + buffer as % of line length
        buffer = df_layers.dist.max() * kwargs.get("map_buffer", 0.3)
        map_reg = regions.alter_region(
            vd.get_region((df_layers.easting, df_layers.northing)), zoom=-buffer
        )

        # Set figure parameters
        if subplot_orientation == "horizontal":
            # if shifting horizontally, set map height to match graph height
            map_proj, map_proj_ll, map_width, _map_height = utils.set_proj(
                map_reg,
                fig_height=fig_height,
                hemisphere=hemisphere,
            )
            # shift map to the left with 1 cm margin
            if data_dict is not None:
                fig.shift_origin(
                    xshift=f"-{map_width+1}c", yshift=f"-{data_height+.5}c"
                )
            else:
                fig.shift_origin(xshift=f"-{map_width+1}c")
        elif subplot_orientation == "vertical":
            # if shifting vertically, set map width to match graph width
            map_proj, map_proj_ll, map_width, _map_height = utils.set_proj(
                map_reg,
                fig_width=fig_width,
                hemisphere=hemisphere,
            )
            # shift map up with a 1/2 cm margin
            if data_dict is not None:
                fig.shift_origin(yshift=f"{layers_height+.5}c")
            else:
                fig.shift_origin(yshift=f"{fig_height+.5}c")
        else:
            msg = "invalid subplot_orientation string"
            raise ValueError(msg)

        # plot imagery, or supplied grid as background
        # can't use maps.plot_grd because it reset projection
        background = kwargs.get("map_background")
        if background is None:
            if hemisphere == "south":
                background = fetch.imagery()
                map_cmap = True
            elif hemisphere == "north":
                background = fetch.modis(hemisphere=hemisphere)
                pygmt.makecpt(
                    cmap="grayC",
                    series=[15000, 17000, 1],
                    verbose="e",
                )
                map_cmap = True
            else:
                msg = (
                    "if add_map is True and hemisphere is not provided, must provide "
                    "argument `map_background`"
                )
                raise ValueError(msg)
        else:
            map_cmap = kwargs.get("map_cmap", "viridis")
        if kwargs.get("map_grd2cpt", False) is True:
            pygmt.grd2cpt(
                cmap=map_cmap,
                grid=background,
                region=map_reg,
                background=True,
                continuous=True,
                verbose="q",
            )
            cmap = True
        else:
            cmap = map_cmap
        fig.grdimage(
            region=map_reg,
            projection=map_proj,
            grid=background,
            shading=kwargs.get("map_shading", False),
            cmap=cmap,
            verbose="q",
        )

        # plot groundingline and coastlines
        if coast is True:
            coast_version = kwargs.get("coast_version")
            if coast_version is None:
                if hemisphere == "north":
                    coast_version = "BAS"
                elif hemisphere == "south":
                    coast_version = "measures-v2"
                elif hemisphere is None:
                    msg = (
                        "if coast is True and hemisphere is not provided, must provide "
                        "argument `coast_version`"
                    )
                    raise ValueError(msg)

            maps.add_coast(
                fig,
                hemisphere=hemisphere,
                region=map_reg,
                projection=map_proj,
                pen=kwargs.get("coast_pen", "1.2p,black"),
                no_coast=kwargs.get("no_coast", False),
                version=kwargs.get("coast_version"),
            )

        # add lat long grid lines
        if gridlines is True:
            maps.add_gridlines(
                fig,
                map_reg,
                map_proj_ll,
                x_spacing=kwargs.get("x_spacing"),
                y_spacing=kwargs.get("y_spacing"),
            )

        # plot profile location, and endpoints on map
        fig.plot(
            projection=map_proj,
            region=map_reg,
            x=df_layers.easting,
            y=df_layers.northing,
            pen=kwargs.get("map_line_pen", "2p,red"),
        )
        fig.text(
            x=df_layers.loc[df_layers.dist.idxmin()].easting,
            y=df_layers.loc[df_layers.dist.idxmin()].northing,
            text=kwargs.get("start_label", "A"),
            fill="white",
            font="12p,Helvetica,black",
            justify="CM",
            clearance="+tO",
        )
        fig.text(
            x=df_layers.loc[df_layers.dist.idxmax()].easting,
            y=df_layers.loc[df_layers.dist.idxmax()].northing,
            text=kwargs.get("end_label", "B"),
            fill="white",
            font="12p,Helvetica,black",
            justify="CM",
            clearance="+tO",
        )

        # add x,y points to plot
        if map_points is not None:
            fig.plot(
                x=map_points.easting,
                y=map_points.northing,
                style=kwargs.get("map_points_style", "x.15c"),
                pen=kwargs.get("map_points_pen", ".2p,black"),
                fill=kwargs.get("map_points_color", "black"),
            )

        # add inset map
        if inset is True:
            maps.add_inset(
                fig,
                hemisphere=hemisphere,
                region=map_reg,
                inset_position=kwargs.get("inset_position", "jTL+jTL+o0/0"),
                inset_pos=kwargs.get("inset_pos"),
                inset_width=kwargs.get("inset_width", 0.25),
                inset_reg=kwargs.get("inset_reg"),
            )

    if kwargs.get("save") is True:
        if kwargs.get("path") is None:
            msg = f"If save = {kwargs.get('save')}, 'path' must be set."
            raise ValueError(msg)
        fig.savefig(kwargs.get("path"), dpi=300)

    return fig, df_layers, df_data


def plot_data(
    method: str,
    data_dict: typing.Any | None = None,
    add_map: bool = False,
    fig_height: float = 9,
    fig_width: float = 14,
    hemisphere: str | None = None,
    **kwargs: typing.Any,
) -> tuple[pygmt.Figure, pd.DataFrame]:
    """
    Show sampled data on a cross section, with an optional location map.

    Parameters
    ----------
    method : str
        Choose sampling method, either "points", "shapefile", or "polyline"
    data_dict : dict, optional
        nested dictionary of data to include in option graph, construct with
        `profiles.make_data_dict`, by default is gravity and magnetic anomalies.
    add_map : bool
        Choose whether to add a location map, by default is False.
    fig_height : float, optional
        Set the height of the figure (excluding the map) in cm, by default is 9.
    fig_width : float, optional
        Set the width of the figure (excluding the map) in cm, by default is 14.
    hemisphere : str, optional
        choose between plotting in the "north" or "south" hemispheres, by default None
    Keyword Args
    ------------
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
        'viridis'.
    map_buffer: float
        Change map zoom as relative percentage of profile length (0-1), by default is
        0.3.
    data_buffer: float
        Change vertical white space within data graph (0-1), by default is 0.1.
    data_legend_loc: str
        Change the legend location with a GMT position string, by default is
        "JBR+jBL+o0c".
    inset : bool
        choose to plot inset map showing figure location, by default is True
    inset_position : str
        position for inset map with PyGMT syntax, by default is "jTL+jTL+o0/0"
    save: bool
        Choose to save the image, by default is False.
    path: str
        Filename for saving image, by default is None.
    """
    try:
        hemisphere = utils.default_hemisphere(hemisphere)
    except KeyError:
        hemisphere = None

    inset = kwargs.get("inset", True)
    subplot_orientation = kwargs.get("subplot_orientation", "horizontal")
    gridlines = kwargs.get("gridlines", True)
    map_points = kwargs.get("map_points")
    coast = kwargs.get("coast", True)

    # create dataframe of points
    points = create_profile(method, **kwargs)

    # create default data dictionary
    if data_dict == "default":
        # with redirect_stdout(None), redirect_stderr(None):
        data_dict = default_data(
            region=vd.get_region((points.easting, points.northing)),
            hemisphere=hemisphere,
        )

    # sample data grids
    df_data = points.copy()
    if data_dict is not None:
        points = points[["easting", "northing", "dist"]].copy()
        for k, v in data_dict.items():
            df_data = sample_grids(df_data, v["grid"], sampled_name=k)

    # shorten profiles
    if kwargs.get("clip") is True:  # noqa: SIM102
        if (kwargs.get("max_dist") or kwargs.get("min_dist")) is None:
            msg = f"If clip = {kwargs.get('clip')}, max_dist and min_dist must be set."
            raise ValueError(msg)
    df_data = shorten(
        df_data,
        max_dist=kwargs.get("max_dist"),
        min_dist=kwargs.get("min_dist"),
    )

    fig = pygmt.Figure()

    # height of data, plus 0.5cm margin equals total figure height
    data_height = fig_height - 0.5

    data_projection = f"X{fig_width}c/{data_height}c"

    # get axes from data dict
    axes = pd.Series([v["axis"] for k, v in data_dict.items()])  # type: ignore[union-attr]

    # for each axis get overall max and min values
    ax0_min_max = []
    ax1_min_max = []
    for k, v in data_dict.items():  # type: ignore[union-attr]
        if v["axis"] == axes.unique()[0]:
            ax0_min_max.append(utils.get_min_max(df_data[k]))
        else:
            ax1_min_max.append(utils.get_min_max(df_data[k]))

    frames = kwargs.get("data_frame")

    if isinstance(frames, (str, type(None))):  # noqa: SIM114
        frames = [frames]
    elif isinstance(frames, list) and isinstance(frames[0], str):
        frames = [frames]

    for i, (k, v) in enumerate(data_dict.items()):  # type: ignore[union-attr]
        if v["axis"] == axes.unique()[0]:
            data_min = np.min([a for (a, b) in ax0_min_max])
            data_max = np.max([b for (a, b) in ax0_min_max])
            if data_min == data_max:
                data_min -= 0.00001
                data_max += 0.00001
            if frames[0] is None:
                frame = [
                    "neSW",
                    f"xag+l{kwargs.get('data_x_label',' ')}",
                    f"yag+l{kwargs.get('data_y0_label',' ')}",
                ]
            else:
                frame = frames[0]
        else:
            data_min = next(np.min(a) for (a, b) in ax1_min_max)
            data_max = next(np.max(b) for (a, b) in ax1_min_max)
            if data_min == data_max:
                data_min -= 0.00001
                data_max += 0.00001
            try:
                if frames[1] is None:
                    frame = [
                        "nEsw",
                        f"ya+l{kwargs.get('data_y1_label',' ')}",
                    ]
                else:
                    frame = frames[1]
            except IndexError:
                frame = [
                    "nEsw",
                    f"ya+l{kwargs.get('data_y1_label',' ')}",
                ]

        if kwargs.get("data_ylims") is not None:
            data_min = kwargs["data_ylims"][0]
            data_max = kwargs["data_ylims"][1]
        # add space above and below top and bottom of graph
        y_buffer = (data_max - data_min) * kwargs.get("data_buffer", 0.1)

        # set region for data
        data_reg = [
            df_data.dist.min(),
            df_data.dist.max(),
            data_min - y_buffer,
            data_max + y_buffer,
        ]

        # plot data
        if kwargs.get("data_line_cmap") is None:
            # plot data as lines
            data_pen = kwargs.get("data_pen")
            if isinstance(data_pen, list):
                data_pen = data_pen[i]
            if data_pen is not None:
                pen = data_pen
            else:
                thick = kwargs.get("data_pen_thickness", 1)
                if isinstance(thick, (float, int)):
                    thick = [thick] * len(data_dict.items())  # type: ignore[union-attr]

                color = kwargs.get("data_pen_color")
                if isinstance(color, list):
                    color = color[i]
                if color is None:
                    color = v["color"]

                style = kwargs.get("data_pen_style")
                if isinstance(style, list):
                    style = style[i]
                if style is None:
                    style = ""
            pen = f"{thick[i]}p,{color},{style}"

            data_line_style = kwargs.get("data_line_style")
            if isinstance(data_line_style, list):
                data_line_style = data_line_style[i]

            fig.plot(
                region=data_reg,
                projection=data_projection,
                frame=frame,
                x=df_data.dist,
                y=df_data[k],
                pen=pen,
                style=data_line_style,
                label=v["name"],
            )

        else:
            pygmt.makecpt(
                cmap=kwargs.get("data_line_cmap"),
                series=[
                    np.min([v["color"] for k, v in data_dict.items()]),  # type: ignore[union-attr]
                    np.max([v["color"] for k, v in data_dict.items()]),  # type: ignore[union-attr]
                ],
            )

            fig.plot(
                region=data_reg,
                projection=data_projection,
                frame=frame,
                x=df_data.dist,
                y=df_data[k],
                pen=f"{kwargs.get('data_pen', [1]*len(data_dict.items()))[i]}p,+z",  # type: ignore[union-attr]
                label=v["name"],
                cmap=True,
                zvalue=v["color"],
            )
    with pygmt.config(
        FONT_ANNOT_PRIMARY=kwargs.get("data_legend_font", "10p,Helvetica,black"),
    ):
        if kwargs.get("data_legend", True) is True:
            fig.legend(
                position=kwargs.get("data_legend_loc", "JBR+jBL+o0c"),
                box=kwargs.get("data_legend_box", False),
                S=kwargs.get("data_legend_scale", 1),
            )
    if kwargs.get("data_line_cmap") is not None:
        fig.colorbar(
            cmap=True,
            frame=f"a+l{kwargs.get('data_line_cmap_label', ' ')}",
            position=f"JMR+o0.5c/0c+w{data_height*.8}c/{data_height*.16}c",
        )

    # plot A, A'  locations
    fig.text(
        x=data_reg[0],
        y=data_reg[3],
        text=kwargs.get("start_label", "A"),
        font="18p,Helvetica,black",
        justify="BR",
        # fill="white",
        no_clip=True,
        offset="-0.1c/0.1c",
    )
    fig.text(
        x=data_reg[1],
        y=data_reg[3],
        text=kwargs.get("end_label", "B"),
        font="18p,Helvetica,black",
        justify="BL",
        # fill="white",
        no_clip=True,
        offset="0.1c/0.1c",
    )

    if add_map is True:
        # Automatic data extent + buffer as % of line length
        buffer = df_data.dist.max() * kwargs.get("map_buffer", 0.3)
        map_reg = regions.alter_region(
            vd.get_region((df_data.easting, df_data.northing)), zoom=-buffer
        )

        # Set figure parameters
        if subplot_orientation == "horizontal":
            # if shifting horizontally, set map height to match graph height
            map_proj, map_proj_ll, map_width, _map_height = utils.set_proj(
                map_reg,
                fig_height=fig_height,
                hemisphere=hemisphere,
            )
            # shift map to the left with 1 cm margin
            if data_dict is not None:
                fig.shift_origin(
                    xshift=f"-{map_width+1}c", yshift=f"-{data_height+.5}c"
                )
            else:
                fig.shift_origin(xshift=f"-{map_width+1}c")
        elif subplot_orientation == "vertical":
            # if shifting vertically, set map width to match graph width
            map_proj, map_proj_ll, map_width, _map_height = utils.set_proj(
                map_reg,
                fig_width=fig_width,
                hemisphere=hemisphere,
            )
            # shift map up with a 1/2 cm margin
            if data_dict is not None:
                fig.shift_origin(yshift=f"{data_height+.5}c")
            else:
                fig.shift_origin(yshift=f"{fig_height+.5}c")
        else:
            msg = "invalid subplot_orientation string"
            raise ValueError(msg)

        # plot imagery, or supplied grid as background
        # can't use maps.plot_grd because it reset projection
        background = kwargs.get("map_background")
        if background is None:
            if hemisphere == "south":
                background = fetch.imagery()
            elif hemisphere == "north":
                background = fetch.modis(hemisphere=hemisphere)

        if kwargs.get("map_grd2cpt", False) is True:
            pygmt.grd2cpt(
                cmap=kwargs.get("map_cmap", "viridis"),
                grid=background,
                region=map_reg,
                background=True,
                continuous=True,
                verbose="q",
            )
            cmap = True
        else:
            cmap = kwargs.get("map_cmap", "viridis")
        fig.grdimage(
            region=map_reg,
            projection=map_proj,
            grid=background,
            shading=kwargs.get("map_shading", False),
            cmap=cmap,
            verbose="q",
        )

        # plot groundingline and coastlines
        if coast is True:
            maps.add_coast(
                fig,
                hemisphere=hemisphere,
                region=map_reg,
                projection=map_proj,
                pen=kwargs.get("coast_pen", "1.2p,black"),
                no_coast=kwargs.get("no_coast", False),
                version=kwargs.get("coast_version"),
            )

        # add lat long grid lines
        if gridlines is True:
            maps.add_gridlines(
                fig,
                map_reg,
                map_proj_ll,
                x_spacing=kwargs.get("x_spacing"),
                y_spacing=kwargs.get("y_spacing"),
            )

        # plot profile location, and endpoints on map
        fig.plot(
            projection=map_proj,
            region=map_reg,
            x=df_data.easting,
            y=df_data.northing,
            pen=kwargs.get("map_line_pen", "2p,red"),
        )
        fig.text(
            x=df_data.loc[df_data.dist.idxmin()].easting,
            y=df_data.loc[df_data.dist.idxmin()].northing,
            text=kwargs.get("start_label", "A"),
            fill="white",
            font="12p,Helvetica,black",
            justify="CM",
            clearance="+tO",
        )
        fig.text(
            x=df_data.loc[df_data.dist.idxmax()].easting,
            y=df_data.loc[df_data.dist.idxmax()].northing,
            text=kwargs.get("end_label", "B"),
            fill="white",
            font="12p,Helvetica,black",
            justify="CM",
            clearance="+tO",
        )

        # add x,y points to plot
        if map_points is not None:
            fig.plot(
                x=map_points.easting,
                y=map_points.northing,
                style=kwargs.get("map_points_style", "x.15c"),
                pen=kwargs.get("map_points_pen", ".2p,blue"),
                fill=kwargs.get("map_points_color", "blue"),
            )

        # add inset map
        if inset is True:
            maps.add_inset(
                fig,
                hemisphere=hemisphere,
                region=map_reg,
                inset_position=kwargs.get("inset_position", "jTL+jTL+o0/0"),
                inset_pos=kwargs.get("inset_pos"),
                inset_width=kwargs.get("inset_width", 0.25),
                inset_reg=kwargs.get("inset_reg", [-2800e3, 2800e3, -2800e3, 2800e3]),
            )

    if kwargs.get("save") is True:
        if kwargs.get("path") is None:
            msg = f"If save = {kwargs.get('save')}, 'path' must be set."
            raise ValueError(msg)
        fig.savefig(kwargs.get("path"), dpi=300)

    return fig, df_data


def rel_dist(
    df: pd.DataFrame,
    reverse: bool = False,
) -> pd.DataFrame:
    """
    calculate distance between x,y points in a dataframe, relative to the previous row.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing columns x and y in meters.
    reverse : bool, optional,
        choose whether to reverse the profile, by default is False

    Returns
    -------
    pandas.DataFrame
        Returns original dataframe with additional column rel_dist
    """
    df = df.copy()
    if reverse is True:
        df1 = df[::-1].reset_index(drop=True)
    elif reverse is False:
        df1 = df.copy()

    # from https://stackoverflow.com/a/75824992/18686384
    df1["x_lag"] = df1.easting.shift(1)  # pylint: disable=used-before-assignment
    df1["y_lag"] = df1.northing.shift(1)
    df1["rel_dist"] = np.sqrt(
        (df1.easting - df1["x_lag"]) ** 2 + (df1.northing - df1["y_lag"]) ** 2
    )
    df1 = df1.drop(["x_lag", "y_lag"], axis=1)
    return df1.dropna(subset=["rel_dist"])

    # from sklearn.metrics import pairwise_distances
    # from https://stackoverflow.com/a/72754753/18686384
    # df1["rel_dist"] = 0
    # for i in range(1, len(df1)):
    #     if i == 0:
    #         pass
    #     else:
    #         df1.loc[i, 'rel_dist'] = pairwise_distances(
    #             df1.loc[i, ['x', 'y']],
    #             df1.loc[i-1, ['x', 'y']],
    #         )

    # issue, raised pandas future warning
    # df1["rel_dist"] = 0
    # for i in range(1, len(df1)):
    #     if i == 0:
    #         pass
    #     else:
    #         df1.loc[i, "rel_dist"] = np.sqrt(
    #             (df1.loc[i, "x"] - df1.loc[i - 1, "x"]) ** 2
    #             + (df1.loc[i, "y"] - df1.loc[i - 1, "y"]) ** 2
    #         )


def cum_dist(df: pd.DataFrame, **kwargs: typing.Any) -> pd.DataFrame:
    """
    calculate cumulative distance of points along a line.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing columns x, y, and rel_dist

    Returns
    -------
    pandas.DataFrame
        Returns original dataframe with additional column dist
    """
    reverse = kwargs.get("reverse", False)
    df1 = df.copy()
    df1 = rel_dist(df1, reverse=reverse)
    df1["dist"] = df1.rel_dist.cumsum()
    return df1


def draw_lines(**kwargs: typing.Any) -> list[typing.Any]:
    """
    Plot an interactive map, and use the "Draw a Polyline" button to create vertices of
    a line. Vertices will be returned as the output of the function.

    Returns
    -------
    list[typing.Any]
        Returns a list of list of vertices for each polyline in lat long.
    """

    if ipyleaflet is None:
        msg = """
            Missing optional dependency 'ipyleaflet' required for interactive plotting.
        """
        raise ImportError(msg)

    if display is None:
        msg = "Missing optional dependency 'ipython' required for interactive plotting."

        raise ImportError(msg)

    m = maps.interactive_map(**kwargs, show=False)

    def clear_m() -> None:
        global lines  # noqa: PLW0603 # pylint:disable=global-variable-undefined
        lines = []  # type: ignore[name-defined]

    clear_m()

    mydrawcontrol = ipyleaflet.DrawControl(
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

    def handle_line_draw(self: typing.Any, action: str, geo_json: typing.Any) -> None:  # noqa: ARG001 # pylint:disable=unused-argument
        global lines  # noqa: PLW0602 # pylint:disable=global-variable-not-assigned
        shapes = []
        for coords in geo_json["geometry"]["coordinates"]:
            shapes.append(list(coords))
        shapes = list(shapes)
        if action == "created":
            lines.append(shapes)  # type: ignore[name-defined]

    mydrawcontrol.on_draw(handle_line_draw)
    m.add_control(mydrawcontrol)

    clear_m()
    display(m)

    return lines  # type: ignore[name-defined, no-any-return]
