# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#
from __future__ import annotations


import numpy as np
import pandas as pd
import pygmt
import pyogrio
import verde as vd

if TYPE_CHECKING:
    import xarray as xr

from antarctic_plots import fetch, maps, utils

try:
    import ipyleaflet
except ImportError:


def create_profile(
    method: str,
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
    if method == "points":
        if num is None:
            num = 1000
        if any(a is None for a in [start, stop]):
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
        shp = pyogrio.read_dataframe(shapefile)
        df = pd.DataFrame()
        df["coords"] = shp.geometry[0].coords[:]
        coordinates_rel = df.coords.apply(pd.Series, index=["x", "y"])
        # for shapefiles, dist is cumulative from previous points
        coordinates = cum_dist(coordinates_rel, **kwargs)

    elif method == "polyline":
        if polyline is None:
        # for shapefiles, dist is cumulative from previous points
        coordinates = cum_dist(polyline, **kwargs)


    if method in ["shapefile", "polyline"]:
        try:
            if num is not None:
                dist_resampled = np.linspace(
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
        except ValueError:
            )
    else:



def sample_grids(
    df: pd.DataFrame,
    """
    Sample data at every point along a line

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing columns 'x', 'y', or columns with names defined by kwarg
        "coor_names".
    grid : str or xr.DataArray
        Grid to sample, either file name or xr.DataArray
        Name for sampled column

    Returns
    -------
    pd.DataFrame
    """

    # drop name column if it already exists
    try:
    except KeyError:

    # reset the index

    x, y = kwargs.get("coord_names", ("x", "y"))
    # get points to sample at

    # sample the grid at all x,y points
    sampled = pygmt.grdtrack(
        points=points,
        grid=grid,
        verbose=kwargs.get("verbose", "w"),
        interpolation=kwargs.get("interpolation", "c"),
    )


    # reset index to previous

    # reset index name to be same as originals




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


def make_data_dict(
    names: list,
    grids: list,
    colors: list,
) -> dict:
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
        y axes to use for each data. By default all data are on axis 0.
    Returns
    -------
    dict[dict]
        Nested dictionaries of grids and attributes
    """

        f"{i}": {
            "grid": grids[i],
            "color": colors[i],
            "axis": axes[i] if axes is not None else 0,
        }
        for i, j in enumerate(names)
    }


def default_layers(
    """
    Fetch default ice surface, ice base, and bed layers.

    Parameters
    ----------
    version : str
        choose between 'bedmap2' and 'bedmachine' layers

    Returns
    -------
        Nested dictionary of earth layers and attributes
    """

    if version == "bedmap2":
        if reference is None:
            reference = "eigen-gl04c"

    elif version == "bedmachine":
        if reference is None:
            reference = "eigen-6c4"

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

        for i, j in enumerate(layer_names)
    }


    """
    Fetch default gravity and magnetic datasets.

    Parameters
    ----------

    Returns
    -------
    dict[dict]
        Nested dictionary of data and attributes
    """
    mag = fetch.magnetics(
        version="admap1",
        # spacing=10e3,
    )
        version="antgg-update",
        anomaly_type="FA",
        # spacing=10e3,
    )
    data_names = [
        "ADMAP-1 magnetics",
        "ANT-4d Free-air grav",
    ]
    data_grids = [
        mag,
    ]
    data_colors = [
        "red",
        "blue",
    ]



def plot_profile(
    method: str,
    add_map: bool = False,
    layers_version="bedmap2",
    fig_height: float = 9,
    fig_width: float = 14,
    **kwargs,
):
    """
    Show sampled layers and/or data on a cross section, with an optional location map.

    Parameters
    ----------
    method : str
        Choose sampling method, either "points", "shapefile", or "polyline"
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
    fig_height : float, optional
        Set the height of the figure (excluding the map) in cm, by default is 9.
    fig_width : float, optional
        Set the width of the figure (excluding the map) in cm, by default is 14.
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
    subplot_orientation = kwargs.get("subplot_orientation", "horizontal")
    gridlines = kwargs.get("gridlines", True)
    map_points = kwargs.get("map_points", None)
    coast = kwargs.get("coast", True)

    # create dataframe of points
    points = create_profile(method, **kwargs)

    # if no layers supplied, use default
    if layers_dict is None:
        with redirect_stdout(None), redirect_stderr(None):
            layers_dict = default_layers(
                layers_version,
                region=vd.get_region((points.x, points.y)),
                reference=kwargs.get("default_layers_reference", None),
            )

    # create default data dictionary
    if data_dict == "default":
        with redirect_stdout(None), redirect_stderr(None):
            data_dict = default_data(region=vd.get_region((points.x, points.y)))

    # sample cross-section layers from grids
    df_layers = points.copy()
    for k, v in layers_dict.items():

    # fill layers with above layer's values
    if kwargs.get("fillnans", True) is True:
        df_layers = fill_nans(df_layers)

    # sample data grids
    df_data = points.copy()
    if data_dict is not None:
        points = points[["x", "y", "dist"]].copy()
        for k, v in data_dict.items():

    # shorten profiles
    if kwargs.get("clip") is True:
        if (kwargs.get("max_dist") or kwargs.get("min_dist")) is None:
        df_layers = shorten(df_layers, **kwargs)
        if data_dict is not None:
            df_data = shorten(df_data, **kwargs)

    fig = pygmt.Figure()

    # PLOT CROSS SECTION AND DATA
    # get max and min of all the layers
    layers_min = df_layers[df_layers.columns[3:]].min().min()
    layers_max = df_layers[df_layers.columns[3:]].max().max()
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

        frames = kwargs.get("data_frame", None)

            frames = [frames]

        for i, (k, v) in enumerate(data_dict.items()):
            if v["axis"] == axes.unique()[0]:
                data_min = np.min([a for (a, b) in ax0_min_max])
                data_max = np.max([b for (a, b) in ax0_min_max])

                if frames[0] is None:
                    frame = [
                        "neSW",
                        f"xag+l{kwargs.get('data_x_label',' ')}",
                        f"yag+l{kwargs.get('data_y0_label',' ')}",
                    ]
                else:
                    frame = frames[0]
            else:
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
            if kwargs.get("data_line_cmap", None) is None:
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

                    color = kwargs.get("data_pen_color", None)
                    if isinstance(color, list):
                        color = color[i]
                    if color is None:
                        color = v["color"]

                    style = kwargs.get("data_pen_style", None)
                    if isinstance(style, list):
                        style = style[i]
                    if style is None:
                        style = ""
                pen = f"{thick[i]}p,{color},{style}"

                data_line_style = kwargs.get("data_line_style", None)
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

        if kwargs.get("data_line_cmap", None) is not None:
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
        if kwargs.get("fill_layers", True) is True:
            fig.plot(
                x=df_layers.dist,
                y=df_layers[k],
                # close the polygons,
                close="+yb",
                fill=v["color"],
                frame=kwargs.get("layers_frame", ["nSew", "a"]),
                transparency=kwargs.get(
                    "layer_transparency", [0] * len(layers_dict.items())
                )[i],
                label=v["name"],
            )
            # plot lines between df_layers
            layers_pen = kwargs.get("layers_pen")
            if isinstance(layers_pen, list):
                layers_pen = layers_pen[i]
            if layers_pen is not None:
                pen = layers_pen
            else:
                thick = kwargs.get("layers_pen_thickness", 1)
                if isinstance(thick, (float, int)):
                    thick = [thick] * len(layers_dict.items())

                color = kwargs.get("layers_pen_color", None)
                if isinstance(color, list):
                    color = color[i]
                if color is None:
                    color = v["color"]

                style = kwargs.get("layers_pen_style", None)
                if isinstance(style, list):
                    style = style[i]
                if style is None:
                    style = ""
            pen = f"{thick[i]}p,{color},{style}"

            layers_line_style = kwargs.get("layers_line_style", None)
            if isinstance(layers_line_style, list):
                layers_line_style = layers_line_style[i]

            fig.plot(
                x=df_layers.dist,
                y=df_layers[k],
                pen=pen,
                style=layers_line_style,
            )

        else:
            if kwargs.get("layers_line_cmap", None) is None:
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

                    color = kwargs.get("layers_pen_color", None)
                    if isinstance(color, list):
                        color = color[i]
                    if color is None:
                        color = v["color"]

                    style = kwargs.get("layers_pen_style", None)
                    if isinstance(style, list):
                        style = style[i]
                    if style is None:
                        style = ""
                pen = f"{thick[i]}p,{color},{style}"

                fig.plot(
                    x=df_layers.dist,
                    y=df_layers[k],
                    pen=pen,
                    frame=kwargs.get("layers_frame", ["nSew", "a"]),
                    label=v["name"],
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
                    frame=kwargs.get("layers_frame", ["nSew", "a"]),
                    label=v["name"],
                    cmap=True,
                    zvalue=v["color"],
                )

    if kwargs.get("layers_line_cmap", None) is not None:
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
        map_reg = utils.alter_region(
            vd.get_region((df_layers.x, df_layers.y)), buffer=buffer
        )[1]

        # Set figure parameters
        if subplot_orientation == "horizontal":
            # if shifting horizontally, set map height to match graph height
                map_reg,
                fig_height=fig_height,
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
                map_reg,
                fig_width=fig_width,
            )
            # shift map up with a 1/2 cm margin
            if data_dict is not None:
                fig.shift_origin(yshift=f"{layers_height+.5}c")
            else:
                fig.shift_origin(yshift=f"{fig_height+.5}c")
        else:

        # plot imagery, or supplied grid as background
        # cant use maps.plot_grd becauseit reset projection
        fig.grdimage(
            region=map_reg,
            projection=map_proj,
            grid=kwargs.get("map_background", fetch.imagery()),
            cmap=kwargs.get("map_cmap", "earth"),
            verbose="q",
        )

        # plot groundingline and coastlines
        if coast is True:
            maps.add_coast(
                fig,
                map_reg,
                map_proj,
                pen=kwargs.get("coast_pen", "1.2p,black"),
                no_coast=kwargs.get("no_coast", False),
                version=kwargs.get("coast_version", "depoorter-2013"),
            )

        # add lat long grid lines
        if gridlines is True:
            maps.add_gridlines(
                fig,
                map_reg,
                map_proj_ll,
                x_spacing=kwargs.get("x_spacing", None),
                y_spacing=kwargs.get("y_spacing", None),
            )

        # plot profile location, and endpoints on map
        fig.plot(
            projection=map_proj,
            region=map_reg,
            x=df_layers.x,
            y=df_layers.y,
            pen=kwargs.get("map_line_pen", "2p,red"),
        )
        fig.text(
            x=df_layers.loc[df_layers.dist.idxmin()].x,
            y=df_layers.loc[df_layers.dist.idxmin()].y,
            text=kwargs.get("start_label", "A"),
            fill="white",
            font="12p,Helvetica,black",
            justify="CM",
            clearance="+tO",
        )
        fig.text(
            x=df_layers.loc[df_layers.dist.idxmax()].x,
            y=df_layers.loc[df_layers.dist.idxmax()].y,
            text=kwargs.get("end_label", "B"),
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
                fill=kwargs.get("map_points_color", "blue"),
            )

        # add inset map
        if inset is True:
            maps.add_inset(
                fig,
                region=map_reg,
                inset_pos=kwargs.get("inset_pos", "TL"),
                inset_width=kwargs.get("inset_width", 0.25),
                inset_reg=kwargs.get("inset_reg", [-2800e3, 2800e3, -2800e3, 2800e3]),
            )

    if kwargs.get("save") is True:
        if kwargs.get("path") is None:
        fig.savefig(kwargs.get("path"), dpi=300)

    return fig, df_layers, df_data


def plot_data(
    method: str,
    add_map: bool = False,
    fig_height: float = 9,
    fig_width: float = 14,
    **kwargs,
):
    """
    Show sampled data on a cross section, with an optional location map.

    Parameters
    ----------
    method : str
        Choose sampling method, either "points", "shapefile", or "polyline"
    data_dict : dict, optional
        nested dictionary of data to include in option graph, construct with
        `profile.make_data_dict`, by default is gravity and magnetic anomalies.
    add_map : bool = False
        Choose whether to add a location map, by default is False.
    fig_height : float, optional
        Set the height of the figure (excluding the map) in cm, by default is 9.
    fig_width : float, optional
        Set the width of the figure (excluding the map) in cm, by default is 14.
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
        'earth'.
    map_buffer: float (0-1)
        Change map zoom as relative percentage of profile length, by default is 0.3.
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
    subplot_orientation = kwargs.get("subplot_orientation", "horizontal")
    gridlines = kwargs.get("gridlines", True)
    map_points = kwargs.get("map_points", None)
    coast = kwargs.get("coast", True)

    # create dataframe of points
    points = create_profile(method, **kwargs)

    # create default data dictionary
    if data_dict == "default":
        with redirect_stdout(None), redirect_stderr(None):
            data_dict = default_data(region=vd.get_region((points.x, points.y)))

    # sample data grids
    df_data = points.copy()
    if data_dict is not None:
        points = points[["x", "y", "dist"]].copy()
        for k, v in data_dict.items():

    # shorten profiles
        if (kwargs.get("max_dist") or kwargs.get("min_dist")) is None:
    df_data = shorten(df_data, **kwargs)

    fig = pygmt.Figure()

    # height of data, plus 0.5cm margin equals total figure height
    data_height = fig_height - 0.5

    data_projection = f"X{fig_width}c/{data_height}c"

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

    frames = kwargs.get("data_frame", None)

        frames = [frames]

    for i, (k, v) in enumerate(data_dict.items()):
        if v["axis"] == axes.unique()[0]:
            data_min = np.min([a for (a, b) in ax0_min_max])
            data_max = np.max([b for (a, b) in ax0_min_max])

            if frames[0] is None:
                frame = [
                    "neSW",
                    f"xag+l{kwargs.get('data_x_label',' ')}",
                    f"yag+l{kwargs.get('data_y0_label',' ')}",
                ]
            else:
                frame = frames[0]
        else:
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
        if kwargs.get("data_line_cmap", None) is None:
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

                color = kwargs.get("data_pen_color", None)
                if isinstance(color, list):
                    color = color[i]
                if color is None:
                    color = v["color"]

                style = kwargs.get("data_pen_style", None)
                if isinstance(style, list):
                    style = style[i]
                if style is None:
                    style = ""
            pen = f"{thick[i]}p,{color},{style}"

            data_line_style = kwargs.get("data_line_style", None)
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
    if kwargs.get("data_line_cmap", None) is not None:
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
        map_reg = utils.alter_region(
            vd.get_region((df_data.x, df_data.y)), buffer=buffer
        )[1]

        # Set figure parameters
        if subplot_orientation == "horizontal":
            # if shifting horizontally, set map height to match graph height
                map_reg,
                fig_height=fig_height,
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
                map_reg,
                fig_width=fig_width,
            )
            # shift map up with a 1/2 cm margin
            if data_dict is not None:
                fig.shift_origin(yshift=f"{data_height+.5}c")
            else:
                fig.shift_origin(yshift=f"{fig_height+.5}c")
        else:

        # plot imagery, or supplied grid as background
        # cant use maps.plot_grd becauseit reset projection
        fig.grdimage(
            region=map_reg,
            projection=map_proj,
            grid=kwargs.get("map_background", fetch.imagery()),
            cmap=kwargs.get("map_cmap", "earth"),
            verbose="q",
        )

        # plot groundingline and coastlines
        if coast is True:
            maps.add_coast(
                fig,
                map_reg,
                map_proj,
                pen=kwargs.get("coast_pen", "1.2p,black"),
                no_coast=kwargs.get("no_coast", False),
                version=kwargs.get("coast_version", "depoorter-2013"),
            )

        # add lat long grid lines
        if gridlines is True:
            maps.add_gridlines(
                fig,
                map_reg,
                map_proj_ll,
                x_spacing=kwargs.get("x_spacing", None),
                y_spacing=kwargs.get("y_spacing", None),
            )

        # plot profile location, and endpoints on map
        fig.plot(
            projection=map_proj,
            region=map_reg,
            x=df_data.x,
            y=df_data.y,
            pen=kwargs.get("map_line_pen", "2p,red"),
        )
        fig.text(
            x=df_data.loc[df_data.dist.idxmin()].x,
            y=df_data.loc[df_data.dist.idxmin()].y,
            text=kwargs.get("start_label", "A"),
            fill="white",
            font="12p,Helvetica,black",
            justify="CM",
            clearance="+tO",
        )
        fig.text(
            x=df_data.loc[df_data.dist.idxmax()].x,
            y=df_data.loc[df_data.dist.idxmax()].y,
            text=kwargs.get("end_label", "B"),
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
                fill=kwargs.get("map_points_color", "blue"),
            )

        # add inset map
        if inset is True:
            maps.add_inset(
                fig,
                region=map_reg,
                inset_pos=kwargs.get("inset_pos", "TL"),
                inset_width=kwargs.get("inset_width", 0.25),
                inset_reg=kwargs.get("inset_reg", [-2800e3, 2800e3, -2800e3, 2800e3]),
            )

    if kwargs.get("save") is True:
        if kwargs.get("path") is None:
        fig.savefig(kwargs.get("path"), dpi=300)

    return fig, df_data


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
    df = df.copy()
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
    df1 = df.copy()
    df1 = rel_dist(df1, reverse=reverse)
    df1["dist"] = df1.rel_dist.cumsum()
    return df1


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

    clear_m()

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

        shapes = []
        for coords in geo_json["geometry"]["coordinates"]:
            shapes.append(list(coords))
        shapes = list(shapes)
        if action == "created":
            lines.append(shapes)


    clear_m()

    return lines
