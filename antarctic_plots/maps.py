# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#

import warnings
from math import floor, log10
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt
import pyogrio
import xarray as xr

from antarctic_plots import fetch, regions, utils

try:
    import ipyleaflet
    import ipywidgets
except ImportError:
    _has_ipyleaflet = False
else:
    _has_ipyleaflet = True


def basemap(
    region: Union[str or np.ndarray] = None,
    fig_height: float = 15,
    fig_width: float = None,
    origin_shift: str = "initialize",
    **kwargs,
):
    # set figure projection and size from input region and figure dimensions
    # by default use figure height to set projection
    if fig_width is None:
        proj, proj_latlon, fig_width, fig_height = utils.set_proj(
            region,
            fig_height=fig_height,
        )
    # if fig_width is set, use it to set projection
    else:
        proj, proj_latlon, fig_width, fig_height = utils.set_proj(
            region,
            fig_width=fig_width,
        )

    # initialize figure or shift for new subplot
    if origin_shift == "initialize":
        fig = pygmt.Figure()
    elif origin_shift == "xshift":
        fig = kwargs.get("fig")
        fig.shift_origin(xshift=(kwargs.get("xshift_amount", 1) * (fig_width + 0.4)))
    elif origin_shift == "yshift":
        fig = kwargs.get("fig")
        fig.shift_origin(yshift=(kwargs.get("yshift_amount", 1) * (fig_height + 3)))
    elif origin_shift == "both_shift":
        fig = kwargs.get("fig")
        fig.shift_origin(
            xshift=(kwargs.get("xshift_amount", 1) * (fig_width + 0.4)),
            yshift=(kwargs.get("yshift_amount", 1) * (fig_height + 3)),
        )
    elif origin_shift == "no_shift":
        fig = kwargs.get("fig")

    # create blank basemap
    fig.basemap(
        region=region,
        projection=proj,
        frame=[f"nwse+g{kwargs.get('background', 'white')}", "xf100000", "yf100000"],
        verbose="e",
    )

    # plot coast
    if kwargs.get("coast", False) is True:
        add_coast(
            fig,
            region,
            proj,
            pen=kwargs.get("coast_pen", None),
            no_coast=kwargs.get("no_coast", False),
        )

    # add lat long grid lines
    if kwargs.get("gridlines", True) is True:
        add_gridlines(
            fig,
            region=region,
            projection=proj_latlon,
            x_spacing=kwargs.get("x_spacing", None),
            y_spacing=kwargs.get("y_spacing", None),
        )

    # add inset map to show figure location
    if kwargs.get("inset", False) is True:
        add_inset(
            fig,
            fig_width,
            inset_pos=kwargs.get("inset_pos", "TL"),
        )

    # add scalebar
    if kwargs.get("scalebar", False) is True:
        add_scalebar(
            fig,
            region,
            proj_latlon,
            font_color=kwargs.get("scale_font_color", "black"),
            scale_length=kwargs.get("scale_length"),
            length_perc=kwargs.get("scale_length_perc", 0.25),
            position=kwargs.get("scale_position", "n.5/.05"),
        )

    # blank plotting call to reset projection to EPSG:3031, optionall add title
    if kwargs.get("title", None) is None:
        fig.basemap(
            region=region,
            projection=proj,
            frame="wesn",
        )
    else:
        fig.basemap(
            region=region,
            projection=proj,
            frame=f"wesn+t{kwargs.get('title')}",
        )

    return fig


def plot_grd(
    grid: Union[str or xr.DataArray],
    cmap: str = "viridis",
    region: Union[str or np.ndarray] = None,
    coast: bool = False,
    origin_shift: str = "initialize",
    **kwargs,
):
    """
    Helps easily create PyGMT maps, individually or as subplots.

    Parameters
    ----------
    grid : Union[str or xr.DataArray]
        grid file to plot, either loaded xr.DataArray or string of a filename
    cmap : str, optional
        GMT color scale to use, by default 'viridis'
    region : Union[str or np.ndarray], optional
        region to plot, by default is extent of input grid
    coast : bool, optional
        choose whether to plot Antarctic coastline and grounding line, by default False
    origin_shift : str, optional
        automatically will create a new figure, set to 'xshift' to instead add plot to
        right of previous plot, or 'yshift' to add plot above previous plot, by
        default 'initialize'.

    Keyword Args
    ------------
    image : bool
        set to True if plotting imagery to correctly set colorscale.
    grd2cpt : bool
        use GMT module grd2cpt to set color scale from grid values, by default is False
    cmap_region : Union[str or np.ndarray]
        region to use to define color scale if grd2cpt is True, by default is
        region
    cbar_label : str
        label to add to colorbar.
    points : pd.DataFrame
        points to plot on map, must contain columns 'x' and 'y'.
    show_region : np.ndarray
        GMT-format region to use to plot a bounding regions.
    cpt_lims : Union[str or tuple]
        limits to use for color scale max and min, by default is max and min of data.
    fig : pygmt.Figure()
        if adding subplots, set the first returned figure to a variable, and add that
        variable as the kwargs 'fig' to subsequent calls to plot_grd.
    gridlines : bool
        choose to plot lat/long grid lines, by default is False
    inset : bool
        choose to plot inset map showing figure location, by default is False
    inset_pos : str
        position for inset map; either 'TL', 'TR', BL', 'BR', by default is 'TL'
    fig_height : int or float
        height in cm for figures, by default is 15cm.
    scalebar: bool
        choose to add a scalebar to the plot, by default is False. See
        `maps.add_scalebar` for additional kwargs.
    colorbar: bool
        choose to add a colorbar to the plot, by default is True
    Returns
    -------
    PyGMT.Figure()
        Returns a figure object, which can be passed to the `fig` kwarg to add subplots
        or other `PyGMT` plotting methods.

    Example
    -------
    >>> from antarctic_plots import maps
    ...
    >>> fig = maps.plot_grd('grid1.nc')
    >>> fig = maps.plot_grd(
    ... 'grid2.nc',
    ... origin_shift = 'xshift',
    ... fig = fig,
    ... )
    ...
    >>> fig.show()
    """

    warnings.filterwarnings("ignore", message="pandas.Int64Index")
    warnings.filterwarnings("ignore", message="pandas.Float64Index")

    if region is None:
        try:
            region = utils.get_grid_info(grid)[1]
        except Exception:  # (ValueError, pygmt.exceptions.GMTInvalidInput):
            # raise
            print("grid region can't be extracted, using antarctic region.")
            region = regions.antarctica

    # print(f"plot region is: {region}")

    cmap_region = kwargs.get("cmap_region", region)
    show_region = kwargs.get("show_region", None)
    cpt_lims = kwargs.get("cpt_lims", None)
    grd2cpt = kwargs.get("grd2cpt", False)
    image = kwargs.get("image", False)
    gridlines = kwargs.get("gridlines", False)
    points = kwargs.get("points", None)
    inset = kwargs.get("inset", False)
    title = kwargs.get("title", None)
    fig_height = kwargs.get("fig_height", 15)
    scalebar = kwargs.get("scalebar", False)
    colorbar = kwargs.get("colorbar", True)
    reverse_cpt = kwargs.get("reverse_cpt", False)

    # set figure projection and size from input region
    proj, proj_latlon, fig_width, fig_height = utils.set_proj(region, fig_height)

    # initialize figure or shift for new subplot
    if origin_shift == "initialize":
        fig = pygmt.Figure()
    elif origin_shift == "xshift":
        fig = kwargs.get("fig")
        fig.shift_origin(xshift=(kwargs.get("xshift_amount", 1) * (fig_width + 0.4)))
    elif origin_shift == "yshift":
        fig = kwargs.get("fig")
        fig.shift_origin(yshift=(kwargs.get("yshift_amount", 1) * (fig_height + 3)))
    elif origin_shift == "both_shift":
        fig = kwargs.get("fig")
        fig.shift_origin(
            xshift=(kwargs.get("xshift_amount", 1) * (fig_width + 0.4)),
            yshift=(kwargs.get("yshift_amount", 1) * (fig_height + 3)),
        )
    elif origin_shift == "no_shift":
        fig = kwargs.get("fig")

    # set cmap
    if image is True:
        pygmt.makecpt(
            cmap=cmap,
            series="15000/17000/1",
            verbose="e",
        )
        colorbar = False
    elif grd2cpt is True:
        pygmt.grd2cpt(
            cmap=cmap,
            grid=grid,
            region=cmap_region,
            background=True,
            continuous=kwargs.get("continuous", True),
            color_model=kwargs.get("color_model", "R"),
            categorical=kwargs.get("categorical", False),
            reverse=reverse_cpt,
            verbose="e",
        )
    elif cpt_lims is not None:
        try:
            pygmt.makecpt(
                cmap=cmap,
                series=cpt_lims,
                background=True,
                continuous=kwargs.get("continuous", False),
                color_model=kwargs.get("color_model", "R"),
                categorical=kwargs.get("categorical", False),
                reverse=reverse_cpt,
                verbose="e",
            )
        except (pygmt.exceptions.GMTCLibError):
            pygmt.makecpt(
                cmap=cmap,
                background=True,
                continuous=kwargs.get("continuous", False),
                color_model=kwargs.get("color_model", "R"),
                categorical=kwargs.get("categorical", False),
                reverse=reverse_cpt,
                verbose="e",
            )
    else:
        try:
            zmin, zmax = utils.get_grid_info(grid)[2], utils.get_grid_info(grid)[3]
            pygmt.makecpt(
                cmap=cmap,
                background=True,
                continuous=kwargs.get("continuous", True),
                series=(zmin, zmax),
                reverse=reverse_cpt,
                verbose="e",
            )
        except Exception:  # (ValueError, pygmt.exceptions.GMTInvalidInput):
            # raise
            print("grid region can't be extracted.")
            pygmt.makecpt(
                cmap=cmap,
                background=True,
                continuous=kwargs.get("continuous", True),
                reverse=reverse_cpt,
                verbose="e",
            )

    # display grid
    fig.grdimage(
        grid=grid,
        cmap=True,
        projection=proj,
        region=region,
        nan_transparent=True,
        frame=[f"+g{kwargs.get('background', 'white')}"],
        shading=kwargs.get("shading", None),
        verbose="q",
    )

    # display colorbar
    if colorbar is True:
        fig.colorbar(
            cmap=True,
            position=f"jBC+w{fig_width*.8}c+jTC+h+o0c/.2c+e",
            frame=[
                f"xaf+l{kwargs.get('cbar_label',' ')}",
                f"y+l{kwargs.get('cbar_unit',' ')}",
            ],
        )

    # plot groundingline and coastlines
    if coast is True:
        add_coast(
            fig,
            region,
            proj,
            pen=kwargs.get("coast_pen", None),
            no_coast=kwargs.get("no_coast", False),
        )

    # add datapoints
    if points is not None:
        fig.plot(
            x=points.x,
            y=points.y,
            style="c.2c",
            color="black",
        )

    # add box showing region
    if show_region is not None:
        add_box(fig, show_region)

    # add lat long grid lines
    if gridlines is True:
        add_gridlines(
            fig,
            region=region,
            projection=proj_latlon,
            x_spacing=kwargs.get("x_spacing", None),
            y_spacing=kwargs.get("y_spacing", None),
        )

    # add inset map to show figure location
    if inset is True:
        add_inset(
            fig,
            fig_width,
            inset_pos=kwargs.get("inset_pos", "TL"),
        )

    # add scalebar
    if scalebar is True:
        add_scalebar(
            fig,
            region,
            proj_latlon,
            font_color=kwargs.get("font_color", "black"),
            scale_length=kwargs.get("scale_length"),
            length_perc=kwargs.get("length_perc", 0.25),
            position=kwargs.get("position", "n.5/.05"),
        )
    # reset region and projection
    if title is None:
        fig.basemap(region=region, projection=proj, frame="wesn")
    else:
        fig.basemap(region=region, projection=proj, frame=f"wesn+t{title}")

    return fig


def add_coast(
    fig: pygmt.figure,
    region: Union[str or np.ndarray] = None,
    projection: str = None,
    no_coast: bool = False,
    pen=None,
):
    """
    add coastline and groundingline to figure.

    Parameters
    ----------
    fig : pygmt.figure
    region : Union[str or np.ndarray], optional
        region for the figure, by default is last used by PyGMT
    projection : str, optional
        GMT projection string, by default is last used by PyGMT
    no_coast : bool
        If True, only plot groundingline, not coastline, by default is False
    pen : None
        GMT pen string, by default "0.6p,black"
    """
    if pen is None:
        pen = "0.6p,black"

    gdf = pyogrio.read_dataframe(fetch.groundingline())

    if no_coast is False:
        data = gdf
    elif no_coast is True:
        data = gdf[gdf.Id_text == "Grounded ice or land"]

    fig.plot(
        data,
        projection=projection,
        region=region,
        pen=pen,
    )


def add_gridlines(
    fig: pygmt.figure,
    region: Union[str or np.ndarray] = None,
    projection: str = None,
    **kwargs,
):
    """
    add lat lon grid lines and annotations to a figure. Use kwargs x_spacing and
    y_spacing to customize the interval of gridlines and annotations.

    Parameters
    ----------
    fig : PyGMT.figure instance
    region : Union[str or np.ndarray], optional
        region for the figure
    projection : str, optional
        GMT projection string in lat lon, if your previous pygmt.Figure() call used a
        cartesian projection, you will need to provide a projection in lat/lon here, use
        utils.set_proj() to make this projection.

    """

    x_spacing = kwargs.get("x_spacing", None)
    y_spacing = kwargs.get("y_spacing", None)

    if x_spacing is None:
        x_frames = ["xag", "xa"]
    else:
        x_frames = [
            f"xa{x_spacing}g{x_spacing/2}",
            f"xa{x_spacing}",
        ]

    if y_spacing is None:
        y_frames = ["yag", "ya"]
    else:
        y_frames = [
            f"ya{y_spacing}g{y_spacing/2}",
            f"ya{y_spacing}",
        ]

    with pygmt.config(
        MAP_ANNOT_OFFSET_PRIMARY=kwargs.get(
            "MAP_ANNOT_OFFSET_PRIMARY", "20p"
        ),  # move annotations in/out radially
        MAP_ANNOT_MIN_ANGLE=0,
        MAP_FRAME_TYPE="inside",
        MAP_ANNOT_OBLIQUE=0,  # rotate relative to lines
        FONT_ANNOT_PRIMARY="8p,black,-=2p,white",
        MAP_GRID_PEN_PRIMARY="auto,gray",
        MAP_TICK_LENGTH_PRIMARY="-5p",
        MAP_TICK_PEN_PRIMARY="auto,gray",
        # FORMAT_GEO_MAP="dddF",
        # MAP_POLAR_CAP="90/90",
    ):
        # plot semi-transparent lines and annotations with black font and white shadow
        fig.basemap(
            projection=projection,
            region=region,
            frame=[
                "NSWE",
                x_frames[0],
                y_frames[0],
            ],
            transparency=50,
            verbose="q",
        )
        # re-plot annotations with no transparency
        with pygmt.config(FONT_ANNOT_PRIMARY="8p,black"):
            fig.basemap(
                projection=projection,
                region=region,
                frame=[
                    "NSWE",
                    x_frames[0],
                    y_frames[0],
                ],
                verbose="q",
            )


def add_inset(
    fig: pygmt.figure,
    fig_width: Union[int, float],
    region: Union[str or np.ndarray] = None,
    inset_pos: str = "TL",
    inset_width: float = 0.25,
    inset_reg: list = [-2800e3, 2800e3, -2800e3, 2800e3],
    **kwargs,
):
    """
    add an inset map showing the figure region relative to the Antarctic continent.

    Parameters
    ----------
    fig : PyGMT.figure instance
    fig_width : float or int
        width of figure in cm, if you didn't explicitly set this in creating the figure
        find the value with utils.set_proj()
    region : Union[str or np.ndarray], optional
        region for the figure
    inset_pos : str, optional
        GMT location string for inset map, by default 'TL' (top left)
    inset_width : float, optional
        Inset width as percentage of the total figure width, by default is 25% (0.25)
    inset_reg : list, optional
        Region of Antarctica to plot for the inset map, by default is whole continent
    """
    coast_pen = kwargs.get("coast_pen", "0.2,black")

    inset_map = f"X{fig_width*inset_width}c"

    # if no region supplied, get region of current PyGMT figure
    if region is None:
        with pygmt.clib.Session() as lib:
            region = lib.extract_region()

    with fig.inset(
        position=f"J{inset_pos}+j{inset_pos}+w{fig_width*inset_width}c",
        verbose="q",
    ):
        gdf = pyogrio.read_dataframe(fetch.groundingline())
        fig.plot(
            projection=inset_map,
            region=inset_reg,
            data=gdf[gdf.Id_text == "Ice shelf"],
            color="skyblue",
        )
        fig.plot(data=gdf[gdf.Id_text == "Grounded ice or land"], color="grey")
        fig.plot(data=fetch.groundingline(), pen=coast_pen)

        fig.plot(
            x=[
                region[0],
                region[0],
                region[1],
                region[1],
                region[0],
            ],
            y=[
                region[2],
                region[3],
                region[3],
                region[2],
                region[2],
            ],
            pen="1p,black",
        )


def add_scalebar(
    fig: pygmt.figure,
    region: Union[str or np.ndarray] = None,
    projection: str = None,
    **kwargs,
):
    """
    add lat lon grid lines and annotations to a figure.

    Parameters
    ----------
    fig : PyGMT.figure instance
    region : np.ndarray, optional
        region for the figure
    projection : str, optional
        GMT projection string in lat lon, if your previous pygmt.Figure() call used a
        cartesian projection, you will need to provide a projection in lat/lon here, use
        utils.set_proj() to make this projection.

    """
    font_color = kwargs.get("font_color", "black")
    scale_length = kwargs.get("scale_length")
    length_perc = kwargs.get("length_perc", 0.25)
    position = kwargs.get("position", "n.5/.05")

    # if no region supplied, get region of current PyGMT figure
    if region is None:
        with pygmt.clib.Session() as lib:
            region = lib.extract_region()

    def round_to_1(x):
        return round(x, -int(floor(log10(abs(x)))))

    if scale_length is None:
        scale_length = round_to_1((abs(region[1] - region[0])) / 1000 * length_perc)

    with pygmt.config(
        FONT_ANNOT_PRIMARY=f"10p,{font_color}",
        FONT_LABEL=f"10p,{font_color}",
        MAP_SCALE_HEIGHT="6p",
        MAP_TICK_PEN_PRIMARY=f"0.5p,{font_color}",
    ):
        fig.basemap(
            region=region,
            projection=projection,
            map_scale=f'{position}+w{scale_length}k+f+l"km"+ar',
            verbose="e",
        )


def add_box(
    fig: pygmt.figure,
    box: Union[list or np.ndarray],
    pen="2p,black",
):
    """
    Plot a GMT region as a box.

    Parameters
    ----------
    fig : pygmt.figure
        Figure to plot on
    box : Union[list or np.ndarray]
        region in EPSG3031 in format [e,w,n,s] in meters
    pen : str, optional
        GMT pen string used for the box, by default "2p,black"
    """
    fig.plot(
        x=[box[0], box[0], box[1], box[1], box[0]],
        y=[box[2], box[3], box[3], box[2], box[2]],
        pen=pen,
    )


def interactive_map(
    center_yx: list = None,
    zoom: float = 0,
    display_xy: bool = True,
    show: bool = True,
    points: pd.DataFrame = None,
    **kwargs,
):
    """
    Plot an interactive map with satellite imagery. Clicking gives the cursor location
    in EPSG:3031 [x,y]. Requires ipyleaflet

    Parameters
    ----------
    center_yx : list, optional
        choose center coordinates in EPSG3031 [y,x], by default [0,0]
    zoom : float, optional
        choose zoom level, by default 0
    display_xy : bool, optional
        choose if you want clicks to show the xy location, by default True
    show : bool, optional
        choose whether to displat the map, by default True
    points : pd.DataFrame, optional
        choose to plot points suppied as columns x, y, in EPSG:3031 in a dataframe
    """

    if not _has_ipyleaflet:
        raise ImportError(
            "ipyleaflet is required to plot an interactive map. Install with `mamba install ipyleaflet`."  # noqa
        )
    layout = ipywidgets.Layout(
        width=kwargs.get("width", "auto"),
        height=kwargs.get("height", None),
    )

    # if points are supplied, center map on them and plot them
    if points is not None:
        if kwargs.get("points_as_latlon", False) is True:
            center_ll = [points.lon.mean(), points.lat.mean()]
        else:
            # convert points to lat lon
            points_ll = utils.epsg3031_to_latlon(points)
            # if points supplied, center map on points
            center_ll = [np.nanmedian(points_ll.lat), np.nanmedian(points_ll.lon)]
            # add points to geodataframe
            gdf = gpd.GeoDataFrame(
                points_ll,
                geometry=gpd.points_from_xy(points_ll.lon, points_ll.lat),
            )
            geo_data = ipyleaflet.GeoData(
                geo_dataframe=gdf,
                # style={'radius': .5, 'color': 'red', 'weight': .5},
                point_style={"radius": 1, "color": "red", "weight": 1},
            )
    else:
        # if no points, center map on 0, 0
        center_ll = utils.epsg3031_to_latlon([0, 0])

    if center_yx is not None:
        center_ll = utils.epsg3031_to_latlon(center_yx)

    # create the map
    m = ipyleaflet.Map(
        center=center_ll,
        zoom=zoom,
        layout=layout,
        basemap=ipyleaflet.basemaps.NASAGIBS.BlueMarble3031,
        crs=ipyleaflet.projections.EPSG3031,
        dragging=True,
    )

    if points is not None:
        m.add_layer(geo_data)

    m.default_style = {"cursor": "crosshair"}
    if display_xy is True:
        label_xy = ipywidgets.Label()
        display(label_xy)  # noqa

        def handle_click(**kwargs):
            if kwargs.get("type") == "click":
                latlon = kwargs.get("coordinates")
                label_xy.value = str(utils.latlon_to_epsg3031(latlon))

    m.on_interaction(handle_click)

    if show is True:
        display(m)  # noqa

    return m


def subplots(
    grids: list,
    region: Union[str or np.ndarray] = None,
    dims: tuple = None,
    **kwargs,
):
    """
    Plot a series of grids as individual suplots. This will automatically configure the
    layout to be closest to a square. Add any parameters from `plot_grd()` here as
    keyword arguments for further customization.

    Parameters
    ----------
    grids : list
        list of xr.DataArray's to be plotted
    region : Union[str or np.ndarray], optional
        choose to subset the grids to a specified region, by default None
    dims : tuple, optional
        customize the subplot dimensions (# rows, # columns), by default will use
        `utils.square_subplots()` to make a square(~ish) layout.

    Returns
    -------
    PyGMT.Figure()
        Returns a figure object, which can be used by other PyGMT plotting functions.

    """
    # if no define region, get from first grid in list
    if region is None:
        try:
            region = utils.get_grid_info(grids[0])[1]
        except Exception:  # (ValueError, pygmt.exceptions.GMTInvalidInput):
            # raise
            print("grid region can't be extracted, using antarctic region.")
            region = regions.antarctica

    # get square dimensions for subplot
    if dims is None:
        subplot_dimensions = utils.square_subplots(len(grids))
    else:
        subplot_dimensions = dims

    # get subplot size
    proj, proj_latlon, fig_width, fig_height = utils.set_proj(
        region, kwargs.get("fig_height", 15)
    )

    # initialize figure
    fig = pygmt.Figure()

    with fig.subplot(
        nrows=subplot_dimensions[0],
        ncols=subplot_dimensions[1],
        subsize=(fig_width, fig_height),
        frame=kwargs.get("frame", "f"),
        clearance=kwargs.get("clearance", None),
        title=kwargs.get("fig_title", None),
        margins=kwargs.get("margins", "0.5c"),
        autolabel=kwargs.get("autolabel"),
    ):
        for i, j in enumerate(grids):
            with fig.set_panel(panel=i):

                # if list of cmaps provided, use them
                if kwargs.get("cmaps", None) is not None:
                    cmap = kwargs.get("cmaps", None)[i]
                # if not, use viridis
                else:
                    cmap = "viridis"

                # if list of titles provided, use them
                if kwargs.get("subplot_titles", None) is not None:
                    sub_title = kwargs.get("subplot_titles", None)[i]
                else:
                    sub_title = None

                # if list of colorbar labels provided, use them
                if kwargs.get("cbar_labels", None) is not None:
                    cbar_label = kwargs.get("cbar_labels", None)[i]
                else:
                    cbar_label = " "

                # if list of colorbar units provided, use them
                if kwargs.get("cbar_units", None) is not None:
                    cbar_unit = kwargs.get("cbar_units", None)[i]
                else:
                    cbar_unit = " "

                # plot the grids
                plot_grd(
                    j,
                    fig=fig,
                    origin_shift="no_shift",
                    region=region,
                    cmap=cmap,
                    title=sub_title,
                    cbar_label=cbar_label,
                    cbar_unit=cbar_unit,
                    **kwargs,
                )
    return fig


def plot_3d(
    grids: list,
    cmaps: list,
    exaggeration: list,
    view: list = [170, 30],
    vlims: list = [-10000, 1000],
    region: Union[str or np.ndarray] = None,
    shp_mask: Union[str or gpd.GeoDataFrame] = None,
    polygon_mask: list = None,
    colorbar=True,
    grd2cpt=True,
    **kwargs,
):
    """
    _summary_

    Parameters
    ----------
    grids : list
        _description_
    cmaps : list
        _description_
    exaggeration : list
        _description_
    view : list, optional
        _description_, by default [170, 30]
    vlims : list, optional
        _description_, by default [-10000, 1000]
    region : Union[str or np.ndarray], optional
        _description_, by default None
    shp_mask : Union[str or gpd.GeoDataFrame], optional
        _description_, by default None
    cpt_lims : list, optional
        _description_, by default None
    colorbar : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    fig_height = kwargs.get("fig_height", 15)

    # if plot region not specified, try to pull from grid info
    if region is None:
        try:
            region = utils.get_grid_info(grids[0])[1]
        except Exception:  # (ValueError, pygmt.exceptions.GMTInvalidInput):
            # raise
            print("first grids' region can't be extracted, using antarctic region.")
            region = regions.antarctica

    # set figure projection and size from input region
    proj, proj_latlon, fig_width, fig_height = utils.set_proj(region, fig_height)

    # set vertical limits
    region = region + vlims

    # initialize the figure
    fig = pygmt.Figure()

    # iterate through grids and plot them
    for i, j in enumerate(grids):
        grid = j
        # if provided, mask grid with shapefile
        if shp_mask is not None:
            grid = utils.mask_from_shp(
                shp_mask,
                xr_grid=grid,
                masked=True,
                invert=kwargs.get("invert", False),
            )
            grid.to_netcdf("tmp.nc")
            grid = xr.load_dataset("tmp.nc")["z"]
        # if provided, mask grid with polygon from interactive map via
        # regions.draw_region
        elif polygon_mask is not None:
            grid = utils.mask_from_polygon(
                polygon_mask,
                grid=grid,
            )
        # create colorscales
        if grd2cpt is True:
            pygmt.grd2cpt(
                cmap=cmaps[i],
                grid=grid,
                background=True,
                continuous=True,
                verbose="e",
            )
        else:
            try:
                zmin, zmax = utils.get_grid_info(grid)[2], utils.get_grid_info(grid)[3]
                pygmt.makecpt(
                    cmap=cmaps[i],
                    background=True,
                    continuous=True,
                    series=(zmin, zmax),
                )
            except Exception:  # (ValueError, pygmt.exceptions.GMTInvalidInput):
                # raise
                print("grid region can't be extracted.")
                pygmt.makecpt(
                    cmap=cmaps[i],
                    background=True,
                    continuous=True,
                )

        # set transparency values
        transparencies = kwargs.get("transparencies", None)
        if transparencies is None:
            transparency = 0
        else:
            transparency = transparencies[i]

        # plot as perspective view
        fig.grdview(
            grid=grid,
            cmap=True,  # cmaps[i],
            projection=proj,
            region=region,
            frame=None,
            perspective=view,
            zsize=f"{exaggeration[i]}c",
            surftype="c",
            transparency=transparency,
            # plane='-9000+ggrey',
            # shading=True, #'grdgradient+a45+ne.5+m-.2'
        )

        # display colorbar
        if colorbar is True:
            cbar_xshift = kwargs.get("cbar_xshift", None)
            cbar_yshift = kwargs.get("cbar_yshift", None)

            if cbar_xshift is None:
                xshift = 0
            else:
                xshift = cbar_xshift[i]

            if cbar_yshift is None:
                yshift = fig_height / 2
            else:
                yshift = cbar_yshift[i]

            fig.shift_origin(yshift=f"{yshift}c", xshift=f"{xshift}c")
            fig.colorbar(
                cmap=True,
                position=f"jMR+w{fig_width*.4}c/.5c+v+e+m",
                frame=f"xaf+l{kwargs.get('cbar_labels',' ')[i]}",
                perspective=True,
                box="+gwhite+c3p",
            )
            fig.shift_origin(yshift=f"{-yshift}c", xshift=f"{-xshift}c")

        # shift up for next grid
        if kwargs.get("zshifts", None) is None:
            fig.shift_origin(yshift=f"{fig_height/2}c")
        else:
            fig.shift_origin(yshift=f"{kwargs.get('zshifts')[i]}c")

    return fig
