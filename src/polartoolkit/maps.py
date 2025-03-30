# Copyright (c) 2024 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#
# pylint: disable=too-many-lines
from __future__ import annotations

import copy
import pathlib
import string
import typing
import warnings
from math import floor, log10

import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
from numpy.typing import NDArray

from polartoolkit import fetch, logger, regions, utils

try:
    import pyogrio  # pylint: disable=unused-import

    ENGINE = "pyogrio"
except ImportError:
    pyogrio = None
    ENGINE = "fiona"

try:
    from IPython.display import display
except ImportError:
    display = None

try:
    import geoviews as gv
except ImportError:
    gv = None

try:
    from cartopy import crs
except ImportError:
    crs = None

try:
    import ipyleaflet
except ImportError:
    ipyleaflet = None

try:
    import ipywidgets
except ImportError:
    ipywidgets = None


def _set_figure_spec(
    region: tuple[float, float, float, float],
    origin_shift: str | None = "initialize",
    fig: pygmt.Figure | None = None,
    fig_height: float | None = None,
    fig_width: float | None = None,
    hemisphere: str | None = None,
    yshift_amount: float = -1,
    xshift_amount: float = 1,
    xshift_extra: float = 0.4,
    yshift_extra: float = 0.4,
) -> tuple[pygmt.Figure, str, str | None, float, float]:
    """determine what to do with figure"""

    # initialize figure or shift for new subplot
    if origin_shift == "initialize":
        fig = pygmt.Figure()
        # set figure projection and size from input region and figure dimensions
        # by default use figure height to set projection
        if fig_width is None:
            if fig_height is None:
                fig_height = 15
            proj, proj_latlon, fig_width, fig_height = utils.set_proj(
                region,
                fig_height=fig_height,
                hemisphere=hemisphere,
            )
        # if fig_width is set, use it to set projection
        else:
            proj, proj_latlon, fig_width, fig_height = utils.set_proj(
                region,
                fig_width=fig_width,
                hemisphere=hemisphere,
            )
    else:
        if fig is None:
            msg = (
                "If origin_shift is not 'initialize', a figure instance must be "
                "provided."
            )
            raise ValueError(msg)

        # allow various alternative strings for origin_shift
        if (origin_shift == "x_shift") | (origin_shift == "xshift"):
            origin_shift = "x"
            msg = "`origin_shift` parameter has changed, use 'x' instead."
            warnings.warn(
                msg,
                DeprecationWarning,
                stacklevel=2,
            )
        if (origin_shift == "y_shift") | (origin_shift == "yshift"):
            origin_shift = "y"
            msg = "`origin_shift` parameter has changed, use 'y' instead."
            warnings.warn(
                msg,
                DeprecationWarning,
                stacklevel=2,
            )
        if origin_shift == "both_shift":
            origin_shift = "both"
            msg = "`origin_shift='both_shift'` is deprecated, use 'both' instead."
            warnings.warn(
                msg,
                DeprecationWarning,
                stacklevel=2,
            )
        if origin_shift == "no_shift":
            origin_shift = None
            msg = "origin_shift 'no_shift' is deprecated, use None instead."
            warnings.warn(
                msg,
                DeprecationWarning,
                stacklevel=2,
            )

        # get figure height if not set
        if fig_height is None:
            fig_height = utils.get_fig_height()

        # get existing figure parameters
        proj, proj_latlon, fig_width, fig_height = utils.set_proj(
            region,
            fig_height=fig_height,
            hemisphere=hemisphere,
        )

        # determine default values for x and y shift
        # add .4 to account for the space between figures
        xshift = xshift_amount * (fig_width + xshift_extra)
        yshift = yshift_amount * (fig_height + yshift_extra)

        # add 3 to account for colorbar and titles
        # colorbar widths are automatically 80% figure width
        # colorbar heights are 4% of colorbar width
        # colorbar histograms are automatically 4*colorbar height
        # yshift = yshift_amount * (fig_height + 0.4)

        # shift origin of figure depending on origin_shift
        if origin_shift == "x":
            fig.shift_origin(xshift=xshift)
        elif origin_shift == "y":
            fig.shift_origin(yshift=yshift)
        elif origin_shift == "both":
            fig.shift_origin(xshift=xshift, yshift=yshift)
        elif origin_shift is None:
            pass
        else:
            msg = "invalid string for origin shift"
            raise ValueError(msg)

    return fig, proj, proj_latlon, fig_width, fig_height


def basemap(
    region: tuple[float, float, float, float] | None = None,
    hemisphere: str | None = None,
    coast: bool = False,
    north_arrow: bool = False,
    scalebar: bool = False,
    faults: bool = False,
    simple_basemap: bool = False,
    imagery_basemap: bool = False,
    modis_basemap: bool = False,
    title: str | None = None,
    inset: bool = False,
    points: pd.DataFrame | None = None,
    gridlines: bool = False,
    origin_shift: str = "initialize",
    fig: pygmt.Figure | None = None,
    **kwargs: typing.Any,
) -> pygmt.Figure:
    """
    Create a figure basemap in polar stereographic projection, and add a range of
    features such as coastline and grounding lines, inset figure location maps,
    background imagery, scalebars, gridlines and northarrows. Plot supplied points with
    either constant color or colored by a colormap. Reuse the figure instance to either
    plot additional features on top, or shift the plot to create subplots. There are
    many keyword arguments which can either be passed along to the various functions in
    the `maps` module, or specified specifically. Kwargs can be passed directly to the
    following functions: `add_colorbar`, `add_north_arrow`, `add_scalebar`, `add_inset`,
    `set_cmap`. Other kwargs are specified below.

    Parameters
    ----------
    region : tuple[float, float, float, float] | None, optional
        region for the figure in format [xmin, xmax, ymin, ymax], by default None
    hemisphere : str, optional
        set whether to plot in "north" hemisphere (EPSG:3413) or "south" hemisphere
        (EPSG:3031), can be set manually, or will read from the environment variable:
        "POLARTOOLKIT_HEMISPHERE"
    coast : bool, optional
        choose whether to plot coastline and grounding line, by default False. Version
        of shapefiles to plots depends on `hemisphere`, and can be changed with kwargs
        `coast_version`, which defaults to `BAS` for the northern hemisphere and
        `measures-v2` for the southern.
    north_arrow : bool, optional
        choose to add a north arrow to the plot, by default is False.
    scalebar : bool, optional
        choose to add a scalebar to the plot, by default is False. See `add_scalebar`
        for additional kwargs
    faults : bool, optional
        choose to plot faults on the map, by default is False
    simple_basemap: bool, optional
        choose to plot a simple basemap with floating ice colored blue and grounded ice
        colored grey, with boarders defined by `simple_basemap_version`.
    simple_basemap_transparency : int, optional
        transparency to use for the simple basemap, by default is 0
    simple_basemap_version : str, optional
        version of the simple basemap to plot, by default is None
    imagery_basemap : bool, optional
        choose to add a background imagery basemap, by default is False. If true, will
        use LIMA for southern hemisphere and MODIS MoG for the northern hemisphere.
    imagery_transparency : int, optional
        transparency to use for the imagery basemap, by default is 0
    modis_basemap : bool, optional
        choose to add a MODIS background imagery basemap, by default is False.
    modis_transparency : int, optional
        transparency to use for the MODIS basemap, by default is 0
    modis_version : str, optional
        version of the MODIS basemap to plot, by default is None
    title : str | None, optional
        title to add to the figure, by default is None
    inset : bool, optional
        choose to plot inset map showing figure location, by default is False
    points : pandas.DataFrame | None, optional
        points to plot on map, must contain columns 'x' and 'y' or
        'easting' and 'northing'.
    gridlines : bool, optional
        choose to plot lat/lon grid lines, by default is False
    origin_shift : str, | None, optional
        choose what to do with the plot when creating the figure. By default is
        'initialize' which will create a new figure instance. To plot additional grids
        on top of the existing figure provide a figure instance to `fig` and set
        origin_shift to None. To create subplots, provide the existing figure instance
        to `fig`, and set `origin_shift` to 'x' to add the the new plot to the right of
        previous plot, 'y' to add the new plot above the previous plot, or 'both' to add
        the new plot to the right and above the old plot. By default each of this shifts
        will be the width/height of the figure instance, this can be changed with kwargs
        `xshift_amount` and `yshift_amount`, which are in multiples of figure
        width/height.
    fig : pygmt.Figure, optional
        supply a figure instance for adding subplots or using other PyGMT plotting
        methods, by default None
    fig_height : int or float
        height in cm for figures, by default is 15cm.
    fig_width : int or float
        width in cm for figures, by default is None and is determined by fig_height and
        the projection.
    xshift_amount : int or float
        amount to shift the origin in the x direction in multiples of current figure
        instance width, by default is 1.
    yshift_amount : int or float
        amount to shift the origin in the y direction in multiples of current figure
        instance height, by default is -1.
    frame : str | bool
        GMT frame string to use for the basemap, by default is "nesw+gwhite"
    frame_pen : str
        GMT pen string to use for the frame, by default is "auto"
    frame_font : str
        GMT font string to use for the frame, by default is "auto"
    transparency : int
        transparency to use for the basemap, by default is 0
    inset_position : str
        position for inset map with PyGMT syntax, by default is "jTL+jTL+o0/0"
    title_font : str
        font to use for the title, by default is 'auto'
    show_region : tuple[float, float, float, float]
        show a rectangular region on the map, in the format [xmin, xmax, ymin, ymax].
    region_pen : str
        GMT pen string to use for the region box, by default is None
    x_spacing : float
        spacing for x gridlines in degrees, by default is None
    y_spacing : float
        spacing for y gridlines in degrees, by default is None
    points_style : str
        style of points to plot in GMT format, by default is 'c.2c'.
    points_fill : str
        fill color of points, either string of color name or column name to color
        points by, by default is 'black'.
    points_pen : str
        pen color and width of points, by default is '1p,black' if constant color or
        None if using a cmap.
    points_label : str
        label to add to legend, by default is None
    points_cmap : str
        GMT color scale to use for coloring points, by default 'viridis'. If True, will
        use the last used in PyGMT.
    cpt_lims : str or tuple]
        limits to use for color scale max and min, by default is max and min of data.
    cmap_region : str or tuple[float, float, float, float]
        region to use to define color scale limits, in format [xmin, xmax, ymin, ymax],
        by default is region
    robust : bool
        use the 2nd and 98th percentile (or those specified with 'robust_percentiles')
        of the data to set color scale limits, by default is False.
    robust_percentiles : tuple[float, float]
        percentiles to use for robust colormap limits, by default is (0.02, 0.98).
    reverse_cpt : bool
        reverse the color scale, by default is False.
    cbar_label : str
        label to add to colorbar.
    colorbar : bool
        choose to add a colorbar for the points to the plot, by default is False.
    scalebar_font_color : str
        color of the scalebar font, by default is 'black'.
    scale_font_color : str
        deprecated, use scalebar_font_color.
    scalebar_length_perc : float
        percentage of the min dimension of the figure region to use for the scalebar,
        by default is 0.25.
    scale_length_perc : float
        deprecated, use scalebar_length_perc.
    scalebar_position : str
        position of the scalebar on the figure, by default is 'n.5/.05' which is bottom
        center of the plot.
    scale_position : str
        deprecated, use scalebar_position.
    coast_pen : str
        GMT pen string to use for the coastlines, by default is None
    no_coast : bool
        choose to not plot coastlines, just grounding lines, by default is False
    coast_version : str
        version of coastlines to plot, by default depends on the hemisphere
    coast_label : str
        label to add to coastlines, by default is None
    fault_label : str
        label to add to faults, by default is None
    fault_pen : str
        GMT pen string to use for the faults, by default is None
    fault_style : str
        GMT style string to use for the faults, by default is None
    fault_activity : str
        column name in faults to use for activity, by default is None
    fault_motion : str
        column name in faults to use for motion, by default is None
    fault_exposure : str
        column name in faults to use for exposure, by default is None

    Returns
    -------
    pygmt.Figure
        Returns a figure object, which can be passed to the `fig` kwarg to add subplots
        or other `PyGMT` plotting methods.

    Example
    -------
    >>> from polartoolkit import maps, regions
    ...
    >>> fig = maps.basemap(region=regions.ross_ice_shelf)
    ...
    >>> fig.show()
    """
    kwargs = copy.deepcopy(kwargs)

    try:
        hemisphere = utils.default_hemisphere(hemisphere)
    except KeyError:
        hemisphere = None
    # if region not set, either use region of existing figure or use antarctic or
    # greenland regions
    if region is None:
        if fig is not None:
            with pygmt.clib.Session() as lib:
                region = tuple(lib.extract_region())
                assert len(region) == 4
        elif hemisphere == "north":
            region = regions.greenland
        elif hemisphere == "south":
            region = regions.antarctica
        else:
            msg = "Region must be specified if hemisphere is not specified."
            raise ValueError(msg)
    logger.debug("using %s for the basemap region", region)

    # need fig width to determine real x/y shift amounts
    _, _, _, fig_width, _ = _set_figure_spec(
        region=region,
        origin_shift="initialize",
        fig_height=kwargs.get("fig_height"),
        fig_width=kwargs.get("fig_width"),
        hemisphere=hemisphere,
    )

    # need to determine if colorbar will be plotted for setting y shift
    # only colorbar if points, and points_fill is a pd.Series
    # not a string indicating a constant color
    if points is None:
        colorbar = False
    else:
        points_fill = kwargs.get("points_fill", "black")
        if points_fill in points.columns:
            colorbar = kwargs.get("colorbar", True)
        else:
            colorbar = False

    # if currently plotting colorbar, or histogram, assume the past plot did as well and
    # account for it in the y shift
    yshift_extra = kwargs.get("yshift_extra", 0.4)
    if colorbar is True:
        # for thickness of cbar
        yshift_extra += (kwargs.get("cbar_width_perc", 0.8) * fig_width) * 0.04
        if kwargs.get("hist"):
            # for histogram thickness
            yshift_extra += kwargs.get("cbar_hist_height", 1.5)
            # for gap between cbar and map above and below
            yshift_extra += kwargs.get("cbar_yoffset", 0.2)
        else:
            # for gap between cbar and map above and below
            yshift_extra += kwargs.get("cbar_yoffset", 0.4)
        # for cbar label text
        if kwargs.get("cbar_label"):
            yshift_extra += 1
    if title is not None:
        # for title text
        yshift_extra += 1

    fig, proj, proj_latlon, fig_width, _ = _set_figure_spec(
        region=region,
        fig=fig,
        origin_shift=origin_shift,
        fig_height=kwargs.get("fig_height"),
        fig_width=kwargs.get("fig_width"),
        hemisphere=hemisphere,
        xshift_amount=kwargs.get("xshift_amount", 1),
        yshift_amount=kwargs.get("yshift_amount", -1),
        xshift_extra=kwargs.get("xshift_extra", 0.4),
        yshift_extra=yshift_extra,
    )

    show_region = kwargs.get("show_region")
    frame = kwargs.get("frame", "nesw+gwhite")
    if frame is None:
        frame = False
    if title is None:
        title = ""
    # plot basemap with optional colored background (+gwhite) and frame
    with pygmt.config(
        MAP_FRAME_PEN=kwargs.get("frame_pen", "auto"),
        FONT=kwargs.get("frame_font", "auto"),
    ):
        if frame is True:
            fig.basemap(
                region=region,
                projection=proj,
                frame=frame,
                verbose="e",
                transparency=kwargs.get("transparency", 0),
            )
        elif frame is False:
            pass
        elif isinstance(frame, list):
            fig.basemap(
                region=region,
                projection=proj,
                frame=frame,
                verbose="e",
                transparency=kwargs.get("transparency", 0),
            )
        else:
            fig.basemap(
                region=region,
                projection=proj,
                frame=frame,
                verbose="e",
                transparency=kwargs.get("transparency", 0),
            )

    with pygmt.config(FONT_TITLE=kwargs.get("title_font", "auto")):
        fig.basemap(
            region=region,
            projection=proj,
            frame=f"+t{title}",
            verbose="e",
        )

    # add satellite imagery (LIMA for Antarctica)
    if imagery_basemap is True:
        logger.debug("adding background imagery")
        add_imagery(
            fig,
            hemisphere=hemisphere,
            transparency=kwargs.get("imagery_transparency", 0),
        )

    # add MODIS imagery as basemap
    if modis_basemap is True:
        logger.debug("adding MODIS imagery")
        add_modis(
            fig,
            hemisphere=hemisphere,
            version=kwargs.get("modis_version"),
            transparency=kwargs.get("modis_transparency", 0),
        )

    # add simple basemap
    if simple_basemap is True:
        logger.debug("adding simple basemap")
        add_simple_basemap(
            fig,
            hemisphere=hemisphere,
            version=kwargs.get("simple_basemap_version"),
            transparency=kwargs.get("simple_basemap_transparency", 0),
            pen=kwargs.get("simple_basemap_pen", "0.2p,black"),
            grounded_color=kwargs.get("simple_basemap_grounded_color", "grey"),
            floating_color=kwargs.get("simple_basemap_floating_color", "skyblue"),
        )
    # add lat long grid lines
    if gridlines is True:
        logger.debug("adding gridlines")
        if hemisphere is None:
            logger.warning(
                "Argument `hemisphere` not specified, will use meters for gridlines."
            )

        add_gridlines(
            fig,
            region=region,
            projection=proj_latlon,
            x_spacing=kwargs.get("x_spacing"),
            y_spacing=kwargs.get("y_spacing"),
        )

    # plot groundingline and coastlines
    if coast is True:
        logger.debug("adding coastlines")
        add_coast(
            fig,
            hemisphere=hemisphere,
            region=region,
            projection=proj,
            pen=kwargs.get("coast_pen"),
            no_coast=kwargs.get("no_coast", False),
            version=kwargs.get("coast_version"),
            label=kwargs.get("coast_label", None),
        )

    # plot faults
    if faults is True:
        logger.debug("adding faults")
        add_faults(
            fig=fig,
            region=region,
            projection=proj,
            label=kwargs.get("fault_label"),
            pen=kwargs.get("fault_pen"),
            style=kwargs.get("fault_style"),
            fault_activity=kwargs.get("fault_activity"),
            fault_motion=kwargs.get("fault_motion"),
            fault_exposure=kwargs.get("fault_exposure"),
        )

    # add box showing region
    if show_region is not None:
        logger.debug("adding region box")
        add_box(
            fig,
            show_region,
            pen=kwargs.get("region_pen"),  # type: ignore[arg-type]
        )

    # add datapoints
    if points is not None:
        logger.debug("adding points")

        # subset points to plot region
        points = points.copy()
        points = utils.points_inside_region(
            points,
            region=region,
        )
        if ("x" in points.columns) and ("y" in points.columns):
            x_col, y_col = "x", "y"
        elif ("easting" in points.columns) and ("northing" in points.columns):
            x_col, y_col = "easting", "northing"
        else:
            msg = "points must contain columns 'x' and 'y' or 'easting' and 'northing'."
            raise ValueError(msg)
        # plot points
        if points_fill in points.columns:
            cmap, _, cpt_lims = set_cmap(
                kwargs.get("points_cmap", "viridis"),
                points=points[points_fill],
                hemisphere=hemisphere,
                **kwargs,
            )
            fig.plot(
                x=points[x_col],
                y=points[y_col],
                style=kwargs.get("points_style", "c.2c"),
                fill=points[points_fill],
                pen=kwargs.get("points_pen"),
                label=kwargs.get("points_label"),
                cmap=cmap,
            )
        else:
            fig.plot(
                x=points[x_col],
                y=points[y_col],
                style=kwargs.get("points_style", "c.2c"),
                fill=points_fill,
                pen=kwargs.get("points_pen", "1p,black"),
                label=kwargs.get("points_label"),
            )
            colorbar = False

        # display colorbar
        if colorbar is True:
            # removed duplicate kwargs before passing to add_colorbar
            cbar_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key
                not in [
                    "cpt_lims",
                    "fig_width",
                    "fig",
                ]
            }
            logger.debug("kwargs passed to 'add_colorbar': %s", cbar_kwargs)
            if cbar_kwargs.get("hist") is True:
                add_colorbar(
                    fig,
                    cmap=cmap,
                    hist_cmap=cmap,
                    grid=points[[x_col, y_col, points_fill]],
                    cpt_lims=cpt_lims,  # pylint: disable=possibly-used-before-assignment
                    region=region,
                    **cbar_kwargs,
                )
            else:
                add_colorbar(
                    fig,
                    cmap=cmap,
                    cpt_lims=cpt_lims,
                    region=region,
                    **cbar_kwargs,
                )
    # add inset map to show figure location
    if inset is True:
        # removed duplicate kwargs before passing to add_inset
        new_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in [
                "fig",
            ]
        }
        add_inset(
            fig,
            region=region,
            hemisphere=hemisphere,
            **new_kwargs,
        )

    # add scalebar
    if scalebar is True:
        if proj_latlon is None:
            msg = "Argument `hemisphere` needs to be specified for plotting a scalebar"
            raise ValueError(msg)

        scalebar_font_color = kwargs.get("scalebar_font_color", "black")
        scalebar_length_perc = kwargs.get("scalebar_length_perc", 0.25)
        scalebar_position = kwargs.get("scalebar_position", "n.5/.05")

        if kwargs.get("scale_font_color", None) is not None:
            msg = "`scale_font_color` is deprecated, use `scalebar_font_color` instead."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            scalebar_font_color = kwargs.get("scale_font_color", "black")

        if kwargs.get("scale_length_perc", None) is not None:
            msg = (
                "`scale_length_perc` is deprecated, use `scalebar_length_perc` instead."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            scalebar_length_perc = kwargs.get("scale_length_perc", 0.25)

        if kwargs.get("scale_position", None) is not None:
            msg = "`scale_position` is deprecated, use `scalebar_position` instead."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            scalebar_position = kwargs.get("scale_position", "n.5/.05")

        add_scalebar(
            fig=fig,
            region=region,
            projection=proj_latlon,
            font_color=scalebar_font_color,
            length_perc=scalebar_length_perc,
            position=scalebar_position,
            **kwargs,
        )

    # add north arrow
    if north_arrow is True:
        if proj_latlon is None:
            msg = (
                "Argument `hemisphere` needs to be specified for plotting a north arrow"
            )
            raise ValueError(msg)

        add_north_arrow(
            fig,
            region=region,
            projection=proj_latlon,
            **kwargs,
        )

    # reset region and projection
    fig.basemap(region=region, projection=proj, frame="+t")

    return fig


def set_cmap(
    cmap: str | bool,
    grid: str | xr.DataArray | None = None,
    points: pd.Series | NDArray | None = None,
    modis: bool = False,
    grd2cpt: bool = False,
    cpt_lims: tuple[float, float] | None = None,
    cmap_region: tuple[float, float, float, float] | None = None,
    robust: bool = False,
    robust_percentiles: tuple[float, float] = (0.02, 0.98),
    reverse_cpt: bool = False,
    shp_mask: gpd.GeoDataFrame | str | None = None,
    hemisphere: str | None = None,
    colorbar: bool = True,
    **kwargs: typing.Any,
) -> tuple[str | bool, bool, tuple[float, float] | None]:
    """
    Function used to set the PyGMT colormap for a figure.

    Parameters
    ----------
    cmap : str | bool
        a string of either a PyGMT cpt file (.cpt), or a preset PyGMT color ramp, or
        alternatively a value of True will use the last used cmap.
    grid : str | xarray.DataArray | None, optional
       grid used to determine colormap limits and grd2cpt colormap equalization, by
       default None
    points : pandas.Series | numpy.ndarray | None, optional
        point values to use to determine colormap limits, by default None
    modis : bool, optional
        choose appropriate cmap for plotting modis data, by default False
    grd2cpt : bool, optional
        equalized the colormap to the grid data values, by default False
    cpt_lims : tuple[float, float] | None, optional
        limits to set for the colormap, by default None
    cmap_region : tuple[float, float, float, float] | None, optional
        extract colormap limits from a subset of the grid or points, in format
        [xmin, xmax, ymin, ymax], by default None
    robust : bool, optional
        use the 2nd and 98th percentile of the data from the grid or points, by default
        False
    robust_percentiles : tuple[float, float], optional
        percentiles to use for robust colormap limits, by default (0.02, 0.98)
    reverse_cpt : bool, optional
        change the direction of the cmap, by default False
    shp_mask : geopandas.GeoDataFrame | str | None, optional
        a shapefile to mask the grid or points by before extracting limits, by default
        None
    hemisphere : str | None, optional
        "north" or "south" hemisphere needed for using shp_mask, by default None
    colorbar : bool, optional
        tell subsequent plotting functions whether to add a colorbar, by default True

    Returns
    -------
    tuple[str | bool, bool, tuple[float,float] | None]
        a tuple with the pygmt colormap, as a string or boolean, a boolean of whether to
        plot the colorbar, and a tuple of 2 floats with the cpt limits.
    """

    if (grid is not None) and (points is not None):
        msg = "Only one of `grid` or `points` can be passed to `set_cmap`."
        raise ValueError(msg)

    # set cmap
    if isinstance(cmap, str) and cmap.endswith(".cpt"):
        # skip everything if cpt file is passed
        def warn_msg(x: str) -> str:
            return f"Since a .cpt file was passed to `cmap`, parameter `{x}` is unused."

        if modis is True:
            warnings.warn(
                warn_msg("modis"),
                stacklevel=2,
            )
        if grd2cpt is True:
            warnings.warn(
                warn_msg("grd2cpt"),
                stacklevel=2,
            )
        if cpt_lims is not None:
            warnings.warn(
                warn_msg("cpt_lims"),
                stacklevel=2,
            )
        if cmap_region is not None:
            warnings.warn(
                warn_msg("cmap_region"),
                stacklevel=2,
            )
        if robust is True:
            warnings.warn(
                warn_msg("robust"),
                stacklevel=2,
            )
        if reverse_cpt is True:
            warnings.warn(
                warn_msg("reverse_cpt"),
                stacklevel=2,
            )
        if shp_mask is not None:
            warnings.warn(
                warn_msg("shp_mask"),
                stacklevel=2,
            )

    elif modis is True:
        # create a cmap to use specifically with MODIS imagery
        pygmt.makecpt(
            cmap="grayC",
            series=[15000, 17000, 1],
            verbose="e",
        )
        colorbar = False
        cmap = True

    elif grd2cpt is True:
        # gets here if
        # 1) cmap doesn't end in .cpt
        # 2) modis is False
        if grid is None:
            warnings.warn(
                "`grd2cpt` ignored since no grid was passed",
                stacklevel=2,
            )
        else:
            if cpt_lims is None and isinstance(grid, (xr.DataArray)):
                zmin, zmax = utils.get_min_max(
                    grid,
                    shp_mask,
                    region=cmap_region,
                    robust=robust,
                    hemisphere=hemisphere,
                    robust_percentiles=robust_percentiles,
                )
            elif cpt_lims is None and isinstance(grid, (str)):
                with xr.load_dataarray(grid) as da:
                    zmin, zmax = utils.get_min_max(
                        da,
                        shp_mask,
                        region=cmap_region,
                        robust=robust,
                        hemisphere=hemisphere,
                        robust_percentiles=robust_percentiles,
                    )
            else:
                if cpt_lims is None:
                    zmin, zmax = None, None
                else:
                    zmin, zmax = cpt_lims
            if cpt_lims is not None:

                def warn_msg(x: str) -> str:
                    return (
                        f"Since limits were passed to `cpt_lims`, parameter `{x}` is"
                        "unused."
                    )

                if cmap_region is not None:
                    warnings.warn(
                        warn_msg("cmap_region"),
                        stacklevel=2,
                    )
                if robust is True:
                    warnings.warn(
                        warn_msg("robust"),
                        stacklevel=2,
                    )
                if shp_mask is not None:
                    warnings.warn(
                        warn_msg("shp_mask"),
                        stacklevel=2,
                    )

            pygmt.grd2cpt(
                cmap=cmap,
                grid=grid,
                region=cmap_region,
                background=True,
                limit=(zmin, zmax),
                continuous=kwargs.get("continuous", True),
                color_model=kwargs.get("color_model", "R"),
                categorical=kwargs.get("categorical", False),
                reverse=reverse_cpt,
                verbose="e",
            )
            cmap = True
    elif cpt_lims is not None:
        # gets here if
        # 1) cmap doesn't end in .cpt
        # 2) modis is False
        # 3) grd2cpt is False
        zmin, zmax = cpt_lims

        def warn_msg(x: str) -> str:
            return f"Since limits were passed to `cpt_lims`, parameter `{x}` is unused."

        if cmap_region is not None:
            warnings.warn(
                warn_msg("cmap_region"),
                stacklevel=2,
            )
        if robust is True:
            warnings.warn(
                warn_msg("robust"),
                stacklevel=2,
            )
        if shp_mask is not None:
            warnings.warn(
                warn_msg("shp_mask"),
                stacklevel=2,
            )
        try:
            pygmt.makecpt(
                cmap=cmap,
                series=(zmin, zmax),
                background=True,
                continuous=kwargs.get("continuous", False),
                color_model=kwargs.get("color_model", "R"),
                categorical=kwargs.get("categorical", False),
                reverse=reverse_cpt,
                verbose="e",
            )
        except pygmt.exceptions.GMTCLibError as e:
            logger.exception(e)
            pygmt.makecpt(
                cmap=cmap,
                background=True,
                continuous=kwargs.get("continuous", False),
                color_model=kwargs.get("color_model", "R"),
                categorical=kwargs.get("categorical", False),
                reverse=reverse_cpt,
                verbose="e",
            )
        cmap = True
    else:
        # gets here if
        # 1) cmap doesn't end in .cpt
        # 2) modis is False
        # 3) grd2cpt is False
        # 4) cpt_lims aren't set
        try:
            if points is not None:
                values = points
            elif isinstance(grid, (xr.DataArray)):
                values = grid
            else:
                values = xr.load_dataarray(grid)
            zmin, zmax = utils.get_min_max(
                values,
                shp_mask,
                region=cmap_region,
                robust=robust,
                hemisphere=hemisphere,
                robust_percentiles=robust_percentiles,
            )
            pygmt.makecpt(
                cmap=cmap,
                background=True,
                continuous=kwargs.get("continuous", True),
                series=(zmin, zmax),
                reverse=reverse_cpt,
                verbose="e",
            )
        except (pygmt.exceptions.GMTCLibError, Exception) as e:  # pylint: disable=broad-exception-caught
            if "Option T: min >= max" in str(e):
                logger.warning("supplied min value is greater or equal to max value")
                logger.exception(e)
                pygmt.makecpt(
                    cmap=cmap,
                    background=True,
                    reverse=reverse_cpt,
                    verbose="e",
                )
            else:
                logger.exception(e)
                pygmt.makecpt(
                    cmap=cmap,
                    background=True,
                    continuous=kwargs.get("continuous", True),
                    reverse=reverse_cpt,
                    verbose="e",
                )
        cmap = True
        if zmin is None or zmax is None:  # noqa: SIM108
            cpt_lims = None
        else:
            cpt_lims = (zmin, zmax)

    return cmap, colorbar, cpt_lims


def plot_grd(
    grid: str | xr.DataArray,
    region: tuple[float, float, float, float] | None = None,
    hemisphere: str | None = None,
    cmap: str | bool = "viridis",
    coast: bool = False,
    north_arrow: bool = False,
    scalebar: bool = False,
    faults: bool = False,
    simple_basemap: bool = False,
    imagery_basemap: bool = False,
    modis_basemap: bool = False,
    title: str | None = None,
    inset: bool = False,
    points: pd.DataFrame | None = None,
    gridlines: bool = False,
    origin_shift: str | None = "initialize",
    fig: pygmt.Figure | None = None,
    **kwargs: typing.Any,
) -> pygmt.Figure:
    """
    Plot a grid (either a filename or a load dataarray) with PyGMT in a polar
    stereographic projection, and add a range of features such as coastline and
    grounding lines, inset figure location maps, background imagery, colorbar histogram,
    scalebars, gridlines and northarrows. Reuse the figure instance to either plot
    additional features on top, or shift the plot to create subplots. There are many
    keyword arguments which can either be passed along to the various functions in the
    `maps` module, or specified specifically. Kwargs can be passed directly to the
    following functions: `add_colorbar`, `add_north_arrow`, `add_scalebar`, `add_inset`,
    `set_cmap`. Other kwargs are specified below.

    Parameters
    ----------
    grid : str or xarray.DataArray
        grid file to plot, either loaded xarray.DataArray or string of the path to a
        gridded data file, such as a netCDF, geotiff or zarr file.
    region : tuple[float, float, float, float], optional
        region for the figure in format [xmin, xmax, ymin, ymax], by default is the
        extent of the input grid. If provided, the grid will be cut to this region
        before plotting.
    hemisphere : str, optional
        set whether to plot in "north" hemisphere (EPSG:3413) or "south" hemisphere
        (EPSG:3031), can be set manually, or will read from the environment variable:
        "POLARTOOLKIT_HEMISPHERE"
    cmap : str or bool, optional
        GMT color scale to use, by default 'viridis'. If True, will use the last use
        cmap from PyGMT. See available options at https://docs.generic-mapping-tools.org/6.2/cookbook/cpts.html.
    coast : bool, optional
        choose whether to plot coastline and grounding line, by default False. Version
        of shapefiles to plots depends on `hemisphere`, and can be changed with kwargs
        `coast_version`, which defaults to `BAS` for the northern hemisphere and
        `measures-v2` for the southern.
    north_arrow : bool, optional
        choose to add a north arrow to the plot, by default is False.
    scalebar : bool, optional
        choose to add a scalebar to the plot, by default is False. See `add_scalebar`
        for additional kwargs
    faults : bool, optional
        choose to plot faults on the map, by default is False
    simple_basemap: bool, optional
        choose to plot a simple basemap with floating ice colored blue and grounded ice
        colored grey.
    simple_basemap_transparency : int, optional
        transparency to use for the simple basemap, by default is 0
    simple_basemap_version : str, optional
        version of the simple basemap to plot, by default is None
    imagery_basemap : bool, optional
        choose to add a background imagery basemap, by default is False. If true, will
        use LIMA for southern hemisphere and MODIS MoG for the northern hemisphere.
    imagery_transparency : int, optional
        transparency to use for the imagery basemap, by default is 0
    modis_basemap : bool, optional
        choose to add a MODIS background imagery basemap, by default is False.
    modis_transparency : int, optional
        transparency to use for the MODIS basemap, by default is 0
    modis_version : str, optional
        version of the MODIS basemap to plot, by default is None
    title : str | None, optional
        title to add to the figure, by default is None
    inset : bool, optional
        choose to plot inset map showing figure location, by default is False
    points : pandas.DataFrame | None, optional
        points to plot on map, must contain columns 'x' and 'y' or
        'easting' and 'northing'.
    gridlines : bool, optional
        choose to plot lat/lon grid lines, by default is False
    origin_shift : str, | None, optional
        choose what to do with the plot when creating the figure. By default is
        'initialize' which will create a new figure instance. To plot additional grids
        on top of the existing figure provide a figure instance to `fig` and set
        origin_shift to None. To create subplots, provide the existing figure instance
        to `fig`, and set `origin_shift` to 'x' to add the the new plot to the right of
        previous plot, 'y' to add the new plot above the previous plot, or 'both' to add
        the new plot to the right and above the old plot. By default each of this shifts
        will be the width/height of the figure instance, this can be changed with kwargs
        `xshift_amount` and `yshift_amount`, which are in multiples of figure
        width/height.
    fig : pygmt.Figure, optional
        supply a figure instance for adding subplots or using other PyGMT plotting
        methods, by default None
    fig_height : int or float
        height in cm for figures, by default is 15cm.
    fig_width : int or float
        width in cm for figures, by default is None and is determined by fig_height and
        the projection.
    xshift_amount : int or float
        amount to shift the origin in the x direction in multiples of current figure
        instance width, by default is 1.
    yshift_amount : int or float
        amount to shift the origin in the y direction in multiples of current figure
        instance height, by default is -1.
    frame : str | bool
        GMT frame string to use for the basemap, by default is "nesw+gwhite"
    frame_pen : str
        GMT pen string to use for the frame, by default is "auto"
    frame_font : str
        GMT font string to use for the frame, by default is "auto"
    transparency : int
        transparency to use for the basemap, by default is 0
    modis : bool
        set to True if plotting MODIS data to use a nice colorscale.
    grd2cpt : bool
        use GMT module grd2cpt to set color scale from grid values, by default is False
    cpt_lims : str or tuple]
        limits to use for color scale max and min, by default is max and min of data.
    cmap_region : str or tuple[float, float, float, float]
        region to use to define color scale limits, in format [xmin, xmax, ymin, ymax],
        by default is region
    robust : bool
        use the 2nd and 98th percentile (or those specified with 'robust_percentiles')
        of the data to set color scale limits, by default is False.
    robust_percentiles : tuple[float, float]
        percentiles to use for robust colormap limits, by default is (0.02, 0.98).
    reverse_cpt : bool
        reverse the color scale, by default is False.
    shp_mask : geopandas.GeoDataFrame | str
        shapefile to use to mask the grid before extracting limits, by default is None.
    colorbar : bool
        choose to add a colorbar to the plot, by default is True.
    cbar_label : str
        label to add to colorbar.
    shading : str
        GMT shading string to use for the basemap, by default is None
    grid_transparency : int
        transparency of the grid, by default is 0
    inset_position : str
        position for inset map with PyGMT syntax, by default is "jTL+jTL+o0/0"
    title_font : str
        font to use for the title, by default is 'auto'
    show_region : tuple[float, float, float, float]
        show a rectangular region on the map, in the format [xmin, xmax, ymin, ymax].
    region_pen : str
        GMT pen string to use for the region box, by default is None
    x_spacing : float
        spacing for x gridlines in degrees, by default is None
    y_spacing : float
        spacing for y gridlines in degrees, by default is None
    points_style : str
        style of points to plot in GMT format, by default is 'c.2c'.
    points_fill : str
        fill color of points, either string of color name or column name to color
        points by, by default is 'black'.
    points_pen : str
        pen color and width of points, by default is '1p,black' if constant color or
        None if using a cmap.
    points_label : str
        label to add to legend, by default is None
    points_cmap : str
        colormap to use for points, by default is None.
    scalebar_font_color : str
        color of the scalebar font, by default is 'black'.
    scale_font_color : str
        deprecated, use scalebar_font_color.
    scalebar_length_perc : float
        percentage of the min dimension of the figure region to use for the scalebar,
        by default is 0.25.
    scale_length_perc : float
        deprecated, use scalebar_length_perc.
    scalebar_position : str
        position of the scalebar on the figure, by default is 'n.5/.05' which is bottom
        center of the plot.
    scale_position : str
        deprecated, use scalebar_position.
    coast_pen : str
        GMT pen string to use for the coastlines, by default is None
    no_coast : bool
        choose to not plot coastlines, just grounding lines, by default is False
    coast_version : str
        version of coastlines to plot, by default depends on the hemisphere
    coast_label : str
        label to add to coastlines, by default is None
    fault_label : str
        label to add to faults, by default is None
    fault_pen : str
        GMT pen string to use for the faults, by default is None
    fault_style : str
        GMT style string to use for the faults, by default is None
    fault_activity : str
        column name in faults to use for activity, by default is None
    fault_motion : str
        column name in faults to use for motion, by default is None
    fault_exposure : str
        column name in faults to use for exposure, by default is None

    Returns
    -------
    pygmt.Figure
        Returns a figure object, which can be passed to the `fig` kwarg to add subplots
        or other `PyGMT` plotting methods.

    Example
    -------
    >>> from polartoolkit import maps
    ...
    >>> fig = maps.plot_grd('grid1.nc')
    >>> fig = maps.plot_grd(
    ... 'grid2.nc',
    ... origin_shift = 'x',
    ... fig = fig,
    ... )
    ...
    >>> fig.show()
    """
    if isinstance(grid, str):
        pass
    else:
        grid = grid.copy()

    if isinstance(grid, xr.Dataset):
        msg = "grid must be a DataArray, not a Dataset."
        raise ValueError(msg)
    try:
        hemisphere = utils.default_hemisphere(hemisphere)
    except KeyError:
        hemisphere = None

    warnings.filterwarnings("ignore", message="pandas.Int64Index")
    warnings.filterwarnings("ignore", message="pandas.Float64Index")

    # clip grid if region supplied
    if region is not None and isinstance(grid, xr.DataArray):
        grid = pygmt.grdcut(
            grid,
            region=region,
            verbose="q",
        )
    # if region not set, either use region of existing figure or get from grid
    if region is None:
        if fig is not None:
            with pygmt.clib.Session() as lib:
                region = tuple(lib.extract_region())
                assert len(region) == 4
        else:
            try:
                region = utils.get_grid_info(grid)[1]
            except Exception as e:  # pylint: disable=broad-exception-caught
                msg = "grid's region can't be extracted, please provide with `region`"
                raise ValueError(msg) from e

    region = typing.cast(tuple[float, float, float, float], region)
    logger.debug("using %s for the basemap region", region)

    # need fig width to determine real x/y shift amounts
    _, _, _, fig_width, _ = _set_figure_spec(
        region=region,
        origin_shift="initialize",
        fig_height=kwargs.get("fig_height"),
        fig_width=kwargs.get("fig_width"),
        hemisphere=hemisphere,
    )

    _, colorbar, _ = set_cmap(
        cmap,
        grid=grid,
        hemisphere=hemisphere,
        **kwargs,
    )

    # if currently plotting colorbar, or histogram, assume the past plot did as well and
    # account for it in the y shift
    yshift_extra = kwargs.get("yshift_extra", 0.4)
    if colorbar is True:
        # for thickness of cbar
        yshift_extra += (kwargs.get("cbar_width_perc", 0.8) * fig_width) * 0.04
        if kwargs.get("hist"):
            # for histogram thickness
            yshift_extra += kwargs.get("cbar_hist_height", 1.5)
            # for gap between cbar and map above and below
            yshift_extra += kwargs.get("cbar_yoffset", 0.2)
        else:
            # for gap between cbar and map above and below
            yshift_extra += kwargs.get("cbar_yoffset", 0.4)
        # for cbar label text
        if kwargs.get("cbar_label"):
            yshift_extra += 1
    if title is not None:
        # for title text
        yshift_extra += 1

    fig, proj, proj_latlon, fig_width, _ = _set_figure_spec(
        region=region,
        fig=fig,
        origin_shift=origin_shift,
        fig_height=kwargs.get("fig_height"),
        fig_width=kwargs.get("fig_width"),
        hemisphere=hemisphere,
        xshift_amount=kwargs.get("xshift_amount", 1),
        yshift_amount=kwargs.get("yshift_amount", -1),
        xshift_extra=kwargs.get("xshift_extra", 0.4),
        yshift_extra=yshift_extra,
    )

    show_region = kwargs.get("show_region")
    frame = kwargs.get("frame", "nesw+gwhite")
    if frame is None:
        frame = False
    if title is None:
        title = ""
    # plot basemap with optional colored background (+gwhite) and frame
    with pygmt.config(
        MAP_FRAME_PEN=kwargs.get("frame_pen", "auto"),
        FONT=kwargs.get("frame_font", "auto"),
    ):
        logger.debug("adding blank basemap")
        if frame is True:
            fig.basemap(
                region=region,
                projection=proj,
                frame=frame,
                verbose="e",
                transparency=kwargs.get("transparency", 0),
            )
        elif frame is False:
            pass
        elif isinstance(frame, list):
            fig.basemap(
                region=region,
                projection=proj,
                frame=frame,
                verbose="e",
                transparency=kwargs.get("transparency", 0),
            )
        else:
            fig.basemap(
                region=region,
                projection=proj,
                frame=frame,
                verbose="e",
                transparency=kwargs.get("transparency", 0),
            )

    with pygmt.config(FONT_TITLE=kwargs.get("title_font", "auto")):
        fig.basemap(
            region=region,
            projection=proj,
            frame=f"+t{title}",
            verbose="e",
        )

    # add satellite imagery (LIMA for Antarctica)
    if imagery_basemap is True:
        logger.debug("adding background imagery")
        add_imagery(
            fig,
            hemisphere=hemisphere,
            transparency=kwargs.get("imagery_transparency", 0),
        )
    # add MODIS imagery as basemap
    if modis_basemap is True:
        logger.debug("adding MODIS imagery")
        add_modis(
            fig,
            hemisphere=hemisphere,
            version=kwargs.get("modis_version"),
            transparency=kwargs.get("modis_transparency", 0),
        )
    # add simple basemap
    if simple_basemap is True:
        logger.debug("adding simple basemap")
        add_simple_basemap(
            fig,
            hemisphere=hemisphere,
            version=kwargs.get("simple_basemap_version"),
            transparency=kwargs.get("simple_basemap_transparency", 0),
            pen=kwargs.get("simple_basemap_pen", "0.2p,black"),
            grounded_color=kwargs.get("simple_basemap_grounded_color", "grey"),
            floating_color=kwargs.get("simple_basemap_floating_color", "skyblue"),
        )

    shading = kwargs.get("shading")
    if shading is not None:  # noqa: SIM108
        nan_transparent = False
    else:
        nan_transparent = True

    cmap, colorbar, cpt_lims = set_cmap(
        cmap,
        grid=grid,
        hemisphere=hemisphere,
        **kwargs,
    )
    # display grid
    logger.debug("plotting grid")
    fig.grdimage(
        grid=grid,
        cmap=cmap,
        projection=proj,
        region=region,
        nan_transparent=nan_transparent,
        frame=kwargs.get("frame"),
        shading=shading,
        transparency=kwargs.get("grid_transparency", 0),
    )

    # add datapoints
    if points is not None:
        logger.debug("adding points")

        # subset points to plot region
        points = points.copy()
        points = utils.points_inside_region(
            points,
            region=region,
        )
        if ("x" in points.columns) and ("y" in points.columns):
            x_col, y_col = "x", "y"
        elif ("easting" in points.columns) and ("northing" in points.columns):
            x_col, y_col = "easting", "northing"
        else:
            msg = "points must contain columns 'x' and 'y' or 'easting' and 'northing'."
            raise ValueError(msg)
        if kwargs.get("points_cmap") is not None:
            msg = "`points_cmap` is ignored since grid's cmap is being used."
            logger.warning(msg)
        # plot points
        points_fill = kwargs.get("points_fill", "black")
        if points_fill in points.columns:
            fig.plot(
                x=points[x_col],
                y=points[y_col],
                style=kwargs.get("points_style", "c.2c"),
                fill=points[points_fill],
                pen=kwargs.get("points_pen"),
                label=kwargs.get("points_label"),
                cmap=cmap,
            )
        else:
            fig.plot(
                x=points[x_col],
                y=points[y_col],
                style=kwargs.get("points_style", "c.2c"),
                fill=points_fill,
                pen=kwargs.get("points_pen", "1p,black"),
                label=kwargs.get("points_label"),
            )

    # add box showing region
    if show_region is not None:
        logger.debug("adding region box")
        add_box(
            fig,
            show_region,
            pen=kwargs.get("region_pen"),  # type: ignore[arg-type]
        )

    # plot groundingline and coastlines
    if coast is True:
        logger.debug("adding coastlines")
        add_coast(
            fig,
            hemisphere=hemisphere,
            region=region,
            projection=proj,
            pen=kwargs.get("coast_pen"),
            no_coast=kwargs.get("no_coast", False),
            version=kwargs.get("coast_version"),
            label=kwargs.get("coast_label"),
        )

    # plot faults
    if faults is True:
        logger.debug("adding faults")
        add_faults(
            fig=fig,
            region=region,
            projection=proj,
            label=kwargs.get("fault_label"),
            pen=kwargs.get("fault_pen"),
            style=kwargs.get("fault_style"),
            fault_activity=kwargs.get("fault_activity"),
            fault_motion=kwargs.get("fault_motion"),
            fault_exposure=kwargs.get("fault_exposure"),
        )

    # add lat long grid lines
    if gridlines is True:
        logger.debug("adding gridlines")
        if hemisphere is None:
            logger.warning(
                "Argument `hemisphere` not specified, will use meters for gridlines."
            )

        add_gridlines(
            fig,
            region=region,
            projection=proj_latlon,
            x_spacing=kwargs.get("x_spacing"),
            y_spacing=kwargs.get("y_spacing"),
        )

    # add inset map to show figure location
    if inset is True:
        logger.debug("adding inset")
        # removed duplicate kwargs before passing to add_inset
        new_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in [
                "fig",
            ]
        }
        add_inset(
            fig,
            region=region,
            hemisphere=hemisphere,
            **new_kwargs,
        )

    # add scalebar
    if scalebar is True:
        logger.debug("adding scalebar")
        if proj_latlon is None:
            msg = "Argument `hemisphere` needs to be specified for plotting a scalebar"
            raise ValueError(msg)

        scalebar_font_color = kwargs.get("scalebar_font_color", "black")
        scalebar_length_perc = kwargs.get("scalebar_length_perc", 0.25)
        scalebar_position = kwargs.get("scalebar_position", "n.5/.05")

        if kwargs.get("scale_font_color") is not None:
            msg = "`scale_font_color` is deprecated, use `scalebar_font_color` instead."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            scalebar_font_color = kwargs.get("scale_font_color", "black")

        if kwargs.get("scale_length_perc") is not None:
            msg = (
                "`scale_length_perc` is deprecated, use `scalebar_length_perc` instead."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            scalebar_length_perc = kwargs.get("scale_length_perc", 0.25)

        if kwargs.get("scale_position") is not None:
            msg = "`scale_position` is deprecated, use `scalebar_position` instead."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            scalebar_position = kwargs.get("scale_position", "n.5/.05")

        add_scalebar(
            fig=fig,
            region=region,
            projection=proj_latlon,
            font_color=scalebar_font_color,
            length_perc=scalebar_length_perc,
            position=scalebar_position,
            **kwargs,
        )

    # add north arrow
    if north_arrow is True:
        logger.debug("adding north arrow")
        if proj_latlon is None:
            msg = (
                "Argument `hemisphere` needs to be specified for plotting a north arrow"
            )
            raise ValueError(msg)

        add_north_arrow(
            fig,
            region=region,
            projection=proj_latlon,
            **kwargs,
        )

    # display colorbar
    if colorbar is True:
        logger.debug("adding colorbar")
        # removed duplicate kwargs before passing to add_colorbar
        cbar_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in [
                "cpt_lims",
                "grid",
                "fig",
            ]
        }
        try:
            add_colorbar(
                fig,
                hist_cmap=cmap,
                grid=grid,
                cpt_lims=cpt_lims,
                region=region,
                **cbar_kwargs,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception(e)
            logger.error("error with plotting colorbar, skipping")

    logger.debug("plotting complete, resetting projection and region")
    # reset region and projection
    fig.basemap(
        region=region,
        projection=proj,
        frame="+t",
    )

    return fig


def add_colorbar(
    fig: pygmt.Figure,
    hist: bool = False,
    cpt_lims: tuple[float, float] | None = None,
    cbar_frame: list[str] | str | None = None,
    verbose: str = "w",
    **kwargs: typing.Any,
) -> None:
    """
    Add a colorbar based on the last cmap used by PyGMT and optionally a histogram of
    the data values.

    Parameters
    ----------
    fig : pygmt.Figure
        pygmt figure instance to add to
    hist : bool, optional
        choose whether to add a colorbar histogram, by default False
    cpt_lims : tuple[float, float], optional
        cpt lims to use for the colorbar histogram, must match those used to create the
        colormap. If not supplied, will attempt to get values from kwargs `grid`, by
        default None
    cbar_frame : list[str] | str, optional
        frame for the colorbar, by default None
    verbose : str, optional
        verbosity level for pygmt, by default "w" for warnings
    **kwargs : typing.Any
        additional keyword arguments to pass
    """
    logger.debug("kwargs supplied to 'add_colorbar': %s", kwargs)

    # get the current figure width
    fig_width = utils.get_fig_width()

    # set colorbar width as percentage of total figure width
    cbar_width_perc = kwargs.get("cbar_width_perc", 0.8)

    # offset colorbar vertically from plot by 0.4cm, or 0.2 + histogram height
    if hist is True:
        cbar_hist_height = kwargs.get("cbar_hist_height", 1.5)
        cbar_yoffset = kwargs.get("cbar_yoffset", 0.2 + cbar_hist_height)
    else:
        cbar_yoffset = kwargs.get("cbar_yoffset", 0.4)
    logger.debug("offset cbar vertically by %s", cbar_yoffset)

    if cbar_frame is None:
        cbar_frame = [
            f"pxaf+l{kwargs.get('cbar_label',' ')}",
            f"+u{kwargs.get('cbar_unit_annot',' ')}",
            f"py+l{kwargs.get('cbar_unit',' ')}",
        ]

    # vertical or horizontal colorbar
    orientation = kwargs.get("cbar_orientation", "h")

    # text location
    text_location = kwargs.get("cbar_text_location")

    # add colorbar
    logger.debug("adding colorbar")
    with pygmt.config(
        FONT=kwargs.get("cbar_font", "12p,Helvetica,black"),
    ):
        position = (
            f"jBC+jTC+w{fig_width*cbar_width_perc}c+{orientation}{text_location}"
            f"+o{kwargs.get('cbar_xoffset', 0)}c/{cbar_yoffset}c+e"
        )
        logger.debug("cbar frame; %s", cbar_frame)
        logger.debug("cbar position: %s", position)

        fig.colorbar(
            cmap=kwargs.get("cmap", True),
            position=position,
            frame=cbar_frame,
            scale=kwargs.get("cbar_scale", 1),
            log=kwargs.get("cbar_log"),
            # verbose=verbose, # this is causing issues
        )
        logger.debug("finished standard colorbar plotting")
    # add histogram to colorbar
    # Note, depending on data and hist_type, you may need to manually set kwarg
    # `hist_ymax` to an appropriate value
    if hist is True:
        logger.debug("adding histogram to colorbar")
        # get values to use
        values = kwargs.get("grid")

        hist_cmap = kwargs.get("hist_cmap", True)

        if values is None:
            msg = "if hist is True, grid must be provided."
            raise ValueError(msg)

        # define plot region
        region = kwargs.get("region")
        # if no region supplied, get region of current PyGMT figure
        if region is None:
            with pygmt.clib.Session() as lib:
                region = tuple(lib.extract_region())
                assert len(region) == 4

        logger.debug("using histogram region: %s", region)
        # clip values to plot region
        if isinstance(values, (xr.DataArray | str)):
            if region != utils.get_grid_info(values)[1]:
                values_clipped = utils.subset_grid(values, region)
                # if subplotting, region will be in figure units and grid will be
                # clipped incorrectly, hacky solution is to check if clipped figure is
                # smaller than a few data points, if so, use grids full region
                if len(values_clipped[list(values_clipped.sizes.keys())[0]].values) < 5:  # noqa: RUF015
                    reg = kwargs.get("region")
                    if reg is None:
                        msg = (
                            "Issue with detecting figure region for adding colorbar "
                            "histogram, please provide region kwarg."
                        )
                        raise ValueError(msg)
                    values_clipped = utils.subset_grid(values, reg)
                values = values_clipped
                logger.debug("clipped grid to region")

        elif isinstance(values, pd.DataFrame):  # type: ignore[unreachable]
            values_clipped = utils.points_inside_region(values, region)
            # if subplotting, region will be in figure units and points will be clipped
            # incorrectly, hacky solution is to check if clipped figure is smaller than
            # a few data points, if so, use points full region
            if len(values_clipped) < 5:
                reg = kwargs.get("region")
                if reg is None:
                    msg = (
                        "Issue with detecting figure region for adding colorbar "
                        "histogram, please provide region kwarg."
                    )
                    raise ValueError(msg)
                values_clipped = utils.points_inside_region(values, reg)
            values = values_clipped
            logger.debug("clipped points to region")

        if isinstance(hist_cmap, str) and hist_cmap.endswith(".cpt"):
            # extract cpt_lims from cmap
            p = pathlib.Path(hist_cmap)
            with p.open(encoding="utf-8") as cptfile:
                # read the lines into memory
                lows, highs = [], []
                for x in cptfile:
                    line = x.strip()

                    # skip empty lines
                    if not line:
                        continue

                    # skip other comments
                    if line.startswith("#"):
                        continue

                    # skip BFN info
                    if line.startswith(("B", "F", "N")):
                        continue

                    # split at tabs
                    split = line.split("\t")
                    lows.append(float(split[0]))
                    highs.append(float(split[2]))

                zmin, zmax = min(lows), max(highs)
                cpt_lims = (zmin, zmax)

        elif (cpt_lims is None) or (np.isnan(cpt_lims).any()):
            warnings.warn(
                "getting max/min values from grid/points, if cpt_lims were used to "
                "create the colorscale, histogram will not properly align with "
                "colorbar!",
                stacklevel=2,
            )
            zmin, zmax = utils.get_min_max(
                values,
                shapefile=kwargs.get("shp_mask"),
                region=kwargs.get("cmap_region"),
                robust=kwargs.get("robust", False),
                hemisphere=kwargs.get("hemisphere"),
                robust_percentiles=kwargs.get("robust_percentiles", (0.02, 0.98)),
            )
        else:
            zmin, zmax = cpt_lims
        logger.debug("using %s, %s for histogram limits", zmin, zmax)

        # get grid's/point's data for histogram
        logger.debug("subsetting histogram data")
        if isinstance(values, xr.DataArray):
            df = vd.grid_to_table(values)
        elif isinstance(values, pd.DataFrame):
            df = values
        else:
            df = values
        df2 = df.iloc[:, -1:].squeeze()

        # subset data between cbar min and max
        data = df2[df2.between(zmin, zmax)]

        bin_width = kwargs.get("hist_bin_width")
        bin_num = kwargs.get("hist_bin_num", 50)

        logger.debug("calculating bin widths; %s", bin_width)
        if bin_width is not None:
            # if bin width is set, will plot x amount of bins of width=bin_width
            bins = np.arange(zmin, zmax, step=bin_width)
        else:
            # if bin width isn't set, will plot bin_num of bins, by default = 100
            bins, bin_width = np.linspace(zmin, zmax, num=bin_num, retstep=True)

        # set hist type
        hist_type = kwargs.get("hist_type", 0)

        logger.debug("generating bin data for histogram")
        if hist_type == 0:
            # if histogram type is counts
            bins = np.histogram(data, bins=bins)[0]
            max_bin_height = bins.max()
        elif hist_type == 1:
            # if histogram type is frequency percent
            bins = np.histogram(
                data,
                density=True,
                bins=bins,
            )[0]
            max_bin_height = bins.max() / bins.sum() * 100
        else:
            msg = "hist_type must be 0 or 1"
            raise ValueError(msg)

        if zmin == zmax:
            msg = "Grid/points are a constant value, can't make a colorbar histogram!"
            logger.warning(msg)
            return

        # define histogram region
        hist_reg = [
            zmin,
            zmax,
            kwargs.get("hist_ymin", 0),
            kwargs.get("hist_ymax", max_bin_height * 1.1),
        ]
        logger.debug("defined hist reg; %s", hist_reg)
        # shift figure to line up with top left of cbar
        xshift = kwargs.get("cbar_xoffset", 0) + ((1 - cbar_width_perc) * fig_width) / 2
        try:
            fig.shift_origin(xshift=f"{xshift}c", yshift=f"{-cbar_yoffset}c")
            logger.debug("shifting origin")
        except pygmt.exceptions.GMTCLibError as e:
            logger.warning(e)
            logger.warning("issue with plotting histogram, skipping...")

        # plot histograms above colorbar
        try:
            logger.debug("plotting histogram")
            fig.histogram(
                data=data,
                projection=f"X{fig_width*cbar_width_perc}c/{cbar_hist_height}c",
                region=hist_reg,
                frame=kwargs.get("hist_frame", False),
                cmap=hist_cmap,
                fill=kwargs.get("hist_fill"),
                pen=kwargs.get("hist_pen", "default"),
                barwidth=kwargs.get("hist_barwidth"),
                center=kwargs.get("hist_center", False),
                distribution=kwargs.get("hist_distribution", False),
                cumulative=kwargs.get("hist_cumulative", False),
                extreme=kwargs.get("hist_extreme", "b"),
                stairs=kwargs.get("hist_stairs", False),
                series=f"{zmin}/{zmax}/{bin_width}",
                histtype=hist_type,
                verbose=verbose,
            )
        except pygmt.exceptions.GMTCLibError as e:
            logger.warning(e)
            logger.warning("issue with plotting histogram, skipping...")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception("An error occurred: %s", e)

        # shift figure back
        try:
            fig.shift_origin(xshift=f"{-xshift}c", yshift=f"{cbar_yoffset}c")
        except pygmt.exceptions.GMTCLibError as e:
            logger.warning(e)
            logger.warning("issue with plotting histogram, skipping...")
        logger.debug("finished plotting histogram")


def add_coast(
    fig: pygmt.Figure,
    hemisphere: str | None = None,
    region: tuple[float, float, float, float] | None = None,
    projection: str | None = None,
    no_coast: bool = False,
    pen: str | None = None,
    version: str | None = None,
    label: str | None = None,
) -> None:
    """
    add coastline and or groundingline to figure.

    Parameters
    ----------
    fig : pygmt.Figure
    hemisphere : str, optional
        choose between plotting in the "north" or "south" hemispheres
    region : tuple[float, float, float, float], optional
        region for the figure in format [xmin, xmax, ymin, ymax], if not provided will
        try to extract from the current figure.
    projection : str, optional
        GMT projection string, by default is last used by PyGMT
    no_coast : bool
        If True, only plot groundingline, not coastline, by default is False
    pen : None
        GMT pen string, by default "0.6p,black"
    version : str, optional
        version of groundingline to plot, by default is 'BAS' for north hemisphere and
        'measures-v2' for south hemisphere
    label : str, optional
        label to add to the legend, by default is None
    """
    try:
        hemisphere = utils.default_hemisphere(hemisphere)
    except KeyError:
        hemisphere = None

    if pen is None:
        pen = "0.6p,black"

    if version is None:
        if hemisphere == "north":
            version = "BAS"
        elif hemisphere == "south":
            version = "measures-v2"
        elif hemisphere is None:
            msg = "if version is not provided, must provide hemisphere"
            raise ValueError(msg)
        else:
            msg = "hemisphere must be either north or south"
            raise ValueError(msg)

    if version == "depoorter-2013":
        if no_coast is False:
            data = fetch.groundingline(version=version)
        elif no_coast is True:
            gdf = gpd.read_file(fetch.groundingline(version=version), engine=ENGINE)
            data = gdf[gdf.Id_text == "Grounded ice or land"]
    elif version == "measures-v2":
        if no_coast is False:
            gl = gpd.read_file(fetch.groundingline(version=version), engine=ENGINE)
            coast = gpd.read_file(
                fetch.antarctic_boundaries(version="Coastline"), engine=ENGINE
            )
            data = pd.concat([gl, coast])
        elif no_coast is True:
            data = fetch.groundingline(version=version)
    elif version in ("BAS", "measures-greenland"):
        data = fetch.groundingline(version=version)
    else:
        msg = "invalid version string"
        raise ValueError(msg)

    fig.plot(
        data,  # pylint: disable=used-before-assignment
        projection=projection,
        region=region,
        pen=pen,
        label=label,
    )


def add_gridlines(
    fig: pygmt.Figure,
    region: tuple[float, float, float, float] | None = None,
    projection: str | None = None,
    x_spacing: float | None = None,
    y_spacing: float | None = None,
    annotation_offset: str = "20p",
) -> None:
    """
    add lat lon grid lines and annotations to a figure. Use kwargs x_spacing and
    y_spacing to customize the interval of gridlines and annotations.

    Parameters
    ----------
    fig : pygmt.Figure
    region : tuple[float, float, float, float], optional
        region for the figure in format [xmin, xmax, ymin, ymax], if not provided will
        try to extract from the current figure.
    projection : str, optional
        GMT projection string in lat lon, if your previous pygmt.Figure call used a
        cartesian projection, you will need to provide a projection in lat/lon here, use
        utils.set_proj() to make this projection.
    x_spacing : float, optional
        spacing for x gridlines in degrees, by default is None
    y_spacing : float, optional
        spacing for y gridlines in degrees, by default is None
    annotation_offset : str, optional
        offset for gridline annotations, by default "20p"
    """

    # if no region supplied, get region of current PyGMT figure
    if region is None:
        with pygmt.clib.Session() as lib:
            region = tuple(lib.extract_region())
            assert len(region) == 4

    region_converted = (*region, "+ue")  # codespell:ignore ue

    if x_spacing is None:
        x_frames = ["xag", "xa"]
    else:
        x_frames = [
            f"xa{x_spacing*2}g{x_spacing}",
            f"xa{x_spacing*2}",
        ]

    if y_spacing is None:
        y_frames = ["yag", "ya"]
    else:
        y_frames = [
            f"ya{y_spacing*2}g{y_spacing}",
            f"ya{y_spacing*2}",
        ]

    with pygmt.config(
        MAP_ANNOT_OFFSET_PRIMARY=annotation_offset,  # move annotations in/out
        MAP_ANNOT_MIN_ANGLE=0,
        MAP_ANNOT_MIN_SPACING="auto",
        MAP_FRAME_TYPE="inside",
        MAP_ANNOT_OBLIQUE="anywhere",
        FONT_ANNOT_PRIMARY="8p,black,-=2p,white",
        MAP_GRID_PEN_PRIMARY="auto,gray",
        MAP_TICK_LENGTH_PRIMARY="auto",
        MAP_TICK_PEN_PRIMARY="auto,gray",
    ):
        # plot semi-transparent lines and annotations with black font and white shadow
        fig.basemap(
            projection=projection,
            region=region_converted,
            frame=[
                "NSWE",
                x_frames[0],
                y_frames[0],
            ],
            transparency=50,
        )
        # re-plot annotations with no transparency
        with pygmt.config(FONT_ANNOT_PRIMARY="8p,black"):
            fig.basemap(
                projection=projection,
                region=region_converted,
                frame=[
                    "NSWE",
                    x_frames[0],
                    y_frames[0],
                ],
            )


def add_faults(
    fig: pygmt.Figure,
    region: tuple[float, float, float, float] | None = None,
    projection: str | None = None,
    fault_activity: str | None = None,
    fault_motion: str | None = None,
    fault_exposure: str | None = None,
    pen: str | None = None,
    style: str | None = None,
    label: str | None = None,
) -> None:
    """
    add various types of faults from GeoMap to a map, from
    :footcite:t:`coxcontinentwide2023` and :footcite:t:`coxgeomap2023`

    Parameters
    ----------
    fig : pygmt.Figure
    region : tuple[float, float, float, float], optional
        region for the figure in format [xmin, xmax, ymin, ymax], if not provided will
        try to extract from the current figure.
    projection : str, optional
        GMT projection string in lat lon, if your previous pygmt.Figure call used a
        cartesian projection, you will need to provide a projection in lat/lon here, use
        utils.set_proj() to make this projection.
    fault_activity : str, optional
        type of fault activity, options are active or inactive, by default both
    fault_motion : str, optional
        type of fault motion, options are sinistral, dextral, normal, or reverse, by
        default all
    fault_exposure : str, optional
        type of fault exposure, options are exposed or inferred, by default both
    pen : str, optional
        GMT pen string, by default "1p,magenta,-"
    style : str, optional
        GMT style string, by default None
    label : str, optional
        label to add to the legend, by default None
    """

    # if no region supplied, get region of current PyGMT figure
    if region is None:
        with pygmt.clib.Session() as lib:
            region = tuple(lib.extract_region())
            assert len(region) == 4

    faults = fetch.geomap(version="faults", region=region)

    legend_label = "Fault types: "

    # subset by activity type (active or inactive)
    if fault_activity is None:
        legend_label = legend_label + "active and inactive"
    elif fault_activity == "active":
        faults = faults[faults.ACTIVITY.isin(["active", "possibly active"])]
        legend_label = legend_label + "active"
    elif fault_activity == "inactive":
        faults = faults[faults.ACTIVITY.isin(["inactive", "probably inactive"])]
        legend_label = legend_label + "inactive"

    # subset by motion type
    if fault_motion is None:
        legend_label = legend_label + " / all motion types"
    elif fault_motion == "sinistral":  # left lateral
        faults = faults[faults.TYPENAME.isin(["sinistral strike slip fault"])]
        legend_label = legend_label + ", sinistral"
        # if style is None:
        #     #f for front,
        #     # -1 for 1 arrow,
        #     # .3c for size of arrow,
        #     # +r for left side,
        #     # +s45 for arrow angle
        #     style = 'f-1c/.3c+r+s45'
    elif fault_motion == "dextral":  # right lateral
        faults = faults[faults.TYPENAME.isin(["dextral strike slip fault"])]
        legend_label = legend_label + " / dextral"
        # if style is None:
        #     style = 'f-1c/.3c+l+s45'
    elif fault_motion == "normal":
        faults = faults[
            faults.TYPENAME.isin(["normal fault", "high angle normal fault"])
        ]
        legend_label = legend_label + " / normal"
    elif fault_motion == "reverse":
        faults = faults[faults.TYPENAME.isin(["thrust fault", "high angle reverse"])]
        legend_label = legend_label + " / reverse"

    # subset by exposure type
    if fault_exposure is None:
        legend_label = legend_label + " / exposed and inferred"
    elif fault_exposure == "exposed":
        faults = faults[faults.EXPOSURE.isin(["exposed"])]
        legend_label = legend_label + " / exposed"
    elif fault_exposure == "inferred":
        faults = faults[faults.EXPOSURE.isin(["concealed", "unknown"])]
        legend_label = legend_label + " / inferred"

    if pen is None:
        pen = "1p,magenta,-"

    # if no subsetting of faults, shorten the label
    if all(x is None for x in [fault_activity, fault_motion, fault_exposure]):
        legend_label = "Faults"

    # if label supplied, use that
    if label is None:
        label = legend_label

    fig.plot(
        faults, projection=projection, region=region, pen=pen, label=label, style=style
    )


def add_imagery(
    fig: pygmt.Figure,
    hemisphere: str | None = None,
    transparency: int = 0,
) -> None:
    """
    Add satellite imagery to a figure. For southern hemisphere uses LIMA imagery, but
    for northern hemisphere uses MODIS imagery.

    Parameters
    ----------
    fig : pygmt.Figure
        PyGMT figure instance to add to
    hemisphere : str | None, optional
        hemisphere to get data for, by default None
    transparency : int, optional
        transparency of the imagery, by default 0
    """

    hemisphere = utils.default_hemisphere(hemisphere)

    if hemisphere == "north":
        image = fetch.modis(version="500m", hemisphere="north")
        cmap, _, _ = set_cmap(
            True,
            modis=True,
        )
    elif hemisphere == "south":
        image = fetch.imagery()
        cmap = None
    else:
        msg = "hemisphere must be north or south"
        raise ValueError(msg)

    fig.grdimage(
        grid=image,
        cmap=cmap,
        transparency=transparency,
    )


def add_modis(
    fig: pygmt.Figure,
    hemisphere: str | None = None,
    version: str | None = None,
    transparency: int = 0,
) -> None:
    """
    Add MODIS imagery to a figure.

    Parameters
    ----------
    fig : pygmt.Figure
        PyGMT figure instance to add to
    hemisphere : str | None, optional
        hemisphere to get MODIS data for, by default None
    version : str | None, optional
        which version (resolution) of MODIS imagery to use, by default "750m" for
        southern hemisphere and "500m" for northern hemisphere.
    transparency : int, optional
        transparency of the MODIS imagery, by default 0
    """

    hemisphere = utils.default_hemisphere(hemisphere)

    if hemisphere == "north":
        if version is None:
            version = "500m"
    elif hemisphere == "south":
        if version is None:
            version = "750m"
    else:
        msg = "hemisphere must be north or south"
        raise ValueError(msg)

    image = fetch.modis(version=version, hemisphere=hemisphere)

    imagery_cmap, _, _ = set_cmap(
        True,
        modis=True,
    )
    fig.grdimage(
        grid=image,
        cmap=imagery_cmap,
        transparency=transparency,
    )


def add_simple_basemap(
    fig: pygmt.Figure,
    hemisphere: str | None = None,
    version: str | None = None,
    transparency: int = 0,
    pen: str = "0.2p,black",
    grounded_color: str = "grey",
    floating_color: str = "skyblue",
) -> None:
    """
    Add a simple basemap to a figure with grounded ice shown as grey and floating ice as
    blue.

    Parameters
    ----------
    fig : pygmt.Figure
        PyGMT figure instance to add to
    hemisphere : str | None, optional
        hemisphere to get coastline data for, by default None
    version : str | None, optional
        which version of shapefiles to use for grounding line / coastline, by default
        "measures-v2" for southern hemisphere and "BAS" for northern hemisphere
    transparency : int, optional
        transparency of all the plotted elements, by default 0
    pen : str, optional
        GMT pen string for the coastline, by default "0.2,black"
    grounded_color : str, optional
        color for the grounded ice, by default "grey"
    floating_color : str, optional
        color for the floating ice, by default "skyblue"
    """

    hemisphere = utils.default_hemisphere(hemisphere)

    if hemisphere == "north":
        if version is None:
            version = "BAS"

        if version == "BAS":
            gdf = gpd.read_file(fetch.groundingline("BAS"), engine=ENGINE)
            fig.plot(
                data=gdf,
                fill=grounded_color,
                transparency=transparency,
            )
            fig.plot(
                data=gdf,
                pen=pen,
                transparency=transparency,
            )
        else:
            msg = "version must be BAS for northern hemisphere"
            raise ValueError(msg)

    elif hemisphere == "south":
        if version is None:
            version = "measures-v2"

        if version == "depoorter-2013":
            gdf = gpd.read_file(fetch.groundingline("depoorter-2013"), engine=ENGINE)
            # plot floating ice as blue
            fig.plot(
                data=gdf[gdf.Id_text == "Ice shelf"],
                fill=floating_color,
                transparency=transparency,
            )
            # plot grounded ice as gray
            fig.plot(
                data=gdf[gdf.Id_text == "Grounded ice or land"],
                fill=grounded_color,
                transparency=transparency,
            )
            # plot coastline on top
            fig.plot(
                data=gdf,
                pen=pen,
                transparency=transparency,
            )
        elif version == "measures-v2":
            fig.plot(
                data=fetch.antarctic_boundaries(version="Coastline"),
                fill=floating_color,
                transparency=transparency,
            )
            fig.plot(
                data=fetch.groundingline(version="measures-v2"),
                fill=grounded_color,
                transparency=transparency,
            )
            fig.plot(
                fetch.groundingline(version="measures-v2"),
                pen=pen,
                transparency=transparency,
            )

    else:
        msg = "hemisphere must be north or south"
        raise ValueError(msg)


def add_inset(
    fig: pygmt.Figure,
    hemisphere: str | None = None,
    region: tuple[float, float, float, float] | None = None,
    inset_position: str = "jTL+jTL+o0/0",
    inset_width: float = 0.25,
    inset_reg: tuple[float, float, float, float] | None = None,
    **kwargs: typing.Any,
) -> None:
    """
    add an inset map showing the figure region relative to the Antarctic continent.

    Parameters
    ----------
    fig : pygmt.Figure
    hemisphere : str, optional
        choose between plotting in the "north" or "south" hemispheres
    region : tuple[float, float, float, float], optional
        region for the figure in format [xmin, xmax, ymin, ymax], if not provided will
        try to extract from the current figure.
    inset_position : str, optional
        GMT location string for inset map, by default 'jTL+jTL+o0/0' (top left)
    inset_width : float, optional
        Inset width as percentage of the smallest figure dimension, by default is 25%
        (0.25)
    inset_reg : tuple[float, float, float, float], optional
        Region of Antarctica/Greenland to plot for the inset map, by default is whole
        area
    """
    hemisphere = utils.default_hemisphere(hemisphere)

    if kwargs.get("inset_pos") is not None:
        inset_position = kwargs.get("inset_pos")  # type: ignore[assignment]
        msg = "inset_pos is deprecated, use inset_position instead"
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    if kwargs.get("inset_offset") is not None:
        inset_position = inset_position + f"+o{kwargs.get('inset_offset')}"
        msg = (
            "inset_offset is deprecated, add offset via '+o0c/0c' to inset_position "
            "instead"
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    fig_width = utils.get_fig_width()
    fig_height = utils.get_fig_height()

    inset_width = inset_width * (min(fig_width, fig_height))
    inset_map = f"X{inset_width}c"

    # if no region supplied, get region of current PyGMT figure
    if region is None:
        with pygmt.clib.Session() as lib:
            region = tuple(lib.extract_region())
            assert len(region) == 4
    logger.debug("using region; %s", region)

    position = f"{inset_position}+w{inset_width}c"
    logger.debug("using position; %s", position)

    with fig.inset(
        position=position,
        box=kwargs.get("inset_box", False),
    ):
        if hemisphere == "north":
            if inset_reg is None:
                if "L" in inset_position[0:3]:
                    # inset reg needs to be square,
                    # if on left side, make square by adding to right side of region
                    inset_reg = (-800e3, 2000e3, -3400e3, -600e3)
                elif "R" in inset_position[0:3]:
                    inset_reg = (-1800e3, 1000e3, -3400e3, -600e3)
                else:
                    inset_reg = (-1300e3, 1500e3, -3400e3, -600e3)

            if inset_reg[1] - inset_reg[0] != inset_reg[3] - inset_reg[2]:
                logger.warning(
                    "Inset region should be square or else projection will be off."
                )
            gdf = gpd.read_file(fetch.groundingline("BAS"), engine=ENGINE)
            fig.plot(
                projection=inset_map,
                region=inset_reg,
                data=gdf,
                fill="grey",
            )
            fig.plot(
                data=gdf,
                pen=kwargs.get("inset_coast_pen", "0.2,black"),
            )
        elif hemisphere == "south":
            if inset_reg is None:
                inset_reg = regions.antarctica
            if inset_reg[1] - inset_reg[0] != inset_reg[3] - inset_reg[2]:
                logger.warning(
                    "Inset region should be square or else projection will be off."
                )
            logger.debug("plotting floating ice")
            fig.plot(
                projection=inset_map,
                region=inset_reg,
                data=fetch.antarctic_boundaries(version="Coastline"),
                fill="skyblue",
            )
            logger.debug("plotting grounded ice")
            fig.plot(
                data=fetch.groundingline(version="measures-v2"),
                fill="grey",
            )
            logger.debug("plotting coastline")
            gl = gpd.read_file(
                fetch.groundingline(version="measures-v2"),
                engine=ENGINE,
            )
            coast = gpd.read_file(
                fetch.antarctic_boundaries(version="Coastline"), engine=ENGINE
            )
            data = pd.concat([gl, coast])
            fig.plot(
                data,
                pen=kwargs.get("inset_coast_pen", "0.2,black"),
            )
        else:
            msg = "hemisphere must be north or south"
            raise ValueError(msg)
        logger.debug("add inset box")
        add_box(
            fig,
            box=region,
            pen=kwargs.get("inset_box_pen", "1p,red"),
        )
        logger.debug("inset complete")


def add_scalebar(
    fig: pygmt.Figure,
    region: tuple[float, float, float, float] | None = None,
    projection: str | None = None,
    **kwargs: typing.Any,
) -> None:
    """
    add a scalebar to a figure.

    Parameters
    ----------
    fig : pygmt.Figure
    region : tuple[float, float, float, float], optional
        region for the figure in format [xmin, xmax, ymin, ymax], if not provided will
        try to extract from the current figure.
    projection : str, optional
        GMT projection string in lat lon, if your previous pygmt.Figure call used a
        cartesian projection, you will need to provide a projection in lat/lon here, use
        utils.set_proj() to make this projection.

    """
    font_color = kwargs.get("font_color", "black")
    length = kwargs.get("length")
    length_perc = kwargs.get("length_perc", 0.25)
    position = kwargs.get("position", "n.5/.05")

    # if no region supplied, get region of current PyGMT figure
    if region is None:
        with pygmt.clib.Session() as lib:
            region = tuple(lib.extract_region())
            assert len(region) == 4

    def round_to_1(x: float) -> float:
        return round(x, -int(floor(log10(abs(x)))))

    region_converted = (*region, "+ue")  # codespell:ignore ue

    if length is None:
        length = typing.cast(float, length)
        # get shorter of east-west vs north-sides
        width = abs(region[1] - region[0])
        height = abs(region[3] - region[2])
        length = round_to_1((min(width, height)) / 1000 * length_perc)

    with pygmt.config(
        FONT_ANNOT_PRIMARY=f"10p,{font_color}",
        FONT_LABEL=f"10p,{font_color}",
        MAP_SCALE_HEIGHT="6p",
        MAP_TICK_PEN_PRIMARY=f"0.5p,{font_color}",
    ):
        fig.basemap(
            region=region_converted,
            projection=projection,
            map_scale=f"{position}+w{length}k+f+lkm+ar",
            box=kwargs.get("scalebar_box", "+gwhite"),
        )


def add_north_arrow(
    fig: pygmt.Figure,
    region: tuple[float, float, float, float] | None = None,
    projection: str | None = None,
    **kwargs: typing.Any,
) -> None:
    """
    add a north arrow to a figure

    Parameters
    ----------
    fig : pygmt.Figure
    region : tuple[float, float, float, float], optional
        region for the figure in format [xmin, xmax, ymin, ymax], if not provided will
        try to extract from the current figure.
    projection : str, optional
        GMT projection string in lat lon, if your previous pygmt.Figure call used a
        cartesian projection, you will need to provide a projection in lat/lon here, use
        utils.set_proj() to make this projection.

    """
    rose_size = kwargs.get("rose_size", "1c")

    position = kwargs.get("position", "n.1/.05")

    # if no region supplied, get region of current PyGMT figure
    if region is None:
        with pygmt.clib.Session() as lib:
            region = tuple(lib.extract_region())
            assert len(region) == 4

    region_converted = (*region, "+ue")  # codespell:ignore ue

    rose_str = kwargs.get("rose_str", f"{position}+w{rose_size}")

    fig.basemap(
        region=region_converted,
        projection=projection,
        rose=rose_str,
        box=kwargs.get("rose_box", False),
        perspective=kwargs.get("perspective", False),
    )


def add_box(
    fig: pygmt.Figure,
    box: tuple[float, float, float, float],
    pen: str = "2p,black",
    verbose: str = "w",
) -> None:
    """
    Plot a GMT region as a box.

    Parameters
    ----------
    fig : pygmt.Figure
        Figure to plot on
    box : tuple[float, float, float, float]
        region in EPSG3031 in format [xmin, xmax, ymin, ymax] in meters
    pen : str, optional
        GMT pen string used for the box, by default "2p,black"
    verbose : str, optional
        verbosity level for pygmt, by default "w" for warnings
    """
    fig.plot(
        x=[box[0], box[0], box[1], box[1], box[0]],
        y=[box[2], box[3], box[3], box[2], box[2]],
        pen=pen,
        verbose=verbose,
    )


def interactive_map(
    hemisphere: str | None = None,
    center_yx: tuple[float] | None = None,
    zoom: float = 0,
    display_xy: bool = True,
    show: bool = True,
    points: pd.DataFrame | None = None,
    basemap_type: str = "BlueMarble",
    **kwargs: typing.Any,
) -> ipyleaflet.Map:
    """
    Plot an interactive map with satellite imagery. Clicking gives the cursor location
    in a Polar Stereographic projection [x,y]. Requires ipyleaflet

    Parameters
    ----------
    hemisphere : str, optional
        choose between plotting in the "north" or "south" hemispheres
    center_yx : tuple, optional
        choose center coordinates in EPSG3031 [y,x], by default [0,0]
    zoom : float, optional
        choose zoom level, by default 0
    display_xy : bool, optional
        choose if you want clicks to show the xy location, by default True
    show : bool, optional
        choose whether to display the map, by default True
    points : pandas.DataFrame, optional
        choose to plot points supplied as columns 'x', 'y', or 'easting', 'northing', in
        EPSG:3031 in a dataframe
    basemap_type : str, optional
        choose what basemap to plot, options are 'BlueMarble', 'Imagery', 'Basemap', and
        "IceVelocity", by default 'BlueMarble'

    Returns
    -------
    ipyleaflet.Map
        interactive map
    """
    hemisphere = utils.default_hemisphere(hemisphere)

    if ipyleaflet is None:
        msg = """
            Missing optional dependency 'ipyleaflet' required for interactive plotting.
        """
        raise ImportError(msg)

    if ipywidgets is None:
        msg = """
            Missing optional dependency 'ipywidgets' required for interactive plotting.
        """
        raise ImportError(msg)

    if display is None:
        msg = "Missing optional dependency 'ipython' required for interactive plotting."
        raise ImportError(msg)

    layout = ipywidgets.Layout(
        width=kwargs.get("width", "auto"),
        height=kwargs.get("height"),
    )

    # if points are supplied, center map on them and plot them
    if points is not None:
        if kwargs.get("points_as_latlon", False) is True:
            center_ll = [points.lon.mean(), points.lat.mean()]
        else:
            # convert points to lat lon
            if hemisphere == "south":
                points_ll: pd.DataFrame = utils.epsg3031_to_latlon(points)
            elif hemisphere == "north":
                points_ll = utils.epsg3413_to_latlon(points)
            else:
                msg = "hemisphere must be north or south"
                raise ValueError(msg)
            # if points supplied, center map on points
            center_ll = [np.nanmedian(points_ll.lat), np.nanmedian(points_ll.lon)]
            # add points to geodataframe
            gdf = gpd.GeoDataFrame(
                points_ll,
                geometry=gpd.points_from_xy(points_ll.lon, points_ll.lat),
            )
            geo_data = ipyleaflet.GeoData(
                geo_dataframe=gdf,
                point_style={"radius": 1, "color": "red", "weight": 1},
            )
    else:
        # if no points, center map on 0, 0
        if hemisphere == "south":
            center_ll = (-90, 0)  # type: ignore[assignment]
        elif hemisphere == "north":
            center_ll = (90, -45)  # type: ignore[assignment]
        else:
            msg = "hemisphere must be north or south"
            raise ValueError(msg)
    if center_yx is not None:
        if hemisphere == "south":
            center_ll = utils.epsg3031_to_latlon(center_yx)  # type: ignore[assignment]
        elif hemisphere == "north":
            center_ll = utils.epsg3413_to_latlon(center_yx)  # type: ignore[assignment]
        else:
            msg = "hemisphere must be north or south"
            raise ValueError(msg)

    if hemisphere == "south":
        if basemap_type == "BlueMarble":
            base = ipyleaflet.basemaps.NASAGIBS.BlueMarbleBathymetry3031  # pylint: disable=no-member
            proj = ipyleaflet.projections.EPSG3031.NASAGIBS
        elif basemap_type == "Imagery":
            base = ipyleaflet.basemaps.Esri.AntarcticImagery  # pylint: disable=no-member
            proj = ipyleaflet.projections.EPSG3031.ESRIImagery
        elif basemap_type == "Basemap":
            base = ipyleaflet.basemaps.Esri.AntarcticBasemap  # pylint: disable=no-member
            proj = ipyleaflet.projections.EPSG3031.ESRIBasemap
        elif basemap_type == "IceVelocity":
            base = ipyleaflet.basemaps.NASAGIBS.MEaSUREsIceVelocity3031  # pylint: disable=no-member
            proj = ipyleaflet.projections.EPSG3031.NASAGIBS
        else:
            msg = "invalid string for basemap_type"
            raise ValueError(msg)
    elif hemisphere == "north":
        if basemap_type == "BlueMarble":
            base = ipyleaflet.basemaps.NASAGIBS.BlueMarbleBathymetry3413  # pylint: disable=no-member
            proj = ipyleaflet.projections.EPSG3413.NASAGIBS
        # elif basemap_type == "Imagery":
        #   base = ipyleaflet.basemaps.Esri.ArcticImagery  # pylint: disable=no-member
        #   proj = ipyleaflet.projections.EPSG5936.ESRIImagery
        # elif basemap_type == "Basemap":
        #   base = ipyleaflet.basemaps.Esri.OceanBasemap  # pylint: disable=no-member
        #   proj = ipyleaflet.projections.EPSG5936.ESRIBasemap
        #   base = ipyleaflet.basemaps.Esri.ArcticOceanBase  # pylint: disable=no-member
        #   proj = ipyleaflet.projections.EPSG5936.ESRIBasemap
        elif basemap_type == "IceVelocity":
            base = ipyleaflet.basemaps.NASAGIBS.MEaSUREsIceVelocity3413  # pylint: disable=no-member
            proj = ipyleaflet.projections.EPSG3413.NASAGIBS

        else:
            msg = "invalid string for basemap_type"
            raise ValueError(msg)
    else:
        msg = "hemisphere must be north or south"
        raise ValueError(msg)
    # create the map
    m = ipyleaflet.Map(
        center=center_ll,
        zoom=zoom,
        layout=layout,
        basemap=base,
        crs=proj,
        dragging=True,
    )

    if points is not None:
        m.add_layer(geo_data)

    m.default_style = {"cursor": "crosshair"}
    if display_xy is True:
        label_xy = ipywidgets.Label()
        display(label_xy)

        def handle_click(**kwargs: typing.Any) -> None:
            if kwargs.get("type") == "click":
                latlon = kwargs.get("coordinates")
                if hemisphere == "south":
                    label_xy.value = str(utils.latlon_to_epsg3031(latlon))
                elif hemisphere == "north":
                    label_xy.value = str(utils.latlon_to_epsg3413(latlon))

    m.on_interaction(handle_click)

    if show is True:
        display(m)

    return m


def subplots(
    grids: list[xr.DataArray],
    hemisphere: str | None = None,
    region: tuple[float, float, float, float] | None = None,
    dims: tuple[int, int] | None = None,
    fig_title: str | None = None,
    fig_x_axis_title: str | None = None,
    fig_y_axis_title: str | None = None,
    fig_title_font: str = "30p,Helvetica-Bold",
    subplot_labels: bool = True,
    subplot_labels_loc: str = "TL",
    row_titles: list[str] | None = None,
    column_titles: list[str] | None = None,
    **kwargs: typing.Any,
) -> pygmt.Figure:
    """
    Plot a series of grids as individual suplots. This will automatically configure the
    layout to be closest to a square. Add any parameters from `plot_grd()` here as
    keyword arguments for further customization.

    Parameters
    ----------
    grids : list
        list of xarray.DataArray's to be plotted
    hemisphere : str, optional
        choose between plotting in the "north" or "south" hemispheres, by default None
    region : tuple[float, float, float, float], optional
        choose to subset the grids to a specified region, in format
        [xmin, xmax, ymin, ymax], by default None
    dims : tuple, optional
        customize the subplot dimensions (# rows, # columns), by default will use
        `utils.square_subplots()` to make a square(~ish) layout.
    fig_title : str, optional
        add a title to the figure, by default None
    fig_x_axis_title : str, optional
        add a title to the x axis of the figure, by default None
    fig_y_axis_title : str, optional
        add a title to the y axis of the figure, by default None
    fig_title_font : str, optional
        font for the figure title, by default "30p,Helvetica-Bold"
    subplot_labels : bool, optional
        add subplot labels (a, b, c ...), by default True
    subplot_labels_loc : str, optional
        location of subplot labels, by default "TL"
    row_titles : list, optional
        add titles to the left of each row, by default None
    column_titles : list, optional
        add titles above each column, by default None

    Returns
    -------
    pygmt.Figure
        Returns a figure object, which can be used by other PyGMT plotting functions.
    """

    kwargs = copy.deepcopy(kwargs)

    # if no defined region, get from first grid in list
    if region is None:
        try:
            region = utils.get_grid_info(grids[0])[1]
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception(e)
            logger.warning("grid region can't be extracted, using antarctic region.")
            region = regions.antarctica
    region = typing.cast(tuple[float, float, float, float], region)

    # get best dimensions for subplot
    nrows, ncols = utils.square_subplots(len(grids)) if dims is None else dims

    # get amounts to shift each figure (multiples of figure width and height)
    xshift_amount = kwargs.pop("xshift_amount", 1)
    yshift_amount = kwargs.pop("yshift_amount", -1)

    # extra lists of args for each grid
    cpt_limits = kwargs.pop("cpt_limits", None)
    cmaps = kwargs.pop("cmaps", None)
    titles = kwargs.pop("titles", kwargs.pop("subplot_titles", None))
    cbar_labels = kwargs.pop("cbar_labels", None)
    cbar_units = kwargs.pop("cbar_units", None)
    point_sets = kwargs.pop("point_sets", None)
    row_titles_font = kwargs.pop("row_titles_font", "38p,Helvetica,black")
    column_titles_font = kwargs.pop("column_titles_font", "38p,Helvetica,black")
    fig_x_axis_title_y_offset = kwargs.pop("fig_x_axis_title_y_offset", "2c")
    fig_y_axis_title_x_offset = kwargs.pop("fig_y_axis_title_x_offset", "2c")
    fig_axis_title_font = kwargs.pop("fig_axis_title_font", "30p,Helvetica-Bold")
    fig_title_y_offset = kwargs.pop("fig_title_y_offset", "2c")
    reverse_cpts = kwargs.pop("reverse_cpts", None)
    insets = kwargs.pop("insets", None)
    scalebars = kwargs.pop("scalebars", None)

    new_kwargs = {
        "cpt_lims": cpt_limits,
        "cmap": cmaps,
        "title": titles,
        "cbar_label": cbar_labels,
        "cbar_unit": cbar_units,
        "points": point_sets,
        "reverse_cpt": reverse_cpts,
        "inset": insets,
        "scalebar": scalebars,
    }
    # check in not none they are the correct length
    for k, v in new_kwargs.items():
        if v is not None:
            if len(v) != len(grids):
                msg = (
                    f"Length of supplied list of `{k}` must match the number of grids."
                )
                raise ValueError(msg)
            if not isinstance(v, list):
                msg = f"`{k}` must be a list."
    row_num = 0
    for i, g in enumerate(grids):
        xshift = xshift_amount
        yshift = yshift_amount

        kwargs2 = copy.deepcopy(kwargs)

        if i == 0:
            fig = (None,)
            origin_shift = "initialize"
        elif i % ncols == 0:
            origin_shift = "both"
            xshift = (-ncols + 1) * xshift
            row_num += 1
        else:
            origin_shift = "x"

        for k, v in new_kwargs.items():
            if (v is not None) & (kwargs2.get(k) is None):
                kwargs2[k] = v[i]

        fig = plot_grd(
            g,
            fig=fig,
            origin_shift=origin_shift,
            xshift_amount=xshift,
            yshift_amount=yshift,
            region=region,
            hemisphere=hemisphere,
            **kwargs2,
        )

        # add overall title
        if (fig_title is not None) & (i == 0):
            fig_width = utils.get_fig_width()
            fig.text(  # type: ignore[attr-defined]
                text=fig_title,
                position="TC",
                font=fig_title_font,
                offset=f"{(((fig_width*xshift)/2)*(ncols-1))}c/{fig_title_y_offset}",
                no_clip=True,
            )
        if (fig_x_axis_title is not None) & (i == int(ncols / 2)):
            fig.text(  # type: ignore[attr-defined]
                text=fig_x_axis_title,
                position="TC",
                justify="BC",
                font=fig_axis_title_font,
                offset=f"0c/{fig_x_axis_title_y_offset}",
                no_clip=True,
            )
        if (
            (fig_y_axis_title is not None)
            & (row_num == int(nrows / 2))
            & (i % ncols == 0)
        ):
            fig.text(  # type: ignore[attr-defined]
                text=fig_y_axis_title,
                position="ML",
                justify="BC",
                font=fig_axis_title_font,
                offset=f"-{fig_y_axis_title_x_offset}/0c",
                no_clip=True,
                angle=90,
            )

        if subplot_labels:
            if i < 26:
                label = string.ascii_lowercase[i]
            elif i < 26 * 2:
                label = f"a{string.ascii_lowercase[i-26]}"
            elif i < 26 * 3:
                label = f"b{string.ascii_lowercase[i-(26*2)]}"
            elif i < 26 * 4:
                label = f"b{string.ascii_lowercase[i-(26*3)]}"
            elif i < 26 * 5:
                label = f"b{string.ascii_lowercase[i-(26*4)]}"
            elif i < 26 * 6:
                label = f"b{string.ascii_lowercase[i-(26*5)]}"
            else:
                label = None

            fig.text(  # type: ignore[attr-defined]
                position=subplot_labels_loc,
                justify="TL",
                text=f"{label})",
                font="18p,Helvetica,black",
                offset="j.1c",
                no_clip=True,
                fill="white",
            )

        # add vertical title to left of each row
        if (row_titles is not None) & (i % ncols == 0):
            fig.text(  # type: ignore[attr-defined]
                justify="BC",
                position="ML",
                offset="-.5c/0c",
                text=row_titles[int(i / ncols)],  # type: ignore[index]
                angle=90,
                font=row_titles_font,
                no_clip=True,
            )

        # add horizontal title above each column
        if (column_titles is not None) & (i < ncols):
            fig.text(  # type: ignore[attr-defined]
                justify="BC",
                position="TC",
                text=column_titles[i],  # type: ignore[index]
                font=column_titles_font,
                no_clip=True,
            )

    return fig


def plot_3d(
    grids: list[xr.DataArray] | xr.DataArray,
    cmaps: list[str] | str,
    exaggeration: list[float] | float,
    view: tuple[float, float] = (170, 30),
    vlims: tuple[float, float] | None = None,
    region: tuple[float, float, float, float] | None = None,
    hemisphere: str | None = None,
    shp_mask: str | gpd.GeoDataFrame | None = None,
    polygon_mask: list[float] | None = None,
    colorbar: bool = True,
    cbar_perspective: bool = True,
    **kwargs: typing.Any,
) -> pygmt.Figure:
    """
    create a 3D perspective plot of a list of grids

    Parameters
    ----------
    grids : list or xarray.DataArray
        xarray DataArrays to be plotted in 3D
    cmaps : list or str
        list of PyGMT colormap names to use for each grid
    exaggeration : list or float
        list of vertical exaggeration factors to use for each grid
    view : tuple, optional
        tuple of azimuth and elevation angles for the view, by default [170, 30]
    vlims : tuple, optional
        tuple of vertical limits for the plot, by default is z range of grids
    region : tuple[float, float, float, float], optional
        region for the figure in format [xmin, xmax, ymin, ymax], by default None
    hemisphere : str, optional
        choose between plotting in the "north" or "south" hemispheres, by default None
    shp_mask : Union[str or geopandas.GeoDataFrame], optional
        shapefile or geodataframe to clip the grids with, by default None
    colorbar : bool, optional
        whether to plot a colorbar, by default True
    cbar_perspective : bool, optional
        whether to plot the colorbar in perspective, by default True

    Returns
    -------
    pygmt.Figure
        Returns a figure object, which can be used by other PyGMT plotting functions.
    """
    fig_height = kwargs.get("fig_height", 15)
    fig_width = kwargs.get("fig_width")

    cbar_labels = kwargs.get("cbar_labels")

    # colormap kwargs
    modis = kwargs.get("modis", False)
    grd2cpt = kwargs.get("grd2cpt", False)
    cmap_region = kwargs.get("cmap_region")
    robust = kwargs.get("robust", False)
    reverse_cpt = kwargs.get("reverse_cpt", False)
    cpt_lims_list = kwargs.get("cpt_lims")

    if not isinstance(grids, list):
        grids = [grids]

    # number of grids to plot
    num_grids = len(grids)

    # if not provided as a list, make it a list the length of num_grids
    if not isinstance(cbar_labels, list):
        cbar_labels = [cbar_labels] * num_grids
    if not isinstance(modis, list):
        modis = [modis] * num_grids
    if not isinstance(grd2cpt, list):
        grd2cpt = [grd2cpt] * num_grids
    if not isinstance(cmap_region, list):
        cmap_region = [cmap_region] * num_grids
    if not isinstance(robust, list):
        robust = [robust] * num_grids
    if not isinstance(reverse_cpt, list):
        reverse_cpt = [reverse_cpt] * num_grids
    if not isinstance(cmaps, list):
        cmaps = [cmaps] * num_grids
    if not isinstance(exaggeration, list):
        exaggeration = [exaggeration] * num_grids
    if cpt_lims_list is None:
        cpt_lims_list = [None] * num_grids
    elif (
        (isinstance(cpt_lims_list, list))
        & (len(cpt_lims_list) == 2)
        & (all(isinstance(x, float) for x in cpt_lims_list))
    ):
        cpt_lims_list = [cpt_lims_list] * num_grids
    if (
        isinstance(cmap_region, list)
        & (len(cmap_region) == 4)
        & (all(isinstance(x, float) for x in cmap_region))
    ):
        cmap_region = [cmap_region] * num_grids

    # if plot region not specified, try to pull from grid info
    if region is None:
        try:
            region = utils.get_grid_info(grids[0])[1]
        except Exception as e:  # pylint: disable=broad-exception-caught
            # pygmt.exceptions.GMTInvalidInput:
            msg = "first grids' region can't be extracted, please provide with `region`"
            raise ValueError(msg) from e

    region = typing.cast(tuple[float, float, float, float], region)

    # set figure projection and size from input region and figure dimensions
    # by default use figure height to set projection
    if fig_width is None:
        proj, _proj_latlon, fig_width, fig_height = utils.set_proj(
            region,
            fig_height=fig_height,
            hemisphere=hemisphere,
        )
    # if fig_width is set, use it to set projection
    else:
        proj, _proj_latlon, fig_width, fig_height = utils.set_proj(
            region,
            fig_width=fig_width,
            hemisphere=hemisphere,
        )

    # set vertical limits
    if vlims is None:
        vlims = utils.get_combined_min_max(grids)

    new_region = region + vlims

    # initialize the figure
    fig = pygmt.Figure()

    # iterate through grids and plot them
    for i, grid in enumerate(grids):
        # if provided, mask grid with shapefile
        if shp_mask is not None:
            grid = utils.mask_from_shp(  # noqa: PLW2901
                shp_mask,
                xr_grid=grid,
                masked=True,
                invert=kwargs.get("invert", False),
                hemisphere=hemisphere,
            )
            grid.to_netcdf("tmp.nc")
            grid = xr.load_dataset("tmp.nc")["z"]  # noqa: PLW2901
            pathlib.Path("tmp.nc").unlink()
        # if provided, mask grid with polygon from interactive map via
        # regions.draw_region
        elif polygon_mask is not None:
            grid = utils.mask_from_polygon(  # noqa: PLW2901
                polygon_mask,
                grid=grid,
                hemisphere=hemisphere,
            )
        # create colorscales
        cpt_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in [
                "modis",
                "grd2cpt",
                "cpt_lims",
                "cmap_region",
                "robust",
                "reverse_cpt",
                "shp_mask",
            ]
        }
        cmap, colorbar, _ = set_cmap(
            cmaps[i],
            grid=grid,
            modis=modis[i],
            grd2cpt=grd2cpt[i],
            cpt_lims=cpt_lims_list[i],
            cmap_region=cmap_region[i],
            robust=robust[i],
            reverse_cpt=reverse_cpt[i],
            hemisphere=hemisphere,
            colorbar=colorbar,
            **cpt_kwargs,
        )

        # set transparency values
        transparencies = kwargs.get("transparencies")
        transparency = 0 if transparencies is None else transparencies[i]

        # plot as perspective view
        fig.grdview(
            grid=grid,
            cmap=cmap,
            projection=proj,
            region=new_region,
            frame=None,
            perspective=view,
            zsize=f"{exaggeration[i]}c",
            surftype="c",
            transparency=transparency,
            # plane='-9000+ggrey',
            shading=kwargs.get("shading", False),
        )

        # display colorbar
        if colorbar is True:
            cbar_xshift = kwargs.get("cbar_xshift")
            cbar_yshift = kwargs.get("cbar_yshift")

            xshift = 0 if cbar_xshift is None else cbar_xshift[i]
            # yshift = fig_height / 2 if cbar_yshift is None else cbar_yshift[i]
            yshift = 0 if cbar_yshift is None else cbar_yshift[i]

            fig.shift_origin(yshift=f"{yshift}c", xshift=f"{xshift}c")
            fig.colorbar(
                cmap=cmap,
                # position=f"g{np.max(region[0:2])}/{np.mean(region[2:4])}+w{fig_width*.4}c/.5c+v+e+m", #noqa: E501
                # # vertical, with triangles, text opposite
                position=f"jMR+w{fig_width*.4}c/.5c+v+e+m",  # vertical, with triangles, text opposite #noqa: E501
                frame=f"xaf+l{cbar_labels[i]}",
                perspective=cbar_perspective,
                box="+gwhite+c3p",
            )
            fig.shift_origin(yshift=f"{-yshift}c", xshift=f"{-xshift}c")

        # shift up for next grid
        if i < len(grids) - 1:
            zshifts = kwargs.get("zshifts")
            zshift = 0 if zshifts is None else zshifts[i]

            if zshifts is not None:
                fig.shift_origin(yshift=f"{zshift}c")

    return fig


def interactive_data(
    hemisphere: str | None = None,
    coast: bool = True,
    grid: xr.DataArray | None = None,
    grid_cmap: str = "inferno",
    points: pd.DataFrame = None,
    points_z: str | None = None,
    points_color: str = "red",
    points_cmap: str = "viridis",
    **kwargs: typing.Any,
) -> typing.Any:
    """
    plot points or grids on an interactive map using GeoViews

    Parameters
    ----------
    hemisphere : str, optional
        set whether to plot in "north" hemisphere (EPSG:3413) or "south" hemisphere
        (EPSG:3031)
    coast : bool, optional
        choose whether to plot coastline data, by default True
    grid : xarray.DataArray, optional
        display a grid on the map, by default None
    grid_cmap : str, optional
        colormap to use for the grid, by default 'inferno'
    points : pandas.DataFrame, optional
        points to display on the map, must have columns 'x' and 'y', by default None
    points_z : str, optional
        name of column to color points by, by default None
    points_color : str, optional
        if no `points_z` supplied, color to use for all points, by default 'red'
    points_cmap : str, optional
        colormap to use for the points, by default 'viridis'

    Returns
    -------
    holoviews.Overlay
        holoview/geoviews map instance

    Example
    -------
    >>> from polartoolkit import regions, utils, maps
    ...
    >>> bedmap2_bed = fetch.bedmap2(layer='bed', region=regions.ross_ice_shelf)
    >>> GHF_point_data = fetch.ghf(version='burton-johnson-2020', points=True)
    ...
    >>> image = maps.interactive_data(
    ...    hemisphere="south",
    ...    grid = bedmap2_bed,
    ...    points = GHF_point_data[['x','y','GHF']],
    ...    points_z = 'GHF',
    ...    )
    >>> image
    """
    hemisphere = utils.default_hemisphere(hemisphere)

    if gv is None:
        msg = (
            "Missing optional dependency 'geoviews' required for interactive plotting."
        )
        raise ImportError(msg)
    if crs is None:
        msg = "Missing optional dependency 'cartopy' required for interactive plotting."
        raise ImportError(msg)

    # set the plot style
    gv.extension("bokeh")

    # initialize figure with coastline
    if hemisphere == "north":
        coast_gdf = gpd.read_file(fetch.groundingline(version="BAS"), engine=ENGINE)
        crsys = crs.NorthPolarStereo()
    elif hemisphere == "south":
        coast_gdf = gpd.read_file(
            fetch.groundingline(version="measures-v2"), engine=ENGINE
        )
        crsys = crs.SouthPolarStereo()
    else:
        msg = "hemisphere must be north or south"
        raise ValueError(msg)

    coast_fig = gv.Path(
        coast_gdf,
        crs=crsys,
    )
    # set projection, and change groundingline attributes
    coast_fig.opts(
        projection=crsys,
        color=kwargs.get("coast_color", "black"),
        data_aspect=1,
    )

    figure = coast_fig

    # display grid
    if grid is not None:
        # turn grid into geoviews dataset
        dataset = gv.Dataset(
            grid,
            [grid.dims[1], grid.dims[0]],
            crs=crsys,
        )
        # turn geoviews dataset into image
        gv_grid = dataset.to(gv.Image)

        # change options
        gv_grid.opts(cmap=grid_cmap, colorbar=True, tools=["hover"])

        # add to figure
        figure = figure * gv_grid

    # display points
    if points is not None:
        gv_points = geoviews_points(
            points=points,
            points_z=points_z,
            points_color=points_color,
            points_cmap=points_cmap,
            **kwargs,
        )
        # if len(points.columns) < 3:
        #     # if only 2 cols are given, give points a constant color
        #     # turn points into geoviews dataset
        #     gv_points = gv.Points(
        #         points,
        #         crs=crs.SouthPolarStereo(),
        #         )

        #     # change options
        #     gv_points.opts(
        #         color=points_color,
        #         cmap=points_cmap,
        #         colorbar=True,
        #         colorbar_position='top',
        #         tools=['hover'],
        #         marker=kwargs.get('marker', 'circle'),
        #         alpha=kwargs.get('alpha', 1),
        #         size= kwargs.get('size', 4),
        #         )

        # else:
        #     # if more than 2 columns, color points by third column
        #     # turn points into geoviews dataset
        #     gv_points = gv.Points(
        #         data = points,
        #         vdims = [points_z],
        #         crs = crs.SouthPolarStereo(),
        #         )

        #     # change options
        #     gv_points.opts(
        #         color=points_z,
        #         cmap=points_cmap,
        #         colorbar=True,
        #         colorbar_position='top',
        #         tools=['hover'],
        #         marker=kwargs.get('marker', 'circle'),
        #         alpha=kwargs.get('alpha', 1),
        #         size= kwargs.get('size', 4),
        #         )

        # add to figure
        figure = figure * gv_points

    # optionally plot coast again, so it's on top
    if coast is True:
        figure = figure * coast_fig

    # trying to get datashader to auto scale colormap based on current map extent
    # from holoviews.operation.datashader import regrid
    # from holoviews.operation.datashader import rasterize

    return figure


def geoviews_points(
    points: pd.DataFrame,
    points_z: str | None = None,
    points_color: str = "red",
    points_cmap: str = "viridis",
    **kwargs: typing.Any,
) -> gv.Points:
    """
    Add points to a geoviews map instance.
    Parameters
    ----------
    points : pandas.DataFrame
        points to plot on the map, by default None
    points_z : str | None, optional
        column name to color the points by, by default None
    points_color : str, optional
        color for the points, by default "red"
    points_cmap : str, optional
        colormap to use to color the points based on `points_z`, by default "viridis"

    Returns
    -------
    holoviews.element.Points
        the instance of points

    """
    if gv is None:
        msg = (
            "Missing optional dependency 'geoviews' required for interactive plotting."
        )
        raise ImportError(msg)
    if crs is None:
        msg = "Missing optional dependency 'cartopy' required for interactive plotting."
        raise ImportError(msg)

    gv_points = gv.Points(
        data=points,
        crs=crs.SouthPolarStereo(),
    )

    if len(points.columns) < 3:
        # if only 2 cols are given, give points a constant color
        # turn points into geoviews dataset
        gv_points.opts(
            color=points_color,
            cmap=points_cmap,
            colorbar=True,
            colorbar_position="top",
            tools=["hover"],
            marker=kwargs.get("marker", "circle"),
            alpha=kwargs.get("alpha", 1),
            size=kwargs.get("size", 4),
        )
    else:
        if points_z is None:
            # change options
            gv_points.opts(
                tools=["hover"],
                marker=kwargs.get("marker", "circle"),
                alpha=kwargs.get("alpha", 1),
                size=kwargs.get("size", 4),
            )
        else:
            # if more than 2 columns, color points by third column
            # turn points into geoviews dataset
            clim = kwargs.get("cpt_lims")
            if clim is None:
                clim = utils.get_min_max(
                    points[points_z],
                    robust=kwargs.get("robust", True),
                )
            gv_points.opts(
                color=points_z,
                cmap=points_cmap,
                clim=clim,
                colorbar=True,
                colorbar_position="top",
                tools=["hover"],
                marker=kwargs.get("marker", "circle"),
                alpha=kwargs.get("alpha", 1),
                size=kwargs.get("size", 4),
            )
    gv_points.opts(
        projection=crs.SouthPolarStereo(),
        data_aspect=1,
    )

    return gv_points
