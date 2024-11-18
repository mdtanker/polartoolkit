# Copyright (c) 2024 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#
# Copyright (c) 2022 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#
# pylint: disable=too-many-lines
from __future__ import annotations

import logging
import pathlib
import typing
import warnings
from math import floor, log10

import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr

from polartoolkit import fetch, regions, utils

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
    yshift_amount: float = 1,
    xshift_amount: float = 1,
) -> tuple[pygmt.Figure, str, str | None, float, float]:
    """determine what to do with figure"""

    # initialize figure or shift for new subplot
    if origin_shift == "initialize":
        fig = pygmt.Figure()
        # set figure projection and size from input region and figure dimensions
        # by default use figure height to set projection
        if fig_width is None:
            if fig_height is None:
                msg = "fig_height must be set if fig_width is not set."
                raise ValueError(msg)
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
        if origin_shift == "x":
            if fig_height is None:
                fig_height = utils.get_fig_height()
            proj, proj_latlon, fig_width, fig_height = utils.set_proj(
                region,
                fig_height=fig_height,
                hemisphere=hemisphere,
            )
            fig.shift_origin(xshift=xshift_amount * (fig_width + 0.4))
        elif origin_shift == "y":
            if fig_height is None:
                fig_height = utils.get_fig_height()
            proj, proj_latlon, fig_width, fig_height = utils.set_proj(
                region,
                fig_height=fig_height,
                hemisphere=hemisphere,
            )
            fig.shift_origin(yshift=yshift_amount * (fig_height + 3))
        elif origin_shift == "both":
            if fig_height is None:
                fig_height = utils.get_fig_height()
            proj, proj_latlon, fig_width, fig_height = utils.set_proj(
                region,
                fig_height=fig_height,
                hemisphere=hemisphere,
            )
            fig.shift_origin(
                xshift=xshift_amount * (fig_width + 0.4),
                yshift=yshift_amount * (fig_height + 3),
            )
        elif origin_shift is None:
            if fig_height is None:
                fig_height = utils.get_fig_height()
            proj, proj_latlon, fig_width, fig_height = utils.set_proj(
                region,
                fig_height=fig_height,
                hemisphere=hemisphere,
            )
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
    imagery_basemap: bool = False,
    title: str | None = None,
    inset: bool = False,
    points: pd.DataFrame | None = None,
    gridlines: bool = False,
    origin_shift: str = "initialize",
    fig: pygmt.Figure | None = None,
    **kwargs: typing.Any,
) -> pygmt.Figure:
    """
    Create a figure without plotting a grid. Can be used as a basemap to plot your own
    data, or features can be automatically data such as coastline and grounding lines,
    inset figure location maps, background imagery, scalebars, gridlines and
    northarrows.

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
        `depoorter-2013` for the southern.
    north_arrow : bool, optional
        choose to add a north arrow to the plot, by default is False.
    scalebar : bool, optional
        choose to add a scalebar to the plot, by default is False. See `add_scalebar`
        for additional kwargs
    faults : bool, optional
        choose to plot faults on the map, by default is False
    imagery_basemap : bool, optional
        choose to add a background imagery basemap, by default is False. If true, will
        use LIMA for southern hemisphere and MODIS MoG for the northern hemisphere.
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
        instance height, by default is 1.
    frame : str
        GMT frame string to use for the basemap, by default is None
    transparency : int
        transparency to use for the background imagery, by default is 0
    inset_pos : str
        position for inset map; either 'TL', 'TR', BL', 'BR', by default is 'TL'
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
        fill color of points, by default is 'black'.
    points_pen : str
        pen color and width of points, by default is '1p,black'.
    points_cmap : str
        colormap to use for points, by default is None.
    colorbar : bool
        choose to add a colorbar for the points to the plot, by default is False.
    scale_font_color : str
        color of the scalebar font, by default is 'black'.
    scale_length_perc : float
        percentage of the figure width to use for the scalebar, by default is 0.25.
    scale_position : str
        position of the scalebar on the figure, by default is 'n.5/.05' which is bottom
        center of the plot.
    coast_pen : str
        GMT pen string to use for the coastlines, by default is None
    no_coast : bool
        choose to not plot coastlines, just grounding lines, by default is False
    coast_version : str
        version of coastlines to plot, by default depends on the hemisphere
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

    try:
        hemisphere = utils.default_hemisphere(hemisphere)
    except KeyError:
        hemisphere = None

    # if region not set, use antarctic or greenland regions
    if region is None:
        if hemisphere == "north":
            region = regions.greenland
        if hemisphere == "south":
            region = regions.antarctica
        else:
            msg = "Region must be specified if hemisphere is not specified."
            raise ValueError(msg)
    fig, proj, proj_latlon, fig_width, _ = _set_figure_spec(
        region=region,
        fig=fig,
        origin_shift=origin_shift,
        fig_height=kwargs.get("fig_height", 15),
        fig_width=kwargs.get("fig_width"),
        hemisphere=hemisphere,
        yshift_amount=kwargs.get("yshift_amount", 1),
        xshift_amount=kwargs.get("xshift_amount", 1),
    )

    show_region = kwargs.get("show_region")

    if imagery_basemap is True:
        if hemisphere == "north":
            image = fetch.modis(version="500m", hemisphere="north")
            imagery_cmap, _, _ = set_cmap(
                True,
                modis=True,
            )
        else:
            image = fetch.imagery()
            imagery_cmap = None
        fig.grdimage(
            grid=image,
            cmap=imagery_cmap,
            projection=proj,
            region=region,
            transparency=kwargs.get("transparency", 0),
            frame=kwargs.get("frame", "nwse+gwhite"),
        )
    else:
        # create blank basemap
        fig.basemap(
            region=region,
            projection=proj,
            frame=kwargs.get("frame", "nwse+gwhite"),
            verbose="e",
        )

    # add lat long grid lines
    if gridlines is True:
        if hemisphere is None:
            logging.warning(
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
        add_coast(
            fig,
            hemisphere=hemisphere,
            region=region,
            projection=proj,
            pen=kwargs.get("coast_pen"),
            no_coast=kwargs.get("no_coast", False),
            version=kwargs.get("coast_version"),
        )

    # plot faults
    if faults is True:
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
        add_box(
            fig,
            show_region,
            pen=kwargs.get("region_pen"),  # type: ignore[arg-type]
        )

    # add datapoints
    if points is not None:
        if ("x" in points.columns) and ("y" in points.columns):
            x, y = points.x, points.y
        elif ("easting" in points.columns) and ("northing" in points.columns):
            x, y = points.easting, points.northing
        else:
            msg = "points must contain columns 'x' and 'y' or 'easting' and 'northing'."
            raise ValueError(msg)
        fig.plot(
            x=x,
            y=y,
            style=kwargs.get("points_style", "c.2c"),
            fill=kwargs.get("points_fill", "black"),
            pen=kwargs.get("points_pen", "1p,black"),
            cmap=kwargs.get("points_cmap"),
        )
        # display colorbar
        if kwargs.get("colorbar", False) is True:
            # removed duplicate kwargs before passing to add_colorbar
            cbar_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key
                not in [
                    "fig_width",
                    "fig",
                ]
            }
            add_colorbar(
                fig,
                cmap=kwargs.get("points_cmap"),
                fig_width=fig_width,
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
            hemisphere=hemisphere,
            **new_kwargs,
        )

    # add scalebar
    if scalebar is True:
        if proj_latlon is None:
            msg = "Argument `hemisphere` needs to be specified for plotting a scalebar"
            raise ValueError(msg)

        add_scalebar(
            fig=fig,
            region=region,
            projection=proj_latlon,
            font_color=kwargs.get("scale_font_color", "black"),
            length_perc=kwargs.get("scale_length_perc", 0.25),
            position=kwargs.get("scale_position", "n.5/.05"),
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
    if title is None:
        fig.basemap(
            region=region,
            projection=proj,
            frame="wesn",
        )
    else:
        with pygmt.config(FONT_TITLE=kwargs.get("title_font", "auto")):
            fig.basemap(
                region=region,
                projection=proj,
                frame=f"wesn+t{title}",
            )

    return fig


def set_cmap(
    cmap: str | bool,
    grid: str | xr.DataArray | None = None,
    modis: bool = False,
    grd2cpt: bool = False,
    cpt_lims: tuple[float, float] | None = None,
    cmap_region: tuple[float, float, float, float] | None = None,
    robust: bool = False,
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
       grid used for grd2cpt colormap equalization, by default None
    modis : bool, optional
        choose appropriate cmap for plotting modis data, by default False
    grd2cpt : bool, optional
        equalized the colormap to the grid data values, by default False
    cpt_lims : tuple[float, float] | None, optional
        limits to set for the colormap, by default None
    cmap_region : tuple[float, float, float, float] | None, optional
        extract colormap limits from a subset of the grid, in format
        [xmin, xmax, ymin, ymax], by default None
    robust : bool, optional
        use the 2nd and 98th percentile of the data from the grid, by default False
    reverse_cpt : bool, optional
        change the direction of the cmap, by default False
    shp_mask : geopandas.GeoDataFrame | str | None, optional
        a shapefile to mask the grid by before extracting limits, by default None
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
        if cpt_lims is None and isinstance(grid, (xr.DataArray)):
            zmin, zmax = utils.get_min_max(
                grid,
                shp_mask,
                region=cmap_region,
                robust=robust,
                hemisphere=hemisphere,
            )
        elif cpt_lims is None and isinstance(grid, (str)):
            with xr.load_dataarray(grid) as da:
                zmin, zmax = utils.get_min_max(
                    da,
                    shp_mask,
                    region=cmap_region,
                    robust=robust,
                    hemisphere=hemisphere,
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
        except pygmt.exceptions.GMTCLibError:
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
            if isinstance(grid, (xr.DataArray)):
                zmin, zmax = utils.get_min_max(
                    grid,
                    shp_mask,
                    region=cmap_region,
                    robust=robust,
                    hemisphere=hemisphere,
                )
            else:
                with xr.load_dataarray(grid) as da:
                    zmin, zmax = utils.get_min_max(
                        da,
                        shp_mask,
                        region=cmap_region,
                        robust=robust,
                        hemisphere=hemisphere,
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
                logging.warning("grid's min value is greater or equal to its max value")
                pygmt.makecpt(
                    cmap=cmap,
                    background=True,
                    reverse=reverse_cpt,
                    verbose="e",
                )
            else:
                logging.exception(e)
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
    imagery_basemap: bool = False,
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
        `depoorter-2013` for the southern.
    north_arrow : bool, optional
        choose to add a north arrow to the plot, by default is False.
    scalebar : bool, optional
        choose to add a scalebar to the plot, by default is False. See `add_scalebar`
        for additional kwargs
    faults : bool, optional
        choose to plot faults on the map, by default is False
    imagery_basemap : bool, optional
        choose to add a background imagery basemap, by default is False. If true, will
        use LIMA for southern hemisphere and MODIS MoG for the northern hemisphere.
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
        instance height, by default is 1.
    frame : str
        GMT frame string to use for the basemap, by default is None
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
        use the 2nd and 98th percentile of the data to set color scale limits, by
        default is False.
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
    transparency : int
        transparency of the grid, by default is 0
    inset_pos : str
        position for inset map; either 'TL', 'TR', BL', 'BR', by default is 'TL'
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
        fill color of points, by default is 'black'.
    points_pen : str
        pen color and width of points, by default is '1p,black'.
    points_cmap : str
        colormap to use for points, by default is None.
    scale_font_color : str
        color of the scalebar font, by default is 'black'.
    scale_length_perc : float
        percentage of the figure width to use for the scalebar, by default is 0.25.
    scale_position : str
        position of the scalebar on the figure, by default is 'n.5/.05' which is bottom
        center of the plot.
    coast_pen : str
        GMT pen string to use for the coastlines, by default is None
    no_coast : bool
        choose to not plot coastlines, just grounding lines, by default is False
    coast_version : str
        version of coastlines to plot, by default depends on the hemisphere
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

    try:
        hemisphere = utils.default_hemisphere(hemisphere)
    except KeyError:
        hemisphere = None

    warnings.filterwarnings("ignore", message="pandas.Int64Index")
    warnings.filterwarnings("ignore", message="pandas.Float64Index")

    # get region from grid or use supplied region
    if region is not None:
        if isinstance(grid, xr.DataArray):
            grid = pygmt.grdcut(
                grid,
                region=region,
                verbose="q",
            )
    else:
        try:
            region = utils.get_grid_info(grid)[1]
        except Exception as e:  # pylint: disable=broad-exception-caught
            msg = "grid's region can't be extracted, please provide with `region`"
            raise ValueError(msg) from e

    region = typing.cast(tuple[float, float, float, float], region)

    fig, proj, proj_latlon, fig_width, _ = _set_figure_spec(
        region=region,
        fig=fig,
        origin_shift=origin_shift,
        fig_height=kwargs.get("fig_height", 15),
        fig_width=kwargs.get("fig_width"),
        hemisphere=hemisphere,
        yshift_amount=kwargs.get("yshift_amount", 1),
        xshift_amount=kwargs.get("xshift_amount", 1),
    )

    show_region = kwargs.get("show_region")

    if imagery_basemap is True:
        if hemisphere == "north":
            image = fetch.modis(version="500m", hemisphere="north")
            imagery_cmap, _, _ = set_cmap(
                True,
                modis=True,
            )
        else:
            image = fetch.imagery()
            imagery_cmap = None
        fig.grdimage(
            grid=image,
            cmap=imagery_cmap,
            projection=proj,
            region=region,
        )

    cmap, colorbar, cpt_lims = set_cmap(
        cmap,
        grid=grid,
        hemisphere=hemisphere,
        **kwargs,
    )

    # display grid
    fig.grdimage(
        grid=grid,
        cmap=cmap,
        projection=proj,
        region=region,
        nan_transparent=True,
        frame=kwargs.get("frame"),
        shading=kwargs.get("shading"),
        transparency=kwargs.get("transparency", 0),
        verbose="q",
    )

    # add datapoints
    if points is not None:
        if ("x" in points.columns) and ("y" in points.columns):
            x, y = points.x, points.y
        elif ("easting" in points.columns) and ("northing" in points.columns):
            x, y = points.easting, points.northing
        else:
            msg = "points must contain columns 'x' and 'y' or 'easting' and 'northing'."
            raise ValueError(msg)
        fig.plot(
            x=x,
            y=y,
            style=kwargs.get("points_style", "c.2c"),
            fill=kwargs.get("points_fill", "black"),
            pen=kwargs.get("points_pen", "1p,black"),
            cmap=kwargs.get("points_cmap"),
        )

    # add box showing region
    if show_region is not None:
        add_box(
            fig,
            show_region,
            pen=kwargs.get("region_pen"),  # type: ignore[arg-type]
        )

    # plot groundingline and coastlines
    if coast is True:
        add_coast(
            fig,
            hemisphere=hemisphere,
            region=region,
            projection=proj,
            pen=kwargs.get("coast_pen"),
            no_coast=kwargs.get("no_coast", False),
            version=kwargs.get("coast_version"),
        )

    # plot faults
    if faults is True:
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
        if hemisphere is None:
            logging.warning(
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
            hemisphere=hemisphere,
            **new_kwargs,
        )

    # add scalebar
    if scalebar is True:
        if proj_latlon is None:
            msg = "Argument `hemisphere` needs to be specified for plotting a scalebar"
            raise ValueError(msg)

        add_scalebar(
            fig=fig,
            region=region,
            projection=proj_latlon,
            font_color=kwargs.get("scale_font_color", "black"),
            length_perc=kwargs.get("scale_length_perc", 0.25),
            position=kwargs.get("scale_position", "n.5/.05"),
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
                "grid",
                "fig",
            ]
        }

        add_colorbar(
            fig,
            hist_cmap=cmap,
            grid=grid,
            cpt_lims=cpt_lims,
            fig_width=fig_width,
            region=region,
            **cbar_kwargs,
        )

    # reset region and projection
    if title is None:
        fig.basemap(region=region, projection=proj, frame="wesn")
    else:
        with pygmt.config(FONT_TITLE=kwargs.get("title_font", "auto")):
            fig.basemap(region=region, projection=proj, frame=f"wesn+t{title}")

    return fig


def add_colorbar(
    fig: pygmt.Figure,
    hist: bool = False,
    cpt_lims: tuple[float, float] | None = None,
    cbar_frame: list[str] | str | None = None,
    **kwargs: typing.Any,
) -> None:
    """
    Add a colorbar and optionally a histogram based on the last cmap used by PyGMT.

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
    """
    # get the current figure width
    fig_width = utils.get_fig_width()

    # set colorbar width as percentage of total figure width
    cbar_width_perc = kwargs.get("cbar_width_perc", 0.8)

    # if plotting a histogram add 2cm of spacing instead of .2cm
    if hist is True:
        cbar_yoffset = kwargs.get("cbar_yoffset", 2)
    else:
        cbar_yoffset = kwargs.get("cbar_yoffset", 0.2)

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
    with pygmt.config(
        FONT=kwargs.get("cbar_font", "12p,Helvetica,black"),
    ):
        fig.colorbar(
            cmap=kwargs.get("cmap", True),
            position=(
                f"jBC+w{fig_width*cbar_width_perc}c+jTC+{orientation}{text_location}"
                f"+o{kwargs.get('cbar_xoffset', 0)}c/{cbar_yoffset}c+e"
            ),
            frame=cbar_frame,
            scale=kwargs.get("cbar_scale", 1),
            log=kwargs.get("cbar_log"),
        )

    # add histogram to colorbar
    # Note, depending on data and hist_type, you may need to manually set kwarg
    # `hist_ymax` to an appropriate value
    if hist is True:
        # get grid to use
        grid = kwargs.get("grid")

        hist_cmap = kwargs.get("hist_cmap", True)

        if grid is None:
            msg = "if hist is True, grid must be provided."
            raise ValueError(msg)

        # clip grid to plot region
        region = kwargs.get("region")
        # if no region supplied, get region of current PyGMT figure
        if region is None:
            with pygmt.clib.Session() as lib:
                region = tuple(lib.extract_region())
                assert len(region) == 4

        # clip grid to plot region
        if region != utils.get_grid_info(grid)[1]:
            grid_clipped = utils.subset_grid(grid, region)

            # if subplotting, region will be in figure units and grid will be clipped
            # incorrectly, hacky solution is to check if clipped figure is smaller than
            # a few data points, if so, use grids full region
            if len(grid_clipped[list(grid_clipped.sizes.keys())[0]].values) < 5:  # noqa: RUF015
                reg = kwargs.get("region")
                if reg is None:
                    msg = (
                        "Issue with detecting figure region for adding colorbar "
                        "histogram, please provide region kwarg."
                    )
                    raise ValueError(msg)
                grid_clipped = utils.subset_grid(grid, reg)

            grid = grid_clipped

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
                "getting max/min values from grid, if cpt_lims were used to create the "
                "colorscale, histogram will not properly align with colorbar!",
                stacklevel=2,
            )
            zmin, zmax = utils.get_min_max(
                grid,
                shapefile=kwargs.get("shp_mask"),
                region=kwargs.get("cmap_region"),
                robust=kwargs.get("robust", False),
                hemisphere=kwargs.get("hemisphere"),
            )
        else:
            zmin, zmax = cpt_lims

        # get grid's data for histogram
        df = vd.grid_to_table(grid)
        df2 = df.iloc[:, -1:].squeeze()

        # subset data between cbar min and max
        data = df2[df2.between(zmin, zmax)]

        bin_width = kwargs.get("hist_bin_width")
        bin_num = kwargs.get("hist_bin_num", 100)

        if bin_width is not None:
            # if bin width is set, will plot x amount of bins of width=bin_width
            bins = np.arange(zmin, zmax, step=bin_width)
        else:
            # if bin width isn't set, will plot bin_num of bins, by default = 100
            bins, bin_width = np.linspace(zmin, zmax, num=bin_num, retstep=True)

        # set hist type
        hist_type = kwargs.get("hist_type", 0)

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
            msg = "Grid is a constant value, can't make a colorbar histogram!"
            logging.warning(msg)
            return

        # define histogram region
        hist_reg = [
            zmin,
            zmax,
            kwargs.get("hist_ymin", 0),
            kwargs.get("hist_ymax", max_bin_height * 1.1),
        ]

        # shift figure to line up with top left of cbar
        xshift = kwargs.get("cbar_xoffset", 0) + ((1 - cbar_width_perc) * fig_width) / 2
        try:
            fig.shift_origin(xshift=f"{xshift}c", yshift=f"-{cbar_yoffset}c")
        except pygmt.exceptions.GMTCLibError as e:
            logging.warning(e)
            logging.warning("issue with plotting histogram, skipping...")

        # plot histograms above colorbar
        try:
            fig.histogram(
                data=data,
                projection=f"X{fig_width*cbar_width_perc}c/{cbar_yoffset-.1}c",
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
            )
        except pygmt.exceptions.GMTCLibError as e:
            logging.warning(e)
            logging.warning("issue with plotting histogram, skipping...")

        # shift figure back
        try:
            fig.shift_origin(xshift=f"{-xshift}c", yshift=f"{cbar_yoffset}c")
        except pygmt.exceptions.GMTCLibError as e:
            logging.warning(e)
            logging.warning("issue with plotting histogram, skipping...")


def add_coast(
    fig: pygmt.Figure,
    hemisphere: str | None = None,
    region: tuple[float, float, float, float] | None = None,
    projection: str | None = None,
    no_coast: bool = False,
    pen: str | None = None,
    version: str | None = None,
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
        version of groundingline to plot, by default is BAS for north hemisphere and
        Depoorter-2013 for south hemisphere
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
            version = "depoorter-2013"
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
            gdf = gpd.read_file(fetch.groundingline(version=version))
            data = gdf[gdf.Id_text == "Grounded ice or land"]
    elif version == "measures-v2":
        if no_coast is False:
            gl = gpd.read_file(fetch.groundingline(version=version))
            coast = gpd.read_file(fetch.antarctic_boundaries(version="Coastline"))
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

    region_converted = (*region, "+ue")  # codespell-ignore

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


def add_inset(
    fig: pygmt.Figure,
    hemisphere: str | None = None,
    region: tuple[float, float, float, float] | None = None,
    inset_pos: str = "TL",
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
    inset_pos : str, optional
        GMT location string for inset map, by default 'TL' (top left)
    inset_width : float, optional
        Inset width as percentage of the total figure width, by default is 25% (0.25)
    inset_reg : tuple[float, float, float, float], optional
        Region of Antarctica/Greenland to plot for the inset map, by default is whole
        area
    """
    hemisphere = utils.default_hemisphere(hemisphere)

    fig_width = utils.get_fig_width()

    inset_map = f"X{fig_width*inset_width}c"

    # if no region supplied, get region of current PyGMT figure
    if region is None:
        with pygmt.clib.Session() as lib:
            region = tuple(lib.extract_region())
            assert len(region) == 4

    with fig.inset(
        position=(
            f"J{inset_pos}+j{inset_pos}+w{fig_width*inset_width}c"
            f"+o{kwargs.get('inset_offset', '0/0')}"
        ),
        # verbose="q",
        box=kwargs.get("inset_box", False),
    ):
        if hemisphere == "north":
            if inset_reg is None:
                if "L" in inset_pos:
                    # inset reg needs to be square,
                    # if on left side, make square by adding to right side of region
                    inset_reg = (-800e3, 2000e3, -3400e3, -600e3)
                elif "R" in inset_pos:
                    inset_reg = (-1800e3, 1000e3, -3400e3, -600e3)
                else:
                    inset_reg = (-1300e3, 1500e3, -3400e3, -600e3)

            if inset_reg[1] - inset_reg[0] != inset_reg[3] - inset_reg[2]:
                logging.warning(
                    "Inset region should be square or else projection will be off."
                )
            gdf = gpd.read_file(fetch.groundingline("BAS"))
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
                logging.warning(
                    "Inset region should be square or else projection will be off."
                )
            gdf = gpd.read_file(fetch.groundingline("depoorter-2013"))
            fig.plot(
                projection=inset_map,
                region=inset_reg,
                data=gdf[gdf.Id_text == "Ice shelf"],
                fill="skyblue",
            )
            fig.plot(
                data=gdf[gdf.Id_text == "Grounded ice or land"],
                fill="grey",
            )
            fig.plot(data=gdf, pen=kwargs.get("inset_coast_pen", "0.2,black"))
        else:
            msg = "hemisphere must be north or south"
            raise ValueError(msg)

        add_box(
            fig,
            box=region,
            pen=kwargs.get("inset_box_pen", "1p,red"),
        )


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
    scale_length = kwargs.get("scale_length")
    length_perc = kwargs.get("length_perc", 0.25)
    position = kwargs.get("position", "n.5/.05")

    # if no region supplied, get region of current PyGMT figure
    if region is None:
        with pygmt.clib.Session() as lib:
            region = tuple(lib.extract_region())
            assert len(region) == 4

    region_converted = (*region, "+ue")  # codespell-ignore

    def round_to_1(x: float) -> float:
        return round(x, -int(floor(log10(abs(x)))))

    if scale_length is None:
        scale_length = typing.cast(float, scale_length)
        scale_length = round_to_1((abs(region[1] - region[0])) / 1000 * length_perc)

    with pygmt.config(
        FONT_ANNOT_PRIMARY=f"10p,{font_color}",
        FONT_LABEL=f"10p,{font_color}",
        MAP_SCALE_HEIGHT="6p",
        MAP_TICK_PEN_PRIMARY=f"0.5p,{font_color}",
    ):
        fig.basemap(
            region=region_converted,
            projection=projection,
            map_scale=f"{position}+w{scale_length}k+f+lkm+ar",
            # verbose="e",
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

    region_converted = (*region, "+ue")  # codespell-ignore

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
    """
    fig.plot(
        x=[box[0], box[0], box[1], box[1], box[0]],
        y=[box[2], box[3], box[3], box[2], box[2]],
        pen=pen,
    )


def interactive_map(
    hemisphere: str | None = None,
    center_yx: list[float] | None = None,
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
    center_yx : list, optional
        choose center coordinates in EPSG3031 [y,x], by default [0,0]
    zoom : float, optional
        choose zoom level, by default 0
    display_xy : bool, optional
        choose if you want clicks to show the xy location, by default True
    show : bool, optional
        choose whether to display the map, by default True
    points : pandas.DataFrame, optional
        choose to plot points supplied as columns x, y, in EPSG:3031 in a dataframe
    basemap_type : str, optional
        choose what basemap to plot, options are 'BlueMarble', 'Imagery', and 'Basemap',
        by default 'BlueMarble'

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
            center_ll = [-90, 0]
        elif hemisphere == "north":
            center_ll = [90, -45]
        else:
            msg = "hemisphere must be north or south"
            raise ValueError(msg)
    if center_yx is not None:
        if hemisphere == "south":
            center_ll = utils.epsg3031_to_latlon(center_yx)
        elif hemisphere == "north":
            center_ll = utils.epsg3413_to_latlon(center_yx)
        else:
            msg = "hemisphere must be north or south"
            raise ValueError(msg)

    if hemisphere == "south":
        if basemap_type == "BlueMarble":
            base = ipyleaflet.basemaps.NASAGIBS.BlueMarble3031  # pylint: disable=no-member
            proj = ipyleaflet.projections.EPSG3031.NASAGIBS
        elif basemap_type == "Imagery":
            base = ipyleaflet.basemaps.Esri.AntarcticImagery  # pylint: disable=no-member
            proj = ipyleaflet.projections.EPSG3031.ESRIImagery
        elif basemap_type == "Basemap":
            base = ipyleaflet.basemaps.Esri.AntarcticBasemap  # pylint: disable=no-member
            proj = ipyleaflet.projections.EPSG3031.ESRIBasemap
        else:
            msg = "invalid string for basemap_type"
            raise ValueError(msg)
    elif hemisphere == "north":
        if basemap_type == "BlueMarble":
            base = ipyleaflet.basemaps.NASAGIBS.BlueMarble3413  # pylint: disable=no-member
            proj = ipyleaflet.projections.EPSG3413.NASAGIBS
        # elif basemap_type == "Imagery":
        #     base = ipyleaflet.basemaps.Esri.ArcticImagery  # pylint: disable=no-member
        #     proj = ipyleaflet.projections.EPSG5936.ESRIImagery
        # elif basemap_type == "Basemap":
        #     base = ipyleaflet.basemaps.Esri.OceanBasemap  # pylint: disable=no-member
        #     proj = ipyleaflet.projections.EPSG5936.ESRIBasemap
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
    region : tuple[float, float, float, float], optional
        choose to subset the grids to a specified region, in format
        [xmin, xmax, ymin, ymax], by default None
    dims : tuple, optional
        customize the subplot dimensions (# rows, # columns), by default will use
        `utils.square_subplots()` to make a square(~ish) layout.
    hemisphere : str, optional
        choose between plotting in the "north" or "south" hemispheres, by default None
    Returns
    -------
    pygmt.Figure
        Returns a figure object, which can be used by other PyGMT plotting functions.

    """
    # if no define region, get from first grid in list
    if region is None:
        try:
            region = utils.get_grid_info(grids[0])[1]
        except Exception as e:  # pylint: disable=broad-exception-caught
            # pygmt.exceptions.GMTInvalidInput:
            logging.exception(e)
            logging.warning("grid region can't be extracted, using antarctic region.")
            region = regions.antarctica

    region = typing.cast(tuple[float, float, float, float], region)

    # get square dimensions for subplot
    subplot_dimensions = utils.square_subplots(len(grids)) if dims is None else dims

    # set subplot projection and size from input region and figure dimensions
    # by default use figure height to set projection
    if kwargs.get("fig_width") is None:
        _proj, _proj_latlon, fig_width, fig_height = utils.set_proj(
            region,
            fig_height=kwargs.pop("fig_height", 15),
            hemisphere=hemisphere,
        )
    # if fig_width is set, use it to set projection
    else:
        _proj, _proj_latlon, fig_width, fig_height = utils.set_proj(
            region,
            fig_width=kwargs.get("fig_width"),
            hemisphere=hemisphere,
        )

    # initialize figure
    fig = pygmt.Figure()

    with fig.subplot(
        nrows=subplot_dimensions[0],
        ncols=subplot_dimensions[1],
        subsize=(fig_width, fig_height),
        frame=kwargs.get("frame", "f"),
        clearance=kwargs.get("clearance"),  # edges of figure
        title=kwargs.get("fig_title"),
        margins=kwargs.get("margins", "0.5c"),  # between suplots
        autolabel=kwargs.get("autolabel"),
    ):
        for i, j in enumerate(grids):
            with fig.set_panel(panel=i):
                # if list of cmaps provided, use them
                cmaps = kwargs.get("cmaps")
                cmap = cmaps[i] if cmaps is not None else "viridis"

                # if list of titles provided, use them
                subplot_titles = kwargs.get("subplot_titles")
                sub_title = subplot_titles[i] if subplot_titles is not None else None

                # if list of colorbar labels provided, use them
                cbar_labels = kwargs.get("cbar_labels")
                cbar_label = cbar_labels[i] if cbar_labels is not None else " "

                # if list of colorbar units provided, use them
                cbar_units = kwargs.get("cbar_units")
                cbar_unit = cbar_units[i] if cbar_units is not None else " "

                # if list of cmaps limits provided, use them
                cpt_limits = kwargs.get("cpt_limits")
                cpt_lims = cpt_limits[i] if cpt_limits is not None else None

                # plot the grids
                plot_grd(
                    j,
                    fig=fig,
                    fig_height=fig_height,
                    origin_shift=None,
                    region=region,
                    cmap=cmap,
                    title=sub_title,
                    cbar_label=cbar_label,
                    cbar_unit=cbar_unit,
                    cpt_lims=cpt_lims,
                    hemisphere=hemisphere,
                    **kwargs,
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
        coast_gdf = gpd.read_file(fetch.groundingline(version="BAS"))
        # crsys=crs.epsg(3413)
        crsys = crs.NorthPolarStereo()
    elif hemisphere == "south":
        coast_gdf = gpd.read_file(fetch.groundingline(version="depoorter-2013"))
        # crsys=crs.epsg(3031)
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

    if len(points.columns) < 3:
        # if only 2 cols are given, give points a constant color
        # turn points into geoviews dataset
        gv_points = gv.Points(
            points,
            crs=crs.SouthPolarStereo(),
        )

        # change options
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
        # if more than 2 columns, color points by third column
        # turn points into geoviews dataset
        gv_points = gv.Points(
            data=points,
            vdims=[points_z],
            crs=crs.SouthPolarStereo(),
        )

        # change options
        gv_points.opts(
            color=points_z,
            cmap=points_cmap,
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
