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

# import polartoolkit.fetch as fetch
from polartoolkit import fetch, regions, utils

# import polartoolkit.regions as regions
# import polartoolkit.utils as utils

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


def basemap(
    region: tuple[float, float, float, float] | None = None,
    fig_height: float = 15,
    fig_width: float | None = None,
    origin_shift: str = "initialize",
    **kwargs: typing.Any,
) -> pygmt.Figure:
    """
    create a blank basemap figure, or add a basemap to an existing figure / subplot.

    Parameters
    ----------
    region : tuple[float, float, float, float] | None, optional
        bounding region in GMT format, by default None
    fig_height : float, optional
        height of figure, by default 15
    fig_width : float | None, optional
        width of figure, by default None
    origin_shift : str, optional
        choose to start new figure, or shift origin of existing figure to add a subplot,
        by default "initialize"

    Returns
    -------
    pygmt.Figure
        a new or update figure instance with a basemap.
    """
    # if region not set, use antarctic region
    if region is None:
        region = regions.antarctica

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
        frame=kwargs.get("frame", "nwse+gwhite"),
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
            version=kwargs.get("coast_version", "depoorter-2013"),
        )

    # add lat long grid lines
    if kwargs.get("gridlines", False) is True:
        add_gridlines(
            fig,
            region=region,
            projection=proj_latlon,
            x_spacing=kwargs.get("x_spacing", None),
            y_spacing=kwargs.get("y_spacing", None),
        )

    # add inset map to show figure location
    if kwargs.get("inset", False) is True:
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
            # inset_pos=kwargs.get("inset_pos", "TL"),
            **new_kwargs,
        )

    # add scalebar
    if kwargs.get("scalebar", False) is True:
        add_scalebar(
            fig=fig,
            region=region,
            projection=proj_latlon,
            font_color=kwargs.get("scale_font_color", "black"),
            scale_length=kwargs.get("scale_length"),
            length_perc=kwargs.get("scale_length_perc", 0.25),
            position=kwargs.get("scale_position", "n.5/.05"),
            **kwargs,
        )

    # blank plotting call to reset projection to EPSG:3031, optionally add title
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
    grid: str | xr.DataArray,
    cmap: str | bool = "viridis",
    region: tuple[float, float, float, float] | None = None,
    coast: bool = False,
    origin_shift: str = "initialize",
    fig: pygmt.Figure | None = None,
    **kwargs: typing.Any,
) -> pygmt.Figure:
    r"""
    Helps easily create PyGMT maps, individually or as subplots.

    Parameters
    ----------
    grid : str or xr.DataArray
        grid file to plot, either loaded xr.DataArray or string of a filename
    cmap : str or bool, optional
        GMT color scale to use, by default 'viridis'
    region : tuple[float, float, float, float], optional
        region to plot, by default is extent of input grid
    coast : bool, optional
        choose whether to plot Antarctic coastline and grounding line, by default False
    origin_shift : str, optional
        automatically will create a new figure, set to 'xshift' to instead add plot to
        right of previous plot, or 'yshift' to add plot above previous plot, by
        default 'initialize'.
    fig : pygmt.Figure(), optional
        supply figure instance for adding subplots or other PyGMT plotting methods, by
        default None

    Keyword Args
    ------------
    image : bool
        set to True if plotting imagery to correctly set colorscale.
    grd2cpt : bool
        use GMT module grd2cpt to set color scale from grid values, by default is False
    cmap_region : str or tuple[float, float, float, float]
        region to use to define color scale if grd2cpt is True, by default is
        region
    cbar_label : str
        label to add to colorbar.
    points : pd.DataFrame
        points to plot on map, must contain columns 'x' and 'y'.
    show_region : tuple[float, float, float, float]
        GMT-format region to use to plot a bounding regions.
    cpt_lims : str or tuple]
        limits to use for color scale max and min, by default is max and min of data.
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
    >>> from polartoolkit import maps
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

    # get region from grid or use supplied region
    if region is None:
        try:
            region = utils.get_grid_info(grid)[1]
        except Exception as e:  # pylint: disable=broad-exception-caught
            # pygmt.exceptions.GMTInvalidInput:
            logging.exception(e)
            logging.warning("grid region can't be extracted, using antarctic region.")
            region = regions.antarctica

    region = typing.cast(tuple[float, float, float, float], region)

    # initialize figure or shift for new subplot
    if origin_shift == "initialize":
        fig = pygmt.Figure()
        fig_height = kwargs.get("fig_height", 15)
        fig_width = kwargs.get("fig_width", None)
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
    else:
        if origin_shift == "xshift":
            fig_height = kwargs.get("fig_height", utils.get_fig_height())
            proj, proj_latlon, fig_width, fig_height = utils.set_proj(
                region,
                fig_height=fig_height,
            )
            fig.shift_origin(  # type: ignore[union-attr]
                xshift=(kwargs.get("xshift_amount", 1) * (fig_width + 0.4))
            )
        elif origin_shift == "yshift":
            fig_height = kwargs.get("fig_height", utils.get_fig_height())
            proj, proj_latlon, fig_width, fig_height = utils.set_proj(
                region,
                fig_height=fig_height,
            )
            fig.shift_origin(yshift=(kwargs.get("yshift_amount", 1) * (fig_height + 3)))  # type: ignore[union-attr]
        elif origin_shift == "both_shift":
            fig_height = kwargs.get("fig_height", utils.get_fig_height())
            proj, proj_latlon, fig_width, fig_height = utils.set_proj(
                region,
                fig_height=fig_height,
            )
            fig.shift_origin(  # type: ignore[union-attr]
                xshift=(kwargs.get("xshift_amount", 1) * (fig_width + 0.4)),
                yshift=(kwargs.get("yshift_amount", 1) * (fig_height + 3)),
            )
        elif origin_shift == "no_shift":
            proj, proj_latlon, fig_width, fig_height = utils.set_proj(
                region,
                fig_height=kwargs.get("fig_height", 15),
            )

        else:
            msg = "invalid string for origin shift"
            raise ValueError(msg)

    cmap_region = kwargs.get("cmap_region", region)
    show_region = kwargs.get("show_region", None)
    robust = kwargs.get("robust", False)
    cpt_lims = kwargs.get("cpt_lims", None)
    grd2cpt = kwargs.get("grd2cpt", False)
    image = kwargs.get("image", False)
    gridlines = kwargs.get("gridlines", False)
    points = kwargs.get("points", None)
    inset = kwargs.get("inset", False)
    title = kwargs.get("title", None)
    scalebar = kwargs.get("scalebar", False)
    north_arrow = kwargs.get("north_arrow", False)
    reverse_cpt = kwargs.get("reverse_cpt", False)
    colorbar = kwargs.get("colorbar", True)
    shp_mask = kwargs.get("shp_mask", None)

    if kwargs.get("imagery_basemap", False) is True:
        fig.grdimage(  # type: ignore[union-attr]
            grid=fetch.imagery(),
            cmap=None,
            projection=proj,
            region=region,
        )

    # set cmap
    if cmap is True:
        # use cmap from most recent pygmt session
        pass
    elif image is True:
        # create a cmap to use with imagery
        pygmt.makecpt(
            cmap=cmap,
            series="15000/17000/1",
            verbose="e",
        )
        colorbar = False
    elif grd2cpt is True:
        if cpt_lims is None and isinstance(grid, (xr.DataArray)):
            zmin, zmax = utils.get_min_max(grid, shp_mask, robust=robust)
        elif cpt_lims is None and isinstance(grid, (str)):
            with xr.load_dataarray(grid) as da:
                zmin, zmax = utils.get_min_max(da, shp_mask, robust=robust)
        else:
            zmin, zmax = cpt_lims
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
    elif cpt_lims is not None:
        zmin, zmax = cpt_lims
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
    else:
        try:
            if isinstance(grid, (xr.DataArray)):
                zmin, zmax = utils.get_min_max(grid, shp_mask, robust=robust)
            else:
                with xr.load_dataarray(grid) as da:
                    zmin, zmax = utils.get_min_max(da, shp_mask, robust=robust)
            pygmt.makecpt(
                cmap=cmap,
                background=True,
                continuous=kwargs.get("continuous", True),
                series=(zmin, zmax),
                reverse=reverse_cpt,
                verbose="e",
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            # pygmt.exceptions.GMTInvalidInput:
            logging.exception(e)
            logging.warning("grid region can't be extracted.")
            pygmt.makecpt(
                cmap=cmap,
                background=True,
                continuous=kwargs.get("continuous", True),
                reverse=reverse_cpt,
                verbose="e",
            )

    # display grid
    fig.grdimage(  # type: ignore[union-attr]
        grid=grid,
        cmap=True,
        projection=proj,
        region=region,
        nan_transparent=True,
        frame=kwargs.get("frame", None),
        shading=kwargs.get("shading", None),
        transparency=kwargs.get("transparency", 0),
        verbose="q",
    )

    cmap_region = kwargs.get("cmap_region", region)

    # add datapoints
    if points is not None:
        fig.plot(  # type: ignore[union-attr]
            x=points.x,
            y=points.y,
            style=kwargs.get("points_style", "c.2c"),
            fill=kwargs.get("points_fill", "black"),
            pen=kwargs.get("points_pen", "1p,black"),
            cmap=kwargs.get("points_cmap", None),
        )

    # add box showing region
    if show_region is not None:
        add_box(fig, show_region)

    # plot groundingline and coastlines
    if coast is True:
        add_coast(
            fig,
            region,
            proj,
            pen=kwargs.get("coast_pen", None),
            no_coast=kwargs.get("no_coast", False),
            version=kwargs.get("coast_version", "depoorter-2013"),
        )

    # plot faults
    if kwargs.get("add_faults", False) is True:
        add_faults(
            fig=fig,
            region=region,
            projection=proj,
            label=kwargs.get("fault_label", None),
            pen=kwargs.get("fault_pen", None),
            style=kwargs.get("fault_style", None),
            fault_activity=kwargs.get("fault_activity", None),
            fault_motion=kwargs.get("fault_motion", None),
            fault_exposure=kwargs.get("fault_exposure", None),
        )

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
            # inset_pos=kwargs.get("inset_pos", "TL"),
            **new_kwargs,
        )

    # add scalebar
    if scalebar is True:
        add_scalebar(
            fig=fig,
            region=region,
            projection=proj_latlon,
            font_color=kwargs.get("font_color", "black"),
            scale_length=kwargs.get("scale_length"),
            length_perc=kwargs.get("length_perc", 0.25),
            position=kwargs.get("scale_position", "n.5/.05"),
            **kwargs,
        )

    # add north arrow
    if north_arrow is True:
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
                "hist",
                "grid",
                "fig",
            ]
        }
        add_colorbar(
            fig,
            hist=kwargs.get("hist", False),
            grid=grid,
            cpt_lims=(zmin, zmax),
            fig_width=fig_width,
            region=region,
            **cbar_kwargs,
        )

    # reset region and projection
    if title is None:
        fig.basemap(region=region, projection=proj, frame="wesn")  # type: ignore[union-attr]
    else:
        with pygmt.config(FONT_TITLE=kwargs.get("title_font", "auto")):
            fig.basemap(region=region, projection=proj, frame=f"wesn+t{title}")  # type: ignore[union-attr]

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
    text_location = kwargs.get("cbar_text_location", None)

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
            log=kwargs.get("cbar_log", None),
        )

    # add histogram to colorbar
    # Note, depending on data and hist_type, you may need to manually set kwarg
    # `hist_ymax` to an appropriate value
    if hist is True:
        # get grid to use
        grid = kwargs.get("grid", None)

        if grid is None:
            msg = "if hist is True, grid must be provided."
            raise ValueError(msg)

        # clip grid to plot region
        region = kwargs.get("region", None)
        # if no region supplied, get region of current PyGMT figure
        if region is None:
            with pygmt.clib.Session() as lib:
                region = list(lib.extract_region())
                assert len(region) == 4

        # clip grid to plot region
        if region != utils.get_grid_info(grid)[1]:
            # grid = fetch.resample_grid(grid, region=region)
            ew = [region[0], region[1]]
            ns = [region[2], region[3]]
            grid_clipped = grid.sel(
                {
                    list(grid.sizes.keys())[1]: slice(min(ew), max(ew)),
                    list(grid.sizes.keys())[0]: slice(max(ns), min(ns)),  # noqa: RUF015
                }
            )
            # if subplotting, region will be in figure units and grid will be clipped
            # incorrectly, hacky solution is to check if clipped figure is smaller than
            # a few data points, if so, use grids full region
            if len(grid_clipped[list(grid_clipped.sizes.keys())[0]].values) < 5:  # noqa: RUF015
                reg = kwargs.get("region", None)
                if reg is None:
                    msg = (
                        "Issue with detecting figure region for adding colorbar "
                        "histogram, please provide region kwarg."
                    )
                    raise ValueError(msg)
                grid_clipped = grid.sel(
                    {
                        list(grid.sizes.keys())[1]: slice(reg[0], reg[1]),
                        list(grid.sizes.keys())[0]: slice(reg[2], reg[3]),  # noqa: RUF015
                    }
                )
            grid = grid_clipped

        if (cpt_lims is None) or (np.isnan(cpt_lims).any()):
            warnings.warn(
                "getting max/min values from grid, if cpt_lims were used to create the "
                "colorscale, histogram will not properly align with colorbar!",
                stacklevel=2,
            )
            zmin, zmax = utils.get_min_max(
                grid, kwargs.get("shp_mask", None), robust=kwargs.get("robust", False)
            )
        else:
            zmin, zmax = cpt_lims

        # get grid's data for histogram
        df = vd.grid_to_table(grid)
        df2 = df.iloc[:, -1:].squeeze()

        # subset between cbar min and max
        data = df2[df2.between(zmin, zmax)]

        bin_width = kwargs.get("hist_bin_width", None)
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

        assert zmin != zmax, "Grids are all the same value!"

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
                cmap=kwargs.get("hist_cmap", True),
                fill=kwargs.get("hist_fill", None),
                pen=kwargs.get("hist_pen", "default"),
                barwidth=kwargs.get("hist_barwidth", None),
                center=kwargs.get("hist_center", False),
                distribution=kwargs.get("hist_distribution", False),
                cumulative=kwargs.get("hist_cumulative", False),
                extreme=kwargs.get("hist_extreme", "b"),
                stairs=kwargs.get("hist_stairs", False),
                # horizontal=kwargs.get('hist_horizontal', False),
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
    region: tuple[float, float, float, float] | None = None,
    projection: str | None = None,
    no_coast: bool = False,
    pen: str | None = None,
    version: str = "depoorter-2013",
) -> None:
    """
    add coastline and groundingline to figure.

    Parameters
    ----------
    fig : pygmt.Figure
    region : tuple[float, float, float, float], optional
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

    if version == "depoorter-2013":
        gdf = gpd.read_file(fetch.groundingline(version=version))
        if no_coast is False:
            data = gdf
        elif no_coast is True:
            data = gdf[gdf.Id_text == "Grounded ice or land"]
    elif version == "measures-v2":
        gl = gpd.read_file(fetch.groundingline(version=version))
        if no_coast is False:
            coast = gpd.read_file(fetch.measures_boundaries(version="Coastline"))
            data = pd.concat([gl, coast])
        elif no_coast is True:
            data = gpd.read_file(fetch.groundingline(version=version))

    fig.plot(
        data,
        projection=projection,
        region=region,
        pen=pen,
    )


def add_gridlines(
    fig: pygmt.Figure,
    region: tuple[float, float, float, float] | None = None,
    projection: str | None = None,
    **kwargs: typing.Any,
) -> None:
    """
    add lat lon grid lines and annotations to a figure. Use kwargs x_spacing and
    y_spacing to customize the interval of gridlines and annotations.

    Parameters
    ----------
    fig : pygmt.Figure instance
    region : tuple[float, float, float, float], optional
        region for the figure
    projection : str, optional
        GMT projection string in lat lon, if your previous pygmt.Figure() call used a
        cartesian projection, you will need to provide a projection in lat/lon here, use
        utils.set_proj() to make this projection.

    """

    x_spacing = kwargs.get("x_spacing", None)
    y_spacing = kwargs.get("y_spacing", None)

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
            region=region_converted,
            frame=[
                "NSWE",
                x_frames[0],
                y_frames[0],
            ],
            transparency=50,
            # verbose="q",
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
                # verbose="q",
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
    fig : pygmt.Figure instance
    region : tuple[float, float, float, float], optional
        region for the figure
    projection : str, optional
        GMT projection string in lat lon, if your previous pygmt.Figure() call used a
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
    region: tuple[float, float, float, float] | None = None,
    inset_pos: str = "TL",
    inset_width: float = 0.25,
    inset_reg: tuple[float, float, float, float] = (-2800e3, 2800e3, -2800e3, 2800e3),
    **kwargs: typing.Any,
) -> None:
    """
    add an inset map showing the figure region relative to the Antarctic continent.

    Parameters
    ----------
    fig : pygmt.Figure instance
    region : tuple[float, float, float, float], optional
        region for the figure
    inset_pos : str, optional
        GMT location string for inset map, by default 'TL' (top left)
    inset_width : float, optional
        Inset width as percentage of the total figure width, by default is 25% (0.25)
    inset_reg : tuple[float, float, float, float], optional
        Region of Antarctica to plot for the inset map, by default is whole continent
    """

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
        gdf = gpd.read_file(fetch.groundingline())
        fig.plot(
            projection=inset_map,
            region=inset_reg,
            data=gdf[gdf.Id_text == "Ice shelf"],
            fill="skyblue",
        )
        fig.plot(data=gdf[gdf.Id_text == "Grounded ice or land"], fill="grey")
        fig.plot(
            data=fetch.groundingline(), pen=kwargs.get("inset_coast_pen", "0.2,black")
        )

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
    fig : pygmt.Figure instance
    region : tuple[float, float, float, float], optional
        region for the figure
    projection : str, optional
        GMT projection string in lat lon, if your previous pygmt.Figure() call used a
        cartesian projection, you will need to provide a projection in lat/lon here, use
        utils.set_proj() to make this projection.

    """
    font_color = kwargs.get("font_color", "black")
    scale_length = kwargs.get("scale_length", None)
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
            map_scale=f'{position}+w{scale_length}k+f+l"km"+ar',
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
    fig : pygmt.Figure instance
    region : tuple[float, float, float, float], optional
        region for the figure
    projection : str, optional
        GMT projection string in lat lon, if your previous pygmt.Figure() call used a
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
        verbose="e",
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
        choose whether to display the map, by default True
    points : pd.DataFrame, optional
        choose to plot points supplied as columns x, y, in EPSG:3031 in a dataframe
    basemap_type : str, optional
        choose what basemap to plot, options are 'BlueMarble', 'Imagery', and 'Basemap',
        by default 'BlueMarble'
    """

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
        height=kwargs.get("height", None),
    )

    # if points are supplied, center map on them and plot them
    if points is not None:
        if kwargs.get("points_as_latlon", False) is True:
            center_ll = [points.lon.mean(), points.lat.mean()]
        else:
            # convert points to lat lon
            points_ll: pd.DataFrame = utils.epsg3031_to_latlon(points)
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

    if basemap_type == "BlueMarble":
        base = ipyleaflet.basemaps.NASAGIBS.BlueMarble3031  # pylint: disable=no-member
        proj = ipyleaflet.projections.EPSG3031.NASAGIBS
    elif basemap_type == "Imagery":
        base = ipyleaflet.basemaps.Esri.AntarcticImagery  # pylint: disable=no-member
        proj = ipyleaflet.projections.EPSG3031.ESRIImagery
    elif basemap_type == "Basemap":
        base = ipyleaflet.basemaps.Esri.AntarcticBasemap  # pylint: disable=no-member
        proj = ipyleaflet.projections.EPSG3031.ESRIBasemap

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
                label_xy.value = str(utils.latlon_to_epsg3031(latlon))

    m.on_interaction(handle_click)

    if show is True:
        display(m)

    return m


def subplots(
    grids: list[xr.DataArray],
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
        list of xr.DataArray's to be plotted
    region : tuple[float, float, float, float], optional
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
    if kwargs.get("fig_width", None) is None:
        _proj, _proj_latlon, fig_width, fig_height = utils.set_proj(
            region,
            fig_height=kwargs.pop("fig_height", 15),
        )
    # if fig_width is set, use it to set projection
    else:
        _proj, _proj_latlon, fig_width, fig_height = utils.set_proj(
            region,
            fig_width=kwargs.get("fig_width", None),
        )

    # initialize figure
    fig = pygmt.Figure()

    with fig.subplot(
        nrows=subplot_dimensions[0],
        ncols=subplot_dimensions[1],
        subsize=(fig_width, fig_height),
        frame=kwargs.get("frame", "f"),
        clearance=kwargs.get("clearance", None),  # edges of figure
        title=kwargs.get("fig_title", None),
        margins=kwargs.get("margins", "0.5c"),  # between suplots
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

                # if list of cmaps limits provided, use them
                if kwargs.get("cpt_limits", None) is not None:
                    cpt_lims = kwargs.get("cpt_limits", None)[i]
                else:
                    cpt_lims = None

                # plot the grids
                plot_grd(
                    j,
                    fig=fig,
                    fig_height=fig_height,
                    origin_shift="no_shift",
                    region=region,
                    cmap=cmap,
                    title=sub_title,
                    cbar_label=cbar_label,
                    cbar_unit=cbar_unit,
                    cpt_lims=cpt_lims,
                    **kwargs,
                )
    return fig


def plot_3d(
    grids: list[xr.DataArray],
    cmaps: list[str],
    exaggeration: list[float],
    view: tuple[float, float] = (170, 30),
    vlims: tuple[float, float] = (-10000, 1000),
    region: tuple[float, float, float, float] | None = None,
    shp_mask: str | gpd.GeoDataFrame | None = None,
    polygon_mask: list[float] | None = None,
    colorbar: bool = True,
    grd2cpt: bool = True,
    **kwargs: typing.Any,
) -> pygmt.Figure:
    """
    create a 3D perspective plot of a list of grids

    Parameters
    ----------
    grids : list
        xarray DataArrays to be plotted in 3D
    cmaps : list
        list of PyGMT colormap names to use for each grid
    exaggeration : list
        list of vertical exaggeration factors to use for each grid
    view : list, optional
        list of azimuth and elevation angles for the view, by default [170, 30]
    vlims : list, optional
        list of vertical limits for the plot, by default [-10000, 1000]
    region : tuple[float, float, float, float], optional
        region for the plot, by default None
    shp_mask : Union[str or gpd.GeoDataFrame], optional
        shapefile or geodataframe to clip the grids with, by default None
    cpt_lims : list, optional
        list of colorbar limits for each grid, by default None
    colorbar : bool, optional
        whether to plot a colorbar, by default True

    Returns
    -------
    PyGMT.Figure()
        Returns a figure object, which can be used by other PyGMT plotting functions.
    """
    fig_height = kwargs.get("fig_height", 15)
    fig_width = kwargs.get("fig_width", None)

    # if plot region not specified, try to pull from grid info
    if region is None:
        try:
            region = utils.get_grid_info(grids[0])[1]
        except Exception as e:  # pylint: disable=broad-exception-caught
            # pygmt.exceptions.GMTInvalidInput:
            logging.exception(e)
            logging.warning(
                "first grids' region can't be extracted, using antarctic region."
            )
            region = regions.antarctica

    region = typing.cast(tuple[float, float, float, float], region)

    # set figure projection and size from input region and figure dimensions
    # by default use figure height to set projection
    if fig_width is None:
        proj, _proj_latlon, fig_width, fig_height = utils.set_proj(
            region,
            fig_height=fig_height,
        )
    # if fig_width is set, use it to set projection
    else:
        proj, _proj_latlon, fig_width, fig_height = utils.set_proj(
            region,
            fig_width=fig_width,
        )
    # set vertical limits
    new_region = region + vlims

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
            pathlib.Path("tmp.nc").unlink()
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
                cpt_lims = kwargs.get("cpt_lims", None)
                if cpt_lims is None:
                    zmin, zmax = (
                        utils.get_grid_info(grid)[2],
                        utils.get_grid_info(grid)[3],
                    )
                else:
                    zmin, zmax = cpt_lims[i]
                pygmt.makecpt(
                    cmap=cmaps[i],
                    background=True,
                    continuous=True,
                    series=(zmin, zmax),
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                # pygmt.exceptions.GMTInvalidInput:
                logging.exception(e)
                logging.warning("grid region can't be extracted.")
                pygmt.makecpt(
                    cmap=cmaps[i],
                    background=True,
                    continuous=True,
                )

        # set transparency values
        transparencies = kwargs.get("transparencies", None)
        transparency = 0 if transparencies is None else transparencies[i]

        # plot as perspective view
        fig.grdview(
            grid=grid,
            cmap=True,
            projection=proj,
            region=new_region,
            frame=None,
            perspective=view,
            zsize=f"{exaggeration[i]}c",
            surftype="c",
            transparency=transparency,
            # plane='-9000+ggrey',
            shading=kwargs.get("shading", True),
        )

        # display colorbar
        if colorbar is True:
            cbar_xshift = kwargs.get("cbar_xshift", None)
            cbar_yshift = kwargs.get("cbar_yshift", None)

            xshift = 0 if cbar_xshift is None else cbar_xshift[i]

            yshift = fig_height / 2 if cbar_yshift is None else cbar_yshift[i]

            fig.shift_origin(yshift=f"{yshift}c", xshift=f"{xshift}c")
            cbar_labels = kwargs.get("cbar_labels", None)
            cbar_label = " " if cbar_labels is None else cbar_labels[i]
            fig.colorbar(
                cmap=True,
                position=f"jMR+w{fig_width*.4}c/.5c+v+e+m",
                frame=f"xaf+l{cbar_label}",
                perspective=True,
                box="+gwhite+c3p",
            )
            fig.shift_origin(yshift=f"{-yshift}c", xshift=f"{-xshift}c")

        # shift up for next grid
        zshifts: list[float] | None = kwargs.get("zshifts", None)
        if zshifts is None:
            fig.shift_origin(yshift=f"{fig_height/2}c")
        else:
            fig.shift_origin(yshift=f"{zshifts[i]}c")

    return fig


def interactive_data(
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
    coast : bool, optional
        choose whether to plot Antarctic coastline data, by default True
    grid : xr.DataArray, optional
        display a grid on the map, by default None
    grid_cmap : str, optional
        colormap to use for the grid, by default 'inferno'
    points : pd.DataFrame, optional
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
    """
    # Example
    # -------

    # image = maps.interactive_data(
    #     grid = bedmap2_bed,
    #     points = point_data,
    #     points_z = 'z_ellipsoidal',
    #     )

    # image
    #     >>> from polartoolkit import maps, regions, fetch
    #     ...
    #     >>> bedmap2_bed = fetch.bedmap2(layer='bed', region=regions.ross_ice_shelf)
    #     >>> GHF_point_data = fetch.ghf(version='burton-johnson-2020', points=True)
    #     ...
    #     >>> image = maps.interactive_data(
    #     ... grid = bedmap2_bed,
    #     ... points = GHF_point_data[['x','y','GHF']],
    #     ... points_z = 'GHF',
    #     ... )
    #     >>> image
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
    coast_fig = gv.Path(
        gpd.read_file(fetch.groundingline()),
        crs=crs.SouthPolarStereo(),
    )
    # set projection, and change groundingline attributes
    coast_fig.opts(
        projection=crs.SouthPolarStereo(),
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
            crs=crs.SouthPolarStereo(),
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
    _summary_

    Parameters
    ----------
    points : pd.DataFrame
        _description_, by default None
    points_z : str | None, optional
        _description_, by default None
    points_color : str, optional
        _description_, by default "red"
    points_cmap : str, optional
        _description_, by default "viridis"

    Returns
    -------
    gv.Points
        _description_

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
