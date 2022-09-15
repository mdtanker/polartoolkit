# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#

import warnings
from typing import TYPE_CHECKING, Union
from math import log10, floor

import pygmt
import pyogrio

from antarctic_plots import fetch, utils

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr


def plot_grd(
    grid: Union[str or xr.DataArray],
    cmap: str = "viridis",
    plot_region: Union[str or np.ndarray] = None,
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
    plot_region : Union[str or np.ndarray], optional
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
        plot_region
    cbar_label : str
        label to add to colorbar.
    points : pd.DataFrame
        points to plot on map, must contain columns 'x' and 'y'.
    square : np.ndarray
        GMT-format region to use to plot a square.
    cpt_lims : Union[str or tuple]
        limits to use for color scale max and min, by default is max and min of data.
    fig : pygmt.Figure()
        if adding subplots, set the first returned figure to a variable, and add that
        variable as the kwargs 'fig' to subsequent calls to plot_grd.
    grid_lines : bool
        choose to plot lat/long grid lines, by default is False
    inset : bool
        choose to plot inset map showing figure location, by default is False
    inset_pos : str
        position for inset map; either 'TL', 'TR', BL', 'BR', by default is 'TL'
    fig_height : int or float
        height in cm for figures, by default is 15cm.

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

    if plot_region is None:
        plot_region = utils.get_grid_info(grid)[1]

    cmap_region = kwargs.get("cmap_region", plot_region)
    square = kwargs.get("square", None)
    cpt_lims = kwargs.get("cpt_lims", None)
    grd2cpt = kwargs.get("grd2cpt", False)
    image = kwargs.get("image", False)
    grid_lines = kwargs.get("grid_lines", False)
    points = kwargs.get("points", None)
    inset = kwargs.get("inset", False)
    title = kwargs.get("title", None)
    fig_height = kwargs.get("fig_height", 15)
    scalebar = kwargs.get('scalebar', False)

    # set figure projection and size from input region
    proj, proj_latlon, fig_width, fig_height = utils.set_proj(plot_region, fig_height)

    # initialize figure or shift for new subplot
    if origin_shift == "initialize":
        fig = pygmt.Figure()
    elif origin_shift == "xshift":
        fig = kwargs.get("fig")
        fig.shift_origin(xshift=(fig_width + 0.4))
    elif origin_shift == "yshift":
        fig = kwargs.get("fig")
        fig.shift_origin(yshift=(fig_height + 3))

    # set cmap
    if image is True:
        pygmt.makecpt(
            cmap=cmap,
            series="15000/17000/1",
        )
    elif grd2cpt is True:
        pygmt.grd2cpt(
            cmap=cmap,
            grid=grid,
            region=cmap_region,
            background=True,
            continuous=True,
        )
    elif cpt_lims is not None:
        pygmt.makecpt(
            cmap=cmap,
            background=True,
            # continuous=True,
            series=cpt_lims,
        )
    else:
        zmin, zmax = utils.get_grid_info(grid)[2], utils.get_grid_info(grid)[3]
        pygmt.makecpt(
            cmap=cmap,
            background=True,
            continuous=True,
            series=(zmin, zmax),
        )

    # display grid
    fig.grdimage(
        grid=grid,
        cmap=True,
        projection=proj,
        region=plot_region,
        nan_transparent=True,
        frame=[f'+gwhite'],
        verbose="q",
    )

    # display colorbar
    if image is not True:
        fig.colorbar(
            cmap=True,
            position=f"jBC+w{fig_width*.8}c+jTC+h+o0c/.2c+e",
            frame=f"xaf+l{kwargs.get('cbar_label',' ')}",
        )

    # plot groundingline and coastlines
    if coast is True:
        add_coast(
            fig, 
            plot_region,
            proj, 
            no_coast = kwargs.get('no_coast', False)
            )
        # fig.plot(
        #     data=fetch.groundingline(),
        #     pen=".6p,black",
        # )

    # add datapoints
    if points is not None:
        fig.plot(
            x=points.x,
            y=points.y,
            style="c1.2p",
            color="black",
        )

    # add square
    if square is not None:
        fig.plot(
            x=[square[0], square[0], square[1], square[1], square[0]],
            y=[square[2], square[3], square[3], square[2], square[2]],
            pen="2p,black",
        )

    # add lat long grid lines
    if grid_lines is True:
        add_gridlines(fig, plot_region, proj_latlon, 
            x_annots = kwargs.get("x_annots", 30),
            y_annots = kwargs.get("y_annots", 4),
        )

    # add inset map to show figure location
    if inset is True:
        add_inset(fig, plot_region, fig_width, kwargs.get("inset_pos", "TL"))

    # add scalebar
    if scalebar is True:
        add_scalebar(fig, plot_region, proj_latlon, 
            font_color=kwargs.get('font_color', 'black'), 
            scale_length=kwargs.get('scale_length'),
            length_perc=kwargs.get('length_perc', .25),
            position=kwargs.get('position', "n.5/.05"),
            )
    # reset region and projection
    if title is None:
        fig.basemap(region=plot_region, projection=proj, frame="wesn")
    else:
        fig.basemap(region=plot_region, projection=proj, frame=f'wesn+t{title}')

    return fig

def add_coast(
    fig: pygmt.figure,
    region: Union[str or np.ndarray],
    projection: str,
    no_coast: bool = False,
    pen: str = "0.6p,black",
):
    """
    add coastline and groundingline to figure.

    Parameters
    ----------
    fig : pygmt.figure
    region : Union[str or np.ndarray]
        region for the figure
    projection : str
        GMT projection string
    no_coast : bool
        If True, only plot groundingline, not coastline, by default is False
    pen : str, optional
        GMT pen string, by default "0.6p,black"
    """
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
    region: Union[str or np.ndarray],
    projection: str,
    x_annots: int =30,
    y_annots: int =4,
):
    """
    add lat lon grid lines and annotations to a figure.

    Parameters
    ----------
    fig : PyGMT.figure instance
    region : np.ndarray
        region for the figure
    projection : str
        GMT projection string in lat lon
    x_annots : int, optional
        interval for longitude lines in degrees, by default 30
    y_annots : int, optional
        interval for latitude lines in degrees, by default 4
    """
    with pygmt.config(
        MAP_ANNOT_OFFSET_PRIMARY="-2p",
        MAP_FRAME_TYPE="inside",
        MAP_ANNOT_OBLIQUE=0,
        FONT_ANNOT_PRIMARY="8p,black,-=2p,white",
        MAP_GRID_PEN_PRIMARY="gray",
        MAP_TICK_LENGTH_PRIMARY="-5p",
        MAP_TICK_PEN_PRIMARY="thinnest,gray",
        FORMAT_GEO_MAP="dddF",
        MAP_POLAR_CAP="90/90",
    ):
        fig.basemap(
            projection=projection,
            region=region,
            frame=[
                "NSWE",
                f"xa{x_annots}g{x_annots/2}",
                f"ya{y_annots}g{y_annots/2}",
            ],
            verbose="q",
        )
        with pygmt.config(FONT_ANNOT_PRIMARY="8p,black"):
            fig.basemap(
                projection=projection,
                region=region,
                frame=["NSWE", f"xa{x_annots}", f"ya{y_annots}"],
                verbose="q",
            )

def add_inset(
    fig: pygmt.figure,
    region: Union[str or np.ndarray],
    fig_width: Union[int, float], 
    inset_pos: str ='TL', 
):
    """
    add an inset map showing the figure region relative to the Antarctic continent.

    Parameters
    ----------
    fig : PyGMT.figure instance
    region : np.ndarray
        region for the figure
    fig_width : float or int
        width of figure in cm
    inset_pos : str, optional
        GMT location string for inset map, by default 'TL' (top left)
    """
    inset_reg = [-2800e3, 2800e3, -2800e3, 2800e3]
    inset_map = f"X{fig_width*.25}c"

    with fig.inset(
        position=f"J{inset_pos}+j{inset_pos}+w{fig_width*.25}c",
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
        fig.plot(data=fetch.groundingline(), pen="0.2p,black")

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
    region: Union[str or np.ndarray],
    projection: str,
    **kwargs
):
    """
    add lat lon grid lines and annotations to a figure.

    Parameters
    ----------
    fig : PyGMT.figure instance
    region : np.ndarray
        region for the figure
    projection : str
        GMT projection string in lat lon

    """
    font_color = kwargs.get('font_color', 'black')
    scale_length = kwargs.get('scale_length')
    length_perc = kwargs.get('length_perc', .25)
    position = kwargs.get('position', "n.5/.05")

    def round_to_1(x):
        return round(x, -int(floor(log10(abs(x)))))

    if scale_length is None:
        scale_length = round_to_1((abs(region[1])-abs(region[0]))/1000*length_perc)

    with pygmt.config(
        FONT_ANNOT_PRIMARY = f'10p,{font_color}', 
        FONT_LABEL = f'10p,{font_color}', 
        MAP_SCALE_HEIGHT='6p', 
        MAP_TICK_PEN_PRIMARY = f'0.5p,{font_color}'
    ):
        fig.basemap(
            region=region,
            projection = projection, 
            map_scale=f'{position}+w{scale_length}k+f+l"km"+ar', 
            verbose='e'
            )