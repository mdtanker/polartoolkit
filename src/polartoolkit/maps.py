# pylint: disable=too-many-lines
import copy
import pathlib
import string
import typing
import warnings
from math import floor, log10

import deprecation
import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
from numpy.typing import NDArray

import polartoolkit
from polartoolkit import fetch, logger, regions, utils

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


class Figure(pygmt.Figure):  # type: ignore[misc]
    """
    A simple class that inherits from a pygmt Figure instance and stores additional
    figure parameters for easy access and provides addition methods.
    """

    def __init__(
        self,
        fig: pygmt.Figure | None = None,
        reg: tuple[float, float, float, float] | None = None,
        hemisphere: str | None = None,
        height: float | None = None,
        width: float | None = None,
    ) -> None:
        if fig is None and reg is None:
            msg = "Either a figure instance or a region (`reg`) must be provided."
            raise ValueError(msg)

        # if figure provided but not region, extract region from it
        if fig is not None and reg is None:
            with pygmt.clib.Session() as lib:
                reg = tuple(lib.extract_region().data)
                assert len(reg) == 4

        # if figure passed, use it, if not, initialize a new one
        if fig is None:
            super().__init__()
        else:
            # super().__init__()
            # copy attributes from the provided figure
            self.__dict__.update(fig.__dict__)
            # self._preview(fmt='png', dpi=1)
            # self.show(method="none")
            # reactivate the figure
            # self._activate_figure()

            # extract width and height from the figure
            width = utils.get_fig_width()
            height = utils.get_fig_height()

        try:
            hemisphere = utils.default_hemisphere(hemisphere)
        except KeyError:
            hemisphere = None
        if hemisphere is None:
            msg = (
                "you must provide the argument hemisphere as either 'north' or 'south'"
            )
            raise ValueError(msg)

        reg = typing.cast(tuple[float, float, float, float], reg)
        self.reg = reg
        self.hemisphere: str = hemisphere

        if self.hemisphere not in ["north", "south"]:
            msg = "hemisphere must be either 'north' or 'south'"
            raise ValueError(msg)

        # use default height if not set
        if width is None and height is None:
            height = 15

        self.proj, self.proj_latlon, self.width, self.height = utils.set_proj(
            self.reg,
            fig_height=height,
            fig_width=width,
            hemisphere=self.hemisphere,
        )

        self.reg_latlon = "/".join(map(str, self.reg)) + "/+ue"  # codespell:ignore ue
        self.origin_shift: str | None = None

    def shift_figure(
        self,
        origin_shift: str | None = None,
        yshift_amount: float = -1,
        xshift_amount: float = 1,
        xshift_extra: float = 0.4,
        yshift_extra: float = 0.4,
    ) -> None:
        """Determine if and how much to shift origin of figure"""

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
        if origin_shift == "initialize":
            origin_shift = None
            msg = "origin_shift 'initialize' is deprecated, use None instead."
            warnings.warn(
                msg,
                DeprecationWarning,
                stacklevel=2,
            )

        self.origin_shift = origin_shift

        if xshift_amount is None:
            xshift_amount = 0  # type: ignore[unreachable]
        if yshift_amount is None:
            yshift_amount = 0  # type: ignore[unreachable]
        # determine default values for x and y shift
        # add .4 to account for the space between figures
        xshift = xshift_amount * (self.width + xshift_extra)
        yshift = yshift_amount * (self.height + yshift_extra)

        if self.origin_shift is None:
            xshift = 0.0
            yshift = 0.0
        elif self.origin_shift == "x":
            yshift = 0.0
        elif self.origin_shift == "y":
            xshift = 0.0

        # add 3 to account for colorbar and titles
        # colorbar widths are automatically 80% figure width
        # colorbar heights are 4% of colorbar width
        # colorbar histograms are automatically 4*colorbar height
        # yshift = yshift_amount * (fig.height + 0.4)

        # shift origin of figure depending on origin_shift
        if self.origin_shift == "x":
            self.shift_origin(xshift=xshift)
        elif origin_shift == "y":
            self.shift_origin(yshift=yshift)
        elif self.origin_shift == "both":
            self.shift_origin(xshift=xshift, yshift=yshift)
        elif self.origin_shift is None:
            pass
        else:
            msg = "invalid string for `origin_shift`"
            raise ValueError(msg)

    def add_imagery(
        self,
        transparency: int = 0,
    ) -> None:
        """
        Add satellite imagery to a figure. For southern hemisphere uses LIMA imagery,
        but for northern hemisphere uses MODIS imagery.

        Parameters
        ----------
        transparency : int, optional
            transparency of the imagery, by default 0
        """

        if self.hemisphere == "north":
            image = fetch.modis(version="500m", hemisphere="north")
            cmap, _, _ = set_cmap(
                True,
                modis=True,
            )
        elif self.hemisphere == "south":
            image = fetch.imagery()
            cmap = None

        self.grdimage(
            grid=image,  # pylint: disable=possibly-used-before-assignment
            cmap=cmap,  # pylint: disable=possibly-used-before-assignment
            transparency=transparency,
            projection=self.proj,
            region=self.reg,
            verbose="e",
        )

    def add_coast(
        self,
        no_coast: bool = False,
        pen: str | None = None,
        version: str | None = None,
        label: str | None = None,
    ) -> None:
        """
        add coastline and or groundingline to figure.

        Parameters
        ----------
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

        if pen is None:
            pen = "0.6p,black"

        if version is None:
            if self.hemisphere == "north":
                version = "BAS"
            elif self.hemisphere == "south":
                version = "measures-v2"

        if version == "depoorter-2013":
            if no_coast is False:
                data = fetch.groundingline(version=version)
            elif no_coast is True:
                gdf = gpd.read_file(
                    fetch.groundingline(version=version), engine="pyogrio"
                )
                data = gdf[gdf.Id_text == "Grounded ice or land"]
        elif version == "measures-v2":
            if no_coast is False:
                gl = gpd.read_file(
                    fetch.groundingline(version=version), engine="pyogrio"
                )
                coast = gpd.read_file(
                    fetch.antarctic_boundaries(version="Coastline"), engine="pyogrio"
                )
                data = pd.concat([gl, coast])
            elif no_coast is True:
                data = fetch.groundingline(version=version)
        elif version in ("BAS", "measures-greenland"):
            data = fetch.groundingline(version=version)
        else:
            msg = "invalid version string"
            raise ValueError(msg)

        self.plot(
            data,  # pylint: disable=used-before-assignment
            projection=self.proj,
            region=self.reg,
            pen=pen,
            label=label,
        )

    def add_gridlines(
        self,
        x_spacing: float | None = None,
        y_spacing: float | None = None,
        annotation_offset: str = "20p",
    ) -> None:
        """
        add lat lon grid lines and annotations to a figure. Use kwargs x_spacing and
        y_spacing to customize the interval of gridlines and annotations.

        Parameters
        ----------
        x_spacing : float, optional
            spacing for x gridlines in degrees, by default is None
        y_spacing : float, optional
            spacing for y gridlines in degrees, by default is None
        annotation_offset : str, optional
            offset for gridline annotations, by default "20p"
        """

        if x_spacing is None:
            x_frames = ["xag", "xa"]
        else:
            x_frames = [
                f"xa{x_spacing * 2}g{x_spacing}",
                f"xa{x_spacing * 2}",
            ]

        if y_spacing is None:
            y_frames = ["yag", "ya"]
        else:
            y_frames = [
                f"ya{y_spacing * 2}g{y_spacing}",
                f"ya{y_spacing * 2}",
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
            self.basemap(
                projection=self.proj_latlon,
                region=self.reg_latlon,
                frame=[
                    "NSWE",
                    x_frames[0],
                    y_frames[0],
                ],
                transparency=50,
            )
            # re-plot annotations with no transparency
            with pygmt.config(FONT_ANNOT_PRIMARY="8p,black"):
                self.basemap(
                    projection=self.proj_latlon,
                    region=self.reg_latlon,
                    frame=[
                        "NSWE",
                        x_frames[0],
                        y_frames[0],
                    ],
                )

    def add_faults(
        self,
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
        if self.hemisphere == "north":
            msg = "Faults are not available for the northern hemisphere."
            raise NotImplementedError(msg)

        faults = fetch.geomap(version="faults", region=self.reg)

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
            faults = faults[
                faults.TYPENAME.isin(["thrust fault", "high angle reverse"])
            ]
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

        self.plot(
            faults,
            projection=self.proj,
            region=self.reg,
            pen=pen,
            label=label,
            style=style,
        )

    def add_modis(
        self,
        version: str | None = None,
        transparency: int = 0,
        cmap: str = "grayC",
    ) -> None:
        """
        Add MODIS imagery to a figure.

        Parameters
        ----------
        version : str | None, optional
            which version (resolution) of MODIS imagery to use, by default "750m" for
            southern hemisphere and "500m" for northern hemisphere.
        transparency : int, optional
            transparency of the MODIS imagery, by default 0
        cmap : str, optional
            colormap to use for MODIS imagery, by default "grayC"
        """

        if self.hemisphere == "north" and version is None:
            version = "500m"
        elif self.hemisphere == "south" and version is None:
            version = "750m"

        image = fetch.modis(version=version, hemisphere=self.hemisphere)

        imagery_cmap, _, _ = set_cmap(
            True,
            modis=True,
            modis_cmap=cmap,
        )
        self.grdimage(
            grid=image,
            cmap=imagery_cmap,
            transparency=transparency,
            projection=self.proj,
            region=self.reg,
            verbose="e",
        )

    def add_simple_basemap(
        self,
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
        version : str | None, optional
            which version of shapefiles to use for grounding line / coastline, by default
            "measures-v2" for southern hemisphere and "BAS" for northern hemisphere
        transparency : int, optional
            transparency of all the plotted elements, by default 0
        pen : str, optional
            GMT pen string for the coastline, by default "0.2p,black"
        grounded_color : str, optional
            color for the grounded ice, by default "grey"
        floating_color : str, optional
            color for the floating ice, by default "skyblue"
        """

        if self.hemisphere == "north":
            if version is None:
                version = "BAS"

            if version == "BAS":
                gdf = gpd.read_file(fetch.groundingline("BAS"), engine="pyogrio")
                self.plot(
                    data=gdf,
                    fill=grounded_color,
                    transparency=transparency,
                    projection=self.proj,
                    region=self.reg,
                )
                self.plot(
                    data=gdf,
                    pen=pen,
                    transparency=transparency,
                    projection=self.proj,
                    region=self.reg,
                )
            else:
                msg = "version must be BAS for northern hemisphere"
                raise ValueError(msg)

        elif self.hemisphere == "south":
            if version is None:
                version = "measures-v2"

            if version == "depoorter-2013":
                gdf = gpd.read_file(
                    fetch.groundingline("depoorter-2013"), engine="pyogrio"
                )
                # plot floating ice as blue
                self.plot(
                    data=gdf[gdf.Id_text == "Ice shelf"],
                    fill=floating_color,
                    transparency=transparency,
                    projection=self.proj,
                    region=self.reg,
                )
                # plot grounded ice as gray
                self.plot(
                    data=gdf[gdf.Id_text == "Grounded ice or land"],
                    fill=grounded_color,
                    transparency=transparency,
                    projection=self.proj,
                    region=self.reg,
                )
                # plot coastline on top
                self.plot(
                    data=gdf,
                    pen=pen,
                    transparency=transparency,
                )
            elif version == "measures-v2":
                self.plot(
                    data=fetch.antarctic_boundaries(version="Coastline"),
                    fill=floating_color,
                    transparency=transparency,
                    projection=self.proj,
                    region=self.reg,
                )
                self.plot(
                    data=fetch.groundingline(version="measures-v2"),
                    fill=grounded_color,
                    transparency=transparency,
                )
                self.plot(
                    fetch.groundingline(version="measures-v2"),
                    pen=pen,
                    transparency=transparency,
                    projection=self.proj,
                    region=self.reg,
                )

    def add_box(
        self,
        box: tuple[float, float, float, float],
        pen: str = "2p,black",
        verbose: str = "w",
    ) -> None:
        """
        Plot a GMT region as a box.

        Parameters
        ----------
        box : tuple[float, float, float, float]
            region in EPSG3031 in format [xmin, xmax, ymin, ymax] in meters
        pen : str, optional
            GMT pen string used for the box, by default "2p,black"
        verbose : str, optional
            verbosity level for pygmt, by default "w" for warnings
        """
        logger.debug("adding box to figure; %s", box)
        self.plot(
            x=[box[0], box[0], box[1], box[1], box[0]],
            y=[box[2], box[3], box[3], box[2], box[2]],
            pen=pen,
            verbose=verbose,
        )

    def add_inset(
        self,
        inset_position: str = "jTL+jTL+o0/0",
        inset_width: float = 0.25,
        inset_reg: tuple[float, float, float, float] | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """
        add an inset map showing the figure region relative to the Antarctic continent.

        Parameters
        ----------
        inset_position : str, optional
            GMT location string for inset map, by default 'jTL+jTL+o0/0' (top left)
        inset_width : float, optional
            Inset width as percentage of the smallest figure dimension, by default is 25%
            (0.25)
        inset_reg : tuple[float, float, float, float], optional
            Region of Antarctica/Greenland to plot for the inset map, by default is whole
            area
        """

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

        inset_width = inset_width * (min(self.width, self.height))
        inset_map = f"X{inset_width}c"

        position = f"{inset_position}+w{inset_width}c"
        logger.debug("using position; %s", position)

        with self.inset(
            position=position,
            box=kwargs.get("inset_box", False),
        ):
            if self.hemisphere == "north":
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
                gdf = gpd.read_file(fetch.groundingline("BAS"), engine="pyogrio")
                self.plot(
                    projection=inset_map,
                    region=inset_reg,
                    data=gdf,
                    fill="grey",
                )
                self.plot(
                    data=gdf,
                    pen=kwargs.get("inset_coast_pen", "0.2p,black"),
                )
            elif self.hemisphere == "south":
                if inset_reg is None:
                    inset_reg = regions.antarctica
                if inset_reg[1] - inset_reg[0] != inset_reg[3] - inset_reg[2]:
                    logger.warning(
                        "Inset region should be square or else projection will be off."
                    )
                logger.debug("plotting floating ice")
                self.plot(
                    projection=inset_map,
                    region=inset_reg,
                    data=fetch.antarctic_boundaries(version="Coastline"),
                    fill="skyblue",
                )
                logger.debug("plotting grounded ice")
                self.plot(
                    data=fetch.groundingline(version="measures-v2"),
                    fill="grey",
                )
                logger.debug("plotting coastline")
                gl = gpd.read_file(
                    fetch.groundingline(version="measures-v2"),
                    engine="pyogrio",
                )
                coast = gpd.read_file(
                    fetch.antarctic_boundaries(version="Coastline"), engine="pyogrio"
                )
                data = pd.concat([gl, coast])
                self.plot(
                    data,
                    pen=kwargs.get("inset_coast_pen", "0.2p,black"),
                )
            else:
                msg = "hemisphere must be north or south"
                raise ValueError(msg)
            logger.debug("add inset box")
            self.add_box(
                box=self.reg,
                pen=kwargs.get("inset_box_pen", "1p,red"),
            )
            logger.debug("inset complete")

    def add_scalebar(
        self,
        **kwargs: typing.Any,
    ) -> None:
        """
        add a scalebar to a figure.

        Parameters
        ----------
        kwargs : typing.Any

        """
        font_color = kwargs.get("font_color", "black")
        length = kwargs.get("length")
        length_perc = kwargs.get("length_perc", 0.25)
        position = kwargs.get("position", "n.5/.05")

        def round_to_1(x: float) -> float:
            return round(x, -floor(log10(abs(x))))

        if length is None:
            length = typing.cast(float, length)
            # get shorter of east-west vs north-sides
            width = abs(self.reg[1] - self.reg[0])
            height = abs(self.reg[3] - self.reg[2])
            length = round_to_1((min(width, height)) / 1000 * length_perc)

        with pygmt.config(
            FONT_ANNOT_PRIMARY=f"10p,{font_color}",
            FONT_LABEL=f"10p,{font_color}",
            MAP_SCALE_HEIGHT="6p",
            MAP_TICK_PEN_PRIMARY=f"0.5p,{font_color}",
        ):
            self.basemap(
                region=self.reg_latlon,
                projection=self.proj_latlon,
                map_scale=f"{position}+w{length}k+f+lkm+ar",
                box=kwargs.get("scalebar_box", "+gwhite"),
            )

    def add_north_arrow(
        self,
        **kwargs: typing.Any,
    ) -> None:
        """
        add a north arrow to a figure

        Parameters
        ----------
        kwargs : typing.Any
        """
        rose_size = kwargs.get("rose_size", "1c")

        position = kwargs.get("position", "n.1/.05")

        rose_str = kwargs.get("rose_str", f"{position}+w{rose_size}")

        self.basemap(
            region=self.reg_latlon,
            projection=self.proj_latlon,
            rose=rose_str,
            box=kwargs.get("rose_box", False),
            perspective=kwargs.get("perspective", False),
        )

    def add_grid(
        self,
        grid: str | xr.DataArray,
        cmap: str | bool = "viridis",
        shading: bool | str = False,
        nan_transparent: bool = True,
        colorbar: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        """
        Add a grid to the figure.

        Parameters
        ----------
        grid : str or xarray.DataArray
            Path to the grid file or an xarray DataArray containing the grid data.
        cmap : str or bool, optional
            Colormap to use for the grid, by default "viridis". If True, last used
            colormap will be used.
        shading : bool or str, optional
            If True, apply shading to the grid. If a string, it will be passed to
            pygmt.grdshade as the `shading` argument. By default, False (no shading).
        nan_transparent : bool, optional
            If True, set NaN values to be transparent in the grid image. Default is True
            unless shading is False, in which case it is set to False.
        colorbar : bool, optional
            If True, add a colorbar to the figure. Default is True.
        **kwargs : typing.Any
            Additional keyword arguments to pass
        """

        kwargs = copy.deepcopy(kwargs)

        # clip grid to region
        try:
            grid = pygmt.grdcut(
                grid,
                region=self.reg,
                verbose="q",
            )
        except ValueError as e:
            logger.error(e)
            msg = "clipping grid to plot region failed!"
            logger.error(msg)

        # if using shading, nan_transparent needs to be False
        if shading is False or shading is None:
            pass
        else:
            nan_transparent = False

        cmap, colorbar, cpt_lims = set_cmap(
            cmap,
            grid=grid,
            colorbar=colorbar,
            **kwargs,
        )

        # display grid
        logger.debug("plotting grid")
        self.grdimage(
            grid=grid,
            cmap=cmap,
            nan_transparent=nan_transparent,
            shading=shading,
            frame=kwargs.get("frame"),
            transparency=kwargs.get("grid_transparency", 0),
            projection=self.proj,
            region=self.reg,
            verbose="e",
        )

        if colorbar is True:
            kwargs["cpt_lims"] = cpt_lims
            logger.debug("adding colorbar to figure")
            self.add_colorbar(
                grid=grid,
                cmap=cmap,
                **kwargs,
            )

    def add_points(
        self,
        points: pd.DataFrame | gpd.GeoDataFrame,
        cmap: str | bool = "viridis",
        fill: str = "black",
        style: str = "c.2c",
        pen: str | None = None,
        label: str | None = None,
        colorbar: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        """
        Add points to the figure.

        Parameters
        ----------
        points : pandas.DataFrame or geopandas.GeoDataFrame
            DataFrame containing point data with columns 'x' and 'y' or 'easting' and
            'northing'.
        cmap : str or bool, optional
            Colormap to use for the points, by default "viridis". If True, last used
            colormap will be used.
        **kwargs : typing.Any
            Additional keyword arguments.
        """
        logger.debug("adding points")

        # subset points to plot region
        points = points.copy()
        points = utils.points_inside_region(
            points,
            region=self.reg,
        )
        if ("x" in points.columns) and ("y" in points.columns):
            x_col, y_col = "x", "y"
        elif ("easting" in points.columns) and ("northing" in points.columns):
            x_col, y_col = "easting", "northing"
        else:
            msg = "points must contain columns 'x' and 'y' or 'easting' and 'northing'."
            raise ValueError(msg)

        # plot points
        if fill in points.columns:
            cmap, colorbar, cpt_lims = set_cmap(
                cmap,
                points=points[fill],
                hemisphere=self.hemisphere,
                colorbar=colorbar,
                **kwargs,
            )
            self.plot(
                x=points[x_col],
                y=points[y_col],
                style=style,
                fill=points[fill],
                pen=pen,
                label=label,
                cmap=cmap,
                region=self.reg,
                projection=self.proj,
            )
            if colorbar is True:
                kwargs["cpt_lims"] = cpt_lims
                logger.debug("adding colorbar to figure")
                self.add_colorbar(
                    grid=points[[x_col, y_col, fill]],
                    cmap=cmap,
                    **kwargs,
                )
        else:
            cpt_lims = None
            if pen is None:
                pen = "1p,black"
            self.plot(
                x=points[x_col],
                y=points[y_col],
                style=style,
                fill=fill,
                pen=pen,
                label=label,
                region=self.reg,
                projection=self.proj,
            )

    def add_colorbar(
        self,
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

        # get kwargs
        values = kwargs.get("grid")
        cbar_end_triangles = kwargs.get("cbar_end_triangles")
        cmap = kwargs.get("cmap", True)

        if hist is True and values is None:
            msg = "if hist is True, grid or point values must be provided to 'grid'."
            raise ValueError(msg)

        # clip provided data to plot region if plotting a histogram or cbar end
        # triangles are not set.
        if hist is True or cbar_end_triangles is None or values is not None:
            # clip values to plot region
            if isinstance(values, (xr.DataArray | str)):
                if self.reg != utils.get_grid_info(values)[1]:
                    try:
                        values = utils.subset_grid(values, self.reg)
                        logger.debug("clipped grid to region")
                    except ValueError as e:
                        logger.error(e)
                        msg = "clipping grid to plot region failed! "
                        logger.error(msg)
                        return
                vals = vd.grid_to_table(values).iloc[:, -1].dropna().to_numpy()
            elif isinstance(values, pd.DataFrame):
                values = utils.points_inside_region(values, self.reg)
                vals = values.iloc[:, 2].to_numpy()
                logger.debug("clipped points to region")

        # set colorbar width as percentage of total figure width
        cbar_width_perc = kwargs.get("cbar_width_perc", 0.8)

        # set colorbar height as percentage of cbar width
        cbar_height_perc = kwargs.get("cbar_height_perc", 0.04)

        if hist is True:  # noqa: SIM102
            if kwargs.get("cbar_log", False) or kwargs.get("cpt_log", False):
                msg = (
                    "logarithmic colorbar is not supported for histogram, please set "
                    "`cbar_log` and `cpt_log` to False."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
                hist = False

        # offset colorbar vertically from plot by 0.4cm, or 0.2 + histogram height
        if hist is True:
            cbar_hist_height = kwargs.get("cbar_hist_height", 1.5)
            cbar_yoffset = kwargs.get("cbar_yoffset", 0.2 + cbar_hist_height)
        else:
            cbar_yoffset = kwargs.get("cbar_yoffset", 0.4)
        logger.debug("offset cbar vertically by %s", cbar_yoffset)

        if cbar_frame is None:
            cbar_label = kwargs.get("cbar_label", " ")
            cbar_frame = [
                f"pxaf+l{cbar_label}",
                f"+u{kwargs.get('cbar_unit_annot', ' ')}",
                f"py+l{kwargs.get('cbar_unit', ' ')}",
            ]

        # vertical or horizontal colorbar
        orientation = kwargs.get("cbar_orientation", "h")

        # text location
        text_location = kwargs.get("cbar_text_location")

        # add triangles to ends of colorbar to indicate if colorbar extends beyond
        # the data limits
        if cbar_end_triangles is None:
            if values is None:
                logger.warning(
                    "plotted values not provided via 'grid', so cannot determine if to "
                    "add colorbar end triangles or not."
                )
                cbar_end_triangles = ""
            elif cpt_lims is None:
                cbar_end_triangles = ""
            elif (cpt_lims[0] > vals.min()) & (cpt_lims[1] < vals.max()):  # pylint: disable=possibly-used-before-assignment
                cbar_end_triangles = "+e"
            elif cpt_lims[0] > vals.min():
                cbar_end_triangles = "+eb"
            elif cpt_lims[1] < vals.max():
                cbar_end_triangles = "+ef"
            else:
                cbar_end_triangles = ""

        # add colorbar
        logger.debug("adding colorbar")
        with pygmt.config(
            FONT=kwargs.get("cbar_font", "12p,Helvetica,black"),
        ):
            cbar_width = self.width * cbar_width_perc
            cbar_height = cbar_width * cbar_height_perc
            position = (
                f"jBC+jTC+w{cbar_width}/{cbar_height}c+{orientation}{text_location}"
                f"+o{kwargs.get('cbar_xoffset', 0)}c/{cbar_yoffset}c{cbar_end_triangles}"
            )
            logger.debug("cbar frame; %s", cbar_frame)
            logger.debug("cbar position: %s", position)

            self.colorbar(
                cmap=cmap,
                position=position,
                frame=cbar_frame,
                scale=kwargs.get("cbar_scale", 1),
                log=kwargs.get("cbar_log"),
                # verbose=verbose, # this is causing issues
            )
            logger.debug("finished standard colorbar plotting")

            # # update figure height to account for colorbar
            # if cbar_label == ' ':
            #     label_height = 0
            # else:
            #     label_height = 1
            # self.height = self.height + cbar_yoffset + label_height

        # add histogram to colorbar
        # Note, depending on data and hist_type, you may need to manually set kwarg
        # `hist_ymax` to an appropriate value
        if hist is True:
            logger.debug("adding histogram to colorbar")

            if isinstance(cmap, str) and cmap.endswith(".cpt"):
                # extract cpt_lims from cmap
                p = pathlib.Path(cmap)
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
                    "getting max/min values from grid/points since cpt_lims were not "
                    "supplied, if cpt_lims were used to create the colorscale, pass "
                    "them there or else histogram will not properly align with "
                    "colorbar!",
                    stacklevel=2,
                )
                zmin, zmax = utils.get_min_max(
                    vals,
                    shapefile=kwargs.get("shp_mask"),
                    region=kwargs.get("cmap_region"),
                    robust=kwargs.get("robust", False),
                    hemisphere=self.hemisphere,
                    robust_percentiles=kwargs.get("robust_percentiles", (0.02, 0.98)),
                    absolute=kwargs.get("absolute", False),
                )
            else:
                zmin, zmax = cpt_lims
            logger.debug("using %s, %s for histogram limits", zmin, zmax)

            vals = pd.Series(vals)
            # subset data between cbar min and max
            data = vals[vals.between(zmin, zmax)]

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
                msg = (
                    "Grid/points are a constant value, can't make a colorbar histogram!"
                )
                logger.warning(msg)
                return

            # define histogram region
            hist_reg = [
                zmin,
                zmax,
                kwargs.get("hist_ymin", 0),
                kwargs.get("hist_ymax", max_bin_height * 1.1),
            ]
            logger.debug("defined histogram region; %s", hist_reg)
            # shift figure to line up with top left of cbar
            xshift = (
                kwargs.get("cbar_xoffset", 0) + ((1 - cbar_width_perc) * self.width) / 2
            )
            try:
                self.shift_origin(xshift=f"{xshift}c", yshift=f"{-cbar_yoffset}c")
                logger.debug("shifting origin")
            except pygmt.exceptions.GMTCLibError as e:
                logger.warning(e)
                logger.warning("issue with plotting histogram, skipping...")

            # plot histograms above colorbar
            try:
                hist_proj = f"X{self.width * cbar_width_perc}c/{cbar_hist_height}c"
                logger.debug("histogram projection; %s", hist_proj)
                hist_series = f"{zmin}/{zmax}/{bin_width}"
                logger.debug("histogram series; %s", hist_series)
                logger.debug("plotting histogram")
                self.histogram(
                    data=data,
                    projection=hist_proj,
                    region=hist_reg,
                    frame=kwargs.get("hist_frame", False),
                    cmap=cmap,
                    fill=kwargs.get("hist_fill"),
                    pen=kwargs.get("hist_pen", "default"),
                    barwidth=kwargs.get("hist_barwidth"),
                    center=kwargs.get("hist_center", False),
                    distribution=kwargs.get("hist_distribution", False),
                    cumulative=kwargs.get("hist_cumulative", False),
                    extreme=kwargs.get("hist_extreme", "b"),
                    stairs=kwargs.get("hist_stairs", False),
                    series=hist_series,
                    histtype=hist_type,
                    verbose=verbose,
                )
                logger.debug(
                    "plotting histogram complete, resetting region and projection"
                )
                # reset region and projection
                self.basemap(
                    region=self.reg,
                    projection=self.proj,
                    frame="+t",
                )

                # # update figure height to account for colorbar
                # if cbar_label == ' ':
                #     label_height = 0
                # else:
                #     label_height = 1
                # self.height = self.height + cbar_yoffset + label_height

            except pygmt.exceptions.GMTCLibError as e:
                logger.warning(e)
                logger.warning("issue with plotting histogram, skipping...")
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.exception("An error occurred: %s", e)

            # shift figure back
            try:
                self.shift_origin(xshift=f"{-xshift}c", yshift=f"{cbar_yoffset}c")
            except pygmt.exceptions.GMTCLibError as e:
                logger.warning(e)
                logger.warning("issue with plotting histogram, skipping...")
            logger.debug("finished plotting histogram")


@deprecation.deprecated(
    deprecated_in="1.0.7",
    removed_in="2.0.0",
    current_version=polartoolkit.__version__,
    details="`add_coast` has been replaced with a class method.",
)
def add_coast(
    fig: pygmt.Figure,
    hemisphere: str | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    region: tuple[float, float, float, float] | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    projection: str | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    no_coast: bool = False,
    pen: str | None = None,
    version: str | None = None,
    label: str | None = None,
) -> None:
    """deprecated function, use class method `add_coast` instead"""
    fig.add_coast(
        no_coast=no_coast,
        pen=pen,
        version=version,
        label=label,
    )


@deprecation.deprecated(
    deprecated_in="1.0.7",
    removed_in="2.0.0",
    current_version=polartoolkit.__version__,
    details="`add_gridlines` has been replaced with a class method.",
)
def add_gridlines(
    fig: pygmt.Figure,
    region: tuple[float, float, float, float] | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    projection: str | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    x_spacing: float | None = None,
    y_spacing: float | None = None,
    annotation_offset: str = "20p",
) -> None:
    """deprecated function, use class method `add_gridlines` instead"""
    fig.add_gridlines(
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        annotation_offset=annotation_offset,
    )


@deprecation.deprecated(
    deprecated_in="1.0.7",
    removed_in="2.0.0",
    current_version=polartoolkit.__version__,
    details="`add_faults` has been replaced with a class method.",
)
def add_faults(
    fig: pygmt.Figure,
    region: tuple[float, float, float, float] | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    projection: str | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    fault_activity: str | None = None,
    fault_motion: str | None = None,
    fault_exposure: str | None = None,
    pen: str | None = None,
    style: str | None = None,
    label: str | None = None,
) -> None:
    """deprecated function, use class method `add_faults` instead"""
    fig.add_faults(
        fault_activity=fault_activity,
        fault_motion=fault_motion,
        fault_exposure=fault_exposure,
        pen=pen,
        style=style,
        label=label,
    )


@deprecation.deprecated(
    deprecated_in="1.0.7",
    removed_in="2.0.0",
    current_version=polartoolkit.__version__,
    details="`add_imagery` has been replaced with a class method.",
)
def add_imagery(
    fig: pygmt.Figure,
    hemisphere: str | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    transparency: int = 0,
) -> None:
    """deprecated function, use class method `add_imagery` instead"""
    fig.add_imagery(
        transparency=transparency,
    )


@deprecation.deprecated(
    deprecated_in="1.0.7",
    removed_in="2.0.0",
    current_version=polartoolkit.__version__,
    details="`add_modis` has been replaced with a class method.",
)
def add_modis(
    fig: pygmt.Figure,
    hemisphere: str | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    version: str | None = None,
    transparency: int = 0,
    cmap: str = "grayC",
) -> None:
    """deprecated function, use class method `add_modis` instead"""
    fig.add_modis(
        version=version,
        transparency=transparency,
        cmap=cmap,
    )


@deprecation.deprecated(
    deprecated_in="1.0.7",
    removed_in="2.0.0",
    current_version=polartoolkit.__version__,
    details="`add_simple_basemap` has been replaced with a class method.",
)
def add_simple_basemap(
    fig: pygmt.Figure,
    hemisphere: str | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    version: str | None = None,
    transparency: int = 0,
    pen: str = "0.2p,black",
    grounded_color: str = "grey",
    floating_color: str = "skyblue",
) -> None:
    """deprecated function, use class method `add_simple_basemap` instead"""
    fig.add_simple_basemap(
        version=version,
        transparency=transparency,
        pen=pen,
        grounded_color=grounded_color,
        floating_color=floating_color,
    )


@deprecation.deprecated(
    deprecated_in="1.0.7",
    removed_in="2.0.0",
    current_version=polartoolkit.__version__,
    details="`add_inset` has been replaced with a class method.",
)
def add_inset(
    fig: pygmt.Figure,
    hemisphere: str | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    region: tuple[float, float, float, float] | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    inset_position: str = "jTL+jTL+o0/0",
    inset_width: float = 0.25,
    inset_reg: tuple[float, float, float, float] | None = None,
    **kwargs: typing.Any,
) -> None:
    """deprecated function, use class method `add_inset` instead"""
    fig.add_inset(
        inset_position=inset_position,
        inset_width=inset_width,
        inset_reg=inset_reg,
        **kwargs,
    )


@deprecation.deprecated(
    deprecated_in="1.0.7",
    removed_in="2.0.0",
    current_version=polartoolkit.__version__,
    details="`add_scalebar` has been replaced with a class method.",
)
def add_scalebar(
    fig: pygmt.Figure,
    region: tuple[float, float, float, float] | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    projection: str | None = None,  # noqa: ARG001 # pylint: disable=unused-argument
    **kwargs: typing.Any,
) -> None:
    """deprecated function, use class method `add_scalebar` instead"""
    fig.add_scalebar(
        **kwargs,
    )


@deprecation.deprecated(
    deprecated_in="1.0.7",
    removed_in="2.0.0",
    current_version=polartoolkit.__version__,
    details="`add_north_arrow` has been replaced with a class method.",
)
def add_north_arrow(
    fig: pygmt.Figure,
    **kwargs: typing.Any,
) -> None:
    """deprecated function, use class method `add_north_arrow` instead"""
    fig.add_north_arrow(
        **kwargs,
    )


@deprecation.deprecated(
    deprecated_in="1.0.7",
    removed_in="2.0.0",
    current_version=polartoolkit.__version__,
    details="`add_box` has been replaced with a class method.",
)
def add_box(
    fig: pygmt.Figure,
    box: tuple[float, float, float, float],
    pen: str = "2p,black",
    verbose: str = "w",
) -> None:
    """deprecated function, use class method `add_box` instead"""
    fig.add_box(
        box=box,
        pen=pen,
        verbose=verbose,
    )


@deprecation.deprecated(
    deprecated_in="1.0.7",
    removed_in="2.0.0",
    current_version=polartoolkit.__version__,
    details="`add_colorbar` has been replaced with a class method.",
)
def add_colorbar(
    fig: pygmt.Figure,
    hist: bool = False,
    cpt_lims: tuple[float, float] | None = None,
    cbar_frame: list[str] | str | None = None,
    verbose: str = "w",
    **kwargs: typing.Any,
) -> None:
    """deprecated function, use class method `add_colorbar` instead"""
    fig.add_colorbar(
        hist=hist,
        cpt_lims=cpt_lims,
        cbar_frame=cbar_frame,
        verbose=verbose,
        **kwargs,
    )


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
    origin_shift: str | None = None,
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

    if fig is None:
        if region is None:
            if points is None:
                msg = "If no figure or points are provided, a region must be specified."
                raise ValueError(msg)
            # if no region is specified, use the points to determine the region
            if ("x" in points.columns) and ("y" in points.columns):
                x_col, y_col = "x", "y"
            elif ("easting" in points.columns) and ("northing" in points.columns):
                x_col, y_col = "easting", "northing"
            else:
                msg = "points must contain columns 'x' and 'y' or 'easting' and 'northing'."
                raise ValueError(msg)
            region = vd.get_region(points[[x_col, y_col]].values)
            logger.debug("using region %s from points", region)
    elif region is None:
        region = fig.reg
        logger.debug("using region %s from figure", region)
    else:
        logger.debug("using region %s from input", region)

    frame = kwargs.get("frame", "nesw+gwhite")

    if fig is not None and origin_shift is None and frame is not None:
        msg = (
            "Argument `frame` is ignored since you are plotting over an existing figure"
        )
        logger.warning(msg)
        frame = None
    # else:
    #     frame = kwargs.get("frame", "nesw+gwhite")

    # initialize figure
    fig = Figure(
        fig=fig,
        reg=region,
        hemisphere=hemisphere,
        height=kwargs.get("fig_height"),
        width=kwargs.get("fig_width"),
    )
    # need to mock show the figure for pygmt to set the temp file
    fig.show(method="none")

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
        yshift_extra += (kwargs.get("cbar_width_perc", 0.8) * fig.width) * kwargs.get(
            "cbar_height_perc", 0.04
        )
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

    # shift figure origin if needed
    fig.shift_figure(
        origin_shift=origin_shift,
        yshift_amount=kwargs.get("yshift_amount", -1),
        xshift_amount=kwargs.get("xshift_amount", 1),
        yshift_extra=yshift_extra,
        xshift_extra=kwargs.get("xshift_extra", 0.4),
    )

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
                region=fig.reg,
                projection=fig.proj,
                frame=frame,
                verbose="e",
                transparency=kwargs.get("transparency", 0),
            )
        elif frame is False:
            pass
        elif isinstance(frame, list):
            fig.basemap(
                region=fig.reg,
                projection=fig.proj,
                frame=frame,
                verbose="e",
                transparency=kwargs.get("transparency", 0),
            )
        else:
            fig.basemap(
                region=fig.reg,
                projection=fig.proj,
                frame=frame,
                verbose="e",
                transparency=kwargs.get("transparency", 0),
            )

    with pygmt.config(FONT_TITLE=kwargs.get("title_font", "auto")):
        fig.basemap(
            region=fig.reg,
            projection=fig.proj,
            frame=f"+t{title}",
            verbose="e",
        )

    # add satellite imagery (LIMA for Antarctica)
    if imagery_basemap is True:
        logger.debug("adding background imagery")
        fig.add_imagery(
            transparency=kwargs.get("imagery_transparency", 0),
        )

    # add MODIS imagery as basemap
    if modis_basemap is True:
        logger.debug("adding MODIS imagery")
        fig.add_modis(
            version=kwargs.get("modis_version"),
            transparency=kwargs.get("modis_transparency", 0),
            cmap=kwargs.get("modis_cmap", "grayC"),
        )

    # add simple basemap
    if simple_basemap is True:
        logger.debug("adding simple basemap")
        fig.add_simple_basemap(
            version=kwargs.get("simple_basemap_version"),
            transparency=kwargs.get("simple_basemap_transparency", 0),
            pen=kwargs.get("simple_basemap_pen", "0.2p,black"),
            grounded_color=kwargs.get("simple_basemap_grounded_color", "grey"),
            floating_color=kwargs.get("simple_basemap_floating_color", "skyblue"),
        )

    # add lat long grid lines
    if gridlines is True:
        logger.debug("adding gridlines")
        fig.add_gridlines(
            x_spacing=kwargs.get("x_spacing"),
            y_spacing=kwargs.get("y_spacing"),
        )

    # plot groundingline and coastlines
    if coast is True:
        logger.debug("adding coastlines")
        fig.add_coast(
            pen=kwargs.get("coast_pen"),
            no_coast=kwargs.get("no_coast", False),
            version=kwargs.get("coast_version"),
            label=kwargs.get("coast_label", None),
        )

    # plot faults
    if faults is True:
        logger.debug("adding faults")
        fig.add_faults(
            label=kwargs.get("fault_label"),
            pen=kwargs.get("fault_pen"),
            style=kwargs.get("fault_style"),
            fault_activity=kwargs.get("fault_activity"),
            fault_motion=kwargs.get("fault_motion"),
            fault_exposure=kwargs.get("fault_exposure"),
        )

    # add box showing region
    if kwargs.get("show_region") is not None:
        logger.debug("adding region box")
        fig.add_box(
            box=kwargs.get("show_region"),  # type: ignore[arg-type]
            pen=kwargs.get("region_pen", "2p,black"),
        )

    # add datapoints
    if points is not None:
        logger.debug("adding points")

        fig.add_points(
            points=points,
            cmap=kwargs.get("points_cmap", "viridis"),
            fill=points_fill,
            style=kwargs.get("points_style", "c.2c"),
            pen=kwargs.get("points_pen"),
            label=kwargs.get("points_label"),
            **kwargs,
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
        fig.add_inset(
            **new_kwargs,
        )

    # add scalebar
    if scalebar is True:
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

        fig.add_scalebar(
            font_color=scalebar_font_color,
            length_perc=scalebar_length_perc,
            position=scalebar_position,
            **kwargs,
        )

    # add north arrow
    if north_arrow is True:
        fig.add_north_arrow(
            **kwargs,
        )

    # reset region and projection
    fig.basemap(region=fig.reg, projection=fig.proj, frame="+t")

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
    absolute: bool = False,
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
    absolute : bool, optional
        use the absolute value of the data from the grid or points, by default False
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
    if cmap is True and modis is False:
        colorbar = True
    elif isinstance(cmap, str) and cmap.endswith(".cpt"):
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
            cmap=kwargs.get("modis_cmap", "grayC"),
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
                    absolute=absolute,
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
                        absolute=absolute,
                    )
            elif cpt_lims is None:
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
                log=kwargs.get("cpt_log", False),
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
                log=kwargs.get("cpt_log", False),
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
                log=kwargs.get("cpt_log", False),
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
                absolute=absolute,
            )
            pygmt.makecpt(
                cmap=cmap,
                background=True,
                continuous=kwargs.get("continuous", True),
                series=(zmin, zmax),
                reverse=reverse_cpt,
                verbose="e",
                log=kwargs.get("cpt_log", False),
            )
        except (pygmt.exceptions.GMTCLibError, Exception) as e:  # pylint: disable=broad-exception-caught
            if "Option T: min >= max" in str(e):
                logger.warning("supplied min value is greater or equal to max value")
                pygmt.makecpt(
                    cmap=cmap,
                    background=True,
                    reverse=reverse_cpt,
                    verbose="e",
                    log=kwargs.get("cpt_log", False),
                )
            else:
                logger.exception(e)
                pygmt.makecpt(
                    cmap=cmap,
                    background=True,
                    continuous=kwargs.get("continuous", True),
                    reverse=reverse_cpt,
                    verbose="e",
                    log=kwargs.get("cpt_log", False),
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
    kwargs = copy.deepcopy(kwargs)

    if isinstance(grid, str):
        pass
    else:
        grid = grid.copy()

    if isinstance(grid, xr.Dataset):
        msg = "grid must be a DataArray, not a Dataset."
        raise ValueError(msg)

    if fig is None:
        if region is None:
            # if no region is specified, use the grid to determine the region
            region = utils.get_grid_info(grid)[1]
            logger.debug("using region %s from grid", region)
    elif region is None:
        region = fig.reg
        logger.debug("using region %s from figure", region)
    else:
        logger.debug("using region %s from input", region)

    frame = kwargs.get("frame", "nesw+gwhite")

    if fig is not None and origin_shift is None and frame is not None:
        msg = (
            "Argument `frame` is ignored since you are plotting over an existing figure"
        )
        logger.warning(msg)
        frame = None
    # else:
    #     frame = kwargs.get("frame", "nesw+gwhite")

    # initialize figure
    fig = Figure(
        fig=fig,
        reg=region,
        hemisphere=hemisphere,
        height=kwargs.get("fig_height"),
        width=kwargs.get("fig_width"),
    )
    # need to mock show the figure for pygmt to set the temp file
    fig.show(method="none")

    # decide if colorbar should be plotted
    colorbar = kwargs.pop("colorbar", True)
    if colorbar is None:
        if kwargs.get("modis", False) is True:
            colorbar = False
        elif cmap is True:
            colorbar = True
        else:
            colorbar = False

    # if currently plotting colorbar, or histogram, assume the past plot did as well and
    # account for it in the y shift
    yshift_extra = kwargs.get("yshift_extra", 0.4)
    if colorbar is True:
        # for thickness of cbar
        yshift_extra += (kwargs.get("cbar_width_perc", 0.8) * fig.width) * kwargs.get(
            "cbar_height_perc", 0.04
        )
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

    # shift figure origin if needed
    fig.shift_figure(
        origin_shift=origin_shift,
        yshift_amount=kwargs.get("yshift_amount", -1),
        xshift_amount=kwargs.get("xshift_amount", 1),
        yshift_extra=yshift_extra,
        xshift_extra=kwargs.get("xshift_extra", 0.4),
    )

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
                region=fig.reg,
                projection=fig.proj,
                frame=frame,
                verbose="e",
                transparency=kwargs.get("transparency", 0),
            )
        elif frame is False:
            pass
        elif isinstance(frame, list):
            fig.basemap(
                region=fig.reg,
                projection=fig.proj,
                frame=frame,
                verbose="e",
                transparency=kwargs.get("transparency", 0),
            )
        else:
            fig.basemap(
                region=fig.reg,
                projection=fig.proj,
                frame=frame,
                verbose="e",
                transparency=kwargs.get("transparency", 0),
            )

    with pygmt.config(FONT_TITLE=kwargs.get("title_font", "auto")):
        fig.basemap(
            region=fig.reg,
            projection=fig.proj,
            frame=f"+t{title}",
            verbose="e",
        )

    # add satellite imagery (LIMA for Antarctica)
    if imagery_basemap is True:
        logger.debug("adding background imagery")
        fig.add_imagery(
            transparency=kwargs.get("imagery_transparency", 0),
        )

    # add MODIS imagery as basemap
    if modis_basemap is True:
        logger.debug("adding MODIS imagery")
        fig.add_modis(
            version=kwargs.get("modis_version"),
            transparency=kwargs.get("modis_transparency", 0),
            cmap=kwargs.get("modis_cmap", "grayC"),
        )

    # add simple basemap
    if simple_basemap is True:
        logger.debug("adding simple basemap")
        fig.add_simple_basemap(
            version=kwargs.get("simple_basemap_version"),
            transparency=kwargs.get("simple_basemap_transparency", 0),
            pen=kwargs.get("simple_basemap_pen", "0.2p,black"),
            grounded_color=kwargs.get("simple_basemap_grounded_color", "grey"),
            floating_color=kwargs.get("simple_basemap_floating_color", "skyblue"),
        )

    # add the grid
    fig.add_grid(
        grid=grid,
        cmap=cmap,
        colorbar=colorbar,
        **kwargs,
    )
    # add lat long grid lines
    if gridlines is True:
        logger.debug("adding gridlines")
        fig.add_gridlines(
            x_spacing=kwargs.get("x_spacing"),
            y_spacing=kwargs.get("y_spacing"),
        )

    # plot groundingline and coastlines
    if coast is True:
        logger.debug("adding coastlines")
        fig.add_coast(
            pen=kwargs.get("coast_pen"),
            no_coast=kwargs.get("no_coast", False),
            version=kwargs.get("coast_version"),
            label=kwargs.get("coast_label", None),
        )

    # plot faults
    if faults is True:
        logger.debug("adding faults")
        fig.add_faults(
            label=kwargs.get("fault_label"),
            pen=kwargs.get("fault_pen"),
            style=kwargs.get("fault_style"),
            fault_activity=kwargs.get("fault_activity"),
            fault_motion=kwargs.get("fault_motion"),
            fault_exposure=kwargs.get("fault_exposure"),
        )

    # add box showing region
    if kwargs.get("show_region") is not None:
        logger.debug("adding region box")
        fig.add_box(
            box=kwargs.get("show_region"),  # type: ignore[arg-type]
            pen=kwargs.get("region_pen", "2p,black"),
        )

    # add datapoints
    if points is not None:
        logger.debug("adding points")
        kwargs["hist"] = False

        if kwargs.get("points_cmap") is not None:
            msg = "`points_cmap` is ignored since grid's cmap is being used."
            logger.warning(msg)

        fig.add_points(
            points=points,
            colorbar=kwargs.get("points_colorbar", False),
            fill=kwargs.get("points_fill", "black"),
            style=kwargs.get("points_style", "c.2c"),
            pen=kwargs.get("points_pen"),
            label=kwargs.get("points_label"),
            cmap=cmap,
            **kwargs,
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
        fig.add_inset(
            **new_kwargs,
        )

    # add scalebar
    if scalebar is True:
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

        fig.add_scalebar(
            font_color=scalebar_font_color,
            length_perc=scalebar_length_perc,
            position=scalebar_position,
            **kwargs,
        )

    # add north arrow
    if north_arrow is True:
        fig.add_north_arrow(
            **kwargs,
        )

    # reset region and projection
    fig.basemap(region=fig.reg, projection=fig.proj, frame="+t")

    return fig


def interactive_map(
    hemisphere: str | None = None,
    center_yx: tuple[float] | None = None,
    zoom: float | None = None,
    display_xy: bool = True,
    points: pd.DataFrame | None = None,
    basemap_type: str | None = None,
    **kwargs: typing.Any,
) -> typing.Any:
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
        choose zoom level, by default None
    display_xy : bool, optional
        choose if you want clicks to show the xy location, by default True
    show : bool, optional
        choose whether to display the map, by default True
    points : pandas.DataFrame, optional
        choose to plot points supplied as columns 'x', 'y', or 'easting', 'northing', in
        EPSG:3031 in a dataframe
    basemap_type : str, optional
        choose what basemap to plot, options are 'BlueMarble', 'Imagery', 'Basemap', and
        "IceVelocity", by default 'BlueMarble' for northern hemisphere and 'Imagery' for
        southern hemisphere.

    Returns
    -------
    typing.Any
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
    # if no points, center map on 0, 0
    elif hemisphere == "south":
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

    if basemap_type is None:
        if hemisphere == "south":
            basemap_type = "Imagery"
        elif hemisphere == "north":
            basemap_type = "BlueMarble"

    if hemisphere == "south":
        if basemap_type == "BlueMarble":
            base = ipyleaflet.basemaps.NASAGIBS.BlueMarbleBathymetry3031  # pylint: disable=no-member
            proj = ipyleaflet.projections.EPSG3031.NASAGIBS
        elif basemap_type == "Imagery":
            base = ipyleaflet.basemaps.Esri.AntarcticImagery  # pylint: disable=no-member
            proj = ipyleaflet.projections.EPSG3031.ESRIImagery
            if zoom is None:
                zoom = 5
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

    if zoom is None:
        zoom = 0

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
                latlon = kwargs.get("coordinates")[::-1]  # type: ignore[index]
                if hemisphere == "south":
                    label_xy.value = str(utils.latlon_to_epsg3031(latlon))
                elif hemisphere == "north":
                    label_xy.value = str(utils.latlon_to_epsg3413(latlon))

        m.on_interaction(handle_click)

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

    if isinstance(grids, xr.DataArray):
        grids = [grids]

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
            fig = None
            origin_shift = None
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
            fig.text(
                text=fig_title,
                position="TC",
                font=fig_title_font,
                offset=f"{(((fig_width * xshift) / 2) * (ncols - 1))}c/{fig_title_y_offset}",
                no_clip=True,
            )
        if (fig_x_axis_title is not None) & (i == int(ncols / 2)):
            fig.text(
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
            fig.text(
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
                label = f"a{string.ascii_lowercase[i - 26]}"
            elif i < 26 * 3:
                label = f"b{string.ascii_lowercase[i - (26 * 2)]}"
            elif i < 26 * 4:
                label = f"b{string.ascii_lowercase[i - (26 * 3)]}"
            elif i < 26 * 5:
                label = f"b{string.ascii_lowercase[i - (26 * 4)]}"
            elif i < 26 * 6:
                label = f"b{string.ascii_lowercase[i - (26 * 5)]}"
            else:
                label = None

            fig.text(
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
            fig.text(
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
            fig.text(
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
    drapegrids: list[xr.DataArray] | None = None,
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

    if not isinstance(grids, (list, tuple)):
        grids = [grids]

    # number of grids to plot
    num_grids = len(grids)

    # if not provided as a list, make it a list the length of num_grids
    if not isinstance(cbar_labels, (list, tuple)):
        cbar_labels = [cbar_labels] * num_grids
    if not isinstance(modis, (list, tuple)):
        modis = [modis] * num_grids
    if not isinstance(grd2cpt, (list, tuple)):
        grd2cpt = [grd2cpt] * num_grids
    if not isinstance(cmap_region, (list, tuple)):
        cmap_region = [cmap_region] * num_grids
    if not isinstance(robust, (list, tuple)):
        robust = [robust] * num_grids
    if not isinstance(reverse_cpt, (list, tuple)):
        reverse_cpt = [reverse_cpt] * num_grids
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * num_grids
    if not isinstance(exaggeration, (list, tuple)):
        exaggeration = [exaggeration] * num_grids
    if not isinstance(drapegrids, (list, tuple)):
        if drapegrids is None:
            drapegrids = [None] * num_grids
        else:
            drapegrids = [drapegrids] * num_grids  # type: ignore[unreachable]
    if cpt_lims_list is None:
        cpt_lims_list = [None] * num_grids
    elif (
        (isinstance(cpt_lims_list, (list, tuple)))
        & (len(cpt_lims_list) == 2)
        & (all(isinstance(x, float) for x in cpt_lims_list))
    ):
        cpt_lims_list = [cpt_lims_list] * num_grids
    if (
        isinstance(cmap_region, (list, tuple))
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
        vlims = utils.get_combined_min_max(grids)  # type: ignore[arg-type]

    new_region = region + vlims

    # initialize the figure
    fig = pygmt.Figure()

    # iterate through grids and plot them
    for i, grid in enumerate(grids):
        # if provided, mask grid with shapefile
        if shp_mask is not None:
            grid = utils.mask_from_shp(  # noqa: PLW2901
                shp_mask,
                grid=grid,
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
            drapegrid=drapegrids[i],
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
                # position=f"g{np.max(region[0:2])}/{np.mean(region[2:4])}+w{fig_width*.4}c/.5c+v+e+m",
                # # vertical, with triangles, text opposite
                position=f"jMR+w{fig_width * 0.4}c/.5c+v+e+m",  # vertical, with triangles, text opposite
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
        coast_gdf = gpd.read_file(fetch.groundingline(version="BAS"), engine="pyogrio")
        crsys = crs.NorthPolarStereo()
    elif hemisphere == "south":
        coast_gdf = gpd.read_file(
            fetch.groundingline(version="measures-v2"), engine="pyogrio"
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
) -> typing.Any:
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
    elif points_z is None:
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
                absolute=kwargs.get("absolute", False),
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
