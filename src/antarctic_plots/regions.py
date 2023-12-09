# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#
"""
Bounding regions for commonly plotted Antarctic regions. In Polar Stereographic
Projection (EPSG:3031). The format is (East, West, North, South), in meters.
"""
from __future__ import annotations

import typing

import pandas as pd
import verde as vd

from antarctic_plots import (  # pylint: disable=import-self
    maps,
    regions,  # noqa: PLW0406
    utils,
)

# import antarctic_plots.maps as maps
# import antarctic_plots.regions as regions
# import antarctic_plots.utils as utils

try:
    import ipyleaflet
except ImportError:
    ipyleaflet = None


try:
    from IPython.display import display
except ImportError:
    display = None


# regions
antarctica = (-2800e3, 2800e3, -2800e3, 2800e3)
west_antarctica = (-2740e3, 570e3, -2150e3, 1670e3)
east_antarctica = (-840e3, 2880e3, -2400e3, 2600e3)
antarctic_peninsula = (-2600e3, -1200e3, 170e3, 1800e3)
marie_byrd_land = (-1500e3, -500e3, -1350e3, -800e3)
victoria_land = (100e3, 1000e3, -2200e3, -1000e3)
# wilkes_land
# queen_maud_land

# study_sites
roosevelt_island = (-480e3, -240e3, -1220e3, -980e3)
ross_island = (210e3, 360e3, -1400e3, -1250e3)
minna_bluff = (210e3, 390e3, -1310e3, -1120e3)
# discovery_deep
mcmurdo_dry_valleys = (320e3, 480e3, -1400e3, -1220e3)
siple_coast = (-700e3, 30e3, -1110e3, -450e3)
crary_ice_rise = (-330e3, -40e3, -830e3, -480e3)
siple_dome = (-630e3, -270e3, -970e3, -630e3)

# ice_shelves
ross_ice_shelf = (-680e3, 470e3, -1420e3, -310e3)
# getz_ice_shelf = ()
# abbott_ice_shelf = ()
# george_vi_ice_shelf = ()
# wilkins_ice_shelf = ()
larsen_ice_shelf = (-2430e3, -1920e3, 900e3, 1400e3)
ronne_filchner_ice_shelf = (-1550e3, -500e3, 80e3, 1200e3)
# riiser_larsen_ice_shelf = ()
# fimbul_ice_shelf = ()
amery_ice_shelf = (1530e3, 2460e3, 430e3, 1000e3)
# west_ice_shelf = ()
# shackleton_ice_shelf = ()
# brunt_ice_shelf = ()

# glaciers
# byrd_glacier
# nimrod_glacier
pine_island_glacier = (-1720e3, -1480e3, -380e3, -70e3)
thwaites_glacier = (-1650e3, -1200e3, -600e3, -300e3)
kamb_ice_stream = (-620e3, -220e3, -800e3, -400e3)
# whillans_ice_stream = ()

# seas
ross_sea = (-500e3, 450e3, -2100e3, -1300e3)
# amundsen_sea
# bellinghausen_sea
# weddell_sea

# subglacial lakes
lake_vostok = (1100e3, 1535e3, -470e3, -230e3)
# ice catchements


def get_regions() -> dict[str, tuple[float, float, float, float]]:
    """
    get all the regions defined in this module.

    Returns
    -------
    dict[str, tuple[float, float, float, float] ]
        dictionary of each defined region's name and values
    """
    exclude_list = [
        "__",
        "pd",
        "vd",
        "utils",
        "regions",
        "TYPE_CHECKING",
        "Union",
        "maps",
        "ipyleaflet",
        "ipywidgets",
        "combine_regions",
        "draw_region",
        "get_regions",
        "annotations",
        "typing",
        "display",
    ]

    return {
        k: v
        for k, v in vars(regions).items()
        if (k not in exclude_list) & (not k.startswith("_"))
    }


def combine_regions(
    region1: tuple[float, float, float, float],
    region2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Get the bounding region of 2 regions.

    Parameters
    ----------
    region1 : tuple[float, float, float, float]
        first region
    region2 : tuple[float, float, float, float]
        second region

    Returns
    -------
    tuple[float, float, float, float]
        Bounding region of the 2 supplied regions.
    """
    coords1 = utils.region_to_df(region1)
    coords2 = utils.region_to_df(region2)
    coords_combined = pd.concat((coords1, coords2))
    reg: tuple[float, float, float, float] = vd.get_region(
        (coords_combined.x, coords_combined.y)
    )
    return reg


def draw_region(**kwargs: typing.Any) -> typing.Any:
    """
    Plot an interactive map, and use the "Draw a Rectangle" button to draw a rectangle
    and get the bounding region. Verticles will be returned as the output of the
    function.

    Returns
    -------
    typing.Any
        Returns a list of list of vertices for each polyline.
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
        global poly  # noqa: PLW0603 # pylint: disable=global-variable-undefined
        poly = []  # type: ignore[name-defined]

    clear_m()

    mydrawcontrol = ipyleaflet.DrawControl(
        polygon={
            "shapeOptions": {
                "fillColor": "#fca45d",
                "color": "#fca45d",
                "fillOpacity": 0.5,
            }
        },
        polyline={},
        circlemarker={},
        rectangle={},
    )

    def handle_rect_draw(self: typing.Any, action: str, geo_json: typing.Any) -> None:  # noqa: ARG001 # pylint: disable=unused-argument
        global poly  # noqa: PLW0602 # pylint: disable=global-variable-not-assigned
        shapes = []
        for coords in geo_json["geometry"]["coordinates"][0][:-1][:]:
            shapes.append(list(coords))
        shapes = list(shapes)
        if action == "created":
            poly.append(shapes)  # type: ignore[name-defined]

    mydrawcontrol.on_draw(handle_rect_draw)
    m.add_control(mydrawcontrol)

    clear_m()
    display(m)

    return poly  # type: ignore[name-defined]
