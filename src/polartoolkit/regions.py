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
"""
Bounding regions for commonly plotted polar regions. In stereographic projections. The
format is (xmin, xmax, ymin, ymax), in meters.
"""

from __future__ import annotations

import typing

import pandas as pd
import verde as vd

from polartoolkit import (  # pylint: disable=import-self
    maps,
    regions,  # noqa: PLW0406
    utils,
)

try:
    import ipyleaflet
except ImportError:
    ipyleaflet = None


try:
    from IPython.display import display
except ImportError:
    display = None

#####
#####
# Antarctica
#####
#####

# regions
antarctica = (-2800e3, 2800e3, -2800e3, 2800e3)
west_antarctica = (-2740e3, 570e3, -2150e3, 1670e3)
east_antarctica = (-840e3, 2880e3, -2400e3, 2600e3)
antarctic_peninsula = (-2600e3, -1200e3, 170e3, 1800e3)
marie_byrd_land = (-1500e3, -500e3, -1350e3, -800e3)
victoria_land = (100e3, 1000e3, -2200e3, -1000e3)
# wilkes_land
# queen_maud_land
saunders_coast = (-980e3, -600e3, -1350e3, -1100e3)

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
# WEST ANTARCTICA
ross_ice_shelf = (-680e3, 470e3, -1420e3, -310e3)
# withrow_ice_shelf =
# swinburne_ice_shelf =
# sulzberger_ice_shelf =
nickerson_ice_shelf = (-980e3, -787e3, -1327e3, -1210e3)
# land_ice_shelf =
# getz_ice_shelf = ()
# dotson_ice_shelf = ()
# crosson_ice_shelf = ()
# thwaites_ice_shelf = ()
# pine_island_ice_shelf = ()
# cosgrove_ice_shelf = ()
# abbott_ice_shelf = ()
# venable_ice_shelf = ()
# ferrigno_ice_shelf = ()
# stange_ice_shelf = ()
# bach_ice_shelf = ()
# wilkins_ice_shelf = ()
george_vi_ice_shelf = (-2150e3, -1690e3, 540e3, 860e3)
# wordie_ice_shelf = ()
larsen_ice_shelf = (-2430e3, -1920e3, 900e3, 1400e3)
# larson_b_ice_shelf = ()
# larson_c_ice_shelf = ()
# larson_d_ice_shelf = ()
# larson_e_ice_shelf = ()
# larson_f_ice_shelf = ()
# larson_g_ice_shelf = ()
ronne_filchner_ice_shelf = (-1550e3, -500e3, 80e3, 1100e3)
ronne_ice_shelf = (-1550e3, -725e3, 45e3, 970e3)
# filchner_ice_shelf =

# EAST ANTARCTICA
# stancomb_brunt_ice_shelf = ()
# riiser_larsen_ice_shelf = ()
# quar_ice_shelf = ()
# ekstrom_ice_shelf = ()
# atka_ice_shelf = ()
# jelbart_ice_shelf = ()
fimbul_ice_shelf = (-260e3, 430e3, 1900e3, 2350e3)
# vigrid_ice_shelf = ()
# nivl_ice_shelf = ()
# lazarev_ice_shelf = ()
# borchgrevink_ice_shelf = ()
baudouin_ice_shelf = (855e3, 1250e3, 1790e3, 2080e3)
# prince_harald_ice_shelf = ()
# shirase_ice_shelf = ()
# rayner_ice_shelf = ()
# edward_vii_ice_shelf = ()
# wilma_ice_shelf = ()
# robert_ice_shelf = ()
# downer_ice_shelf = ()
amery_ice_shelf = (1530e3, 2460e3, 430e3, 1000e3)
# publications_ice_shelf = ()
# west_ice_shelf = ()
# shackleton_ice_shelf = ()
# tracy_tremenchus_ice_shelf = ()
# conger_ice_shelf = ()
# vicennes_ice_shelf = ()
# totten_ice_shelf = ()
# moscow_university_ice_shelf = ()
# holmes_ice_shelf = ()
# dibble_ice_shelf = ()
# mertz_ice_shelf = ()
# ninnis_ice_shelf = ()
# cook_east_ice_shelf = ()
# rennick_ice_shelf = ()
# lillie_ice_shelf = ()
# mariner_ice_shelf = ()
# aviator_ice_shelf = ()
# nansen_ice_shelf = ()
# drygalski_ice_shelf = ()

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

#####
#####
# Greenland
#####
#####

# regions
greenland = (-700e3, 900e3, -3400e3, -600e3)
north_greenland = (-500e3, 600e3, -1200e3, -650e3)
# northwest_greenland = ()
# northeast_greenland = ()
# west_greenland = ()
# east_greenland = ()
# southeast_greenland = ()
# southwest_greenland = ()

# glaciers
kangerlussuaq_glacier = (380e3, 550e3, -2340e3, -2140e3)


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
        "alter_region",
    ]

    return {
        k: v
        for k, v in vars(regions).items()
        if (k not in exclude_list) & (not k.startswith("_"))
    }


def alter_region(
    starting_region: tuple[float, float, float, float],
    zoom: float = 0,
    n_shift: float = 0,
    w_shift: float = 0,
) -> tuple[float, float, float, float]:
    """
    Change a bounding region by shifting the box east/west or north/south, or zooming in
    or out.

    Parameters
    ----------
    starting_region : tuple[float, float, float, float]
        Initial region in meters in format [xmin, xmax, ymin, ymax]
    zoom : float, optional
        zoom in or out, in meters, by default 0
    n_shift : float, optional
        shift north, or south if negative, in meters, by default 0
    w_shift : float, optional
        shift west, or east if negative, in meters, by default 0
    buffer : float, optional
        create new region which is zoomed out in all direction, in meters, by default 0

    Returns
    -------
    tuple[float, float, float, float]
        Returns the altered region
    """
    starting_e, starting_w = starting_region[0], starting_region[1]
    starting_n, starting_s = starting_region[2], starting_region[3]

    xmin = starting_e + zoom + w_shift
    xmax = starting_w - zoom + w_shift

    ymin = starting_n + zoom - n_shift
    ymax = starting_s - zoom - n_shift

    return (xmin, xmax, ymin, ymax)


def combine_regions(
    region1: tuple[float, float, float, float],
    region2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Get the bounding region of 2 regions.

    Parameters
    ----------
    region1 : tuple[float, float, float, float]
        first region, in the format (xmin, xmax, ymin, ymax)
    region2 : tuple[float, float, float, float]
        second region in the format (xmin, xmax, ymin, ymax)

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
    and get the bounding region. Vertices will be returned as the output of the
    function.

    Returns
    -------
    typing.Any
        Returns a list of list of vertices for each polyline.

    Example
    -------
    >>> from polartoolkit import regions, utils
    ...
    >>> polygon = regions.draw_region()
    >>> region = utils.polygon_to_region(polygon, hemisphere="north")
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
