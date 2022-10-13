# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#
"""
Bounding regions for commonly plotted Antarctic regions. In Polar Stereographic
Projection (EPSG:3031). The format is [East, West, North, South], in meters.
"""
import pandas as pd
import verde as vd

from antarctic_plots import maps, utils

try:
    import ipyleaflet
except ImportError:
    _has_ipyleaflet = False
else:
    _has_ipyleaflet = True

# regions
antarctica = [-2800e3, 2800e3, -2800e3, 2800e3]
west_antarctica = [-2740e3, 570e3, -2150e3, 1670e3]
east_antarctica = [-840e3, 2880e3, -2400e3, 2600e3]
antarctic_peninsula = [-2600e3, -1200e3, 170e3, 1800e3]
marie_byrd_land = [-1500e3, -500e3, -1350e3, -800e3]
victoria_land = [100e3, 1000e3, -2200e3, -1000e3]
# wilkes_land
# queen_maud_land

# study_sites
roosevelt_island = [-480e3, -240e3, -1220e3, -980e3]
ross_island = [210e3, 360e3, -1400e3, -1250e3]
minna_bluff = [210e3, 390e3, -1310e3, -1120e3]
# discovery_deep
mcmurdo_dry_valleys = [320e3, 480e3, -1400e3, -1220e3]
siple_coast = [-700e3, 30e3, -1110e3, -450e3]
crary_ice_rise = [-330e3, -40e3, -830e3, -480e3]
siple_dome = [-630e3, -270e3, -970e3, -630e3]

# ice_shelves
ross_ice_shelf = [-680e3, 470e3, -1420e3, -310e3]
# getz_ice_shelf = []
# abbott_ice_shelf = []
# george_vi_ice_shelf = []
# wilkins_ice_shelf = []
larsen_ice_shelf = [-2430e3, -1920e3, 900e3, 1400e3]
ronne_filchner_ice_shelf = [-1550e3, -500e3, 80e3, 1200e3]
# riiser_larsen_ice_shelf = []
# fimbul_ice_shelf = []
amery_ice_shelf = [1530e3, 2460e3, 430e3, 1000e3]
# west_ice_shelf = []
# shackleton_ice_shelf = []
# brunt_ice_shelf = []

# glaciers
# byrd_glacier
# nimrod_glacier
pine_island_glacier = [-1720e3, -1480e3, -380e3, -70e3]
thwaites_glacier = [-1650e3, -1200e3, -600e3, -300e3]
kamb_ice_stream = [-620e3, -220e3, -800e3, -400e3]
# whillans_ice_stream = []

# seas
ross_sea = [-500e3, 450e3, -2100e3, -1300e3]
# amundsen_sea
# bellinghausen_sea
# weddell_sea

# ice catchements


def combine_regions(
    region1: list,
    region2: list,
):
    """
    Get the bounding region of 2 regions.

    Parameters
    ----------
    region1 : list
        first region
    region2 : list
        second region

    Returns
    -------
    list
        Bounding region of the 2 supplied regions.
    """
    coords1 = utils.reg_str_to_df(region1)
    coords2 = utils.reg_str_to_df(region2)
    coords_combined = pd.concat((coords1, coords2))
    region = vd.get_region((coords_combined.x, coords_combined.y))

    return region


def draw_region(**kwargs):
    """
    Plot an interactive map, and use the "Draw a Rectangle" button to draw a rectangle
    and get the bounding region. Verticles will be returned as the output of the
    function.

    Returns
    -------
    tuple
        Returns a tuple of list of vertices for each polyline.
    """

    m = maps.interactive_map(**kwargs, show=False)

    def clear_m():
        global poly
        poly = list()

    clear_m()

    myDrawControl = ipyleaflet.DrawControl(
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

    def handle_rect_draw(self, action, geo_json):
        global poly
        shapes = []
        for coords in geo_json["geometry"]["coordinates"][0][:-1][:]:
            shapes.append(list(coords))
        shapes = list(shapes)
        if action == "created":
            poly.append(shapes)

    myDrawControl.on_draw(handle_rect_draw)
    m.add_control(myDrawControl)

    clear_m()
    display(m)  # noqa

    return poly
