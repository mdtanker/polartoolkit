# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#
"""
Tests for fetch module. Use pre-determined results of utils.get_grid_info() to verify 
grids have been properly fetch. Also tests the `resample_grid()` function
Follow this formate:
def test_():
    grid = fetch.()
    expected = 
    assert utils.get_grid_info(grid) == expected
"""
#%%
from antarctic_plots import utils, fetch
import pytest

#%%
# ice_vel

# def test_ice_vel_lowres():
#     resolution='lowres'
#     grid = fetch.ice_vel(resolution=resolution)
#     expected = ('5000', [-2800000.0, 2795000.0, -2795000.0, 2800000.0], -15.5856771469, 4201.70605469, 'g')
#     assert utils.get_grid_info(grid) == expected

# def test_ice_vel_highres():
#     resolution='highres'
#     grid = fetch.ice_vel(resolution=resolution)
#     expected = ('450', [-2800000.0, 2799800.0, -2799800.0, 2800000.0], 2.34232032881e-07, 4218.26513672, 'g')
#     assert utils.get_grid_info(grid) == expected

# grid = fetch.ice_vel(resolution='lowres')
# utils.get_grid_info(grid)

#%%
# modis_moa

# def test_modis_moa():
#     version=750
#     grid = fetch.modis_moa(version=version)
#     expected = ('750', [-3174450.0, 2867550.0, -2815925.0, 2405575.0], 0.0, 42374.0, 'p')
#     assert utils.get_grid_info(grid) == expected

# version=125 not testing since too large

# grid = fetch.modis_moa(version=750)
# utils.get_grid_info(grid)

#%%
# imagery, not testing, too large

# grid = fetch.imagery()
# utils.get_grid_info(grid)

#%%
# basement

def test_basement():
    grid = fetch.basement()
    expected = ('5000', [-3330000.0, 1900000.0, -3330000.0, 1850000.0], -8503.13378906, 78.269317627, 'p')
    assert utils.get_grid_info(grid) == expected

# grid = fetch.basement()
# utils.get_grid_info(grid)

#%%
# bedmachine
# test for all layers, but only test reference models with 1 layer

# test = [
#     ('icebase', 
#         ('500', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], -3856.74609375, 4818.15527344, 'g')
#     ),
#     ('surface', 
#         ('500', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], 0.0, 4818.15527344, 'g')
#     ),
#     ('thickness', 
#         ('500', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], 0.0, 4726.54638672, 'g')
#     ),
#     ('bed', 
#         ('500', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], -8151.48730469, 4818.15527344, 'g')
#     ),
#     ('geoid', 
#         ('500', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], -66.0, 52.0, 'g')
#     )
# ]
# @pytest.mark.parametrize("test_input,expected", test)
# def test_bedmachine(test_input, expected):
#     grid = fetch.bedmachine(test_input)
#     assert utils.get_grid_info(grid) == expected

# def test_bedmachine_reference():
#     grid = fetch.bedmachine(layer='surface', reference="ellipsoid")
#     expected = ('500', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], -66.0, 4797.15527344, 'g')
#     assert utils.get_grid_info(grid) == expected

# grid = fetch.bedmachine(layer='surface', reference="ellipsoid")
# utils.get_grid_info(grid)

# #%%
# # bedmap2
# # test for all layers, but only test reference models with 1 layer

# test = [
#     ('icebase', 
#         ('1000', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], -2736.0, 3972.0, 'g')
#     ),
#     ('surface', 
#         ('1000', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], 0.0, 4082.0, 'g')
#     ),
#     ('thickness', 
#         ('1000', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], 0.0, 4621.0, 'g')
#     ),
#     ('bed', 
#         ('1000', [-3333500.0, 3333500.0, -3332500.0, 3332500.0], -7054.0, 3972.0, 'p')
#     ),
#     ('gl04c_geiod_to_WGS84', 
#         ('1000', [-3333500.0, 3333500.0, -3332500.0, 3332500.0], -65.8680496216, 36.6361198425, 'p')
#     )
# ]
# @pytest.mark.parametrize("test_input,expected", test)
# def test_bedmap(test_input, expected):
#     grid = fetch.bedmap2(test_input)
#     assert utils.get_grid_info(grid) == expected

# def test_bedmap2_reference():
#     grid = fetch.bedmap2(layer='surface', reference="ellipsoid")
#     expected = ('1000', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], 0.0, 8164.0, 'g')
#     assert utils.get_grid_info(grid) == expected

# # grid = fetch.bedmap2(layer='gl04c_geiod_to_WGS84')
# # utils.get_grid_info(grid)

# #%%
# # deepbedmap

# def test_deepbedmap():
#     grid = fetch.deepbedmap()
#     expected = ('250', [-2700000.0, 2800000.0, -2199750.0, 2299750.0], -6156.0, 4215.0, 'p')
#     assert utils.get_grid_info(grid) == expected

# # grid = fetch.deepbedmap()
# # utils.get_grid_info(grid)

# #%%
# # gravity
# # only testing 1 anomaly type (FA) for each version

# test = [
#     ('antgg', 
#         ('10000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -384.5, 204.800003052, 'g')
#     ),
#     ('antgg-update', 
#         ('10000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -237.559997559, 171.86000061, 'g')
#     ),
#     ('eigen',   
#         ('5000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], 977835.3125, 980167.75, 'g')
#     )
# ]

# @pytest.mark.parametrize("test_input,expected", test)
# def test_gravity(test_input, expected):
#     grid = fetch.gravity(test_input, anomaly_type='FA')
#     assert utils.get_grid_info(grid) == expected

# # grid = fetch.gravity(version='eigen')
# # utils.get_grid_info(grid)

# #%%
# # magnetics

# test = [
#     ('admap1', 
#         ('5000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -936.875427246, 1766.1373291, 'g')
#     ),
# ]

# @pytest.mark.parametrize("test_input,expected", test)
# def test_magnetics(test_input, expected):
#     grid = fetch.magnetics(test_input)
#     assert utils.get_grid_info(grid) == expected

# # grid = fetch.magnetics(version='admap1')
# # utils.get_grid_info(grid)

# #%%
# # geothermal

# test = [
#     ('an-2015', 
#         ('5000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], 26.547088623, 102.363540649, 'g')
#     ),
#     ('martos-2017', 
#         ('15000', [-2535000.0, 2715000.0, -2130000.0, 2220000.0], 42.6263694763, 240.510910034, 'g')
#     ),
#     ('burton-johnson-2020',   
#         ('17000', [-2543500.0, 2624500.0, -2121500.0, 2213500.0], 42.2533454895, 106.544433594, 'p')
#     )
# ]

# @pytest.mark.parametrize("test_input,expected", test)
# def test_geothermal(test_input, expected):
#     grid = fetch.geothermal(test_input)
#     assert utils.get_grid_info(grid) == expected

# # grid = fetch.geothermal(version='burton-johnson-2020')
# # utils.get_grid_info(grid)

# #%%
# # gia

# test = [
#     ('stal-2020', 
#        ('10000', [-2800000.0, 2800000.0, -2800000.0, 2800000.0], -2953.8605957, 3931.43554688, 'p') 
#     ),
# ]

# @pytest.mark.parametrize("test_input,expected", test)
# def test_gia(test_input, expected):
#     grid = fetch.gia(test_input)
#     assert utils.get_grid_info(grid) == expected

# # grid = fetch.gia(version='stal-2020')
# # utils.get_grid_info(grid)

# #%%
# # crustal_thickness

# test = [
#     ('shen-2018', 
#        ('10000', [-2800000.0, 2800000.0, -2800000.0, 2800000.0], 17216.1484375, 57233.3320313, 'g')
#     ),
#     ('an-2015',
#         ('5000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], 6264.68212891, 65036.3632813, 'g')
#     )
# ]

# @pytest.mark.parametrize("test_input,expected", test)
# def test_crustal_thickness(test_input, expected):
#     grid = fetch.crustal_thickness(test_input)
#     assert utils.get_grid_info(grid) == expected

# # grid = fetch.crustal_thickness(version='an-2015')
# # utils.get_grid_info(grid)

# #%%
# # moho   

# test = [
#     ('shen-2018', 
#        ('10000', [-2800000.0, 2800000.0, -2800000.0, 2800000.0], -57223.5273438, -17218.0996094, 'g')
#     ),
#     ('an-2015',
#         ('5000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -65036.3632813, -6264.68212891, 'g')
#     )
# ]

# @pytest.mark.parametrize("test_input,expected", test)
# def test_moho(test_input, expected):
#     grid = fetch.moho(test_input)
#     assert utils.get_grid_info(grid) == expected

# # grid = fetch.moho(version='shen-2018')
# # utils.get_grid_info(grid)

# # %%