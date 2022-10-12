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
Follow this format:
def test_():
    grid = fetch.()
    expected = 
    assert utils.get_grid_info(grid) == expected
"""
#%%
from antarctic_plots import utils, fetch
import pytest, os

try:
    earthdata_login = [
        os.environ.get("EARTHDATA_USERNAME", None), 
        os.environ.get("EARTHDATA_PASSWORD", None)]
except:
    earthdata_login = [None, None]

# create skipif decorator for fetch calls which use NSIDC Earthdata logins
skip_earthdata = pytest.mark.skipif(
    earthdata_login == [None, None], 
    reason='requires earthdata login credentials set as environment variables')

#%%
# resample_grid
test = [
    # no inputs
    (dict(),
    ('10000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -384.5, 204.800003052, 'g')
    ),
    # return original with given initials
    (dict(
        initial_region=[-3330000.0, 3330000.0, -3330000.0, 3330000.0],
        initial_spacing=10e3,
        initial_registration='g',
    ), 
    ('10000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -384.5, 204.800003052, 'g')
    ),
    # give false initial values, return actual initial values
    (dict(
        initial_region=[-2800000.0, 2800000.0, -2800000.0, 2800000.0],
        initial_spacing=8e3,
        initial_registration='p',
    ),
    ('10000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -384.5, 204.800003052, 'g')
    ),
    # Only registration is different
    (dict(
        registration='p',
    ),
    ('10000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -337.490234375, 170.69921875, 'p')
    ),
    # smaller spacing, uneven, reset region to keep exact spacing
    (dict(spacing=8212),
    ('8212', [-3325860.0, 3325860.0, -3325860.0, 3325860.0], -374.47366333, 182.33392334, 'g')
    ),
    # larger spacing, uneven, reset region to keep exact spacing
    (dict(spacing=10119),
    ('10119', [-3329151.0, 3329151.0, -3329151.0, 3329151.0], -318.772613525, 177.986114502, 'g')
    ),
    # uneven subregion, reset region to keep exact spacing
    (dict(region=[210012.0, 390003.0, -1310217.0, -1121376.0]),
    ('10000', [210000.0, 400000.0, -1320000.0, -1120000.0], -175.400009155, 54.1000022888, 'g')
    ),
    # uneven subregion with diff reg, reset region to keep exact spacing
    (dict(
        region=[210012.0, 390003.0, -1310217.0, -1121376.0],
        registration='p',
    ),
    ('10000', [210000.0, 400000.0, -1320000.0, -1120000.0], -156.026565552, 46.8070335388, 'p')
    ),
    # uneven spacing (smaller) and uneven region, reset region to keep exact spacing
    (dict(
        spacing=8212,
        region=[210012.0, 390003.0, -1310217.0, -1121376.0],
    ),
    ('8212', [205300.0, 402388.0, -1322132.0, -1116832.0], -170.436401367, 47.9773178101, 'g')
    ),
    # uneven spacing (larger) and uneven region, reset region to keep exact spacing
    (dict(
        spacing=10119,
        region=[210012.0, 390003.0, -1310217.0, -1121376.0],
    ),
    ('10119', [212499.0, 384522.0, -1305351.0, -1123209.0], -173.363143921, 50.2054672241, 'g')
    ),
    # larger than initial region, return initial region
    (dict(region=[-3400e3, 3400e3, -3400e3, 34030e3]),
    ('10000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -384.5, 204.800003052, 'g')
    ),
]

@pytest.mark.parametrize("test_input,expected", test)
def test_resample_grid(test_input, expected):
    grid = fetch.gravity(version='antgg', anomaly_type='FA')
    resampled = fetch.resample_grid(grid, **test_input)
    assert utils.get_grid_info(resampled) == pytest.approx(expected, rel=0.1)

# test_input = dict(
#     spacing=10119,
#     region=[-3400e3, 3400e3, -3400e3, 34030e3],
#     registration='p',
# )
# grid = fetch.gravity(version='antgg', anomaly_type='FA')
# resampled = fetch.resample_grid(grid, **test_input)
# utils.get_grid_info(resampled)

#%%
# ice_vel
@pytest.mark.earthdata
@skip_earthdata
def test_ice_vel_lowres():
    resolution='lowres'
    grid = fetch.ice_vel(resolution=resolution)
    expected = ('5000', [-2800000.0, 2795000.0, -2795000.0, 2800000.0], -15.5856771469, 4201.70605469, 'g')
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

@pytest.mark.slow
@pytest.mark.earthdata
@skip_earthdata
def test_ice_vel_highres():
    resolution='highres'
    grid = fetch.ice_vel(resolution=resolution)
    expected = ('450', [-2800000.0, 2799800.0, -2799800.0, 2800000.0], 2.34232032881e-07, 4218.26513672, 'g')
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

# grid = fetch.ice_vel(resolution='lowres')
# utils.get_grid_info(grid)

#%%
# modis_moa
@pytest.mark.slow
@pytest.mark.earthdata
@skip_earthdata
def test_modis_moa():
    version=750
    grid = fetch.modis_moa(version=version)
    expected = ('750', [-3174450.0, 2867550.0, -2815925.0, 2405575.0], 0.0, 42374.0, 'p')
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

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
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

# grid = fetch.basement()
# utils.get_grid_info(grid)

#%%
# bedmachine
# test for all layers, but only test reference models with 1 layer

test = [
    ('icebase', 
        ('500', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], -3856.74609375, 4818.15527344, 'g')
    ),
    ('surface', 
        ('500', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], 0.0, 4818.15527344, 'g')
    ),
    ('thickness', 
        ('500', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], 0.0, 4726.54638672, 'g')
    ),
    ('bed', 
        ('500', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], -8151.48730469, 4818.15527344, 'g')
    ),
    ('geoid', 
        ('500', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], -66.0, 52.0, 'g')
    )
]
@pytest.mark.slow
@pytest.mark.earthdata
@skip_earthdata
@pytest.mark.parametrize("test_input,expected", test)
def test_bedmachine(test_input, expected):
    grid = fetch.bedmachine(test_input)
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

@pytest.mark.earthdata
@skip_earthdata
def test_bedmachine_reference():
    grid = fetch.bedmachine(layer='surface', reference="ellipsoid")
    expected = ('500', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], -66.0, 4797.15527344, 'g')
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

# grid = fetch.bedmachine(layer='surface', reference="ellipsoid")
# utils.get_grid_info(grid)

#%%
# bedmap2
# test for all layers, but only test reference models with 1 layer

test = [
    ('icebase', 
        ('1000', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], -2736.0, 3972.0, 'g')
    ),
    ('surface', 
        ('1000', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], 0.0, 4082.0, 'g')
    ),
    ('thickness', 
        ('1000', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], 0.0, 4621.0, 'g')
    ),
    ('bed', 
        ('1000', [-3333500.0, 3333500.0, -3332500.0, 3332500.0], -7054.0, 3972.0, 'p')
    ),
    ('gl04c_geiod_to_WGS84', 
        ('1000', [-3333500.0, 3333500.0, -3332500.0, 3332500.0], -65.8680496216, 36.6361198425, 'p')
    )
]
@pytest.mark.slow
@pytest.mark.parametrize("test_input,expected", test)
def test_bedmap(test_input, expected):
    grid = fetch.bedmap2(test_input)
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

def test_bedmap2_reference():
    grid = fetch.bedmap2(layer='surface', reference="ellipsoid")
    expected = ('1000', [-3333000.0, 3333000.0, -3333000.0, 3333000.0], 0.0, 8164.0, 'g')
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

# grid = fetch.bedmap2(layer='gl04c_geiod_to_WGS84')
# utils.get_grid_info(grid)

#%%
# deepbedmap
@pytest.mark.slow
def test_deepbedmap():
    grid = fetch.deepbedmap()
    expected = ('250', [-2700000.0, 2800000.0, -2199750.0, 2299750.0], -6156.0, 4215.0, 'p')
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

# grid = fetch.deepbedmap()
# utils.get_grid_info(grid)

#%%
# gravity
# only testing 1 anomaly type (FA) for each version

test = [
    ('antgg', 
        ('10000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -384.5, 204.800003052, 'g')
    ),
    ('antgg-update', 
        ('10000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -237.559997559, 171.86000061, 'g')
    ),
    ('eigen',   
        ('5000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], 977835.3125, 980167.75, 'g')
    )
]

@pytest.mark.parametrize("test_input,expected", test)
def test_gravity(test_input, expected):
    grid = fetch.gravity(test_input, anomaly_type='FA')
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

# grid = fetch.gravity(version='eigen')
# utils.get_grid_info(grid)

#%%
# magnetics

test = [
    ('admap1', 
        ('5000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -936.875427246, 1766.1373291, 'g')
    ),
]

@pytest.mark.parametrize("test_input,expected", test)
def test_magnetics(test_input, expected):
    grid = fetch.magnetics(test_input)
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

# grid = fetch.magnetics(version='admap1')
# utils.get_grid_info(grid)

#%%
# ghf

test = [
    ('an-2015', 
        ('5000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], 26.547088623, 102.363540649, 'g')
    ),
    ('martos-2017', 
        ('15000', [-2535000.0, 2715000.0, -2130000.0, 2220000.0], 42.6263694763, 240.510910034, 'g')
    ),
    ('burton-johnson-2020',   
        ('17000', [-2543500.0, 2624500.0, -2121500.0, 2213500.0], 42.2533454895, 106.544433594, 'p')
    )
]

@pytest.mark.parametrize("test_input,expected", test)
def test_ghf(test_input, expected):
    grid = fetch.ghf(test_input)
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

def test_ghf_points():
    df = fetch.ghf(version='burton-johnson-2020', points=True)
    expected = [-56.5667, 34.1833, 'C11-44', 0.0, 11, 300, 0.77, 229.0, -5372.0,
        'Anderson1977', 'https://doi.org/10.1594/PANGAEA.796541', 'S3', 
        'Unconsolidated sediments', 2098568.3517061966, 3089886.43259545,229.002]
    assert df.iloc[0].dropna().tolist() == pytest.approx(expected, rel=0.1)

# df = fetch.ghf(version='burton-johnson-2020', points=True)
# df.iloc[0].dropna().tolist()

# grid = fetch.ghf(version='burton-johnson-2020')
# utils.get_grid_info(grid)

#%%
# gia

test = [
    ('stal-2020', 
       ('10000', [-2800000.0, 2800000.0, -2800000.0, 2800000.0], -2953.8605957, 3931.43554688, 'p') 
    ),
]

@pytest.mark.parametrize("test_input,expected", test)
def test_gia(test_input, expected):
    grid = fetch.gia(test_input)
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

# grid = fetch.gia(version='stal-2020')
# utils.get_grid_info(grid)

#%%
# crustal_thickness

test = [
    ('shen-2018', 
       ('10000', [-2800000.0, 2800000.0, -2800000.0, 2800000.0], 17216.1484375, 57233.3320313, 'g')
    ),
    ('an-2015',
        ('5000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], 6264.68212891, 65036.3632813, 'g')
    )
]

@pytest.mark.parametrize("test_input,expected", test)
def test_crustal_thickness(test_input, expected):
    grid = fetch.crustal_thickness(test_input)
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

# grid = fetch.crustal_thickness(version='an-2015')
# utils.get_grid_info(grid)

#%%
# moho   

test = [
    ('shen-2018', 
       ('10000', [-2800000.0, 2800000.0, -2800000.0, 2800000.0], -57223.5273438, -17218.0996094, 'g')
    ),
    ('an-2015',
        ('5000', [-3330000.0, 3330000.0, -3330000.0, 3330000.0], -65036.3632813, -6264.68212891, 'g')
    )
]

@pytest.mark.parametrize("test_input,expected", test)
def test_moho(test_input, expected):
    grid = fetch.moho(test_input)
    assert utils.get_grid_info(grid) == pytest.approx(expected, rel=0.1)

# grid = fetch.moho(version='shen-2018')
# utils.get_grid_info(grid)

# %%