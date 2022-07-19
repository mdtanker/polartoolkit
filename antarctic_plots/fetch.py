# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package: Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#
import pandas as pd
import pooch
import pygmt
import xarray as xr
from pyproj import Transformer


def sample_shp(name):
    """
    load 1 of 2 sample shapefiles
    name =is either 'Disco_deep_transect' or 'Roosevelt_Island'
    """
    path = pooch.retrieve(
        url=f"https://github.com/mdtanker/antarctic_plots/raw/main/data/{name}.zip",
        processor=pooch.Unzip(),
        known_hash=None,
    )
    file = [p for p in path if p.endswith(".shp")][0]
    return file


def imagery():
    """
    Antarctic imagery from LIMA: https://lima.usgs.gov/fullcontinent.php
    will replace with below once figured out login issue with pooch
     MODIS Mosaic of Antarctica: https://doi.org/10.5067/68TBT0CGJSOJ
    Assessed from https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0730_MEASURES_MOA2014_v01/geotiff/
    """
    path = pooch.retrieve(
        # url="https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0730_MEASURES_MOA2014_v01/geotiff/moa750_2014_hp1_v01.tif",
        url="https://lima.usgs.gov/tiff_90pct.zip",
        processor=pooch.Unzip(),
        known_hash=None,
        progressbar=True,
    )
    file = [p for p in path if p.endswith(".tif")][0]
    return file


def groundingline():
    """
    Antarctic groundingline shape file, from https://doi.pangaea.de/10.1594/PANGAEA.819147
    Supplement to Depoorter et al. 2013: https://doi.org/10.1038/nature12567
    """
    path = pooch.retrieve(
        url="https://doi.pangaea.de/10013/epic.42133.d001",
        known_hash=None,
        processor=pooch.Unzip(),
        progressbar=True,
    )  # [3]
    file = [p for p in path if p.endswith(".shp")][0]
    return file


def basement(plot=False, info=False):
    """
    Offshore and sub-Ross Ice Shelf basement topography.
    from Tankersley et al. 2022: https://onlinelibrary.wiley.com/doi/10.1029/2021GL097371
    offshore data from Lindeque et al. 2016: https://doi.org/10.1002/2016GC006401
    """
    path = pooch.retrieve(
        url="https://download.pangaea.de/dataset/941238/files/Ross_Embayment_basement_filt.nc",
        known_hash=None,
        progressbar=True,
    )
    grd = xr.load_dataarray(path)
    if plot == True:
        grd.plot(robust=True)
    if info == True:
        print(pygmt.grdinfo(grd))
    return grd


def bedmap2(layer, plot=False, info=False):
    """
    bedmap2 data, from https://doi.org/10.5194/tc-7-375-2013.
    layer is one of following strings:
        'thickness', 'bed', 'surface', 'geiod_to_WGS84'
    """
    path = pooch.retrieve(
        url="https://secure.antarctica.ac.uk/data/bedmap2/bedmap2_tiff.zip",
        known_hash=None,
        processor=pooch.Unzip(),
        progressbar=True,
    )
    file = [p for p in path if p.endswith(f"{layer}.tif")][0]
    grd = xr.load_dataarray(file)
    grd = grd.squeeze()
    if layer == "surface":
        grd = grd.fillna(0)
        print('filling grid nans with "0"')
    if plot == True:
        grd.plot(robust=True)
    if info == True:
        print(pygmt.grdinfo(grd))
    return grd


def deepbedmap(plot=False, info=False, region=None, spacing=10e3):
    """
    DeepBedMap, from Leong and Horgan, 2020: https://doi.org/10.5194/tc-14-3687-2020
    Accessed from https://zenodo.org/record/4054246#.Ysy344RByp0
    """
    if region == None:
        region = (-2700000, 2800000, -2200000, 2300000)
    path = pooch.retrieve(
        url="https://zenodo.org/record/4054246/files/deepbedmap_dem.tif?download=1",
        known_hash=None,
        progressbar=True,
    )
    grd = pygmt.grdfilter(
        grid=path,
        filter=f"g{spacing}",
        spacing=spacing,
        region=region,
        distance="0",
        nans="r",
        verbose="q",
    )
    if plot == True:
        grd.plot(robust=True)
    if info == True:
        print(pygmt.grdinfo(grd))
    return grd


def gravity(type, plot=False, info=False, region=None, spacing=5e3):
    """
    Preliminary compilation of Antarctica gravity and gravity gradient data.
    Updates on 2016 AntGG compilation.
    Accessed from https://ftp.space.dtu.dk/pub/RF/4D-ANTARCTICA/.
    type is either 'FA' or 'BA', for free-air and bouguer anomalies, respectively.
    """
    if region == None:
        region = (-3330000, 3330000, -3330000, 3330000)
    path = pooch.retrieve(
        url="https://ftp.space.dtu.dk/pub/RF/4D-ANTARCTICA/ant4d_gravity.zip",
        known_hash=None,
        processor=pooch.Unzip(),
        progressbar=True,
    )
    file = [p for p in path if p.endswith(".dat")][0]
    df = pd.read_csv(
        file,
        delim_whitespace=True,
        skiprows=3,
        names=["id", "lat", "lon", "FA", "Err", "DG", "BA"],
    )
    transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
    df["x"], df["y"] = transformer.transform(df.lat.tolist(), df.lon.tolist())
    df = pygmt.blockmedian(
        df[["x", "y", type]], spacing=spacing, region=region, verbose="q"
    )
    grd = pygmt.surface(
        data=df[["x", "y", type]], spacing=spacing, region=region, M="2c", verbose="q"
    )
    if plot == True:
        grd.plot(robust=True)
    if info == True:
        print(pygmt.grdinfo(grd))
    return grd


def magnetics(version, plot=False, info=False, region=None, spacing=5e3):
    """
    version, one of following strings:
        'admap2', 'admap1', 'admap2_eq_src', 'admap2_gdb'
    1)
    ADMAP2 magnetic anomaly compilation of Antarctica.
    Non-geosoft specific files provide from Sasha Golynsky.
    2)
    ADMAP-2001 magnetic anomaly compilation of Antarctica.
    https://admap.kongju.ac.kr/databases.html
    3)
    ADMAP2 eqivalent sources, from https://admap.kongju.ac.kr/admapdata/
    4)
    Geosoft-specific .gdb abridged files.
    Accessed from https://doi.pangaea.de/10.1594/PANGAEA.892722?format=html#download

    """
    if region == None:
        region = (-3330000, 3330000, -3330000, 3330000)
    if version == "admap2":
        path = "../data/ADMAP_2B_2017_R9_BAS_.tif"
        grd = xr.load_dataarray(path)
        grd = grd.squeeze()
        grd = pygmt.grdfilter(
            grid=grd,
            filter=f"g{spacing}",
            spacing=spacing,
            region=region,
            distance="0",
            nans="r",
            verbose="q",
        )
    elif version == "admap1":
        path = pooch.retrieve(
            url="https://admap.kongju.ac.kr/admapdata/ant_new.zip",
            known_hash=None,
            processor=pooch.Unzip(),
            progressbar=True,
        )[0]
        df = pd.read_csv(
            path, delim_whitespace=True, header=None, names=["lat", "lon", "nT"]
        )
        transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
        df["x"], df["y"] = transformer.transform(df.lat.tolist(), df.lon.tolist())
        df = pygmt.blockmedian(
            df[["x", "y", "nT"]], spacing=spacing, region=region, verbose="q"
        )
        grd = pygmt.surface(
            data=df[["x", "y", "nT"]],
            spacing=spacing,
            region=region,
            M="2c",
            verbose="q",
        )
    elif version == "admap2_eq_src":
        path = pooch.retrieve(
            url="https://admap.kongju.ac.kr/admapdata/ADMAP_2S_EPS_20km.zip",
            known_hash=None,
            processor=pooch.Unzip(),
            progressbar=True,
        )[0]
        df = pd.read_csv(
            path,
            delim_whitespace=True,
            header=None,
            names=["lat", "lon", "z", "eq_source"],
        )
        transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
        df["x"], df["y"] = transformer.transform(df.lat.tolist(), df.lon.tolist())
        df = pygmt.blockmedian(
            df[["x", "y", "eq_source"]], spacing=spacing, region=region, verbose="q"
        )
        grd = pygmt.surface(
            data=df[["x", "y", "eq_source"]],
            spacing=spacing,
            region=region,
            M="2c",
            verbose="q",
        )
    elif version == "admap2_gdb":
        files = pooch.retrieve(
            url="https://hs.pangaea.de/mag/airborne/Antarctica/ADMAP2A.zip",
            known_hash=None,
            processor=pooch.Unzip(),
            progressbar=True,
        )
    else:
        print("invalid version string")
    if plot == True:
        grd.plot(robust=True)
    if info == True:
        print(pygmt.grdinfo(grd))
    return grd


# def geothermal(plot=False, info=False):
#     """
#     Mean geothermal heat flow from various models.
#     From Burton-Johnson et al. 2020: Review article: Geothermal heat flow in Antarctica: current and future directions
#     """
#     path = pooch.retrieve(
#         url="https://doi.org/10.5194/tc-14-3843-2020-supplement",
#         known_hash=None,
#         processor=pooch.Unzip(
#             extract_dir='Burton_Johnson_2020',),
#         progressbar=True,)
#     file = [p for p in path if p.endswith('Mean.tif')][0]
#     grd = xr.load_dataarray(file)
#     grd = grd.squeeze()

#     if plot==True:
#         grd.plot(robust=True)
#     if info==True:
#         print(pygmt.grdinfo(grd))
#     return grd
