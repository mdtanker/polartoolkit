# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#
import pandas as pd
import pooch
import pygmt
import xarray as xr
from pyproj import Transformer


def sample_shp(name: str) -> str:
    """
    Load the file path of sample shapefiles

    Parameters
    ----------
    name : str
        chose which sample shapefile to load, either 'Disco_deep_transect' or
        'Roosevelt_Island'

    Returns
    -------
    str
        file path as a string
    """
    path = pooch.retrieve(
        url=f"https://github.com/mdtanker/antarctic_plots/raw/main/data/{name}.zip",
        processor=pooch.Unzip(),
        known_hash=None,
    )
    file = [p for p in path if p.endswith(".shp")][0]
    return file


def imagery() -> str:
    """
    Load the file path of Antarctic imagery geotiff from LIMA:
    https://lima.usgs.gov/fullcontinent.php
    will replace with below once figured out login issue with pooch
    MODIS Mosaic of Antarctica: https://doi.org/10.5067/68TBT0CGJSOJ
    Assessed from https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0730_MEASURES_MOA2014_v01/geotiff/ # noqa

    Returns
    -------
    str
        file path
    """
    path = pooch.retrieve(
        # url="https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0730_MEASURES_MOA2014_v01/geotiff/moa750_2014_hp1_v01.tif", # noqa
        url="https://lima.usgs.gov/tiff_90pct.zip",
        processor=pooch.Unzip(),
        known_hash=None,
        progressbar=True,
    )
    file = [p for p in path if p.endswith(".tif")][0]
    return file


def groundingline() -> str:
    """
    Load the file path of a groundingline shapefile
    Antarctic groundingline shape file, from
    https://doi.pangaea.de/10.1594/PANGAEA.819147
    Supplement to Depoorter et al. 2013: https://doi.org/10.1038/nature12567

    Returns
    -------
    str
        file path
    """
    path = pooch.retrieve(
        url="https://doi.pangaea.de/10013/epic.42133.d001",
        known_hash=None,
        processor=pooch.Unzip(),
        progressbar=True,
    )
    file = [p for p in path if p.endswith(".shp")][0]
    return file


def basement(plot: bool = False, info: bool = False) -> xr.DataArray:
    """
    Load a grid of basement topography.
    Offshore and sub-Ross Ice Shelf basement topography.
    from Tankersley et al. 2022:
    https://onlinelibrary.wiley.com/doi/10.1029/2021GL097371
    offshore data from Lindeque et al. 2016: https://doi.org/10.1002/2016GC006401

    Parameters
    ----------
    plot : bool, optional
        plot the fetched grid, by default False
    info : bool, optional
        print info on the fetched grid, by default False

    Returns
    -------
    xr.DataArray
        dataarray of basement depths
    """
    path = pooch.retrieve(
        url="https://download.pangaea.de/dataset/941238/files/Ross_Embayment_basement_filt.nc",  # noqa
        known_hash=None,
        progressbar=True,
    )
    grd = xr.load_dataarray(path)
    if plot is True:
        grd.plot(robust=True)
    if info is True:
        print(pygmt.grdinfo(grd))
    return grd


def bedmap2(
    layer: str,
    plot: bool = False,
    info: bool = False,
) -> xr.DataArray:
    """
    Load bedmap2 data. Note, nan's in surface grid are set to 0.
    from https://doi.org/10.5194/tc-7-375-2013.

    Parameters
    ----------
    layer : str
        choose which layer to fetch, 'thickness', 'bed', 'surface', or 'geiod_to_WGS84'
    plot : bool, optional
        choose to plot grid, by default False
    info : bool, optional
        choose to print info on grid, by default False

    Returns
    -------
    xr.DataArray
        Returns a bedmap2 grid
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
    if plot is True:
        grd.plot(robust=True)
    if info is True:
        print(pygmt.grdinfo(grd))
    return grd


def deepbedmap(
    plot: bool = False, info: bool = False, region=None, spacing=10e3
) -> xr.DataArray:
    """
    Load DeepBedMap data,  from Leong and Horgan, 2020:
    https://doi.org/10.5194/tc-14-3687-2020
    Accessed from https://zenodo.org/record/4054246#.Ysy344RByp0

    Parameters
    ----------
    plot : bool, optional
        choose to plot grid, by default False
    info : bool, optional
        choose to print info on grid, by default False
    region : str or np.ndarray, optional
        GMT-format region to clip the loaded grid to, by default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 10e3

    Returns
    -------
    xr.DataArray
        Returns a loaded, and optional clip/resampled grid of DeepBedMap.
    """

    if region is None:
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
    if plot is True:
        grd.plot(robust=True)
    if info is True:
        print(pygmt.grdinfo(grd))
    return grd


def gravity(
    type: str, plot: bool = False, info: bool = False, region=None, spacing=10e3
) -> xr.DataArray:
    """
    Loads an Antarctic gravity grid
    Preliminary compilation of Antarctica gravity and gravity gradient data.
    Updates on 2016 AntGG compilation.
    Accessed from https://ftp.space.dtu.dk/pub/RF/4D-ANTARCTICA/.

    Parameters
    ----------
    type : str
        either 'FA' or 'BA', for free-air and bouguer anomalies, respectively.
    plot : bool, optional
        choose to plot grid, by default False
    info : bool, optional
        choose to print info on grid, by default False
    region : str or np.ndarray, optional
        GMT-format region to clip the loaded grid to, by default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 10e3

    Returns
    -------
    xr.DataArray
        Returns a loaded, and optional clip/resampled grid of either free-air or
        Bouguer gravity anomalies.
    """

    if region is None:
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
    if plot is True:
        grd.plot(robust=True)
    if info is True:
        print(pygmt.grdinfo(grd))
    return grd


def magnetics(
    version: str, plot: bool = False, info: bool = False, region=None, spacing=10e3
) -> xr.DataArray:
    """
    Load 1 of 4 'versions' of Antarctic magnetic anomaly grid.
    version='admap2'
    ADMAP2 magnetic anomaly compilation of Antarctica. Non-geosoft specific files
    provide from Sasha Golynsky.
    version='admap1'
    ADMAP-2001 magnetic anomaly compilation of Antarctica.
    https://admap.kongju.ac.kr/databases.html
    version='admap2_eq_src'
    ADMAP2 eqivalent sources, from https://admap.kongju.ac.kr/admapdata/
    version='admap2_gdb'
    Geosoft-specific .gdb abridged files. Accessed from
    https://doi.pangaea.de/10.1594/PANGAEA.892722?format=html#download

    Parameters
    ----------
    version : str
        Either 'admap2', 'admap1', 'admap2_eq_src', or 'admap2_gdb'
    plot : bool, optional
        choose to plot grid, by default False
    info : bool, optional
        choose to print info on grid, by default False
    region : str or np.ndarray, optional
        GMT-format region to clip the loaded grid to, by default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 10e3

    Returns
    -------
    xr.DataArray
        Returns a loaded, and optional clip/resampled grid of magnetic anomalies.
    """
    if region is None:
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
    if plot is True:
        grd.plot(robust=True)
    if info is True:
        print(pygmt.grdinfo(grd))
    return grd


# def geothermal(plot=False, info=False)-> xr.DataArray:
#
#    # Mean geothermal heat flow from various models.
#    # From Burton-Johnson et al. 2020: Review article: Geothermal heat flow in
# Antarctica: current and future directions
#
#     path = pooch.retrieve(
#         url="https://doi.org/10.5194/tc-14-3843-2020-supplement",
#         known_hash=None,
#         processor=pooch.Unzip(
#             extract_dir='Burton_Johnson_2020',),
#         progressbar=True,)
#     file = [p for p in path if p.endswith('Mean.tif')][0]
#     grd = xr.load_dataarray(file)
#     grd = grd.squeeze()

#     if plot is True:
#         grd.plot(robust=True)
#     if info is True:
#         print(pygmt.grdinfo(grd))
#     return grd
