# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#
from antarctic_plots import utils
from typing import Union
import pandas as pd
import pooch
import pygmt
import xarray as xr
import rioxarray
from pyproj import Transformer
import os


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


def imagery(
    # plot: bool = False,
    # region= None,
) -> xr.DataArray:
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
    image = [p for p in path if p.endswith(".tif")][0]

    # if region is not None:
    #     grd = pygmt.grdcut(
    #         grid=image,
    #         region=region,  
    #         verbose='q')
        
    # grd = rioxarray.open_rasterio(image)

    # if region is not None:
    #     grd = grd.rio.clip_box(
    #         minx=region[0],
    #         maxx=region[1],
    #         miny=region[2],
    #         maxy=region[3],
    #         )

    # spacing = kwargs.get('spacing', None)
    
    # region = utils.get_grid_info(image)[1]
    # spacing = utils.get_grid_info(image)[0]

    # if region is not None and spacing is None:
    #     print('using input region with grdcut')
    #     grd = pygmt.grdcut(
    #         grid=image,
    #         region=region,  
    #         verbose='q')
    # elif spacing is not None and region is None:
    #     print('using input spacing')
    #     grd = pygmt.grdfilter(
    #         grid=image,
    #         filter=f"g{spacing}",
    #         spacing=spacing,
    #         region=utils.get_grid_info(image)[1],
    #         distance="0",
    #         nans="r",
    #         verbose="q",
    #     )
    # elif region and spacing is not None:
    #     print('using input region and spacing')
    #     grd = pygmt.grdfilter(
    #         grid=image,
    #         filter=f"g{spacing}",
    #         spacing=spacing,
    #         region=region,
    #         distance="0",
    #         nans="r",
    #         verbose="q",
    #     )
    # else:
    #     grd = image

    # if plot is True:
    #     grd.plot.imshow()

    return image


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


def bedmachine(
    layer: str,
    reference: str = "geoid",
    plot: bool = False,
    info: bool = False,
    region=None,
    spacing=10e3,
) -> xr.DataArray:
    """
    Load BedMachine data,  from Morlighem et al. 2020:
    https://doi.org/10.1038/s41561-019-0510-8

    orignally from https://nsidc.org/data/nsidc-0756/versions/1.
    Added to Google Bucket as described in the following notebook:
    https://github.com/ldeo-glaciology/pangeo-bedmachine/blob/master/load_plot_bedmachine.ipynb # noqa

    Parameters
    ----------
    layer : str
        choose which layer to fetch:
        'surface', 'thickness', 'bed', 'firn', 'geoid', 'mapping', 'mask', 'errbed', 
        'source'; 'icebase' will give results of surface-thickness
    reference : str
        choose whether heights are referenced to 'geoid' (EIGEN-6C4) or 'ellipsoid' 
        (WGS84), by default is 'geoid'
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
        Returns a loaded, and optional clip/resampled grid of Bedmachine.
    """

    if region is None:
        region = (-2800e3, 2800e3, -2800e3, 2800e3)
    path = pooch.retrieve(
        url="https://storage.googleapis.com/ldeo-glaciology/bedmachine/BedMachineAntarctica_2019-11-05_v01.nc", # noqa
        known_hash=None,
        progressbar=True,
    )

    if layer == "icebase":
        surface = pygmt.grdfilter(
            grid=f"{path}?surface",
            filter=f"g{spacing}",
            spacing=spacing,
            region=region,
            distance="0",
            nans="r",
            verbose="q",
        )
        thickness = pygmt.grdfilter(
            grid=f"{path}?thickness",
            filter=f"g{spacing}",
            spacing=spacing,
            region=region,
            distance="0",
            nans="r",
            verbose="q",
        )
        grd = surface - thickness

    else:
        grd = pygmt.grdfilter(
            grid=f"{path}?{layer}",
            filter=f"g{spacing}",
            spacing=spacing,
            region=region,
            distance="0",
            nans="r",
            verbose="q",
        )

    if reference == "ellipsoid":
        geoid = pygmt.grdfilter(
            grid=f"{path}?geoid",
            filter=f"g{spacing}",
            spacing=spacing,
            region=region,
            distance="0",
            nans="r",
            verbose="q",
        )
        grd = grd + geoid

    if plot is True:
        grd.plot(robust=True)
    if info is True:
        print(pygmt.grdinfo(grd))
    return grd


def bedmap2(
    layer: str,
    reference: str = "geoid",
    plot: bool = False,
    info: bool = False,
    region=None,
    spacing=10e3,
) -> xr.DataArray:
    """
    Load bedmap2 data. All grids are by default referenced to the g104c geoid. Use the 'reference' parameter to convert to the ellipsoid.
    Note, nan's in surface grid are set to 0.
    from https://doi.org/10.5194/tc-7-375-2013.

    Parameters
    ----------
    layer : str
        choose which layer to fetch:
        'surface', 'thickness', 'bed', 'gl04c_geiod_to_WGS84', 'icebase' will give results of 
        surface-thickness
    reference : str
        choose whether heights are referenced to 'geoid' (g104c) or 'ellipsoid' 
        (WGS84), by default is 'geoid'
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
        Returns a loaded, and optional clip/resampled grid of Bedmap2.
    """
    if region is None:
        region = (-2800e3, 2800e3, -2800e3, 2800e3)
    path = pooch.retrieve(
        url="https://secure.antarctica.ac.uk/data/bedmap2/bedmap2_tiff.zip",
        known_hash=None,
        processor=pooch.Unzip(),
        progressbar=True,
    )

    if layer == "icebase":
        surface_file = [p for p in path if p.endswith(f"surface.tif")][0]
        # fill nans with 0 for surface
        surface = pygmt.grdfilter(
            grid=surface_file,
            filter=f"g{spacing}",
            spacing=spacing,
            region=region,
            distance="0",
            nans="r",
            verbose="q",
        )
        thickness_file = [p for p in path if p.endswith(f"thickness.tif")][0]
        thickness = pygmt.grdfilter(
            grid=thickness_file,
            filter=f"g{spacing}",
            spacing=spacing,
            region=region,
            distance="0",
            nans="r",
            verbose="q",
        )
        grd = surface - thickness

    else:
        file = [p for p in path if p.endswith(f"{layer}.tif")][0]
        # grd = xr.load_dataarray(file)
        # grd = rioxarray.open_rasterio(file)
        # grd = grd.squeeze()
        grd = pygmt.grdfilter(
            grid=file,
            filter=f"g{spacing}",
            spacing=spacing,
            region=region,
            distance="0",
            nans="r",
            verbose="q",
        )

    if reference == "ellipsoid" and layer != 'thickness':
        geoid_file = [p for p in path if p.endswith("gl04c_geiod_to_WGS84.tif")][0]
        geoid = pygmt.grdfilter(
            grid=geoid_file,
            filter=f"g{spacing}",
            spacing=spacing,
            region=region,
            distance="0",
            nans="r",
            verbose="q",
        )
        grd = grd + geoid

    if layer == "surface" or "thickness":
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
        region = (-2700e3, 2800e3, -2200e3, 2300e3)
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
    type: str,
    plot: bool = False,
    info: bool = False,
    region=None,
    spacing=10e3,
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
        region = (-2800e3, 2800e3, -2800e3, 2800e3)
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
    version: str,
    plot: bool = False,
    info: bool = False,
    region=None,
    spacing=10e3,
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
        Either 'admap1', 'admap2_eq_src', or 'admap2_gdb'
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
        region = (-2800e3, 2800e3, -2800e3, 2800e3)
    # if version == "admap2":
    #     path = "../data/ADMAP_2B_2017_R9_BAS_.tif"
    #     grd = xr.load_dataarray(path)
    #     grd = grd.squeeze()
    #     grd = pygmt.grdfilter(
    #         grid=grd,
    #         filter=f"g{spacing}",
    #         spacing=spacing,
    #         region=region,
    #         distance="0",
    #         nans="r",
    #         verbose="q",
    #     )
    if version == "admap1":
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
        path = pooch.retrieve(
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


def geothermal(
    version: str,
    plot: bool = False,
    info: bool = False,
    region=None,
    spacing: int = None,
) -> xr.DataArray:
    """
    Load 1 of 3 'versions' of Antarctic geothermal heat flux grids.
    version='burton-johnson-2020'
    From Burton-Johnson et al. 2020: Review article: Geothermal heat flow in Antarctica: 
    current and future directions, https://doi.org/10.5194/tc-14-3843-2020
    Accessed from supplementary material
    version='losing-ebbing-2021'
    From Losing and Ebbing 2021: Predicting Geothermal Heat Flow in Antarctica With a 
    Machine Learning Approach. Journal of Geophysical Research: Solid Earth, 126(6), 
    https://doi.org/10.1029/2020JB021499
    Accessed from https://doi.pangaea.de/10.1594/PANGAEA.930237
    version='aq1'
    From Stal et al. 2021: Antarctic Geothermal Heat Flow Model: Aq1. DOI: 
    https://doi.org/10.1029/2020GC009428 
    Accessed from https://doi.pangaea.de/10.1594/PANGAEA.924857
    verion='shen-2020':
    From Shen et al. 2020; A Geothermal Heat Flux Map of Antarctica Empirically 
    Constrained by Seismic Structure. https://doi.org/ 10.1029/2020GL086955
    Accessed from https://sites.google.com/view/weisen/research-products?authuser=0
    
    Parameters
    ----------
    version : str
        Either 'burton-johnson-2020', 'losing-ebbing-2021', 'aq1', 
    plot : bool, optional
        choose to plot grid, by default False
    info : bool, optional
        choose to print info on grid, by default False
    region : str or np.ndarray, optional
        GMT-format region to clip the loaded grid to, by default doesn't clip
    spacing : int, optional
       grid spacing to resample the loaded grid to, by default spacing is read from downloaded files

    Returns
    -------
    xr.DataArray
         Returns a loaded, and optional clip/resampled grid of GHF data.
    """
    # if region is None:
    #     region = (-3330000, 3330000, -3330000, 3330000)
    if version == "burton-johnson-2020":
        path = pooch.retrieve(
            url="https://doi.org/10.5194/tc-14-3843-2020-supplement",
            known_hash=None,
            processor=pooch.Unzip(
                extract_dir='Burton_Johnson_2020',),
            progressbar=True,)
        try:
            os.rename(
                'C:\\Users\\matthewt\\AppData\\Local\\pooch\\pooch\\Cache\\Burton_Johnson_2020\\Geophysical GHF Summary Statistics Maps',
                'C:\\Users\\matthewt\\AppData\\Local\\pooch\\pooch\\Cache\\Burton_Johnson_2020\\Geophysical_GHF_Summary_Statistics_Maps'
            )
        except:
            pass
        file = [p for p in path if p.endswith('Mean.tif')][0]
        if region is None:
            region = utils.get_grid_info(file)[1]
        if spacing is None:
            spacing = utils.get_grid_info(file)[0]

        grd = pygmt.grdfilter(
            grid=file,
            filter=f"g{spacing}",
            spacing=spacing,
            region=region,
            distance="0",
            nans="r",
            verbose="q",
        )

    elif version == "losing-ebbing-2021":
        path = pooch.retrieve(
            url="https://download.pangaea.de/dataset/930237/files/HF_Min_Max_MaxAbs-1.csv",
            known_hash=None,
            progressbar=True,)
        df = pd.read_csv(path)
        transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
        df["x"], df["y"] = transformer.transform(df.Lat.tolist(), df.Lon.tolist())
       
        if region is None:
            region = (-2800e3, 2800e3, -2800e3, 2800e3)
        if spacing is None:
            spacing = 20e3

        df = pygmt.blockmedian(
            df[["x", "y", "HF [mW/m2]"]], spacing=spacing, region=region, verbose="q"
        )
        grd = pygmt.surface(
            data=df[["x", "y", "HF [mW/m2]"]],
            spacing=spacing,
            region=region,
            M="2c",
            verbose="q",
        )
    elif version == "aq1":
        path = pooch.retrieve(
            url="https://download.pangaea.de/dataset/924857/files/aq1_01_20.nc",
            known_hash=None,
            progressbar=True,)
        file = xr.load_dataset(path)['Q']
        if region is None:
            region = utils.get_grid_info(file)[1]
        if spacing is None:
            spacing = utils.get_grid_info(file)[0]

        grd = pygmt.grdfilter(
            grid=file,
            filter=f"g{spacing}",
            spacing=spacing,
            region=region,
            distance="0",
            nans="r",
            verbose="q",
        )
        grd=grd*1000
      
    else:
        print("invalid version string")
    if plot is True:
        grd.plot(robust=True)
    if info is True:
        print(pygmt.grdinfo(grd))
    return grd
