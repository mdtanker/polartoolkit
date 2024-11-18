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
# pylint: disable=too-many-lines
from __future__ import annotations

import glob
import logging
import pathlib
import re
import shutil
import typing
from inspect import getmembers, isfunction
from pathlib import Path

import deprecation
import earthaccess

if typing.TYPE_CHECKING:
    import geopandas as gpd

import harmonica as hm
import pandas as pd
import pooch
import pygmt
import pyogrio
import requests
import xarray as xr
import zarr
from dotenv import load_dotenv
from pyproj import Transformer

import polartoolkit
from polartoolkit import (  # pylint: disable=import-self
    fetch,  # noqa: PLW0406
    regions,
    utils,
)

load_dotenv()


def get_fetches() -> list[str]:
    """
    get all the fetch functions defined in this module.

    Returns
    -------
    list[str, tuple[float, float, float, float] ]
        names of each fetch function
    """

    fetch_functions = [i[0] for i in getmembers(fetch, isfunction)]

    remove = [
        "load_dotenv",
        "isfunction",
        "getmembers",
        "resample_grid",
        "sample_shp",
        "get_fetches",
    ]

    return [x for x in fetch_functions if x not in remove]


def resample_grid(
    grid: str | xr.DataArray,
    initial_spacing: float | None = None,
    initial_region: tuple[float, float, float, float] | None = None,
    initial_registration: str | None = None,
    spacing: float | None = None,
    region: tuple[float, float, float, float] | None = None,
    registration: str | None = None,
    **kwargs: dict[str, str],
) -> str | xr.DataArray:
    """
    Resample a grid to a new spacing, region, and/or registration. Method of resampling
    depends on comparison with initial and supplied values for spacing, region, and
    registration. If initial values not supplied, will try and extract them from the
    grid.

    Parameters
    ----------
    grid : str | xarray.DataArray
        grid to resample
    initial_spacing : float | None, optional
        spacing of input grid, if known, by default None
    initial_region : tuple[float, float, float, float] | None, optional
        region of input grid, if known, in format [xmin, xmax, ymin, ymax] by default
        None
    initial_registration : str | None, optional
        registration of input grid, if known, by default None
    spacing : float | None, optional
        new spacing for grid, by default None
    region : tuple[float, float, float, float] | None, optional
        new region for grid in format [xmin, xmax, ymin, ymax], by default None
    registration : str | None, optional
        new registration for grid, by default None

    Returns
    -------
    str | xarray.DataArray
        grid, either resampled or same as original depending on inputs. If no
        resampling, and supplied grid is a filepath, returns filepath.
    """

    # get coordinate names
    # original_dims = list(grid.sizes.keys())
    verbose = kwargs.get("verbose", "w")

    # if initial values not given, extract from supplied grid
    grd_info = utils.get_grid_info(grid)
    if initial_spacing is None:
        initial_spacing = grd_info[0]
        initial_spacing = typing.cast(float, initial_spacing)
    if initial_region is None:
        initial_region = grd_info[1]
        initial_region = typing.cast(tuple[float, float, float, float], initial_region)
    if initial_registration is None:
        initial_registration = grd_info[4]
        initial_registration = typing.cast(str, initial_registration)

    # if new values not given, set equal to initial values
    if spacing is None:
        spacing = initial_spacing
    if region is None:
        region = initial_region
    if registration is None:
        registration = initial_registration

    # if all specs are same as original, return original
    rules = [
        spacing == initial_spacing,
        region == initial_region,
        registration == initial_registration,
    ]
    if all(rules):
        logging.info("returning original grid")
        resampled = grid

    # if spacing is smaller, return resampled
    elif spacing < initial_spacing:
        logging.warning(
            "Warning, requested spacing (%s) is smaller than the original (%s).",
            spacing,
            initial_spacing,
        )
        cut = pygmt.grdcut(
            grid=grid,
            region=region,
            verbose=verbose,
        )
        resampled = pygmt.grdsample(
            grid=grid,
            region=pygmt.grdinfo(cut, spacing=f"{spacing}r")[2:-1],
            spacing=f"{spacing}+e",
            registration=registration,
            verbose=verbose,
        )

    # if spacing is larger, return filtered / resampled
    elif spacing > initial_spacing:
        logging.info("spacing larger than original, filtering and resampling")
        filtered = pygmt.grdfilter(
            grid=grid,
            filter=f"g{spacing}",
            region=region,
            distance=kwargs.get("distance", "0"),
            # nans=kwargs.get('nans',"r"),
            verbose=verbose,
        )
        resampled = pygmt.grdsample(
            grid=filtered,
            region=pygmt.grdinfo(filtered, spacing=f"{spacing}r")[2:-1],
            spacing=spacing,
            registration=registration,
            verbose=verbose,
        )

    else:
        if verbose == "w":
            logging.info(
                "returning grid with new region and/or registration, same spacing"
            )

        cut = pygmt.grdcut(
            grid=grid,
            region=region,
            extend="",
            verbose=verbose,
        )
        resampled = pygmt.grdsample(
            grid=grid,
            spacing=f"{spacing}+e",
            region=pygmt.grdinfo(cut, spacing=f"{spacing}r")[2:-1],
            registration=registration,
            verbose=verbose,
        )
        resampled = pygmt.grdcut(
            grid=resampled,
            region=region,
            extend="",
            verbose=verbose,
        )

    # # reset coordinate names if changed
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", message="rename '")
    #     resampled = resampled.rename(
    #         {
    #             list(resampled.dims)[0]: original_dims[0],
    #             list(resampled.dims)[1]: original_dims[1],
    #         }
    #     )

    return resampled


class EarthDataDownloader:
    """
    Either pulls login details from pre-set environment variables, or prompts user to
    input username and password. Will persist the entered details within the python
    session.
    """

    def __init__(self) -> None:
        earthaccess.login()

    def __call__(self, url: str, output_file: str, dataset: typing.Any) -> None:
        creds = earthaccess.auth_environ()
        auth = creds.get("EARTHDATA_USERNAME"), creds.get("EARTHDATA_PASSWORD")
        downloader = pooch.HTTPDownloader(auth=auth, progressbar=True)
        login = requests.get(url, timeout=30)
        downloader(login.url, output_file, dataset)


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

    if name == "Disco_deep_transect":
        known_hash = (
            None  # "ffffeef15d7556cd60305e6222852e3b4e09da3b6c628a094c1e99ac6d605303"
        )
    elif name == "Roosevelt_Island":
        known_hash = (
            None  # "f3821b8a4d24dd676f75db4b7f2b532a328de18e0bdcce8cee6a6abb3b3e70f6"
        )
    else:
        msg = "invalid name string"
        raise ValueError(msg)

    path = pooch.retrieve(
        url=f"doi:10.6084/m9.figshare.26039578.v1/{name}.zip",
        path=f"{pooch.os_cache('pooch')}/polartoolkit/shapefiles",
        fname=name,
        processor=pooch.Unzip(),
        known_hash=known_hash,
    )
    val: str = next(p for p in path if p.endswith(".shp"))
    return val


def mass_change(
    version: str | None = None,
    hemisphere: str | None = None,
) -> typing.Any:
    """
    Ice-sheet height and thickness changes from ICESat to ICESat-2 for both Antarctica
    and Greenland from :footcite:t:`smithpervasive2020`.

    Choose a version of the data to download with the format: "ICESHEET_VERSION_TYPE"
    where ICESHEET is "ais" or "gris", for Antarctica or Greenland, which is
    automatically set via the hemisphere variable. VERSION is "dhdt" for total thickness
    change or "dmdt" for corrected for firn-air content. For Antarctica data, TYPE is
    "floating" or "grounded".

    add "_filt" to retrieve a filtered version of the data.

    accessed from https://digital.lib.washington.edu/researchworks/handle/1773/45388

    Units are in m/yr

    Parameters
    ----------
    version : str, optional,
        choose which version to retrieve, by default is "dhdt_grounded" for Antarctica
        and "dhdt" for Greenland.
    hemisphere : str, optional
        choose which hemisphere to retrieve data for, "north" or "south", by default
        None

    Returns
    -------
    xarray.DataArray
        Returns a calculated grid of Antarctic ice mass change in meters/year.

    References
    ----------
    .. footbibliography::
    """

    hemisphere = utils.default_hemisphere(hemisphere)

    # This is the path to the processed (magnitude) grid
    url = (
        "https://digital.lib.washington.edu/researchworks/bitstream/handle/1773/"
        "45388/ICESat1_ICESat2_mass_change_updated_2_2021%20%281%29.zip?sequence"
        "=4&isAllowed=y"
    )

    zip_fname = "ICESat1_ICESat2_mass_change_updated_2_2021.zip"

    if version is None:
        if hemisphere == "south":
            version = "dhdt_grounded"
        elif hemisphere == "north":
            version = "dhdt"

    if hemisphere == "south":
        version = f"ais_{version}"
    elif hemisphere == "north":
        version = f"gris_{version}"

    if "dhdt" in version:  # type: ignore[operator]
        fname = f"dhdt/{version}.tif"
    elif "dmdt" in version:  # type: ignore[operator]
        fname = f"dmdt/{version}.tif"

    path = pooch.retrieve(
        url=url,
        fname=zip_fname,
        path=f"{pooch.os_cache('pooch')}/polartoolkit/mass_change",
        known_hash=None,
        progressbar=True,
        processor=pooch.Unzip(
            extract_dir="Smith_2020",
        ),
    )
    fname1 = next(p for p in path if p.endswith(fname))

    return (
        xr.load_dataarray(
            fname1,
            engine="rasterio",
        )
        .squeeze()
        .drop_vars(["band", "spatial_ref"])
    )


def basal_melt(variable: str = "w_b") -> typing.Any:
    """
    Antarctic ice shelf basal melt rates for 1994-2018 from satellite radar altimetry.
    from :footcite:t:`adusumilliinterannual2020`.

    accessed from http://library.ucsd.edu/dc/object/bb0448974g

    reading files and preprocessing from supplied jupyternotebooks:
    https://github.com/sioglaciology/ice_shelf_change/blob/master/read_melt_rate_file.ipynb

    Units are in m/yr

    Parameters
    ----------
    variable : str
        choose which variable to load, either 'w_b' for basal melt rate, 'w_b_interp',
        for basal melt rate with interpolated values, and 'w_b_uncert' for uncertainty

    Returns
    -------
    xarray.DataArray
        Returns a dataarray of basal melt rate values

    References
    ----------
    .. footbibliography::
    """

    # This is the path to the processed (magnitude) grid
    url = "http://library.ucsd.edu/dc/object/bb0448974g/_3_1.h5/download"

    fname = "ANT_iceshelf_melt_rates_CS2_2010-2018_v0.h5"

    def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Download the .h5 file, save to .zarr and return fname"
        fname1 = Path(fname)

        # Rename to the file to ***.zarr
        fname_processed = fname1.with_suffix(".zarr")

        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            # load .h5 file
            grid = xr.load_dataset(
                fname1,
                engine="netcdf4",
                # engine='h5netcdf',
                # phony_dims='sort',
            )

            # Remove extra dimension
            grid = grid.squeeze()

            # Assign variables as coords
            grid = grid.assign_coords({"easting": grid.x, "northing": grid.y})

            # Swap dimensions with coordinate names
            grid = grid.swap_dims({"phony_dim_1": "easting", "phony_dim_0": "northing"})

            # Drop coordinate variables
            grid = grid.drop_vars(["x", "y"])

            # Save to .zarr file
            compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
            enc = {x: {"compressor": compressor} for x in grid}
            grid.to_zarr(
                fname_processed,
                encoding=enc,
            )

        return str(fname_processed)

    path = pooch.retrieve(
        url=url,
        fname=fname,
        path=f"{pooch.os_cache('pooch')}/polartoolkit/mass_change/Admusilli_2020",
        known_hash="c14f7059876e6808e3869853a91a3a17a776c95862627c4a3d674c12e4477d2a",
        progressbar=True,
        processor=preprocessing,
    )

    return xr.open_zarr(
        path,  # consolidated=False,
    )[variable]


def ice_vel(
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    hemisphere: str | None = None,
    **kwargs: typing.Any,
) -> xr.DataArray:
    """
    MEaSUREs Phase-Based Ice Velocity Maps for Antarctica and Greenland.

    Antarctica: version 1 from :footcite:t:`mouginotcontinent2019` and
    :footcite:t:`mouginotmeasures2019`.

    accessed from https://nsidc.org/data/nsidc-0754/versions/1#anchor-1
    Data part of https://doi.org/10.1029/2019GL083826

    Greenland: version 1 from :footcite:t:`measures2020`

    accessed from https://nsidc.org/data/nsidc-0670/versions/1

    Units are in m/yr

    Requires an EarthData login, see Tutorials/Download Polar datasets for how to
    configure this.

    Parameters
    ----------
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : float, optional,
        grid spacing to resample the loaded grid to, by default is 5km for Antarctica
        (original data is 450m), and 250m for Greenland
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.
    hemisphere : str, optional
        choose which hemisphere to retrieve data for, "north" or "south", by default
        None
    kwargs : typing.Any
        additional keyword arguments to pass to resample_grid

    Returns
    -------
    xarray.DataArray
        Returns a calculated grid of ice velocity in meters/year.

    References
    ----------
    .. footbibliography::
    """

    hemisphere = utils.default_hemisphere(hemisphere)

    if hemisphere == "south":
        if spacing is None:
            spacing = 5e3
        original_spacing = 450

        # preprocessing for full, 450m resolution
        def preprocessing_fullres(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the .nc file, calculate velocity magnitude, save it back"
            fname1 = Path(fname)
            # Rename to the file to ***_preprocessed.nc
            fname_processed = fname1.with_stem(fname1.stem + "_preprocessed_fullres")
            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                msg = (
                    "WARNING; this file is large (~7Gb) and may take some time to "
                    "download!"
                )
                logging.warning(msg)
                msg = (
                    "WARNING; preprocessing this grid in full resolution is very "
                    "computationally demanding, consider choosing a lower resolution "
                    "using the parameter `spacing`."
                )
                logging.warning(msg)
                with xr.open_dataset(fname1) as ds:
                    processed = (ds.VX**2 + ds.VY**2) ** 0.5
                    # Save to disk
                    processed.to_netcdf(fname_processed)
            return str(fname_processed)

        # preprocessing for filtered 5k resolution
        def preprocessing_5k(fname: str, action: str, _pooch2: typing.Any) -> str:
            """
            Load the .nc file, calculate velocity magnitude, resample to 5k, save it
            back
            """

            fname1 = Path(fname)
            # Rename to the file to ***_preprocessed_5k.nc
            fname_processed = fname1.with_stem(fname1.stem + "_preprocessed_5k")
            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                msg = (
                    "WARNING; this file is large (~7Gb) and may take some time to "
                    "download!"
                )
                logging.warning(msg)
                msg = "WARNING; preprocessing this grid may take a long time."
                logging.warning(msg)
                initial_region = (-2800000.0, 2799800.0, -2799800.0, 2800000.0)
                initial_spacing = original_spacing
                initial_registration = "g"
                with xr.open_dataset(fname1) as ds:
                    vx_5k = resample_grid(
                        ds.VX,
                        initial_spacing=initial_spacing,  # pylint: disable=possibly-used-before-assignment
                        initial_region=initial_region,  # pylint: disable=possibly-used-before-assignment
                        initial_registration=initial_registration,  # pylint: disable=possibly-used-before-assignment
                        spacing=5e3,
                        region=initial_region,
                        registration=initial_registration,
                        **kwargs,
                    )
                    vx_5k = typing.cast(xr.DataArray, vx_5k)
                    vy_5k = resample_grid(
                        ds.VY,
                        initial_spacing=initial_spacing,
                        initial_region=initial_region,
                        initial_registration=initial_registration,
                        spacing=5e3,
                        region=initial_region,
                        registration=initial_registration,
                        **kwargs,
                    )
                    vy_5k = typing.cast(xr.DataArray, vy_5k)

                    processed_lowres = (vx_5k**2 + vy_5k**2) ** 0.5
                    # Save to disk
                    processed_lowres.to_netcdf(fname_processed)
            return str(fname_processed)

        # determine which resolution of preprocessed grid to use
        if spacing < 5000:
            preprocessor = preprocessing_fullres
            initial_region = (-2800000.0, 2799800.0, -2799800.0, 2800000.0)
            initial_spacing = original_spacing
            initial_registration = "g"
        elif spacing >= 5000:
            logging.info("using preprocessed 5km grid since spacing is > 5km")
            preprocessor = preprocessing_5k
            initial_region = (-2800000.0, 2795000.0, -2795000.0, 2800000.0)
            initial_spacing = 5000
            initial_registration = "g"

        if region is None:
            region = initial_region  # pylint: disable=possibly-used-before-assignment
        if registration is None:
            registration = initial_registration  # pylint: disable=possibly-used-before-assignment

        # This is the path to the processed (magnitude) grid
        path = pooch.retrieve(
            url="https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0754.001/1996.01.01/antarctic_ice_vel_phase_map_v01.nc",
            fname="measures_ice_vel_phase_map.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/ice_velocity",
            downloader=EarthDataDownloader(),
            known_hash="fa0957618b8bd98099f4a419d7dc0e3a2c562d89e9791b4d0ed55e6017f52416",
            progressbar=True,
            processor=preprocessor,  # pylint: disable=possibly-used-before-assignment
        )

        with xr.open_dataarray(path) as grid:
            resampled = resample_grid(
                grid,
                initial_spacing=initial_spacing,  # pylint: disable=possibly-used-before-assignment
                initial_region=initial_region,
                initial_registration=initial_registration,
                spacing=spacing,
                region=region,
                registration=registration,
                **kwargs,
            )

    elif hemisphere == "north":
        if spacing is None:
            spacing = 250

        initial_region = (-645000.0, 859750.0, -3370000.0, -640250.0)
        initial_spacing = 250
        initial_registration = "g"

        base_fname = "greenland_vel_mosaic250"
        registry = {
            f"{base_fname}_vx_v1.tif": None,
            f"{base_fname}_vy_v1.tif": None,
        }
        base_url = "https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0670.001/1995.12.01/"
        path = f"{pooch.os_cache('pooch')}/polartoolkit/ice_velocity"

        pup = pooch.create(
            path=path,
            base_url=base_url,
            registry=registry,
        )
        for k, _ in registry.items():
            pup.fetch(
                fname=k,
                downloader=EarthDataDownloader(),
                progressbar=True,
            )

        # pick the requested files
        fname_x = glob.glob(f"{path}/*vx_v1.tif")[0]  # noqa: PTH207
        fname_y = glob.glob(f"{path}/*vy_v1.tif")[0]  # noqa: PTH207

        # print(fname_x)
        # print(fname_y)

        # fname_processed = f"{fname_x[0:-10]}.zarr"
        # print(fname_processed)

        # load and merge data into dataset
        grid_x = (
            xr.load_dataarray(
                fname_x,
                engine="rasterio",
            )
            .squeeze()
            .drop_vars(["band", "spatial_ref"])
        ).rename("VX")
        grid_y = (
            xr.load_dataarray(
                fname_y,
                engine="rasterio",
            )
            .squeeze()
            .drop_vars(["band", "spatial_ref"])
        ).rename("VY")
        grid = xr.merge([grid_x, grid_y])

        processed = (grid.VX**2 + grid.VY**2) ** 0.5

        resampled = resample_grid(
            processed,
            initial_spacing=initial_spacing,  # pylint: disable=possibly-used-before-assignment
            initial_region=initial_region,
            initial_registration=initial_registration,
            spacing=spacing,
            region=region,
            registration=registration,
            **kwargs,
        )

    return typing.cast(xr.DataArray, resampled)  # pylint: disable=possibly-used-before-assignment


def modis(
    version: str | None = None,
    hemisphere: str | None = None,
) -> str:
    """
    Load the MODIS Mosaic of Antarctica (MoA) or Greenland (MoG) imagery.

    Antarctica:
    from :footcite:t:`haranmodis2021` and :footcite:t:`scambosmodisbased2007`.

    accessed from https://nsidc.org/data/nsidc-0593/versions/2

    Greenland:
    from :footcite:t:`haranmeasures2018`
    accessed from https://nsidc.org/data/nsidc-0547/versions/2

    Requires an EarthData login, see Tutorials/Download Polar datasets for how to
    configure this.

    Parameters
    ----------
    version : str, optional
        for Antarctica, choose between "750m" or "125m" resolutions, by default "750m",
        for Greenland, choose between "500m" or "100m" resolutions, by default "500m"
    hemisphere : str, optional
        choose which hemisphere to retrieve data for, "north" or "south", by default
        None

    Returns
    -------
    str
       filepath for MODIS Imagery

    References
    ----------
    .. footbibliography::
    """

    hemisphere = utils.default_hemisphere(hemisphere)

    if version is None:
        if hemisphere == "south":
            version = "750m"
        elif hemisphere == "north":
            version = "500m"

    if hemisphere == "south":
        if version == "125m":
            url = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0593_moa2009_v02/geotiff/moa125_2009_hp1_v02.0.tif.gz"
            fname = "moa125.tif.gz"
            name = "moa125_2009_hp1_v02.0.tif"
            known_hash = (
                "101fa22295f94f6eab487d208c051cf81c9af925355b124a04e3d96463af5b72"
            )
        elif version == "750m":
            url = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0593_moa2009_v02/geotiff/moa750_2009_hp1_v02.0.tif.gz"
            fname = "moa750.tif.gz"
            name = "moa750_2009_hp1_v02.0.tif"
            known_hash = (
                "90d1718ea0971795ec102482c47f308ba08ba2b88383facb9fe210877e80282c"
            )
        else:
            msg = "invalid version string for southern hemisphere"
            raise ValueError(msg)
        path: str = pooch.retrieve(
            url=url,
            fname=fname,
            path=f"{pooch.os_cache('pooch')}/polartoolkit/imagery",
            downloader=EarthDataDownloader(),
            processor=pooch.Decompress(method="gzip", name=name),
            known_hash=known_hash,
            progressbar=True,
        )

    if hemisphere == "north":
        if version == "100m":
            url = "https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0547.002/2015.03.12/mog100_2015_hp1_v02.tif"
            fname = "mog100.tif"
            known_hash = (
                "673745b96b08bf7118c47ad458f7999fb715b8260328d1112c9faf062c4664e9"
            )
        elif version == "500m":
            url = "https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0547.002/2015.03.12/mog500_2015_hp1_v02.tif"
            fname = "mog500.tif"
            known_hash = (
                "5a5d3f5771e72750db69eeb1ddc2860101933ca45a5d5e0f43e54e1f86aae14b"
            )
        else:
            msg = "invalid version string for northern hemisphere"
            raise ValueError(msg)
        path = pooch.retrieve(
            url=url,
            fname=fname,
            path=f"{pooch.os_cache('pooch')}/polartoolkit/imagery",
            downloader=EarthDataDownloader(),
            known_hash=known_hash,
            progressbar=True,
        )
    return path  # pylint: disable=possibly-used-before-assignment


@deprecation.deprecated(
    deprecated_in="0.4.0",
    removed_in="0.8.0",
    current_version=polartoolkit.__version__,
    details="Use the new function modis(hemisphere='south') instead",
)
def modis_moa(version: str = "750m") -> str:
    """deprecated function, use modis(hemisphere="south") instead"""
    return modis(version=version, hemisphere="south")


@deprecation.deprecated(
    deprecated_in="0.4.0",
    removed_in="0.8.0",
    current_version=polartoolkit.__version__,
    details="Use the new function modis(hemisphere='north') instead",
)
def modis_mog(version: str = "500m") -> str:
    """deprecated function, use modis(hemisphere="north") instead"""
    return modis(version=version, hemisphere="north")


def imagery() -> str:
    """
    Load the file path of Antarctic imagery geotiff from LIMA from
    :footcite:t:`bindschadlerlandsat2008`. accessed from https://lima.usgs.gov/

    will replace with below once figured out login issue with pooch
    MODIS Mosaic of Antarctica: https://doi.org/10.5067/68TBT0CGJSOJ
    Assessed from https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0730_MEASURES_MOA2014_v01/geotiff/

    Returns
    -------
    str
        file path

    References
    ----------
    .. footbibliography::
    """

    path = pooch.retrieve(
        url="https://lima.usgs.gov/tiff_90pct.zip",
        fname="lima.zip",
        path=f"{pooch.os_cache('pooch')}/polartoolkit/imagery",
        processor=pooch.Unzip(),
        known_hash="7e7daa7af128f1ad18ac597d95d716ba26f745e75f8abb81c10049419a070c37",
        progressbar=True,
    )
    return typing.cast(str, next(p for p in path if p.endswith(".tif")))


def geomap(
    version: str = "faults",
    region: tuple[float, float, float, float] | None = None,
) -> gpd.GeodataFrame:
    """
    Data from GeoMAP accessed from
    https://doi.pangaea.de/10.1594/PANGAEA.951482?format=html#download

    from :footcite:t:`coxcontinentwide2023` and :footcite:t:`coxgeomap2023`.

    Parameters
    ----------
    version : str, optional
        choose which version to retrieve, "faults", "units", "sources", or "quality",
        by default "faults"
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip

    Returns
    -------
    geopandas.GeoDataFrame
        Returns a geodataframe

    References
    ----------
    .. footbibliography::
    """

    fname = "ATA_SCAR_GeoMAP_v2022_08_QGIS.zip"
    url = "https://download.pangaea.de/dataset/951482/files/ATA_SCAR_GeoMAP_v2022_08_QGIS.zip"

    path = pooch.retrieve(
        url=url,
        fname=fname,
        path=f"{pooch.os_cache('pooch')}/polartoolkit/shapefiles/geomap",
        known_hash="0dd1ec3f276d7aec1bddface7280ae3c10a40d8dea1efd9271803284a1120f84",
        processor=pooch.Unzip(extract_dir="geomap"),
        progressbar=True,
    )
    fname = "ATA_SCAR_GeoMAP_Geology_v2022_08.gpkg"

    fname1 = next(p for p in path if p.endswith(fname))
    fname2 = Path(fname1)

    # found layer names with: fiona.listlayers(fname)

    if version == "faults":
        layer = "ATA_GeoMAP_faults_v2022_08"
    elif version == "units":
        layer = "ATA_GeoMAP_geological_units_v2022_08"
        qml_fname = "ATA geological units - Simple geology.qml"
        qml = next(p for p in path if p.endswith(qml_fname))
        with Path(qml).open(encoding="utf8") as f:
            contents = f.read().replace("\n", "")
        symbol = re.findall(r'<rule symbol="(.*?)"', contents)
        simpcode = re.findall(r'filter="SIMPCODE = (.*?)"', contents)
        simple_geol = pd.DataFrame(
            {
                "SIMPsymbol": symbol,
                "SIMPCODE": simpcode,
            }
        )

        symbol_infos = re.findall(r"<symbol name=(.*?)</layer>", contents)

        symbol_names = []
        symbol_colors = []
        for i in symbol_infos:
            symbol_names.append(re.findall(r'"(.*?)"', i)[0])
            color = re.findall(r'/>          <prop v="(.*?),255" k="color"', i)[0]
            symbol_colors.append(str(color))

        assert len(symbol) == len(simpcode) == len(symbol_names) == len(symbol_colors)

        colors = pd.DataFrame(
            {
                "SIMPsymbol": symbol_names,
                "SIMPcolor": symbol_colors,
            },
        )
        unit_symbols = simple_geol.merge(colors)
        unit_symbols["SIMPCODE"] = unit_symbols.SIMPCODE.astype(int)
        unit_symbols["SIMPcolor"] = unit_symbols.SIMPcolor.str.replace(",", "/")

    elif version == "sources":
        layer = "ATA_GeoMAP_sources_v2022_08"
    elif version == "quality":
        layer = "ATA_GeoMAP_quality_v2022_08"
    else:
        msg = "invalid version string"
        raise ValueError(msg)

    if region is None:
        data = pyogrio.read_dataframe(fname2, layer=layer)
    else:
        data = pyogrio.read_dataframe(
            fname2,
            bbox=tuple(utils.region_to_bounding_box(region)),
            layer=layer,
        )

    if version == "units":
        data = data.merge(unit_symbols)
        data["SIMPsymbol"] = data.SIMPsymbol.astype(float)
        data = data.sort_values("SIMPsymbol")

    return data


def groundingline(
    version: str = "depoorter-2013",
) -> str:
    """
    Load the file path of two versions of groundingline shapefiles

    version = "depoorter-2013"
    from :footcite:t:`depoorterantarctic2013`.
    Supplement to :footcite:t:`depoortercalving2013`.

    version = "measures-v2"
    from :footcite:t:`mouginotmeasures2017`.
    accessed at https://nsidc.org/data/nsidc-0709/versions/2

    version = "BAS"
    from :footcite:t:`gerrishcoastline2020`.
    accessed at https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=8cecde06-8474-4b58-a9cb-b820fa4c9429

    version = "measures-greenland"
    from :footcite:t:`haranmeasures2018`.

    Some versions require an EarthData login, see Tutorials/Download Polar datasets for
    how to configure this.

    Parameters
    ----------
    version : str, optional
        choose which version to retrieve, by default "depoorter-2013"

    Returns
    -------
    str
        file path

    References
    ----------
    .. footbibliography::
    """

    if version == "depoorter-2013":
        path = pooch.retrieve(
            url="https://doi.pangaea.de/10013/epic.42133.d001",
            fname="groundingline_depoorter_2013.d001",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/shapefiles/depoorter-2013",
            known_hash="e4c5918240e334680aed1329f109527efd8f43b6a277bb8e77b66f84f8c16619",
            processor=pooch.Unzip(),
            progressbar=True,
        )
        fname: str = next(p for p in path if p.endswith(".shp"))

    elif version == "measures-v2":
        registry = {
            "GroundingLine_Antarctica_v02.dbf": None,
            "GroundingLine_Antarctica_v02.prj": None,
            "GroundingLine_Antarctica_v02.shp": None,
            "GroundingLine_Antarctica_v02.shx": None,
            "GroundingLine_Antarctica_v02.xml": None,
        }
        base_url = "https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0709.002/1992.02.07/"
        path = f"{pooch.os_cache('pooch')}/polartoolkit/shapefiles/measures"
        pup = pooch.create(
            path=path,
            base_url=base_url,
            # The registry specifies the files that can be fetched
            registry=registry,
        )

        for k, _ in registry.items():
            pup.fetch(
                fname=k,
                downloader=EarthDataDownloader(),
                progressbar=True,
            )
        # pick the requested file
        fname = glob.glob(f"{path}/GroundingLine*.shp")[0]  # noqa: PTH207

    elif version == "BAS":
        url = "https://ramadda.data.bas.ac.uk/repository/entry/get/Greenland_coast.zip?entryid=synth:8cecde06-8474-4b58-a9cb-b820fa4c9429:L0dyZWVubGFuZF9jb2FzdC56aXA="
        path = pooch.retrieve(
            url=url,
            fname="Greenland_coast.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/shapefiles/greenland",
            known_hash=None,
            processor=pooch.Unzip(),
            progressbar=True,
        )
        fname = next(p for p in path if p.endswith(".shp"))

    elif version == "measures-greenland":
        name = "mog100_geus_coastline_v02"
        # name = "mog100_gimp_iceedge_v02"  # shows islands
        # name = "mog500_geus_coastline_v02" # corrupted
        # name = "mog500_gimp_iceedge_v02" # shows islands
        registry = {
            f"{name}.dbf": None,
            f"{name}.prj": None,
            f"{name}.shp": None,
            f"{name}.shx": None,
            f"{name}.xml": None,
        }
        base_url = "https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0547.002/2005.03.12/"
        path = f"{pooch.os_cache('pooch')}/polartoolkit/shapefiles/measures"
        pup = pooch.create(
            path=path,
            base_url=base_url,
            # The registry specifies the files that can be fetched
            registry=registry,
        )
        for k, _ in registry.items():
            pup.fetch(
                fname=k,
                downloader=EarthDataDownloader(),
                progressbar=True,
            )
        # pick the requested files
        fname = glob.glob(f"{path}/{name}*.shp")[0]  # noqa: PTH207
    else:
        msg = "invalid version string"
        raise ValueError(msg)

    return fname


@deprecation.deprecated(
    deprecated_in="0.4.0",
    removed_in="0.8.0",
    current_version=polartoolkit.__version__,
    details="Use the new function antarctic_boundaries instead",
)
def measures_boundaries(
    version: str | None = None,
) -> str:
    """Deprecated, see the new function antarctic_boundaries instead"""
    return antarctic_boundaries(version)  # type: ignore[arg-type]


def antarctic_boundaries(
    version: str,
) -> str:
    """
    Load various files from the MEaSUREs Antarctic Boundaries for IPY 2007-2009

    from :footcite:t:`mouginotmeasures2017`.
    accessed at https://nsidc.org/data/nsidc-0709/versions/2

    Requires an EarthData login, see Tutorials/Download Polar datasets for how to
    configure this.

    Parameters
    ----------
    version : str,
        choose which file to retrieve from the following list:
        "Coastline", "Basins_Antarctica", "Basins_IMBIE", "IceBoundaries", "IceShelf",
        "Mask"

    Returns
    -------
    str
        file path

    References
    ----------
    .. footbibliography::
    """

    # path to store the downloaded files
    path = f"{pooch.os_cache('pooch')}/polartoolkit/shapefiles/measures"

    # coastline shapefile is in a different directory
    if version == "Coastline":
        base_url = "https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0709.002/2008.01.01/"
        registry = {
            "Coastline_Antarctica_v02.dbf": None,
            "Coastline_Antarctica_v02.prj": None,
            "Coastline_Antarctica_v02.shp": None,
            "Coastline_Antarctica_v02.shx": None,
            "Coastline_Antarctica_v02.xml": None,
        }
        pup = pooch.create(
            path=path,
            base_url=base_url,
            # The registry specifies the files that can be fetched
            registry=registry,
        )
        for k, _ in registry.items():
            pup.fetch(
                fname=k,
                downloader=EarthDataDownloader(),
                progressbar=True,
            )
        # pick the requested file
        fname = glob.glob(f"{path}/{version}*.shp")[0]  # noqa: PTH207
    elif version in [
        "Basins_Antarctica",
        "Basins_IMBIE",
        "IceBoundaries",
        "IceShelf",
        "Mask",
    ]:
        base_url = "https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0709.002/1992.02.07/"
        registry = {
            "Basins_Antarctica_v02.dbf": None,
            "Basins_Antarctica_v02.prj": None,
            "Basins_Antarctica_v02.shp": None,
            "Basins_Antarctica_v02.shx": None,
            "Basins_Antarctica_v02.xml": None,
            "Basins_IMBIE_Antarctica_v02.dbf": None,
            "Basins_IMBIE_Antarctica_v02.prj": None,
            "Basins_IMBIE_Antarctica_v02.shp": None,
            "Basins_IMBIE_Antarctica_v02.shx": None,
            "Basins_IMBIE_Antarctica_v02.xml": None,
            "IceBoundaries_Antarctica_v02.dbf": None,
            "IceBoundaries_Antarctica_v02.prj": None,
            "IceBoundaries_Antarctica_v02.shp": None,
            "IceBoundaries_Antarctica_v02.shx": None,
            "IceBoundaries_Antarctica_v02.xml": None,
            "IceShelf_Antarctica_v02.dbf": None,
            "IceShelf_Antarctica_v02.prj": None,
            "IceShelf_Antarctica_v02.shp": None,
            "IceShelf_Antarctica_v02.shx": None,
            "IceShelf_Antarctica_v02.xml": None,
            "Mask_Antarctica_v02.bmp": None,
            "Mask_Antarctica_v02.tif": None,
            "Mask_Antarctica_v02.xml": None,
        }
        pup = pooch.create(
            path=path,
            base_url=base_url,
            # The registry specifies the files that can be fetched
            registry=registry,
        )
        for k, _ in registry.items():
            pup.fetch(
                fname=k,
                downloader=EarthDataDownloader(),
                progressbar=True,
            )
        # pick the requested file
        if version == "Mask":
            fname = glob.glob(f"{path}/{version}*.tif")[0]  # noqa: PTH207
        else:
            fname = glob.glob(f"{path}/{version}*.shp")[0]  # noqa: PTH207
    else:
        msg = "invalid version string"
        raise ValueError(msg)

    return fname


def sediment_thickness(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
) -> xr.DataArray:
    """
    Load 1 of 4 'versions' of sediment thickness data.

    version='ANTASed'
    From :footcite:t:`baranovantased2021`.
    Accessed from https://www.itpz-ran.ru/en/activity/current-projects/antased-a-new-sediment-model-for-antarctica/

    version='tankersley-2022'
    From :footcite:t:`tankersleybasement2022`, :footcite:t:`tankersleybasement2022a`.

    version='lindeque-2016'
    From :footcite:t:`lindequepreglacial2016a` and :footcite:t:`lindequepreglacial2016`.

    version='GlobSed'
    From  :footcite:t:`straumeglobsed2019`.
    Accessed from https://ngdc.noaa.gov/mgg/sedthick/

    Parameters
    ----------
    version : str,
        choose which version of data to fetch.
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 10e3
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of sediment thickness.

    References
    ----------
    .. footbibliography::
    """

    if version == "ANTASed":
        # found with df.describe()
        initial_region = (-2350000.0, 2490000.0, -1990000.0, 2090000.0)
        initial_spacing = 10e3
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, grid the .dat file, and save it back as a .nc"
            path = pooch.Unzip(
                extract_dir="Baranov_2021_sediment_thickness",
            )(fname, action, _pooch2)
            fname1 = next(p for p in path if p.endswith(".dat"))
            fname2 = Path(fname1)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname2.with_stem(fname2.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".nc")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load data
                df = pd.read_csv(
                    fname2,
                    header=None,
                    sep=r"\s+",
                    names=["x_100km", "y_100km", "thick_km"],
                )
                # change units to meters
                df["x"] = df.x_100km * 100000
                df["y"] = df.y_100km * 100000
                df["thick"] = df.thick_km * 1000

                # block-median and grid the data
                df = pygmt.blockmedian(
                    df[["x", "y", "thick"]],
                    spacing=initial_spacing,
                    region=initial_region,
                    registration=initial_registration,
                )
                processed = pygmt.xyz2grd(
                    data=df[["x", "y", "thick"]],
                    region=initial_region,
                    spacing=initial_spacing,
                    registration=initial_registration,
                )
                # Save to disk
                processed.to_netcdf(fname_processed)
            return str(fname_processed)

        path = pooch.retrieve(
            url="https://www.itpz-ran.ru/wp-content/uploads/2021/04/0.1_lim_b.dat_.zip",
            fname="ANTASed.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/sediment_thickness",
            known_hash="7ca77e5be871b1d2ff8a42166b2f9c4e779604fe2dfed5e70c029a2d03bc866b",
            processor=preprocessing,
            progressbar=True,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing=initial_spacing,
            initial_region=initial_region,
            initial_registration=initial_registration,
            spacing=spacing,
            region=region,
            registration=registration,
        )

    elif version == "tankersley-2022":
        # found with utils.get_grid_info()
        initial_region = (-3330000.0, 1900000.0, -3330000.0, 1850000.0)
        initial_spacing = 5e3
        initial_registration = "p"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        path = pooch.retrieve(
            url="https://download.pangaea.de/dataset/941238/files/Ross_Embayment_sediment.nc",
            fname="tankersley_2022_sediment_thickness.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/sediment_thickness",
            known_hash="0a39e57875780e6f499b570f738a464588059cb195cb709785f2436934c1c4e7",
            progressbar=True,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing=initial_spacing,
            initial_region=initial_region,
            initial_registration=initial_registration,
            spacing=spacing,
            region=region,
            registration=registration,
        )

    elif version == "lindeque-2016":
        # found with utils.get_grid_info()
        initial_region = (-4600000.0, 1900000.0, -3900000.0, 1850000.0)
        initial_spacing = 5e3
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        path = pooch.retrieve(
            url="https://store.pangaea.de/Publications/WobbeF_et_al_2016/sedthick_total_v2_5km_epsg3031.nc",
            fname="lindeque_2016_total_sediment_thickness.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/sediment_thickness",
            known_hash="c156a9e9960d965e0599e449a393709f6720af1d1a25c22613c6be7726a213d7",
            progressbar=True,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing=initial_spacing,
            initial_region=initial_region,
            initial_registration=initial_registration,
            spacing=spacing,
            region=region,
            registration=registration,
        )

    elif version == "GlobSed":
        # was in lat long, so just using standard values here
        initial_region = (-3330000, 3330000, -3330000, 3330000)
        initial_spacing = 1e3  # given as 5 arc min (0.08333 degrees), which is
        # ~0.8km at -85deg, or 3km at -70deg
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, reproject the grid, and save it back as a .nc"
            path = pooch.Unzip(
                extract_dir="GlobSed",
            )(fname, action, _pooch2)
            fname1 = next(p for p in path if p.endswith("GlobSed-v3.nc"))
            fname2 = Path(fname1)

            # Rename to the file to ***_preprocessed.nc
            fname_processed = fname2.with_stem(fname2.stem + "_preprocessed")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load data
                grid = xr.load_dataarray(fname2)

                # write the current projection
                grid = grid.rio.write_crs("EPSG:4326")

                # set names of coordinates
                grid = grid.rename({"lon": "x", "lat": "y"})

                # clip to antarctica
                grid = grid.rio.clip_box(
                    *utils.region_to_bounding_box(initial_region),
                    crs="EPSG:3031",
                )

                # reproject to polar stereographic
                reprojected = grid.rio.reproject(
                    "epsg:3031", resolution=initial_spacing
                )

                # need to save to .nc and reload, issues with pygmt
                reprojected.to_netcdf("tmp.nc")
                processed = xr.load_dataset("tmp.nc").z

                # resample and save to disk
                pygmt.grdsample(
                    processed,
                    region=initial_region,
                    spacing=initial_spacing,
                    registration=initial_registration,
                    outgrid=fname_processed,
                )

                # remove tmp file
                pathlib.Path("tmp.nc").unlink()

            return str(fname_processed)

        path = pooch.retrieve(
            url="https://ngdc.noaa.gov/mgg/sedthick/data/version3/GlobSed.zip",
            fname="GlobSed.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/sediment_thickness",
            known_hash="e063ee6603d65c9cee6420cb52a4c6afb520143711b12d618f1a2f591d248bd9",
            processor=preprocessing,
            progressbar=True,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing=initial_spacing,
            initial_region=initial_region,
            initial_registration=initial_registration,
            spacing=spacing,
            region=region,
            registration=registration,
        )

    else:
        msg = "invalid version string"
        raise ValueError(msg)

    return typing.cast(xr.DataArray, resampled)


def ibcso_coverage(
    region: tuple[float, float, float, float],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load IBCSO v2 data, from :footcite:t:`dorschelinternational2022` and
    :footcite:t:`dorschelinternational2022a`.

    Parameters
    ----------
    region : tuple[float, float, float, float]
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        Returns two geodataframes; points and polygons for a subset of IBCSO v2 point
        measurement locations.

    References
    ----------
    .. footbibliography::
    """

    # download / retrieve the geopackage file
    path = pooch.retrieve(
        url="https://download.pangaea.de/dataset/937574/files/IBCSO_v2_coverage.gpkg",
        fname="IBCSO_v2_coverage.gpkg",
        path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
        known_hash="b89e54f26c03b74f0b0f8d826f0f130573eac2c8240de2eb178c8840f0aa99a0",
        progressbar=True,
    )

    # extract the geometries which are within the supplied region
    data = pyogrio.read_dataframe(
        path,
        layer="IBCSO_coverage",
        bbox=tuple(utils.region_to_bounding_box(region)),
    )

    # expand from multipoint/mulitpolygon to point/polygon
    data_coords = data.explode(index_parts=False)

    # extract the single points/polygons within region
    data_subset = data_coords.clip(mask=utils.region_to_bounding_box(region))

    # separate points and polygons
    points = data_subset[data_subset.geometry.type == "Point"]
    polygons = data_subset[data_subset.geometry.type == "Polygon"]

    # this isn't working currently
    # points_3031 = points.to_crs(epsg=3031)
    # polygons_3031 = polygons.to_crs(epsg=3031)

    return (points, polygons)


def ibcso(
    layer: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | int | None = None,
    registration: str | None = None,
) -> xr.DataArray:
    """
    Load IBCSO v2 data, from :footcite:t:`dorschelinternational2022` and
    :footcite:t:`dorschelinternational2022a`.

    Parameters
    ----------
    layer : str
        choose which layer to fetch:
        'surface', 'bed'
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of IBCSO data.

    References
    ----------
    .. footbibliography::
    """

    original_spacing = 500

    # preprocessing for full, 500m resolution
    def preprocessing_fullres(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Load the .nc file, reproject, and save it back"
        fname1 = Path(fname)
        # Rename to the file to ***_preprocessed.nc
        fname_processed = fname1.with_stem(fname1.stem + "_preprocessed_fullres")
        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            # give warning about time
            logging.warning(
                "WARNING; preprocessing for this grid (reprojecting to EPSG:3031) for"
                " the first time can take several minutes!"
            )

            # load grid
            grid = xr.load_dataset(fname1).z
            logging.info(utils.get_grid_info(grid))

            # subset to a smaller region (buffer by 1 cell width)
            cut = pygmt.grdcut(
                grid=grid,
                region=regions.alter_region(
                    regions.antarctica,
                    zoom=-original_spacing,
                ),
            )
            logging.info(utils.get_grid_info(cut))

            # set the projection
            cut = cut.rio.write_crs("EPSG:9354")

            # reproject to EPSG:3031
            reprojected = cut.rio.reproject("epsg:3031")

            # need to save to .nc and reload, issues with pygmt
            reprojected.to_netcdf("tmp.nc")
            processed = xr.load_dataset("tmp.nc").z

            # resample to correct spacing (remove buffer) and region and save to .nc
            pygmt.grdsample(
                grid=processed,
                spacing=original_spacing,
                region=regions.antarctica,
                registration="p",
                outgrid=fname_processed,
            )

            # remove tmp file
            pathlib.Path("tmp.nc").unlink()

        return str(fname_processed)

    # preprocessing for filtered 5k resolution
    def preprocessing_5k(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Load the .nc file, reproject and resample to 5km, and save it back"
        fname1 = Path(fname)
        # Rename to the file to ***_preprocessed.nc
        fname_processed = fname1.with_stem(fname1.stem + "_preprocessed_5k")
        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            # give warning about time
            logging.warning(
                "WARNING; preprocessing for this grid (reprojecting to EPSG:3031) for"
                " the first time can take several minutes!"
            )

            # load grid
            grid = xr.load_dataset(fname1).z
            logging.info(utils.get_grid_info(grid))

            # cut and change spacing, with 1 cell buffer
            cut = resample_grid(
                grid,
                initial_spacing=original_spacing,
                initial_region=(-4800000, 4800000, -4800000, 4800000),
                initial_registration="p",
                spacing=5e3,
                region=regions.alter_region(regions.antarctica, zoom=-5e3),
                registration="p",
            )
            cut = typing.cast(xr.DataArray, cut)

            logging.info(utils.get_grid_info(cut))

            # set the projection
            cut = cut.rio.write_crs("EPSG:9354")

            cut = typing.cast(xr.DataArray, cut)

            # reproject to EPSG:3031
            reprojected = cut.rio.reproject("epsg:3031")

            # need to save to .nc and reload, issues with pygmt
            reprojected.to_netcdf("tmp.nc")
            processed = xr.load_dataset("tmp.nc").z

            # resample to correct spacing (remove buffer) and region and save to .nc
            pygmt.grdsample(
                grid=processed,
                spacing=5e3,
                region=regions.antarctica,
                registration="p",
                outgrid=fname_processed,
            )

            # remove tmp file
            pathlib.Path("tmp.nc").unlink()

        return str(fname_processed)

    if spacing is None:
        spacing = original_spacing

    # determine which resolution of preprocessed grid to use
    if spacing < 5e3:
        preprocessor = preprocessing_fullres
        initial_region = regions.antarctica
        initial_spacing = original_spacing
        initial_registration = "p"
    elif spacing >= 5000:
        logging.info("using preprocessed 5km grid since spacing is > 5km")
        preprocessor = preprocessing_5k
        initial_region = regions.antarctica
        initial_spacing = 5000
        initial_registration = "p"

    if region is None:
        region = initial_region  # pylint: disable=possibly-used-before-assignment
    if registration is None:
        registration = initial_registration  # pylint: disable=possibly-used-before-assignment

    if layer == "surface":
        path = pooch.retrieve(
            url="https://download.pangaea.de/dataset/937574/files/IBCSO_v2_ice-surface.nc",
            fname="IBCSO_ice_surface.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
            known_hash="7748a79fffa41024c175cff7142066940b3e88f710eaf4080193c46b2b59e1f0",
            progressbar=True,
            processor=preprocessor,  # pylint: disable=possibly-used-before-assignment
        )
    elif layer == "bed":
        path = pooch.retrieve(
            url="https://download.pangaea.de/dataset/937574/files/IBCSO_v2_bed.nc",
            fname="IBCSO_bed.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
            known_hash="74d55acb219deb87dc5be019d6dafeceb7b1ebcf9095866f257671d12670a5e2",
            progressbar=True,
            processor=preprocessor,  # pylint: disable=possibly-used-before-assignment
        )
    else:
        msg = "invalid layer string"
        raise ValueError(msg)

    grid = xr.load_dataset(path).z

    resampled = resample_grid(
        grid,
        initial_spacing=initial_spacing,  # pylint: disable=possibly-used-before-assignment
        initial_region=initial_region,
        initial_registration=initial_registration,
        spacing=spacing,
        region=region,
        registration=registration,
    )

    return typing.cast(xr.DataArray, resampled)


def bedmachine(
    layer: str,
    reference: str = "eigen-6c4",
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    hemisphere: str | None = None,
) -> xr.DataArray:
    """
    Load BedMachine topography data from either Greenland (v5) or Antarctica (v3),  from
    :footcite:t:`morlighemmeasures2022` or  :footcite:t:`icebridge2020a`.

    Antarctica:
    Accessed from NSIDC via https://nsidc.org/data/nsidc-0756/versions/3.
    Also available from
    https://github.com/ldeo-glaciology/pangeo-bedmachine/blob/master/load_plot_bedmachine.ipynb

    Greenland:
    Accessed from NSIDC via https://nsidc.org/data/idbmg4/versions/5

    Referenced to the EIGEN-6C4 geoid. To convert to be ellipsoid-referenced, we add
    the geoid grid. use `reference='ellipsoid'` to include this conversion in the
    fetch call.

    For Antarctica: Surface and ice thickness are in ice equivalents. Actual snow
    surface is from REMA :footcite:p:`howatreference2019`, and has had firn thickness
    added(?) to it to get Bedmachine Surface.

    To get snow surface: surface+firn
    To get firn and ice thickness: thickness+firn

    Here, icebase will return a grid of surface-thickness
    This should be the same as snow-surface - (firn and ice thickness)

    Requires an EarthData login, see Tutorials/Download Polar datasets for how to
    configure this.

    Parameters
    ----------
    layer : str
        choose which layer to fetch:
        'bed', 'dataid', 'errbed', 'firn', 'geoid', 'mask', 'source',
        'surface', 'thickness'; 'icebase' will give results of surface-thickness
    reference : str
        choose whether heights are referenced to 'eigen-6c4' geoid or the
        'ellipsoid' (WGS84), by default is eigen-6c4'
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 500m
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.
    hemisphere : str, optional
        choose which hemisphere to retrieve data for, "north" or "south", by default
        None

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of Bedmachine.

    References
    ----------
    .. footbibliography::
    """

    hemisphere = utils.default_hemisphere(hemisphere)

    if hemisphere == "north":
        # found with utils.get_grid_info()
        initial_region = (-653000.0, 879700.0, -3384350.0, -632750.0)
        initial_spacing = 150
        initial_registration = "p"

        url = (
            "https://n5eil01u.ecs.nsidc.org/ICEBRIDGE/IDBMG4.005/1993.01.01/"
            "BedMachineGreenland-v5.nc"
        )
        fname = "bedmachine_v5.nc"
        known_hash = "f7116b8e9e3840649075dcceb796ce98aaeeb5d279d15db489e6e7668e0d80db"

        # greenland dataset doesn't have firn layer
        if layer == "firn":
            msg = "invalid layer string"
            raise ValueError(msg)

    elif hemisphere == "south":
        # found with utils.get_grid_info()
        initial_region = (-3333000.0, 3333000.0, -3333000.0, 3333000.0)
        initial_spacing = 500
        initial_registration = "g"

        url = (
            "https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0756.003/1970.01.01/"
            "BedMachineAntarctica-v3.nc"
        )
        fname = "bedmachine_v3.nc"
        known_hash = "d34390f585e61c4dba0cecd9e275afcc9586b377ba5ccc812e9a004566a9e159"
    else:
        msg = "invalid hemisphere string"
        raise ValueError(msg)

    if region is None:
        region = initial_region
    if spacing is None:
        spacing = initial_spacing
    if registration is None:
        registration = initial_registration

    path = pooch.retrieve(
        url=url,
        fname=fname,
        path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
        downloader=EarthDataDownloader(),
        known_hash=known_hash,
        progressbar=True,
    )

    # calculate icebase as surface-thickness
    if layer == "icebase":
        with xr.open_dataset(path) as ds:
            grid = ds["surface"] - ds["thickness"]
            # utils.get_grid_info(ds["thickness"], print_info=True)
            # restore registration type
            grid.gmt.registration = ds["surface"].gmt.registration

    elif layer in [
        "bed",
        "dataid",
        "errbed",
        "firn",
        "geoid",
        "mask",
        "source",
        "surface",
        "thickness",
    ]:
        with xr.open_dataset(path) as ds:
            grid = ds[layer]

    else:
        msg = "invalid layer string"
        raise ValueError(msg)

    # change layer elevation to be relative to different reference frames.
    if layer in ["surface", "icebase", "bed"]:
        if reference == "ellipsoid":
            logging.info("converting to be reference to the WGS84 ellipsoid")
            with xr.open_dataset(path) as ds:
                geoid_grid = ds["geoid"]
            # save grid registration type
            reg = grid.gmt.registration
            # convert to the ellipsoid
            grid = grid + geoid_grid
            # restore registration type
            if grid.gmt.registration != reg:
                grid.gmt.registration = reg

        elif reference == "eigen-6c4":
            pass
        else:
            msg = "invalid reference string"
            raise ValueError(msg)

    resampled = resample_grid(
        grid,
        initial_spacing=initial_spacing,
        initial_region=initial_region,
        initial_registration=initial_registration,
        spacing=spacing,
        region=region,
        registration=registration,
    )

    return typing.cast(xr.DataArray, resampled)


def bedmap_points(
    version: str,
    region: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """
    Load bedmap point data, choose from Bedmap 1, 2 or 3

    version == 'bedmap1'
    from :footcite:t:`lythebedmap2001`.
    accessed from https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/01619

    version == 'bedmap2'
    from :footcite:t:`fretwellbedmap22013`.
    accessed from https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/01616

    version == 'bedmap3'
    from :footcite:t:`fremandbedmap32022`.
    accessed from https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/01614

    Parameters
    ----------
    version : str
        choose between 'bedmap1', 'bedmap2', or 'bedmap3' point data
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip

    Returns
    -------
    pandas.DataFrame
        Return a dataframe, optionally subset by a region

    References
    ----------
    .. footbibliography::
    """

    if version == "bedmap1":
        url = (
            "https://ramadda.data.bas.ac.uk/repository/entry/get/BEDMAP1_1966-2000_"
            "AIR_BM1.csv?entryid=synth%3Af64815ec-4077-4432-9f55-"
            "0ce230f46029%3AL0JFRE1BUDFfMTk2Ni0yMDAwX0FJUl9CTTEuY3N2"
        )
        fname = pooch.retrieve(
            url=url,
            fname="BEDMAP1_1966-2000_AIR_BM1.csv",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
            known_hash="77d10a0c41ff3401a2a3da1467ba292861a919c6a43a933c91a51d2e1ebe5f6e",
            progressbar=True,
        )

        df = pd.read_csv(
            fname,
            skiprows=18,
            na_values=[-9999],  # set additional nan value
        )

        # drop columns with no entries
        df = df.drop(
            columns=[
                "trace_number",
                "date",
                "time_UTC",
                "two_way_travel_time (m)",
                "aircraft_altitude (m)",
                "along_track_distance (m)",
            ],
        )

        # convert from lat lon to EPSG3031
        df = utils.latlon_to_epsg3031(
            df,
            input_coord_names=("longitude (degree_east)", "latitude (degree_north)"),
        )

    elif version == "bedmap2":
        msg = "fetch bedmap2 point data not implemented yet"
        raise ValueError(msg)
    elif version == "bedmap3":
        msg = "fetch bedmap3 point data not implemented yet"
        raise ValueError(msg)
    else:
        msg = "invalid layer string"
        raise ValueError(msg)

    # get subset inside of region
    if region is not None:
        df = utils.points_inside_region(df, region)

    return df


def bedmap2(
    layer: str,
    reference: str = "eigen-gl04c",
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    fill_nans: bool = False,
) -> xr.DataArray:
    """
    Load bedmap2 data as xarray.DataArrays
    from :footcite:t:`fretwellbedmap22013`.
    accessed from https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=fa5d606c-dc95-47ee-9016-7a82e446f2f2.

    All grids are by default referenced to the EIGEN-GL04C geoid. Use the
    reference='ellipsoid' to convert to the WGS-84 ellipsoid or reference='eigen-6c4' to
    convert to the EIGEN-6c4 geoid.

    Unlike Bedmachine data, Bedmap2 surface and icethickness contain NaN's over the
    ocean, instead of 0's. To fill these NaN's with 0's, set `fill_nans=True`.
    Note, this only makes since if the reference is the geoid, therefore, if
    `reference='ellipsoid` and `fill_nans=True`, the nan's will be filled before
    converting the results to the geoid (just for surface, since thickness isn't
    relative to anything).

    Parameters
    ----------
    layer : str
        choose which layer to fetch:
        "bed", "coverage", "grounded_bed_uncertainty", "icemask_grounded_and_shelves",
        "lakemask_vostok", "rockmask", "surface", "thickness",
        "thickness_uncertainty_5km", "gl04c_geiod_to_WGS84", "icebase",
        "water_thickness"
    reference : str
        choose whether heights are referenced to the 'eigen-6c4' geoid, the WGS84
        ellipsoid, 'ellipsoid', or by default the 'eigen-gl04c' geoid.
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 10e3
    registration : str, optional,
        choose between 'g' (gridline) or 'p' (pixel) registration types, by default is
        the original type of the grid
    fill_nans : bool, optional,
        choose whether to fill nans in 'surface' and 'thickness' with 0. If converting
        to reference to the geoid, will fill nan's before conversion, by default is
        False

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of Bedmap2.

    References
    ----------
    .. footbibliography::
    """

    # download url
    url = (
        "https://ramadda.data.bas.ac.uk/repository/entry/show/Polar+Data+Centre/"
        "DOI/BEDMAP2+-+Ice+thickness%2C+bed+and+surface+elevation+for+Antarctica"
        "+-+gridding+products/bedmap2_tiff?entryid=synth%3Afa5d606c-dc95-47ee-9016"
        "-7a82e446f2f2%3AL2JlZG1hcDJfdGlmZg%3D%3D&output=zip.zipgroup"
    )
    known_hash = None
    # Declare initial grid values, of .nc files not .tiff files
    # use utils.get_grid_info(xr.load_dataset(file).band_data
    # several of the layers have different values
    if layer == "lakemask_vostok":
        initial_region = (1190000.0, 1470000.0, -402000.0, -291000.0)
        initial_spacing = 1e3
        initial_registration = "g"

    elif layer == "thickness_uncertainty_5km":
        initial_region = (-3399000.0, 3401000.0, -3400000.0, 3400000.0)
        initial_spacing = 5e3
        initial_registration = "g"

    elif layer in [
        "bed",
        "coverage",
        "grounded_bed_uncertainty",
        "icemask_grounded_and_shelves",
        "rockmask",
        "surface",
        "thickness",
        "gl04c_geiod_to_WGS84",
        "icebase",
        "water_thickness",
    ]:
        initial_region = (-3333000, 3333000, -3333000, 3333000)
        initial_spacing = 1e3
        initial_registration = "g"

    else:
        msg = "invalid layer string"
        raise ValueError(msg)

    if region is None:
        region = initial_region
    if spacing is None:
        spacing = initial_spacing
    if registration is None:
        registration = initial_registration

    def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Unzip the folder, convert the tiffs to compressed .zarr files"
        # extract each layer to it's own folder
        if layer == "gl04c_geiod_to_WGS84":
            member = ["bedmap2_tiff/gl04c_geiod_to_WGS84.tif"]
        else:
            member = [f"bedmap2_tiff/bedmap2_{layer}.tif"]
        fname1 = pooch.Unzip(
            extract_dir=f"bedmap2_{layer}",
            members=member,
        )(fname, action, _pooch2)[0]
        # get the path to the layer's tif file
        fname2 = Path(fname1)

        # Rename to the file to ***.zarr
        fname_processed = fname2.with_suffix(".zarr")

        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            # load data
            grid = (
                xr.load_dataarray(
                    fname2,
                    engine="rasterio",
                )
                .squeeze()
                .drop_vars(["band", "spatial_ref"])
            )
            grid = grid.to_dataset(name=layer)

            # Save to disk
            compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
            grid.to_zarr(
                fname_processed,
                encoding={layer: {"compressor": compressor}},
            )

        return str(fname_processed)

    # calculate icebase as surface-thickness
    if layer == "icebase":
        # set layer variable so pooch retrieves correct file
        layer = "surface"
        fname = pooch.retrieve(
            url=url,
            fname="bedmap2_tiff.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
            known_hash=known_hash,
            processor=preprocessing,
            progressbar=True,
        )
        # load zarr as a dataarray
        surface = xr.open_zarr(fname)[layer]

        layer = "thickness"
        # set layer variable so pooch retrieves correct file
        fname = pooch.retrieve(
            url=url,
            fname="bedmap2_tiff.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
            known_hash=known_hash,
            processor=preprocessing,
            progressbar=True,
        )
        # load zarr as a dataarray
        thickness = xr.open_zarr(fname)[layer]

        # calculate icebase
        grid = surface - thickness

        # reset layer variable
        layer = "icebase"
        logging.info("calculating icebase from surface and thickness grids")
    elif layer == "water_thickness":
        icebase = bedmap2(layer="icebase")
        bed = bedmap2(layer="bed")

        # calculate water thickness
        grid = icebase - bed
        logging.info("calculating water thickness from bed and icebase grids")
    elif layer in [
        "bed",
        "coverage",
        "grounded_bed_uncertainty",
        "icemask_grounded_and_shelves",
        "lakemask_vostok",
        "rockmask",
        "surface",
        "thickness",
        "thickness_uncertainty_5km",
        "gl04c_geiod_to_WGS84",
    ]:
        # download/unzip all files, retrieve the specified layer file and convert to
        # .zarr
        fname = pooch.retrieve(
            url=url,
            fname="bedmap2_tiff.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
            known_hash=known_hash,
            processor=preprocessing,
            progressbar=True,
        )
        # load zarr as a dataarray
        grid = xr.open_zarr(fname)[layer]

    else:
        msg = "invalid layer string"
        raise ValueError(msg)

    # replace nans with 0's in surface, thickness or icebase grids
    if fill_nans is True and layer in ["surface", "thickness", "icebase"]:
        # pygmt.grdfill(final_grid, mode='c0') # doesn't work, maybe grid is too big
        # this changes the registration from pixel to gridline
        grid = grid.fillna(0)

    # change layer elevation to be relative to different reference frames.
    if layer in ["surface", "icebase", "bed"]:
        if reference == "ellipsoid":
            logging.info("converting to be referenced to the WGS84 ellipsoid")
            # set layer variable so pooch retrieves the geoid conversion file
            layer = "gl04c_geiod_to_WGS84"
            fname = pooch.retrieve(
                url=url,
                fname="bedmap2_tiff.zip",
                path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
                known_hash=known_hash,
                processor=preprocessing,
                progressbar=True,
            )
            # load zarr as a dataarray
            geoid_2_ellipsoid = xr.open_zarr(fname)[layer]

            # convert to the ellipsoid
            grid = grid + geoid_2_ellipsoid
        elif reference == "eigen-6c4":
            logging.info("converting to be referenced to the EIGEN-6C4")
            # set layer variable so pooch retrieves the geoid conversion file
            layer = "gl04c_geiod_to_WGS84"
            fname = pooch.retrieve(
                url=url,
                fname="bedmap2_tiff.zip",
                path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
                known_hash=known_hash,
                processor=preprocessing,
                progressbar=True,
            )
            # load zarr as a dataarray
            geoid_2_ellipsoid = xr.open_zarr(fname)[layer]

            # convert to the ellipsoid
            grid = grid + geoid_2_ellipsoid

            # get a grid of EIGEN geoid values matching the user's input
            eigen_correction = geoid(
                spacing=initial_spacing,
                region=initial_region,
                registration=initial_registration,
                hemisphere="south",
            )
            # convert from ellipsoid back to eigen geoid
            grid = grid - eigen_correction
        elif reference == "eigen-gl04c":
            pass
        else:
            msg = "invalid reference string"
            raise ValueError(msg)

    resampled = resample_grid(
        grid,
        initial_spacing=initial_spacing,
        initial_region=initial_region,
        initial_registration=initial_registration,
        spacing=spacing,
        region=region,
        registration=registration,
    )

    return typing.cast(xr.DataArray, resampled)


def rema(
    version: str = "1km",
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
) -> xr.DataArray:
    """
    Load the REMA surface elevation data from :footcite:t:`howatreference2019`. The data
    are in EPSG3031 and reference to the WGS84 ellipsoid. To convert the data to be
    geoid-referenced, subtract a geoid model, which you can get from fetch.geoid().

    Choose between "1km" or "500m" resolutions with parameter `version`.

    accessed from https://www.pgc.umn.edu/data/rema/

    Parameters
    ----------
    version : str, optional,
        choose which resolution to fetch, either "1km" or "500m", by default is "1km"
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 10e3
    registration : str, optional,
        choose between 'g' (gridline) or 'p' (pixel) registration types, by default is
        the original type of the grid

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of the REMA DEM.

    References
    ----------
    .. footbibliography::
    """

    if version == "500m":
        # found with utils.get_grid_info(grid)
        initial_region = (-2700250.0, 2750250.0, -2500250.0, 3342250.0)
        initial_spacing = 500
        initial_registration = "g"
        # url and file name for download
        url = (
            "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/500m/rema_mosaic_"
            "500m_v2.0_filled_cop30.tar.gz"
        )
        fname = "rema_mosaic_500m_v2.0_filled_cop30.tar.gz"
        members = ["rema_mosaic_500m_v2.0_filled_cop30_dem.tif"]
        known_hash = "dd59df24d1ee570654d79afc099c260aeb4128f67232f8c7258a8a7803ef3e0c"
    elif version == "1km":
        # found with utils.get_grid_info(grid)
        initial_region = (-2700500.0, 2750500.0, -2500500.0, 3342500.0)
        initial_spacing = 1000
        initial_registration = "g"
        # url and file name for download
        url = (
            "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/1km/rema_mosaic_"
            "1km_v2.0_filled_cop30.tar.gz"
        )
        fname = "rema_mosaic_1km_v2.0_filled_cop30.tar.gz"
        members = ["rema_mosaic_1km_v2.0_filled_cop30_dem.tif"]
        known_hash = "143ab56b79a0fdcae6769a895202af117fb0dbfe1fa2a0a17db9df2091338d21"
    else:
        msg = "invalid version"
        raise ValueError(msg)

    if region is None:
        region = initial_region
    if spacing is None:
        spacing = initial_spacing
    if registration is None:
        registration = initial_registration

    def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Untar the folder, convert the tiffs to compressed .zarr files"
        # extract the files and get the surface grid
        path = pooch.Untar(members=members)(fname, action, _pooch2)[0]
        # fname = [p for p in path if p.endswith("dem.tif")]#[0]
        tiff_file = Path(path)
        # Rename to the file to ***.zarr
        fname_processed = tiff_file.with_suffix(".zarr")

        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            # load data
            with (
                xr.open_dataarray(tiff_file)
                .squeeze()
                .drop_vars(["band", "spatial_ref"]) as grid
            ):
                ds = grid.to_dataset(name="surface")

                # Save to disk
                compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
                ds.to_zarr(
                    fname_processed,
                    encoding={"surface": {"compressor": compressor}},
                )
                ds.close()

        # delete the unzipped file
        # os.remove(fname)

        return str(fname_processed)

    # download/untar file convert to .zarr and return the path
    zarr_file = pooch.retrieve(
        url=url,
        fname=fname,
        path=f"{pooch.os_cache('pooch')}/polartoolkit/topography/REMA",
        known_hash=known_hash,
        progressbar=True,
        processor=preprocessing,
    )

    # load zarr as a dataarray
    grid = xr.open_zarr(zarr_file)["surface"]

    resampled = resample_grid(
        grid,
        initial_spacing=initial_spacing,
        initial_region=initial_region,
        initial_registration=initial_registration,
        spacing=spacing,
        region=region,
        registration=registration,
    )

    return typing.cast(xr.DataArray, resampled)


def deepbedmap(
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
) -> str:
    """
    Load DeepBedMap data,  from :footcite:t:`leongdeepbedmap2020` and
    :footcite:t:`leongdeepbedmap2020a`.

    Parameters
    ----------
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 10e3
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.

    Returns
    -------
    str
        Returns the filepath of DeepBedMap.

    References
    ----------
    .. footbibliography::
    """

    # found with utils.get_grid_info()
    initial_region = (-2700000.0, 2800000.0, -2199750.0, 2299750.0)
    initial_spacing = 250
    initial_registration = "p"

    if region is None:
        region = initial_region
    if spacing is None:
        spacing = initial_spacing
    if registration is None:
        registration = initial_registration

    path: str = pooch.retrieve(
        url="https://zenodo.org/record/4054246/files/deepbedmap_dem.tif?download=1",
        fname="deepbedmap.tif",
        path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
        known_hash="d8bca00bf6e999d6f77490943454d730f519ebd04b8a9511b9913313c7d954b1",
        progressbar=True,
    )

    with xr.open_dataarray(path) as da:
        grid = da.squeeze().drop_vars(["band", "spatial_ref"])

    return resample_grid(
        grid,
        initial_spacing=initial_spacing,
        initial_region=initial_region,
        initial_registration=initial_registration,
        spacing=spacing,
        region=region,
        registration=registration,
    )


def gravity(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
) -> xr.DataArray:
    """
    Loads 1 of 3 'versions' of Antarctic gravity grids.

    version='antgg'
    Antarctic-wide gravity data compilation of ground-based, airborne, and shipborne
    data, from :footcite:t:`scheinertnew2016`.
    Accessed from https://doi.pangaea.de/10.1594/PANGAEA.848168

    version='antgg-update'
    Preliminary compilation of Antarctica gravity and gravity gradient data.
    Updates on 2016 AntGG compilation.
    Accessed from https://ftp.space.dtu.dk/pub/RF/4D-ANTARCTICA/

    version='antgg-2021'
    Updates on 2016 AntGG compilation.
    Accessed from https://doi.pangaea.de/10.1594/PANGAEA.971238?format=html#download

    version='eigen'
    Earth gravity grid (eigen-6c4) at 10 arc-min resolution at 10km geometric
    (ellipsoidal) height from :footcite:t:`forsteeigen6c42014`.
    originally from https://dataservices.gfz-potsdam.de/icgem/showshort.php?id=escidoc:1119897
    Accessed via the Fatiando data repository https://github.com/fatiando-data/earth-gravity-10arcmin

    Parameters
    ----------
    version : str
        choose which version of gravity data to fetch.
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 10e3
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.
    kwargs : typing.Any
        additional kwargs to pass to resample_grid and set the anomaly_type.

    Keyword Args
    ------------
    anomaly_type : str
        either 'FA' or 'BA', for free-air and bouguer anomalies, respectively. For
        antgg-update can also be 'DG' for gravity disturbance, or 'Err' for error
        estimates.

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of either observed, free-air
        or Bouguer gravity anomalies.

    References
    ----------
    .. footbibliography::
    """

    anomaly_type = kwargs.get("anomaly_type")

    if version == "antgg":
        # found with utils.get_grid_info()
        initial_region = (-3330000.0, 3330000.0, -3330000.0, 3330000.0)
        initial_spacing = 10e3
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        path = pooch.retrieve(
            url="https://hs.pangaea.de/Maps/antgg2015/antgg2015.nc",
            fname="antgg.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/gravity",
            known_hash="ad94d16f7e4895c5a09bbf9ca9d64750f1edd9b01f2bff21ca49e10b6fe47d72",
            progressbar=True,
        )

        if anomaly_type == "FA":
            anomaly_type = "free_air_anomaly"
        elif anomaly_type == "BA":
            anomaly_type = "bouguer_anomaly"
        else:
            msg = "invalid anomaly type"
            raise ValueError(msg)

        file = xr.load_dataset(path)[anomaly_type]

        # convert coordinates from km to m
        file_meters = file.copy()
        file_meters["x"] = file.x * 1000
        file_meters["y"] = file.y * 1000

        resampled = resample_grid(
            file_meters,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
            **kwargs,
        )

    elif version == "antgg-update":
        # found in documentation
        initial_region = (-3330000.0, 3330000.0, -3330000.0, 3330000.0)
        initial_spacing = 10e3
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        available_anomalies = ["FA", "DG", "BA", "Err"]
        if anomaly_type not in available_anomalies:
            msg = "anomaly_type must be either 'FA', 'BA', 'Err' or 'DG'"
            raise ValueError(msg)

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, grid the .dat file, and save it back as a .nc"
            path = pooch.Unzip()(fname, action, _pooch2)
            fname1 = next(p for p in path if p.endswith(".dat"))
            fname2 = Path(fname1)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname2.with_stem(fname2.stem + f"_{anomaly_type}_preprocessed")
            fname_processed = fname_pre.with_suffix(".nc")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load data
                df = pd.read_csv(
                    fname2,
                    sep=r"\s+",
                    skiprows=3,
                    names=["id", "lat", "lon", "FA", "Err", "DG", "BA"],
                )
                # re-project to polar stereographic
                transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
                df["x"], df["y"] = transformer.transform(  # pylint: disable=unpacking-non-sequence
                    df.lat.tolist(), df.lon.tolist()
                )

                # block-median and grid the data
                df = pygmt.blockmedian(
                    df[["x", "y", anomaly_type]],
                    spacing=initial_spacing,
                    region=initial_region,
                    registration=initial_registration,
                )
                processed = pygmt.surface(
                    data=df[["x", "y", anomaly_type]],
                    spacing=initial_spacing,
                    region=initial_region,
                    registration=initial_registration,
                    maxradius="1c",
                )
                # Save to disk
                processed.to_netcdf(fname_processed)
            return str(fname_processed)

        path = pooch.retrieve(
            url="https://ftp.space.dtu.dk/pub/RF/4D-ANTARCTICA/ant4d_gravity.zip",
            fname="antgg_update.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/gravity",
            known_hash="1013a8fa610c16198bc3c901039fd535cf939e4f21f99e6434849bb505094974",
            processor=preprocessing,
            progressbar=True,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
            **kwargs,
        )

    elif version == "antgg-2021":
        # found with utils.get_grid_info()
        initial_region = (-3330000.0, 3330000.0, -3330000.0, 3330000.0)
        initial_spacing = 5e3
        initial_registration = "g"

        # if region is None:
        #     region = initial_region
        # if spacing is None:
        #     spacing = initial_spacing
        # if registration is None:
        #     registration = initial_registration

        if anomaly_type == "FA":
            url = "https://download.pangaea.de/dataset/971238/files/AntGG2021_Gravity-anomaly.nc"
            fname = "antgg_2021_FA.nc"
        elif anomaly_type == "DG":
            url = "https://download.pangaea.de/dataset/971238/files/AntGG2021_Gravity_disturbance_at-surface.nc"
            fname = "antgg_2021_DG.nc"
        elif anomaly_type == "BA":
            url = "https://download.pangaea.de/dataset/971238/files/AntGG2021_Bouguer-anomaly.nc"
            fname = "antgg_2021_BA.nc"
        elif anomaly_type == "Err":
            url = "https://download.pangaea.de/dataset/971238/files/AntGG2021_Standard-deviation_GA-from-LSC.nc"
            fname = "antgg_2021_Err.nc"
        else:
            msg = "invalid anomaly type"
            raise ValueError(msg)

        path = pooch.retrieve(
            url=url,
            fname=fname,
            path=f"{pooch.os_cache('pooch')}/polartoolkit/gravity",
            known_hash=None,
            progressbar=True,
        )

        file = xr.load_dataset(path)

        if anomaly_type == "FA":
            file = file.grav_anom
        elif anomaly_type == "DG":
            file = file.grav_dist
        elif anomaly_type == "BA":
            file = file.Boug_anom
        elif anomaly_type == "Err":
            file = file.std_grav_anom

        resampled = resample_grid(
            file,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
            **kwargs,
        )

    elif version == "eigen":
        initial_region = (-3330000.0, 3330000.0, -3330000.0, 3330000.0)
        initial_spacing = 5e3
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the .nc file, reproject, and save it back"
            fname1 = Path(fname)
            # Rename to the file to ***_preprocessed.nc
            fname_processed = fname1.with_stem(fname1.stem + "_preprocessed")
            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load grid
                grid = xr.load_dataset(fname1).gravity

                # reproject to polar stereographic
                grid2 = pygmt.grdproject(
                    grid,
                    projection="EPSG:3031",
                    spacing=initial_spacing,
                )
                # get just antarctica region
                processed = pygmt.grdsample(
                    grid2,
                    region=initial_region,
                    spacing=initial_spacing,
                    registration=initial_registration,
                )
                # Save to disk
                processed.to_netcdf(fname_processed)
            return str(fname_processed)

        path = pooch.retrieve(
            url="doi:10.5281/zenodo.5882207/earth-gravity-10arcmin.nc",
            fname="eigen.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/gravity",
            known_hash="d55134501da0d984f318c0f92e1a15a8472176ec7babde5edfdb58855190273e",
            progressbar=True,
            processor=preprocessing,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing=initial_spacing,
            initial_region=initial_region,
            initial_registration=initial_registration,
            spacing=spacing,
            region=region,
            registration=registration,
            **kwargs,
        )

    else:
        msg = "invalid version string"
        raise ValueError(msg)

    return typing.cast(xr.DataArray, resampled)


def etopo(
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    hemisphere: str | None = None,
) -> xr.DataArray:
    """
    Loads a grid of Antarctic topography from ETOPO1 from :footcite:t:`etopo12009`.
    Originally at 10 arc-min resolution, reference to mean sea-level (geoid).

    originally from https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc.mgg.dem:316
    Accessed via the Fatiando data repository https://github.com/fatiando-data/earth-topography-10arcmin

    Parameters
    ----------

    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 10e3
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.
    hemisphere : str, optional
        choose which hemisphere to retrieve data for, "north" or "south", by default
        None

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of topography.

    References
    ----------
    .. footbibliography::
    """

    hemisphere = utils.default_hemisphere(hemisphere)

    initial_region = (-3500000.0, 3500000.0, -3500000.0, 3500000.0)
    initial_spacing = 5e3
    initial_registration = "g"

    if hemisphere == "south":
        proj = "EPSG:3031"
        fname = "etopo_south.nc"
    elif hemisphere == "north":
        proj = "EPSG:3413"
        fname = "etopo_north.nc"

    if region is None:
        region = initial_region
    if spacing is None:
        spacing = initial_spacing
    if registration is None:
        registration = initial_registration

    def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Load the .nc file, reproject, and save it back"
        fname1 = Path(fname)
        # Rename to the file to ***_preprocessed.nc
        fname_processed = fname1.with_stem(fname1.stem + "_preprocessed")
        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            # load grid
            grid = xr.load_dataset(fname1).topography

            # reproject to polar stereographic
            grid2 = pygmt.grdproject(
                grid,
                projection=proj,  # pylint: disable=possibly-used-before-assignment
                spacing=initial_spacing,
            )
            # get just needed region
            processed = pygmt.grdsample(
                grid2,
                region=initial_region,
                spacing=initial_spacing,
                registration=initial_registration,
            )
            # Save to disk
            processed.to_netcdf(fname_processed)
        return str(fname_processed)

    path = pooch.retrieve(
        url="doi:10.5281/zenodo.5882203/earth-topography-10arcmin.nc",
        fname=fname,  # pylint: disable=possibly-used-before-assignment
        path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
        known_hash="e45628a3f559ec600a4003587a2b575402d22986651ee48806930aa909af4cf6",
        progressbar=True,
        processor=preprocessing,
    )

    grid = xr.load_dataarray(path)

    resampled = resample_grid(
        grid,
        initial_spacing=initial_spacing,
        initial_region=initial_region,
        initial_registration=initial_registration,
        spacing=spacing,
        region=region,
        registration=registration,
    )

    return typing.cast(xr.DataArray, resampled)


def geoid(
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    hemisphere: str | None = None,
) -> xr.DataArray:
    """
    Loads a grid of Antarctic geoid heights derived from the EIGEN-6C4 from
    :footcite:t:`forsteeigen6c42014` spherical harmonic model of Earth's gravity field.
    Originally at 10 arc-min resolution.
    Negative values indicate the geoid is below the ellipsoid surface and vice-versa.
    To convert a topographic grid which is referenced to the ellipsoid to be referenced
    to the geoid, add this grid.
    To convert a topographic grid which is referenced to the geoid to be reference to
    the ellipsoid, subtract this grid.

    originally from https://dataservices.gfz-potsdam.de/icgem/showshort.php?id=escidoc:1119897
    Accessed via the Fatiando data repository https://github.com/fatiando-data/earth-geoid-10arcmin
    DOI: 10.5281/zenodo.5882205

    Parameters
    ----------
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 10e3
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.
    hemisphere : str, optional
        choose which hemisphere to retrieve data for, "north" or "south", by default
        None

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of geoid height.

    References
    ----------
    .. footbibliography::
    """

    hemisphere = utils.default_hemisphere(hemisphere)

    initial_region = (-3500000.0, 3500000.0, -3500000.0, 3500000.0)
    initial_spacing = 5e3
    initial_registration = "g"

    if hemisphere == "south":
        proj = "EPSG:3031"
        fname = "eigen_geoid_south.nc"
    elif hemisphere == "north":
        proj = "EPSG:3413"
        fname = "eigen_geoid_north.nc"

    if region is None:
        region = initial_region
    if spacing is None:
        spacing = initial_spacing
    if registration is None:
        registration = initial_registration

    def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Load the .nc file, reproject, and save it back"
        fname1 = Path(fname)
        # Rename to the file to ***_preprocessed.nc
        fname_processed = fname1.with_stem(fname1.stem + "_preprocessed")
        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            # load grid
            grid = xr.load_dataset(fname1).geoid

            # reproject to polar stereographic
            grid2 = pygmt.grdproject(
                grid,
                projection=proj,  # pylint: disable=possibly-used-before-assignment
                spacing=initial_spacing,
            )
            # get just needed region
            processed = pygmt.grdsample(
                grid2,
                region=initial_region,
                spacing=initial_spacing,
                registration=initial_registration,
            )
            # Save to disk
            processed.to_netcdf(fname_processed)
        return str(fname_processed)

    path = pooch.retrieve(
        url="doi:10.5281/zenodo.5882204/earth-geoid-10arcmin.nc",
        fname=fname,  # pylint: disable=possibly-used-before-assignment
        path=f"{pooch.os_cache('pooch')}/polartoolkit/geoid",
        known_hash="e98dd544c8b4b8e5f11d1a316684dfbc2612e2860af07b946df46ed9f782a0f6",
        progressbar=True,
        processor=preprocessing,
    )

    grid = xr.load_dataarray(path)

    resampled = resample_grid(
        grid,
        initial_spacing=initial_spacing,
        initial_region=initial_region,
        initial_registration=initial_registration,
        spacing=spacing,
        region=region,
        registration=registration,
    )

    return typing.cast(xr.DataArray, resampled)


def magnetics(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
) -> xr.DataArray | None:
    """
    Load 1 of 3 'versions' of Antarctic magnetic anomaly grid.
    from :footcite:t:`golynskynew2018` and :footcite:t:`golynskyadmap2006`.

    version='admap1'
    ADMAP-2001 magnetic anomaly compilation of Antarctica.
    Accessed from http://admap.kopri.re.kr/databases.html

    version='admap2'
    ADMAP2 magnetic anomaly compilation of Antarctica from
    :footcite:t:`golynskyadmap22018a`.
    Accessed from https://doi.pangaea.de/10.1594/PANGAEA.892723?format=html#download

    version='admap2_gdb'
    Geosoft-specific .gdb abridged files :footcite:t:`golynskyadmap22018`.
    Accessed from https://doi.pangaea.de/10.1594/PANGAEA.892722?format=html#download

    Parameters
    ----------
    version : str
        Either 'admap1', 'admap2', or 'admap2_gdb'
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 10e3
    registration : str, optional,
        choose between 'g' (gridline) or 'p' (pixel) registration types, by default is
        the original type of the grid
    kwargs : typing.Any
        key word arguments to pass to resample_grid.

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of magnetic anomalies.

    References
    ----------
    .. footbibliography::
    """

    if version == "admap1":
        # was in lat long, so just using standard values here
        initial_region = (-3330000.0, 3330000.0, -3330000.0, 3330000.0)
        initial_spacing = 5e3
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, grid the .dat file, and save it back as a .nc"
            path = pooch.Unzip(
                # extract_dir="Baranov_2021_sediment_thickness",
            )(fname, action, _pooch2)
            fname1 = next(p for p in path if p.endswith(".dat"))
            fname2 = Path(fname1)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname2.with_stem(fname2.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".nc")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                logging.info("unzipping %s", fname)
                # load data
                df = pd.read_csv(
                    fname1,
                    sep=r"\s+",
                    header=None,
                    names=["lat", "lon", "nT"],
                )

                # re-project to polar stereographic
                transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
                df["x"], df["y"] = transformer.transform(  # pylint: disable=unpacking-non-sequence
                    df.lat.tolist(), df.lon.tolist()
                )

                # block-median and grid the data
                df = pygmt.blockmedian(
                    df[["x", "y", "nT"]],
                    spacing=initial_spacing,
                    region=initial_region,
                    registration=initial_registration,
                )
                processed = pygmt.surface(
                    data=df[["x", "y", "nT"]],
                    spacing=initial_spacing,
                    region=initial_region,
                    registration=initial_registration,
                    maxradius="1c",
                )
                # Save to disk
                processed.to_netcdf(fname_processed)

                logging.info(".dat file gridded and saved as %s", fname_processed)

            return str(fname_processed)

        path = pooch.retrieve(
            url="http://admap.kopri.re.kr/admapdata/ant_new.zip",
            fname="admap1.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/magnetics",
            known_hash="3fe567c4dfe75be9d5a7772623c72cd27fa386b09a82f03bdb0153a7d64e8524",
            processor=preprocessing,
            progressbar=True,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
            **kwargs,
        )

    elif version == "admap2":
        initial_region = (-3423000.0, 3426000.0, -3424500.0, 3426000.0)
        initial_spacing = 1500
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "convert geosoft grd to xarray dataarray and save it back as a .nc"
            fname1 = Path(fname)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname1.with_stem(fname1.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".nc")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # convert to dataarray
                processed = hm.load_oasis_montaj_grid(fname1)
                # Save to disk
                processed.to_netcdf(fname_processed)
            return str(fname_processed)

        url = "https://hs.pangaea.de/mag/airborne/Antarctica/grid/ADMAP_2B_2017.grd"
        fname = "ADMAP_2B_2017.grd"
        path = pooch.retrieve(
            url=url,
            fname=fname,
            path=f"{pooch.os_cache('pooch')}/polartoolkit/magnetics",
            known_hash="87db037e0b8c134ec4198f261d85c75c2bd5d144d8358ca37759cf8b87ae8c40",
            progressbar=True,
            processor=preprocessing,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
            **kwargs,
        )

    elif version == "admap2_gdb":
        path = pooch.retrieve(
            url="https://hs.pangaea.de/mag/airborne/Antarctica/ADMAP2A.zip",
            fname="admap2_gdb.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/magnetics",
            known_hash="a587555677350257dadbbf615838deac67e7d183a16525996ea0954eb23d83e8",
            processor=pooch.Unzip(),
            progressbar=True,
        )
        resampled = path
    else:
        msg = "invalid version string"
        raise ValueError(msg)

    return typing.cast(xr.DataArray, resampled)


def ghf(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
) -> xr.DataArray:
    """
    Load 1 of 6 'versions' of Antarctic geothermal heat flux data.

    version='an-2015'
    From :footcite:t:`antemperature2015`.
    Accessed from http://www.seismolab.org/model/antarctica/lithosphere/index.html

    version='martos-2017'
    From :footcite:t:`martosheat2017` and :footcite:t:`martosantarctic2017`.

    version='shen-2020':
    From :footcite:t:`shengeothermal2020`.
    Accessed from https://sites.google.com/view/weisen/research-products?authuser=0
    Used https://paperform.co/templates/apps/direct-download-link-google-drive/ to
    generate a direct download link from google drive page.
    https://drive.google.com/uc?export=download&id=1Fz7dAHTzPnlytuyRNctk6tAugCAjiqzR

    version='burton-johnson-2020'
    From :footcite:t:`burton-johnsongeothermal2020`.
    Accessed from supplementary material
    Choose for either of grid, or the point measurements

    version='losing-ebbing-2021'
    From :footcite:t:`losingpredicting2021` and :footcite:t:`losingpredicted2021`.

    version='aq1'
    From :footcite:t:`stalantarctic2021` and :footcite:t:`stalantarctic2020a`.

    Parameters
    ----------
    version : str
        Either 'burton-johnson-2020', 'losing-ebbing-2021', 'aq1',
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : int, optional
       grid spacing to resample the loaded grid to, by default spacing is read from
       downloaded files
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.
    kwargs : typing.Any
        if version='burton-johnson-2020', then kwargs are passed to return point
        measurements instead of the grid.

    Returns
    -------
    xarray.DataArray
         Returns a loaded, and optional clip/resampled grid of GHF data.

    References
    ----------
    .. footbibliography::
    """

    if version == "an-2015":
        # was in lat long, so just using standard values here
        initial_region = (-3330000.0, 3330000.0, -3330000.0, 3330000.0)
        initial_spacing = 5e3
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, reproject the .nc file, and save it back"
            fname = pooch.Untar()(fname, action, _pooch2)[0]
            fname1 = Path(fname)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname1.with_stem(fname1.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".nc")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load grid
                grid = xr.load_dataarray(fname1)

                # write the current projection
                grid = grid.rio.write_crs("EPSG:4326")

                # set names of coordinates
                grid = grid.rename({"lon": "x", "lat": "y"})

                # reproject to polar stereographic
                reprojected = grid.rio.reproject("epsg:3031")

                # need to save to .nc and reload, issues with pygmt
                reprojected.to_netcdf("tmp.nc")
                processed = xr.load_dataset("tmp.nc").z

                # remove tmp file
                pathlib.Path("tmp.nc").unlink()

                # get just antarctica region and save to disk
                pygmt.grdsample(
                    processed,
                    region=initial_region,
                    spacing=initial_spacing,
                    registration=initial_registration,
                    outgrid=fname_processed,
                )

            return str(fname_processed)

        path = pooch.retrieve(
            url="http://www.seismolab.org/model/antarctica/lithosphere/AN1-HF.tar.gz",
            fname="an_2015_.tar.gz",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/ghf",
            known_hash="9834439cdf99d5ee62fb88a008fa34dbc8d1848e9b00a1bd9cbc33194dd7d402",
            progressbar=True,
            processor=preprocessing,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
        )

    elif version == "martos-2017":
        # found from df.describe()
        initial_region = (-2535e3, 2715e3, -2130e3, 2220e3)
        initial_spacing = 15e3
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the .xyz file, grid it, and save it back as a .nc"
            fname1 = Path(fname)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname1.with_stem(fname1.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".nc")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load the data
                df = pd.read_csv(
                    fname1, header=None, sep=r"\s+", names=["x", "y", "GHF"]
                )

                # grid the data
                processed = pygmt.xyz2grd(
                    df,
                    region=initial_region,
                    registration=initial_registration,
                    spacing=initial_spacing,
                )

                # Save to disk
                processed.to_netcdf(fname_processed)
            return str(fname_processed)

        path = pooch.retrieve(
            url="https://store.pangaea.de/Publications/Martos-etal_2017/Antarctic_GHF.xyz",
            fname="martos_2017.xyz",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/ghf",
            known_hash="a5814bd0432986e111d0d48bfbd950cce66ba247b26b37f9a7499e66d969eb1f",
            progressbar=True,
            processor=preprocessing,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
        )

    elif version == "burton-johnson-2020":
        # found from utils.get_grid_info(grid)
        initial_region = (-2543500.0, 2624500.0, -2121500.0, 2213500.0)
        initial_spacing = 17e3
        initial_registration = "p"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        path = pooch.retrieve(
            url="https://doi.org/10.5194/tc-14-3843-2020-supplement",
            fname="burton_johnson_2020.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/ghf",
            known_hash="66b1f7acd06eeb6a6362c89b05db07034f510c81e3115cefbd4d11a584f143b2",
            processor=pooch.Unzip(extract_dir="burton_johnson_2020"),
            progressbar=True,
        )

        if kwargs.get("points", False) is True:
            url = "https://github.com/RicardaDziadek/Antarctic-GHF-DB/raw/master/ANT_GHF_DB_V004.xlsx"
            file = pooch.retrieve(
                url=url,
                fname="ANT_GHF_DB_V004.xlsx",
                path=f"{pooch.os_cache('pooch')}/polartoolkit/ghf",
                known_hash="192ad3862770de66f7ba82e9bc0bf1156ae3fccaabc76e9791edfb6c8fd4ff5f",
                progressbar=True,
            )

            # read the excel file with pandas
            df = pd.read_excel(file)

            # drop 2 extra columns
            df = df.drop(columns=["Unnamed: 15", "Unnamed: 16"])

            # remove numbers from all column names
            df.columns = df.columns.str[4:]

            # rename some columns, remove symbols
            df = df.rename(
                columns={
                    "Latitude": "lat",
                    "Longitude": "lon",
                    "grad (°C/km)": "grad",
                    "GHF (mW/m²)": "GHF",
                    "Err (mW/m²)": "err",
                },
            )

            # drop few rows without coordinates
            df = df.dropna(subset=["lat", "lon"])

            # re-project the coordinates to Polar Stereographic
            transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
            df["x"], df["y"] = transformer.transform(  # pylint: disable=unpacking-non-sequence
                df["lat"].tolist(),
                df["lon"].tolist(),
            )

            resampled = df

        elif kwargs.get("points", False) is False:
            file = next(p for p in path if p.endswith("Mean.tif"))
            # pygmt gives issues when original filepath has spaces in it. To get around
            # this, we will copy the file into the parent directory.
            try:
                new_file = shutil.copyfile(
                    file,
                    f"{pooch.os_cache('pooch')}/polartoolkit/ghf/burton_johnson_2020/Mean.tif",
                )
            except shutil.SameFileError:
                new_file = file

            grid = (
                xr.load_dataarray(new_file).squeeze().drop_vars(["band", "spatial_ref"])
            )

            resampled = resample_grid(
                grid,
                initial_spacing,
                initial_region,
                initial_registration,
                spacing,
                region,
                registration,
            )

    elif version == "losing-ebbing-2021":
        # was in lat long, so just using standard values here
        initial_region = (-3330000.0, 3330000.0, -3330000.0, 3330000.0)
        initial_spacing = 5e3  # given as 0.5degrees, which is ~3.5km at the pole
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the .csv file, grid it, and save it back as a .nc"
            fname1 = Path(fname)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname1.with_stem(fname1.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".nc")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load data
                df = pd.read_csv(fname1)

                # block-median and grid the data
                df = pygmt.blockmedian(
                    df[["Lon", "Lat", "HF [mW/m2]"]],
                    spacing="30m",
                    coltypes="g",
                    region="AQ",
                    registration=initial_registration,
                )
                grid = pygmt.surface(
                    data=df[["Lon", "Lat", "HF [mW/m2]"]],
                    spacing="30m",
                    coltypes="g",
                    region="AQ",
                    registration=initial_registration,
                )

                # re-project to polar stereographic
                reprojected = pygmt.grdproject(
                    grid,
                    projection="EPSG:3031",
                    # spacing=initial_spacing,
                )

                pygmt.grdsample(
                    reprojected,
                    spacing=initial_spacing,
                    region=initial_region,
                    registration=initial_registration,
                    outgrid=fname_processed,
                )

            return str(fname_processed)

        path = pooch.retrieve(
            url="https://download.pangaea.de/dataset/930237/files/HF_Min_Max_MaxAbs-1.csv",
            fname="losing_ebbing_2021_ghf.csv",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/ghf",
            known_hash="ecdae882083d8eb3503fab5be2ef862c96229f89ecbae1f95e56a8f43fb912e2",
            progressbar=True,
            processor=preprocessing,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
        )

    elif version == "aq1":
        # found from utils.get_grid_info(grid)
        initial_region = (-2800000.0, 2800000.0, -2800000.0, 2800000.0)
        initial_spacing = 20e3  # was actually 20071.6845878
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        path = pooch.retrieve(
            url="https://download.pangaea.de/dataset/924857/files/aq1_01_20.nc",
            fname="aq1.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/ghf",
            known_hash="946ae69e0a3d15a7500d7252fe0ce4f5cb126eaeb6170555ade0acdc38b86d7f",
            progressbar=True,
        )
        grid = xr.load_dataset(path)["Q"]

        grid = grid * 1000

        resampled = resample_grid(
            grid,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
        )

    elif version == "shen-2020":
        # was in lat long, so just using standard values here
        initial_region = regions.antarctica
        initial_spacing = 10e3
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the .csv file, grid it, and save it back as a .nc"
            fname1 = Path(fname)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname1.with_stem(fname1.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".nc")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load data
                df = pd.read_csv(
                    fname1,
                    sep=r"\s+",
                    header=None,
                    names=["lon", "lat", "GHF"],
                )
                # re-project to polar stereographic
                transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
                df["x"], df["y"] = transformer.transform(  # pylint: disable=unpacking-non-sequence
                    df.lat.tolist(), df.lon.tolist()
                )

                # block-median and grid the data
                df = pygmt.blockmedian(
                    df[["x", "y", "GHF"]],
                    spacing=initial_spacing,
                    region=initial_region,
                    registration=initial_registration,
                )
                processed = pygmt.surface(
                    data=df[["x", "y", "GHF"]],
                    spacing=initial_spacing,
                    region=initial_region,
                    registration=initial_registration,
                    maxradius="1c",
                )
                # Save to disk
                processed.to_netcdf(fname_processed)
            return str(fname_processed)

        path = pooch.retrieve(
            url="https://drive.google.com/uc?export=download&id=1Fz7dAHTzPnlytuyRNctk6tAugCAjiqzR",
            fname="shen_2020_ghf.xyz",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/ghf",
            known_hash="d6164c3680da52f8f03584293b1a271c937852df9a64f3c98d68debc44e02533",
            processor=preprocessing,
            progressbar=True,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
        )
    else:
        msg = "invalid version string"
        raise ValueError(msg)

    return typing.cast(xr.DataArray, resampled)  # pylint: disable=possibly-used-before-assignment


def gia(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
) -> xr.DataArray | None:
    """
    Load 1 of 1 'versions' of Antarctic glacial isostatic adjustment grids.

    version='stal-2020'
    From :footcite:t:`stalantarctic2020` and :footcite:t:`stalantarctic2020b`.

    Parameters
    ----------
    version : str
        For now the only option is 'stal-2020',
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : int, optional
       grid spacing to resample the loaded grid to, by default spacing is read from
       downloaded files
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.

    Returns
    -------
    xarray.DataArray
         Returns a loaded, and optional clip/resampled grid of GIA data.

    References
    ----------
    .. footbibliography::
    """

    if version == "stal-2020":
        # found from utils.get_grid_info(grid)
        initial_region = (-2800000.0, 2800000.0, -2800000.0, 2800000.0)
        initial_spacing = 10e3
        initial_registration = "p"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        path = pooch.retrieve(
            url="https://zenodo.org/record/4003423/files/ant_gia_dem_0.tiff?download=1",
            fname="stal_2020_gia.tiff",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/gia",
            known_hash="cb579c9606f98dfd28239183ba28de33e6e288a4256b27da7249c3741a24b7e8",
            progressbar=True,
        )
        grid = xr.load_dataarray(path).squeeze().drop_vars(["band", "spatial_ref"])

        resampled = resample_grid(
            grid,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
        )

    else:
        msg = "invalid version string"
        raise ValueError(msg)

    return typing.cast(xr.DataArray, resampled)


def crustal_thickness(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
) -> xr.DataArray | None:
    """
    Load 1 of x 'versions' of Antarctic crustal thickness grids.

    version='shen-2018'
    Crustal thickness (excluding ice layer) from :footcite:t:`shencrust2018`.
    Accessed from https://sites.google.com/view/weisen/research-products?authuser=0

    version='an-2015'
    Crustal thickness (distance from solid (ice and rock) top to Moho discontinuity)
    from :footcite:t:`ansvelocity2015`.
    Accessed from http://www.seismolab.org/model/antarctica/lithosphere/index.html#an1s
    File is the AN1-CRUST model, paper states "Moho depths and crustal thicknesses
    referred to below are the distance from the solid surface to the Moho. We note that
    this definition of Moho depth is different from that in the compilation of AN-MOHO".
    Unclear, but seems moho depth is just negative of crustal thickness. Not sure if its
    to the ice surface or ice base.

    Parameters
    ----------
    version : str
        Either 'shen-2018',
        will add later: 'lamb-2020',  'an-2015', 'baranov', 'chaput', 'crust1',
        'szwillus', 'llubes', 'pappa', 'stal'
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : int, optional
       grid spacing to resample the loaded grid to, by default spacing is read from
       downloaded files
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.

    Returns
    -------
    xarray.DataArray
         Returns a loaded, and optional clip/resampled grid of crustal thickness.

    References
    ----------
    .. footbibliography::
    """

    if version == "shen-2018":
        msg = "the link to the shen-2018 data appears to be broken"
        raise ValueError(msg)
        # was in lat long, so just using standard values here
        # initial_region = regions.antarctica
        # initial_spacing = 10e3  # given as 0.5degrees, which is ~3.5km at the pole
        # initial_registration = "g"

        # if region is None:
        #     region = initial_region
        # if spacing is None:
        #     spacing = initial_spacing
        # if registration is None:
        #     registration = initial_registration

        # def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
        #     "Load the .dat file, grid it, and save it back as a .nc"
        #     fname1 = Path(fname)

        #     # Rename to the file to ***_preprocessed.nc
        #     fname_pre = fname1.with_stem("shen_2018_crustal_thickness_preprocessed")
        #     fname_processed = fname_pre.with_suffix(".nc")

        #     # Only recalculate if new download or the processed file doesn't exist yet
        #     if action in ("download", "update") or not fname_processed.exists():
        #         # load data
        #         df = pd.read_csv(
        #             fname1,
        #             sep='\s+',
        #             header=None,
        #             names=["lon", "lat", "thickness"],
        #         )
        #         # convert to meters
        #         df.thickness = df.thickness * 1000

        #         # re-project to polar stereographic
        #         transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
        #         df["x"], df["y"] = transformer.transform(  # pylint: disable=unpacking-non-sequence # noqa: E501
        #             df.lat.tolist(), df.lon.tolist()
        #         )

        #         # block-median and grid the data
        #         df = pygmt.blockmedian(
        #             df[["x", "y", "thickness"]],
        #             spacing=initial_spacing,
        #             region=initial_region,
        #             registration=initial_registration,
        #         )
        #         processed = pygmt.surface(
        #             data=df[["x", "y", "thickness"]],
        #             spacing=initial_spacing,
        #             region=initial_region,
        #             registration=initial_registration,
        #             maxradius="1c",
        #         )
        #         # Save to disk
        #         processed.to_netcdf(fname_processed)
        #     return str(fname_processed)

        # url = "http://www.google.com/url?q=http%3A%2F%2Fweisen.wustl.edu%2FFor_Comrades%2Ffor_self%2Fmoho.WCANT.dat&sa=D&sntz=1&usg=AOvVaw0XC8VjO2gPVIt96QvzqFtw"

        # path = pooch.retrieve(
        #     url=url,
        #     known_hash=None,
        #     fname="shen_2018_crustal_thickness.dat",
        #     path=f"{pooch.os_cache('pooch')}/polartoolkit/crustal_thickness",
        #     processor=preprocessing,
        #     progressbar=True,
        # )

        # grid = xr.load_dataarray(path)

        # resampled = resample_grid(
        #     grid,
        #     initial_spacing,
        #     initial_region,
        #     initial_registration,
        #     spacing,
        #     region,
        #     registration,
        # )

    if version == "an-2015":
        # was in lat long, so just using standard values here
        initial_region = (-3330000.0, 3330000.0, -3330000.0, 3330000.0)
        initial_spacing = 5e3
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, reproject the .nc file, and save it back"
            path = pooch.Untar(
                extract_dir="An_2015_crustal_thickness", members=["AN1-CRUST.grd"]
            )(fname, action, _pooch2)
            fname1 = Path(path[0])
            # Rename to the file to ***_preprocessed.nc
            fname_processed = fname1.with_stem(fname1.stem + "_preprocessed")
            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load grid
                grid = xr.load_dataarray(fname1)

                # convert to meters
                grid = grid * 1000

                # write the current projection
                grid = grid.rio.write_crs("EPSG:4326")

                # set names of coordinates
                grid = grid.rename({"lon": "x", "lat": "y"})

                # reproject to polar stereographic
                reprojected = grid.rio.reproject("EPSG:3031")

                # get just antarctica region and save to disk
                pygmt.grdsample(
                    reprojected,
                    region=initial_region,
                    spacing=initial_spacing,
                    registration=initial_registration,
                    outgrid=fname_processed,
                )

            return str(fname_processed)

        path = pooch.retrieve(
            url="http://www.seismolab.org/model/antarctica/lithosphere/AN1-CRUST.tar.gz",
            fname="an_2015_crustal_thickness.tar.gz",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/crustal_thickness",
            known_hash="486da67ccbe76bb7f79725981c6078a0429a2cd2797d447702b90302e2b7b1e5",
            progressbar=True,
            processor=preprocessing,
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
        )

    else:
        msg = "invalid version string"
        raise ValueError(msg)

    return typing.cast(xr.DataArray, resampled)


def moho(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
) -> xr.DataArray | None:
    """
    Load 1 of x 'versions' of Antarctic Moho depth grids.

    version='shen-2018'
    Depth to the Moho relative to the surface of solid earth (bottom of ice/ocean)
    from :footcite:t:`shencrust2018`.
    Accessed from https://sites.google.com/view/weisen/research-products?authuser=0
    Appears to be almost identical to crustal thickness from Shen et al. 2018

    version='an-2015'
    This is fetch.crustal_thickness(version='an-2015)* -1
    Documentation is unclear whether the An crust model from
    :footcite:t:`ansvelocity2015` is crustal thickness or moho depths, or whether it
    makes a big enough difference to matter.

    version='pappa-2019'
    from :footcite:t:`pappamoho2019`.
    Accessed from supplement material

    Parameters
    ----------
    version : str
        Either 'shen-2018', 'an-2015', 'pappa-2019',
        will add later: 'lamb-2020', 'baranov', 'chaput', 'crust1',
        'szwillus', 'llubes',
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : int, optional
       grid spacing to resample the loaded grid to, by default spacing is read from
       downloaded files
    registration : str, optional,
        choose between 'g' (gridline) or 'p' (pixel) registration types, by default is
        the original type of the grid

    Returns
    -------
    xarray.DataArray
         Returns a loaded, and optional clip/resampled grid of crustal thickness.

    References
    ----------
    .. footbibliography::
    """

    if version == "shen-2018":
        # was in lat long, so just using standard values here
        initial_region = regions.antarctica
        initial_spacing = 10e3  # given as 0.5degrees, which is ~3.5km at the pole
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the .dat file, grid it, and save it back as a .nc"
            path = pooch.Untar(
                extract_dir="Shen_2018_moho", members=["WCANT_MODEL/moho.final.dat"]
            )(fname, action, _pooch2)
            fname1 = next(p for p in path if p.endswith("moho.final.dat"))
            fname2 = Path(fname1)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname2.with_stem(fname2.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".nc")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load data
                df = pd.read_csv(
                    fname2,
                    sep=r"\s+",
                    header=None,
                    names=["lon", "lat", "depth"],
                )
                # convert to meters
                df.depth = df.depth * -1000

                # re-project to polar stereographic
                transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
                df["x"], df["y"] = transformer.transform(  # pylint: disable=unpacking-non-sequence
                    df.lat.tolist(), df.lon.tolist()
                )

                # block-median and grid the data
                df = pygmt.blockmedian(
                    df[["x", "y", "depth"]],
                    spacing=initial_spacing,
                    region=initial_region,
                    registration=initial_registration,
                )
                processed = pygmt.surface(
                    data=df[["x", "y", "depth"]],
                    spacing=initial_spacing,
                    region=initial_region,
                    registration=initial_registration,
                    maxradius="1c",
                )
                # Save to disk
                processed.to_netcdf(fname_processed)
            return str(fname_processed)

        path = pooch.retrieve(
            url="https://drive.google.com/uc?export=download&id=1huoGe54GMNc-WxDAtDWYmYmwNIUGrmm0",
            fname="shen_2018_moho.tar",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/moho",
            known_hash=None,
            progressbar=True,
            processor=preprocessing,
            downloader=pooch.HTTPDownloader(timeout=60),
        )

        grid = xr.load_dataarray(path)

        resampled = resample_grid(
            grid,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
        )

    elif version == "an-2015":
        # was in lat long, so just using standard values here
        initial_region = (-3330000.0, 3330000.0, -3330000.0, 3330000.0)
        initial_spacing = 5e3
        initial_registration = "g"

        if region is None:
            region = initial_region
        if spacing is None:
            spacing = initial_spacing
        if registration is None:
            registration = initial_registration

        grid = crustal_thickness(version="an-2015") * -1  # type: ignore[operator]

        resampled = resample_grid(
            grid,
            initial_spacing,
            initial_region,
            initial_registration,
            spacing,
            region,
            registration,
        )

    elif version == "pappa-2019":
        msg = "Pappa et al. 2019 moho model download is not working currently."
        raise ValueError(msg)
        # resampled = pooch.retrieve(
        #     url="https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1029%2F2018GC008111&file=GGGE_21848_DataSetsS1-S6.zip",  # noqa: E501
        #     fname="pappa_moho.zip",
        #     path=f"{pooch.os_cache('pooch')}/polartoolkit/moho",
        #     known_hash=None,
        #     progressbar=True,
        #     processor=pooch.Unzip(extract_dir="pappa_moho"),
        # )
        # fname='/Volumes/arc_04/tankerma/Datasets/Pappa_et_al_2019_data/2018GC008111_Moho_depth_inverted_with_combined_depth_points.grd' # noqa: E501
        # grid = pygmt.load_dataarray(fname)
        # Moho_Pappa = grid.to_dataframe().reset_index()
        # Moho_Pappa.z=Moho_Pappa.z.apply(lambda x:x*-1000)

        # transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
        # Moho_Pappa['x'], Moho_Pappa['y'] = transformer.transform(
        #   Moho_Pappa.lat.tolist(),
        # Moho_Pappa.lon.tolist())

        # Moho_Pappa = pygmt.blockmedian(
        #   Moho_Pappa[['x','y','z']],
        #   spacing=10e3,
        #   registration='g',
        #   region='-1560000/1400000/-2400000/560000',
        # )

        # fname='inversion_layers/Pappa_moho.nc'

        # pygmt.surface(
        #   Moho_Pappa[['x','y','z']],
        #   region='-1560000/1400000/-2400000/560000',
        #   spacing=10e3,
        #   registration='g',
        #   maxradius='1c',
        #   outgrid=fname,
        # )

    else:
        msg = "invalid version string"
        raise ValueError(msg)

    return typing.cast(xr.DataArray, resampled)
