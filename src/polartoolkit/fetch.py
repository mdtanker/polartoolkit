# pylint: disable=too-many-lines
import glob
import pathlib
import re
import shutil
import typing
import warnings
from inspect import getmembers, isfunction

import deprecation
import earthaccess
import geopandas as gpd
import harmonica as hm
import numpy as np
import pandas as pd
import pooch
import pygmt
import requests
import xarray as xr
from dotenv import load_dotenv
from tqdm.autonotebook import tqdm

import polartoolkit
from polartoolkit import (  # pylint: disable=import-self
    fetch,  # noqa: PLW0406
    logger,
    regions,
    utils,
)

try:
    import pyarrow as pa  # pylint: disable=unused-import # noqa: F401

    USE_ARROW = True
except ImportError:
    USE_ARROW = False


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
    grid: xr.DataArray,
    spacing: float | None = None,
    region: tuple[float, float, float, float] | None = None,
    registration: str | None = None,
    **kwargs: dict[str, str],
) -> xr.DataArray:
    """
    Resample a grid to a new spacing, region, and/or registration. Method of resampling
    depends on comparison with initial and supplied values for spacing, region, and
    registration. If initial values not supplied, will try and extract them from the
    grid.

    Parameters
    ----------
    grid : xarray.DataArray
        grid to resample
    spacing : float | None, optional
        new spacing for grid, by default None
    region : tuple[float, float, float, float] | None, optional
        new region for grid in format [xmin, xmax, ymin, ymax], by default None
    registration : str | None, optional
        new registration for grid, by default None

    Returns
    -------
    xarray.DataArray
        grid, either resampled or same as original depending on inputs. If no
        resampling, and supplied grid is a filepath, returns filepath.
    """

    # get coordinate names
    # original_dims = tuple(grid.sizes.keys())

    verbose = kwargs.get("verbose", "e")

    if (spacing is None) & (region is None) & (registration is None):
        logger.info("returning original grid")
        return grid

    grd_info = utils.get_grid_info(grid)
    initial_spacing, initial_region, _, _, initial_registration = grd_info

    logger.debug(
        "using initial values: %s, %s, %s",
        initial_spacing,
        initial_region,
        initial_registration,
    )

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
    # no changes
    if all(rules):
        logger.info("returning original grid")
        return grid
    # only spacing changes
    if not rules[0] and rules[1] and rules[2]:
        logger.info("changing grid spacing")
        new_region = tuple(  # pylint: disable=consider-using-generator
            [
                float(x)
                for x in pygmt.grdinfo(grid, spacing=f"{spacing}r")[2:-1].split("/")
            ]
        )
        if new_region != region:
            logger.info(
                "region (%s) updated to %s to match desired spacing",
                region,
                new_region,
            )
        if spacing < initial_spacing:  # type: ignore[operator]
            logger.warning(
                "requested spacing (%s) is smaller than the original (%s).",
                spacing,
                initial_spacing,
            )
            resampled = grid
        elif spacing > initial_spacing:  # type: ignore[operator]
            logger.info(
                "requested spacing (%s) larger than original (%s), filtering before "
                "resampling",
                spacing,
                initial_spacing,
            )
            resampled = pygmt.grdfilter(
                grid=grid,
                filter=f"g{spacing}",
                # region=new_region,
                distance=kwargs.get("distance", "0"),
                verbose=verbose,
            )
        resampled = pygmt.grdsample(
            grid=resampled,
            region=new_region,
            spacing=f"{spacing}+e",
            verbose=verbose,
        )
        assert spacing == utils.get_grid_info(resampled)[0], (
            "spacing not correctly updated"
        )
        return resampled
    # only region changes
    if rules[0] and not rules[1] and rules[2]:
        logger.info("changing grid region")
        # get region to nearest multiple of spacing
        new_region = tuple([spacing * round(x / spacing) for x in region])  # type: ignore[operator, union-attr] # pylint: disable=consider-using-generator
        if new_region != region:
            logger.info(
                "supplied region (%s) updated to %s to match spacing",
                region,
                new_region,
            )
        cut = pygmt.grdcut(
            grid=grid,
            region=new_region,
            extend="",
            verbose=verbose,
        )
        return cut  # noqa: RET504
    # only registration changes
    if rules[0] and rules[1] and not rules[2]:
        logger.info("changing grid registration")
        try:
            return utils.change_reg(grid)
        except ValueError as e:
            logger.exception(e)
            logger.error("changing registration failed")
            return grid
        # changed = check_registration_changed(grid, resampled)
        # if not changed:
        # resampled = utils.change_reg(resampled)
    # other combinations
    else:
        # if both spacing and region changed
        # if rules[0] and rules[1]:
        logger.info("changing grid region, spacing and/or registration")
        # for speed, first subset then change spacing
        # get region to nearest multiple of spacing
        new_region = tuple([spacing * round(x / spacing) for x in region])  # type: ignore[operator, union-attr] # pylint: disable=consider-using-generator
        if new_region != region:
            logger.info(
                "supplied region (%s) updated to %s to match desired spacing",
                region,
                new_region,
            )
        cut = pygmt.grdcut(
            grid=grid,
            region=new_region,
            extend="",
            verbose=verbose,
        )
        # new_reg = [float(x) for x in pygmt.grdinfo(grid, spacing=f"{spacing}r")[2:-1]]
        if spacing < initial_spacing:  # type: ignore[operator]
            logger.warning(
                "requested spacing (%s) is smaller than the original (%s).",
                spacing,
                initial_spacing,
            )
            resampled = cut
        elif spacing > initial_spacing:  # type: ignore[operator]
            logger.info(
                "requested spacing (%s) larger than original (%s), filtering before "
                "resampling",
                spacing,
                initial_spacing,
            )
            resampled = pygmt.grdfilter(
                grid=cut,
                filter=f"g{spacing}",
                # region=new_region,
                distance=kwargs.get("distance", "0"),
                verbose=verbose,
            )
        else:
            resampled = cut
        resampled = pygmt.grdsample(
            grid=resampled,
            region=new_region,
            spacing=f"{spacing}+e",
            verbose=verbose,
            registration=registration,
        )
        # if new region entirely within original, check region has been updated
        original_region = utils.get_grid_info(grid)[1]
        if all(
            [
                new_region[0] > original_region[0],  # type: ignore[index]
                new_region[1] < original_region[1],  # type: ignore[index]
                new_region[2] > original_region[2],  # type: ignore[index]
                new_region[3] < original_region[3],  # type: ignore[index]
            ]
        ):
            if new_region != utils.get_grid_info(resampled)[1]:
                msg = "region not correctly updated"
                warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            pass
        assert spacing == utils.get_grid_info(resampled)[0], (
            "spacing not correctly updated"
        )
        return resampled


class EarthDataDownloader:
    """
    Either pulls login details from pre-set environment variables, or prompts user to
    input username and password. Will persist the entered details within the python
    session.
    """

    def __call__(self, url: str, output_file: str, dataset: typing.Any) -> None:
        auth = earthaccess.login()
        auth = earthaccess.__auth__

        if auth.authenticated is False:
            msg = (
                "EarthData login failed, please check your Username and Password are "
                "correct"
            )
            raise ValueError(msg)

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
        msg = "name must be either 'Disco_deep_transect' or 'Roosevelt_Island'"
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
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
) -> typing.Any:
    """
    Ice-sheet height and thickness changes from ICESat to ICESat-2 for both Antarctica
    and Greenland from :footcite:t:`smithpervasive2020`.

    Choose a version of the data to download with the format: "ICESHEET_VERSION_TYPE"
    where ICESHEET is "ais" or "gris", for Antarctica or Greenland, VERSION is "dhdt"
    for total thickness change or "dmdt" for corrected for firn-air content. For
    Antarctic data, TYPE is "floating" or "grounded".

    add "_filt" to retrieve a filtered version of the data for some versions.

    accessed from https://digital.lib.washington.edu/researchworks/handle/1773/45388

    Units are in m/yr

    Parameters
    ----------
    version : str, optional,
        choose which version to retrieve, by default is "ais_dhdt_grounded" for
        Antarctica and "gris_dhdt" for Greenland.
    hemisphere : str, optional
        choose which hemisphere to retrieve data for, "north" or "south", by default
        None
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : float, optional,
        grid spacing to resample the loaded grid to, by default is 5km
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is "p".
    kwargs : typing.Any
        additional keyword arguments to pass to resample_grid

    Returns
    -------
    xarray.DataArray
        Returns a grid of Antarctic or Greenland ice mass change in meters/year.

    References
    ----------
    .. footbibliography::
    """
    if version is None:
        hemisphere = utils.default_hemisphere(hemisphere)
        if hemisphere == "south":
            version = "ais_dhdt_grounded"
        elif hemisphere == "north":
            version = "gris_dhdt"
        else:
            msg = "if version is None, must provide 'hemisphere'"
            raise ValueError(msg)

    # This is the path to the processed (magnitude) grid
    url = (
        "https://digital.lib.washington.edu/researchworks/bitstreams/"
        "cc12195c-b71e-4e26-bf85-0978dd9ce933/download"
    )

    zip_fname = "ICESat1_ICESat2_mass_change_updated_2_2021.zip"

    valid_versions = [
        # dhdt
        "ais_dhdt_floating",
        "ais_dhdt_floating_filt",
        "ais_dhdt_grounded",
        "ais_dhdt_grounded_filt",
        "gris_dhdt",
        "gris_dhdt_filt",
        # dmdt
        "ais_dmdt_floating",
        "ais_dmdt_floating_filt",
        "ais_dmdt_grounded",
        "ais_dmdt_grounded_filt",
        "gris_dmdt",
        "gris_dmdt_filt",
    ]
    if version not in valid_versions:
        msg = "version must be one of " + ", ".join(valid_versions)
        raise ValueError(msg)

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

    fname = next(p for p in path if p.endswith(f"{version}.tif"))

    grid = (
        xr.load_dataarray(
            fname,
            engine="rasterio",
        )
        .squeeze()
        .drop_vars(["band", "spatial_ref"])
    )

    return resample_grid(
        grid,
        spacing=spacing,
        region=region,
        registration=registration,
        **kwargs,
    )


def basal_melt(
    version: str = "w_b",
    variable: str | None = None,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
) -> typing.Any:
    """
    Antarctic ice shelf basal melt rates for 1994-2018 from satellite radar altimetry.
    from :footcite:t:`adusumilliinterannual2020`.

    accessed from http://library.ucsd.edu/dc/object/bb0448974g

    reading files and preprocessing from supplied jupyternotebooks:
    https://github.com/sioglaciology/ice_shelf_change/blob/master/read_melt_rate_file.ipynb

    Units are in m/yr

    Parameters
    ----------
    version : str
        choose which version to load, either 'w_b' for basal melt rate, 'w_b_interp',
        for basal melt rate with interpolated values, and 'w_b_uncert' for uncertainty

    Returns
    -------
    xarray.DataArray
        Returns a dataarray of basal melt rate values

    References
    ----------
    .. footbibliography::
    """

    if variable is not None:
        version = variable
        msg = "variable parameter is deprecated, please use version parameter instead"
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    # This is the path to the processed (magnitude) grid
    url = "https://library.ucsd.edu/dc/object/bb0448974g/_3_1.h5/download"
    fname = "ANT_iceshelf_melt_rates_CS2_2010-2018_v0.h5"

    def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Download the .h5 file, save to .zarr and return fname"
        fname1 = pathlib.Path(fname)

        # Rename to the file to ***.zarr
        fname_processed = fname1.with_suffix(".zarr")

        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            # load .h5 file
            try:
                grid = xr.load_dataset(
                    fname1,
                    engine="netcdf4",
                )
            except OSError as e:
                msg = (
                    "Unfortunately, this dataset is not available for download at the "
                    "moment, follow here for details: "
                    "https://github.com/mdtanker/polartoolkit/issues/250"
                )
                raise OSError(msg) from e

            # Remove extra dimension
            grid = grid.squeeze()

            # Assign variables as coords
            grid = grid.assign_coords({"easting": grid.x, "northing": grid.y})

            # Swap dimensions with coordinate names
            grid = grid.swap_dims({"phony_dim_1": "easting", "phony_dim_0": "northing"})

            # Drop coordinate variables
            grid = grid.drop_vars(["x", "y"])

            # Save to .zarr file
            grid.to_zarr(
                fname_processed,
            )

        return str(fname_processed)

    # known_hash="c14f7059876e6808e3869853a91a3a17a776c95862627c4a3d674c12e4477d2a"
    known_hash = None
    path = pooch.retrieve(
        url=url,
        fname=fname,
        path=f"{pooch.os_cache('pooch')}/polartoolkit/mass_change/Admusilli_2020",
        known_hash=known_hash,
        progressbar=True,
        processor=preprocessing,
    )

    grid = xr.open_zarr(
        path,
        consolidated=False,
    )[version]

    return resample_grid(
        grid,
        spacing=spacing,
        region=region,
        registration=registration,
        **kwargs,
    )


def buttressing(
    version: str,
    variable: str | None = None,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
) -> typing.Any:
    """
    Antarctic ice shelf buttressing.
    from :footcite:t:`furstsafety2016`.

    accessed from https://nsidc.org/data/nsidc-0664/versions/1

    Units are in m/yr

    Parameters
    ----------
    version : str
        choose which version to load, either 'max' for maximum buttressing, 'min' for
        minimum buttressing, 'flow' for along-flow buttressing, or 'viscosity' for
        estimated ice viscosity values

    Returns
    -------
    xarray.DataArray
        Returns a dataarray of buttressing or viscosity values

    References
    ----------
    .. footbibliography::
    """

    if variable is not None:
        version = variable
        msg = "variable parameter is deprecated, please use version parameter instead"
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    base_url = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0664_antarctic_iceshelf_buttress/"

    if version == "max":
        var = "bmax"
    elif version == "min":
        var = "bmin"
    elif version == "flow":
        var = "bflow"
    elif version == "viscosity":
        var = "visc"
    else:
        msg = "version must be one of 'max', 'min', 'flow', or 'viscosity'"
        raise ValueError(msg)

    fname = f"{var}_nsidc_sumer_buttressing_v1.0.nc"
    url = base_url + fname

    path = pooch.retrieve(
        url=url,
        fname=fname,
        path=f"{pooch.os_cache('pooch')}/polartoolkit/buttressing/",
        known_hash=None,
        progressbar=True,
        downloader=EarthDataDownloader(),
    )

    grid = xr.load_dataset(path)[var]

    return resample_grid(
        grid,
        spacing=spacing,
        region=region,
        registration=registration,
        **kwargs,
    )


def ice_vel(
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    hemisphere: str | None = None,
    **kwargs: typing.Any,
) -> xr.DataArray:
    """
    MEaSUREs Phase-Based Ice Velocity Maps for Antarctica and Greenland.

    Antarctica: version 1 from :footcite:t:`mouginotcontinentwide2019` and
    :footcite:t:`mouginotmeasures2019`.

    accessed from https://cmr.earthdata.nasa.gov/virtual-directory/collections/C3298047930-NSIDC_CPRD
    Data part of https://doi.org/10.1029/2019GL083826

    Greenland: version 1 from :footcite:t:`measures2020`

    accessed from https://cmr.earthdata.nasa.gov/virtual-directory/collections/C3291956575-NSIDC_CPRD

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

        # preprocessing for full, 450m resolution
        def preprocessing_fullres(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the .nc file, calculate velocity magnitude, save it to a .zarr"

            fname1 = pathlib.Path(fname)
            # Rename to the file to ***_preprocessed.zarr
            fname_pre = fname1.with_stem(fname1.stem + "_preprocessed_fullres")
            fname_processed = fname_pre.with_suffix(".zarr")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                msg = "this file is large (~7Gb) and may take some time to download!"
                warnings.warn(msg, stacklevel=2)
                msg = (
                    "preprocessing this grid in full resolution is very "
                    "computationally demanding, consider choosing a lower resolution "
                    "using the parameter `spacing`."
                )
                warnings.warn(msg, stacklevel=2)
                with xr.open_dataset(fname1) as ds:
                    processed = (ds.VX**2 + ds.VY**2) ** 0.5
                    # restore registration type
                    processed.gmt.registration = ds.VX.gmt.registration
                    # Save to disk
                    processed = processed.to_dataset(name="vel")
                    processed.to_zarr(fname_processed)

            return str(fname_processed)

        # preprocessing for filtered 5k resolution
        def preprocessing_5k(fname: str, action: str, _pooch2: typing.Any) -> str:
            """
            Load the .nc file, calculate velocity magnitude, resample to 5k, save it
            to a .zarr
            """

            fname1 = pathlib.Path(fname)
            # Rename to the file to ***_preprocessed_5k.zarr
            fname_pre = fname1.with_stem(fname1.stem + "_preprocessed_5k")
            fname_processed = fname_pre.with_suffix(".zarr")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                msg = "this file is large (~7Gb) and may take some time to download!"
                warnings.warn(msg, stacklevel=2)
                with xr.open_dataset(fname1) as ds:
                    vx_5k = resample_grid(
                        ds.VX,
                        spacing=5e3,
                        **kwargs,
                    )
                    vx_5k = typing.cast(xr.DataArray, vx_5k)
                    vy_5k = resample_grid(
                        ds.VY,
                        spacing=5e3,
                        **kwargs,
                    )
                    vy_5k = typing.cast(xr.DataArray, vy_5k)
                    processed = (vx_5k**2 + vy_5k**2) ** 0.5
                    # restore registration type
                    processed.gmt.registration = ds.VX.gmt.registration
                    # Save to disk
                    processed = processed.to_dataset(name="vel")
                    processed.to_zarr(fname_processed)

            return str(fname_processed)

        # determine which resolution of preprocessed grid to use
        if spacing < 5000:
            preprocessor = preprocessing_fullres
        elif spacing >= 5000:
            logger.info("using preprocessed 5km grid since spacing is > 5km")
            preprocessor = preprocessing_5k
        # This is the path to the processed (magnitude) grid
        path = pooch.retrieve(
            url="https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/MEASURES/NSIDC-0754/1/1996/01/01/antarctic_ice_vel_phase_map_v01.nc",
            fname="measures_ice_vel_phase_map.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/ice_velocity",
            downloader=EarthDataDownloader(),
            known_hash="fa0957618b8bd98099f4a419d7dc0e3a2c562d89e9791b4d0ed55e6017f52416",
            progressbar=True,
            processor=preprocessor,  # pylint: disable=possibly-used-before-assignment
        )
        grid = xr.open_zarr(
            path,
            consolidated=False,
        )["vel"]
        resampled = resample_grid(
            grid,
            spacing=spacing,
            region=region,
            registration=registration,
            **kwargs,
        )

    elif hemisphere == "north":
        if spacing is None:
            spacing = 250

        base_fname = "greenland_vel_mosaic250"
        registry = {
            f"{base_fname}_vx_v1.tif": None,
            f"{base_fname}_vy_v1.tif": None,
        }
        base_url = "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/MEASURES/NSIDC-0670/1/1995/12/01/"
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

        # restore registration type
        processed.gmt.registration = grid_x.gmt.registration

        resampled = resample_grid(
            processed,
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
    removed_in="2.0.0",
    current_version=polartoolkit.__version__,
    details="Use the new function modis(hemisphere='south') instead",
)
def modis_moa(version: str = "750m") -> str:
    """deprecated function, use modis(hemisphere="south") instead"""
    return modis(version=version, hemisphere="south")


@deprecation.deprecated(
    deprecated_in="0.4.0",
    removed_in="2.0.0",
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
) -> gpd.GeoDataFrame:
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
    fname2 = pathlib.Path(fname1)

    # found layer names with: fiona.listlayers(fname)

    if version == "faults":
        layer = "ATA_GeoMAP_faults_v2022_08"
    elif version == "units":
        layer = "ATA_GeoMAP_geological_units_v2022_08"
        qml_fname = "ATA geological units - Simple geology.qml"
        qml = next(p for p in path if p.endswith(qml_fname))
        with pathlib.Path(qml).open(encoding="utf8") as f:
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
        msg = "version must be one of 'faults', 'units', 'sources', or 'quality'"
        raise ValueError(msg)

    if region is None:
        data = gpd.read_file(
            fname2,
            layer=layer,
            use_arrow=USE_ARROW,
            engine="pyogrio",
        )
    else:
        data = gpd.read_file(
            fname2,
            bbox=utils.region_to_bounding_box(region),
            layer=layer,
            use_arrow=USE_ARROW,
            engine="pyogrio",
        )

    if version == "units":
        data = data.merge(unit_symbols)
        data["SIMPsymbol"] = data.SIMPsymbol.astype(float)
        data = data.sort_values("SIMPsymbol")

    return data


def groundingline(
    version: str = "measures-v2",
) -> str:
    """
    Load the file path of two versions of groundingline shapefiles

    version = "depoorter-2013"
    from :footcite:t:`depoorterantarctic2013`.
    Supplement to :footcite:t:`depoortercalving2013`.
    accessed at https://doi.pangaea.de/10.1594/PANGAEA.819147

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
        choose which version to retrieve, by default "measures-v2"

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
        msg = (
            "version must be one of 'depoorter-2013', 'measures-v2', 'BAS', or"
            "'measures-greenland'"
        )
        raise ValueError(msg)

    return fname


@deprecation.deprecated(
    deprecated_in="0.4.0",
    removed_in="2.0.0",
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
        msg = (
            "version must be one of 'Coastline', 'Basins_Antarctica', 'Basins_IMBIE',"
            "'IceBoundaries','IceShelf','Mask'"
        )
        raise ValueError(msg)

    return fname


def sediment_thickness(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
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
    **kwargs : typing.Any
        additional keyword arguments to pass to the resample_grid function

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of sediment thickness.

    References
    ----------
    .. footbibliography::
    """

    if version == "ANTASed":

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, grid the .dat file, and save it back as a .nc"
            path = pooch.Unzip(
                extract_dir="Baranov_2021_sediment_thickness",
            )(fname, action, _pooch2)
            fname1 = next(p for p in path if p.endswith(".dat"))
            fname2 = pathlib.Path(fname1)

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
                    region=(-2350000.0, 2490000.0, -1990000.0, 2090000.0),
                    spacing=10e3,
                    registration="g",
                )
                processed = pygmt.xyz2grd(
                    data=df[["x", "y", "thick"]],
                    region=(-2350000.0, 2490000.0, -1990000.0, 2090000.0),
                    spacing=10e3,
                    registration="g",
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
            spacing=spacing,
            region=region,
            registration=registration,
            **kwargs,
        )

    elif version == "tankersley-2022":
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
            spacing=spacing,
            region=region,
            registration=registration,
            **kwargs,
        )

    elif version == "lindeque-2016":
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
            spacing=spacing,
            region=region,
            registration=registration,
            **kwargs,
        )

    elif version == "GlobSed":

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, reproject the grid, and save it back as a .zarr"
            path = pooch.Unzip(
                extract_dir="GlobSed",
            )(fname, action, _pooch2)
            fname1 = next(p for p in path if p.endswith("GlobSed-v3.nc"))
            fname2 = pathlib.Path(fname1)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname2.with_stem(fname2.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".zarr")

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
                    *utils.region_to_bounding_box(
                        regions.alter_region(regions.antarctica, -1000e3)
                    ),
                    crs="EPSG:3031",
                )

                # reproject to EPSG:3031
                reprojected = grid.rio.reproject("EPSG:3031", resolution=1000)

                # resample to correct spacing, region and registration
                resampled = resample_grid(
                    reprojected,
                    spacing=1e3,
                    region=regions.antarctica,
                    registration="g",
                    **kwargs,
                )

                # Save to .zarr file
                resampled = resampled.to_dataset(name="sediment_thickness")
                resampled.to_zarr(
                    fname_processed,
                )

            return str(fname_processed)

        path = pooch.retrieve(
            url="https://ngdc.noaa.gov/mgg/sedthick/data/version3/GlobSed.zip",
            fname="GlobSed.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/sediment_thickness",
            known_hash="e063ee6603d65c9cee6420cb52a4c6afb520143711b12d618f1a2f591d248bd9",
            processor=preprocessing,
            progressbar=True,
        )

        grid = xr.open_zarr(
            path,
            consolidated=False,
        )["sediment_thickness"]

        resampled = resample_grid(
            grid,
            spacing=spacing,
            region=region,
            registration=registration,
            **kwargs,
        )

    else:
        msg = (
            "version must be one of 'ANTASed', 'tankersley-2022', 'lindeque-2016', or "
            "'GlobSed'"
        )
        raise ValueError(msg)

    return typing.cast(xr.DataArray, resampled)


def ibcso_coverage(
    region: tuple[float, float, float, float] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load IBCSO v2 data, from :footcite:t:`dorschelinternational2022` and
    :footcite:t:`dorschelinternational2022a`.

    Parameters
    ----------
    region : tuple[float, float, float, float] or None
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        Returns two geodataframes; points and polygons for a subset of IBCSO v2 point
        measurement locations. Column 'dataset_tid' is the type identifier from IBCSO.
        The points geodataframe contains all individual point measurements, including
        single-beam (TID 10), seismic points (TID 12), isolated soundings (TID 13),
        ENC sounding (TID 14), grounded iceberg draft (TID 46), and gravity-inverted
        bathymetry (TID 45). The polygon geodataframe contains all polygon measurements,
        including multi-beam (swath) (TID 11), contours from charts (TID 42), or
        other unknown sources (TID 71).

    References
    ----------
    .. footbibliography::
    """

    # download / retrieve the geopackage file
    fname = pooch.retrieve(
        url="https://download.pangaea.de/dataset/937574/files/IBCSO_v2_coverage.gpkg",
        fname="IBCSO_v2_coverage.gpkg",
        path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
        known_hash="b89e54f26c03b74f0b0f8d826f0f130573eac2c8240de2eb178c8840f0aa99a0",
        progressbar=True,
    )

    # extract the geometries which are within the supplied region
    if region is None:
        region = regions.antarctica
        msg = (
            "this file is large, if you only need a subset of data please provide "
            "a bounding box region via `region` to subset the data, using "
            "`regions.antarctica` as a default"
        )
        warnings.warn(msg, stacklevel=2)

    # users supply region in EPSG:3031, but the data is in EPSG:9354
    reg_df = utils.region_to_df(region)
    region_epsg_9354 = utils.reproject(
        reg_df,
        input_crs="epsg:3031",
        output_crs="epsg:9354",
        reg=True,
        input_coord_names=("easting", "northing"),
        output_coord_names=("easting", "northing"),
    )
    bbox = utils.region_to_bounding_box(region_epsg_9354)  # type: ignore[arg-type]

    data = gpd.read_file(
        fname,
        bbox=bbox,
        use_arrow=USE_ARROW,
        engine="pyogrio",
    )

    # expand from multipoint/mulitpolygon to point/polygon
    # this is slow!
    data_coords = data.explode(index_parts=False).copy()

    # extract the single points/polygons within region
    if region is not None:
        data_coords = data_coords.clip(mask=bbox).copy()

    # separate points and polygons
    points = data_coords[data_coords.geometry.type == "Point"].copy()
    polygons = data_coords[data_coords.geometry.type == "Polygon"].copy()

    # reproject to EPSG3031
    points = points.to_crs(epsg=3031)
    polygons = polygons.to_crs(epsg=3031)

    # extract reprojected coordinates
    points["easting"] = points.get_coordinates().x
    points["northing"] = points.get_coordinates().y

    return (points, polygons)


def ibcso(
    layer: str,
    reference: str = "geoid",
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
) -> xr.DataArray:
    """
    Load IBCSO v2 data, from :footcite:t:`dorschelinternational2022` and
    :footcite:t:`dorschelinternational2022a`.

    By default the elevations are relative to Mean Sea Level (the geoid). To convert
    them to be relative to the WGS84 ellipsoid, set `reference="ellipsoid` which will
    add the EIGEN-6C4 geoid anomaly.

    Parameters
    ----------
    layer : str
        choose which layer to fetch:
        'surface', 'bed'
    reference : str, optional
        choose which vertical reference to use, 'geoid' or 'ellipsoid', by default
        'geoid'
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 500m. If spacing >=
        5000m, will resample the grid to 5km, and save it as a preprocessed grid, so
        future fetch calls are performed faster.
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.
    **kwargs : typing.Any
        additional keyword arguments to pass to the resample_grid function

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of IBCSO data.

    References
    ----------
    .. footbibliography::
    """

    def preprocessing_fullres(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Load the .nc file, reproject, and save it back"
        fname1 = pathlib.Path(fname)

        # Rename to the file to ***_preprocessed.zarr
        fname_pre = fname1.with_stem(fname1.stem + "_preprocessed")
        fname_processed = fname_pre.with_suffix(".zarr")

        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            # give warning about time
            msg = (
                "preprocessing for this grid (reprojecting to EPSG:3031) for"
                " the first time can take several minutes!"
            )
            warnings.warn(msg, stacklevel=2)
            # load grid
            grid = xr.load_dataset(fname1).z

            # set the projection
            grid = grid.rio.write_crs("EPSG:9354")

            # reproject to EPSG:3031
            reprojected = grid.rio.reproject("EPSG:3031")

            # resample to correct spacing, region and registration
            resampled = pygmt.grdsample(
                grid=reprojected,
                spacing=500,
                region=(-3500000.0, 3500000.0, -3500000.0, 3500000.0),
                registration="g",
            )

            # Save to .zarr file
            resampled = resampled.to_dataset(name=layer)
            resampled.to_zarr(fname_processed)

        return str(fname_processed)

    def preprocessing_5k(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Load preprocessed full-res grid, resample to 5km and save to .zarr file"

        # get the path to the .nc file
        fname1 = pathlib.Path(fname)

        # add _5k to .zarr file name
        fname_pre = fname1.with_stem(fname1.stem + "_preprocessed_5k")
        fname_processed = fname_pre.with_suffix(".zarr")

        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            msg = "Resampling IBCSO data to 5km resolution, this may take a while!"
            warnings.warn(msg, stacklevel=2)

            # load the full-res preprocessed grid
            grid = ibcso(layer=layer)

            # resample to 5km
            grid = resample_grid(grid, spacing=5e3)

            # Save to disk
            grid = grid.to_dataset(name=layer)
            grid.to_zarr(fname_processed)

        return str(fname_processed)

    # determine which resolution of preprocessed grid to use
    if spacing is None or spacing < 5000:
        preprocessor = preprocessing_fullres
    elif spacing >= 5000:
        logger.info("using preprocessed 5km grid since spacing is > 5km")
        preprocessor = preprocessing_5k
    else:
        msg = "spacing must be either None or a float greater than 0"
        raise ValueError(msg)

    if layer == "surface":
        path = pooch.retrieve(
            url="https://download.pangaea.de/dataset/937574/files/IBCSO_v2_ice-surface.nc",
            fname="IBCSO_v2_ice_surface.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
            known_hash="7748a79fffa41024c175cff7142066940b3e88f710eaf4080193c46b2b59e1f0",
            progressbar=True,
            processor=preprocessor,  # pylint: disable=possibly-used-before-assignment
        )
    elif layer == "bed":
        path = pooch.retrieve(
            url="https://download.pangaea.de/dataset/937574/files/IBCSO_v2_bed.nc",
            fname="IBCSO_v2_bed.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
            known_hash="74d55acb219deb87dc5be019d6dafeceb7b1ebcf9095866f257671d12670a5e2",
            progressbar=True,
            processor=preprocessor,  # pylint: disable=possibly-used-before-assignment
        )
    else:
        msg = "layer must be 'surface' or 'bed'"
        raise ValueError(msg)

    grid = xr.open_zarr(
        path,
        consolidated=False,
    )[layer]

    grid = resample_grid(
        grid,
        spacing=spacing,
        region=region,
        registration=registration,
        **kwargs,
    )

    if reference == "ellipsoid":
        logger.info("converting to be reference to the WGS84 ellipsoid")
        # get a grid of EIGEN geoid values matching IBCSO grid

        initial_spacing, initial_region, _, _, initial_registration = (
            utils.get_grid_info(grid)
        )
        eigen_correction = geoid(
            spacing=initial_spacing,
            region=initial_region,
            registration=initial_registration,
            hemisphere="south",
            **kwargs,
        )

        initial_registration_num = grid.gmt.registration

        # convert from geoidal heights to ellipsoidal heights
        grid = grid + eigen_correction

        # restore registration type
        grid.gmt.registration = initial_registration_num
    elif reference == "geoid":
        pass
    else:
        msg = "reference must be 'geoid' or 'ellipsoid'"
        raise ValueError(msg)

    return typing.cast(xr.DataArray, grid)


def bedmachine(
    layer: str,
    reference: str = "eigen-6c4",
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    hemisphere: str | None = None,
    **kwargs: typing.Any,
) -> xr.DataArray:
    """
    Load BedMachine topography data from either Greenland (v5) or Antarctica (v3),  from
    :footcite:t:`morlighemmeasures2022` or  :footcite:t:`icebridge2020`.

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
        'surface', 'ice_thickness'; 'icebase' will give results of surface-thickness
    reference : str
        choose whether heights are referenced to 'eigen-6c4' geoid or the
        'ellipsoid' (WGS84), by default is eigen-6c4'
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 500m. If spacing >=
        5000m, will resample the grid to 5km, and save it as a preprocessed grid, so
        future fetch calls are performed faster.
    registration : str, optional
        change registration with either 'p' for pixel or 'g' for gridline registration,
        by default is None.
    hemisphere : str, optional
        choose which hemisphere to retrieve data for, "north" or "south", by default
        None
    **kwargs : typing.Any
        additional keyword arguments to pass to the resample_grid function
    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of Bedmachine.

    References
    ----------
    .. footbibliography::
    """
    logger.debug("Loading Bedmachine data for %s", layer)

    hemisphere = utils.default_hemisphere(hemisphere)

    if layer == "thickness":
        layer = "ice_thickness"
        msg = "'thickness' is deprecated, use 'ice_thickness' instead"
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    # users use 'ice_thickness' but the dataset uses 'thickness'
    if layer == "ice_thickness":
        layer = "thickness"

    def preprocessing_fullres(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Load the .nc file and save it as a zarr"
        fname1 = pathlib.Path(fname)

        # Rename to the file
        if hemisphere == "south":
            fname_pre = fname1.with_stem(fname1.stem + "_antarctica")
        elif hemisphere == "north":
            fname_pre = fname1.with_stem(fname1.stem + "_greenland")
        else:
            msg = "hemisphere must be 'north' or 'south'"
            raise ValueError(msg)

        fname_processed = fname_pre.with_suffix(".zarr")

        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            # load grid
            grid = xr.load_dataset(fname1)

            # Save to .zarr file
            grid.to_zarr(
                fname_processed,
            )

        return str(fname_processed)

    def preprocessing_5k(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Load the .nc file, resample to 5km resolution, and save it as a zarr"
        fname1 = pathlib.Path(fname)

        # Rename to the file to ***_5k.zarr
        if hemisphere == "south":
            fname_pre = fname1.with_stem(fname1.stem + "_antarctica_5k")
        elif hemisphere == "north":
            fname_pre = fname1.with_stem(fname1.stem + "_greenland_5k")
        else:
            msg = "hemisphere must be 'north' or 'south'"
            raise ValueError(msg)

        fname_processed = fname_pre.with_suffix(".zarr")

        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            msg = "resampling this file to 5 km may take some time!"
            warnings.warn(msg, stacklevel=2)

            # load grid
            grid = xr.load_dataset(fname1)

            var_names = [
                "bed",
                "dataid",
                "errbed",
                "firn",
                "geoid",
                "mask",
                "source",
                "surface",
                "thickness",
            ]

            if hemisphere == "north":
                var_names.remove("firn")

            # resample each data variable to 5 km
            for _i, var in enumerate(
                tqdm(var_names, total=len(var_names), desc="dataset variables")
            ):
                da = resample_grid(
                    grid[var],
                    spacing=spacing,
                ).rename(var)
                # append to .zarr file
                da.to_zarr(
                    fname_processed,
                    mode="a",  # append to existing zarr
                )

        return str(fname_processed)

    if hemisphere == "north":
        url = (
            "https://n5eil01u.ecs.nsidc.org/ICEBRIDGE/IDBMG4.005/1993.01.01/"
            "BedMachineGreenland-v5.nc"
        )

        fname = "bedmachine_v5.nc"
        known_hash = "f7116b8e9e3840649075dcceb796ce98aaeeb5d279d15db489e6e7668e0d80db"

        # greenland dataset doesn't have firn layer
        if layer == "firn":
            msg = "firn layer not available for Greenland"
            raise ValueError(msg)

        if spacing is None:
            spacing = 150

    elif hemisphere == "south":
        url = (
            "https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0756.003/1970.01.01/"
            "BedMachineAntarctica-v3.nc"
        )
        fname = "bedmachine_v3.nc"
        known_hash = "d34390f585e61c4dba0cecd9e275afcc9586b377ba5ccc812e9a004566a9e159"

        if spacing is None:
            spacing = 500
    else:
        msg = "hemisphere must be 'north' or 'south'"
        raise ValueError(msg)

    # determine which resolution of preprocessed grid to use
    if spacing < 5000:
        preprocessor = preprocessing_fullres
    elif spacing >= 5000:
        logger.info("using preprocessed 5km grid since spacing is > 5km")
        preprocessor = preprocessing_5k
    else:
        msg = "spacing must be a float greater than 0"
        raise ValueError(msg)

    path = pooch.retrieve(
        url=url,
        fname=fname,
        path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
        downloader=EarthDataDownloader(),
        known_hash=known_hash,
        progressbar=True,
        processor=preprocessor,
    )
    ds = xr.open_zarr(
        path,
        consolidated=False,
    )

    # calculate icebase as surface-thickness
    if layer == "icebase":
        logger.info("calculating icebase from surface and thickness grids")
        grid = ds.surface - ds.thickness
        # restore registration type
        grid.gmt.registration = ds.surface.gmt.registration

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
        grid = ds[layer]
    else:
        msg = (
            "layer must be one of 'bed', 'dataid', 'errbed', 'firn', 'geoid', "
            "'mask', 'source', 'surface', 'thickness', or 'icebase'"
        )
        raise ValueError(msg)

    grid = resample_grid(
        grid,
        spacing=spacing,
        region=region,
        registration=registration,
        **kwargs,
    )

    # change layer elevation to be relative to different reference frames.
    if layer in ["surface", "icebase", "bed"]:
        if reference == "ellipsoid":
            logger.info("converting to be reference to the WGS84 ellipsoid")
            geoid_grid = ds["geoid"]
            geoid_grid = resample_grid(
                geoid_grid,
                spacing=spacing,
                region=region,
                registration=registration,
                **kwargs,
            )
            initial_registration_num = grid.gmt.registration

            # convert to the ellipsoid
            grid = grid + geoid_grid

            # restore registration type
            grid.gmt.registration = initial_registration_num

        elif reference == "eigen-6c4":
            pass
        else:
            msg = "reference must be 'eigen-6c4' or 'ellipsoid'"
            raise ValueError(msg)

    return typing.cast(xr.DataArray, grid)


def bedmap_points(
    version: str,
    region: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """
    Load bedmap point data, choose from Bedmap 1, 2 or 3 or all combined.

    All elevations are in meters above the WGS84 ellipsoid.

    version == 'bedmap1'
    from :footcite:t:`lythebedmap2001`.
    accessed from https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/01619

    version == 'bedmap2'
    from :footcite:t:`fretwellbedmap22013`.
    accessed from https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/01616

    version == 'bedmap3'
    from :footcite:t:`fremandbedmap32022`.
    accessed from  https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/01614#access-data
    download link was found from https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=61100714-1e32-44af-a237-0a517529bc49
    under DOI/BEDMAP3 datapoints, right click on the download link and copy link address

    Parameters
    ----------
    version : str
        choose between 'bedmap1', 'bedmap2', 'bedmap3', or 'all', point data
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

    # warn that pyarrow is faster
    if not USE_ARROW:
        msg = (
            "Consider installing pyarrow for faster performance when reading "
            "geodataframes."
        )
        warnings.warn(msg, stacklevel=2)

    if region is not None:
        bbox = utils.region_to_bounding_box(region)
    else:
        bbox = None
        msg = (
            "this file is large, if you only need a subset of data please provide "
            "a bounding box region via `region` to subset the data."
        )
        warnings.warn(msg, stacklevel=2)

    if version == "bedmap1":

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the csv and save as a geopackage"
            # name to give the processed file
            fname_processed = pathlib.Path(
                f"{pooch.os_cache('pooch')}/polartoolkit/topography/bedmap1_point_data.gpkg"
            )

            # Only perform if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                df = pd.read_csv(
                    fname,
                    skiprows=18,
                    na_values=[-9999],  # set additional nan value
                )

                # reproject from lat/lon to EPSG:3031
                df = utils.latlon_to_epsg3031(
                    df,
                    input_coord_names=(
                        "longitude (degree_east)",
                        "latitude (degree_north)",
                    ),
                )

                df["project"] = np.nan

                # convert to a geodataframe
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df["easting"], df["northing"]),
                    crs="EPSG:3031",
                )

                # save the dataframe
                gdf.to_file(
                    fname_processed,
                    driver="GPKG",
                    use_arrow=USE_ARROW,
                    engine="pyogrio",
                )

            return str(fname_processed)

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
            processor=preprocessing,
        )
        df = gpd.read_file(
            fname,
            use_arrow=USE_ARROW,
            engine="pyogrio",
            bbox=bbox,
        )

    elif version == "bedmap2":

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, load csvs into pandas dataframe and save as a geopackage"
            # name to give the processed file
            fname_processed = pathlib.Path(
                f"{pooch.os_cache('pooch')}/polartoolkit/topography/bedmap2_point_data/bedmap2_point_data.gpkg"
            )

            # Only unzip and merge csv's if new download or the processed file doesn't
            # exist yet
            if action in ("download", "update") or not fname_processed.exists():
                msg = (
                    "this file is large and will take some time to "
                    "download and preprocess!"
                )
                warnings.warn(msg, stacklevel=2)

                # extract the files and get list of csv paths
                path = pooch.Unzip(extract_dir="bedmap2_point_data")(
                    fname, action, _pooch2
                )

                # get folder name
                fold = str(pathlib.Path(path[0]).parent)

                # make new clean folder name
                new_fold = fold.replace("-", "")
                new_fold = new_fold.replace(",", "")
                new_fold = new_fold.replace(" ", "_")
                shutil.move(fold, new_fold)

                # get all csv files
                fnames = glob.glob(  # noqa: PTH207
                    f"{pooch.os_cache('pooch')}/polartoolkit/topography/bedmap2_point_data/*/*.csv"
                )

                # append all csv files into a gpkg file
                for i, f in enumerate(
                    tqdm(fnames, total=len(fnames), desc="csv files")
                ):
                    df = pd.read_csv(
                        f,
                        skiprows=18,  # metadata in first 18 rows
                        na_values=[-9999],  # set additional nan value
                        low_memory=False,
                    )
                    df["project"] = pathlib.Path(f).stem

                    # reproject from lat/lon to EPSG:3031
                    df = utils.latlon_to_epsg3031(
                        df,
                        input_coord_names=(
                            "longitude (degree_east)",
                            "latitude (degree_north)",
                        ),
                    )
                    # convert to a geodataframe
                    df = gpd.GeoDataFrame(
                        df,
                        geometry=gpd.points_from_xy(df["easting"], df["northing"]),
                        crs="EPSG:3031",
                    )

                    # need to use "string" instead of str to preserve NaNs
                    df["time_UTC"] = df.time_UTC.astype("string")
                    df["date"] = df.date.astype("string")
                    df["trajectory_id"] = df.trajectory_id.astype("string")

                    # save / append to a geopackage file
                    try:
                        df.to_file(
                            fname_processed,
                            driver="GPKG",
                            use_arrow=USE_ARROW,
                            engine="pyogrio",
                            append=True,
                            geometry_type="Point",
                        )
                    except Exception as e:
                        logger.exception(
                            "Error writing to geopackage for file number %s, deleting "
                            "geopackage file",
                            i,
                        )
                        # delete the file
                        pathlib.Path.unlink(fname_processed)
                        raise e

                # delete the folder with csv files
                shutil.rmtree(new_fold)

            return str(fname_processed)

        url = (
            "https://ramadda.data.bas.ac.uk/repository/entry/show/UK+Polar+Data+Centre/"
            "DOI/BEDMAP2+-+Ice+thickness%2C+bed+and+surface+elevation+for+Antarctica+-+"
            "standardised+data+points?entryid=2fd95199-365e-4da1-ae26-3b6d48b3e6ac&"
            "output=zip.zipgroup"
        )

        fname = pooch.retrieve(
            url=url,
            path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
            fname="bedmap2_point_data.zip",
            known_hash=None,
            progressbar=True,
            processor=preprocessing,
        )
        df = gpd.read_file(
            fname,
            use_arrow=USE_ARROW,
            engine="pyogrio",
            bbox=bbox,
        )

    elif version == "bedmap3":

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, load csvs into pandas dataframe and save as a geopackage"
            # name to give the processed file
            fname_processed = pathlib.Path(
                f"{pooch.os_cache('pooch')}/polartoolkit/topography/bedmap3_point_data/bedmap3_point_data.gpkg"
            )

            # Only unzip and merge csv's if new download or the processed file doesn't
            # exist yet
            if action in ("download", "update") or not fname_processed.exists():
                msg = (
                    "this file is large (14 Gb!) and will take some time to "
                    "download and preprocess!"
                )
                warnings.warn(msg, stacklevel=2)

                # extract the files and get list of csv paths
                path = pooch.Unzip(extract_dir="bedmap3_point_data")(
                    fname, action, _pooch2
                )

                # get folder name
                fold = str(pathlib.Path(path[0]).parent)

                # get all csv files
                fnames = glob.glob(  # noqa: PTH207
                    f"{pooch.os_cache('pooch')}/polartoolkit/topography/bedmap3_point_data/*/*.csv"
                )

                # append all csv files into a gpkg file
                for i, f in enumerate(
                    tqdm(fnames, total=len(fnames), desc="csv files")
                ):
                    df = pd.read_csv(
                        f,
                        skiprows=18,  # metadata in first 18 rows
                        na_values=[-9999],  # set additional nan value
                        low_memory=False,
                    )
                    df["project"] = pathlib.Path(f).stem

                    # reproject from lat/lon to EPSG:3031
                    df = utils.latlon_to_epsg3031(
                        df,
                        input_coord_names=(
                            "longitude (degree_east)",
                            "latitude (degree_north)",
                        ),
                    )

                    # convert to a geodataframe
                    df = gpd.GeoDataFrame(
                        df,
                        geometry=gpd.points_from_xy(df["easting"], df["northing"]),
                        crs="EPSG:3031",
                    )

                    df["time_UTC"] = df.time_UTC.astype("string")
                    df["date"] = df.date.astype("string")
                    df["trajectory_id"] = df.trajectory_id.astype("string")

                    # save / append to a geopackage file
                    try:
                        df.to_file(
                            fname_processed,
                            driver="GPKG",
                            use_arrow=USE_ARROW,
                            engine="pyogrio",
                            append=True,
                            geometry_type="Point",
                        )
                    except Exception as e:
                        logger.exception(
                            "Error writing to geopackage for file number %s, deleting "
                            "geopackage file",
                            i,
                        )
                        # delete the file
                        pathlib.Path.unlink(fname_processed)
                        raise e

                # delete the folder with csv files
                shutil.rmtree(fold)

            return str(fname_processed)

        url = (
            "https://ramadda.data.bas.ac.uk/repository/entry/show/UK+Polar+Data+Centre/"
            "DOI/BEDMAP3+-+Ice+thickness%2C+bed+and+surface+elevation+for+Antarctica+-+"
            "standardised+data+points?entryid=91523ff9-d621-46b3-87f7-ffb6efcd1847&"
            "output=zip.zipgroup"
        )

        fname = pooch.retrieve(
            url=url,
            path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
            fname="bedmap3_point_data.zip",
            # known_hash="c4661e1a8cee93164bb19d126e8fa1112a59f7579ff5e0d993704b5956621ef5",
            known_hash=None,
            progressbar=True,
            processor=preprocessing,
        )
        df = gpd.read_file(
            fname,
            use_arrow=USE_ARROW,
            engine="pyogrio",
            bbox=bbox,
        )

    elif version == "all":
        # get individual dataframes
        bedmap1_points = bedmap_points("bedmap1", region)
        bedmap2_points = bedmap_points("bedmap2", region)
        bedmap3_points = bedmap_points("bedmap3", region)

        # add new columns to identify the source
        bedmap1_points["bedmap_version"] = "bedmap1"
        bedmap2_points["bedmap_version"] = "bedmap2"
        bedmap3_points["bedmap_version"] = "bedmap3"

        df = pd.concat([bedmap1_points, bedmap2_points, bedmap3_points])
    else:
        msg = "version must be 'bedmap1', 'bedmap2', 'bedmap3' or 'all'"
        raise ValueError(msg)

    return df


def bedmap3(
    layer: str,
    reference: str = "eigen-gl04c",
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    fill_nans: bool = False,
    **kwargs: typing.Any,
) -> xr.DataArray:
    """
    Load Bedmap3 data as an xarray.DataArray
    from :footcite:t:`pritchardbedmap32025`.
    accessed from https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=2d0e4791-8e20-46a3-80e4-f5f6716025d2.

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
        "surface", "icebase", "bed", "ice_thickness", "water_thickness",
        "bed_uncertainty", "ice_thickness_uncertainty", and  "mask".
    reference : str
        choose whether heights are referenced to the 'eigen-6c4' geoid, the WGS84
        ellipsoid, 'ellipsoid', or by default the 'eigen-gl04c' geoid.
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 500m. If spacing >=
        5000m, will resample the grid to 5km, and save it as a preprocessed grid, so
        future fetch calls are performed faster.
    registration : str, optional,
        choose between 'g' (gridline) or 'p' (pixel) registration types, by default is
        the original type of the grid
    fill_nans : bool, optional,
        choose whether to fill nans in 'surface' and 'thickness' with 0. If converting
        to reference to the geoid, will fill nan's before conversion, by default is
        False
    **kwargs : optional
        additional keyword arguments to pass to the resample_grid function

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of Bedmap2.

    References
    ----------
    .. footbibliography::
    """
    logger.debug("Loading Bedmap3 data for %s", layer)

    # download url
    url = (
        "https://ramadda.data.bas.ac.uk/repository/entry/get/bedmap3.nc?entryid=synth%"
        "3A2d0e4791-8e20-46a3-80e4-f5f6716025d2%3AL2JlZG1hcDMubmM%3D"
    )
    known_hash = None
    # convert user-supplied strings to names used by Bedmap3
    if layer == "surface":
        layer = "surface_topography"
    if layer == "bed":
        layer = "bed_topography"
    if layer == "ice_thickness_uncertainty":
        layer = "thickness_uncertainty"

    valid_variables = [
        "surface_topography",
        "bed_uncertainty",
        "bed_topography",
        "mask",
        "ice_thickness",
        "thickness_uncertainty",
        # "mapping",
    ]

    def preprocessing_fullres(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Save each layer to individual .zarr file"
        # get the path to the nc file
        fname1 = pathlib.Path(fname)

        # check if all layers have already been processed in to Zarr files
        exists = []
        for lyr in valid_variables:
            fname_processed = (
                pathlib.Path(fname1).with_stem(f"bedmap3_{lyr}").with_suffix(".zarr")
            )
            if action in ("download", "update"):
                exists.append(False)
            elif fname_processed.exists():
                exists.append(True)
            else:
                exists.append(False)

        if all(exists):
            return str(
                pathlib.Path(fname1).with_stem(f"bedmap3_{layer}").with_suffix(".zarr")
            )

        msg = (
            "Preprocessing Bedmap3 data to gridline registration, this may take a "
            "while!"
        )
        warnings.warn(msg, stacklevel=2)
        # go through each layer, update to gridline registration and save to a zarr file
        for lyr in valid_variables:
            with xr.open_dataset(fname1) as ds:
                # Rename to the file to ***.zarr
                fname_processed = (
                    pathlib.Path(fname1)
                    .with_stem(f"bedmap3_{lyr}")
                    .with_suffix(".zarr")
                )
                if fname_processed.exists():
                    continue
                logger.info("Processing %s to %s", lyr, fname_processed)

                grid = ds[lyr]
                # get min max before, and use as limits after resampling
                min_val, max_val = utils.get_min_max(grid)

                # resample to be pixel registered to match most grids in PolarToolkit
                grid = pygmt.grdsample(
                    grid=grid,
                    region=(-3333000.0, 3333000.0, -3333000.0, 3333000.0),
                    spacing=500,
                    registration="g",
                )

                # restore the min max values
                grid = xr.where(grid < min_val, min_val, grid)
                grid = xr.where(grid > max_val, max_val, grid)

                new_min, new_max = utils.get_min_max(grid)
                assert new_min >= min_val, (
                    f"New min value {new_min} is less than the original min value "
                    f"{min_val}"
                )
                assert new_max <= max_val, (
                    f"New max value {new_max} is greater than the original max value "
                    f"{max_val}"
                )
                # Save to disk
                grid.rename("z").to_zarr(fname_processed)

        return str(
            pathlib.Path(fname1).with_stem(f"bedmap3_{layer}").with_suffix(".zarr")
        )

    def preprocessing_5k(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Load preprocessed full-res grid, resample to 5km and save to .zarr file"

        # get the path to the .nc file
        fname1 = pathlib.Path(fname)

        # add _5k to .zarr file name
        fname_processed = (
            pathlib.Path(fname1).with_stem(f"bedmap3_{layer}_5k").with_suffix(".zarr")
        )

        # Only recalculate if new download or the processed file doesn't exist yet
        if action in ("download", "update") or not fname_processed.exists():
            msg = "Resampling Bedmap3 data to 5km resolution, this may take a while!"
            warnings.warn(msg, stacklevel=2)

            # load the full-res preprocessed grid
            grid = bedmap3(layer=layer)

            # resample to 5km
            grid = resample_grid(grid, spacing=5e3)

            # Save to disk
            grid.to_zarr(fname_processed)

        return str(fname_processed)

    # determine which resolution of preprocessed grid to use
    if spacing is None or spacing < 5000:
        preprocessor = preprocessing_fullres
    elif spacing >= 5000:
        logger.info("using preprocessed 5km grid since spacing is > 5km")
        preprocessor = preprocessing_5k
    else:
        msg = "spacing must be either None or a float greater than 0"
        raise ValueError(msg)

    # calculate icebase as surface-thickness
    if layer == "icebase":
        logger.info("calculating icebase from surface and thickness grids")
        surface = bedmap3(layer="surface", spacing=spacing)
        thickness = bedmap3(layer="ice_thickness", spacing=spacing)
        # calculate icebase
        grid = surface - thickness
        # restore registration type
        grid.gmt.registration = surface.gmt.registration
    elif layer == "water_thickness":
        logger.info("calculating water thickness from bed and icebase grids")
        icebase = bedmap3(layer="icebase", spacing=spacing)
        bed = bedmap3(layer="bed", spacing=spacing)
        # calculate water thickness
        grid = icebase - bed
        # ensure no negative values
        grid = xr.where(grid < 0, 0, grid)
        # restore registration type
        grid.gmt.registration = bed.gmt.registration
    elif layer in valid_variables:
        fname = pooch.retrieve(
            url=url,
            fname="bedmap3.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
            known_hash=known_hash,
            processor=preprocessor,
            progressbar=True,
        )

        try:
            # load zarr as a dataarray
            grid = xr.open_zarr(
                fname,
                consolidated=False,
            ).z
        except AttributeError as e:
            msg = (
                "The preprocessing steps for Bedmap3 have been changed but the old data"
                " is still on your disk. Please delete the Bedmap3 grids files from "
                "your polartoolkit cache directory. This cache folder can be found "
                "with the python command: import pooch; print(pooch.os_cache('pooch'))."
            )
            raise ValueError(msg) from e

    else:
        msg = (
            "layer must be one of 'surface_topography', 'bed_uncertainty', "
            "'bed_topography', 'mask', 'ice_thickness', "
            "'thickness_uncertainty', or 'icebase' or 'water_thickness'"
        )
        raise ValueError(msg)

    grid = resample_grid(
        grid,
        spacing=spacing,
        region=region,
        registration=registration,
        **kwargs,
    )

    # replace nans with 0's in surface, thickness or icebase grids
    if fill_nans is True and layer in [
        "surface_topography",
        "ice_thickness",
        "icebase",
    ]:
        # pygmt.grdfill(final_grid, mode='c0') # doesn't work, maybe grid is too big
        # this changes the registration from pixel to gridline
        registration_num = grid.gmt.registration
        grid = grid.fillna(0)
        # restore registration type
        grid.gmt.registration = registration_num

    # change layer elevation to be relative to different reference frames.
    if layer in ["surface_topography", "icebase", "bed_topography"]:
        if reference not in ["ellipsoid", "eigen-6c4", "eigen-gl04c"]:
            msg = "reference must be 'eigen-gl04c', 'eigen-6c4' or 'ellipsoid'"
            raise ValueError(msg)
        spacing, region, _, _, registration = utils.get_grid_info(grid)
        if reference == "ellipsoid":
            logger.info("converting to be referenced to the WGS84 ellipsoid")
            # load bedmap2 grid for converting to wgs84
            geoid_2_ellipsoid = bedmap2(
                layer="gl04c_geiod_to_WGS84",
                region=region,
                spacing=spacing,
                registration=registration,
            )
            registration_num = grid.gmt.registration
            # convert to the ellipsoid
            grid = grid + geoid_2_ellipsoid
            # restore registration type
            grid.gmt.registration = registration_num
        elif reference == "eigen-6c4":
            logger.info("converting to be referenced to the EIGEN-6C4")
            # load bedmap2 grid for converting to wgs84
            geoid_2_ellipsoid = bedmap2(
                layer="gl04c_geiod_to_WGS84",
                region=region,
                spacing=spacing,
                registration=registration,
            )
            registration_num = grid.gmt.registration
            # convert to the ellipsoid
            grid = grid + geoid_2_ellipsoid
            # restore registration type
            grid.gmt.registration = registration_num
            # get a grid of EIGEN geoid values matching the user's input
            eigen_correction = geoid(
                spacing=spacing,
                region=region,
                registration=registration,
                hemisphere="south",
                **kwargs,
            )
            # convert from ellipsoid back to eigen geoid
            grid = grid - eigen_correction
            # restore registration type
            grid.gmt.registration = registration_num
        elif reference == "eigen-gl04c":
            pass

    return typing.cast(xr.DataArray, grid)


def bedmap2(
    layer: str,
    reference: str = "eigen-gl04c",
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    fill_nans: bool = False,
    **kwargs: typing.Any,
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
        "lakemask_vostok", "rockmask", "surface", "ice_thickness",
        "ice_thickness_uncertainty", "gl04c_geiod_to_WGS84", "icebase",
        "water_thickness"
    reference : str
        choose whether heights are referenced to the 'eigen-6c4' geoid, the WGS84
        ellipsoid, 'ellipsoid', or by default the 'eigen-gl04c' geoid.
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 1000m. If spacing >=
        5000m, will resample the grid to 5km, and save it as a preprocessed grid, so
        future fetch calls are performed faster.
    registration : str, optional,
        choose between 'g' (gridline) or 'p' (pixel) registration types, by default is
        the original type of the grid
    fill_nans : bool, optional,
        choose whether to fill nans in 'surface' and 'thickness' with 0. If converting
        to reference to the geoid, will fill nan's before conversion, by default is
        False
    **kwargs : optional
        additional keyword arguments to pass to the resample_grid function

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of Bedmap2.

    References
    ----------
    .. footbibliography::
    """
    logger.debug("Loading Bedmap2 data for %s", layer)

    if layer == "thickness":
        layer = "ice_thickness"
        msg = "'thickness' is deprecated, use 'ice_thickness' instead"
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    if layer == "thickness_uncertainty_5km":
        layer = "ice_thickness_uncertainty"
        msg = (
            "'thickness_uncertainty_5km' is deprecated, use "
            "'ice_thickness_uncertainty' instead"
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    # download url
    url = (
        "https://ramadda.data.bas.ac.uk/repository/entry/show/Polar+Data+Centre/"
        "DOI/BEDMAP2+-+Ice+thickness%2C+bed+and+surface+elevation+for+Antarctica"
        "+-+gridding+products/bedmap2_tiff?entryid=synth%3Afa5d606c-dc95-47ee-9016"
        "-7a82e446f2f2%3AL2JlZG1hcDJfdGlmZg%3D%3D&output=zip.zipgroup"
    )
    known_hash = None
    if layer not in [
        "lakemask_vostok",
        "ice_thickness_uncertainty",
        "bed",
        "coverage",
        "grounded_bed_uncertainty",
        "icemask_grounded_and_shelves",
        "rockmask",
        "surface",
        "ice_thickness",
        "gl04c_geiod_to_WGS84",
        "icebase",
        "water_thickness",
    ]:
        msg = (
            "layer must be one of 'bed', 'coverage', 'grounded_bed_uncertainty', "
            "'icemask_grounded_and_shelves', 'lakemask_vostok', 'rockmask', "
            "'surface', 'ice_thickness', 'ice_thickness_uncertainty', "
            "'gl04c_geiod_to_WGS84', 'icebase', or 'water_thickness'"
        )
        raise ValueError(msg)

    def preprocessing_fullres(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Unzip the folder, convert the tiffs to .zarr files"
        # extract each layer to it's own folder
        if layer == "gl04c_geiod_to_WGS84":
            member = ["bedmap2_tiff/gl04c_geiod_to_WGS84.tif"]
        elif layer == "ice_thickness":
            member = ["bedmap2_tiff/bedmap2_thickness.tif"]
        elif layer == "ice_thickness_uncertainty":
            member = ["bedmap2_tiff/bedmap2_thickness_uncertainty_5km.tif"]
        else:
            member = [f"bedmap2_tiff/bedmap2_{layer}.tif"]
        fname1 = pooch.Unzip(
            extract_dir=f"bedmap2_{layer}",
            members=member,
        )(fname, action, _pooch2)[0]
        # get the path to the layer's tif file
        fname2 = pathlib.Path(fname1)

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
            grid = grid.to_dataset(name="z")

            # Save to disk
            grid.to_zarr(
                fname_processed,
            )

        return str(fname_processed)

    def preprocessing_5k(fname: str, action: str, _pooch2: typing.Any) -> str:
        """
        Unzip the folder, load the tiffs, resample to 5km resolution, and save as .zarr
        files
        """
        # extract each layer to it's own folder
        if layer == "gl04c_geiod_to_WGS84":
            member = ["bedmap2_tiff/gl04c_geiod_to_WGS84.tif"]
        elif layer == "ice_thickness":
            member = ["bedmap2_tiff/bedmap2_thickness.tif"]
        elif layer == "ice_thickness_uncertainty":
            member = ["bedmap2_tiff/bedmap2_thickness_uncertainty_5km.tif"]
        else:
            member = [f"bedmap2_tiff/bedmap2_{layer}.tif"]
        fname1 = pooch.Unzip(
            extract_dir=f"bedmap2_{layer}",
            members=member,
        )(fname, action, _pooch2)[0]
        # get the path to the layer's tif file
        fname2 = pathlib.Path(fname1)

        # add _5k to filename
        fname_pre = fname2.with_stem(fname2.stem + "_5k")

        # Rename to the file to ***.zarr
        fname_processed = fname_pre.with_suffix(".zarr")

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
            # resampe to 5km
            grid = resample_grid(
                grid,
                spacing=5e3,
            )

            grid = grid.to_dataset(name="z")

            # Save to disk
            grid.to_zarr(
                fname_processed,
            )

        return str(fname_processed)

    # determine which resolution of preprocessed grid to use
    if spacing is None or spacing < 5000:
        preprocessor = preprocessing_fullres
    elif spacing >= 5000:
        logger.info("using preprocessed 5km grid since spacing is > 5km")
        preprocessor = preprocessing_5k
    else:
        msg = "spacing must be either None or a float greater than 0"
        raise ValueError(msg)

    # calculate icebase as surface-ice_thickness
    if layer == "icebase":
        logger.info("calculating icebase from surface and ice thickness grids")
        surface = bedmap2(layer="surface", spacing=spacing)
        ice_thickness = bedmap2(layer="ice_thickness", spacing=spacing)
        # calculate icebase
        grid = surface - ice_thickness
        # restore registration type
        grid.gmt.registration = surface.gmt.registration
    elif layer == "water_thickness":
        logger.info("calculating water thickness from bed and icebase grids")
        icebase = bedmap2(layer="icebase", spacing=spacing)
        bed = bedmap2(layer="bed", spacing=spacing)
        # calculate water thickness
        grid = icebase - bed
        # restore registration type
        grid.gmt.registration = bed.gmt.registration
    elif layer in [
        "bed",
        "coverage",
        "grounded_bed_uncertainty",
        "icemask_grounded_and_shelves",
        "lakemask_vostok",
        "rockmask",
        "surface",
        "ice_thickness",
        "ice_thickness_uncertainty",
        "gl04c_geiod_to_WGS84",
    ]:
        # download/unzip all files, retrieve the specified layer file and convert to
        # .zarr
        fname = pooch.retrieve(
            url=url,
            fname="bedmap2_tiff.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/topography",
            known_hash=known_hash,
            processor=preprocessor,
            progressbar=True,
        )
        try:
            # load zarr as a dataarray
            grid = xr.open_zarr(
                fname,
                consolidated=False,
            ).z
        except AttributeError as e:
            msg = (
                "The preprocessing steps for Bedmap2 have been changed but the old data"
                " is still on your disk. Please delete the Bedmap2 grids files from "
                "your polartoolkit cache directory. This cache folder can be found "
                "with the python command: import pooch; print(pooch.os_cache('pooch'))."
            )
            raise ValueError(msg) from e
    else:
        msg = (
            "layer must be one of 'bed', 'coverage', 'grounded_bed_uncertainty', "
            "'icemask_grounded_and_shelves', 'lakemask_vostok', 'rockmask', "
            "'surface', 'ice_thickness', 'ice_thickness_uncertainty', "
            "'gl04c_geiod_to_WGS84', 'icebase', or 'water_thickness'"
        )
        raise ValueError(msg)

    grid = resample_grid(
        grid,
        spacing=spacing,
        region=region,
        registration=registration,
        **kwargs,
    )

    # replace nans with 0's in surface, ice thickness or icebase grids
    if fill_nans is True and layer in ["surface", "ice_thickness", "icebase"]:
        # pygmt.grdfill(final_grid, mode='c0') # doesn't work, maybe grid is too big
        # this changes the registration from pixel to gridline
        registration_num = grid.gmt.registration
        grid = grid.fillna(0)
        # restore registration type
        grid.gmt.registration = registration_num

    # change layer elevation to be relative to different reference frames.
    if layer in ["surface", "icebase", "bed"]:
        if reference not in ["ellipsoid", "eigen-6c4", "eigen-gl04c"]:
            msg = "reference must be 'eigen-gl04c', 'eigen-6c4' or 'ellipsoid'"
            raise ValueError(msg)
        spacing, region, _, _, registration = utils.get_grid_info(grid)
        if reference == "ellipsoid":
            logger.info("converting to be referenced to the WGS84 ellipsoid")
            # load bedmap2 grid for converting to wgs84
            geoid_2_ellipsoid = bedmap2(
                layer="gl04c_geiod_to_WGS84",
                region=region,
                spacing=spacing,
                registration=registration,
            )
            registration_num = grid.gmt.registration
            # convert to the ellipsoid
            grid = grid + geoid_2_ellipsoid
            # restore registration type
            grid.gmt.registration = registration_num
        elif reference == "eigen-6c4":
            logger.info("converting to be referenced to the EIGEN-6C4")
            # load bedmap2 grid for converting to wgs84
            geoid_2_ellipsoid = bedmap2(
                layer="gl04c_geiod_to_WGS84",
                region=region,
                spacing=spacing,
                registration=registration,
            )
            registration_num = grid.gmt.registration
            # convert to the ellipsoid
            grid = grid + geoid_2_ellipsoid
            # restore registration type
            grid.gmt.registration = registration_num
            # get a grid of EIGEN geoid values matching the user's input
            eigen_correction = geoid(
                spacing=spacing,
                region=region,
                registration=registration,
                hemisphere="south",
                **kwargs,
            )
            # convert from ellipsoid back to eigen geoid
            grid = grid - eigen_correction
            # restore registration type
            grid.gmt.registration = registration_num
        elif reference == "eigen-gl04c":
            pass

    return typing.cast(xr.DataArray, grid)


def rema(
    version: str = "1km",
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
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
    **kwargs : optional
        additional keyword arguments to pass to the resample_grid function
    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of the REMA DEM.

    References
    ----------
    .. footbibliography::
    """

    if version == "500m":
        # url and file name for download
        url = (
            "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/500m/rema_mosaic_"
            "500m_v2.0_filled_cop30.tar.gz"
        )
        fname = "rema_mosaic_500m_v2.0_filled_cop30.tar.gz"
        members = ["rema_mosaic_500m_v2.0_filled_cop30_dem.tif"]
        known_hash = "50ab9424f787aa76b5d55e694094f0c6c945144d8f6c8a26b5c5474ff8e29ddc"
    elif version == "1km":
        # url and file name for download
        url = (
            "https://data.pgc.umn.edu/elev/dem/setsm/REMA/mosaic/v2.0/1km/rema_mosaic_"
            "1km_v2.0_filled_cop30.tar.gz"
        )
        fname = "rema_mosaic_1km_v2.0_filled_cop30.tar.gz"
        members = ["rema_mosaic_1km_v2.0_filled_cop30_dem.tif"]
        known_hash = "6e39923c7beabe5a7b1942a06aba9fc632e6e6672fee9823ac3ba108086559b7"
    else:
        msg = "version must be '1km' or '500m'"
        raise ValueError(msg)

    def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Untar the folder, convert the tiffs to compressed .zarr files"
        # extract the files and get the surface grid
        path = pooch.Untar(members=members)(fname, action, _pooch2)[0]
        # fname = [p for p in path if p.endswith("dem.tif")]#[0]
        tiff_file = pathlib.Path(path)
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
                ds.to_zarr(
                    fname_processed,
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
    grid = xr.open_zarr(
        zarr_file,
        consolidated=False,
    )["surface"]

    resampled = resample_grid(
        grid,
        spacing=spacing,
        region=region,
        registration=registration,
        **kwargs,
    )

    return typing.cast(xr.DataArray, resampled)


def deepbedmap(
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
) -> xr.DataArray:
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
    **kwargs : optional
        additional keyword arguments to pass to the resample_grid function
    Returns
    -------
    xarray.DataArray:
        Returns the grid of DeepBedMap.

    References
    ----------
    .. footbibliography::
    """

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
        spacing=spacing,
        region=region,
        registration=registration,
        **kwargs,
    )


def gravity(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    hemisphere: str | None = None,
    **kwargs: typing.Any,
) -> xr.DataArray:
    """
    Loads gravity anomaly data for the Arctic and Antarctic.

    version='antgg'
    Antarctic-wide gravity data compilation of ground-based, airborne, and shipborne
    data, from :footcite:t:`scheinertnew2016`.
    Accessed from https://doi.pangaea.de/10.1594/PANGAEA.848168
    Anomalies are at the ice surface, or bedrock surface in areas of no ice. These
    surfaces are defined by Bedmap2 and are relative to the ellipsoid.

    version='antgg-2021'
    Updates on 2016 AntGG compilation.
    Accessed from https://doi.pangaea.de/10.1594/PANGAEA.971238?format=html#download
    Anomalies are at the ice surface, or bedrock surface in areas of no ice. These
    surfaces are defined by Bedmap2 and are relative to the ellipsoid.

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
    hemisphere : str, optional
        choose which hemisphere to retrieve data for, "north" or "south", by default
        None
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
        path = pooch.retrieve(
            url="https://hs.pangaea.de/Maps/antgg2015/antgg2015.nc",
            fname="antgg.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/gravity",
            known_hash="ad94d16f7e4895c5a09bbf9ca9d64750f1edd9b01f2bff21ca49e10b6fe47d72",
            progressbar=True,
        )

        file = xr.load_dataset(path)
        # convert coordinates from km to m
        file["x"] = file.x * 1000
        file["y"] = file.y * 1000

        # drop some variables
        file = file.drop_vars(
            [
                "longitude",
                "latitude",
                "crs",
            ]
        )

        resampled_vars = []
        for var in file.data_vars:
            resampled_vars.append(
                resample_grid(
                    file[var],
                    spacing,
                    region,
                    registration,
                    **kwargs,
                ).rename(var)
            )

        resampled = xr.merge(resampled_vars)

        # rename variables to match antgg
        resampled = resampled.rename(
            {
                "accuracy_measure": "error",
            }
        )

    elif version == "antgg-2021":
        # Free-air anomaly at the surface
        if anomaly_type == "FA":
            url = "https://download.pangaea.de/dataset/971238/files/AntGG2021_Gravity-anomaly.nc"
            fname = "antgg_2021_FA.nc"
        # Disturbance at the surface
        elif anomaly_type == "DG":
            url = "https://download.pangaea.de/dataset/971238/files/AntGG2021_Gravity_disturbance_at-surface.nc"
            fname = "antgg_2021_DG.nc"
        # Bouguer anomaly
        elif anomaly_type == "BA":
            url = "https://download.pangaea.de/dataset/971238/files/AntGG2021_Bouguer-anomaly.nc"
            fname = "antgg_2021_BA.nc"
        elif anomaly_type == "Err":
            url = "https://download.pangaea.de/dataset/971238/files/AntGG2021_Standard-deviation_GA-from-LSC.nc"
            fname = "antgg_2021_Err.nc"
        else:
            msg = "anomaly_type must be 'FA', 'BA', 'DG' or 'Err'"
            raise ValueError(msg)

        path = pooch.retrieve(
            url=url,
            fname=fname,
            path=f"{pooch.os_cache('pooch')}/polartoolkit/gravity",
            known_hash=None,
            progressbar=True,
        )

        file = xr.load_dataset(path)

        resampled_vars = []
        for var in file.data_vars:
            resampled_vars.append(
                resample_grid(
                    file[var],
                    spacing,
                    region,
                    registration,
                    **kwargs,
                ).rename(var)
            )

        resampled = xr.merge(resampled_vars)

        if "h_ellips" in resampled.data_vars:
            resampled = resampled.rename({"h_ellips": "h_ell"})

        # rename variables to match antgg
        to_rename = [
            {"h_ell": "ellipsoidal_height"},
            {"grav_anom": "free_air_anomaly"},
            {"Boug_anom": "bouguer_anomaly"},
            {"grav_dist": "gravity_disturbance"},
            {"std_grav_anom": "error"},
        ]
        for i in to_rename:
            try:  # noqa: SIM105
                resampled = resampled.rename(i)
            except ValueError:
                pass

    elif version == "eigen":
        hemisphere = utils.default_hemisphere(hemisphere)

        if hemisphere == "south":
            proj = "EPSG:3031"
        elif hemisphere == "north":
            proj = "EPSG:3413"
        else:
            msg = "hemisphere must be 'north' or 'south'"
            raise ValueError(msg)

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the .nc file, reproject, and save it as a zarr"
            fname1 = pathlib.Path(fname)

            # Rename to the file to ***_preprocessed.nc
            if hemisphere == "south":
                fname_pre = fname1.with_stem(fname1.stem + "_epsg3031_preprocessed")
            elif hemisphere == "north":
                fname_pre = fname1.with_stem(fname1.stem + "_epsg3413_preprocessed")
            else:
                msg = "hemisphere must be 'north' or 'south'"
                raise ValueError(msg)
            fname_processed = fname_pre.with_suffix(".zarr")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load grid
                grid = xr.load_dataset(fname1).gravity

                # reproject to polar stereographic
                grid = pygmt.grdproject(
                    grid,
                    projection=proj,
                    spacing=5e3,
                )
                # get just antarctica region
                grid = pygmt.grdsample(
                    grid,
                    region=(-3500000.0, 3500000.0, -3500000.0, 3500000.0),
                    spacing=5e3,
                    verbose=kwargs.get("verbose", "e"),
                ).rename("gravity")

                # add ellipsoidal height of observations
                ellipsoidal_height = xr.ones_like(grid) * 10e3
                grid = xr.merge([grid, ellipsoidal_height.rename("ellipsoidal_height")])

                # Save to .zarr file
                grid.to_zarr(
                    fname_processed,
                )

            return str(fname_processed)

        path = pooch.retrieve(
            url="doi:10.5281/zenodo.5882207/earth-gravity-10arcmin.nc",
            fname="eigen.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/gravity",
            known_hash="d55134501da0d984f318c0f92e1a15a8472176ec7babde5edfdb58855190273e",
            progressbar=True,
            processor=preprocessing,
        )

        try:
            # load zarr as a dataset
            grid = xr.open_zarr(
                path,
                consolidated=False,
            )
        except AttributeError as e:
            msg = (
                "The preprocessing steps for EIGEN gravity have been changed but the "
                "old data is still on your disk. Please delete the EIGEN grids files "
                "from your polartoolkit cache directory. This cache folder can be "
                "found with the python command: import pooch; "
                "print(pooch.os_cache('pooch'))."
            )
            raise ValueError(msg) from e

        resampled_vars = []
        for var in grid.data_vars:
            resampled_vars.append(
                resample_grid(
                    grid[var],
                    spacing=spacing,
                    region=region,
                    registration=registration,
                    **kwargs,
                ).rename(var)
            )
        resampled = xr.merge(resampled_vars)

    else:
        msg = "version must be 'antgg', 'antgg-2021' or 'eigen'"
        raise ValueError(msg)

    return resampled  # typing.cast(xr.Dataset, resampled)


def etopo(
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    hemisphere: str | None = None,
    **kwargs: typing.Any,
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
    **kwargs : optional
        additional keyword arguments to pass to the resample_grid function

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of topography.

    References
    ----------
    .. footbibliography::
    """

    hemisphere = utils.default_hemisphere(hemisphere)

    if hemisphere == "south":
        proj = "EPSG:3031"
        fname = "etopo_south.nc"
    elif hemisphere == "north":
        proj = "EPSG:3413"
        fname = "etopo_north.nc"

    def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Load the .nc file, reproject, and save it back"
        fname1 = pathlib.Path(fname)
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
                spacing=5e3,
            )
            # get just needed region
            processed = pygmt.grdsample(
                grid2,
                region=(-3500000.0, 3500000.0, -3500000.0, 3500000.0),
                spacing=5e3,
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
        spacing=spacing,
        region=region,
        registration=registration,
        **kwargs,
    )

    return typing.cast(xr.DataArray, resampled)


def geoid(
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    hemisphere: str | None = None,
    **kwargs: typing.Any,
) -> xr.DataArray:
    """
    Loads a grid of Antarctic geoid heights derived from the EIGEN-6C4 from
    :footcite:t:`forsteeigen6c42014` spherical harmonic model of Earth's gravity field.
    Originally at 10 arc-min resolution.
    Negative values indicate the geoid is below the ellipsoid surface and vice-versa.
    To convert a topographic grid which is referenced to the ellipsoid to be referenced
    to the geoid, add this grid.
    To convert a topographic grid which is referenced to the geoid to be referencde to
    the ellipsoid, add this grid.

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
    kwargs : typing.Any
        additional kwargs to pass to resample_grid.

    Returns
    -------
    xarray.DataArray
        Returns a loaded, and optional clip/resampled grid of geoid height.

    References
    ----------
    .. footbibliography::
    """

    hemisphere = utils.default_hemisphere(hemisphere)

    if hemisphere == "south":
        proj = "EPSG:3031"
        fname = "eigen_geoid_south.nc"
    elif hemisphere == "north":
        proj = "EPSG:3413"
        fname = "eigen_geoid_north.nc"

    def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
        "Load the .nc file, reproject, and save it back"
        fname1 = pathlib.Path(fname)
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
                spacing=5e3,
                verbose=kwargs.get("verbose", "e"),
            )
            # get just needed region
            processed = pygmt.grdsample(
                grid2,
                region=(-3500000.0, 3500000.0, -3500000.0, 3500000.0),
                spacing=5e3,
                verbose=kwargs.get("verbose", "e"),
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
        spacing=spacing,
        region=region,
        registration=registration,
        **kwargs,
    )

    return typing.cast(xr.DataArray, resampled)


def magnetics(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    hemisphere: str | None = None,
    **kwargs: typing.Any,
) -> xr.DataArray | None:
    """
    Load magnetic anomaly data for the Arctic and Antarctic.
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

    version='LCS-1'
    Satellite-only derived magnetic anomaly at Earth's surface (WGS84 ellipsoid) for
    spherical harmonic degrees n=14-185
    Accessed from https://www.spacecenter.dk/files/magnetic-models/LCS-1/

    Parameters
    ----------
    version : str
        Either 'admap1', 'admap2', 'admap2_gdb' or 'LCS-1'.
    region : tuple[float, float, float, float], optional
        region to clip the loaded grid to, in format [xmin, xmax, ymin, ymax], by
        default doesn't clip
    spacing : str or int, optional
        grid spacing to resample the loaded grid to, by default 10e3
    registration : str, optional,
        choose between 'g' (gridline) or 'p' (pixel) registration types, by default is
        the original type of the grid
    hemisphere : str, optional
        choose which hemisphere to retrieve data for, "north" or "south", by default
        None
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

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, grid the .dat file, and save it back as a .nc"
            path = pooch.Unzip(
                extract_dir="admap1",
            )(fname, action, _pooch2)
            fname1 = next(p for p in path if p.endswith(".dat"))
            fname2 = pathlib.Path(fname1)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname2.with_stem(fname2.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".nc")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load data
                df = pd.read_csv(
                    fname1,
                    sep=r"\s+",
                    header=None,
                    names=["lat", "lon", "nT"],
                )

                # re-project to polar stereographic
                df = utils.reproject(
                    df,
                    input_crs="epsg:4326",
                    output_crs="epsg:3031",
                    input_coord_names=("lon", "lat"),
                    output_coord_names=("x", "y"),
                )

                # block-median and grid the data
                df = pygmt.blockmedian(
                    df[["x", "y", "nT"]],  # type: ignore[call-overload]
                    spacing=5e3,
                    region=(-3330000.0, 3330000.0, -3330000.0, 3330000.0),
                    registration="g",
                )
                processed = pygmt.surface(
                    data=df[["x", "y", "nT"]],
                    spacing=5e3,
                    region=(-3330000.0, 3330000.0, -3330000.0, 3330000.0),
                    registration="g",
                    maxradius="1c",
                )
                # Save to disk
                processed.to_netcdf(fname_processed)

                logger.info(".dat file gridded and saved as %s", fname_processed)

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
            spacing,
            region,
            registration,
            **kwargs,
        )

    elif version == "admap2":

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "convert geosoft grd to xarray dataarray and save it back as a .nc"
            fname1 = pathlib.Path(fname)

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

    elif version == "LCS-1":
        hemisphere = utils.default_hemisphere(hemisphere)
        if hemisphere == "south":
            proj = "EPSG:3031"
        elif hemisphere == "north":
            proj = "EPSG:3413"
        else:
            msg = "hemisphere must be 'north' or 'south'"
            raise ValueError(msg)

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, grid the data, and save it back as a .zarr"
            path = pooch.Unzip()(fname, action, _pooch2)
            fname1 = pathlib.Path(path[0])

            # Rename to the file to ***_preprocessed.nc
            if hemisphere == "south":
                fname_pre = fname1.with_stem(fname1.stem + "epsg3031_preprocessed")
            elif hemisphere == "north":
                fname_pre = fname1.with_stem(fname1.stem + "epsg3413_preprocessed")
            else:
                msg = "hemisphere must be 'north' or 'south'"
                raise ValueError(msg)
            fname_processed = fname_pre.with_suffix(".zarr")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load data
                df = pd.read_csv(
                    fname1,
                    header=None,
                    skiprows=6,
                    sep=r"\s+",
                    names=["lon", "lat", "nT"],
                )

                # re-project to polar stereographic
                df = utils.reproject(
                    df,
                    input_crs="epsg:4326",
                    output_crs=proj,
                    input_coord_names=("lon", "lat"),
                    output_coord_names=("x", "y"),
                )

                # block-median and grid the data
                df = pygmt.blockmedian(
                    df[["x", "y", "nT"]],  # type: ignore[call-overload]
                    spacing=10e3,  # .25 degree resolution, ~20 km,
                    region=(-3500000.0, 3500000.0, -3500000.0, 3500000.0),
                    registration="g",
                )
                processed = pygmt.surface(
                    data=df[["x", "y", "nT"]],
                    spacing=10e3,  # .25 degree resolution, ~20 km,
                    region=(-3500000.0, 3500000.0, -3500000.0, 3500000.0),
                    registration="g",
                    maxradius="1c",
                )
                # Save to disk
                processed = processed.to_dataset(name="mag")
                processed.to_zarr(fname_processed)
            return str(fname_processed)

        url = "https://www.spacecenter.dk/files/magnetic-models/LCS-1/F_LCS-1_ellipsoid_14-185_ASC.zip"

        path = pooch.retrieve(
            url=url,
            fname="F_LCS-1_ellipsoid_14-185_ASC.zip",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/magnetics",
            known_hash=None,
            progressbar=True,
            processor=preprocessing,
        )

        grid = xr.open_zarr(
            path,
            consolidated=False,
        )["mag"]

        resampled = resample_grid(
            grid,
            spacing,
            region,
            registration,
            **kwargs,
        )

    else:
        msg = "version must be 'admap1', 'admap2', 'admap2_gdb' or 'LCS-1'"
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
    Accessed from https://doi.pangaea.de/10.1594/PANGAEA.882503

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
    Accessed from https://doi.pangaea.de/10.1594/PANGAEA.930237

    version='aq1'
    From :footcite:t:`stalantarctic2021` and :footcite:t:`stalantarctic2020a`.
    Accessed from https://doi.pangaea.de/10.1594/PANGAEA.924857

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
    **kwargs : typing.Any
        additional keyword arguments to pass to the resample_grid function

    Returns
    -------
    xarray.DataArray
         Returns a loaded, and optional clip/resampled grid of GHF data.

    References
    ----------
    .. footbibliography::
    """

    if version == "an-2015":

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, reproject the .nc file, and save it back to a zarr"
            fname = pooch.Untar()(fname, action, _pooch2)[0]
            fname1 = pathlib.Path(fname)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname1.with_stem(fname1.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".zarr")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load grid
                grid = xr.load_dataarray(fname1)

                # write the current projection
                grid = grid.rio.write_crs("EPSG:4326")

                # reproject to polar stereographic
                grid = grid.rename({"lon": "x", "lat": "y"})
                reprojected = grid.rio.reproject("epsg:3031")

                resampled = resample_grid(
                    reprojected,
                    region=(-3330000.0, 3330000.0, -3330000.0, 3330000.0),
                    spacing=5e3,
                )
                # Save to disk
                resampled = resampled.to_dataset(name="ghf")
                resampled.to_zarr(fname_processed)

            return str(fname_processed)

        path = pooch.retrieve(
            url="http://www.seismolab.org/model/antarctica/lithosphere/AN1-HF.tar.gz",
            fname="an_2015.tar.gz",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/ghf",
            known_hash="9834439cdf99d5ee62fb88a008fa34dbc8d1848e9b00a1bd9cbc33194dd7d402",
            progressbar=True,
            processor=preprocessing,
        )

        grid = xr.open_zarr(
            path,
            consolidated=False,
        )["ghf"]

        resampled = resample_grid(
            grid,
            spacing,
            region,
            registration,
            **kwargs,
        )

    elif version == "martos-2017":

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the .xyz file, grid it, and save it back as a .zarr"
            fname1 = pathlib.Path(fname)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname1.with_stem(fname1.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".zarr")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load the data
                df = pd.read_csv(
                    fname1, header=None, sep=r"\s+", names=["x", "y", "GHF"]
                )

                # grid the data
                processed = df.set_index(["y", "x"]).to_xarray().GHF

                processed = resample_grid(
                    processed,
                    spacing=15e3,
                    region=tuple(  # type: ignore[arg-type]
                        float(pygmt.grdinfo(processed, per_column="n", o=i)[:-1])
                        for i in range(4)
                    ),
                    registration="g",
                )

                # convert to dataset for zarr
                processed = processed.to_dataset(name="ghf")

                # Save to .zarr file
                processed.to_zarr(
                    fname_processed,
                )

            return str(fname_processed)

        path = pooch.retrieve(
            url="https://store.pangaea.de/Publications/Martos-etal_2017/Antarctic_GHF.xyz",
            fname="martos_2017.xyz",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/ghf",
            known_hash="a5814bd0432986e111d0d48bfbd950cce66ba247b26b37f9a7499e66d969eb1f",
            progressbar=True,
            processor=preprocessing,
        )

        grid = xr.open_zarr(
            path,
            consolidated=False,
        )["ghf"]

        resampled = resample_grid(
            grid,
            spacing,
            region,
            registration,
            **kwargs,
        )

    elif version == "burton-johnson-2020":
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

            # re-project to polar stereographic
            df = utils.reproject(
                df,
                input_crs="epsg:4326",
                output_crs="epsg:3031",
                input_coord_names=("lon", "lat"),
                output_coord_names=("x", "y"),
            )

            # retain only points in the region
            if region is not None:
                df = utils.points_inside_region(
                    df,
                    region,
                )

            resampled = df

        elif kwargs.get("points", False) is False:
            path = pooch.retrieve(
                url="https://doi.org/10.5194/tc-14-3843-2020-supplement",
                fname="burton_johnson_2020.zip",
                path=f"{pooch.os_cache('pooch')}/polartoolkit/ghf",
                known_hash="66b1f7acd06eeb6a6362c89b05db07034f510c81e3115cefbd4d11a584f143b2",
                processor=pooch.Unzip(extract_dir="burton_johnson_2020"),
                progressbar=True,
            )

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
                spacing,
                region,
                registration,
                **kwargs,
            )

    elif version == "losing-ebbing-2021":

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the .csv file, grid it, and save it back as a .zarr"
            fname1 = pathlib.Path(fname)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname1.with_stem(fname1.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".zarr")

            # Only recalculate if new download or the processed file doesn't exist yet
            if action in ("download", "update") or not fname_processed.exists():
                # load data
                df = pd.read_csv(fname1)

                # re-project to polar stereographic
                df = utils.reproject(
                    df,
                    input_crs="epsg:4326",
                    output_crs="epsg:3031",
                    input_coord_names=("Lon", "Lat"),
                    output_coord_names=("x", "y"),
                )

                # block-median and grid the data
                df = pygmt.blockmedian(
                    df[["x", "y", "HF [mW/m2]"]],  # type: ignore[call-overload]
                    spacing=5e3,
                    region=regions.antarctica,
                    registration="g",
                )
                processed = pygmt.surface(
                    data=df[["x", "y", "HF [mW/m2]"]],
                    spacing=5e3,
                    region=regions.antarctica,
                    registration="g",
                )

                # clip to coastline
                shp = gpd.read_file(antarctic_boundaries(version="Coastline"))
                processed = utils.mask_from_shp(
                    shp,
                    hemisphere="south",
                    grid=processed,
                    masked=True,
                    invert=False,
                )

                # resample to ensure correct region and spacing
                processed = resample_grid(
                    processed,
                    spacing=5e3,
                    region=regions.antarctica,
                    registration="g",
                )

                # convert to dataset for zarr
                processed = processed.to_dataset(name="ghf")

                # Save to .zarr file
                processed.to_zarr(
                    fname_processed,
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

        grid = xr.open_zarr(
            path,
            consolidated=False,
        )["ghf"]

        resampled = resample_grid(
            grid,
            spacing,
            region,
            registration,
            **kwargs,
        )

    elif version == "aq1":
        path = pooch.retrieve(
            url="https://download.pangaea.de/dataset/924857/files/aq1_01_20.nc",
            fname="aq1.nc",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/ghf",
            known_hash="946ae69e0a3d15a7500d7252fe0ce4f5cb126eaeb6170555ade0acdc38b86d7f",
            progressbar=True,
        )
        grid = xr.load_dataset(path)["Q"]

        # convert from W/m^2 to mW/m^2
        grid = grid * 1000

        resampled = grid * 1000

        # restore registration type
        resampled.gmt.registration = grid.gmt.registration

        if spacing is None:
            spacing = 20e3

        resampled = resample_grid(
            grid,
            spacing,
            region,
            registration,
            **kwargs,
        )

    elif version == "shen-2020":

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the .csv file, grid it, and save it back as a .zarr"
            fname1 = pathlib.Path(fname)

            # Rename to the file to ***_preprocessed.nc
            fname_pre = fname1.with_stem(fname1.stem + "_preprocessed")
            fname_processed = fname_pre.with_suffix(".zarr")

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
                df = utils.reproject(
                    df,
                    input_crs="epsg:4326",
                    output_crs="epsg:3031",
                    input_coord_names=("lon", "lat"),
                    output_coord_names=("x", "y"),
                )

                # block-median and grid the data
                df = pygmt.blockmedian(
                    df[["x", "y", "GHF"]],  # type: ignore[call-overload]
                    spacing=10e3,
                    region=regions.antarctica,
                    registration="g",
                )
                processed = pygmt.surface(
                    data=df[["x", "y", "GHF"]],
                    spacing=10e3,
                    region=regions.antarctica,
                    registration="g",
                    maxradius="1c",
                )

                # resample to ensure correct region and spacing
                processed = resample_grid(
                    processed,
                    spacing=10e3,
                    region=regions.antarctica,
                    registration="g",
                )

                # convert to dataset for zarr
                processed = processed.to_dataset(name="ghf")

                # Save to .zarr file
                processed.to_zarr(
                    fname_processed,
                )

            return str(fname_processed)

        path = pooch.retrieve(
            url="https://drive.google.com/uc?export=download&id=1Fz7dAHTzPnlytuyRNctk6tAugCAjiqzR",
            fname="shen_2020_ghf.xyz",
            path=f"{pooch.os_cache('pooch')}/polartoolkit/ghf",
            known_hash="d6164c3680da52f8f03584293b1a271c937852df9a64f3c98d68debc44e02533",
            processor=preprocessing,
            progressbar=True,
        )

        grid = xr.open_zarr(
            path,
            consolidated=False,
        )["ghf"]

        resampled = resample_grid(
            grid,
            spacing,
            region,
            registration,
            **kwargs,
        )
    else:
        msg = (
            "version must be 'an-2015', 'martos-2017', 'burton-johnson-2020', "
            "'losing-ebbing-2021', 'aq1', or 'shen-2020'"
        )

        raise ValueError(msg)

    return typing.cast(xr.DataArray, resampled)  # pylint: disable=possibly-used-before-assignment


def gia(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
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
    **kwargs : typing.Any
        additional keyword arguments to pass to the resample_grid function
    Returns
    -------
    xarray.DataArray
         Returns a loaded, and optional clip/resampled grid of GIA data.

    References
    ----------
    .. footbibliography::
    """

    if version == "stal-2020":
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
            spacing,
            region,
            registration,
            **kwargs,
        )

    else:
        msg = "version must be 'stal-2020'"
        raise ValueError(msg)

    return typing.cast(xr.DataArray, resampled)


def crustal_thickness(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
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
    **kwargs : typing.Any
        additional keyword arguments to pass to the resample_grid function
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
        # def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
        #     "Load the .dat file, grid it, and save it back as a .nc"
        #     fname1 = pathlib.Path(fname)

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
        #         df = utils.reproject(
        #             df,
        #             input_crs="epsg:4326",
        #             output_crs="epsg:3031",
        #             input_coord_names=("lon", "lat"),
        #             output_coord_names=("x", "y"),
        #         )

        #         # block-median and grid the data
        #         df = pygmt.blockmedian(
        #             df[["x", "y", "thickness"]],
        #             spacing=10e3  # given as 0.5degrees, which is ~3.5km at the pole,
        #             region=regions.antarctica,
        #             registration="g",
        #         )
        #         processed = pygmt.surface(
        #             data=df[["x", "y", "thickness"]],
        #             spacing=10e3  # given as 0.5degrees, which is ~3.5km at the pole,
        #             region=regions.antarctica,
        #             registration="g",
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
        #     spacing,
        #     region,
        #     registration,
        # )

    if version == "an-2015":

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Unzip the folder, reproject the .nc file, and save it back"
            path = pooch.Untar(
                extract_dir="An_2015_crustal_thickness", members=["AN1-CRUST.grd"]
            )(fname, action, _pooch2)
            fname1 = pathlib.Path(path[0])
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
                reprojected = (
                    grid.rio.reproject(
                        "EPSG:3031",
                        resolution=5e3,
                    )
                    .squeeze()
                    .drop_vars(["spatial_ref"])
                )
                # save to netcdf
                reprojected.to_netcdf(fname_processed)

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
            spacing,
            region,
            registration,
            **kwargs,
        )

    else:
        msg = "version must be 'an-2015' or 'shen-2018'"
        raise ValueError(msg)

    return typing.cast(xr.DataArray, resampled)


def moho(
    version: str,
    region: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
    registration: str | None = None,
    **kwargs: typing.Any,
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
    **kwargs : typing.Any
        additional keyword arguments to pass to the resample_grid function

    Returns
    -------
    xarray.DataArray
         Returns a loaded, and optional clip/resampled grid of crustal thickness.

    References
    ----------
    .. footbibliography::
    """

    if version == "shen-2018":

        def preprocessing(fname: str, action: str, _pooch2: typing.Any) -> str:
            "Load the .dat file, grid it, and save it back as a .nc"
            path = pooch.Untar(
                extract_dir="Shen_2018_moho", members=["WCANT_MODEL/moho.final.dat"]
            )(fname, action, _pooch2)
            fname1 = next(p for p in path if p.endswith("moho.final.dat"))
            fname2 = pathlib.Path(fname1)

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
                df = utils.reproject(
                    df,
                    input_crs="epsg:4326",
                    output_crs="epsg:3031",
                    input_coord_names=("lon", "lat"),
                    output_coord_names=("x", "y"),
                )

                # block-median and grid the data
                df = pygmt.blockmedian(
                    df[["x", "y", "depth"]],  # type: ignore[call-overload]
                    spacing=10e3,  # given as 0.5degrees, which is ~3.5km at the pole,
                    region=regions.antarctica,
                    registration="g",
                )
                processed = pygmt.surface(
                    data=df[["x", "y", "depth"]],
                    spacing=10e3,  # given as 0.5degrees, which is ~3.5km at the pole,
                    region=regions.antarctica,
                    registration="g",
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
            spacing,
            region,
            registration,
            **kwargs,
        )

    elif version == "an-2015":
        grid = crustal_thickness(version="an-2015") * -1  # type: ignore[operator]

        resampled = resample_grid(
            grid,
            spacing,
            region,
            registration,
            **kwargs,
        )

    elif version == "pappa-2019":
        msg = "This link is broken, and the data is not available anymore."
        raise ValueError(msg)
        # resampled = pooch.retrieve(
        #     url="https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1029%2F2018GC008111&file=GGGE_21848_DataSetsS1-S6.zip",
        #     fname="pappa_moho.zip",
        #     path=f"{pooch.os_cache('pooch')}/polartoolkit/moho",
        #     known_hash=None,
        #     progressbar=True,
        #     processor=pooch.Unzip(extract_dir="pappa_moho"),
        # )
        # # fname = "/Volumes/arc_04/tankerma/Datasets/Pappa_et_al_2019_data/2018GC008111_Moho_depth_inverted_with_combined_depth_points.grd"
        # grid = pygmt.load_dataarray(resampled)
        # df = grid.to_dataframe().reset_index()
        # df.z = df.z.apply(lambda x: x * -1000)

        # # re-project to polar stereographic
        # df = utils.reproject(
        #     df,
        #     input_crs="epsg:4326",
        #     output_crs="epsg:3031",
        #     input_coord_names=("lon", "lat"),
        #     output_coord_names=("x", "y"),
        # )

        # df = pygmt.blockmedian(
        #     df[["x", "y", "z"]],
        #     spacing=10e3,
        #     registration="g",
        #     region="-1560000/1400000/-2400000/560000",
        # )

        # fname = "inversion_layers/Pappa_moho.nc"

        # pygmt.surface(
        #     df[["x", "y", "z"]],
        #     region="-1560000/1400000/-2400000/560000",
        #     spacing=10e3,
        #     registration="g",
        #     maxradius="1c",
        #     outgrid=fname,
        # )

    else:
        msg = "version must be 'shen-2018', 'an-2015', or 'pappa-2019'"
        raise ValueError(msg)

    return typing.cast(xr.DataArray, resampled)
