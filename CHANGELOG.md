# Changelog

## Unreleased

### Other
*  ([`90302ba`](https://github.com/mdtanker/antarctic_plots/commit/90302ba4e059c6688dad9468c42cb38c62ad6540))
*  ([`12a5299`](https://github.com/mdtanker/antarctic_plots/commit/12a5299d590c3ce16e797fd3efd18ea8e7b2234c))
*  ([`c962e56`](https://github.com/mdtanker/antarctic_plots/commit/c962e56ca0eca300b9f45bd4104861cd9fdf67dd))
*  ([`95ad63c`](https://github.com/mdtanker/antarctic_plots/commit/95ad63c9af3c9b305fdbc97c984ac6bee952c3bf))
*  ([`30ea5aa`](https://github.com/mdtanker/antarctic_plots/commit/30ea5aaea6cf599299e0cc7253eaf136b1e54a59))
*  ([`d8c821e`](https://github.com/mdtanker/antarctic_plots/commit/d8c821ee79f146bc74171ae358a331815ba1ac7d))
*  ([`f1249f0`](https://github.com/mdtanker/antarctic_plots/commit/f1249f0d7cb848f489bd25f8364d1693e704f886))


## Unreleased

### 💫 Highlights
* dropped support for Python 3.8 in PR #140
* several new datasets!
* added `robust` option to get_min_max(), and mapping functions
* lots of new customization options (kwargs) for plotting functions
* several bug fixes

### 🚀 Features

#### New datasets in `Fetch`
* Ice Mass Changes from Smith et al. 2020
* Basal Melt Rates from Adusumulli et al. 2020
* Faults and geologic unit shapefiles from GEOMAP (Cox et al. 2023)
* ADMAP-2 magnetics compilation
* ROSETTA-Ice airborne magnetics for the Ross Ice Shelf from Tinto et al. 2019
* BedMachine updated to v3

#### New functions in `regions`
* `get_regions()``
    * list all available regions

### 📖 Documentation

### ⛔ Maintenance
* new URL for fetching ADMAP1 magnetic data
* added Harmonica as a dependency for ADMAP-2 geosoft grid conversion
* fix old binder links
* added `north_arrow()` function call to `plot_grd()` with keyword `add_north_arrow`
* fixed issues with clipping of grid for plotting colorbar histogram

### 🧑‍🤝‍🧑 Contributors
[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.6

### 💫 Highlights
* Switched from Poetry to Setuptools
* Can install with conda
* Eased the dependecy constraints

### ⛔ Maintenance
* renamed various util functions

### 🧑‍🤝‍🧑 Contributors
[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.4

### 💫 Highlights
* New mapping function `antarctic_plots.maps`
* Pre-set regions for commonly plotted areas
* Added Gallery examples
* Created a Binder environment
* More datasets included in `fetch`

### 🚀 Features

#### New module `Maps`

* plot_grd

#### New datasets in `Fetch`

* bedmachine
* geothermal

#### New functions in `Utils`

* alter_region
* coherency
* grd_compare
* grd_trend
* make_grid
* raps
* set_proj

### 📖 Documentation

* Added `Tutorials` and `Gallery examples` to the docs
* Added tutorial for modules `fetch` and `region`

### ⛔ Maintenance
* Closed [Issue #6](https://github.com/mdtanker/antarctic_plots/issues/6): Create gallery examples
* Closed [Issue #9](https://github.com/mdtanker/antarctic_plots/issues/9): Code formating
* Closed [Issue #13](https://github.com/mdtanker/antarctic_plots/issues/13): Specify dependency version
* Closed [Issue #15](https://github.com/mdtanker/antarctic_plots/issues/15): Add inset map of Antarctica
* Closed [Issue #16](https://github.com/mdtanker/antarctic_plots/issues/16): Add Welcome Bot message to first time contributors
* Closed [Issue #20](https://github.com/mdtanker/antarctic_plots/issues/20): Add options to use the package online
* Closed [Issue #25](https://github.com/mdtanker/antarctic_plots/issues/25): Add GHF data to fetch module
* Closed [Issue #26](https://github.com/mdtanker/antarctic_plots/issues/26): Add BedMachine Data to fetch
* Closed [Issue #27](https://github.com/mdtanker/antarctic_plots/issues/27): fetch.bedmap2 issue with xarray
* Closed [Issue #28](https://github.com/mdtanker/antarctic_plots/issues/28): Set region strings for commonly plotted areas
* Closed [Issue #22](https://github.com/mdtanker/antarctic_plots/issues/22): Create Zenodo DOI

### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.3

### 💫 Highlights
* Finally succeeded in building the docs!

### 📖 Documentation

* Added `make build-docs` to execute and overwrite .ipynb to use in docs, since `PyGMT` can't be included in dependencies and therefore RTD's can't execute the .ipynb's.

### ⛔ Maintenance

* Closed [Issue #7](https://github.com/mdtanker/antarctic_plots/issues/7)

### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.2

### 💫 Highlights
* Created a [website for the documentation!](https://antarctic-plots.readthedocs.io/en/latest/installation.html#)

* Added `NumPy` formatted docstrings to the modules

* Wrote contribution guide, which outlines the unique case of publishing a package with dependencies which need C packages, like `PyGMT` (`GMT`) and `GeoPandas` (`GDAL`).

* Added `Tips` for generating shapefiles and picking start/end points

### 📖 Documentation

* Re-wrote docstrings to follow `NumPy` format.
* Added type-hints to docstrings.

### ⛔ Maintenance

* Closed [Issue #13](https://github.com/mdtanker/antarctic_plots/issues/13)
* Closed [Issue #9](https://github.com/mdtanker/antarctic_plots/issues/9)
* Closed [Issue #2](https://github.com/mdtanker/antarctic_plots/issues/2)


### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.1

### 💫 Highlights
* also probably should have been published to TestPyPI 🤦♂️

### 🚀 Features

* Added a Makefile for streamlining development, publishing, and doc building.
* Added license notifications to all files.


### 📖 Documentation

* Used `Jupyter-Book` structure, with a table of contents (_toc.yml) and various markdown files.
* added `Sphinx.autodoc` to automatically include API documentation.


### ⛔ Maintenance

* Looks of issues with the Poetry -> Jupyter-Books -> Read the Docs workflow
* Poetry / RTD don't like `PyGMT` or `GeoPandas` since they both rely on C packages which can't be installed via pip (`GMT` and `GDAL`). Next release should fix this.


### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.0

* 🎉 **First release of Antarctic-plots** 🎉

* should have been published to TestPyPI 🤦♂️

### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)