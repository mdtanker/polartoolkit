# Overview

## Modules

**PolarToolkit** is a Python package developed to help with conducting polar
science. The 5 modules shown here provide tools to help with a variety of common
tasks.

### Regions

The {mod}`.regions` module enables pre-defined or interactively chosen geographic regions.
This includes many ice shelves, glaciers, and general geographic regions.
These can be used in functions throughout the package, such as subsetting data or specifying areas to plot.

### Fetch

The {mod}`.fetch` module allows to easily download data sets to your computer, retrieve previously download data
sets, and perform common gridded data manipulations. This module uses {mod}`pooch` to
managed the download, storage, and retrieve of data, and {mod}`pygmt`, {mod}`xarray`, and {mod}`verde` for grid
manipulations.

### Maps

The {mod}`.maps` module can be used to create high-quality maps using {mod}`pygmt` with functions specifically tailored to Antarctica, Greenland and the Arctic. Some included plot types are; 2D, 3D,
subplots, and interactive maps.

### Profiles

The {mod}`.profiles` module has tools to define a line, sample layers & data along it, and plot the results.

### Utils

The {mod}`.utils` module has useful functions for common tasks: coordinate conversion, grid comparison,
masking, de-trending, as well as functions used throughout the rest of the package.
