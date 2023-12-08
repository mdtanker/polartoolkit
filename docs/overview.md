# 🔎 Overview

Antarctic-Plots is a Python package developed to help with conducting Antarctic
science. The 5 modules shown here provide tools to help with a variety of common
tasks.

## Modules

### Regions

Pre-defined or interactively chosen geographic regions. Includes many ice
shelves, glaciers, and general geographic regions. These can be used in
functions throughout, such as subsetting data or specifying areas to plot.

### Fetch

Easily download data sets to your computer, retrieve previously download data
sets, and perform common gridded data manipulations. This module uses `Pooch` to
managed the download, storage, and retrieve of data, and `PyGMT` for grid
manipulations.

## Maps

Create high-quality maps using PyGMT with functions specifically tailored to
Antarctica. plot types: 2D, 3D, subplots, interactive maps

### Profile

Define a line, sample layers & data along it, and plot the results.

### Utils

Useful functions for common tasks: coordinate conversion, grid comparison,
masking, de-trending.
