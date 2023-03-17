# Installation

## Online usage (Binder)

See below for the full installation instructions. If instead you'd like to use this package online, without needing to install anything, check out our [Binder link](https://mybinder.org/v2/gh/mdtanker/antarctic_plots/c88a23c9dfe92c36f0bfdbbc277d926c2de763de), which gives full access the the package in an online environment.

This Binder environment can also be accessed by clicking the Binder icon in any of the {doc}`gallery/gallery` examples.

## Install package

This package and most of it's dependencies can be installed with a simple call to `pip`, but since `PyGMT` requires `GMT` and `GeoPandas` require `GDAL`, both of which are C packages, neither can be installed via pip successfully. The below instructions should successfully install antarctic-plots, and all the dependencies:

If you don't have Python set up on your computer, I recommend setting up python with Miniconda. See the install instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

Here I use mamba to install packages, but conda should work as well:

## Create an environment:

    mamba create --name antarctic_plots python=3.9 pygmt=0.7.0 geopandas=0.11.0
    mamba activate antarctic_plots

## Option 1) Install the PyPI package:

    pip install antarctic-plots

## Option 2) Install the dev version:

    git clone https://github.com/mdtanker/antarctic_plots.git
    cd antarctic_plots

Install the package and PyGMT/GeoPandas:

    make install

Test the install by running any of the {doc}`gallery/gallery` examples.

## Common errors

If you get errors related to GDAL and GMT, try reinstalling Geopandas and PyGMT with the following command:

    mamba install geopandas pygmt --force-reinstall -y

If you get errors related to the PyProj EPSG database, try the following:

    mamba install -c conda-forge proj-data --force-reinstall -y

or

    conda remove --force pyproj -y
    pip install pyproj --force-reinstall

If you get an error related to traitlets run the following command as discussed [here](https://github.com/microsoft/vscode-jupyter/issues/5689#issuecomment-829538285):

    mamba install ipykernel --update-deps --force-reinstall -y
