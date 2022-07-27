# Installation
This package and most of it's dependencies can be installed with a simple call to `pip`, but since `PyGMT` requires `GMT` and `GeoPandas` require `GDAL`, both of which are C packages, neither can be installed via pip successfully. The below instructions should successfully install antarctic-plots, and all the dependencies:

Here I use mamba to install packages, but conda should work as well:

## Create an environment:

    mamba create --name antarctic_plots python=3.9 pygmt=0.7.0 geopandas=0.11.0
    mamba activate antarctic_plots

## Install the package: 

    pip install antarctic-plots

## To install the dev version:

    git clone https://github.com/mdtanker/antarctic_plots.git
    cd antarctic_plots

Make a virtual env to install into:
    mamba create --name antarctic_plots python=3.9 pygmt=0.7.0 geopandas=0.11.0
    mamba activate antarctic_plots

Install the package and PyGMT/GeoPandas:
    make install

Test the install by running the first few cells of `docs/walkthrough.ipynb`.

## Common errors

If you get an error related to traitlets run the following command as discussed [here](https://github.com/microsoft/vscode-jupyter/issues/5689#issuecomment-829538285):

    mamba install ipykernel --update-deps --force-reinstall
