# Automatically plot Antarctic data

<p align="center">
    <img src="cover_fig.png" width="600">
</p> 

## Disclaimer

🚨 **This package is in early stages of design and implementation.** 🚨

I welcome any feedback, ideas, or contributions! Please submit an [issue on Github](https://github.com/mdtanker/antarctic_plots/issues) for problems or feature ideas. 

## About

This python package provides some basic tools for creating maps and plots specific to Antarctica. It includes code to download common continent-wide datasets (i.e. Bedmap2, AntGG, ADMAP), and visualize them in a variety of ways, including cross sections and maps. The Jupyter notebook [examples/examples.ipynb](https://github.com/mdtanker/antarctic_plots/blob/main/examples/examples.ipynb) runs through some of the main functions and usages of this package.

By default the cross-sections include Bedmap2 surface, icebase, and bed as layers, and the data profiles include Free-air gravity and magnetics, but these can be changed to any data, as long as it's supplied as a grid/raster type of file.

Data are sampled along lines either defined by 2 sets of coordinates, or along the path of a shapefile. 

Feel free to use, share, modify, and contribute to this project. I've mostly made this for private usage so for now the documentation is sparse. 

## Getting started

Here I use mamba to install packages, but conda should work as well:

    conda install mamba

Create a new python environment with the required dependencies:

    mamba create --name antarctic_plots python=3.9 pandas numpy pooch xarray pyproj  verde rioxarray pygmt geopandas netCDF4 tqdm

If you will use ipython (i.e. Jupyter notebooks) then include ipykernel and jupyterlab:

        mamba create --name antarctic_plots python=3.9 pandas numpy pooch xarray pyproj  verde rioxarray pygmt geopandas netCDF4 tqdm ipykernel jupyterlab

Activate the new environment:

    mamba activate antarctic_plots

To install the last released version:

    pip install antarctic-plots --no-deps

To install the development version from Github:

    git clone https://github.com/mdtanker/antarctic_plots.git
    cd antarctic_plots
    pip install -e .

Test the install by running the first few cells of [examples/examples.ipynb](https://github.com/mdtanker/antarctic_plots/blob/main/examples/examples.ipynb) or the equivalent [.py file](https://github.com/mdtanker/antarctic_plots/blob/main/examples/examples.py)

If you get an error related to traitlets run the following command as discuss [here](https://github.com/microsoft/vscode-jupyter/issues/5689#issuecomment-829538285):

    conda install ipykernel --update-deps --force-reinstall

Antarctic_plots has the following dependencies:
* pandas 
* numpy 
* pooch 
* xarray 
* pyproj 
* verde 
* rioxarray 
* pygmt 
* geopandas 
* netCDF4 
* tqdm