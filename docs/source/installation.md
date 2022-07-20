# Installation

Here I use mamba to install packages, but conda should work as well:

### Create a new python environment:

    mamba create --name antarctic_plots python=3.9 
    mamba activate antarctic_plots

### install the package: 

    pip install antarctic-plots --no-deps

### install the dependencies seperately:
    
    mamba install pandas numpy pooch xarray pyproj verde rioxarray pygmt geopandas netCDF4 tqdm

Optionally add ipykernel jupyterlab and notebook if you want to use iPython.


### To install the latest development version from Github:

    git clone https://github.com/mdtanker/antarctic_plots.git
    cd antarctic_plots
    pip install -e .

Test the install by running the first few cells of [examples/examples.ipynb](https://github.com/mdtanker/antarctic_plots/blob/main/examples/examples.ipynb) or the equivalent [.py file](https://github.com/mdtanker/antarctic_plots/blob/main/examples/examples.py)

If you get an error related to traitlets run the following command as discuss [here](https://github.com/microsoft/vscode-jupyter/issues/5689#issuecomment-829538285):

    conda install ipykernel --update-deps --force-reinstall