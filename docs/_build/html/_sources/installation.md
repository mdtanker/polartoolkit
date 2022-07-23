# Installation
Here I use mamba to install packages, but conda should work as well:

## Create a new python environment:

    mamba create --name antarctic_plots python=3.9 
    mamba activate antarctic_plots

## install the package: 

    pip install antarctic-plots

This will install most of the dependencies, except for geopandas, which you'll need to install seperately

    mamba install geopandas

You may need to re-install pygmt

    mamba install pygmt

## To install the latest development version from Github:

    git clone https://github.com/mdtanker/antarctic_plots.git
    cd antarctic_plots
    pip install -e .

Test the install by running the first few cells of [examples/examples.ipynb](https://github.com/mdtanker/antarctic_plots/blob/main/examples/examples.ipynb) or the equivalent [.py file](https://github.com/mdtanker/antarctic_plots/blob/main/examples/examples.py)

If you get an error related to traitlets run the following command as discuss [here](https://github.com/microsoft/vscode-jupyter/issues/5689#issuecomment-829538285):

    conda install ipykernel --update-deps --force-reinstall


## To build the docs (for developers only)
### format and check all the code
First run all the .ipynb so their results can be included in the docs.

Then format and check the code with:

    make format
    make check

Fix issues shown in make check.

### Manually build the docs 

    jupyter-book build docs/

Open `index.html` in docs/_build/html/ to view the docs

### Automatically build the docs 

This uses the doc_requirements.txt included in the repository, which was create with the below code:

    conda create --name doc_requirements python=3.9
    conda activate doc_requirements
    mamba install pytest flake8 isort jupyter-book 
    pip install black[jupyer]
    pip list --format=freeze > doc_requirements.txt

This should be included in the .readthedocs.yaml, so it should be the env RTD uses to build.
Since `execute_notebooks: "off"` is set in _config.yml, RTD shouldn't need any other packages installed to build.

Add, commit, and push all changes to Github, and RTD should automatically build the docs

<!-- ## Older instructions



## install the dependencies seperately:
    
    mamba install pandas numpy pooch xarray pyproj verde rioxarray pygmt geopandas netCDF4 tqdm

Optionally add ipykernel jupyterlab and notebook if you want to use iPython.

## to import working env into poetry
    mamba create --name antarctic_plots python=3.8
    mamba activate antarctic_plots
    mamba install pandas numpy pooch xarray pyproj verde rioxarray netCDF4 pygmt geopandas black pytest flake8 isort jupyter-book
    pip list --format=freeze > requirements.txt
    cat requirements.txt | xargs poetry add
    pip insteal -e . 

## to get poetry to work
without hashes
    poetry export -f requirements.txt --output requirements.txt --dev --without-hashes
    pip install -r requirements.txt

or with hashes
    poetry export -f requirements.txt --output requirements.txt --dev 
    pip install --no-deps -r requirements.txt

pip install -e .
conda install pygmt geopandas -->
