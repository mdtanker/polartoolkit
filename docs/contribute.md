# Contribution guide
## Build the docs
The Docs are build with `Sphinx` and `Read the Docs`. Due to issues with included C programs (GMT and GDAL) in a pip-installed package, `PyGMT` and `GeoPandas` aren't included in the package dependencies, so `Read the Docs` can't run the scripts which are part of the docs (i.e. the gallery examples). Because of this the notebooks don't execute on a build, as specified by `execute_notebooks: 'off'` in `_config.yml`.

Additionally we use `Poetry` as a package manager, which also can't include `PyGMT` or `GeoPandas` successfully (since it installs with Pip). To get around this, we will export the poetry venv, add `PyGMT` and `Geopandas` independently, run the .ipynb's for the docs, then build the docs.

### Set up a virtual environment

Set up the poetry virtual environment:

    make poetry_env

This solves the dependencies for the packages listed in pyproject.toml, adds the versions to a .lock file, install them in a poetry virtual environment, and exports the resulting environment to a requirements.txt file.

Next we need to create a conda/mamba env:
    make delete_env
    make new_env

This will create a new conda env `antarctic_plots_dev` and install `PyGMT`.

Activate it and install the package requirements and local antarctic_plots package in editable mode:

    mamba activate antarctic_plots_dev
    make install_reqs

### Run all .ipynb's to update them

    make run_doc_files

### format and check all the code

    make format
    make check

Fix issues shown in `make check`. If lines are too long, split them. If they are urls, and you want flake8 to ignore the line, add `# noqa` at the end of the line. 

### Check the build manually

    make build_docs

This will run the `.ipynb` files, and convert them to markdown to be included in the docs.
Check for returned errors and open `index.html` in docs/_build/html/ to view the docs.

### Automatically build the docs 

Add, commit, and push all changes to Github in a Pull Request, and RTD should automatically build the docs.

## Build and publish the package
Follow all the above instructions for building the docs

Increase the version number in `pyproject.toml`

Then run the following:

    poetry shell
    make publish

This will both build the dist files, and upload to PyPI. Now push the changes to Github and make a release with the matching version number. 

## Update the dependencies
The package uses `Poetry` (v.1.1.14) to handle dependencies, build, and publish. Unfortunately, due to `PyGMT` relying on the C package `GMT`, poetry can't install `PyGMT`. This is the same with `GeoPandas` relygin on `GDAL`. To update any other dependencies, use the below commands:

    poetry add <PACKAGE>

or if the package is only for development/documentation

    poetry add <PACKAGE> -D

Then run through the commands at the top of this page again to update the environement.

## Set up the binder configuration
To run examples online, Read the Docs will automatically create a Binder instance for this package. The configuration file is `/binder/environment.yml`. To create this or update it do the following:

    make binder_yaml

This will create an environment with the core dependencies, and export it to a .yml. Open this file and add the following at the bottom of the list of dependencies:

  - pip
  - pip:
    --ignore-installed --no-deps git+https://github.com/mdtanker/antarctic_plots.git@main

Now, when submitting a PR, RTD will automatically build the docs and update the Binder environement. 

<!-- This uses the doc_requirements.txt included in the repository, which was create with the below code:

    conda create --name doc_requirements python=3.9
    conda activate doc_requirements
    mamba install pytest flake8 isort jupyter-book 
    pip install black[jupyer]
    pip list --format=freeze > doc_requirements.txt

This should be included in the .readthedocs.yaml, so it should be the env RTD uses to build.
Since `execute_notebooks: "off"` is set in _config.yml, RTD shouldn't need any other packages installed to build.

Add, commit, and push all changes to Github, and RTD should automatically build the docs -->

<!-- ### Need local install to build

    conda create --name ant_plots_build --clone doc_requirements
    conda activate ant_plots_build
    conda install pandas numpy pooch xarray pyproj verde rioxarray netCDF4 pygmt geopandas

Export to requirements.txt
    
    pip list --format=freeze > requirements.txt

Add them to poetry.lock file
    cat requirements.txt | xargs poetry add
    pip install -r requirements.txt -->

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
    pip instal -e . 

## to get poetry to work
without hashes
    poetry export -f requirements.txt --output requirements.txt --dev --without-hashes
    pip install -r requirements.txt

or with hashes
    poetry export -f requirements.txt --output requirements.txt --dev 
    pip install --no-deps -r requirements.txt

pip install -e .
conda install pygmt geopandas -->
