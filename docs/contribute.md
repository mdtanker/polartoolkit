# Contribution guide
## Build the docs
The Docs are build with `Sphinx` and `Read the Docs`. Due to issues with included C programs (GMT and GDAL) in a pip-installed package, `PyGMT` and `GeoPandas` aren't included in the package dependencies, so `RTD` can't run the scripts which are part of the docs (i.e. `walkthrough.ipynb`). Because of this the notebooks don't execute on a build, as specified by `execute_notebooks: 'off'` in `_config.yml`.

Additionally we use `Poetry` as a package manager, which also can't include `PyGMT` or `GeoPandas` successfully. To get around this, we will export the poetry venv, add `PyGMT` and `Geopandas`, run the .ipynb's for the docs, then build the docs.

### Set up a virtual environment

Set up the poetry virutal environment:

    poetry install

Export to a requirements.txt:

    poetry export -f requirements.txt --output requirements.txt --dev

Deactivate poetry shell:
    
    deactivate

Create a conda/mamba env:

    mamba create --name antarctic_plots python=3.9 pygmt=0.7.0 geopandas=0.11.0
    mamba activate antarctic_plots

Add pinned dependencies

    pip install --no-deps -r requirements.txt

Install local antarctic_plots in editable mode:

    pip install -e .

Or install from PyPI (docs won't update if you build them!):

    pip install antarctic_plots

### Run all .ipynb's to update them

    make run_doc_files

### format and check all the code

    make format
    make check

Fix issues shown in `make check`. If lines are too long, split them. If they are urls, and you want flake8 to ignore the line, add `# noqa` at the end of the line. 

### Check the build manually

    make build_docs

This will run the `.ipynb` files, and convert them to markdown to be included in the dos.
Check for returned errors and open `index.html` in docs/_build/html/ to view the docs.

### Automatically build the docs 

Add, commit, and push all changes to Github in a Pull Request, and RTD should automatically build the docs.

## Build and publish the package
Follow all the above instructions for building the docs

Increase the version number in `pyproject.toml`

Then run the following:

    poetry shell
    make publish

This will both build the dist files, and upload to PyPI. Now push the changes to Github and make a release with the matching version #. 

## Update the dependencies
The package uses `Poetry` (v.1.1.14) to handle dependencies, build, and publish. Unfortunately, due to `PyGMT` relying on the C package `GMT`, poetry can't install `PyGMT`. This is the same with `GeoPandas` relygin on `GDAL`. To update any other dependencies, use the below commands:

    poetry add <PACKAGE>

or if the package is only for development/documentation

    poetry add <PACKAGE> -D

Then:

    poetry lock
    poetry install <optionally add --remove-untracked>

Note, you may need to deleted the .lock file, and run `poetry install --remove-untracked for removals to take place. This will take some time.

This will solve the dependencies for the added package, re-write the `poetry.lock` file, and install the new lock file. 

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
