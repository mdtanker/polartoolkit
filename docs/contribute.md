# Contribution guide

## Set up a virtual environment

Antarctic-Plots uses `Poetry` as a package manager, which uses `pip` to install packages. Two of the dependencies, `PyGMT` and `GeoPandas`, need to be installed with `conda` since they contain C packages. To navigate this issue, we install `PyGMT` and `Geopandas` independently, then export the `Poetry` env to a file, and use that to add the remaining dependencies.

The file is `requirements.txt` which defines the packages need to use and develop the package.

Run the following to create a conda env "antarctic_plots_dev":

    make new_env

Activate it with:

    conda activate antarctic_plots_dev

Install the necessary packages:

    make install_reqs

This environment contains your local, editable version of Antarctic-Plots, meaning if you alter code in the package, it will automatically include those changes in your environement (you'll need to restart your kernel). 

## Formatting the code

poetry export -f requirements.txt --output env/requirements-dev.txt --only dev

## Testing the code


## Build the docs

The Docs are build with `Sphinx` and `Read the Docs`. Due to the above mentioned issues with the included C programs, `Read the Docs` can't run the scripts which are part of the docs (i.e. the gallery examples). Because of this the notebooks don't execute on a build, as specified by `execute_notebooks: 'off'` in `_config.yml`. Here is how to run/update the docs on your local machine.

### Run all .ipynb's to update them

    make run_doc_files

### format and check all the code

    make format
    make check

Fix issues shown in `make check`. If lines are too long, split them. If they are urls, and you want flake8 to ignore the line, add `# noqa` at the end of the line. 

### run the tests

    make test

### Check the build manually (Optional)

    make build_docs

This will run the `.ipynb` files, and convert them to markdown to be included in the docs.
Check for returned errors and open `index.html` in docs/_build/html/ to view the docs.

### Automatically build the docs 

Add, commit, and push all changes to Github in a Pull Request, and RTD should automatically build the docs.


## Build and publish the package
Follow all the above instructions for building the docs

Increase the version number in `pyproject.toml`

Recreate the poetry environement without the dev packages:

    make poetry_env

Then run the following:

    make test_publish

This will both build the dist files, and upload to TestPyPI.

Make a new environment, activate it:

    make test_env
    mamba activate antarctic_plots_test

 and run the following, replacing the asterisks with the version number:

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ antarctic-plots==******

Run a few gallery examples to make sure this env works, then its read to publish to the real PyPI:

    make publish

 Now push the changes to Github and make a release with the matching version number. 

## Update the dependencies
The package uses `Poetry` (v.1.1.14) to handle dependencies, build, and publish. Unfortunately, due to `PyGMT` relying on the C package `GMT`, poetry can't install `PyGMT`. This is the same with `GeoPandas` relying on `GDAL`. 

To update or add dependencies, use the below commands:

    poetry add <PACKAGE> --lock

or if the package is only for development/documentation

    poetry add <PACKAGE> --group dev --lock

Replace <PACKAGE> with package name, and optionally set the version with the following formats, as defined [here](https://python-poetry.org/docs/dependency-specification/):

    PACKAGE==2.1 (exactly 2.1)
    PACKAGE@^2.1 (>=2.1.0 <3.0.0)

To completely reset Poetry, and reinstall based on the updated .toml file:

    make poetry_env_dev

This solves the dependencies for the packages listed in pyproject.toml, adds the versions to a .lock file, install them in a poetry virtual environment, and exports the resulting environment to a requirements.txt file.

Then run through the commands at the top of this page again to update the conda environement which is based on the requirements.txt file.

If you add a dependency necessary for using the package, make sure to include it in the Binder config file. See below.

## Set up the binder configuration
To run this pacakge online, Read the Docs will automatically create a Binder instance. It will use the configuration file `/binder/environment.yml`. This file is made by running the below Make command. If you've added a dependency with poetry, you'll need to add it to the end of the Makefile command.

    make binder_yml

This will create an environment with the core dependencies, and export it to a .yml. Open this file and add the following at the bottom of the list of dependencies:
```
  - pip
  - pip:
    - -e ..
```
Now, when submitting a PR, RTD will automatically build the docs and update the Binder environement. 