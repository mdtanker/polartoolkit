# Install

## Online usage (Binder)

See below for the full installation instructions. If instead you'd like to use this package online, without needing to install anything, check out our [Binder link](https://mybinder.org/v2/gh/mdtanker/polartoolkit-binder/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fmdtanker%252Fpolartoolkit%26urlpath%3Dtree%252Fpolartoolkit%252Fdocs%252Ftutorial%26branch%3Dmain), which gives full access to the latest released version of the package in an online environment.

## Install Python

Before installing _PolarToolkit_, ensure you have Python 3.9 or greater installed.
If you don't, I recommend setting up Python with Miniforge.
See the install instructions [here](https://github.com/conda-forge/miniforge).

## Install _PolarToolkit_

The fastest way to install PolarToolkit is with the [mamba](https://mamba.readthedocs.io/en/latest/)
or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)
package manager which takes care of setting up a virtual environment, as well as the
installation of PolarToolkit and all its dependencies:

:::: {tab-set}
::: {tab-item} mamba
:sync: mamba
```
mamba create --name polartoolkit --channel conda-forge polartoolkit
```
:::

::: {tab-item} conda
:sync: conda
```
conda create --name polartoolkit --channel conda-forge polartoolkit
```
:::
::::

To activate the virtual environment, you can do:

:::: {tab-set}
::: {tab-item} mamba
:sync: mamba
```
mamba activate polartoolkit
```
:::

::: {tab-item} conda
:sync: conda
```
conda activate polartoolkit
```
:::
::::

## Test your install

After this, check that everything works by running the following in a Python interpreter
(e.g., in a Jupyter notebook):
```python
import polartoolkit

polartoolkit.__version__
```

This should tell you which version was installed.

To further test, you can clone the GitHub repository and run the suite of tests, see the [Contributors Guide](https://polartoolkit.readthedocs.io/en/latest/contributing.html).

A simpler method to ensure the basics are working would be to download any of the Tutorials or How-to guides and run them locally. On the documentation, each of the examples should have a drop down button in the top right corner to download the `.ipynb`.

You are now ready to use PolarToolkit! Start by looking at our [Quickstart Guide](quickstart),
[Tutorials](tutorial/index.md), or [How-To Guides](how_to/index.md). Good luck!

:::{note}
The sections below provide more detailed, step by step instructions to install and test
PolarToolkit for those who may have a slightly different setup or want to install the latest
development version.
:::


## Alternative Install methods

### Pip

Instead, you can use pip to install `polartoolkit`, but first you need to install a few dependencies with conda.
This is because {mod}`pygmt`, {mod}`geopandas`, and {mod}`cartopy` all rely on C packages, which can only be successfully install with conda/mamba and not with pip.

Create a new virtual environment:

:::: {tab-set}
::: {tab-item} mamba
:sync: mamba
```
mamba create --name polartoolkit --channel conda-forge pygmt geopandas cartopy
```
:::

::: {tab-item} conda
:sync: conda
```
conda create --name polartoolkit --channel conda-forge polartoolkit
```
:::
::::

To activate the virtual environment, you can do:

:::: {tab-set}
::: {tab-item} mamba
:sync: mamba
```
mamba activate polartoolkit
```
:::

::: {tab-item} conda
:sync: conda
```
conda activate polartoolkit
```
:::
::::

Install `polartoolkit` with pip:

```
pip install polartoolkit
```

```{note}
to install the optional dependencies, use this instead:
`pip install polartoolkit[all]`
```

### Development version

You can use pip, with the above created environment, to install the latest source from GitHub:

    pip install git+https://github.com/mdtanker/polartoolkit.git

Or you can clone the repository and install:

    git clone https://github.com/mdtanker/polartoolkit.git
    cd polartoolkit
    pip install .

## Common errors

If you get errors related to GDAL and GMT, try reinstalling Geopandas and PyGMT with the following command:

    mamba install geopandas pygmt --force-reinstall -y --channel conda-forge

If you get errors related to the PyProj, try the following:

    mamba install -c conda-forge proj-data --force-reinstall -y --channel conda-forge

or

    mamba remove --force pyproj -y
    pip install pyproj --force-reinstall

If you get an error related to traitlets run the following command as discussed [here](https://github.com/microsoft/vscode-jupyter/issues/5689#issuecomment-829538285):

    mamba install ipykernel --update-deps --force-reinstall -y

If you are still having errors, then you can download the `environment.yml` file from [here](https://github.com/mdtanker/polartoolkit/blob/main/env/environment.yml) and create a conda environment directly from this:

    mamba create --file PATH_TO_FILE

where you replace PATH_TO_FILE with the path to your downloaded `environment.yml` file.
