# 🚀 Install

## Online usage (Binder)

See below for the full installation instructions. If instead you'd like to use
this package online, without needing to install anything, check out our
[Binder link](https://mybinder.org/v2/gh/mdtanker/polartoolkit-binder/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fmdtanker%252Fpolartoolkit%26urlpath%3Dtree%252Fpolartoolkit%252Fdocs%252Fgallery%26branch%3Dmain),
which gives full access the the package in an online environment.

## Install Python

Before installing _PolarToolkit_, ensure you have Python downloaded. If you
don't, I recommend setting up Python with Miniforge. See the install
instructions [here](https://github.com/conda-forge/miniforge).

## Install _PolarToolkit_ Locally

There are 3 main ways to install `polartoolkit`. We show them here in order of
simplest to hardest.

### Conda / Mamba

The easiest way to install this package and it's dependencies is with conda or
mamba into a new virtual environment:

```
mamba create --name polartoolkit --yes --force polartoolkit
```

Activate the environment:

```
conda activate polartoolkit
```

### Pip

Instead, you can use pip to install `polartoolkit`, but first you need to
install a few dependencies with conda. This is because `PyGMT` `GeoPandas`, and
`Cartopy` all rely on C packages, which can only be install with conda/mamba and
not with pip.

```{note}
`conda` and `mamba` are interchangeable
```

Create a new virtual environment:

```
mamba create --name polartoolkit --yes --force pygmt geopandas cartopy
```

activate the environment and use `pip` to install `polartoolkit`:

```
mamba activate polartoolkit
pip install polartoolkit
```

```{note}
to install the optional dependencies, use this instead:
`pip install polartoolkit[all]`
```

### Development version

You can use pip, with the above created environment, to install the latest
source from GitHub:

    pip install git+https://github.com/mdtanker/polartoolkit.git

Or you can clone the repository and install:

    git clone https://github.com/mdtanker/polartoolkit.git
    cd polartoolkit
    pip install .

## Common errors

If you get errors related to GDAL and GMT, try reinstalling Geopandas and PyGMT
with the following command:

    mamba install geopandas pygmt --force-reinstall -y

If you get errors related to the PyProj EPSG database, try the following:

    mamba install -c conda-forge proj-data --force-reinstall -y

or

    mamba remove --force pyproj -y
    pip install pyproj --force-reinstall

If you get an error related to traitlets run the following command as discussed
[here](https://github.com/microsoft/vscode-jupyter/issues/5689#issuecomment-829538285):

    mamba install ipykernel --update-deps --force-reinstall -y
