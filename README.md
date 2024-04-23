<h1 align="center">PolarToolkit</h1>
<h2 align="center">Helpful tools for polar researchers</h2>

<p align="center">
<a href="https://polartoolkit.readthedocs.io"><strong>Documentation Link</strong></a>
</p>

<!-- SPHINX-START1 -->

<p align="center">
<a href="https://mybinder.org/v2/gh/mdtanker/polartoolkit/main"><img src="https://mybinder.org/badge_logo.svg" alt="Binder link"></a>
</p>

<p align="center">
<a href="https://pypi.python.org/pypi/polartoolkit"><img src="https://img.shields.io/pypi/v/polartoolkit?style=flat-square"
alt="Latest version on PyPI"/></a>
<a href="https://github.com/conda-forge/polartoolkit-feedstock"><img src="https://img.shields.io/conda/vn/conda-forge/polartoolkit.svg?style=flat-square"alt="Latest version on conda-forge"/></a>
<a href="https://codecov.io/gh/mdtanker/polartoolkit"><img src="https://img.shields.io/codecov/c/github/mdtanker/polartoolkit/main.svg?style=flat-square" alt="Test coverage status"/></a>
</p>

<p align="center">
<a href="https://pypi.org/project/polartoolkit/"><img src="https://img.shields.io/pypi/pyversions/polartoolkit?style=flat-square" alt="Compatible Python versions."/></a>
<a href="https://zenodo.org/badge/latestdoi/475677039"><img src="https://zenodo.org/badge/475677039.svg?style=flat-square" alt="Zenodo DOI"/></a>
<a href='https://readthedocs.org/projects/polartoolkit/'><img src='https://readthedocs.org/projects/polartoolkit/badge/?version=latest&style=flat-square' alt='Documentation Status' /></a>
 </p>

<!-- SPHINX-END1 -->

![](docs/cover_fig.png)

**PolarToolkit** (formerly known as Antarctic-Plots) is a Python package with the goal of making Polar (i.e. Antarctic, Arctic, Greenland) research more efficient, reproducible, and accessible. The software does this by providing: 

- Convenient functions for downloading and pre-processing a wide range of commonly used polar datasets
- Tools for common geospatial tasks (i.e. changing data resolution, subsetting data by geographic regions)
- Code to easily create publication-quality maps, data profiles, and cross-sections

Additionally, **PolarToolkit** provides an easy means for exploring datasets with pre-defined or interactively-chosen geographic regions.

## Disclaimer

<p align="center">
🚨 **Ready for daily use but still changing.** 🚨
</p>

This means that we are still adding a lot of new features and sometimes we make changes to the ones we already have while we try to improve the software based on users' experience, test new ideas, make better design decisions, etc. Some of these changes could be **backwards incompatible**. Keep that in mind before you update PolarToolkit to a new major version (i.e. from `v1.0.0` to `v2.0.0`).

I welcome any feedback, ideas, or contributions! Please contact us on the
[GitHub discussions page](https://github.com/mdtanker/polartoolkit/discussions) or submit an [issue on GitHub](https://github.com/mdtanker/polartoolkit/issues) for problems or feature ideas.

<!-- SPHINX-START2 -->

The **PolarToolkit** python package provides some basic tools to help in conducting polar research. You can use it to download common datasets (i.e. BedMachine, Bedmap2, MODIA MoA), create maps and plots specific to Antarctica (soon Greenland and the Arctic as well), and visualize data with multiple methods.

Feel free to use, share, modify, and [contribute](https://polartoolkit.readthedocs.io/en/latest/contributing.html) to this project. I've mostly made this for personal usage so expect significant changes. Hopefully, I'll implement more tests and Gallery examples soon.

## Project goals

Below is a list of some features I hope to eventually include. Feel free to make a feature request through
[GitHub Issues](https://github.com/mdtanker/polartoolkit/issues/new/choose).

- Create 3D interactive models to help visualize data.
- Include more datasets to aid in downloading and storage.
- Additional projections and possible support for the Arctic region as well.

<!-- SPHINX-END2 -->

## Installation

There are 3 main ways to install `polartoolkit`. We show them here in order of simplest to hardest.

### Conda / Mamba

The easiest way to install this package and it's dependencies is with conda or mamba into a new virtual environment:

```
mamba create --name polartoolkit --yes --force polartoolkit
```

And activate the environment:

```
conda activate polartoolkit
```

Note that `conda` and `mamba` are interchangeable.
 
### Pip

Instead, you can use pip to install `polartoolkit`, but first you need to
install a few dependencies with conda. This is because `PyGMT` `GeoPandas`, and `Cartopy` all rely on C packages, which can only be install with conda/mamba and not with pip.

To create a new virtual environment:

```
mamba create --name polartoolkit --yes --force pygmt geopandas cartopy
```

And activate the environment, followed by using `pip` to install `polartoolkit`:

```
mamba activate polartoolkit
pip install polartoolkit
```

To install the optional dependencies of `polartoolkit`, use this instead:
```
`pip install polartoolkit[all]`
```

### Development version

You can use pip, with the above created environment, to install the latest
source from GitHub:

```
pip install git+https://github.com/mdtanker/polartoolkit.git
```

Or you can clone the repository and install:

```
git clone https://github.com/mdtanker/polartoolkit.git
cd polartoolkit
pip install .
```
    
## How to contribute

I really welcome all forms of contribution! If you have any questions, comments or suggestions, please open a [discussion]() or [issue (feature request)]() on the [GitHub page](https://github.com/mdtanker/polartoolkit/)!

Also, please feel free to share how you're using PolarToolkit, I'd love to know.

Please, read our [Contributor Guide](https://polartoolkit.readthedocs.io/en/latest/contributing.html) to learn how you can contribute to the project.
