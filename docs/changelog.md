# Changelog

## Release v0.0.2 

### 💫 Highlights
* Created a [website for the documentation!](https://antarctic-plots.readthedocs.io/en/latest/installation.html#) 

* Added `NumPy` formatted docstrings to the modules

* Wrote contribution guide, which outlines the unique case of publishing a package with dependencies which need C packages, like `PyGMT` (`GMT`) and `GeoPandas` (`GDAL`). 

* Added `Tips` for generating shapefiles and picking start/end points

### 🚀 Features

* 
* 


### 📖 Documentation

* 
*

### ⛔ Maintenance

* 
* 


### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.1 

### 💫 Highlights
* also probably should have been published to TestPyPI 🤦♂️

### 🚀 Features

* Added a Makefile for streamlining development, publishing, and doc building.
* Added license notifications to all files.


### 📖 Documentation

* Used `Jupyter-Book` structure, with a table of contents (_toc.yml) and various markdown files.
* added `Sphinx.autodoc` to automatically include API documentation.


### ⛔ Maintenance

* Looks of issues with the Poetry -> Jupyter-Books -> Read the Docs workflow
* Poetry / RTD don't like `PyGMT` or `GeoPandas` since they both rely on C packages which can't be installed via pip (`GMT` and `GDAL`). Next release should fix this. 


### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)

---

## Release v0.0.0 

* 🎉 **First release of Antarctic-plots** 🎉

* should have been published to TestPyPI 🤦♂️

### 🧑‍🤝‍🧑 Contributors

[@mdtanker](https://github.com/mdtanker)
