# Build, package, test, and clean
PROJECT=antarctic_plots
TESTDIR=tmp-test-dir-with-unique-name
PYTEST_ARGS=--cov-config=../.coveragerc --cov-report=term-missing --cov=$(PROJECT) --doctest-modules -v --pyargs
# NUMBATEST_ARGS=--doctest-modules -v --pyargs -m use_numba
STYLE_CHECK_FILES= $(PROJECT) docs tools

help:
	@echo "Commands:"
	@echo ""
	@echo "  install   install in editable mode"
	@echo "  test      run the test suite (including doctests) and report coverage"
	@echo "  format    automatically format the code"
	@echo "  check     run code style and quality checks"
	@echo "  clean     clean up build and generated files"
	@echo ""

install:
	pip install -e .

test:
	coverage run -m pytest -n 0

format: isort black license-add

check: isort-check black-check license-check flake8

black:
	black $(STYLE_CHECK_FILES)

black-check:
	black --check $(STYLE_CHECK_FILES)

isort:
	isort $(STYLE_CHECK_FILES)

isort-check:
	isort --check $(STYLE_CHECK_FILES)

license-add:
	python tools/license_notice.py

license-check:
	python tools/license_notice.py --check

flake8:
	flake8p $(STYLE_CHECK_FILES) --exclude=*/_build/*

run_gallery:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/gallery/*.ipynb

run_tutorials:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/tutorial/*.ipynb

run_doc_files:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/*.ipynb
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/*/*.ipynb

build_docs:
	@echo
	@echo "Building HTML files."
	@echo
	jupyter-book build docs/
	@echo
	@echo "Build finished. The HTML pages are in docs/build/html."

remove_poetry:
	poetry env remove --all

poetry_env: remove_poetry
	poetry install --sync --without dev

poetry_env_dev: remove_poetry
	poetry install --sync 
	poetry export -f requirements.txt --output requirements.txt --with dev

package:
	poetry build

test_publish:
	poetry publish --build -r test-pypi

publish:
	poetry publish --build

delete_env:
	mamba remove --name antarctic_plots_dev --all --yes

new_env: delete_env
	mamba create --name antarctic_plots_dev --yes python=3.9 pygmt=0.7.0 geopandas=0.11.0 

install_reqs:
	pip install --no-deps --requirement requirements.txt
	pip install --editable .

delete_test_env:
	mamba remove --name antarctic_plots_test --all --yes

test_env: delete_test_env
	mamba create --name antarctic_plots_test --yes python=3.9 pygmt=0.7.0 geopandas=0.11.0

binder_yml:
	mamba remove --name antarctic_plots_binder --all --yes
	rm binder/environment.yml -f
	mamba create --name antarctic_plots_binder --yes python=3.9 pygmt=0.7.0 geopandas=0.11.0 pandas=1.5 numpy=1.23.1 pooch=1.6.0 tqdm=4.64.0 verde=1.7.0 xarray=2022.6.0 cfgrib=0.9.10.1 rasterio=1.3.2 cftime=1.6.1 zarr=2.12.0 pydap=3.2.2 scipy=1.9.1 h5netcdf=1.0.2 netcdf4=1.6.1 pyproj=3.4 matplotlib=3.6 pyogrio=0.4.1 ipyleaflet=0.17.1 openpyxl=3.0.10
	conda env export --name antarctic_plots_binder --from-history --no-build > binder/environment.yml

clean:
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name ".coverage.*" -exec rm -v {} \;
	rm -rvf docs/_build dist *.egg-info __pycache__ .coverage .cache .pytest_cache $(PROJECT)/_version.py
	rm -rvf $(TESTDIR) dask-worker-space
