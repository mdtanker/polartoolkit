# Build, package, test, and clean
PROJECT=antarctic_plots
# TESTDIR=tmp-test-dir-with-unique-name
# PYTEST_ARGS=--cov-config=../.coveragerc --cov-report=term-missing --cov=$(PROJECT) --doctest-modules -v --pyargs
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
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(TESTDIR)
	cd $(TESTDIR); NUMBA_DISABLE_JIT=1 MPLBACKEND='agg' pytest $(PYTEST_ARGS) $(PROJECT)
	cp $(TESTDIR)/.coverage* .
	rm -rvf $(TESTDIR)

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
	flake8p $(STYLE_CHECK_FILES)

run_doc_files:
	jupyter nbconvert --execute --inplace docs/*.ipynb

build_docs:
	jupyter-book build docs/

# html-noplot:
#         $(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

publish:
	poetry publish --build

package:
	poetry build

poetry_env: 
	poetry install --sync
	poetry export -f requirements.txt --output requirements.txt --with dev
	
delete_env:
	mamba remove --name antarctic_plots_dev --all --yes

new_env:
	mamba create --name antarctic_plots_dev --yes python=3.9 pygmt=0.7.0 geopandas=0.11.0

install_reqs:
	pip install --no-deps --requirement requirements.txt
	pip install --editable .

clean:
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name ".coverage.*" -exec rm -v {} \;
	rm -rvf docs/_build dist *.egg-info __pycache__ .coverage .cache .pytest_cache $(PROJECT)/_version.py
	rm -rvf $(TESTDIR) dask-worker-space
