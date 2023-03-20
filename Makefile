# Build, package, test, and clean
PROJECT=antarctic_plots
STYLE_CHECK_FILES= $(PROJECT) docs tools
TESTDIR=tmp-test-dir-with-unique-name
PYTEST_ARGS=--cov-config=../.coveragerc --cov-report=term-missing --cov=$(PROJECT) --doctest-modules -v --pyargs

help:
	@echo "Commands:"
	@echo ""
	@echo "  install   install in editable mode"
	@echo "  test      run the test suite (including doctests) and report coverage"
	@echo "  format    automatically format the code"
	@echo "  check     run code style and quality checks"
	@echo "  clean     clean up build and generated files"
	@echo ""
#
#
#
# ENVIRONMENTS
#
#
#
install:
	pip install -e ".[dev]"

# install with conda
conda_install:
	mamba create --name antarctic_plots --yes --force antarctic-plots

# create binder yml
binder_env:
	mamba env export --name antarctic_plots --no-builds > binder/environment.yml

#
#
#
# TESTING
#
#
#
test:
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(TESTDIR)
	cd $(TESTDIR); pytest $(PYTEST_ARGS) $(PROJECT)
	cp $(TESTDIR)/.coverage* .
	rm -rvf $(TESTDIR)

test_fast:
	pytest --cov=. -rs -m "not slow"

test_fast_no_earthdata:
	pytest --cov=. -rs -m "not slow or not earthdata"
#
#
#
# STYLE
#
#
#
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
#
#
#
# DOCUMENTATION
#
#
#
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
#
#
#
# PACKAGING
#
#
#
build:
	python -m build

test_publish:
	twine upload -r testpypi dist/*

test_pypi_env:
	mamba create --name antarctic_plots_test_pypi python=3.10 pygmt ipykernel --yes --force

publish:
	twine upload dist/*
