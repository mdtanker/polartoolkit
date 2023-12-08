# Build, package, test, and clean
PROJECT=antarctic_plots
STYLE_CHECK_FILES=.

create:
	mamba create --name $(PROJECT) --yes --force pygmt geopandas python=3.11

create_test_env:
	mamba create --name test --yes python=3.11

install:
	pip install -e .[all]

install_test:
	pip install antarctic-plots[all]

remove:
	mamba remove --name $(PROJECT) --all

test:
	pytest -m "not earthdata and not issue and not fetch"

format:
	ruff format $(STYLE_CHECK_FILES)

check:
	ruff check --fix $(STYLE_CHECK_FILES)

lint:
	pre-commit run --all-files

pylint:
	pylint antarctic_plots

style: format check lint pylint

mypy:
	mypy src/antarctic_plots

release_check:
	semantic-release --noop version

changelog:
	semantic-release changelog

license-add:
	python tools/license_notice.py

license-check:
	python tools/license_notice.py --check

run_gallery:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/gallery/*.ipynb

run_tutorials:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/tutorial/*.ipynb

run_doc_files:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/*.ipynb
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/*/*.ipynb

run_notebooks: run_gallery run_tutorials run_doc_files

# install with conda
conda_install:
	mamba create --name antarctic_plots --yes --force antarctic-plots

# create binder yml
binder_env:
	mamba env export --name antarctic_plots --no-builds > binder/environment.yml
