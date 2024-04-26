# Build, package, test, and clean
PROJECT=polartoolkit
STYLE_CHECK_FILES=.

####
####
# install commands
####
####

create:
	mamba create --name $(PROJECT) --yes --force pygmt geopandas python=3.11

install:
	pip install -e .[all]

install_test:
	pip install $(PROJECT)[all]

remove:
	mamba remove --name $(PROJECT) --all

conda_install:
	mamba create --name $(PROJECT) --yes --force $(PROJECT)

####
####
# test commands
####
####

test:
	pytest -m "not earthdata and not issue and not fetch"

test_fetch:
	pytest -s -m fetch #-rp

####
####
# style commands
####
####

format:
	ruff format $(STYLE_CHECK_FILES)

check:
	ruff check --fix $(STYLE_CHECK_FILES)

lint:
	pre-commit run --all-files

pylint:
	pylint $(PROJECT)

style: format check lint pylint

mypy:
	mypy src/$(PROJECT)


####
####
# chore commands
####
####

release_check:
	semantic-release --noop version

changelog:
	semantic-release changelog

license-add:
	python tools/license_notice.py

license-check:
	python tools/license_notice.py --check


####
####
# doc commands
####
####

run_gallery:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/gallery/*.ipynb

run_tutorials:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/tutorial/*.ipynb

run_doc_files:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/*.ipynb
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/*/*.ipynb

run_notebooks: run_gallery run_tutorials run_doc_files
