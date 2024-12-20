# Build, package, test, and clean
PROJECT=polartoolkit
VERSION := $(shell grep -m 1 'version =' pyproject.toml | tr -s ' ' | tr -d '"' | tr -d "'" | cut -d' ' -f3)

print-%  : ; @echo $* = $($*)
####
####
# install commands
####
####

create:
	mamba env create --file environment.yml

install:
	pip install --no-deps -e .

remove:
	mamba remove --name $(PROJECT) --all

pip_install:
	pip install $(PROJECT)[all]==$(VERSION)

conda_install:
	mamba create --name $(PROJECT) --yes --force --channel conda-forge $(PROJECT)=$(VERSION) pytest pytest-cov deepdiff ipykernel

conda_export:
	mamba env export --name $(PROJECT) --channel conda-forge --file env/environment.yml
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

check:
	pre-commit run --all-files

pylint:
	pylint $(PROJECT)

style: check pylint

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

clear_datasets:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/datasets/**/*.ipynb

run_datasets: clear_datasets
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/datasets/**/*.ipynb

clear_tutorial:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/tutorial/*.ipynb

run_tutorial: clear_tutorial
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/tutorial/*.ipynb

clear_how_to:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/how_to/*.ipynb

run_how_to: clear_how_to
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/how_to/*.ipynb

clear_docs:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/*.ipynb
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/**/*.ipynb

run_docs: clear_docs
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/*.ipynb
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/**/*.ipynb
