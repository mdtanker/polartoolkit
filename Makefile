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
	mamba env remove --name $(PROJECT) --yes

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

run_gallery:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/gallery/*.ipynb

run_tutorials:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/tutorial/*.ipynb

run_doc_files:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/*.ipynb
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/*/*.ipynb

run_notebooks: run_gallery run_tutorials run_doc_files
