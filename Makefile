# Build, package, test, and clean
PROJECT=polartoolkit
STYLE_CHECK_FILES=.

create:
	mamba create --name $(PROJECT) --yes --force pygmt geopandas python=3.11

create_test_env:
	mamba create --name test --yes python=3.11

install:
	pip install -e .[all]

install_test:
	pip install polartoolkit[all]

remove:
	mamba remove --name $(PROJECT) --all

test:
	pytest -m "not earthdata and not issue and not fetch"

test_fetch:
	pytest -s -m fetch #-rp

format:
	ruff format $(STYLE_CHECK_FILES)

check:
	ruff check --fix $(STYLE_CHECK_FILES)

lint:
	pre-commit run --all-files

pylint:
	pylint polartoolkit

style: format check lint pylint

mypy:
	mypy src/polartoolkit

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
	mamba create --name polartoolkit --yes --force polartoolkit

# create binder yml
binder_env:
	mamba env export --name polartoolkit --no-builds > binder/environment.yml

# create ReadTheDocs yml
RTD_env:
	mamba remove --name RTD_env --all --yes
	mamba create --name RTD_env --yes --force python==3.11 pygmt>=0.10.0
	mamba env export --name RTD_env --no-builds --from-history > env/RTD_env.yml
	# delete last line
	sed -i '$$d' env/RTD_env.yml
	# add pip and install local package
	sed -i '$$a\  - pip\n  - pip:\n    - ../.[docs]' env/RTD_env.yml
