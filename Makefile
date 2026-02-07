PROJECT=polartoolkit
####
####
# install commands
####
####

create:
	mamba env create --file environment.yml --name $(PROJECT)

install:
	pip install --no-deps -e .

update:
	mamba env update --file environment.yml --name $(PROJECT) --prune

remove:
	mamba env remove --name $(PROJECT)

conda_export:
	conda env export --name $(PROJECT) --channel conda-forge --file env/environment.yml

####
####
# doc commands
####
####

clear_datasets:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/datasets/**/*.ipynb
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/datasets/*.ipynb

run_datasets:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/datasets/**/*.ipynb

clear_tutorial:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/tutorial/*.ipynb

run_tutorial:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/tutorial/*.ipynb

clear_how_to:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/how_to/*.ipynb

run_how_to:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/how_to/*.ipynb

clear_gallery:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/gallery/*.ipynb

run_gallery:
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/gallery/*.ipynb

clear_docs: clear_datasets clear_tutorial clear_how_to clear_gallery
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/*.ipynb

run_docs: run_datasets run_tutorial run_how_to run_gallery
	jupyter nbconvert --ExecutePreprocessor.allow_errors=True --execute --inplace docs/*.ipynb
