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
	mamba env export --name $(PROJECT) --channel conda-forge --file env/environment.yml

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
