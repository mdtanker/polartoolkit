# Contribution guide
🎉 Thanks for considering contributing to this package! 🎉

<sub>Adapted from the great contribution guidelines of the [Fatiando a Terra](https://www.fatiando.org/) packages<sub>.

> This document contains some general guidlines to help with contributing to this code. Contributing to a package can be a daunting task, if you want help please reach out on the [Github discussions page](https://github.com/mdtanker/antarctic_plots/discussions)!

Any kind of help would be much appreciated. Here are a few ways to contribute:
* 🐛 Submitting bug reports and feature requests
* 📝 Writing tutorials or examples
* 🔍 Fixing typos and improving to the documentation
* 💡 Writing code for everyone to use

If you get stuck at any point you can create an issue on GitHub (look for the Issues tab in the repository).

For more information on contributing to open source projects,
[GitHub's own guide](https://guides.github.com/activities/contributing-to-open-source/)
is a great starting point if you are new to version control.
Also, checkout the
[Zen of Scientific Software Maintenance](https://jrleeman.github.io/ScientificSoftwareMaintenance/)
for some guiding principles on how to create high quality scientific software
contributions.

## Contents

* [What Can I Do?](#what-can-i-do)
* [Reporting a Bug](#reporting-a-bug)
* [Editing the Documentation](#editing-the-documentation)
* [Contributing Code](#contributing-code)
  - [Setting up your environment](#setting-up-your-environment)
  - [Code style](#code-style)
  - [Testing your code](#testing-your-code)
  - [Documentation](#documentation)
  - [Code Review](#code-review)
* [Release a New Version](#release-a-new-version)
* [Update the Dependencies](#update-the-dependencies)
* [Set up Binder](#set-up-the-binder-configuration)

## What Can I Do?

* Tackle any issue that you wish! Some issues are labeled as **"good first issues"** to
  indicate that they are beginner friendly, meaning that they don't require extensive
  knowledge of the project.
* Make a tutorial or example of how to do something.
* Provide feedback about how we can improve the project or about your particular use
  case.
* Contribute code you already have. It doesn't need to be perfect! We will help you
  clean things up, test it, etc.

## Reporting a Bug

Find the *Issues* tab on the top of the GitHub repository and click *New Issue*.
You'll be prompted to choose between different types of issue, like bug reports and
feature requests.
Choose the one that best matches your need.
The Issue will be populated with one of our templates.
**Please try to fillout the template with as much detail as you can**.
Remember: the more information we have, the easier it will be for us to solve your
problem.

## Editing the Documentation

If you're browsing the documentation and notice a typo or something that could be
improved, please consider letting us know by [creating an issue](#reporting-a-bug) or
submitting a fix (even better 🌟).

You can submit fixes to the documentation pages completely online without having to
download and install anything:

* On each documentation page, there should be a "Suggest edit" link at the very
  top (click on the GitHub logo).
* Click on that link to open the respective source file on GitHub for editing online (you'll need a GitHub account).
* Make your desired changes.
* When you're done, scroll to the bottom of the page.
* Fill out the two fields under "Commit changes": the first is a short title describing
  your fixes; the second is a more detailed description of the changes. Try to be as
  detailed as possible and describe *why* you changed something.
* Click on the "Commit changes" button to open a
  [pull request (see below)](#pull-requests).
* We'll review your changes and then merge them in if everything is OK.
* Done 🎉🍺

Alternatively, you can make the changes offline to the files in the `doc` folder or the
example scripts. See [Contributing Code](#contributing-code) for instructions.

## Contributing Code

**Is this your first contribution?**
Please take a look at these resources to learn about git and pull requests (don't
hesitate to ask questions in the [Github discussions page](https://github.com/mdtanker/antarctic_plots/discussions):

* [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/).
* Aaron Meurer's [tutorial on the git workflow](http://www.asmeurer.com/git-workflow/)
* [How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github)

If you're new to working with git, GitHub, and the Unix Shell, we recommend
starting with the [Software Carpentry](https://software-carpentry.org/) lessons,
which are available in English and Spanish:

* :gb: [Version Control with Git](http://swcarpentry.github.io/git-novice/) / :es: [Control de
versiones con Git](https://swcarpentry.github.io/git-novice-es/)
* :gb: [The Unix Shell](http://swcarpentry.github.io/shell-novice/) / :es:
[La Terminal de Unix](https://swcarpentry.github.io/shell-novice-es/)


### Setting up your environment

Antarctic-Plots uses `Poetry` as a package manager, which uses `pip` to install packages. Two of the dependencies, `PyGMT` and `GeoPandas`, need to be installed with `conda` since they contain C packages. To navigate this issue, we install `PyGMT` and `Geopandas` with conda, and the rest of the dependencies with `Poetry`.

Run the following to create a conda/mamba env "antarctic_plots_dev":

    # switch "win" for either "linux" or "osx" depending on your computer
    conda create --name antarctic_plots_dev --file env/conda-win-64.lock
    conda activate antarctic_plots_dev
    poetry install

This environment contains your local, editable version of Antarctic-Plots, within the virtual environment `antarctic_plots_dev`, meaning if you alter code in the package, it will automatically include those changes in your environement (you may need to restart your kernel). If you need to update the dependencies, see the [update the dependencies](#update-the-dependencies) section below.

> **Note:** You'll need to activate the environment every time you start a new terminal.

### Code style

We use [Black](https://github.com/ambv/black) to format the code so we don't have to
think about it.
Black loosely follows the [PEP8](http://pep8.org) guide but with a few differences.
Regardless, you won't have to worry about formatting the code yourself.
Before committing, run the following to automatically format your code:

    make format

Some formatting changes can't be applied automatically. Running the following to see these.

    make check

Go through the output of this and try to change the code based on the errors. Re-run the check to see if you've fixed it. Somethings can't be resolved (unsplittable urls longer than the line length). For these, add `# noqa` at the end of the line and the check will ignore it.

#### Docstrings

**All docstrings** should follow the
[numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
All functions/classes/methods should have docstrings with a full description of all
arguments and return values.

While the maximum line length for code is automatically set by *Black*, docstrings
must be formatted manually. To play nicely with Jupyter and IPython, **keep docstrings
limited to 88 characters** per line. We don't have a good way of enforcing this
automatically yet, so please do your best.

### Testing your code

Automated testing helps ensure that our code is as free of bugs as it can be.
It also lets us know immediately if a change we make breaks any other part of the code.

All of our test code and data are stored in the `tests` subpackage.
We use the [pytest](https://pytest.org/) framework to run the test suite, and our continuous integration systems with GitHub Actions use CodeCov to display how much of our code is covered by the tests.

Please write tests for your code so that we can be sure that it won't break any of the
existing functionality.
Tests also help us be confident that we won't break your code in the future.

If you're **new to testing**, see existing test files for examples of things to do.
**Don't let the tests keep you from submitting your contribution!**
If you're not sure how to do this or are having trouble, submit your pull request
anyway.
We will help you create the tests and sort out any kind of problem during code review.

Run the tests and calculate test coverage using:

    make test

You can choose to exclude the slow tests:

    make test_fast

To run a specific test by name:

    pytest --cov=. -k "test_name"

The coverage report will let you know which lines of code are touched by the tests.
**Strive to get 100% coverage for the lines you changed.**
It's OK if you can't or don't know how to test something.
Leave a comment in the PR and we'll help you out.

### Documentation

The Docs are build with `Sphinx` and `Read the Docs`. Due to the above mentioned issues with the included C programs, `Read the Docs (RTD)` can't run the scripts which are part of the docs (i.e. the gallery examples). Because of this the notebooks don't execute on a build, as specified by `execute_notebooks: 'off'` in `_config.yml`. Here is how to run/update the docs on your local machine.

> **Note:** The docs are automatically built on PR's by `RTD`, but it's good practise to build them manually before a PR, to check them for errors.

#### Run all .ipynb's to update them

    make run_doc_files

#### Check the build manually (optional)

    make build_docs

This will run the `.ipynb` files, and convert them to markdown to be included in the docs.
Check for returned errors and open `index.html` in docs/_build/html/ to view the docs.

#### Automatically build the docs

Add, commit, and push all changes to Github in a Pull Request, and `RTD` should automatically build the docs.

### Code Review

After you've submitted a pull request, you should expect to hear at least a comment
within a couple of days.
We may suggest some changes or improvements or alternatives.

Some things that will increase the chance that your pull request is accepted quickly:

* Write a good and detailed description of what the PR does.
* Write tests for the code you wrote/modified.
* Readable code is better than clever code (even with comments).
* Write documentation for your code (docstrings) and leave comments explaining the
  *reason* behind non-obvious things.
* Include an example of new features in the gallery or tutorials.
* Follow the [numpy guide](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
  for documentation.
* Run the automatic code formatter and style checks.

If you're PR involves changing the package dependencies, see the below instructions for [updating the dependencies](#update-the-dependencies).

Pull requests will automatically have tests run by GitHub Actions.
This includes running both the unit tests as well as code linters.
Github will show the status of these checks on the pull request.
Try to get them all passing (green).
If you have any trouble, leave a comment in the PR or
[post on the GH discussions page](https://github.com/mdtanker/antarctic_plots/discussions).

## Release a new version

This will almost always be done by the developers, but as a guide for them, here are instructions on how to release a new version of the package.

Follow all the above instructions for building the docs

Increase the version number in `pyproject.toml`

Recreate the poetry environment without the dev packages:

    make poetry_env

Then run the following:

    make test_publish

This will both build the dist files, and upload to TestPyPI.

Make a new environment and activate it:

    make test_pypi_env
    mamba activate antarctic_plots_test_pypi

 and run the following, replacing the asterisks with the version number:

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ antarctic-plots==******

Run a few gallery examples to make sure this env works, then its read to publish to the real PyPI:

    make publish

 Now push the changes to Github and make a release with the matching version number.

## Update the dependencies
The package uses `Poetry` (v.1.2) to handle dependencies, build, and publish. Unfortunately, due to `PyGMT` relying on the C package `GMT`, poetry can't install `PyGMT`. This is the same with `GeoPandas` relying on `GDAL`. To get around this, we install `PyGMT` and `GeoPandas` with conda, and install the rest of the dependencies with `Poetry`.

To update or add dependencies, use the below commands:

    poetry add <PACKAGE> --lock

or if the package is only for development/documentation

    poetry add <PACKAGE> --group dev --lock

or if the package should include its optional dependencies

    poetry add <PACKAGE[extras]>

Replace <PACKAGE> with the package name, and optionally set the version with the following formats, as defined [here](https://python-poetry.org/docs/dependency-specification/):

    PACKAGE==2.1 (exactly 2.1)
    PACKAGE@^2.1 (>=2.1.0 <3.0.0)

To completely reset Poetry, and reinstall based on the updated .toml file:

    make poetry_env_dev

This solves the dependencies for the packages listed in pyproject.toml, adds the versions to a .lock file, install them in a poetry virtual environment, and exports the resulting environment to a requirements.txt file.

Then run through the commands at the top of this page again to update the conda environement which is based on the requirements.txt file.

If you add a dependency necessary for using the package, make sure to include it in the Binder config file. See below.

## Set up the binder configuration

To run this package online, Read the Docs will automatically create a Binder instance. It will use the configuration file `/binder/environment.yml`. This file is made by running the below Make command. If you've added a dependency with poetry, you'll need to add it to the end of the Makefile command.

    make binder_yml

This will create an environment with the core dependencies, and export it to a .yml. Open this file and add the following at the bottom of the list of dependencies:
```
  - pip
  - pip:
    - -e ..
```
Now, when submitting a PR, RTD will automatically build the docs and update the Binder environement.

## Initial setup with conda/poetry
> Note: this is only necessary to do once, by the creator of the package. This instructions are here for referencing only!

This follows a stackoverflow [answer](https://stackoverflow.com/a/71110028/18686384)

Manually make a conda environemnt.yml file which lists the dependencies that need to be installed via conda and place it in the `env` folder:

    name: antarctic_plots
    channels:
      - conda-forge
      # We want to have a reproducible setup, so we don't want default channels,
      # which may be different for different users. All required channels should
      # be listed explicitly here.
      - nodefaults
    dependencies:
      - python=3.9.*
      - mamba
      - pip # pip must be mentioned explicitly, or conda-lock will fail
      - poetry=1.2.* # 1.1 or additional patches
      - pygmt=0.7.0
      - geopandas=0.12.1

In a conda env, which has Poetry, conda-lock, and mamba installed, run the following:

    # Create Conda lock file(s) from environment.yml
    conda-lock --file env/environment.yml --kind explicit --filename-template "env/conda-{platform}.lock" --conda mamba

    # Set up Poetry
    poetry init --python=~3.9  # version spec should match the one from environment.yml

    # Fix package versions installed by Conda to prevent upgrades
    poetry add --lock pygmt=0.7.0 geopandas=0.12.1

    # Add conda-lock (and other packages, as needed) to pyproject.toml and poetry.lock
    poetry add --lock conda-lock pandas pooch[progress] verde xarray[io] pyproj matplotlib pyogrio rioxarray scipy numpy openpyxl

    # Add development packages
    # for Poetry v1.2
    poetry add --group dev --lock pytest flake8 isort Flake8-pyproject black[jupyter] nbsphinx sphinx-gallery jupyter-book "nbconvert<7" pytest-cov

    # for Poetry v1.1
    poetry add --dev --lock pytest flake8 isort Flake8-pyproject black[jupyter] nbsphinx sphinx-gallery jupyter-book "nbconvert<7" pytest-cov

Creating the env (choose between win-64, linux-64):

    conda create --name antarctic_plots --file env/conda-win-64.lock
    conda activate antarctic_plots
    poetry install

Updating the environment

    # Re-generate Conda lock file(s) based on environment.yml
    conda-lock --file env/environment.yml --kind explicit --filename-template "env/conda-{platform}.lock" --conda mamba

    # Update Conda packages based on re-generated lock file
    mamba update --file conda-win-64.lock

    # Update Poetry packages and re-generate poetry.lock
    poetry update