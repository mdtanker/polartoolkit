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
  - [General guidelines](#general-guidelines)
  - [Setting up your environment](#setting-up-your-environment)
  - [Code style](#code-style)
  - [Testing your code](#testing-your-code)
  - [Documentation](#documentation)
  - [Code Review](#code-review)


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












## Set up a virtual environment

Antarctic-Plots uses `Poetry` as a package manager, which uses `pip` to install packages. Two of the dependencies, `PyGMT` and `GeoPandas`, need to be installed with `conda` since they contain C packages. To navigate this issue, we install `PyGMT` and `Geopandas` independently, then export the `Poetry` env to a file, and use that to add the remaining dependencies.

The file is `requirements.txt` which defines the packages need to use and develop the package.

Run the following to create a conda env "antarctic_plots_dev":

    make new_env

Activate it with:

    conda activate antarctic_plots_dev

Install the necessary packages:

    make install_reqs

This environment contains your local, editable version of Antarctic-Plots, meaning if you alter code in the package, it will automatically include those changes in your environement (you'll need to restart your kernel). 

## Formatting the code

poetry export -f requirements.txt --output env/requirements-dev.txt --only dev

## Testing the code


## Build the docs

The Docs are build with `Sphinx` and `Read the Docs`. Due to the above mentioned issues with the included C programs, `Read the Docs` can't run the scripts which are part of the docs (i.e. the gallery examples). Because of this the notebooks don't execute on a build, as specified by `execute_notebooks: 'off'` in `_config.yml`. Here is how to run/update the docs on your local machine.

### Run all .ipynb's to update them

    make run_doc_files

### format and check all the code

    make format
    make check

Fix issues shown in `make check`. If lines are too long, split them. If they are urls, and you want flake8 to ignore the line, add `# noqa` at the end of the line. 

### run the tests

    make test

### Check the build manually (Optional)

    make build_docs

This will run the `.ipynb` files, and convert them to markdown to be included in the docs.
Check for returned errors and open `index.html` in docs/_build/html/ to view the docs.

### Automatically build the docs 

Add, commit, and push all changes to Github in a Pull Request, and RTD should automatically build the docs.


## Build and publish the package
Follow all the above instructions for building the docs

Increase the version number in `pyproject.toml`

Recreate the poetry environement without the dev packages:

    make poetry_env

Then run the following:

    make test_publish

This will both build the dist files, and upload to TestPyPI.

Make a new environment, activate it:

    make test_env
    mamba activate antarctic_plots_test

 and run the following, replacing the asterisks with the version number:

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ antarctic-plots==******

Run a few gallery examples to make sure this env works, then its read to publish to the real PyPI:

    make publish

 Now push the changes to Github and make a release with the matching version number. 

## Update the dependencies
The package uses `Poetry` (v.1.1.14) to handle dependencies, build, and publish. Unfortunately, due to `PyGMT` relying on the C package `GMT`, poetry can't install `PyGMT`. This is the same with `GeoPandas` relying on `GDAL`. 

To update or add dependencies, use the below commands:

    poetry add <PACKAGE> --lock

or if the package is only for development/documentation

    poetry add <PACKAGE> --group dev --lock

Replace <PACKAGE> with package name, and optionally set the version with the following formats, as defined [here](https://python-poetry.org/docs/dependency-specification/):

    PACKAGE==2.1 (exactly 2.1)
    PACKAGE@^2.1 (>=2.1.0 <3.0.0)

To completely reset Poetry, and reinstall based on the updated .toml file:

    make poetry_env_dev

This solves the dependencies for the packages listed in pyproject.toml, adds the versions to a .lock file, install them in a poetry virtual environment, and exports the resulting environment to a requirements.txt file.

Then run through the commands at the top of this page again to update the conda environement which is based on the requirements.txt file.

If you add a dependency necessary for using the package, make sure to include it in the Binder config file. See below.

## Set up the binder configuration
To run this pacakge online, Read the Docs will automatically create a Binder instance. It will use the configuration file `/binder/environment.yml`. This file is made by running the below Make command. If you've added a dependency with poetry, you'll need to add it to the end of the Makefile command.

    make binder_yml

This will create an environment with the core dependencies, and export it to a .yml. Open this file and add the following at the bottom of the list of dependencies:
```
  - pip
  - pip:
    - -e ..
```
Now, when submitting a PR, RTD will automatically build the docs and update the Binder environement. 