# How to contribute
🎉 Thanks for considering contributing to this package! 🎉

<sub>Adapted from the great contribution guidelines of the [Fatiando a Terra](https://www.fatiando.org/) packages<sub>.

> This document contains some general guidelines to help with contributing to this code. Contributing to a package can be a daunting task, if you want help please reach out on the [GitHub discussions page](https://github.com/mdtanker/polartoolkit/discussions)!

Any kind of help would be much appreciated. Here are a few ways to contribute:
* 🐛 Submitting bug reports and feature requests
* 📝 Writing tutorials or examples
* 🔍 Fixing typos and improving to the documentation
* 💡 Writing code for everyone to use

A few easy options:
* Add a new pre-defined region
  * this could simple involve adding 1 line of code!
* Add a new dataset to the `fetch` module
  * most of the code is reused for each function in `fetch`, just find an existing function which has the same input datatype (filetype, whether it needs unzipping, preprocessing, or both), and reused the code.

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
  - [Code style and linting](#code-style-and-linting)
  - [Testing your code](#testing-your-code)
  - [Documentation](#documentation)
  - [Code Review](#code-review)
* [Publish a new release](#publish-a-new-release)
* [Update the Dependencies](#update-the-dependencies)
* [Create a conda environment file](#create-a-conda-environment-file)
* [Set up Binder](#set-up-the-binder-configuration)
* [Release Checklist](#release-checklist)

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
submitting a fix (even better 🌟). You can submit fixes to the documentation pages completely online without having to
download and install anything:

* On each documentation page, there should be a " ✏️ Suggest edit" link at the very
  top (click on the GitHub logo).
* Click on that link to open the respective source file on GitHub for editing online (you'll need a GitHub account).
* Make your desired changes.
* When you're done, scroll to the bottom of the page.
* Fill out the two fields under "Commit changes": the first is a short title describing
  your fixes; the second is a more detailed description of the changes. Try to be as
  detailed as possible and describe *why* you changed something.
* Click on the "Commit changes" button to open a pull request (see below).
* We'll review your changes and then merge them in if everything is OK.
* Done 🎉🍺

Alternatively, you can make the changes offline to the files in the `doc` folder or the
example scripts. See [Contributing Code](#contributing-code) for instructions.

## Contributing Code

**Is this your first contribution?**
Please take a look at these resources to learn about git and pull requests (don't
hesitate to ask questions in the [GitHub discussions page](https://github.com/mdtanker/polartoolkit/discussions)):

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

To get the latest version clone the github repo:

```
git clone https://github.com/mdtanker/polartoolkit.git
```
Change into the directory:

```
cd polartoolkit
```

Run the following command to make a new environment and install the package dependencies:

```
  make create
```
Activate the environment:
```
    conda activate polartoolkit
```
Install your local version:
```
  make install
```
This environment now contains your local, editable version of PolarToolkit, meaning if you alter code in the package, it will automatically include those changes in your environment (you may need to restart your kernel if using Jupyter). If you need to update the dependencies, see the [update the dependencies](#update-the-dependencies) section below.

> **Note:** You'll need to activate the environment every time you start a new terminal.

### Code style and linting

We use [Ruff](https://docs.astral.sh/ruff/) to format the code so we don't have to
think about it. This allows you to not think about proper indentation, line length, or aligning your code while to development. Before committing, or periodically while you code, run the following to automatically format your code:
```
    make format
```
Some formatting changes can't be applied automatically. Running the following to see these.
```
    make check
```
Go through the output of this and try to change the code based on the errors. Search the error codes on the [Ruff documentation](https://docs.astral.sh/ruff/), which should give suggestions. Re-run the check to see if you've fixed it. Somethings can't be resolved (unsplittable urls longer than the line length). For these, add `# noqa: []` at the end of the line and the check will ignore it. Inside the square brackets add the specific error code you want to ignore.

We also use [Pylint](https://pylint.readthedocs.io/en/latest/), which performs static-linting on the code. This checks the code and catches many common bugs and errors, without running any of the code. This check is slightly slower the the `Ruff` check. Run it with the following:
```
make pylint
```
Similar to using `Ruff`, go through the output of this, search the error codes on the [Pylint documentation](https://pylint.readthedocs.io/en/latest/) for help, and try and fix all the errors and warnings. If there are false-positives, or your confident you don't agree with the warning, add ` # pylint: disable=` at the end of the lines, with the warning code following the `=`.

To run all three of the code checks, use:
```
make style
```

#### Docstrings

**All docstrings** should follow the
[numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard).
All functions/classes/methods should have docstrings with a full description of all
arguments and return values.

While the maximum line length for code is automatically set by *Ruff*, docstrings
must be formatted manually. To play nicely with Jupyter and IPython, **keep docstrings
limited to 88 characters** per line. We don't have a good way of enforcing this
automatically yet, so please do your best.

#### Type hints

We have also opted to use type hints throughout the codebase. This means each function/class/method should be fulled typed, including the docstrings. We use [mypy](https://mypy.readthedocs.io/en/stable/) as a type checker.
```
make mypy
```
Try and address all the errors and warnings. If there are complex types, just use `typing.Any`, or if necessary, ignore the line causing the issue by adding `# type: ignore[]` with the error code inside the square brackets.

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

To run a specific test by name:

    pytest --cov=. -k "test_name"

The coverage report will let you know which lines of code are touched by the tests.
**Strive to get 100% coverage for the lines you changed.**
It's OK if you can't or don't know how to test something.
Leave a comment in the PR and we'll help you out.

### Documentation

The Docs are build with `Sphinx` and `Read the Docs`. Due to the above mentioned issues with the included C programs, `Read the Docs (RTD)` can't run the scripts which are part of the docs (i.e. the gallery examples). Because of this the notebooks don't execute on a build, as specified by `execute_notebooks: 'off'` in `_config.yml`. Here is how to run/update the docs on your local machine.

> **Note:** The docs are automatically built on PR's by `RTD`, but it's good practice to build them manually before a PR, to check them for errors.

#### Run all .ipynb's to update them

    make run_doc_files

If your edits haven't changed any part of the core package, then there is no need to re-run the notebooks. If you changed a notebook, just clear it's contents and re-run that one notebook.

#### Check the build manually (optional)

You can build the docs using, but this will require pandoc to be install on your machine:

```bash
nox -s docs
```

You can see a preview with:

```bash
nox -s docs -- --serve
```

#### Automatically build the docs

Add, commit, and push all changes to GitHub in a Pull Request, and `RTD` should automatically build the docs.

In each PR, you will see section of the checks for `RTD`. Click on this to preview the docs for the PR.

`RTD` uses the conda environment specified in `env/RTD_env.yml` when it's building.

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
GitHub will show the status of these checks on the pull request.
Try to get them all passing (green).
If you have any trouble, leave a comment in the PR or
[post on the GH discussions page](https://github.com/mdtanker/polartoolkit/discussions).

## Publish a new release

This will almost always be done by the developers, but as a guide for them, here are instructions on how to release a new version of the package.

Follow all the above instructions for formatting. Push your changes to a new or existing Pull Request. Once the automated GitHub Actions run (and pass), merge the PR into the main branch.

### PyPI (pip)
PyPI release are made automatically via GitHub actions whenever a pull request is merged.

### Conda-Forge
Once the new version is on PyPI, within a few hours a bot will automatically open a new PR in the [PolarToolkit conda-forge feedstock](https://github.com/conda-forge/polartoolkit-feedstock). Go through the checklist on the PR. Most of the time the only actions needs are updated any changes made to the dependencies since the last release. Merge the PR and the new version will be available on conda-forge shortly.

Once the new version is on conda, update the binder .yml file, as below.

## Update the dependencies

To add or update a dependencies, add it to `pyproject.toml` either under `dependencies` or `optional-dependencies`. This will be included in the next build uploaded to PyPI.

If you add a dependency necessary for using the package, make sure to add it to the Binder config file and update the `environment.yml` file in the repository. See below.

## Create a conda environment file

As a backup options for users experiencing install issues, we include a `environment.yml` file in the repo, which users can download and install from. To update this, run the following commands:

```
make remove
make conda_install
make conda_export
```

## Set up the binder configuration

To run this package online, Read the Docs will automatically create a Binder instance based on the configuration file `environment.yml` in a separate repository [`Polartoolkit-Binder`](https://github.com/mdtanker/polartoolkit-binder). This file should reflect the latest release on Conda-Forge. To allow all optional features in Binder, we need to manually add optional dependencies to the `environment.yml` file. Also, to use the latest version of PolarToolkit within Binder, makes sure to update its version in the file after each release.

Once updated, rebuild the Binder environment, look at which package versions Binder used for each specified dependency, and update the environment file with these versions.

Now, when submitting a PR, RTD will automatically build the docs and update the Binder environment.

## Release Checklist
* re-run any relevant notebooks
* check docs are building correctly using the GitHub actions link within the PR
* merge the  PR
* wait for `PyPI` to publish the new version [here](https://pypi.python.org/pypi/polartoolkit)
* wait for a PR to be opened in the [feedstock](https://github.com/conda-forge/polartoolkit-feedstock)
* update any changed dependencies in the feedstock PR and merge
* wait for `conda` to publish the new version [here](https://anaconda.org/conda-forge/polartoolkit)
* update backup `env/environment.yml`
* update polartoolkit version in `environment.yml` in [PolarToolkit-Binder repo](https://github.com/mdtanker/polartoolkit-binder/blob/main/environment.yml)
* test `PyPI` version with `make install_test` and `make test`
* test `conda` version with `make conda_install` and `make test`
