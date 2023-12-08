# How to contribute
🎉 Thanks for considering contributing to this package! 🎉

<sub>Adapted from the great contribution guidelines of the [Fatiando a Terra](https://www.fatiando.org/) packages<sub>.

> This document contains some general guidlines to help with contributing to this code. Contributing to a package can be a daunting task, if you want help please reach out on the [GitHub discussions page](https://github.com/mdtanker/antarctic_plots/discussions)!

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
  - [Code style](#code-style)
  - [Testing your code](#testing-your-code)
  - [Documentation](#documentation)
  - [Code Review](#code-review)
* [Publish a new release](#release-a-new-version)
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
* Click on the "Commit changes" button to open a
  [pull request (see below)](#pull-requests).
* We'll review your changes and then merge them in if everything is OK.
* Done 🎉🍺

Alternatively, you can make the changes offline to the files in the `doc` folder or the
example scripts. See [Contributing Code](#contributing-code) for instructions.

## Contributing Code

**Is this your first contribution?**
Please take a look at these resources to learn about git and pull requests (don't
hesitate to ask questions in the [GitHub discussions page](https://github.com/mdtanker/antarctic_plots/discussions):

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
git clone https://github.com/mdtanker/antarctic_plots.git
```
Change into the directory:

```
cd antarctic_plots
```

Run the following command to make a new environment and install the package dependencies:

```
  make create
```
Activate the environement:
```
    conda activate antarctic_plots
```
Install your local version:
```
  make install
```
This environment now contains your local, editable version of Antarctic-Plots, meaning if you alter code in the package, it will automatically include those changes in your environement (you may need to restart your kernel if using Jupyter). If you need to update the dependencies, see the [update the dependencies](#update-the-dependencies) section below.

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

> **Note:** The docs are automatically built on PR's by `RTD`, but it's good practise to build them manually before a PR, to check them for errors.

#### Run all .ipynb's to update them

    make run_doc_files

#### Check the build manually (optional)

    make build_docs

This will run the `.ipynb` files, and convert them to markdown to be included in the docs.
Check for returned errors and open `index.html` in docs/_build/html/ to view the docs.

#### Automatically build the docs

Add, commit, and push all changes to GitHub in a Pull Request, and `RTD` should automatically build the docs.

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
[post on the GH discussions page](https://github.com/mdtanker/antarctic_plots/discussions).

## Publish a new release

This will almost always be done by the developers, but as a guide for them, here are instructions on how to release a new version of the package.

Follow all the above instructions for formating and building the docs

### PyPI (pip)
Manually increment the version in antarctic_plots/__init__.py:

  version = "X.Y.Z"

Build the package locally into the /dist folder:

  make build

Upload the dist files to Test PyPI:

    make test_publish

This should automatically find the TestPyPI username and token from a `.pypirc` file in your home directory.

Make a new environment and activate it:

    make test_pypi_env
    mamba activate antarctic_plots_test_pypi

 and run the following, replacing the asterisks with the version number:

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ antarctic-plots==******

Run a few gallery examples to make sure this env works, then its ready to publish to the real PyPI:

    make publish

Now push the changes to GitHub and make a release with the matching version number.

### Conda-Forge
Once the new version is on PyPI, we can update the conda-forge feedstock.

Fork the [conda-forge antarctic-plots feedstock](https://github.com/conda-forge/antarctic-plots-feedstock) on github:

Clone the fork and checkout a new branch

    git clone https://github.com/mdtanker/antarctic-plots-feedstock

    git checkout -b update

Update the `meta.yaml` with the new PyPI version with `grayskull`

  grayskull pypi antarctic-plots

Copy the new contents into the old `meta.yaml` file.

Push the changes to GitHub

    git add .

    git commit -m "updating antarctic-plots"

    git push origin update

Open a PR on GitHub with the new branch.

Once the new version is on conda, update the binder .yml file, as below.

## Update the dependencies

To add or update a dependencies, add it to `pyproject.toml` either under `dependencies` or `optional-dependencies`. This will be included in the next build uploaded to PyPI.

After release a new version on PyPI, we will create a new release on conda-forge, and the new dependencies should automatically be included there.

If you add a dependency necessary for using the package, make sure to add it to `env/env_test.ylm` and include it in the Binder config file. See below.

## Set up the binder configuration

To run this package online, Read the Docs will automatically create a Binder instance based on the configuration file `/binder/environment.yml`. This file reflects the latest release on Conda-Forge. Update it with the following commands.

    make conda install

    conda activate antarctic_plots

    make binder_env

Now, when submitting a PR, RTD will automatically build the docs and update the Binder environement.