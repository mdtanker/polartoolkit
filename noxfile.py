import argparse
import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()
PROJECT = nox.project.load_toml()

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run(
        "pre-commit",
        "run",
        "--all-files",
        "--show-diff-on-failure",
        *session.posargs,
    )


@nox.session
def pylint(session: nox.Session) -> None:
    """
    Run PyLint.
    """
    # This needs to be installed into the package environment, and is slower
    # than a pre-commit check
    session.install("-e.", "pylint>=3.2")
    session.run("pylint", "polartoolkit", *session.posargs)


@nox.session
def style(session: nox.Session) -> None:
    """
    Run the linter and Pylint.
    """
    session.notify("lint")
    session.notify("pylint")


@nox.session(venv_backend="mamba", python="3.11")
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.conda_install("pygmt", "geopandas")
    test_deps = nox.project.dependency_groups(PROJECT, "test")
    session.install("-e.", *test_deps)
    session.run(
        "pytest",
        "--cov",
        # "-m",
        # "not earthdata and not issue",
        *session.posargs,
    )


@nox.session(reuse_venv=True, default=False)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass --non-interactive to avoid serving. First positional
    argument is the target directory.
    """

    doc_deps = nox.project.dependency_groups(PROJECT, "docs")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    parser.add_argument("output", nargs="?", help="Output directory")
    args, posargs = parser.parse_known_args(session.posargs)
    serve = args.builder == "html" and session.interactive

    session.install("-e.", *doc_deps, "sphinx-autobuild")

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        args.output or f"docs/_build/{args.builder}",
        *posargs,
    )

    if serve:
        session.run("sphinx-autobuild", "--open-browser", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session
def build(session: nox.Session) -> None:
    """
    Build an SDist and wheel.
    """

    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")
