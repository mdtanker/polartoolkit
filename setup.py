from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='antarctic_plots',
    version='0.1',
    author='Matt Tankersley',
    author_email='matt.d.tankersley@gmail.com',
    description='Functions to automate Antarctic data visualization',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mdtanker/antarctic_plots',
    project_urls = {
        "Bug Tracker": "https://github.com/mdtanker/antarctic_plots/issues"
    },
    license='MIT',
    packages=['antarctic_plots'],
    install_requires=[
        'pygmt',
        'pandas',
        'numpy',
        'pooch',
        'xarray',
        'pyproj',
        'verde',
        'rioxarray',
        ],
)