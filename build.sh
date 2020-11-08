#!/bin/bash

# Set up conda environment for databricks ML runtime
conda env create -f environment.yml
conda install --yes setuptools

# Build
conda run -n fillmore python setup.py bdist_wheel
conda run -n fillmore pip install --upgrade --force-reinstall dist/fillmore-0.1.2-py3-none-any.whl

# Run tests with xml reports and code coverage
python -m unittest discover -s ./tests/ -o ./target/test-reports/

