# Re-compile, test and build library inside a conda environment

# Build
python setup.py bdist_wheel
pip install --upgrade --force-reinstall dist/fillmore-0.2.1-py3-none-any.whl

# Run tests with xml reports and code coverage
# python -m unittest discover -s ./tests/

