#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from setuptools import setup, find_packages

setup(
    name="fillmore",
    version="0.1.8",
    license="SalesforceIQ internal",
    author="yuel",
    author_email="yuel@alumni.stanford.edu",
    description="Semi-unsupervised meta learning for text classification",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    python_requires=">=3.6.9"
)
