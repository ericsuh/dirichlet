#!/usr/bin/env python

import sys
from setuptools import setup

# Note, any changes here should be replicated in the `tox.ini` file

if not sys.version_info[:2] >= (3, 6):
    raise Exception("Only supports Python versions >= 3.6")
install_requires = [
    "scipy >= 1.4.1",
    "numpy >= 1.18.1",
    "matplotlib >= 3.2.0",
]

setup(
    name="dirichlet",
    version="0.9",
    description="Calculates Dirichlet test and plots 2-simplex Dirichlets",
    author="Eric Suh",
    author_email="contact@ericsuh.com",
    packages=["dirichlet"],
    install_requires=install_requires,
    url="http://github.com/ericsuh/dirichlet",
    download_url="https://github.com/ericsuh/dirichlet/zipball/master",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    long_description="""\
Dirichlet Test
--------------

Calculates nested model Dirichlet test of independence by finding maximum
likelihood estimates of Dirichlet distributions for different data sets
and comparing to the null hypothesis of the data being derived from one
distribution.

In addition, ``dirichlet.simplex`` module creates scatter, contour, and filled
contour 2-simplex plots.
""",
)
