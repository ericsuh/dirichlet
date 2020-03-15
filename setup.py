#!/usr/bin/env python
#
# This file is subject to the terms and conditions defined in file
# "LICENSE.txt", which is part of this source code package.

import sys
from setuptools import setup, find_packages

# Note, any changes here should be replicated in the `tox.ini` file

py_version = sys.version_info[:2]
if not (py_version >= (2, 6) and py_version <= (2, 7) or
        py_version >= (3, 6)):
    raise Exception("Only supports Python versions 2.6 to 2.7 and >= 3.6")
if py_version < (3, 0):  # Python 2.6 or 2.7
    install_requires = [
        "scipy >= 0.10.1, <1.3",
        "numpy >= 1.6.2, <1.17",
        "matplotlib >= 1.2.0, <2.3",
    ]
else:  # Python 3.6+
    install_requires = [
        "scipy >= 1.4.1",
        "numpy >= 1.18.1",
        "matplotlib >= 3.2.0",
    ]

setup(
    name="dirichlet",
    version="0.8",
    description="Calculates Dirichlet test and plots 2-simplex Dirichlets",
    author="Eric Suh",
    author_email="contact@ericsuh.com",
    packages=["dirichlet"],
    install_requires = install_requires,
    url="http://github.com/ericsuh/dirichlet",
    download_url="https://github.com/ericsuh/dirichlet/zipball/master",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
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
"""
)
