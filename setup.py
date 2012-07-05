#!/usr/bin/env python

# This file is subject to the terms and conditions defined in file
# 'LICENSE.txt', which is part of this source code package.

from setuptools import setup

setup(
    name='dirichlet',
    version='0.5',
    description='Calculates Dirichlet test',
    author='Eric Suh',
    author_email='contact@ericsuh.com',
    packages = find_packages(),
    install_requires = [
        'scipy >= 0.10.1',
        'numpy >= 1.8.0',
    ],
    url='http://github.com/ericsuh/dirichlet',
    download_url='https://github.com/ericsuh/dirichlet/zipball/master',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
    ],

    long_description="""\
Dirichlet Test
--------------

Calculates nested model Dirichlet test of independence by finding maximum
likelihood estimates of Dirichlet distributions for different data sets
and comparing to the null hypothesis of the data being derived from one
distribution.
"""
)
