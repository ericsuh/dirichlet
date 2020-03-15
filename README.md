Dirichlet
=========

A Python package to estimate the Dirichlet distribution, calculate maximum
likelihood, and test for independence from a variable based on fitting nested
Dirichlet distribution hypotheses.

Most of this package is a port of Thomas P. Minka's wonderful
[Fastfit][fastfit] MATLAB code. Much thanks to him for that and his clear
paper ["Estimating a Dirichlet distribution"][estimating].

[estimating]: http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/
[fastfit]: http://research.microsoft.com/en-us/um/people/minka/software/fastfit/

Dirichlet Test
--------------

This likelihood ratio test for independence will determine whether two
Dirichlet-distributed data sets are likely to be from the same distribution
or from two different ones, much like a chi-square or G-test for independence,
but with Dirichlet models.

Simplex Plots
-------------

The `dirichlet.simplex` module creates scatter, contour, and filled contour 2-simplex plots.

Caveats
-------

Note that this package at the moment doesn't support sparse data vectors due to the
numerical fitting algorithm that uses the gamma function. Possibly some sort of
[additive smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) would
make this package work in your context, but that will depend on your application.

Installation
------------

    pip install git+https://github.com/ericsuh/dirichlet.git

This has only been tested with Python 3.6+. Other versions may work, but they
haven't been tested.

Development
-----------

*Note*: These instructions have only been tested on Ubuntu/Debian.

Dev dependencies are listed in `requirements-dev.txt`. You can install them
with:

    pip install -r requirements-dev.txt

### Code style

Please use [`black`](https://black.readthedocs.io/) to format your code when contributing

### Testing

This project uses [`tox`](https://tox.readthedocs.io/) and
[`pytest`](https://pytest.readthedocs.io/) for testing. To run tests,
generally you can just run:

    tox

To test a particular version of Python, you will need to have it
installed and in your `$PATH` ahead of time.