"""Tests for the functions in the Dirichlet package."""

import sys

import numpy
import pytest
from numpy.linalg import norm
from scipy.special import psi

import dirichlet

TOL = 1.48e-8


def test_ipsi():
    x_input = numpy.logspace(-5, 5)
    assert dirichlet.dirichlet._ipsi(psi(0.01)) - 0.01 < TOL
    assert norm(dirichlet.dirichlet._ipsi(psi(x_input)) - x_input) < TOL * len(x_input)


class TestDirichlet:
    # seed must be convertible to 32 bit unsigned integers
    numpy.random.seed(12987529)
    a0 = numpy.array([100, 299, 100])
    D0 = numpy.random.dirichlet(a0, 1000)
    logl0 = dirichlet.loglikelihood(D0, a0)

    D0a = numpy.random.dirichlet(a0, 1000)
    logl0a = dirichlet.loglikelihood(D0a, a0)

    a1 = numpy.array([50, 50, 90])
    D1 = numpy.random.dirichlet(a1, 1000)
    logl1 = dirichlet.loglikelihood(D1, a1)

    @pytest.mark.parametrize("method", ["fixedpoint", "meanprecision"])
    def test_mle(self, method):
        a0_fit = dirichlet.mle(self.D0, method=method)
        logl0_fit = dirichlet.loglikelihood(self.D0, a0_fit)
        assert norm(self.a0 - a0_fit) / norm(self.a0) < 0.1
        assert abs((logl0_fit - self.logl0) / logl0_fit) < 0.01

    @pytest.mark.parametrize("method", ["fixedpoint", "meanprecision"])
    def test_lltest(self, method):
        D, p, a0_fit, a1_fit, a2_fit = dirichlet.test(self.D0, self.D0a, method=method)
        # assert(p > 0.05) Need to do a uniform KS-test for uniform p-values
        assert norm(self.a0 - a0_fit) / norm(self.a0) < 0.1

        D, p, a0_fit, a1_fit, a2_fit = dirichlet.test(self.D0, self.D1, method=method)
        assert p < 0.05
        assert norm(self.a0 - a1_fit) / norm(self.a0) < 0.1
        assert norm(self.a1 - a2_fit) / norm(self.a1) < 0.1
