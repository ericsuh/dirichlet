# This file is subject to the terms and conditions defined in file
# 'LICENSE.txt', which is part of this source code package.

'''Tests for the functions in the Dirichlet package.'''

import sys
import os.path
import pytest
import numpy
from numpy.linalg import norm
from scipy.special import psi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import dirichlet

TOL = 1.48e-8

def test_ipsi():
    x_input = numpy.logspace(-5, 5)
    assert(dirichlet.dirichlet._ipsi(psi(0.01)) - 0.01 < TOL)
    assert(norm(dirichlet.dirichlet._ipsi(psi(x_input))-x_input)
           < TOL*len(x_input))

class TestDirichlet:
    numpy.random.seed(129875290473)
    a0 = numpy.array([100, 299, 100])
    D0 = numpy.random.dirichlet(a0, 1000)
    logl0 = dirichlet.loglikelihood(D0, a0)

    D0a = numpy.random.dirichlet(a0, 1000)
    logl0a = dirichlet.loglikelihood(D0a, a0)

    a1 = numpy.array([50, 50, 90])
    D1 = numpy.random.dirichlet(a1, 1000)
    logl1 = dirichlet.loglikelihood(D1, a1)

    @pytest.mark.parametrize('method',['fixedpoint','meanprecision'])
    def test_mle(self, method):
        a0_fit = dirichlet.mle(self.D0, method=method)
        logl0_fit = dirichlet.loglikelihood(self.D0, a0_fit)
        assert(norm(self.a0 - a0_fit)/norm(self.a0) < 0.1)
        assert(abs((logl0_fit - self.logl0)/logl0_fit) < 0.01)

    @pytest.mark.parametrize('method',['fixedpoint','meanprecision'])
    def test_lltest(self, method):
        D, p, a0_fit, a1_fit, a2_fit = dirichlet.test(self.D0, self.D0a,
                method=method)
        # assert(p > 0.05) Need to do a uniform KS-test for uniform p-values
        assert(norm(self.a0 - a0_fit)/norm(self.a0) < 0.1)

        D, p, a0_fit, a1_fit, a2_fit = dirichlet.test(self.D0, self.D1,
                method=method)
        assert(p < 0.05)
        assert(norm(self.a0 - a1_fit)/norm(self.a0) < 0.1)
        assert(norm(self.a1 - a2_fit)/norm(self.a1) < 0.1)
