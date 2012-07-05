import pytest
import dirichlet
import numpy
from numpy.linalg import norm
from scipy.special import psi

TOL = 1.48e-8

def test_ipsi():
    x_input = numpy.logspace(-3, 3)
    assert(dirichlet.ipsi(psi(0.01)) - 0.01 < TOL)
    assert(norm(dirichlet.ipsi(psi(x_input))-x_input) < TOL*len(x_input))

def test_dirichlet_mle():
    a0 = numpy.array([100, 299, 100])
    test_data = numpy.random.mtrand.dirichlet(a0, 1000)
    logl0 = dirichlet.loglikelihood(test_data, a0)

    a1 = dirichlet.dirichlet_mle(test_data)
    logl1 = dirichlet.loglikelihood(test_data, a1)

    assert(norm(a0 - a1)/norm(a0) < 0.05)
    assert(abs((logl1 - logl0)/logl1) < 0.01)
