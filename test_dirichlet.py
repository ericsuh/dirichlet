import pytest
import dirichlet
import numpy
from numpy.linalg import norm
from scipy.special import psi

TOL = 1.48e-8

def test_ipsi():
    x_input = numpy.logspace(-5, 5)
    assert(dirichlet._ipsi(psi(0.01)) - 0.01 < TOL)
    assert(norm(dirichlet._ipsi(psi(x_input))-x_input) < TOL*len(x_input))

class TestDirichlet:
    a0 = numpy.array([100, 299, 100])
    D0 = numpy.random.mtrand.dirichlet(a0, 1000)
    logl0 = dirichlet.loglikelihood(D0, a0)

    D0p = numpy.random.mtrand.dirichlet(a0, 1000)
    logl0p = dirichlet.loglikelihood(D0p, a0)

    a1 = numpy.array([50, 50, 90])
    D1 = numpy.random.mtrand.dirichlet(a1, 1000)
    logl1 = dirichlet.loglikelihood(D1, a1)

    @pytest.mark.parametrize('method',['fixedpoint','meanprecision'])
    def test_mle(self, method):
        a0_fit = dirichlet.dirichlet_mle(self.D0, method=method)
        logl0_fit = dirichlet.loglikelihood(self.D0, a0_fit)
        assert(norm(self.a0 - a0_fit)/norm(self.a0) < 0.05)
        assert(abs((logl0_fit - self.logl0)/logl0_fit) < 0.01)

    def test_lltest(self):
        D, p, a0_fit, a1_fit, a2_fit = dirichlet.dirichlet(D0, D0p)
        assert(p > 0.05)
        assert(norm(self.a0 - a0_fit)/norm(self.a0) < 0.05)

        D, p, a0_fit, a1_fit, a2_fit = dirichlet.dirichlet(D0, D1)
        assert(p < 0.05)
        assert(norm(self.a0 - a1_fit)/norm(self.a0) < 0.05)
        assert(norm(self.a1 - a2_fit)/norm(self.a1) < 0.05)
