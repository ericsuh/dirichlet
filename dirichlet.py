import sys
from scipy.stats import chi2
from scipy.special import (psi, polygamma, gammaln)
from scipy.optimize import (fsolve)
from numpy import (array, ones, arange, log, diag)
from numpy.linalg import norm

def ipsi(y):
    '''Inverse of psi (digamma) using iterative root finder. For the purposes
    of Dirichlet MLE, since the parameters a[i] must always
    satisfy a > 0, we define ipsi :: R -> (0,inf).

    The inverse solver can have trouble if numbers in the array span more
    than 6 orders of magnitude. The main workaround is to pass them in
    separately.'''
    ya = array(y)
    return fsolve((lambda x: psi(x) - ya),
                  ones(ya.shape)*0.1, # x0 > 1 creates problems for y < 0
                  fprime=(lambda x: diag(polygamma(1, x))),
                 )

def loglikelihood(D, a):
    '''Compute log likelihood of Dirichlet distribution, i.e. log p(D|a).

    Parameters
    ----------
    D : 2D array
        where ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    a : array
        Parameters for the Dirichlet distribution.

    Returns
    -------
    logl : float
        The log likelihood of the Dirichlet distribution'''
    N, K = D.shape
    logps = (1.0/N)*log(D).sum(axis=0)
    return N*(gammaln(a.sum()) - gammaln(a).sum() + ((a - 1)*logps).sum())

def dirichlet_mle(D, a0=None, tol=1e-9, maxiter=None):
    '''Iteratively computes maximum likelihood Dirichlet distribution
    for an observed data set, i.e. a for which log p(D|a) is maximum.

    Parameters
    ----------
    D : 2D array
        ``N x K`` array of numbers from [0,1] where ``N`` is the number of
        observations, ``K`` is the number of parameters for the Dirichlet
        distribution.
    a0 : array
        Initial guess for parameters. If not given, will be estimated from
        D.
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is
        ``sys.maxint``.

    Returns
    -------
    a : array
        Maximum likelihood parameters for Dirichlet distribution.'''
    N, K = D.shape
    logps = (1.0/N)*log(D).sum(axis=0)

    if a0 is None:
        E = D.mean(axis=0)
        E2 = (D**2).mean(axis=0)
        a0 = ((E[0]-E2[0])/(E2[0]-E[0]**2)) * E

    # Start updating
    a1 = ipsi(psi(a0.sum()) + logps)

    if maxiter is None:
        maxiter = sys.maxint

    for i in xrange(maxiter):
        if norm(a1-a0) < tol:
            return a1
        else:
            a0 = a1
            a1 = ipsi(psi(a0.sum()) + logps)

    raise Exception('Failed to converge after {} iterations, values are {}.'
                    .format(maxiter, a1))

def meanprec(a):
    '''Mean and precision of Dirichlet distribution.

    Parameters
    ----------
    a : array
        Parameters of Dirichlet distribution.

    Returns
    -------
    mean : array
        Numbers [0,1] of the means of the Dirichlet distribution.
    precision : float
        Precision or concentration parameter of the Dirichlet distribution.'''

    s = a.sum()
    m = a / s
    return (m,s)

def dirichlet(D1, D2):
    '''Test for statistical difference between observed proportions.

    Parameters
    ----------
    D1 : array
    D2 : array
        Both ``D1`` and ``D2`` must have the same number of columns, which are
        the different levels or categorical possibilities. Each row of the
        matrices must add up to 1.

    Returns
    -------
    D : float
        Test statistic, which is ``-2 * log`` of likelihood ratios.
    p : float
        p-value of test.
    a0 : array
    a1 : array
    a2 : array
        MLE parameters for the Dirichlet distributions fit to 
        ``D1`` and ``D2`` together, ``D1``, and ``D2``, respectively.'''

    N1, K1 = D1.shape
    N2, K2 = D2.shape
    if K1 != K2:
        raise Exception("D1 and D2 must have the same number of columns")

    D0 = numpy.vstack(D1, D2)
    a0 = dirichlet_mle(D0)
    a1 = dirichlet_mle(D1)
    a2 = dirichlet_mle(D2)

    D = 2 * (loglikelihood(D1, a1) + loglikelihood(D2, a2)
         - loglikelihood(D0, a0))
    return (D, chi2.sf(D, K1), a0, a1, a2)
