# Copyright (C) 2012 Eric J. Suh
#
# This file is subject to the terms and conditions defined in file
# 'LICENSE.txt', which is part of this source code package.

'''Dirichlet.py

Maximum likelihood estimation and likelihood ratio tests of Dirichlet
distribution models of data.

Most of this package is a port of Thomas P. Minka's wonderful Fastfit MATLAB
code. Much thanks to him for that and his clear paper "Estimating a Dirichlet
distribution". See the following URL for more information:

    http://research.microsoft.com/en-us/um/people/minka/'''

import sys
import scipy as sp
import scipy.stats as stats
from scipy.special import (psi, polygamma, gammaln)
from numpy import (array, asanyarray, ones, arange, log, diag, vstack, exp,
        asarray, ndarray, zeros, isscalar)
from numpy.linalg import norm
import numpy as np
import simplex

__all__ = [
    'pdf',
    'test',
    'mle',
    'meanprecision',
    'loglikelihood',
]

euler = -1*psi(1) # Euler-Mascheroni constant

def test(D1, D2, method='meanprecision', maxiter=None):
    '''Test for statistical difference between observed proportions.

    Parameters
    ----------
    D1 : array
    D2 : array
        Both ``D1`` and ``D2`` must have the same number of columns, which are
        the different levels or categorical possibilities. Each row of the
        matrices must add up to 1.
    method : string
        One of ``'fixedpoint'`` and ``'meanprecision'``, designates method by
        which to find MLE Dirichlet distribution. Default is
        ``'meanprecision'``, which is faster.
    maxiter : int
        Maximum number of iterations to take calculations. Default is
        ``sys.maxint``.

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

    D0 = vstack((D1, D2))
    a0 = mle(D0, method=method, maxiter=maxiter)
    a1 = mle(D1, method=method, maxiter=maxiter)
    a2 = mle(D2, method=method, maxiter=maxiter)

    D = 2 * (loglikelihood(D1, a1) + loglikelihood(D2, a2)
         - loglikelihood(D0, a0))
    return (D, stats.chi2.sf(D, K1), a0, a1, a2)

def pdf(alphas):
    '''Returns a Dirichlet PDF function'''
    alphap = alphas - 1
    c = np.exp(gammaln(alphas.sum()) - gammaln(alphas).sum())
    def dirichlet(xs):
        '''N x K array'''
        return c * (xs**alphap).prod(axis=1)
    return dirichlet

def meanprecision(a):
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
    logp = log(D).mean(axis=0)
    return N*(gammaln(a.sum()) - gammaln(a).sum() + ((a - 1)*logp).sum())

def mle(D, tol=1e-9, method='meanprecision', maxiter=None):
    '''Iteratively computes maximum likelihood Dirichlet distribution
    for an observed data set, i.e. a for which log p(D|a) is maximum.

    Parameters
    ----------
    D : 2D array
        ``N x K`` array of numbers from [0,1] where ``N`` is the number of
        observations, ``K`` is the number of parameters for the Dirichlet
        distribution.
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    method : string
        One of ``'fixedpoint'`` and ``'meanprecision'``, designates method by
        which to find MLE Dirichlet distribution. Default is
        ``'meanprecision'``, which is faster.
    maxiter : int
        Maximum number of iterations to take calculations. Default is
        ``sys.maxint``.

    Returns
    -------
    a : array
        Maximum likelihood parameters for Dirichlet distribution.'''

    if method == 'meanprecision':
        return _meanprecision(D, tol=tol, maxiter=maxiter)
    else:
        return _fixedpoint(D, tol=tol, maxiter=maxiter)

def _fixedpoint(D, tol=1e-9, maxiter=None):
    '''Simple fixed point iteration method for MLE of Dirichlet distribution'''
    N, K = D.shape
    logp = log(D).mean(axis=0)
    a0 = _init_a(D)

    # Start updating
    if maxiter is None:
        maxiter = sys.maxint
    for i in xrange(maxiter):
        a1 = _ipsi(psi(a0.sum()) + logp)
        # if norm(a1-a0) < tol:
        if abs(loglikelihood(D, a1)-loglikelihood(D, a0)) < tol: # much faster
            return a1
        a0 = a1
    raise Exception('Failed to converge after {} iterations, values are {}.'
                    .format(maxiter, a1))

def _meanprecision(D, tol=1e-9, maxiter=None):
    '''Mean and precision alternating method for MLE of Dirichlet
    distribution'''
    N, K = D.shape
    logp = log(D).mean(axis=0)
    a0 = _init_a(D)
    s0 = a0.sum()
    if s0 < 0:
        a0 = a0/s0
        s0 = 1
    elif s0 == 0:
        a0 = ones(a.shape) / len(a)
        s0 = 1
    m0 = a0/s0

    # Start updating
    if maxiter is None:
        maxiter = sys.maxint
    for i in xrange(maxiter):
        a1 = _fit_s(D, a0, logp)
        s1 = sum(a1)
        a1 = _fit_m(D, a1, logp)
        m = a1/s1
        # if norm(a1-a0) < tol:
        if abs(loglikelihood(D, a1)-loglikelihood(D, a0)) < tol: # much faster
            return a1
        a0 = a1
    raise Exception('Failed to converge after {} iterations, values are {}.'
                    .format(maxiter, a1))

def _fit_s(D, a0, logp, tol=1e-9, maxiter=100):
    '''Assuming a fixed mean for Dirichlet distribution, maximize likelihood
    for preicision a.k.a. s'''
    N, K = D.shape
    s1 = a0.sum()
    m = a0 / s1
    mlogp = (m*logp).sum()
    for i in xrange(maxiter):
        s0 = s1
        g = psi(s1) - (m*psi(s1*m)).sum() + mlogp
        h = _trigamma(s1) - ((m**2)*_trigamma(s1*m)).sum()

        if g + s1 * h < 0:
            s1 = 1/(1/s0 + g/h/(s0**2))
        if s1 <= 0:
            s1 = s0 * exp(-g/(s0*h + g)) # Newton on log s
        if s1 <= 0:
            s1 = 1/(1/s0 + g/((s0**2)*h + 2*s0*g)) # Newton on 1/s
        if s1 <= 0:
            s1 = s0 - g/h # Newton
        if s1 <= 0:
            raise Exception('Unable to update s from {}'.format(s0))

        a = s1 * m
        if abs(s1 - s0) < tol:
            return a

    raise Exception('Failed to converge after {} iterations, s is {}'
            .format(maxiter, s))

def _fit_m(D, a0, logp, tol=1e-9, maxiter=1000):
    '''With fixed precision s, maximize mean m'''
    N,K = D.shape
    s = a0.sum()

    for i in xrange(maxiter):
        m = a0 / s
        a1 = _ipsi(logp + (m*(psi(a0) - logp)).sum())
        a1 = a1/a1.sum() * s

        if norm(a1 - a0) < tol:
            return a1
        a0 = a1

    raise Exception('Failed to converge after {} iterations, s is {}'
            .format(maxiter, s))

def _piecewise(x, condlist, funclist, *args, **kw):
    '''Fixed version of numpy.piecewise for 0-d arrays'''
    x = asanyarray(x)
    n2 = len(funclist)
    if isscalar(condlist) or \
            (isinstance(condlist, np.ndarray) and condlist.ndim == 0) or \
            (x.ndim > 0 and condlist[0].ndim == 0):
        condlist = [condlist]
    condlist = [asarray(c, dtype=bool) for c in condlist]
    n = len(condlist)

    zerod = False
    # This is a hack to work around problems with NumPy's
    #  handling of 0-d arrays and boolean indexing with
    #  numpy.bool_ scalars
    if x.ndim == 0:
        x = x[None]
        zerod = True
        newcondlist = []
        for k in range(n):
            if condlist[k].ndim == 0:
                condition = condlist[k][None]
            else:
                condition = condlist[k]
            newcondlist.append(condition)
        condlist = newcondlist

    if n == n2-1:  # compute the "otherwise" condition.
        totlist = condlist[0]
        for k in range(1, n):
            totlist |= condlist[k]
        condlist.append(~totlist)
        n += 1
    if (n != n2):
        raise ValueError(
                "function list and condition list must be the same")

    y = zeros(x.shape, x.dtype)
    for k in range(n):
        item = funclist[k]
        if not callable(item):
            y[condlist[k]] = item
        else:
            vals = x[condlist[k]]
            if vals.size > 0:
                y[condlist[k]] = item(vals, *args, **kw)
    if zerod:
        y = y.squeeze()
    return y

def _init_a(D):
    '''Initial guess for Dirichlet alpha parameters given data D'''
    E = D.mean(axis=0)
    E2 = (D**2).mean(axis=0)
    return ((E[0] - E2[0])/(E2[0]-E[0]**2)) * E

def _ipsi(y, tol=1.48e-9, maxiter=10):
    '''Inverse of psi (digamma) using Newton's method. For the purposes
    of Dirichlet MLE, since the parameters a[i] must always
    satisfy a > 0, we define ipsi :: R -> (0,inf).'''
    y = asanyarray(y, dtype='float')
    x0 = _piecewise(y, [y >= -2.22, y < -2.22],
            [(lambda x: exp(x) + 0.5), (lambda x: -1/(x+euler))])
    for i in xrange(maxiter):
        x1 = x0 - (psi(x0) - y)/_trigamma(x0)
        if norm(x1 - x0) < tol:
            return x1
        x0 = x1
    raise Exception(
        'Unable to converge in {} iterations, value is {}'.format(maxiter, x1))

def _trigamma(x):
    return polygamma(1, x)
