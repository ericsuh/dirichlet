"""Tests for the functions in the Simplex module."""

import sys

import numpy as np
import numpy.linalg

from dirichlet import simplex

TOL = 1.48e-8


def assert_almost(x, y, tol=1.48e-8):
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    assert numpy.linalg.norm(x - y) < tol


def test_cartesian():
    assert_almost(simplex.cartesian(np.array([1, 0, 0])), np.array([0, 0]))
    assert_almost(simplex.cartesian(np.array([0, 1, 0])), np.array([1, 0]))
    assert_almost(
        simplex.cartesian(np.array([0, 0, 1])), np.array([0.5, 0.5 * np.sqrt(3)])
    )
    assert_almost(simplex.cartesian(np.array([0.5, 0.5, 0])), np.array([0.5, 0]))
    assert_almost(
        simplex.cartesian(np.array([0, 0.5, 0.5])), np.array([0.75, np.sqrt(3) * 0.25])
    )


def test_simplex():
    assert_almost(simplex.barycentric(np.array([0, 0])), np.array([1, 0, 0]))
    assert_almost(simplex.barycentric(np.array([1, 0])), np.array([0, 1, 0]))
    assert_almost(
        simplex.barycentric(np.array([0.5, 0.5 * np.sqrt(3)])), np.array([0, 0, 1])
    )
