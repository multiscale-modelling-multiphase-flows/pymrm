import pytest
import numpy as np
from pymrm.solve import newton, clip_approach
from scipy.sparse import csc_array

def test_newton():
    def f(x):
        return np.array([x[0]**2 - 2]), csc_array([[2*x[0]]])
    x0 = np.array([1.0])
    sol = newton(f, x0)
    assert np.isclose(sol.x[0]**2, 2, atol=1e-6)

def test_clip_approach():
    def f(x):
        return x**2 - 2
    x = np.array([1.0])
    clip_approach(x, f, lower_bounds=0, upper_bounds=2)
    assert np.all(x >= 0) and np.all(x <= 2)
