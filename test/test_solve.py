import pytest
import numpy as np
from pymrm import newton, clip_approach

def test_newton():
    def f(x):
        return x**2 - 2
    x0 = 1.0
    sol = newton(f, x0)
    assert abs(sol - np.sqrt(2)) < 1e-6

def test_clip_approach():
    def f(x):
        return x**2 - 2
    x0 = 1.0
    sol = clip_approach(x0, f, lower_bounds=0, upper_bounds=2)
    assert 0 <= sol <= 2
