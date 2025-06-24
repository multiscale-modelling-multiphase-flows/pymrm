import pytest
import numpy as np
from pymrm import NumJac

def test_numjac_basic():
    shape = (5,)
    numjac = NumJac(shape)
    def f(x):
        return x**2
    x = np.arange(5.0)
    g, jac = numjac(f, x)
    assert g.shape == (5,)
    assert jac.shape[0] == 5
    assert jac.shape[1] == 5
