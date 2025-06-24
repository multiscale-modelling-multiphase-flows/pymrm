import pytest
import numpy as np
from pymrm import construct_convflux_upwind, construct_convflux_upwind_int, construct_convflux_bc, upwind, minmod, osher, clam, muscl, smart, stoic, vanleer

def test_construct_convflux_upwind():
    shape = 10
    x_f = np.linspace(0, 1, shape + 1)
    conv_matrix, conv_bc = construct_convflux_upwind(shape, x_f)
    assert conv_matrix.shape[0] == shape + 1
    assert conv_matrix.shape[1] == shape
    assert conv_bc.shape[0] == shape + 1

def test_construct_convflux_upwind_int():
    shape = 10
    v = 1.0
    conv_matrix = construct_convflux_upwind_int(shape, v)
    assert conv_matrix.shape[0] == shape + 1
    assert conv_matrix.shape[1] == shape

def test_construct_convflux_bc():
    shape = 10
    x_f = np.linspace(0, 1, shape + 1)
    v = 1.0
    bc = ({'a': 0, 'b': 1, 'd': 1}, {'a': 1, 'b': 0, 'd': 0})
    conv_matrix, conv_bc = construct_convflux_bc(shape, x_f, bc=bc, v=v)
    assert conv_matrix.shape[0] == shape + 1
    assert conv_matrix.shape[1] == shape
    assert conv_bc.shape[0] == shape + 1

def test_tvd_limiters():
    c = np.linspace(-1, 2, 99)
    for func in [upwind, minmod, osher, clam, muscl, smart, stoic, vanleer]:
        result = func(c, 0.5, 0.75)
        assert result.shape == c.shape
