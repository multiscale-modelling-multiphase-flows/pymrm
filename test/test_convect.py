import pytest
import numpy as np
from pymrm.convect import (
    construct_convflux_upwind, construct_convflux_upwind_int, construct_convflux_bc,
    upwind, minmod, osher, clam, muscl, smart, stoic, vanleer
)

def test_construct_convflux_upwind():
    shape = (10,)
    x_f = np.linspace(0, 1, shape[0] + 1)
    conv_matrix, conv_bc = construct_convflux_upwind(shape, x_f)
    assert conv_matrix.shape[0] > 0
    assert conv_bc.shape[0] > 0

def test_construct_convflux_upwind_int():
    shape = (10,)
    v = np.ones(11)
    conv_matrix = construct_convflux_upwind_int(shape, v)
    assert conv_matrix.shape[0] > 0

def test_construct_convflux_bc():
    shape = (10,)
    x_f = np.linspace(0, 1, shape[0] + 1)
    v = np.ones(11)
    bc = ({'a': 0, 'b': 1, 'd': 1}, {'a': 1, 'b': 0, 'd': 0})
    result = construct_convflux_bc(shape, x_f, bc=bc, v=v)
    if isinstance(result, tuple):
        conv_matrix, conv_bc = result[0], result[1]
        assert conv_matrix.shape[0] > 0
        assert conv_bc.shape[0] > 0
    else:
        assert result.shape[0] > 0

def test_tvd_limiters():
    c = np.linspace(-1, 2, 10)
    x_c = 0.5
    x_d = 0.75
    for limiter in [upwind, minmod, osher, clam, muscl, smart, stoic, vanleer]:
        result = limiter(c, x_c, x_d)
        assert result.shape == c.shape
