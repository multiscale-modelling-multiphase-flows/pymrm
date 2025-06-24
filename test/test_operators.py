import pytest
import numpy as np
from pymrm.operators import construct_grad, construct_grad_int, construct_grad_bc, construct_div

def test_construct_grad_and_div():
    shape = (10,)
    x_f = np.linspace(0, 1, shape[0] + 1)
    grad, grad_bc = construct_grad(shape, x_f)
    div = construct_div(shape, x_f)
    assert grad.shape[0] > 0
    assert div.shape[0] > 0

def test_construct_grad_int():
    shape = (10,)
    x_f = np.linspace(0, 1, shape[0] + 1)
    grad_int = construct_grad_int(shape, x_f)
    assert grad_int.shape[0] > 0

def test_construct_grad_bc():
    shape = (10,)
    x_f = np.linspace(0, 1, shape[0] + 1)
    bc = ({'a': 0, 'b': 1, 'd': 1}, {'a': 1, 'b': 0, 'd': 0})
    grad_matrix, grad_bc = construct_grad_bc(shape, x_f, bc=bc)
    assert grad_matrix.shape[0] > 0
    assert grad_bc.shape[0] > 0

def test_construct_div():
    shape = (10,)
    x_f = np.linspace(0, 1, shape[0] + 1)
    div = construct_div(shape, x_f)
    assert div.shape[0] > 0
