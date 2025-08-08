import pytest
import numpy as np
from pymrm.interpolate import (
    interp_stagg_to_cntr, interp_cntr_to_stagg, interp_cntr_to_stagg_tvd,
    create_staggered_array, compute_boundary_values, construct_boundary_value_matrices
)
from pymrm.convect import upwind

def test_interp_stagg_to_cntr():
    x_f = np.linspace(0.0, 1.0, 11)
    arr = 10*x_f + 1.0
    result = interp_stagg_to_cntr(arr, x_f)
    assert result.shape[0] == 10

def test_interp_cntr_to_stagg():
    arr = np.arange(10.0)
    x_f = np.linspace(0, 1, 11)
    result = interp_cntr_to_stagg(arr, x_f)
    assert result.shape[0] == 11

def test_interp_cntr_to_stagg_tvd():
    arr = np.arange(10.0)
    x_f = np.linspace(0, 1, 11)
    x_c = np.linspace(0.05, 0.95, 10)
    bc = ({'a': 0, 'b': 1, 'd': 1}, {'a': 1, 'b': 0, 'd': 0})
    v = 1.0
    result, _ = interp_cntr_to_stagg_tvd(arr, x_f, x_c, bc, v, upwind)
    assert result.shape[0] == 11

def test_create_staggered_array():
    arr = np.arange(10.0)
    shape = (10,)
    x_f = np.linspace(0, 1, 11)
    x_c = np.linspace(0.05, 0.95, 10)
    result = create_staggered_array(arr, shape, 0, x_f=x_f, x_c=x_c)
    assert result.shape[0] == 11

def test_compute_boundary_values():
    tol = 1e-12
    num_x = 2
    x_f = np.linspace(0.0, 1.0, num_x + 1)
    c = 0.5 * (x_f[1:] + x_f[:-1]).copy()

    # Left boundary condition
    bc_left = {'a': -1, 'b': 1, 'd': 1}
    boundary_value, boundary_grad = compute_boundary_values(c, x_f, bc=bc_left, bound_id=0)
    assert abs(boundary_value) < tol
    assert abs(boundary_grad - 1.0) < tol

    # Right boundary condition
    bc_right = {'a': 1, 'b': 1, 'd': 2}
    boundary_value, boundary_grad = compute_boundary_values(c, x_f, bc=bc_right, bound_id=1)
    assert abs(boundary_value - 1.0) < tol
    assert abs(boundary_grad - 1.0) < tol

def test_construct_boundary_value_matrices():
    tol = 1e-12
    num_x = 2
    x_f = np.linspace(0.0, 1.0, num_x + 1)
    c = 0.5 * (x_f[1:] + x_f[:-1]).copy()

    # Left boundary condition
    bc_left = {'a': -1, 'b': 1, 'd': 1}
    matrix, matrix_bc = construct_boundary_value_matrices(c.shape, x_f, bc=bc_left, bound_id=0)
    boundary_value = matrix @ c.reshape((-1, 1)) + matrix_bc
    assert np.allclose(boundary_value, 0.0, atol=tol)

    # Right boundary condition
    bc_right = {'a': 1, 'b': 1, 'd': 2}
    matrix, matrix_bc = construct_boundary_value_matrices(c.shape, x_f, bc=bc_right, bound_id=1)
    boundary_value = matrix @ c.reshape((-1, 1)) + matrix_bc
    assert np.allclose(boundary_value, 1.0, atol=tol)
