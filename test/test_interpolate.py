import pytest
import numpy as np
from pymrm.interpolate import (
    interp_stagg_to_cntr, interp_cntr_to_stagg, interp_cntr_to_stagg_tvd,
    create_staggered_array, compute_boundary_values
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
    arr = np.arange(10.0).reshape((10, 1, 1))
    x_f = np.linspace(0, 1, 11)
    x_c = np.linspace(0.05, 0.95, 10)
    bc = ({'a': 1, 'b': 0, 'd': 0}, {'a': 1, 'b': 0, 'd': 0})
    result = compute_boundary_values(arr, x_f, x_c=x_c, bc=bc, axis=0)
    assert isinstance(result, tuple)
