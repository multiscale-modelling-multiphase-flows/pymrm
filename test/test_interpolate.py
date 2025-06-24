import pytest
import numpy as np
from pymrm import interp_stagg_to_cntr, interp_cntr_to_stagg, interp_cntr_to_stagg_tvd, create_staggered_array, compute_boundary_values

def test_interp_stagg_to_cntr():
    arr = np.arange(10.0)
    result = interp_stagg_to_cntr(arr)
    assert result.shape[0] == arr.shape[0] - 1 or result.shape[0] == arr.shape[0]

def test_interp_cntr_to_stagg():
    arr = np.arange(10.0)
    result = interp_cntr_to_stagg(arr)
    assert result.shape[0] == arr.shape[0] + 1 or result.shape[0] == arr.shape[0]

def test_interp_cntr_to_stagg_tvd():
    arr = np.arange(10.0)
    result, _ = interp_cntr_to_stagg_tvd(arr, np.arange(11.0), np.arange(10.0), ({'a':0,'b':1,'d':1},{'a':1,'b':0,'d':0}), 1.0, upwind)
    assert result.shape[0] == arr.shape[0] + 1

def test_create_staggered_array():
    arr = np.arange(10.0)
    result = create_staggered_array(arr, (10,), 0)
    assert result.shape[0] == 11 or result.shape[0] == 10

def test_compute_boundary_values():
    arr = np.arange(10.0)
    result = compute_boundary_values(arr, (10,), 0)
    assert result.shape[0] == 2
