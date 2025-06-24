import pytest
import numpy as np
from pymrm.helpers import unwrap_bc_coeff
from pymrm import construct_coefficient_matrix

def test_unwrap_bc_coeff():
    shape = (5,)
    bc_coeff = [1, 2, 3, 4, 5]
    result = unwrap_bc_coeff(shape, bc_coeff)
    assert isinstance(result, np.ndarray)
    assert result.shape[-1] == 5

def test_construct_coefficient_matrix():
    coeffs = np.arange(5.0)
    mat = construct_coefficient_matrix(coeffs)
    assert mat.shape[0] == mat.shape[1]
