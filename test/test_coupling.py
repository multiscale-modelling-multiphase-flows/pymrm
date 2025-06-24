import pytest
import numpy as np
from pymrm import translate_indices_to_larger_array, update_csc_array_indices, construct_interface_matrices

def test_translate_indices_to_larger_array():
    indices = np.array([0, 1, 2])
    shape = (3,)
    new_shape = (6,)
    result = translate_indices_to_larger_array(indices, shape, new_shape)
    assert np.all(result >= 0)

def test_update_csc_array_indices():
    from scipy.sparse import csc_matrix
    mat = csc_matrix(np.eye(3))
    shape = (3,)
    new_shape = (6,)
    result = update_csc_array_indices(mat, shape, new_shape)
    assert result.shape[0] == 6
    assert result.shape[1] == 6
