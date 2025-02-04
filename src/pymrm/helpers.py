"""
Helper functions for common operations in the pymrm package.

This module provides utility functions that support various operations across
different submodules, such as boundary condition handling, constructing coefficient
matrices, and creating staggered arrays for finite volume discretizations.

Functions:
- unwrap_bc: Process boundary condition dictionaries for numerical schemes.
- construct_coefficient_matrix: Create diagonal coefficient matrices.
"""

import numpy as np
from scipy.sparse import diags, csc_array

def unwrap_bc(shape, bc):
    """
    Unwrap the boundary conditions for a given shape.

    Args:
        shape (tuple): Shape of the domain.
        bc (dict): Boundary conditions in the form {'a': ..., 'b': ..., 'd': ...}.

    Returns:
        tuple: Unwrapped boundary conditions (a, b, d).
    """
    if not isinstance(shape, (list, tuple)):
        lgth_shape = 1
    else:
        lgth_shape = len(shape)

    if bc is None:
        a = np.zeros((1,) * lgth_shape)
        b = np.zeros((1,) * lgth_shape)
        d = np.zeros((1,) * lgth_shape)
    else:
        a = np.array(bc['a'])
        a = a[(..., *([np.newaxis] * (lgth_shape - a.ndim)))]
        b = np.array(bc['b'])
        b = b[(..., *([np.newaxis] * (lgth_shape - b.ndim)))]
        d = np.array(bc['d'])
        d = d[(..., *([np.newaxis] * (lgth_shape - d.ndim)))]
    return a, b, d


def construct_coefficient_matrix(coefficients, shape=None, axis=None):
    """
    Construct a diagonal matrix with coefficients on its diagonal.

    Args:
        coefficients (ndarray or list): Values of the coefficients.
        shape (tuple, optional): Shape of the multidimensional field.
        axis (int, optional): Axis for broadcasting in staggered grids.

    Returns:
        csc_array: Sparse diagonal matrix of coefficients.
    """
    if shape is None:
        coeff_matrix = csc_array(diags(coefficients.flatten(), format='csc'))
    else:
        shape = list(shape)
        if axis is not None:
            shape[axis] += 1
        coefficients_copy = np.array(coefficients)
        reps = [shape[i] // coefficients_copy.shape[i] if i < len(coefficients_copy.shape) else shape[i] for i in range(len(shape))]
        coefficients_copy = np.tile(coefficients_copy, reps)
        coeff_matrix = csc_array(diags(coefficients_copy.flatten(), format='csc'))
    return coeff_matrix

