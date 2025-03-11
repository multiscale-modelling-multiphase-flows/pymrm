"""
Helper functions for common operations in the pymrm package.

This module provides utility functions that support various operations across
different submodules, such as boundary condition handling, constructing coefficient
matrices, and creating staggered arrays for finite volume discretizations.

Functions:
- unwrap_bc_coeff: Process boundary coefficients for numerical schemes.
- construct_coefficient_matrix: Create diagonal coefficient matrices.
"""

import numpy as np
from scipy.sparse import diags, csc_array

def unwrap_bc_coeff(shape, bc_coeff, axis=0):
    """
    Unwrap the boundary conditions for a given shape.

    Args:
        shape (tuple): Shape of the domain.
        bc_coeff (dict): Boundary condition coefficient, e.g., a, b, d.

    Returns:
        numpy array: Unwrapped boundary condition coefficient
    """
    if not isinstance(shape, (list, tuple)):
        lgth_shape = 1
    else:
        lgth_shape = len(shape)

    a = np.array(bc_coeff)
    if a.ndim == (lgth_shape-1):
        a = np.expand_dims(a,axis=axis)
    elif a.ndim != lgth_shape:
        shape_a = (1,)* (lgth_shape - a.ndim) + a.shape
        a = a.reshape(shape_a)
    return a


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
        coeff_matrix = csc_array(diags(coefficients.ravel(), format='csc'))
    else:
        if axis is not None:
            shape = tuple(s if i != axis else s + 1 for i, s in enumerate(shape))
        coefficients_copy = np.array(coefficients)
        shape_coeff = (1,)* (len(shape) - coefficients_copy.ndim) + coefficients_copy.shape
        coefficients_copy = coefficients_copy.reshape(shape_coeff)
        coefficients_copy = np.broadcast_to(coefficients_copy, shape)
        coeff_matrix = csc_array(diags(coefficients_copy.ravel(), format='csc'))
    return coeff_matrix

