"""
Helper functions for common operations in the pymrm package.

This module provides utility functions that support various operations across
different submodules, such as boundary condition handling, constructing coefficient
matrices, and creating staggered arrays for finite volume discretizations.

Functions:
- unwrap_bc: Process boundary condition dictionaries for numerical schemes.
- construct_coefficient_matrix: Create diagonal coefficient matrices.
- create_staggered_array: Generate staggered arrays for face-centered values.
"""

import numpy as np
from scipy.sparse import diags, csc_array
from pymrm.interpolate import interp_cntr_to_stagg  # Correctly imported from interpolate.py

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


def create_staggered_array(array, shape, axis, x_f=None, x_c=None):
    """
    Create a staggered array by interpolating values to face-centered positions.

    Args:
        array (ndarray): The array to be staggered.
        shape (tuple): Shape of the non-staggered cell-centered field.
        axis (int): Axis along which staggering is applied.
        x_f (ndarray, optional): Face positions. Default is None.
        x_c (ndarray, optional): Cell positions. Default is None.

    Returns:
        ndarray: The staggered array aligned with face positions.
    """
    if not isinstance(shape, (list, tuple)):
        shape_f = [shape]
    else:
        shape_f = list(shape)

    if axis < 0:
        axis += len(shape)
    shape_f[axis] += 1

    array = np.asarray(array)
    if array.shape == tuple(shape_f):
        return array

    if array.size == 1:
        array = np.full(shape_f, array)
    else:
        # Interpolate to staggered positions if necessary
        if array.shape == tuple(shape):
            array = interp_cntr_to_stagg(array, x_f, x_c, axis)
        array = np.broadcast_to(array, shape_f)
    
    return array

