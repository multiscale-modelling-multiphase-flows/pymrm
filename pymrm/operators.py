"""
Operators Submodule for pymrm

This submodule provides numerical operators for spatial discretization, including gradient and divergence operatorss. 
These operators are essential for constructing finite difference and finite volume schemes used in multiphase reactor modeling.

Functions:
- construct_grad: Constructs the gradient matrix for spatial differentiation.
- construct_grad_int: Constructs the gradient matrix for internal faces.
- construct_grad_bc: Constructs the gradient matrix for boundary faces.
- construct_div: Constructs the divergence matrix for flux calculations.

Dependencies:
- numpy
- scipy.sparse
- pymrm.grid (for optional grid generation)
- pymrm.helper (for boundary condition handling)
"""

import math
import numpy as np
from scipy.sparse import csc_array
from pymrm.helper import unwrap_bc
from pymrm.grid import generate_grid


def construct_grad(shape, x_f, x_c=None, bc=(None, None), axis=0):
    """
    Construct the gradient matrix.

    Args:
        shape (tuple): Shape of the domain.
        x_f (ndarray): Face positions.
        x_c (ndarray, optional): Cell center coordinates. If not provided, it is calculated.
        bc (tuple, optional): Boundary conditions. Default is (None, None).
        axis (int, optional): Axis of differentiation. Default is 0.

    Returns:
        csc_array: Gradient matrix.
        csc_array: Gradient contribution from boundary conditions.
    """
    if isinstance(shape, int):
        shape = (shape, )
    x_f, x_c = generate_grid(shape[axis], x_f, generate_x_c=True, x_c=x_c)
    grad_matrix = construct_grad_int(shape, x_f, x_c, axis)

    if bc is None:
        grad_bc = csc_array((math.prod(shape), 1))
    else:
        grad_matrix_bc, grad_bc = construct_grad_bc(shape, x_f, x_c, bc, axis)
        grad_matrix += grad_matrix_bc

    return grad_matrix, grad_bc


def construct_grad_int(shape, x_f, x_c=None, axis=0):
    """
    Construct the gradient matrix for internal faces.

    Args:
        shape (tuple): Shape of the domain.
        x_f (ndarray): Face coordinates.
        x_c (ndarray, optional): Cell center coordinates.
        axis (int, optional): Axis of differentiation. Default is 0.

    Returns:
        csc_array: Gradient matrix.
    """
    if axis < 0:
        axis += len(shape)
    shape_t = [math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1:])]

    if x_c is None:
        x_c = 0.5 * (x_f[:-1] + x_f[1:])

    dx_inv = 1 / (x_c[1:] - x_c[:-1])
    values = np.empty((shape_t[0], shape_t[1] + 1, shape_t[2], 2))
    values[:, 1:, :, 0] = dx_inv
    values[:, :-1, :, 1] = -dx_inv

    grad_matrix = csc_array((values.ravel(), np.arange(values.size)), shape=(math.prod(shape), math.prod(shape)))
    return grad_matrix


def construct_grad_bc(shape, x_f, x_c=None, bc=(None, None), axis=0):
    """
    Construct the gradient matrix for boundary faces.

    Args:
        shape (tuple): Shape of the domain.
        x_f (ndarray): Face coordinates.
        x_c (ndarray, optional): Cell center coordinates.
        bc (tuple, optional): Boundary conditions.
        axis (int, optional): Axis of differentiation.

    Returns:
        csc_array: Gradient matrix for boundary conditions.
        csc_array: Contribution of inhomogeneous boundary conditions.
    """
    a, b, d = unwrap_bc(shape, bc[0])
    grad_bc = csc_array((math.prod(shape), 1))
    grad_matrix = csc_array((math.prod(shape), math.prod(shape)))
    return grad_matrix, grad_bc


def construct_div(shape, x_f, nu=0, axis=0):
    """
    Construct the divergence matrix.

    Args:
        shape (tuple): Shape of the domain.
        x_f (ndarray): Face positions.
        nu (int or callable): Geometry factor (0: flat, 1: cylindrical, 2: spherical).
        axis (int, optional): Axis along which divergence is computed. Default is 0.

    Returns:
        csc_array: Divergence matrix.
    """
    if isinstance(shape, int):
        shape = (shape, )
    x_f = generate_grid(shape[axis], x_f)

    if callable(nu):
        area = nu(x_f)
    elif nu == 0:
        area = np.ones_like(x_f)
    else:
        area = np.power(x_f, nu)

    dx_inv = 1 / (x_f[1:] - x_f[:-1])
    values = np.empty_like(dx_inv)
    values[:-1] = -area[:-1] * dx_inv
    values[1:] = area[1:] * dx_inv

    div_matrix = csc_array((values, np.arange(len(values))), shape=(math.prod(shape), math.prod(shape)))
    return div_matrix
\