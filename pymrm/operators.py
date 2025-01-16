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
from pymrm.helpers import unwrap_bc
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

    if bc == (None, None):
        shape_f = shape.copy()
        if axis < 0:
            axis += len(shape)
        shape_f[axis] += 1
        grad_bc = csc_array(shape=(math.prod(shape_f), 1))
    else:
        grad_matrix_bc, grad_bc = construct_grad_bc(
            shape, x_f, x_c, bc, axis)
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
    shape_t = [math.prod(shape[:axis]),  math.prod(shape[axis:axis+1]), math.prod(shape[axis + 1:])]

    i_f = (shape_t[1]+1) * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1, 1) + shape_t[2] * np.arange(shape_t[1]).reshape((
        1, -1, 1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1, 1)) + np.array([0, shape_t[2]]).reshape((1, 1, 1, -1))

    if x_c is None:
        x_c = 0.5*(x_f[:-1] + x_f[1:])

    dx_inv = np.tile(
        1 / (x_c[1:] - x_c[:-1]).reshape((1, -1, 1)), (shape_t[0], 1, shape_t[2]))
    values = np.empty(i_f.shape)
    values[:, 0, :, 0] = np.zeros((shape_t[0], shape_t[2]))
    values[:, 1:, :, 0] = dx_inv
    values[:, :-1, :, 1] = -dx_inv
    values[:, -1, :, 1] = np.zeros((shape_t[0], shape_t[2]))
    grad_matrix = csc_array((values.ravel(), i_f.ravel(), range(0, i_f.size + 1, 2)),
                            shape=(shape_t[0]*(shape_t[1]+1)*shape_t[2], shape_t[0]*shape_t[1]*shape_t[2]))
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
    if axis < 0:
        axis += len(shape)
    shape_f = list(shape)
    shape_t = [math.prod(shape_f[0:axis]), math.prod(
        shape_f[axis:axis+1]), math.prod(shape_f[axis+1:])]
    shape_f[axis] = shape_f[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_t[1] + 1
    shape_bc = shape_f.copy()
    shape_bc[axis] = 1
    shape_bc_d = [shape_t[0], shape_t[2]]

    # Handle special case with one cell in the dimension axis.
    # This is convenient e.g. for flexibility where you can choose not to
    # spatially discretize a direction, but still use a BC, e.g. with a mass transfer coefficient
    # It is a bit subtle because in this case the two opposite faces influence each other
    if shape_t[1] == 1:
        a, b, d = [None]*2, [None]*2, [None]*2
        # Get a, b, and d for left bc from dictionary
        a[0], b[0], d[0] = unwrap_bc(shape, bc[0])
        # Get a, b, and d for right bc from dictionary
        a[1], b[1], d[1] = unwrap_bc(shape, bc[1])
        if x_c is None:
            x_c = 0.5*(x_f[0:-1] + x_f[1:])
        i_c = shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1)) + np.array(
            [0, 0]).reshape((1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        i_f = shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1)) + shape_t[2] * np.array([0, 1]).reshape((
            1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        values = np.zeros(shape_f_t)
        alpha_1 = (x_f[1] - x_f[0]) / (
            (x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2_left = (x_c[0] - x_f[0]) / (
            (x_f[1] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_0_left = alpha_1 - alpha_2_left
        alpha_2_right = -(x_c[0] - x_f[1]) / (
            (x_f[0] - x_f[1]) * (x_f[0] - x_c[0]))
        alpha_0_right = alpha_1 - alpha_2_right
        fctr = ((b[0] + alpha_0_left * a[0]) * (b[1] +
                                            alpha_0_right * a[1]) - alpha_2_left * alpha_2_right * a[0] * a[1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        value = alpha_1 * \
            b[0] * (a[1] * (alpha_0_right - alpha_2_left) + b[1]) * \
            fctr + np.zeros(shape)
        values[:, 0, :] = np.reshape(value, shape_bc_d)
        value = alpha_1 * \
            b[1] * (a[0] * (-alpha_0_left + alpha_2_right) - b[0]) * \
            fctr + np.zeros(shape)
        values[:, 1, :] = np.reshape(value, shape_bc_d)

        i_f_bc = shape_f_t[1] * shape_f_t[2] * np.arange(shape_f_t[0]).reshape((-1, 1, 1)) + shape_f_t[2] * np.array(
            [0, shape_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(shape_f_t[2]).reshape((1, 1, -1))
        values_bc = np.zeros((shape_t[0], 2, shape_t[2]))
        value = ((a[1] * (-alpha_0_left * alpha_0_right + alpha_2_left * alpha_2_right) - alpha_0_left *
                 b[1]) * d[0] - alpha_2_left * b[0] * d[1]) * fctr + np.zeros(shape_bc)
        values_bc[:, 0, :] = np.reshape(value, shape_bc_d)
        value = ((a[0] * (+alpha_0_left * alpha_0_right - alpha_2_left * alpha_2_right) + alpha_0_right *
                 b[0]) * d[1] + alpha_2_right * b[1] * d[0]) * fctr + np.zeros(shape_bc)
        values_bc[:, 1, :] = np.reshape(value, shape_bc_d)
    else:
        i_c = shape_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1) + shape_t[2] * np.array([0, 1, shape_t[1]-2, shape_t[1]-1]).reshape((
            1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        i_f = shape_f_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1) + shape_t[2] * np.array([0, 0, shape_f_t[1]-1, shape_f_t[1]-1]).reshape((
            1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        i_f_bc = shape_f_t[1] * shape_f_t[2] * np.arange(shape_f_t[0]).reshape((-1, 1, 1)) + shape_f_t[2] * np.array(
            [0, shape_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(shape_f_t[2]).reshape((1, 1, -1))
        values_bc = np.zeros((shape_t[0], 2, shape_t[2]))
        values = np.zeros((shape_t[0], 4, shape_t[2]))
        if x_c is None:
            x_c = 0.5*np.array([x_f[0] + x_f[1], x_f[1] + x_f[2],
                                        x_f[-3] + x_f[-2], x_f[-2] + x_f[-1]])

        # Get a, b, and d for left bc from dictionary
        a, b, d = unwrap_bc(shape, bc[0])
        alpha_1 = (x_c[1] - x_f[0]) / (
            (x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_2 = (x_c[0] - x_f[0]) / (
            (x_c[1] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_0 = alpha_1 - alpha_2
        b = b / alpha_0
        fctr = (a + b)
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        b_fctr = b * fctr
        b_fctr = b_fctr + np.zeros(shape_bc)
        b_fctr = np.reshape(b_fctr, shape_bc_d)
        d_fctr = d * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        values[:, 0, :] = b_fctr * alpha_1
        values[:, 1, :] = -b_fctr * alpha_2
        values_bc[:, 0, :] = -d_fctr

        # Get a, b, and d for right bc from dictionary
        a, b, d = unwrap_bc(shape, bc[1])
        alpha_1 = -(x_c[-2] - x_f[-1]) / (
            (x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_2 = -(x_c[-1] - x_f[-1]) / (
            (x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_0 = alpha_1 - alpha_2
        b = b / alpha_0
        fctr = (a + b)
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        b_fctr = b * fctr
        b_fctr = b_fctr + np.zeros(shape_bc)
        b_fctr = np.reshape(b_fctr, shape_bc_d)
        d_fctr = d * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        values[:, -2, :] = b_fctr * alpha_2
        values[:, -1, :] = -b_fctr * alpha_1
        values_bc[:, -1, :] = d_fctr

    grad_matrix = csc_array((values.ravel(), (i_f.ravel(), i_c.ravel())),
                            shape=(math.prod(shape_f_t), math.prod(shape_t)))
    grad_bc = csc_array((values_bc.ravel(), i_f_bc.ravel(), [
                         0, i_f_bc.size]), shape=(math.prod(shape_f_t), 1))
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
        shape_f = [shape]
        shape = (shape, )
    else:
        shape_f = list(shape)
        shape = tuple(shape)
    x_f = generate_grid(shape[axis], x_f)
    if axis < 0:
        axis += len(shape)
    shape_t = [math.prod(shape_f[0:axis]), math.prod(
        shape_f[axis:axis + 1]), math.prod(shape_f[axis + 1:])]
    shape_f[axis] += 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] += 1

    i_f = (
        shape_f_t[1] * shape_f_t[2] *
        np.arange(shape_t[0]).reshape((-1, 1, 1, 1))
        + shape_f_t[2] * np.arange(shape_t[1]).reshape((1, -1, 1, 1))
        + np.arange(shape_t[2]).reshape((1, 1, -1, 1))
        + np.array([0, shape_t[2]]).reshape((1, 1, 1, -1))
    )

    if callable(nu):
        area = nu(x_f).ravel()
        inv_sqrt3 = 1 / np.sqrt(3)
        x_f_r = x_f.ravel()
        dx_f = x_f_r[1:] - x_f_r[:-1]
        dvol_inv = 1 / (
            (nu(x_f_r[:-1] + (0.5 - 0.5 * inv_sqrt3) * dx_f)
             + nu(x_f_r[:-1] + (0.5 + 0.5 * inv_sqrt3) * dx_f))
            * 0.5 * dx_f
        )
    elif nu == 0:
        area = np.ones(shape_f_t[1])
        dvol_inv = 1 / (x_f[1:] - x_f[:-1])
    else:
        area = np.power(x_f.ravel(), nu)
        vol = area * x_f.ravel() / (nu + 1)
        dvol_inv = 1 / (vol[1:] - vol[:-1])

    values = np.empty((shape_t[1], 2))
    values[:, 0] = -area[:-1] * dvol_inv
    values[:, 1] = area[1:] * dvol_inv
    values = np.tile(values.reshape((1, -1, 1, 2)),
                     (shape_t[0], 1, shape_t[2]))

    num_cells = np.prod(shape_t, dtype=int)
    div_matrix = csc_array(
        (values.ravel(), (np.repeat(np.arange(num_cells), 2),
                          i_f.ravel())),
        shape=(num_cells, np.prod(shape_f_t, dtype=int))
    )
    div_matrix.sort_indices()
    return div_matrix