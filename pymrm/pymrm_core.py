"""
Module Name: pymrm
Author: E.A.J.F. Peters, M. Sint Annaland, M. Galanti, D. Rieder
Date: 14/01/2025
License: MIT License
Version: 2.0.0

This module provides functions for multiphase reactor modeling.

Functions:

- construct_grad(shape, x_f, x_c=None, bc=(None, None), axis=0): Construct the gradient matrix.
- construct_grad_int(shape, x_f, x_c=None, axis=0): Construct the gradient matrix for internal faces.
- construct_grad_bc(shape, x_f, x_c=None, bc=(None, None), axis=0): Construct the gradient matrix for boundary faces.
- construct_div(shape, x_f, nu=0, axis=0): Construct the divergence matrix based on the given parameters.
- construct_convflux_upwind(shape, x_f, x_c=None, bc=(None, None), v=1.0, axis=0): Construct the convective flux matrix using the upwind scheme.
- construct_convflux_upwind_int(shape, v=1.0, axis=0): Construct the convective flux matrix for internal faces using the upwind scheme.
- construct_convflux_upwind_bc(shape, x_f, x_c=None, bc=(None, None), v=1.0, axis=0): Construct the convective flux matrix for boundary faces using the upwind scheme.
- construct_coefficient_matrix(coefficients, shape=None, axis=None): Construct a diagonal matrix with coefficients on its diagonal.
- numjac_local(function, initial_values, epsilon_jac=1e-6, axis=-1): Compute the local numerical Jacobian matrix and function values for the given function and initial values.
- newton(function, initial_guess, args=(), tolerance=1.49012e-08, max_iterations=100, solver=None, callback=None): Perform Newton-Raphson iterations to seek the root of the vector-valued function.
- clip_approach(values, function, lower_bounds=0, upper_bounds=None, factor=0): Filter values with lower and upper bounds using an approach factor.
- interp_stagg_to_cntr(staggered_values, x_f, x_c=None, axis=0): Interpolate values at staggered positions to cell-centers using linear interpolation.
- interp_cntr_to_stagg(cell_centered_values, x_f, x_c=None, axis=0): Interpolate values at cell-centers to staggered positions using linear interpolation.
- interp_cntr_to_stagg_tvd(cell_centered_values, x_f, x_c=None, bc=None, v=0, tvd_limiter=None, axis=0): Interpolate values at staggered positions to cell-centers using TVD interpolation.
- create_staggered_array(array, shape, axis, x_f=None, x_c=None): Create a staggered array by staggering values in the specified axis.
- upwind(normalized_c_c, normalized_x_c, normalized_x_d): Apply the upwind TVD limiter to reduce oscillations in numerical schemes.
- minmod(normalized_c_c, normalized_x_c, normalized_x_d): Apply the Minmod TVD limiter to reduce oscillations in numerical schemes.
- osher(normalized_c_c, normalized_x_c, normalized_x_d): Apply the Osher TVD limiter to reduce oscillations in numerical schemes.
- clam(normalized_c_c, normalized_x_c, normalized_x_d): Apply the CLAM TVD limiter to reduce oscillations in numerical schemes.
- muscl(normalized_c_c, normalized_x_c, normalized_x_d): Apply the MUSCL TVD limiter to reduce oscillations in numerical schemes.
- smart(normalized_c_c, normalized_x_c, normalized_x_d): Apply the SMART TVD limiter to reduce oscillations in numerical schemes.
- stoic(normalized_c_c, normalized_x_c, normalized_x_d): Apply the STOIC TVD limiter to reduce oscillations in numerical schemes.
- vanleer(normalized_c_c, normalized_x_c, normalized_x_d): Apply the van Leer TVD limiter to reduce oscillations in numerical schemes.
- non_uniform_grid(left_bound, right_bound, num_points, dx_inf, factor): Generate a non-uniform grid of points in the interval [left_bound, right_bound].
- generate_grid(size, x_f, generate_x_c=False, x_c=None): Generate a grid of face and optionally cell-centered positions.
- unwrap_bc(shape, bc): Unwrap the boundary conditions for a given shape. Mostly used by other functions.

Note: Please refer to the function descriptions for more details on their arguments and usage.
"""

import math
import numpy as np
from scipy.sparse import csc_array, diags, linalg
from scipy.linalg import norm
from scipy.optimize import OptimizeResult


def construct_grad(shape, x_f, x_c=None, bc=(None, None), axis=0):
    """
    Construct the gradient matrix.

    Args:
        shape (tuple): Shape of the domain.
        x_f (ndarray): Face positions.
        x_c (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        bc (tuple, optional): Boundary conditions. Default is (None, None).
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.

    Returns:
        csc_array: Gradient matrix (grad_matrix).
        csc_array: Contribution of the inhomogeneous BC to the gradient (grad_bc).
    """
    # The contributions to the gradient on internal faces that
    # are independent of boundary conditions are computed separately
    # from those on boundary faces. On boundary faces,
    # in case of inhomogeneous boundary conditions,
    # then there can also be constant contribution, grad_bc
    if isinstance(shape, int):
        shape = (shape, )
        shape_f = [shape]
    else:
        shape = tuple(shape)
        shape_f = list(shape)
    x_f, x_c = generate_grid(shape[axis], x_f, generate_x_c=True, x_c=x_c)
    grad_matrix = construct_grad_int(shape, x_f, x_c, axis)

    if bc is None:
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


def construct_grad_int(shape, x_f,  x_c=None, axis=0):
    """
    Construct the gradient matrix for internal faces.

    Args:
        shape (tuple): Shape of the domain.
        x_f (ndarray): Face coordinates.
        x_c (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.

    Returns:
        csc_array: Gradient matrix (grad_matrix).
    """
    # Explanation of implementation:
    # The vectors of unknown are flattened versions of a multi-dimensional array.
    # In this multi-dimensional arrays some dimensions will represent spatial directions, i.e.,
    # the indices are cell indices. The gradient denotes a finite difference discretization
    # of the spatial differentiation in that direction. The direction of differentiation
    # is denoted by 'axis'.
    # For multi-dimenional arrays the following trick is used: Arrays are reshaped to
    # three-dimensional arrays where the middle dimension now corresponds to the direction of differentiation.

    if axis < 0:
        axis += len(shape)
    shape_t = [math.prod(shape[0:axis]), math.prod(
        # shape of the reshaped 3 dimenional array
        shape[axis:axis+1]), math.prod(shape[axis+1:])]
    # The gradient will be represented by a csc array.
    # Each column corresponds to the contribution of a value cell
    # For each column there are two entries corresponding to the two faces, except for the cells near the boundary
    # The indices of these faces are stored in an array i_f, with i_f.shape = [shape_t[0], shape_t[1]+1 , shape_t[2], 2]
    # Note the shape_t[1]+1, because there is 1 more face than cells.
    # The linear index is: i_f[i,j,k,m] = 2*(shape_t[2]*((shape_t[1]+1)*j + k)) + m
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
    Construct the gradient matrix for the boundary faces.

    Args:
        shape (tuple): Shape of the domain.
        x_f (ndarray): Face coordinates.
        x_c (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        bc (tuple, optional): Boundary conditions. Default is (None, None).
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.

    Returns:
        csc_array: Gradient matrix (grad_matrix).
        csc_array: Contribution of the inhomogeneous BC to the gradient (grad_bc).
    """
    # For the explanation of resizing see construct_grad_int
    if axis < 0:
        axis += len(shape)
    shape_f = list(shape)
    shape_t = [math.prod(shape_f[0:axis]), math.prod(
        shape_f[axis:axis+1]), math.prod(shape_f[axis+1:])]
    # Specify shapes of faces (multi-dimentional shape_f and as a triplet shape_f_t)
    shape_f[axis] = shape_f[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_t[1] + 1
    # Specify shapes of boundary quantities
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
    Construct the divergence matrix based on the given parameters.

    Args:
        shape (tuple): Shape of the multi-dimensional array.
        x_f (ndarray): Face positions.
        nu (int or callable): The integer representing geometry (0: flat, 1: cylindrical, 2: spherical). If it is a function, it specifies an area at position x.
        axis (int): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The divergence matrix (div_matrix).
    """

    # Trick: Reshape to triplet shape_t, see compute_grad_int for explanation
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

def construct_convflux_upwind(shape, x_f, x_c=None, bc=(None, None), v=1.0, axis=0):
    """
    Construct the convective flux matrix using the upwind scheme.

    Args:
        shape (tuple): Shape of the multi-dimensional array.
        x_f (ndarray): Face positions.
        x_c (ndarray, optional): Cell positions. If not provided, it will be calculated based on the face array.
        bc (tuple, optional): The boundary conditions. Default is (None, None).
        v (ndarray): Velocities on face positions.
        axis (int, optional): The axis along which the convection takes place. Default is 0.

    Returns:
        csc_array: The convective flux matrix (conv_matrix).
        csc_array: The convective flux matrix for boundary conditions (conv_bc).
    """
    if isinstance(shape, int):
        shape = (shape,)
    x_f, x_c = generate_grid(shape[axis], x_f, generate_x_c=True, x_c=x_c)

    v_f = create_staggered_array(v, shape, axis, x_f=x_f, x_c=x_c)
    conv_matrix = construct_convflux_upwind_int(shape, v_f, axis)
    if (bc is None or bc == (None, None)):
        shape_f = shape.copy()
        shape_f[axis] += 1
        conv_bc = csc_array(shape=(math.prod(shape_f), 1))
    else:
        conv_matrix_bc, conv_bc = construct_convflux_upwind_bc(
            shape, x_f, x_c, bc, v_f, axis)
        conv_matrix += conv_matrix_bc
    return conv_matrix, conv_bc

def create_staggered_array(array, shape, axis, x_f=None, x_c=None):
    """
    Create a staggered array by staggering values in the specified axis. Values are broadcasted if needed.

    Args:
        array (ndarray): The array that needs to be staggered.
        shape (tuple): The shape of the non-staggered cell-centered fields.
        axis (int): The axis that indicates the staggered direction.
        x_f (ndarray, optional): The face positions. Default is None.
        x_c (ndarray, optional): The cell positions. Default is None.

    Returns:
        ndarray: The staggered array.
    """
    if not isinstance(shape, (list, tuple)):
        shape_f = [shape]
    else:
        shape_f = list(shape)
    if axis < 0:
        axis += len(shape)
    shape_f[axis] = shape_f[axis] + 1
    if isinstance(array, (float, int)):
        array = np.array([array])
    else:
        array = np.asarray(array)
    if array.shape == shape_f:
        return array
    if (len(shape) != 1 and array.ndim == 1):
        shape_new = [1]*len(shape)
        if array.size in (shape[axis],shape_f[axis]):
            shape_new[axis] = -1
        else:
            for i in range(len(shape)-1, -1, -1):
                if array.size == shape[axis]:
                    shape_new[i] = shape[i]
                    break
        array = array.reshape(shape_new)
    print(f"array.ndim: {array.ndim}, len(shape): {len(shape)}")
    if array.ndim != len(shape):
        raise ValueError("The array has the wrong number of dimensions.")
    else:
        if (array.shape[axis] == shape[axis]):
            # interpolate to staggered positions
            array_f = interp_cntr_to_stagg(array, x_f, x_c, axis)
        else:
            array_f = np.asarray(array)
        array_f = np.broadcast_to(array_f, shape_f)
    return array_f

def construct_convflux_upwind_int(shape, v=1.0, axis=0):
    """
    Construct the convective flux matrix for internal faces using the upwind scheme.

    Args:
        shape (tuple): Shape of the ndarrays.
        v (float, ndarray): The velocity array.
        axis (int, optional): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The convective flux matrix (conv_matrix).
    """
    if not isinstance(shape, (list, tuple)):
        shape_f = [shape]
    else:
        shape_f = list(shape)
    if axis < 0:
        axis += len(shape)
    shape_t = [math.prod(shape_f[0:axis]), math.prod(
        shape_f[axis:axis+1]), math.prod(shape_f[axis+1:])]

    shape_f[axis] = shape_f[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_t[1] + 1

    if (isinstance(v, (float, int))):
        v_t = np.broadcast_to(np.array(v), shape_f_t)
    else:
        v_t = v.reshape(shape_f_t)
    fltr_v_pos = (v_t > 0)
    i_f = (shape_t[1]+1) * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1) + shape_t[2] * np.arange(1, shape_t[1]).reshape((
        1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
    i_c = shape_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1)) + shape_t[2] * np.arange(1, shape_t[1]).reshape((
        1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
    i_c = i_c - shape_t[2] * fltr_v_pos[:, 1:-1, :]
    conv_matrix = csc_array((v_t[:, 1:-1, :].ravel(), (i_f.ravel(), i_c.ravel())), shape=(
        math.prod(shape_f_t), math.prod(shape_t)))
    conv_matrix.sort_indices()
    return conv_matrix

def construct_convflux_upwind_bc(shape, x_f, x_c=None, bc=(None, None), v=1.0, axis=0):
    """
    Construct the convective flux matrix for the boundary faces using the upwind scheme.

    Args:
        shape (tuple): Shape of the multi-dimensional array.
        x_f (ndarray): Face positions.
        x_c (ndarray, optional): Cell-centered positions. If not provided, it will be calculated based on the face array.
        bc (tuple, optional): A tuple containing the left and right boundary conditions. Default is (None, None).
        v (float, ndarray): The velocity array.
        axis (int, optional): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The convective flux matrix (conv_matrix).
        csc_array: The convective flux matrix for boundary conditions (conv_bc).
    """

    # Trick: Reshape to triplet shape_t
    if not isinstance(shape, (list, tuple)):
        shape_f = [shape]
    else:
        shape_f = list(shape)
    if axis < 0:
        axis += len(shape)
    shape_t = [math.prod(shape_f[0:axis]), math.prod(
        shape_f[axis:axis+1]), math.prod(shape_f[axis+1:])]

    # Create face arrays
    shape_f[axis] = shape_f[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_t[1] + 1

    # Create boundary quantity shapes
    shape_bc = shape_f.copy()
    shape_bc[axis] = 1
    shape_bc_d = [shape_t[0], shape_t[2]]

    # Handle special case with one cell in the dimension axis
    if shape_t[1] == 1:
        a, b, d = [None]*2, [None]*2, [None]*2
        a[0], b[0], d[0] = unwrap_bc(shape, bc[0])
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
        values = np.empty((shape_t[0], 2, shape_t[2]))
        values[:, 0, :] = ((alpha_1 * a[0] * (a[1] * (alpha_0_right - alpha_2_left) + b[1])
                           * fctr + np.zeros(shape)).reshape(shape_bc_d))
        values[:, 1, :] = ((alpha_1 * a[1] * (a[0] * (alpha_0_left - alpha_2_right) + b[0])
                           * fctr + np.zeros(shape)).reshape(shape_bc_d))
        
        i_f_bc = shape_f_t[1] * shape_f_t[2] * np.arange(shape_f_t[0]).reshape((-1, 1, 1)) + shape_f_t[2] * np.array(
            [0, shape_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(shape_f_t[2]).reshape((1, 1, -1))
        values_bc = np.empty((shape_t[0], 2, shape_t[2]))
        values_bc = np.zeros((shape_t[0], 2, shape_t[2]))
        values_bc[:, 0, :] = ((((a[1] * alpha_0_right + b[1]) * d[0] - alpha_2_left * a[0] * d[1])
                              * fctr + np.zeros(shape_bc)).reshape(shape_bc_d))
        values_bc[:, 1, :] = ((((a[0] * alpha_0_left + b[0]) * d[1] - alpha_2_right * a[1] * d[0])
                              * fctr + np.zeros(shape_bc)).reshape(shape_bc_d))

        if isinstance(v, (float, int)):
            values *= v
            values_bc *= v
        else:
            slicer= [slice(None)]*len(shape)
            slicer[axis] = [0,-1]
            shape_f_b = shape_f
            shape_f_b[axis] = 2
            values = values.reshape(shape_f_b)
            values *= v[tuple(slicer)]
            values_bc = values_bc.reshape(shape_f_b)
            values_bc *= v[tuple(slicer)]
        conv_matrix = csc_array((values.ravel(), (i_f.ravel(), i_c.ravel())), shape=(
            math.prod(shape_f_t), math.prod(shape_t)))
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
        fctr = (alpha_0 * a + b)
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a * fctr
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        values[:, 0, :] = (a_fctr * alpha_1)
        values[:, 1, :] = -a_fctr * alpha_2
        values_bc[:, 0, :] = d_fctr

        # Get a, b, and d for right bc from dictionary
        a, b, d = unwrap_bc(shape, bc[1])
        alpha_1 = -(x_c[-2] - x_f[-1]) / (
            (x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_2 = -(x_c[-1] - x_f[-1]) / (
            (x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a + b)
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a * fctr
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        values[:, -1, :] = (a_fctr * alpha_1)
        values[:, -2, :] = -a_fctr * alpha_2
        values_bc[:, -1, :] = d_fctr
        if isinstance(v, (float, int)):
            values *= v
            values_bc *= v
        else:
            slicer= [slice(None)]*len(shape)
            slicer[axis] = [0,0,-1,-1]
            shape_f_b = shape_f
            shape_f_b[axis] = 4
            values = values.reshape(shape_f_b)
            values *= v[tuple(slicer)]
            shape_f_b[axis] = 2
            slicer[axis] = [0,-1]
            values_bc = values_bc.reshape(shape_f_b)
            values_bc *= v[tuple(slicer)]
        conv_matrix = csc_array((values.ravel(), (i_f.ravel(), i_c.ravel())), shape=(
            math.prod(shape_f_t), math.prod(shape_t)))
        conv_matrix.sort_indices()
    conv_bc = csc_array((values_bc.ravel(), i_f_bc.ravel(), [
                         0, i_f_bc.size]), shape=(math.prod(shape_f_t), 1))
    return conv_matrix, conv_bc


def construct_coefficient_matrix(coefficients, shape=None, axis=None):
    """
    Construct a diagonal matrix with coefficients on its diagonal.

    Args:
        coefficients (ndarray, list): Values of the coefficients in a field.
        shape (tuple, optional): Shape of the multidimensional field. With this option, some of the dimensions of coefficients can be chosen singleton.
        axis (int, optional): In case of broadcasting along 'axis', used shape will be shape[axis+1] (can be useful for face-values).

    Returns:
        csc_array: Matrix with coefficients on its diagonal (coeff_matrix).
    """
    if shape is None:
        shape = coefficients.shape
        coeff_matrix = csc_array(diags(coefficients.flatten(), format='csc'))
    else:
        shape = list(shape)
        if axis is not None:
            shape[axis] += 1
        coefficients_copy = np.array(coefficients)
        reps = [shape[i] // coefficients_copy.shape[i] if i <
                len(coefficients_copy.shape) else shape[i] for i in range(len(shape))]
        coefficients_copy = np.tile(coefficients_copy, reps)
        coeff_matrix = csc_array(diags(coefficients_copy.flatten(), format='csc'))
    return coeff_matrix


def numjac_local(function, initial_values, epsilon_jac=1e-6, axis=-1):
    """
    Compute the local numerical Jacobian matrix and function values for the given function and initial values.

    Args:
        function (callable): The function for which to compute the Jacobian.
        initial_values (ndarray): The value at which the Jacobian should be evaluated.
        epsilon_jac (float, optional): The perturbation value for computing the Jacobian. Defaults to 1e-6.
        axis (int or tuple/list, optional): The axis or axes along which components are mutually coupled. Default is -1.

    Returns:
        csc_array: The Jacobian matrix.
        ndarray: The function values.
    """
    shape = initial_values.shape
    if isinstance(axis, int):
        axis = (axis,)
    # Normalize negative indices
    axis = [a + len(shape) if a < 0 else a for a in axis]
    # Calculate the shape tuple for the reshaping operation
    middle_dim = math.prod([shape[a] for a in axis])
    shape_t = [math.prod(shape[:min(axis)]), middle_dim,
               math.prod(shape[max(axis)+1:])]

    values = np.zeros((*shape_t, shape_t[1]))
    i = shape_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1, 1)) + np.zeros((1, shape_t[1], 1, 1)) + np.arange(
        shape_t[2]).reshape((1, 1, -1, 1)) + shape_t[2] * np.arange(shape_t[1]).reshape((1, 1, 1, -1))
    function_value = function(initial_values,).reshape(shape_t)
    initial_values = initial_values.reshape(shape_t)
    dc = -epsilon_jac * np.abs(initial_values)  # relative deviation
    # If dc is small use absolute deviation
    dc[dc > (-epsilon_jac)] = epsilon_jac
    dc = (initial_values + dc) - initial_values
    for k in range(shape_t[1]):
        initial_values_perturb = np.copy(initial_values)
        initial_values_perturb[:, k,
                               :] = initial_values_perturb[:, k, :] + dc[:, k, :]
        function_perturb = function(
            initial_values_perturb.reshape(shape)).reshape(shape_t)
        values[:, k, :, :] = np.transpose(
            (function_perturb - function_value) / dc[:, [k], :], (0, 2, 1))
    jac_matrix = csc_array((values.flatten(), i.flatten(), np.arange(
        0, i.size + shape_t[1], shape_t[1])), shape=(np.prod(shape_t), np.prod(shape_t)))
    return function_value.reshape(shape), jac_matrix


def newton(function, initial_guess, args=(), tolerance=1.49012e-08, max_iterations=100, solver=None, callback=None):
    """
    Perform Newton-Raphson iterations to seek the root of the vector-valued function.

    Args:
        function (callable): Function that provides the vector-valued function 'g' of which the roots are sought and, as second argument, its Jacobian.
        initial_guess (ndarray): Vector containing the initial guesses for the values of x.
        args (tuple, optional): Extra arguments passed to function.
        tolerance (float, optional): Tolerance used for convergence in the Newton-Raphson iteration. Default is 1.49012e-08.
        max_iterations (int, optional): Maximum number of iterations used in Newton-Raphson procedure. Default is 100.
        solver (str, optional): The method to solve the linearized equations. Default is None.
        callback (callable, optional): Optional callback function. It is called on every iteration as callback(x, f) where x is the current solution and f the corresponding residual.

    Returns:
        OptimizeResult: The solution represented as an OptimizeResult object.
    """
    n = initial_guess.size
    if solver is None:
        if n < 50000:
            solver = 'spsolve'
        else:
            solver = 'bicgstab'

    if solver == 'spsolve':
        linsolver = linalg.spsolve
    elif solver == 'lu':
        def linsolver(jac_matrix, g):
            Jac_lu = linalg.splu(jac_matrix)
            dx_neg = Jac_lu.solve(g)
            return dx_neg
    elif solver == 'cg':
        def linsolver(jac_matrix, g):
            # determine pre-conditioner M via ILU factorization
            Jac_iLU = linalg.spilu(jac_matrix)
            M = linalg.LinearOperator((n, n), Jac_iLU.solve)
            dx_neg, info = linalg.cg(jac_matrix, g, np.zeros(
                n), tol=1e-9, maxiter=1000, M=M)
            if info != 0:
                print(f"solution via cg unsuccessful! info = {info}")
            return dx_neg
    elif solver == 'bicgstab':
        def linsolver(jac_matrix, g):
            # determine pre-conditioner M via ILU factorization
            Jac_iLU = linalg.spilu(jac_matrix)
            M = linalg.LinearOperator((n, n), Jac_iLU.solve)
            dx_neg, info = linalg.bicgstab(
                jac_matrix, g, np.zeros(n), tol=1e-9, maxiter=10, M=M)
            if info != 0:
                print('solution via bicgstab unsuccessful! info = %d' % info)
            return dx_neg
    else:
        linsolver = None
        print("No valid solver selected.")

    converged = False
    it = 0
    x = initial_guess.copy()
    while (not converged) and (it < max_iterations):
        it += 1
        g, jac_matrix = function(x, *args)
        g = g.reshape((-1, 1))
        dx_neg = linsolver(jac_matrix, g)
        defect = norm(dx_neg[:], ord=np.inf)
        x -= dx_neg.reshape(x.shape)
        converged = (defect < tolerance)
        if callback:
            callback(x, g)

    if not converged:
        message = f"Newton stopped after {it} iterations with max. norm {defect}."
    else:
        message = 'The solution converged'

    result = OptimizeResult({
        'x': x,
        'success': converged,
        'message': message,
        'fun': g.reshape(initial_guess.shape),
        'jac': jac_matrix,
        'nit': it
    })
    return result

def clip_approach(values, function, lower_bounds=0, upper_bounds=None, factor=0):
    """
    Filter values with lower and upper bounds using an approach factor.

    Args:
        values (ndarray): The array of values to be filtered.
        function (callable): The function to apply.
        lower_bounds (float or ndarray, optional): The lower bounds. Default is 0.
        upper_bounds (float or ndarray, optional): The upper bounds. Default is None.
        factor (float, optional): The approach factor. Default is 0.
    """
    # filter values with lower and upper bounds using an approach factor
    if factor == 0:
        np.clip(values, lower_bounds, upper_bounds, out=values)
    else:
        if lower_bounds is not None:
            below_lower = (values < lower_bounds)
            if np.any(below_lower):
                broadcasted_lower_bounds = np.broadcast_to(
                    lower_bounds, values.shape)
                values[below_lower] = (
                    1.0 + factor)*broadcasted_lower_bounds[below_lower] - factor*values[below_lower]
        if upper_bounds is not None:
            above_upper = (values > upper_bounds)
            if np.any(above_upper):
                broadcasted_upper_bounds = np.broadcast_to(
                    upper_bounds, values.shape)
                values[above_upper] = (
                    1.0 + factor)*broadcasted_upper_bounds[above_upper] - factor*values[above_upper]


def interp_stagg_to_cntr(staggered_values, x_f, x_c=None, axis=0):
    """
    Interpolate values at staggered positions to cell-centers using linear interpolation.

    Args:
        staggered_values (ndarray): Quantities at staggered positions.
        x_f (ndarray): Positions of cell-faces.
        x_c (ndarray, optional): Cell-centered positions. Default is None.
        axis (int, optional): Dimension that is interpolated. Default is 0.

    Returns:
        ndarray: Interpolated concentrations at the cell-centered positions.
    """
    shape_f = list(staggered_values.shape)
    if axis < 0:
        axis += len(shape_f)
    shape_f_t = [math.prod(shape_f[:axis]), shape_f[axis], math.prod(
        shape_f[axis + 1:])]  # reshape as a triplet
    shape = shape_f.copy()
    shape[axis] = shape[axis] - 1
    staggered_values = np.reshape(staggered_values, shape_f_t)
    if x_c is None:
        cell_centered_values = 0.5 * \
            (staggered_values[:, 1:, :] + staggered_values[:, :-1, :])
    else:
        wght = (x_c - x_f[:-1]) / \
            (x_f[1:] - x_f[:-1])
        cell_centered_values = staggered_values[:, :-1, :] + wght.reshape(
            (1, -1, 1)) * (staggered_values[:, 1:, :] - staggered_values[:, :-1, :])
    cell_centered_values = cell_centered_values.reshape(shape)
    return cell_centered_values


def interp_cntr_to_stagg(cell_centered_values, x_f, x_c=None, axis=0):
    """
    Interpolate values at cell-centers to staggered positions using linear interpolation.

    Args:
        cell_centered_values (ndarray): Quantities at cell-centered positions.
        x_f (ndarray): Positions of cell-faces.
        x_c (ndarray, optional): Cell-centered positions. Default is None.
        axis (int, optional): Dimension along which interpolation is performed. Default is 0.

    Returns:
        ndarray: Interpolated concentrations at staggered positions.
    """
    shape = list(cell_centered_values.shape)
    if axis < 0:
        axis += len(shape)
    shape_t = [math.prod(shape[:axis]), shape[axis], math.prod(
        shape[axis + 1:])]  # reshape as a triplet
    shape_f = shape.copy()
    shape_f[axis] = shape[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_f[axis]
    if x_c is None:
        x_c = 0.5*(x_f[:-1]+x_f[1:])
    wght = (x_f[1:-1] - x_c[:-1]) / \
        (x_c[1:] - x_c[:-1])
    cell_centered_values = cell_centered_values.reshape(shape_t)
    if shape_t[1] == 1:
        staggered_values = np.tile(cell_centered_values, (1, 2, 1))
    else:
        staggered_values = np.empty(shape_f_t)
        staggered_values[:, 1:-1, :] = cell_centered_values[:, :-1, :] + wght.reshape(
            (1, -1, 1)) * (cell_centered_values[:, 1:, :] - cell_centered_values[:, :-1, :])
        staggered_values[:, 0, :] = (cell_centered_values[:, 0, :]*(x_c[1]-x_f[0]) -
                                     cell_centered_values[:, 1, :]*(x_c[0]-x_f[0]))/(x_c[1]-x_c[0])
        staggered_values[:, -1, :] = (cell_centered_values[:, -1, :]*(x_f[-1]-x_c[-2]) -
                                      cell_centered_values[:, -2, :]*(x_f[-1]-x_c[-1]))/(x_c[-1]-x_c[-2])
        staggered_values = staggered_values.reshape(shape_f)
    return staggered_values


def interp_cntr_to_stagg_tvd(cell_centered_values, x_f, x_c=None, bc=None, v=0, tvd_limiter=None, axis=0):
    """
    Interpolate values at staggered positions to cell-centers using TVD interpolation.

    Args:
        cell_centered_values (ndarray): Quantities at cell-centered positions.
        x_f (ndarray): Positions of cell-faces.
        x_c (ndarray, optional): Cell-centered positions. Default is None.
        bc (list, optional): The boundary conditions used to extrapolate to the boundary faces. Default is None.
        v (ndarray, optional): Velocities on face positions. Default is 0.
        tvd_limiter (function, optional): The TVD limiter. Default is None.
        axis (int, optional): Dimension along which interpolation is performed. Default is 0.

    Returns:
        ndarray: Interpolated concentrations at staggered positions.
        ndarray: Delta staggered values.
    """
    shape = list(cell_centered_values.shape)
    if axis < 0:
        axis += len(shape)
    shape_t = [math.prod(shape[:axis]), shape[axis], math.prod(
        shape[axis + 1:])]  # reshape as a triplet
    shape_f = shape.copy()
    shape_f[axis] = shape[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_f[axis]
    shape_bc = shape_f.copy()
    shape_bc[axis] = 1
    shape_bc_d = [shape_t[0], shape_t[2]]

    if x_c is None:
        x_c = 0.5*(x_f[:-1]+x_f[1:])
    cell_centered_values = cell_centered_values.reshape(shape_t)
    staggered_values = np.empty(shape_f_t)

    if shape_t[1] == 1:
        a, b, d = [None]*2, [None]*2, [None]*2
        a[0], b[0], d[0] = unwrap_bc(shape, bc[0])
        a[1], b[1], d[1] = unwrap_bc(shape, bc[1])
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
        fctr_m = (alpha_1 * a[0] * (a[1] * (alpha_0_right - alpha_2_left) + b[1])
                  * fctr)
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)
        staggered_values[:, 0, :] = fctr_m*cell_centered_values[:, 0, :]
        fctr_m = (alpha_1 * a[1] * (a[0] * (alpha_0_left - alpha_2_right) + b[0])
                  * fctr)
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)
        staggered_values[:, 1, :] = fctr_m*cell_centered_values[:, 0, :]
        fctr_m = ((a[1] * alpha_0_right + b[1]) * d[0] -
                  alpha_2_left * a[0] * d[1]) * fctr
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)
        staggered_values[:, 0, :] += fctr_m
        fctr_m = ((a[0] * alpha_0_left + b[0]) * d[1] -
                  alpha_2_right * a[1] * d[0]) * fctr
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)
        staggered_values[:, 1, :] += fctr_m
        staggered_values.reshape(shape_f)
        delta_staggered_values = np.zeros(shape_f)
    else:
        # bc 0
        a, b, d = unwrap_bc(shape, bc[0])
        alpha_1 = (x_c[1] - x_f[0]) / (
            (x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_2 = (x_c[0] - x_f[0]) / (
            (x_c[1] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a + b)
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a * fctr
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        staggered_values[:, 0, :] = (
            d_fctr + a_fctr*(alpha_1*cell_centered_values[:, 0, :] - alpha_2*cell_centered_values[:, 1, :]))
        # bc 1
        a, b, d = unwrap_bc(shape, bc[1])
        alpha_1 = -(x_c[-2] - x_f[-1]) / (
            (x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_2 = -(x_c[-1] - x_f[-1]) / (
            (x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a + b)
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a * fctr
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        staggered_values[:, -1, :] = (d_fctr + a_fctr*(
            alpha_1*cell_centered_values[:, -1, :] - alpha_2*cell_centered_values[:, -2, :]))

        v = np.broadcast_to(np.asarray(v),shape_f)
        v_t = v.reshape(shape_f_t)
        fltr_v_pos = (v_t > 0)

        x_f = x_f.reshape((1, -1, 1))
        x_c = x_c.reshape((1, -1, 1))
        x_d = x_f[:, 1:-1, :]
        x_C = fltr_v_pos[:, 1:-1, :]*x_c[:, :-1, :] + \
            ~fltr_v_pos[:, 1:-1, :]*x_c[:, 1:, :]
        x_U = fltr_v_pos[:, 1:-1, :]*np.concatenate((x_f[:, 0:1, :], x_c[:, 0:-2, :]), axis=1) + \
            ~fltr_v_pos[:, 1:-1, :]*np.concatenate(
                (x_c[:, 2:, :], x_f[:, -1:, :]), axis=1)
        x_D = fltr_v_pos[:, 1:-1, :]*x_c[:, 1:, :] + \
            ~fltr_v_pos[:, 1:-1, :]*x_c[:, :-1, :]
        x_norm_C = (x_C-x_U)/(x_D-x_U)
        x_norm_d = (x_d-x_U)/(x_D-x_U)
        c_C = fltr_v_pos[:, 1:-1, :]*cell_centered_values[:, :-1,
                                                          :] + ~fltr_v_pos[:, 1:-1, :]*cell_centered_values[:, 1:, :]
        c_U = fltr_v_pos[:, 1:-1, :]*np.concatenate((staggered_values[:, 0:1, :], cell_centered_values[:, 0:-2, :]), axis=1) + \
            ~fltr_v_pos[:, 1:-1, :]*np.concatenate(
                (cell_centered_values[:, 2:, :], staggered_values[:, -1:, :]), axis=1)
        c_D = fltr_v_pos[:, 1:-1, :]*cell_centered_values[:, 1:, :] + \
            ~fltr_v_pos[:, 1:-1, :]*cell_centered_values[:, :-1, :]
        c_norm_C = np.zeros_like(c_C)
        dc_DU = (c_D-c_U)
        np.divide((c_C-c_U), dc_DU, out=c_norm_C, where=(dc_DU != 0))
        staggered_values = np.concatenate(
            (staggered_values[:, 0:1, :], c_C, staggered_values[:, -1:, :]), axis=1)
        if tvd_limiter is None:
            delta_staggered_values = np.zeros(shape_f)
            staggered_values = staggered_values.reshape(shape_f)
        else:
            delta_staggered_values = np.zeros(shape_f_t)
            delta_staggered_values[:, 1:-1,
                                   :] = tvd_limiter(c_norm_C, x_norm_C, x_norm_d) * dc_DU
            staggered_values += delta_staggered_values
            delta_staggered_values = delta_staggered_values.reshape(shape_f)
            staggered_values = staggered_values.reshape(shape_f)
    return staggered_values, delta_staggered_values


def upwind(normalized_c_c, normalized_x_c, normalized_x_d):
    """
    Apply the upwind TVD limiter to reduce oscillations in numerical schemes.

    Args:
        normalized_c_c (ndarray): Normalized concentration at cell centers.
        normalized_x_c (ndarray): Normalized position of cell centers.
        normalized_x_d (ndarray): Normalized position of down-wind face.

    Returns:
        ndarray: Normalized concentration difference c_norm_d - c_norm_C.
    """
    normalized_concentration_diff = np.zeros_like(
        normalized_c_c)
    return normalized_concentration_diff


def minmod(normalized_c_c, normalized_x_c, normalized_x_d):
    """
    Apply the Minmod TVD limiter to reduce oscillations in numerical schemes.

    Args:
        normalized_c_c (ndarray): Normalized concentration at cell centers.
        normalized_x_c (ndarray): Normalized position of cell centers.
        normalized_x_d (ndarray): Normalized position of down-wind face.

    Returns:
        ndarray: Normalized concentration difference c_norm_d - c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0, (normalized_x_d-normalized_x_c)*np.minimum(
        normalized_c_c/normalized_x_c, (1-normalized_c_c)/(1-normalized_x_c)))
    return normalized_concentration_diff


def osher(normalized_c_c, normalized_x_c, normalized_x_d):
    """
    Apply the Osher TVD limiter to reduce oscillations in numerical schemes.

    Args:
        normalized_c_c (ndarray): Normalized concentration at cell centers.
        normalized_x_c (ndarray): Normalized position of cell centers.
        normalized_x_d (ndarray): Normalized position of down-wind face.

    Returns:
        ndarray: Normalized concentration difference c_norm_d - c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0, np.where(normalized_c_c < normalized_x_c/normalized_x_d,
                                               (normalized_x_d/normalized_x_c - 1)*normalized_c_c, 1 - normalized_c_c))
    return normalized_concentration_diff


def clam(normalized_c_c, normalized_x_c, normalized_x_d):
    """
    Apply the CLAM TVD limiter to reduce oscillations in numerical schemes.

    Args:
        normalized_c_c (ndarray): Normalized concentration at cell centers.
        normalized_x_c (ndarray): Normalized position of cell centers.
        normalized_x_d (ndarray): Normalized position of down-wind face.

    Returns:
        ndarray: Normalized concentration difference c_norm_d - c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0, np.where(normalized_c_c < normalized_x_c/normalized_x_d,
                                               (normalized_x_d/normalized_x_c - 1)*normalized_c_c, 1 - normalized_c_c))
    return normalized_concentration_diff


def muscl(normalized_c_c, normalized_x_c, normalized_x_d):
    """
    Apply the MUSCL TVD limiter to reduce oscillations in numerical schemes.

    Args:
        normalized_c_c (ndarray): Normalized concentration at cell centers.
        normalized_x_c (ndarray): Normalized position of cell centers.
        normalized_x_d (ndarray): Normalized position of down-wind face.

    Returns:
        ndarray: Normalized concentration difference c_norm_d - c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0, np.where(normalized_c_c < normalized_x_c/(2*normalized_x_d), ((2*normalized_x_d - normalized_x_c)/normalized_x_c - 1)*normalized_c_c,
                                                           np.where(normalized_c_c < 1 + normalized_x_c - normalized_x_d, normalized_x_d - normalized_x_c, 1 - normalized_c_c)))
    return normalized_concentration_diff


def smart(normalized_c_c, normalized_x_c, normalized_x_d):
    """
    Apply the SMART TVD limiter to reduce oscillations in numerical schemes.

    Args:
        normalized_c_c (ndarray): Normalized concentration at cell centers.
        normalized_x_c (ndarray): Normalized position of cell centers.
        normalized_x_d (ndarray): Normalized position of down-wind face.

    Returns:
        ndarray: Normalized concentration difference c_norm_d - c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0, np.where(normalized_c_c < normalized_x_c/3, (normalized_x_d*(1 - 3*normalized_x_c + 2*normalized_x_d)/(normalized_x_c*(1 - normalized_x_c)) - 1)*normalized_c_c,
                                                           np.where(normalized_c_c < normalized_x_c/normalized_x_d*(1 + normalized_x_d - normalized_x_c), (normalized_x_d*(normalized_x_d - normalized_x_c) + normalized_x_d*(1 - normalized_x_d)/normalized_x_c*normalized_c_c)/(1 - normalized_x_c) - normalized_c_c, 1 - normalized_c_c)))
    return normalized_concentration_diff


def stoic(normalized_c_c, normalized_x_c, normalized_x_d):
    """
    Apply the STOIC TVD limiter to reduce oscillations in numerical schemes.

    Args:
        normalized_c_c (ndarray): Normalized concentration at cell centers.
        normalized_x_c (ndarray): Normalized position of cell centers.
        normalized_x_d (ndarray): Normalized position of down-wind face.

    Returns:
        ndarray: Normalized concentration difference c_norm_d - c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0, np.where(normalized_c_c < normalized_x_c*(normalized_x_d - normalized_x_c)/(normalized_x_c + normalized_x_d + 2*normalized_x_d*normalized_x_d - 4*normalized_x_d*normalized_x_c), normalized_x_d*(1 - 3*normalized_x_c + 2*normalized_x_d)/(normalized_x_c*(1 - normalized_x_c)) - normalized_c_c,
                                                           np.where(normalized_c_c < normalized_x_c, (normalized_x_d - normalized_x_c + (1 - normalized_x_d)*normalized_c_c)/(1 - normalized_x_c) - normalized_c_c,
                                                                    np.where(normalized_c_c < normalized_x_c/normalized_x_d*(1 + normalized_x_d - normalized_x_c), (normalized_x_d*(normalized_x_d - normalized_x_c) + normalized_x_d*(1 - normalized_x_d)/normalized_x_c*normalized_c_c)/(1 - normalized_x_c) - normalized_c_c, 1 - normalized_c_c))))
    return normalized_concentration_diff


def vanleer(normalized_c_c, normalized_x_c, normalized_x_d):
    """
    Apply the van Leer TVD limiter to reduce oscillations in numerical schemes.

    Args:
        normalized_c_c (ndarray): Normalized concentration at cell centers.
        normalized_x_c (ndarray): Normalized position of cell centers.
        normalized_x_d (ndarray): Normalized position of down-wind face.

    Returns:
        ndarray: Normalized concentration difference c_norm_d - c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0, normalized_c_c*(1-normalized_c_c)*(
        normalized_x_d-normalized_x_c)/(normalized_x_c*(1-normalized_x_c)))
    return normalized_concentration_diff


def non_uniform_grid(left_bound, right_bound, num_points, dx_inf, factor):
    """
    Generate a non-uniform grid of points in the interval [left_bound, right_bound].

    Args:
        left_bound (float): Start point of the interval.
        right_bound (float): End point of the interval.
        num_points (int): Total number of face positions (including left_bound and right_bound).
        dx_inf (float): Limiting upper-bound grid spacing.
        factor (float): Factor used to increase grid spacing.

    Returns:
        ndarray: Array containing the non-uniform grid points.
    """
    a = np.log(factor)
    unif = np.arange(num_points)
    b = np.exp(-a * unif)
    L = right_bound - left_bound
    c = (np.exp(a * (L / dx_inf - num_points + 1.0)) - b[-1]) / (1 - b[-1])
    x_f = left_bound + unif * dx_inf + \
        np.log((1 - c) * b + c) * (dx_inf / a)
    return x_f

def generate_grid(size, x_f, generate_x_c=False, x_c=None):
    """
    Generate a grid of face and optionally cell-centered positions.

    Args:
        size (int): Number of cells.
        x_f (ndarray): Face positions.
        generate_x_c (bool, optional): Whether to generate cell-centered positions. Default is False.
        x_c (ndarray, optional): Cell-centered positions. Default is None.

    Returns:
        tuple: Face positions (x_f) and optionally cell-centered positions (x_c).
    """
    if (x_f is None or len(x_f) == 0):
        x_f = np.linspace(0.0, 1.0, size+1)
    elif size+1 == len(x_f):
        x_f = np.asarray(x_f)
    elif len(x_f) == 2:
        x_f = np.linspace(x_f[0], x_f[1], size+1)
    else:
        raise ValueError("Grid cannot be generated")
    if generate_x_c:
        if x_c is None:
            x_c = 0.5*(x_f[1:] + x_f[:-1])
        elif len(x_c) == size:
            x_c = np.asarray(x_c)
        else:
            raise ValueError("Cell-centered grid not properly defined")
        return x_f, x_c
    return x_f


def unwrap_bc(shape, bc):
    """
    Unwrap the boundary conditions for a given shape.

    Args:
        shape (tuple): Shape of the domain.
        bc (dict): Boundary conditions.

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
        a = a[(..., *([np.newaxis]*(lgth_shape-a.ndim)))]
        b = np.array(bc['b'])
        b = b[(..., *([np.newaxis]*(lgth_shape-b.ndim)))]
        d = np.array(bc['d'])
        d = d[(..., *([np.newaxis]*(lgth_shape-d.ndim)))]
    return a, b, d
