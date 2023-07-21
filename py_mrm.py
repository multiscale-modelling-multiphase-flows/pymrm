"""
Module Name: py_mrm
Author: E.A.J.F. Peters
License: MIT License
Version: 1.0.0

This module provides functions for multiphase reactor modeling

Usage:
- Function1: Perform operation X on input data.
- Function2: Perform operation Y on input data.

"""


import numpy as np
from scipy.sparse import csc_array
import math


def unwrap_bc(sz, bc):
    """
    Unwrap the boundary conditions for a given size.

    Args:
        sz (tuple): Size of the domain.
        bc (dict): Boundary conditions.

    Returns:
        tuple: Unwrapped boundary conditions (a, b, d).
    """
    a = [None, None]
    a[0] = np.array(bc['a'][0])
    a[0] = a[0][(..., *([np.newaxis]*(len(sz)-a[0].ndim)))]
    a[1] = np.array(bc['a'][1])
    a[1] = a[1][(..., *([np.newaxis]*(len(sz)-a[1].ndim)))]
    b = [None, None]
    b[0] = np.array(bc['b'][0])
    b[0] = b[0][(..., *([np.newaxis]*(len(sz)-b[0].ndim)))]
    b[1] = np.array(bc['b'][1])
    b[1] = b[1][(..., *([np.newaxis]*(len(sz)-b[1].ndim)))]
    d = [None, None]
    d[0] = np.array(bc['d'][0])
    d[0] = d[0][(..., *([np.newaxis]*(len(sz)-d[0].ndim)))]
    d[1] = np.array(bc['d'][1])
    d[1] = d[1][(..., *([np.newaxis]*(len(sz)-d[1].ndim)))]
    return a, b, d

def construct_div(axis, sz, x_f, nu):
    """
    Construct the Div matrix based on the given parameters.

    Args:
        axis (int): The axis along which the numerical differentiation is performed
        sz (tuple): The size of the matrix.
        x_f (ndarray): The face array.
        nu (callable or int): The function or integer representing nu.

    Returns:
        csc_array: The Div matrix.

    """
    # Trick: Reshape sizes to triplet sz_t
    sz_t = [math.prod(sz[0:axis]), math.prod(
        sz[axis:axis + 1]), math.prod(sz[axis + 1:])]

    # Create face arrays
    sz_f = list(sz)
    sz_f[axis] += 1
    sz_f_t = sz_t.copy()
    sz_f_t[1] += 1

    i_f = (
        sz_f_t[1] * sz_f_t[2] *
        np.arange(sz_t[0]).reshape((-1, 1, 1, 1))
        + sz_f_t[2] * np.arange(sz_t[1]).reshape((1, -1, 1, 1))
        + np.arange(sz_t[2]).reshape((1, 1, -1, 1))
        + np.array([0, sz_t[2]]).reshape((1, 1, 1, -1))
    )

    if callable(nu):
        area = nu(x_f).flatten()
        inv_sqrt3 = 1 / np.sqrt(3)
        x_f_r = x_f.flatten()
        dx_f = x_f_r[1:] - x_f_r[:-1]
        dvol_inv = 1 / (
            (nu(x_f_r[:-1] + (0.5 - 0.5 * inv_sqrt3) * dx_f)
             + nu(x_f_r[:-1] + (0.5 + 0.5 * inv_sqrt3) * dx_f))
            * 0.5 * dx_f
        )
    elif nu == 0:
        area = np.ones(sz_f_t[1])
        dvol_inv = 1 / (x_f[1:] - x_f[:-1])
    else:
        area = np.power(x_f.flatten(), nu)
        vol = area * x_f.flatten() / (nu + 1)
        dvol_inv = 1 / (vol[1:] - vol[:-1])

    values = np.empty((sz_t[1],2))
    values[:, 0] = -area[:-1] * dvol_inv
    values[:, 1] =  area[1:] * dvol_inv
    values = np.tile(values.reshape((1,-1,1,2)),(sz_t[0],1,sz_t[2]))

    num_cells = np.prod(sz_t, dtype=int);
    Div = csc_array(
        (values.flatten(),(np.repeat(np.arange(num_cells),2) , 
                           i_f.flatten())),
        shape=(num_cells, np.prod(sz_f_t, dtype=int))
    )
    Div.sort_indices()
    return Div


def construct_convflux_upwind(axis, sz, x_c, x_f, bc, v):
    """
    Construct the Conv and conv_bc matrices based on the given parameters.

    Args:
        axis (int): The axis along which the numerical differentiation is performed.
        sz (tuple): The size of the matrices.
        x_c (ndarray): The cell array.
        x_f (ndarray): The face array.
        bc (list): The boundary conditions.
        v (ndarray): The velocity array.

    Returns:
        csc_array: The Conv matrix.
        csc_array: The conv_bc matrix.

    """
    sz_t = [math.prod(sz[0:axis]), math.prod(
        sz[axis:axis+1]), math.prod(sz[axis+1:])]

    sz_f = list(sz)
    sz_f[axis] = sz_f[axis] + 1
    sz_f_t = sz_t.copy()
    sz_f_t[1] = sz_t[1] + 1

    sz_bc = list(sz)
    sz_bc[axis] = 1
    sz_bc_d = [sz_t[0], sz_t[2]]

    a, b, d = unwrap_bc(sz, bc)

    v = np.array(v) + np.zeros(sz_f)
    v = v.reshape(sz_f_t)
    fltr_v_pos = (v > 0)

    if sz_t[1] == 1:
        i_c = sz_t[1] * sz_t[2] * np.arange(sz_t[0]).reshape((-1, 1, 1)) + np.array(
            (0, 0)).reshape((1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
        values = np.zeros(sz_f_t)
        alpha_1 = (x_f[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2L = (x_c[0] - x_f[0]) / ((x_f[1] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_0L = alpha_1 - alpha_2L
        alpha_2R = -(x_c[0] - x_f[1]) / ((x_f[0] - x_f[1]) * (x_f[0] - x_c[0]))
        alpha_0R = alpha_1 - alpha_2R
        fctr = 1. / ((b[0] + alpha_0L * a[0]) * (b[1] +
                     alpha_0R * a[1]) - alpha_2L * alpha_2R * a[0] * a[1])
        fctr[~np.isfinite(fctr)] = 0
        values[:, 0, :] = ((alpha_1 * a[0] * (a[1] * (alpha_0R - alpha_2L) + b[1])
                           * fctr + np.zeros(sz)).reshape(sz_bc_d) - 1) * fltr_v_pos[:, 0, :] + 1
        values[:, 1, :] = ((alpha_1 * a[1] * (a[0] * (alpha_0L - alpha_2R) + b[0])
                           * fctr + np.zeros(sz)).reshape(sz_bc_d) - 1) * ~fltr_v_pos[:, -1, :] + 1
        values = values * v
        Conv = csr_array((values.flatten(), i_c.flatten(), np.arange(
            0, i_c.size + 1)), shape=(math.prod(sz_f_t), math.prod(sz_t)))

        i_f_bc = sz_f_t[1] * sz_f_t[2] * np.arange(sz_f_t[0]).reshape((-1, 1, 1)) + sz_f_t[2] * np.array(
            [0, sz_f_t[1] - 1]).reshape((1, -1, 1)) + np.arange(sz_f_t[2]).reshape((1, 1, -1))
        values_bc = np.zeros((sz_t[0], 2, sz_t[2]))
        values_bc[:, 0, :] = ((((a[1] * alpha_0R + b[1]) * d[0] - alpha_2L * a[0] * d[1])
                              * fctr + np.zeros(sz_bc)).reshape(sz_bc_d)) * fltr_v_pos[:, 0, :]
        values_bc[:, 1, :] = ((((a[0] * alpha_0L + b[0]) * d[1] - alpha_2R * a[1] * d[0])
                              * fctr + np.zeros(sz_bc)).reshape(sz_bc_d)) * ~fltr_v_pos[:, -1, :]
        values_bc = values_bc * v
    else:
        i_f = np.zeros((sz_f_t[0], sz_f_t[1] + 2, sz_f_t[2]), dtype=int)
        i_f[:, 1:-1, :] = np.arange(math.prod(sz_f_t)).reshape(sz_f_t)
        i_f[:, 0, :] = sz_f_t[1] * sz_f_t[2] * \
            np.arange(sz_f_t[0]).reshape(-1, 1) + \
            np.arange(sz_f_t[2]).reshape(1, -1)
        i_f[:, -1, :] = sz_f_t[1] * sz_f_t[2] * np.arange(sz_f_t[0]).reshape(-1, 1) + sz_f_t[2] * (
            sz_f_t[1] - 1) + np.arange(sz_f_t[2]).reshape(1, -1)
        i_c = np.zeros((sz_f_t[0], sz_f_t[1] + 2, sz_f_t[2]), dtype=int)
        i_c[:, 1:-2, :] = np.arange(math.prod(sz_t)).reshape(sz_t)
        i_c[:, :2, :] = sz_t[1] * sz_t[2] * np.arange(sz_t[0]).reshape((-1, 1, 1)) + np.array(
            (0, sz_t[2])).reshape((1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
        i_c[:, -2:, :] = sz_t[1] * sz_t[2] * np.arange(sz_t[0]).reshape((-1, 1, 1)) + sz_t[2] * np.array(
            (sz_t[1] - 2, sz_t[1] - 1)).reshape((1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
        i_c[:, 2:-2, :] = i_c[:, 2:-2, :] - sz_t[2] * fltr_v_pos[:, 1:-1, :]
        i_f_bc = sz_f_t[1] * sz_f_t[2] * np.arange(sz_f_t[0]).reshape((-1, 1, 1)) + sz_f_t[2] * np.array(
            [0, sz_f_t[1] - 1]).reshape((1, -1, 1)) + np.arange(sz_f_t[2]).reshape((1, 1, -1))
        values = np.zeros((sz_f_t[0], sz_f_t[1] + 2, sz_f_t[2]))
        values[:, 2:-2, :] = v[:, 1:-1, :]
        values_bc = np.zeros((sz_f_t[0], 2, sz_f_t[2]))

        alpha_1 = (x_c[0] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_2 = (x_c[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_0 = alpha_1 - alpha_2
        fctr = 1 / (alpha_0 * a[0] + b[0])
        fctr[~np.isfinite(fctr)] = 0
        a_fctr = a[0] * fctr
        a_fctr = a_fctr + np.zeros(sz_bc)
        a_fctr = np.reshape(a_fctr, sz_bc_d)
        d_fctr = d[0] * fctr
        d_fctr = d_fctr + np.zeros(sz_bc)
        d_fctr = np.reshape(d_fctr, sz_bc_d)
        values[:, 0, :] = (1 + (a_fctr * alpha_1 - 1) *
                           fltr_v_pos[:, 0, :]) * v[:, 0, :]
        values[:, 1, :] = -a_fctr * alpha_2 * fltr_v_pos[:, 0, :] * v[:, 0, :]
        values_bc[:, 0, :] = d_fctr * v[:, 0, :] * fltr_v_pos[:, 0, :]

        alpha_1 = -(x_c[-2] - x_f[-1]) / \
            ((x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_2 = -(x_c[-1] - x_f[-1]) / \
            ((x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_0 = alpha_1 - alpha_2
        fctr = 1 / (alpha_0 * a[-1] + b[-1])
        fctr[~np.isfinite(fctr)] = 0
        a_fctr = a[-1] * fctr
        a_fctr = a_fctr + np.zeros(sz_bc)
        a_fctr = np.reshape(a_fctr, sz_bc_d)
        d_fctr = d[-1] * fctr
        d_fctr = d_fctr + np.zeros(sz_bc)
        d_fctr = np.reshape(d_fctr, sz_bc_d)
        values[:, -1, :] = (1 + (a_fctr * alpha_1 - 1) * ~
                            fltr_v_pos[:, -1, :]) * v[:, -1, :]
        values[:, -2, :] = a_fctr * alpha_2 * \
            ~fltr_v_pos[:, -1, :] * v[:, -1, :]
        values_bc[:, -1, :] = d_fctr * v[:, -1, :] * ~fltr_v_pos[:, -1, :]
        Conv = csc_array((values.flatten(), (i_f.flatten(), i_c.flatten())), shape=(
            math.prod(sz_f_t), math.prod(sz_t)))
        Conv.sort_indices()

    conv_bc = csc_array((values_bc.flatten(), i_f_bc.flatten(), [
                         0, i_f_bc.size]), shape=(math.prod(sz_f_t), 1))
    return Conv, conv_bc

def numjac_local(axis, f, c, eps_jac=1e-6):
    """
    Compute the local numerical Jacobian matrix and function values for the given function and initial values.
    The function f is assumed local, f can be dependent on other components in the array dimension 'axis'.
    numjac_local can be used to compute Jacobians of e.g. reaction, accumulation or mass transfer terms, where there 
    is a only dependence of local components in a spatial cell.
    The best choice is to set up the problem such that axis is the last dimension of the multidimensional array, 
    because this will give to a nicely block structured Jacoian matrix.  

    Args:
        axis (int): The axis along which components are coupled.
        f (callable): The function for which to compute the Jacobian.
        c (ndarray): The value at which the Jacobian should be evaluated.
        eps_jac (float, optional): The perturbation value for computing the Jacobian. Defaults to 1e-6.

    Returns:
        csc_array: The Jacobian matrix.
        ndarray: The function values.

    """
    sz = c.shape
    sz_t = [math.prod(sz[0:axis]), math.prod(
        sz[axis:axis+1]), math.prod(sz[axis+1:])]
    values = np.zeros((*sz_t, sz[axis]))
    i = sz_t[1] * sz_t[2] * np.arange(sz_t[0]).reshape((-1, 1, 1, 1)) + np.zeros((1, sz_t[1], 1, 1)) + np.arange(
    sz_t[2]).reshape((1, 1, -1, 1)) + sz_t[2] * np.arange(sz_t[1]).reshape((1, 1, 1, -1))
    f_value = f(c).reshape(sz_t)
    c = c.reshape(sz_t)
    dc = -eps_jac * np.abs(c)  # relative deviation
    dc[dc > (-eps_jac)] = eps_jac  # If dc is small use absolute deviation
    dc = (c + dc) - c
    for k in range(sz_t[1]):
        c_perturb = np.copy(c)
        c_perturb[:, k, :] = c_perturb[:, k, :] + dc[:, k, :]
        f_perturb = f(c_perturb.reshape(sz)).reshape(sz_t)
        values[:, k, :, :] = np.transpose((f_perturb - f_value) / dc[:, [k], :],(0,2,1))
    Jac = csc_array((values.flatten(), i.flatten(), np.arange(
        0, i.size + sz_t[1], sz_t[1])), shape=(np.prod(sz_t), np.prod(sz_t)))
    return Jac, f_value.reshape(sz)

def construct_grad(axis, sz, x_c, x_f, bc):
    """
    Construct the gradient matrix.

    Args:
        axis (int): Dimension to construct the gradient matrix for.
        sz (tuple): Size of the domain.
        x_c (ndarray): Cell center coordinates.
        x_f (ndarray): Face coordinates.
        bc (dict): Boundary conditions.

    Returns:
        csr_array: Gradient matrix (Grad).
        csc_array: Contribtion of the inhomogeneous BC to the gradient (grad_bc).
    """
    # Trick: Reshape sizes to triplet sz_t
    sz_t = [math.prod(sz[0:axis]), math.prod(
        sz[axis:axis+1]), math.prod(sz[axis+1:])]

    # Create face arrays
    sz_f = list(sz)
    sz_f[axis] = sz_f[axis] + 1
    sz_f_t = list(sz_t.copy())
    sz_f_t[1] = sz_t[1] + 1

    # Create boundary quantity sizes
    sz_bc = list(sz)
    sz_bc[axis] = 1
    sz_bc_d = [sz_t[0], sz_t[2]]

    a, b, d = unwrap_bc(sz, bc)

    # Handle special case with one cell in the dimension axis
    if sz_t[1] == 1:
        i_c = sz_t[2] * np.arange(sz_t[0]).reshape((-1, 1, 1)) + np.array(
            (0, 0)).reshape((1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
        values = np.zeros(sz_f_t)
        alpha_1 = (x_f[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2L = (x_c[0] - x_f[0]) / ((x_f[1] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_0L = alpha_1 - alpha_2L
        alpha_2R = -(x_c[0] - x_f[1]) / ((x_f[0] - x_f[1]) * (x_f[0] - x_c[0]))
        alpha_0R = alpha_1 - alpha_2R
        fctr = 1 / ((b[0] + alpha_0L * a[0]) * (b[1] +
                    alpha_0R * a[1]) - alpha_2L * alpha_2R * a[0] * a[1])
        fctr[~np.isfinite(fctr)] = 0
        value = alpha_1 * \
            b[0] * (a[1] * (alpha_0R - alpha_2L) + b[1]) * fctr + np.zeros(sz)
        values[:, 0, :] = np.reshape(value, sz_bc_d)
        value = alpha_1 * \
            b[1] * (a[0] * (-alpha_0L + alpha_2R) - b[0]) * fctr + np.zeros(sz)
        values[:, 1, :] = np.reshape(value, sz_bc_d)

        i_f_bc = sz_f_t[1] * sz_f_t[2] * np.arange(sz_f_t[0]).reshape((-1, 1, 1)) + sz_f_t[2] * np.array(
            [0, sz_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(sz_f_t[2]).reshape((1, 1, -1))
        values_bc = np.zeros((sz_t[0], 2, sz_t[2]))
        value = ((a[1] * (-alpha_0L * alpha_0R + alpha_2L * alpha_2R) - alpha_0L *
                 b[1]) * d[0] - alpha_2L * b[0] * d[1]) * fctr + np.zeros(sz_bc)
        values_bc[:, 0, :] = np.reshape(value, sz_bc_d)
        value = ((a[0] * (+alpha_0L * alpha_0R - alpha_2L * alpha_2R) + alpha_0R *
                 b[0]) * d[1] + alpha_2R * b[1] * d[0]) * fctr + np.zeros(sz_bc)
        values_bc[:, 1, :] = np.reshape(value, sz_bc_d)
    else:
        i_c = sz_t[1] * sz_t[2] * np.arange(sz_t[0]).reshape(-1, 1, 1, 1) + sz_t[2] * np.arange(sz_f_t[1]).reshape((
            1, -1, 1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1, 1)) + np.array([-sz_t[2], 0]).reshape((1, 1, 1, -1))
        # For the first and last (boundary) faces, the relevant cells are the two neighboring the boundary faces
        i_c[:, 0, :, :] = i_c[:, 0, :, :] + sz_t[2]
        i_c[:, -1, :, :] = i_c[:, -1, :, :] - sz_t[2]

        i_f_bc = sz_f_t[1] * sz_f_t[2] * np.arange(sz_f_t[0]).reshape((-1, 1, 1)) + sz_f_t[2] * np.array(
            [0, sz_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(sz_f_t[2]).reshape((1, 1, -1))
        values_bc = np.zeros((sz_t[0], 2, sz_t[2]))

        values = np.zeros((sz_f_t[0], sz_f_t[1], sz_f_t[2], 2))
        dx_inv = np.tile(
            1 / (x_c[1:] - x_c[:-1]).reshape((1, -1, 1)), (sz_t[0], 1, sz_t[2]))
        values[:, 1:-1, :, 0] = -dx_inv
        values[:, 1:-1, :, 1] = dx_inv

        alpha_1 = (x_c[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_2 = (x_c[0] - x_f[0]) / ((x_c[1] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_0 = alpha_1 - alpha_2
        b[0] = b[0] / alpha_0
        fctr = 1 / (a[0] + b[0])
        fctr[~np.isfinite(fctr)] = 0
        b_fctr = b[0] * fctr
        b_fctr = b_fctr + np.zeros(sz_bc)
        b_fctr = np.reshape(b_fctr, sz_bc_d)
        d_fctr = d[0] * fctr
        d_fctr = d_fctr + np.zeros(sz_bc)
        d_fctr = np.reshape(d_fctr, sz_bc_d)
        values[:, 0, :, 0] = b_fctr * alpha_1
        values[:, 0, :, 1] = -b_fctr * alpha_2
        values_bc[:, 0, :] = -d_fctr

        alpha_1 = -(x_c[-2] - x_f[-1]) / \
            ((x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_2 = -(x_c[-1] - x_f[-1]) / \
            ((x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_0 = alpha_1 - alpha_2
        b[-1] = b[-1] / alpha_0
        fctr = 1 / (a[-1] + b[-1])
        fctr[~np.isfinite(fctr)] = 0
        b_fctr = b[-1] * fctr
        b_fctr = b_fctr + np.zeros(sz_bc)
        b_fctr = np.reshape(b_fctr, sz_bc_d)
        d_fctr = d[-1] * fctr
        d_fctr = d_fctr + np.zeros(sz_bc)
        d_fctr = np.reshape(d_fctr, sz_bc_d)
        values[:, -1, :, 0] = b_fctr * alpha_2
        values[:, -1, :, 1] = -b_fctr * alpha_1
        values_bc[:, -1, :] = d_fctr

    num_f = math.prod(sz_f_t);
    i_f = np.repeat(np.arange(num_f),2)
    Grad = csc_array((values.flatten(), (i_f, i_c.flatten())), 
                      shape=(num_f, math.prod(sz_t)))
    Grad.sort_indices()
        
    grad_bc = csc_array((values_bc.flatten(), i_f_bc.flatten(), [
                         0, i_f_bc.size]), shape=(math.prod(sz_f_t), 1))

    return Grad, grad_bc


def interp_stagg_to_cntr(axis, c_f, x_c, x_f):
    """
    Interpolate values at staggered positions to cell-centers using linear interpolation.

    Args:
        axis (int): Dimension that is interpolated.
        c_f (ndarray): Quantities at cell centered positions.
        x_c (ndarray): Cell-centered positions.
        x_f (ndarray): Positions of cell-faces (numel(x_f) = numel(x_c) + 1).

    Returns:
        ndarray: Interpolated concentrations at the staggered positions.

    """
    sz_f = list(c_f.shape)
    sz_f = sz_f + [1,] * (1 + axis - len(sz_f))  # size padded with extra 1's if needed
    sz_f_t = [math.prod(sz_f[:axis]), sz_f[axis], math.prod(sz_f[axis + 1:])]  # sizes for reshape as a triplet
    sz = sz_f.copy()
    sz[axis] = sz[axis] - 1
    sz_t = sz_f_t.copy()
    sz_t[1] = sz[axis]
    wght = (x_c - x_f[:-1]) / (x_f[1:] - x_f[:-1])
    c_f = c_f.reshape(sz_f_t)
    c_c = c_f[:, :-1, :] + wght.reshape((1,-1,1)) * (c_f[:, 1:, :] - c_f[:, :-1, :])
    c_c = c_c.reshape(sz)

    return c_c
