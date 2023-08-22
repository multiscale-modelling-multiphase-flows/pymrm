"""
Module Name: py_mrm
Author: E.A.J.F. Peters
License: MIT License
Version: 1.1.0

This module provides functions for multiphase reactor modeling.

Functions:

- construct_grad(sz, x_f, x_c=None, bc=None, axis=0): Construct the gradient matrix.
- construct_grad_int(sz, x_f, x_c=None, axis=0): Construct the gradient matrix for internal faces.
- construct_grad_bc(sz, x_f, x_c=None, bc=None, axis=0): Construct the gradient matrix for boundary faces.
- construct_div(sz, x_f, nu=0, axis=0): Construct the divergence matrix based on the given parameters.
- construct_convflux_upwind(sz, x_f, x_c=None, bc=None, v=0, axis=0): Construct the convective flux matrix based on the given parameters.
- numjac_local(f, c, eps_jac=1e-6, axis=0): Compute the local numerical Jacobian matrix and function values for the given function and initial values.
- interp_stagg_to_cntr(c_f, x_f, x_c=None, axis=0): Interpolate values at staggered positions to cell-centers using linear interpolation.
- interp_cntr_to_stagg(c_c, x_f, x_c=None, axis=0): Interpolate values at cell-centered positions to staggered positions using linear interpolation and extrapolation at the wall.
- non_uniform_grid(x_L, x_R, n, dx_inf, factor): Generate a non-uniform grid of points in the interval [x_L, x_R].
- unwrap_bc(sz, bc): Unwrap the boundary conditions for a given size. Mostly used by other functions.

Note: Please refer to the function descriptions for more details on their arguments and usage.
"""


import numpy as np
from scipy.sparse import csc_array
import math

def construct_grad(sz, x_f, x_c=None, bc=None, axis=0):
    """
    Construct the gradient matrix.

    Args:
        sz (tuple): Size of the domain.
        x_f (ndarray): Face coordinates.
        x_c (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        bc (dict, optional): Boundary conditions. Default is None.
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.

    Returns:
        csr_array: Gradient matrix (Grad).
        csc_array: Contribution of the inhomogeneous BC to the gradient (grad_bc).
    """
    if (x_c is None):
        x_c = 0.5*(x_f[0:-1]+x_f[1:])
    Grad = construct_grad_int(sz, x_f, x_c, axis)
    if (bc is None):
        sz_f = sz.copy()
        sz_f[axis] +=1
        grad_bc = csc_array(shape=(math.prod(sz_f),1))
    else:
        Grad_bc, grad_bc = construct_grad_bc(sz, x_f, x_c, bc, axis)
        Grad += Grad_bc
    return Grad, grad_bc

def construct_grad_old(sz, x_f, x_c=None, bc=None, axis=0):
    """
    Construct the gradient matrix.

    Args:
        sz (tuple): Size of the domain.
        x_f (ndarray): Face coordinates.
        x_c (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        bc (dict, optional): Boundary conditions. Default is None.
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.

    Returns:
        csr_array: Gradient matrix (Grad).
        csc_array: Contribution of the inhomogeneous BC to the gradient (grad_bc).
    """
    # Trick: Reshape sizes to triplet sz_t
    if not isinstance(sz, (list, tuple)):
        sz_f = [sz]
    else:
        sz_f = list(sz)
    sz_t = [math.prod(sz_f[0:axis]), math.prod(
        sz_f[axis:axis+1]), math.prod(sz_f[axis+1:])]

    if (x_c is None):
        x_c = 0.5*(x_f[0:-1] + x_f[1:])
    
    # Create face arrays  
    sz_f[axis] = sz_f[axis] + 1
    sz_f_t = sz_t.copy()
    sz_f_t[1] = sz_t[1] + 1

    # Create boundary quantity sizes
    sz_bc = sz_f.copy()
    sz_bc[axis] = 1
    sz_bc_d = [sz_t[0], sz_t[2]]

    a, b, d = unwrap_bc(sz, bc)

    # Handle special case with one cell in the dimension axis
    if (sz_t[1] == 1):
        i_c = sz_t[2] * np.arange(sz_t[0]).reshape((-1, 1, 1)) + np.array(
            (0, 0)).reshape((1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
        values = np.zeros(sz_f_t)
        alpha_1 = (x_f[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2L = (x_c[0] - x_f[0]) / ((x_f[1] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_0L = alpha_1 - alpha_2L
        alpha_2R = -(x_c[0] - x_f[1]) / ((x_f[0] - x_f[1]) * (x_f[0] - x_c[0]))
        alpha_0R = alpha_1 - alpha_2R
        fctr = ((b[0] + alpha_0L * a[0]) * (b[1] +
                    alpha_0R * a[1]) - alpha_2L * alpha_2R * a[0] * a[1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
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
        fctr = (a[0] + b[0])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
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
        fctr = (a[-1] + b[-1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
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
                         0, i_f_bc.size]), shape=(math.prod(sz_f_t),1))

    return Grad, grad_bc

def construct_grad_int(sz, x_f,  x_c=None, axis=0):
    """
    Construct the gradient matrix for internal faces.

    Args:
        sz (tuple): Size of the domain.
        x_f (ndarray): Face coordinates.
        x_c (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.
    
    Returns:
        csr_array: Gradient matrix (Grad).
        csc_array: Contribution of the inhomogeneous BC to the gradient (grad_bc).
    """
    # Trick: Reshape sizes to triplet sz_t
    if not isinstance(sz, (list, tuple)):
        sz_t = [sz]
    else:
        sz_t = list(sz)
    sz_t = [math.prod(sz_t[0:axis]), math.prod(
        sz_t[axis:axis+1]), math.prod(sz_t[axis+1:])]    
    i_f = (sz_t[1]+1) * sz_t[2] * np.arange(sz_t[0]).reshape(-1, 1, 1, 1) + sz_t[2] * np.arange(sz_t[1]).reshape((
        1, -1, 1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1, 1)) + np.array([0, sz_t[2]]).reshape((1, 1, 1, -1))
    if (x_c is None):
        x_c = 0.5*(x_f[:-1] + x_f[1:])
    dx_inv = np.tile(
        1 / (x_c[1:] - x_c[:-1]).reshape((1, -1, 1)), (sz_t[0], 1, sz_t[2]))
    values = np.empty(i_f.shape)
    values[:,0,:,0] = np.zeros((sz_t[0],sz_t[2]))
    values[:, 1:, :, 0] = dx_inv
    values[:, :-1, :, 1] = -dx_inv
    values[:,-1,:,1] = np.zeros((sz_t[0],sz_t[2]))
    Grad_int = csc_array((values.flatten(), i_f.flatten(), range(0,i_f.size + 1,2)), 
                      shape=(sz_t[0]*(sz_t[1]+1)*sz_t[2], sz_t[0]*sz_t[1]*sz_t[2]))
    return Grad_int

def construct_grad_bc(sz, x_f, x_c=None, bc=None, axis=0):
    """
    Construct the gradient matrix for the boundary faces 

    Args:
        sz (tuple): Size of the domain.
        x_f (ndarray): Face coordinates.
        x_c (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        bc (dict, optional): Boundary conditions. Default is None.
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.

    Returns:
        csc_array: Gradient matrix (Grad).
        csc_array: Contribution of the inhomogeneous BC to the gradient (grad_bc).
    """
    # Trick: Reshape sizes to triplet sz_t
    if not isinstance(sz, (list, tuple)):
        sz_f = [sz]
    else:
        sz_f = list(sz)
    sz_t = [math.prod(sz_f[0:axis]), math.prod(
        sz_f[axis:axis+1]), math.prod(sz_f[axis+1:])]
    
    # Create face arrays  
    sz_f[axis] = sz_f[axis] + 1
    sz_f_t = sz_t.copy()
    sz_f_t[1] = sz_t[1] + 1

    # Create boundary quantity sizes
    sz_bc = sz_f.copy()
    sz_bc[axis] = 1
    sz_bc_d = [sz_t[0], sz_t[2]]

    a, b, d = unwrap_bc(sz, bc)

    # Handle special case with one cell in the dimension axis
    if (sz_t[1] == 1):
        if (x_c is None):
            x_c = 0.5*(x_f[0:-1] + x_f[1:])
        i_c = sz_t[2] * np.arange(sz_t[0]).reshape((-1, 1, 1)) + np.array(
            (0, 0)).reshape((1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
        values = np.zeros(sz_f_t)
        alpha_1 = (x_f[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2L = (x_c[0] - x_f[0]) / ((x_f[1] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_0L = alpha_1 - alpha_2L
        alpha_2R = -(x_c[0] - x_f[1]) / ((x_f[0] - x_f[1]) * (x_f[0] - x_c[0]))
        alpha_0R = alpha_1 - alpha_2R
        fctr = ((b[0] + alpha_0L * a[0]) * (b[1] +
                    alpha_0R * a[1]) - alpha_2L * alpha_2R * a[0] * a[1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
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
        i_c = sz_t[1] * sz_t[2] * np.arange(sz_t[0]).reshape(-1, 1, 1) + sz_t[2] * np.array([0,1,sz_t[1]-2, sz_t[1]-1]).reshape((
            1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
        i_f = sz_f_t[1] * sz_t[2] * np.arange(sz_t[0]).reshape(-1, 1, 1) + sz_t[2] * np.array([0,0,sz_f_t[1]-1, sz_f_t[1]-1]).reshape((
            1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
        i_f_bc = sz_f_t[1] * sz_f_t[2] * np.arange(sz_f_t[0]).reshape((-1, 1, 1)) + sz_f_t[2] * np.array(
            [0, sz_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(sz_f_t[2]).reshape((1, 1, -1))
        values_bc = np.zeros((sz_t[0], 2, sz_t[2]))
        values = np.zeros((sz_t[0], 4, sz_t[2]))
        if (x_c is None):
            x_c = 0.5*np.array([x_f[0] + x_f[1], x_f[1] + x_f[2], x_f[-3] + x_f[-2], x_f[-2] + x_f[-1]])
        dx_inv = np.tile(
            1 / (x_c[1:] - x_c[:-1]).reshape((1, -1, 1)), (sz_t[0], 1, sz_t[2]))

        alpha_1 = (x_c[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_2 = (x_c[0] - x_f[0]) / ((x_c[1] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_0 = alpha_1 - alpha_2
        b[0] = b[0] / alpha_0
        fctr = (a[0] + b[0])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        b_fctr = b[0] * fctr
        b_fctr = b_fctr + np.zeros(sz_bc)
        b_fctr = np.reshape(b_fctr, sz_bc_d)
        d_fctr = d[0] * fctr
        d_fctr = d_fctr + np.zeros(sz_bc)
        d_fctr = np.reshape(d_fctr, sz_bc_d)
        values[:, 0, :] = b_fctr * alpha_1
        values[:, 1, :] = -b_fctr * alpha_2
        values_bc[:, 0, :] = -d_fctr

        alpha_1 = -(x_c[-2] - x_f[-1]) / ((x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_2 = -(x_c[-1] - x_f[-1]) / ((x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_0 = alpha_1 - alpha_2
        b[-1] = b[-1] / alpha_0
        fctr = (a[-1] + b[-1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        b_fctr = b[-1] * fctr
        b_fctr = b_fctr + np.zeros(sz_bc)
        b_fctr = np.reshape(b_fctr, sz_bc_d)
        d_fctr = d[-1] * fctr
        d_fctr = d_fctr + np.zeros(sz_bc)
        d_fctr = np.reshape(d_fctr, sz_bc_d)
        values[:, -2, :] = b_fctr * alpha_2
        values[:, -1, :] = -b_fctr * alpha_1
        values_bc[:, -1, :] = d_fctr

    Grad = csc_array((values.flatten(), (i_f.flatten(), i_c.flatten())), 
                      shape=(math.prod(sz_f_t), math.prod(sz_t)))
    grad_bc = csc_array((values_bc.flatten(), i_f_bc.flatten(), [
                         0, i_f_bc.size]), shape=(math.prod(sz_f_t),1))
    return Grad, grad_bc

def construct_div(sz, x_f, nu=0, axis=0):
    """
    Construct the Div matrix based on the given parameters.

    Args:
        sz (tuple): Size of the matrix.
        x_f (ndarray): Face array.
        nu (callable or int): The function or integer representing nu.
        axis (int): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The Div matrix.

    """
    # Rest of the function code...

    # Trick: Reshape sizes to triplet sz_t
    if not isinstance(sz, (list, tuple)):
        sz_f = [sz]
    else:
        sz_f = list(sz)
    sz_t = [math.prod(sz_f[0:axis]), math.prod(
        sz_f[axis:axis + 1]), math.prod(sz_f[axis + 1:])]

    # Create face arrays
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

def construct_convflux_upwind(sz, x_f, x_c=None, bc=None, v=1.0, axis=0):
    """
    Construct the Conv and conv_bc matrices based on the given parameters.

    Args:
        sz (tuple): Size of the matrices.
        x_c (ndarray, optional): The cell array. If not provided, it will be calculated based on the face array.
        x_f (ndarray): The face array.
        bc (list, optional): The boundary conditions. Default is None.
        v (ndarray): The velocity array.
        axis (int, optional): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The Conv matrix.
        csc_array: The conv_bc matrix.

    """
    if (x_c is None):
        x_c = 0.5*(x_f[0:-1]+x_f[1:])
    Grad = construct_convflux_upwind_int(sz, v, axis)
    if (bc is None):
        sz_f = sz.copy()
        sz_f[axis] +=1
        conv_bc = csc_array(shape=(math.prod(sz_f),1))
    else:
        Conv_bc, conv_bc = construct_convflux_upwind_bc(sz, x_f, x_c, bc, v, axis)
        Conv += Conv_bc
    return Conv, conv_bc

def construct_convflux_upwind_int(sz, v=1.0, axis=0):
    """
    Construct the Conv matrix based on the given parameters.

    Args:
        sz (tuple): Size of the matrices.
        v (ndarray): The velocity array.
        axis (int, optional): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The Conv matrix.

    """
    if not isinstance(sz, (list, tuple)):
        sz_f = [sz]
    else:
        sz_f = list(sz)
    sz_t = [math.prod(sz_f[0:axis]), math.prod(
        sz_f[axis:axis+1]), math.prod(sz_f[axis+1:])]

    sz_f[axis] = sz_f[axis] + 1
    sz_f_t = sz_t.copy()
    sz_f_t[1] = sz_t[1] + 1

    v = np.array(v) + np.zeros(sz_f)
    v = v.reshape(sz_f_t)
    fltr_v_pos = (v > 0)
    i_f = (sz_t[1]+1) * sz_t[2] * np.arange(sz_t[0]).reshape(-1, 1, 1) + sz_t[2] * np.arange(1,sz_t[1]).reshape((
        1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
    i_c = sz_t[1] * sz_t[2] * np.arange(sz_t[0]).reshape((-1, 1, 1)) + sz_t[2] * np.arange(1,sz_t[1]).reshape((
        1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
    i_c = i_c - sz_t[2] * fltr_v_pos[:, 1:-1, :]
    Conv = csc_array((v[:, 1:-1, :].flatten(), (i_f.flatten(), i_c.flatten())), shape=(
        math.prod(sz_f_t), math.prod(sz_t)))
    Conv.sort_indices()
    return Conv
    
def construct_convflux_upwind_bc(sz, x_f, x_c=None, bc=None, v=1.0, axis=0):
    """
    Construct the Conv and conv_bc matrices based on the given parameters.

    Args:
        sz (tuple): Size of the matrices.
        x_c (ndarray, optional): The cell array. If not provided, it will be calculated based on the face array.
        x_f (ndarray): The face array.
        bc (list, optional): The boundary conditions. Default is None.
        v (ndarray): The velocity array.
        axis (int, optional): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The Conv matrix.
        csc_array: The conv_bc matrix.

    """

     # Trick: Reshape sizes to triplet sz_t
    if not isinstance(sz, (list, tuple)):
        sz_f = [sz]
    else:
        sz_f = list(sz)
    sz_t = [math.prod(sz_f[0:axis]), math.prod(
        sz_f[axis:axis+1]), math.prod(sz_f[axis+1:])]
    
    # Create face arrays  
    sz_f[axis] = sz_f[axis] + 1
    sz_f_t = sz_t.copy()
    sz_f_t[1] = sz_t[1] + 1

    # Create boundary quantity sizes
    sz_bc = sz_f.copy()
    sz_bc[axis] = 1
    sz_bc_d = [sz_t[0], sz_t[2]]

    a, b, d = unwrap_bc(sz, bc)

    v = np.array(v) + np.zeros(sz_f)
    v = v.reshape(sz_f_t)
    fltr_v_pos = (v > 0)

    # Handle special case with one cell in the dimension axis
    if (sz_t[1] == 1):
        if (x_c is None):
            x_c = 0.5*(x_f[0:-1] + x_f[1:])
        i_c = sz_t[2] * np.arange(sz_t[0]).reshape((-1, 1, 1)) + np.array(
            (0, 0)).reshape((1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
        values = np.zeros(sz_f_t)
        alpha_1 = (x_f[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2L = (x_c[0] - x_f[0]) / ((x_f[1] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_0L = alpha_1 - alpha_2L
        alpha_2R = -(x_c[0] - x_f[1]) / ((x_f[0] - x_f[1]) * (x_f[0] - x_c[0]))
        alpha_0R = alpha_1 - alpha_2R        
        fctr = ((b[0] + alpha_0L * a[0]) * (b[1] +
                     alpha_0R * a[1]) - alpha_2L * alpha_2R * a[0] * a[1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        values = np.empty((sz_t[0],2,sz_t[2]))
        values[:, 0, :] = ((alpha_1 * a[0] * (a[1] * (alpha_0R - alpha_2L) + b[1])
                           * fctr + np.zeros(sz)).reshape(sz_bc_d))
        values[:, 1, :] = ((alpha_1 * a[1] * (a[0] * (alpha_0L - alpha_2R) + b[0])
                           * fctr + np.zeros(sz)).reshape(sz_bc_d))
        values = values * v
        Conv = csc_array((values.flatten(), i_c.flatten(), np.arange(
            0, i_c.size + 1)), shape=(math.prod(sz_f_t), math.prod(sz_t)))        
        
        i_f_bc = sz_f_t[1] * sz_f_t[2] * np.arange(sz_f_t[0]).reshape((-1, 1, 1)) + sz_f_t[2] * np.array(
            [0, sz_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(sz_f_t[2]).reshape((1, 1, -1))
        values_bc = np.empty((sz_t[0], 2, sz_t[2]))
        values_bc = np.zeros((sz_t[0], 2, sz_t[2]))
        values_bc[:, 0, :] = ((((a[1] * alpha_0R + b[1]) * d[0] - alpha_2L * a[0] * d[1])
                              * fctr + np.zeros(sz_bc)).reshape(sz_bc_d))
        values_bc[:, 1, :] = ((((a[0] * alpha_0L + b[0]) * d[1] - alpha_2R * a[1] * d[0])
                              * fctr + np.zeros(sz_bc)).reshape(sz_bc_d))
        values_bc = values_bc * v
    else:
        i_c = sz_t[1] * sz_t[2] * np.arange(sz_t[0]).reshape(-1, 1, 1) + sz_t[2] * np.array([0,1,sz_t[1]-2, sz_t[1]-1]).reshape((
            1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
        i_f = sz_f_t[1] * sz_t[2] * np.arange(sz_t[0]).reshape(-1, 1, 1) + sz_t[2] * np.array([0,0,sz_f_t[1]-1, sz_f_t[1]-1]).reshape((
            1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
        i_f_bc = sz_f_t[1] * sz_f_t[2] * np.arange(sz_f_t[0]).reshape((-1, 1, 1)) + sz_f_t[2] * np.array(
            [0, sz_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(sz_f_t[2]).reshape((1, 1, -1))
        values_bc = np.zeros((sz_t[0], 2, sz_t[2]))
        values = np.zeros((sz_t[0], 4, sz_t[2]))
        if (x_c is None):
            x_c = 0.5*np.array([x_f[0] + x_f[1], x_f[1] + x_f[2], x_f[-3] + x_f[-2], x_f[-2] + x_f[-1]])

        alpha_1 = (x_c[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_2 = (x_c[0] - x_f[0]) / ((x_c[1] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a[0] + b[0])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a[0] * fctr
        a_fctr = a_fctr + np.zeros(sz_bc)
        a_fctr = np.reshape(a_fctr, sz_bc_d)
        d_fctr = d[0] * fctr
        d_fctr = d_fctr + np.zeros(sz_bc)
        d_fctr = np.reshape(d_fctr, sz_bc_d)
        values[:, 0, :] = (a_fctr * alpha_1) * v[:, 0, :]
        values[:, 1, :] = -a_fctr * alpha_2 * v[:, 0, :]
        values_bc[:, 0, :] = d_fctr * v[:, 0, :]

        alpha_1 = -(x_c[-2] - x_f[-1]) / ((x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_2 = -(x_c[-1] - x_f[-1]) / ((x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_0 = alpha_1 - alpha_2
        
        fctr = (alpha_0 * a[-1] + b[-1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a[-1] * fctr
        a_fctr = a_fctr + np.zeros(sz_bc)
        a_fctr = np.reshape(a_fctr, sz_bc_d)
        d_fctr = d[-1] * fctr
        d_fctr = d_fctr + np.zeros(sz_bc)
        d_fctr = np.reshape(d_fctr, sz_bc_d)
        values[:, -1, :] = (1 + (a_fctr * alpha_1 - 1) * ~
                            fltr_v_pos[:, -1, :]) * v[:, -1, :]
        values[:, -2, :] = a_fctr * alpha_2 * v[:, -1, :]
        values_bc[:, -1, :] = d_fctr * v[:, -1, :]
        Conv = csc_array((values.flatten(), (i_f.flatten(), i_c.flatten())), shape=(
            math.prod(sz_f_t), math.prod(sz_t)))
        Conv.sort_indices()

    conv_bc = csc_array((values_bc.flatten(), i_f_bc.flatten(), [
                         0, i_f_bc.size]), shape=(math.prod(sz_f_t),1))
    return Conv, conv_bc

def construct_convflux_upwind_old(sz, x_f, x_c=None, bc=None, v=1.0, axis=0):
    """
    Construct the Conv and conv_bc matrices based on the given parameters.

    Args:
        sz (tuple): Size of the matrices.
        x_c (ndarray, optional): The cell array. If not provided, it will be calculated based on the face array.
        x_f (ndarray): The face array.
        bc (list, optional): The boundary conditions. Default is None.
        v (ndarray): The velocity array.
        axis (int, optional): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The Conv matrix.
        csc_array: The conv_bc matrix.

    """
    if not isinstance(sz, (list, tuple)):
        sz_f = [sz]
    else:
        sz_f = list(sz)
    sz_t = [math.prod(sz_f[0:axis]), math.prod(
        sz_f[axis:axis+1]), math.prod(sz_f[axis+1:])]

    sz_f[axis] = sz_f[axis] + 1
    sz_f_t = sz_t.copy()
    sz_f_t[1] = sz_t[1] + 1

    sz_bc = sz_f.copy()
    sz_bc[axis] = 1
    sz_bc_d = [sz_t[0], sz_t[2]]

    a, b, d = unwrap_bc(sz, bc)

    v = np.array(v) + np.zeros(sz_f)
    v = v.reshape(sz_f_t)
    fltr_v_pos = (v > 0)
    
    if (x_c is None):
        x_c = 0.5*(x_f[0:-1]+x_f[1:])

    if sz_t[1] == 1:
        i_c = sz_t[1] * sz_t[2] * np.arange(sz_t[0]).reshape((-1, 1, 1)) + np.array(
            (0, 0)).reshape((1, -1, 1)) + np.arange(sz_t[2]).reshape((1, 1, -1))
        values = np.zeros(sz_f_t)
        alpha_1 = (x_f[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2L = (x_c[0] - x_f[0]) / ((x_f[1] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_0L = alpha_1 - alpha_2L
        alpha_2R = -(x_c[0] - x_f[1]) / ((x_f[0] - x_f[1]) * (x_f[0] - x_c[0]))
        alpha_0R = alpha_1 - alpha_2R
        fctr = ((b[0] + alpha_0L * a[0]) * (b[1] +
                     alpha_0R * a[1]) - alpha_2L * alpha_2R * a[0] * a[1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        values[:, 0, :] = ((alpha_1 * a[0] * (a[1] * (alpha_0R - alpha_2L) + b[1])
                           * fctr + np.zeros(sz)).reshape(sz_bc_d))
        values[:, 1, :] = ((alpha_1 * a[1] * (a[0] * (alpha_0L - alpha_2R) + b[0])
                           * fctr + np.zeros(sz)).reshape(sz_bc_d))
        values = values * v
        Conv = csc_array((values.flatten(), i_c.flatten(), np.arange(
            0, i_c.size + 1)), shape=(math.prod(sz_f_t), math.prod(sz_t)))

        i_f_bc = sz_f_t[1] * sz_f_t[2] * np.arange(sz_f_t[0]).reshape((-1, 1, 1)) + sz_f_t[2] * np.array(
            [0, sz_f_t[1] - 1]).reshape((1, -1, 1)) + np.arange(sz_f_t[2]).reshape((1, 1, -1))
        values_bc = np.zeros((sz_t[0], 2, sz_t[2]))
        values_bc[:, 0, :] = ((((a[1] * alpha_0R + b[1]) * d[0] - alpha_2L * a[0] * d[1])
                              * fctr + np.zeros(sz_bc)).reshape(sz_bc_d))
        values_bc[:, 1, :] = ((((a[0] * alpha_0L + b[0]) * d[1] - alpha_2R * a[1] * d[0])
                              * fctr + np.zeros(sz_bc)).reshape(sz_bc_d))
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
        fctr = (alpha_0 * a[0] + b[0])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a[0] * fctr
        a_fctr = a_fctr + np.zeros(sz_bc)
        a_fctr = np.reshape(a_fctr, sz_bc_d)
        d_fctr = d[0] * fctr
        d_fctr = d_fctr + np.zeros(sz_bc)
        d_fctr = np.reshape(d_fctr, sz_bc_d)
        values[:, 0, :] = (a_fctr * alpha_1) * v[:, 0, :]
        values[:, 1, :] = -a_fctr * alpha_2 * v[:, 0, :]
        values_bc[:, 0, :] = d_fctr * v[:, 0, :]

        alpha_1 = -(x_c[-2] - x_f[-1]) / \
            ((x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_2 = -(x_c[-1] - x_f[-1]) / \
            ((x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a[-1] + b[-1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a[-1] * fctr
        a_fctr = a_fctr + np.zeros(sz_bc)
        a_fctr = np.reshape(a_fctr, sz_bc_d)
        d_fctr = d[-1] * fctr
        d_fctr = d_fctr + np.zeros(sz_bc)
        d_fctr = np.reshape(d_fctr, sz_bc_d)
        values[:, -1, :] = (1 + (a_fctr * alpha_1 - 1) * ~
                            fltr_v_pos[:, -1, :]) * v[:, -1, :]
        values[:, -2, :] = a_fctr * alpha_2 * v[:, -1, :]
        values_bc[:, -1, :] = d_fctr * v[:, -1, :]
        Conv = csc_array((values.flatten(), (i_f.flatten(), i_c.flatten())), shape=(
            math.prod(sz_f_t), math.prod(sz_t)))
        Conv.sort_indices()

    conv_bc = csc_array((values_bc.flatten(), i_f_bc.flatten(), [
                         0, i_f_bc.size]), shape=(math.prod(sz_f_t),1))
    return Conv, conv_bc


def numjac_local(f, c, eps_jac=1e-6, axis=0):
    """
    Compute the local numerical Jacobian matrix and function values for the given function and initial values.
    
    The function 'f' is assumed to be local, meaning it can be dependent on other components in the array along the 'axis' dimension.
    numjac_local can be used to compute Jacobians of functions like reaction, accumulation, or mass transfer terms, where there 
    is a dependence only on local components in a spatial cell.
    The best choice is to set up the problem such that 'axis' is the last dimension of the multidimensional array, 
    as this will result in a nicely block-structured Jacobian matrix.

    Args:
        f (callable): The function for which to compute the Jacobian.
        c (ndarray): The value at which the Jacobian should be evaluated.
        eps_jac (float, optional): The perturbation value for computing the Jacobian. Defaults to 1e-6.
        axis (int, optional): The axis along which components are coupled. Default is 0.

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


def interp_stagg_to_cntr(c_f, x_f, x_c = None, axis = 0):
    """
    Interpolate values at staggered positions to cell-centers using linear interpolation.

    Args:
        axis (int): Dimension that is interpolated.
        c_f (ndarray): Quantities at staggered positions.
        x_c (ndarray): Cell-centered positions.
        x_f (ndarray): Positions of cell-faces (numel(x_f) = numel(x_c) + 1).

    Returns:
        ndarray: Interpolated concentrations at the cell-centered positions.

    """
    sz_f = list(c_f.shape)
    sz_f = sz_f + [1,] * (1 + axis - len(sz_f))  # size padded with extra 1's if needed
    sz_f_t = [math.prod(sz_f[:axis]), sz_f[axis], math.prod(sz_f[axis + 1:])]  # sizes for reshape as a triplet
    sz = sz_f.copy()
    sz[axis] = sz[axis] - 1
    sz_t = sz_f_t.copy()
    sz_t[1] = sz[axis]
    c_f = c_f.reshape(sz_f_t)
    if (x_c is None):
        c_c =  0.5 * (c_f[:, 1:, :] + c_f[:, :-1, :])
    else:
        wght = (x_c - x_f[:-1]) / (x_f[1:] - x_f[:-1])
        c_c = c_f[:, :-1, :] + wght.reshape((1,-1,1)) * (c_f[:, 1:, :] - c_f[:, :-1, :])
    c_c = c_c.reshape(sz)

    return c_c

def interp_cntr_to_stagg(c_c, x_f, x_c=None, axis=0):
    """
    Interpolate values at staggered positions to cell-centers using linear interpolation.

    Args:
        c_c (ndarray): Quantities at cell-centered positions.
        x_f (ndarray): Positions of cell-faces (numel(x_f) = numel(x_c) + 1).
        x_c (ndarray, optional): Cell-centered positions. If not provided, the interpolated values will be averaged between adjacent staggered positions.
        axis (int, optional): Dimension along which interpolation is performed. Default is 0.

    Returns:
        ndarray: Interpolated concentrations at staggered positions.

    """
    sz = list(c_c.shape)
    sz = sz + [1,] * (1 + axis - len(sz))  # size padded with extra 1's if needed
    sz_t = [math.prod(sz[:axis]), sz[axis], math.prod(sz[axis + 1:])]  # sizes for reshape as a triplet
    sz_f = sz.copy()
    sz_f[axis] = sz[axis] + 1
    sz_f_t = sz_t.copy()
    sz_f_t[1] = sz_f[axis]
    if (x_c is None):
        x_c = 0.5*(x_f[:-1]+x_f[1:])
    wght = (x_f[1:-1] - x_c[:-1]) / (x_c[1:] - x_c[:-1])
    c_c = c_c.reshape(sz_t)
    c_f = np.empty(sz_f_t)
    c_f[:,1:-1,:] = c_c[:, :-1, :] + wght.reshape((1,-1,1)) * (c_c[:, 1:, :] - c_c[:, :-1, :])
    c_f[:,0,:] = (c_c[:,0,:]*(x_c[1]-x_f[0]) - c_c[:,1,:]*(x_c[0]-x_f[0]))/(x_c[1]-x_c[0])
    c_f[:,-1,:] = (c_c[:,-1,:]*(x_f[-1]-x_c[-2]) - c_c[:,-2,:]*(x_f[-1]-x_c[-1]))/(x_c[-1]-x_c[-2])
    c_f = c_f.reshape(sz_f)
    return c_f


def non_uniform_grid(x_L, x_R, n, dx_inf, factor):
    """
    Generate a non-uniform grid of points in the interval [x_L, x_R]
    With factor > 1 the refinement will be at the left wall, 
    with 1/factor you will get the same refinement at the right wall.
    
    Parameters:
        x_L (float): Start point of the interval.
        x_R (float): End point of the interval.
        n (int): Total number of points in the grid (excluding x1 and x2).
        dx_inf (float): Limiting grid spacing.
        factor (float): Factor used to increase grid spacing.

    Returns:
        numpy.ndarray: Array containing the non-uniform grid points.
    """
    a = np.log(factor)
    unif = np.arange(n)
    b = np.exp(-a * unif)
    L = x_R - x_L
    c = (np.exp(a * (L / dx_inf - n + 1.0)) - b[-1]) / (1 - b[-1])
    x_f = x_L + unif * dx_inf + np.log((1 - c) * b + c) * (dx_inf / a)
    return x_f

def unwrap_bc(sz, bc):
    """
    Unwrap the boundary conditions for a given size.

    Args:
        sz (tuple): Size of the domain.
        bc (dict): Boundary conditions.

    Returns:
        tuple: Unwrapped boundary conditions (a, b, d).
    """
    if not isinstance(sz, (list,tuple)):
        lgth_sz = 1
    else:
        lgth_sz = len(sz)
      
    a = [None, None]
    b = [None, None]
    d = [None, None]
    
    if (bc is None):
        a[0] = np.zeros((1,) * lgth_sz)
        a[1] = np.zeros((1,) * lgth_sz)
        b[0] = np.zeros((1,) * lgth_sz)
        b[1] = np.zeros((1,) * lgth_sz)
        d[0] = np.zeros((1,) * lgth_sz)
        d[1] = np.zeros((1,) * lgth_sz)
    else:
        a[0] = np.array(bc['a'][0])
        a[0] = a[0][(..., *([np.newaxis]*(lgth_sz-a[0].ndim)))]
        a[1] = np.array(bc['a'][1])
        a[1] = a[1][(..., *([np.newaxis]*(lgth_sz-a[1].ndim)))]
        b[0] = np.array(bc['b'][0])
        b[0] = b[0][(..., *([np.newaxis]*(lgth_sz-b[0].ndim)))]
        b[1] = np.array(bc['b'][1])
        b[1] = b[1][(..., *([np.newaxis]*(lgth_sz-b[1].ndim)))]
        d[0] = np.array(bc['d'][0])
        d[0] = d[0][(..., *([np.newaxis]*(lgth_sz-d[0].ndim)))]
        d[1] = np.array(bc['d'][1])
        d[1] = d[1][(..., *([np.newaxis]*(lgth_sz-d[1].ndim)))]
    return a, b, d
