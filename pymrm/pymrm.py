"""
Module Name: pymrm
Author: E.A.J.F. Peters, M. Sint Annaland, M. Galanti, D. Rieder
Date: 18/04/2024
License: MIT License
Version: 1.3.1

This module provides functions for multiphase reactor modeling.

Functions:

- construct_grad(shape, x_f, x_c=None, bc=None, axis=0): Construct the gradient matrix.
- construct_grad_int(shape, x_f, x_c=None, axis=0): Construct the gradient matrix for internal faces.
- construct_grad_bc(shape, x_f, x_c=None, bc=None, axis=0): Construct the gradient matrix for boundary faces.
- construct_div(shape, x_f, nu=0, axis=0): Construct the divergence matrix based on the given parameters.
- construct_convflux_upwind(shape, x_f, x_c=None, bc=None, v=0, axis=0): Construct the convective flux matrix
- construct_convflux_upwind_int(shape, v=1.0, axis=0): Construct the convective flux matrix for internal faces
- construct_convflux_upwind_bc(shape, x_f, x_c=None, bc=None, v=0, axis=0): Construct the convective flux matrix for boundary faces.
- construct_coefficient_matrix(coeffs, shape=None): Construct a diagonal matrix with coefficients on its diagonal.
- numjac_local(f, c, eps_jac=1e-6, axis=0): Compute the local numerical Jacobian matrix and function values for the given function and initial values.
- newton(g, c, tol=1e-6, itmax=100, solver='bicgstab', filter=True, **param): Performs a Newton-Raphson iteration to seek the root of A(c)*c = b(c).
- interp_stagg_to_cntr(c_f, x_f, x_c=None, axis=0): Interpolate values at staggered positions to cell-centers using linear interpolation.
- interp_cntr_to_stagg(c_c, x_f, x_c=None, axis=0): Interpolate values at cell-centered positions to staggered positions using linear interpolation and extrapolation at the wall.
- interp_cntr_to_stagg_tvd(c_c, x_f, x_c=None, bc=None, v=0, tvd_limiter=None, axis=0): Interpolate values at cell-centered positions to staggered positions using linear interpolation and extrapolation at the wall.
- upwind(c_norm_C, x_norm_C, x_norm_d) : TVD limiter - Upwind.
- minmod(c_norm_C, x_norm_C, x_norm_d) : TVD limiter - Minmod.
- osher(c_norm_C, x_norm_C, x_norm_d)  : TVD limiter - Osher.
- clam(c_norm_C, x_norm_C, x_norm_d)   : TVD limiter - CLAM.
- muscl(c_norm_C, x_norm_C, x_norm_d)  : TVD limiter - MUSCL.
- smart(c_norm_C, x_norm_C, x_norm_d)  : TVD limiter - SMART.
- stoic(c_norm_C, x_norm_C, x_norm_d)  : TVD limiter - Stoic.
- vanleer(c_norm_C, x_norm_C, x_norm_d): TVD limiter - van Leer.
- non_uniform_grid(x_L, x_R, n, dx_inf, factor): Generate a non-uniform grid of points in the interval [x_L, x_R].
- unwrap_bc(shape, bc): Unwrap the boundary conditions for a given shape. Mostly used by other functions.

Note: Please refer to the function descriptions for more details on their arguments and usage.
"""

import numpy as np
from scipy.sparse import csc_array, diags, linalg, block_diag
from scipy.linalg import norm
from scipy.optimize import OptimizeResult
import math

def construct_grad(shape, x_f, x_c=None, bc=None, axis=0):
    """
    Construct the gradient matrix.

    Args:
        shape (tuple): shape of the domain.
        x_f (ndarray): Face positions
        x_c (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        bc (dict, optional): Boundary conditions. Default is None.
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.

    Returns:
        csc_array: Gradient matrix (Grad).
        csc_array: Contribution of the inhomogeneous BC to the gradient (grad_bc).
    """
    # The contributions to the gradient on internal faces that 
    # are independent of boundary conditions are computed separately
    # from those on boundary faces. On boundary faces, 
    # in case of inhomogeneous boundary conditions, 
    # then there can also be constant contribution, grad_bc   
    if (x_c is None):
        x_c = 0.5*(x_f[0:-1]+x_f[1:])
    
    Grad = construct_grad_int(shape, x_f, x_c, axis)
    
    if (bc is None):
        shape_f = shape.copy()
        if (axis<0):
            axis += len(shape)       
        shape_f[axis] +=1
        grad_bc = csc_array(shape=(math.prod(shape_f),1))
    else:
        Grad_bc, grad_bc = construct_grad_bc(shape, x_f, x_c, bc, axis)
        Grad += Grad_bc
    return Grad, grad_bc

def construct_grad_int(shape, x_f,  x_c=None, axis=0):
    """
    Construct the gradient matrix for internal faces.

    Args:
        shape (tuple): shape of the domain.
        x_f (ndarray): Face coordinates.
        x_c (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.
    
    Returns:
        csc_array: Gradient matrix (Grad).
        csc_array: Contribution of the inhomogeneous BC to the gradient (grad_bc).
    """
    # Explanation of implementation:
    # The vectors of unknown are flattened versions of a multi-dimensional array.
    # In this multi-dimensional arrays some dimensions will represent spatial directions, i.e.,
    # the indices are cell indices. The gradient denotes a finite difference discretization 
    # of the spatial differentiation in that direction. The direction of differentiation 
    # is denoted by 'axis'.
    # For multi-dimenional arrays the following trick is used: Arrays are reshaped to 
    # three-dimensional arrays where the middle dimension now corresponds to the direction of differentiation.
        
    if not isinstance(shape, (list, tuple)):
        shape_t = [shape]
    else:
        shape_t = list(shape)
    if (axis<0):
        axis += len(shape)
    shape_t = [math.prod(shape_t[0:axis]), math.prod(
        shape_t[axis:axis+1]), math.prod(shape_t[axis+1:])] # shape of the reshaped 3 dimenional array   
    # The gradient will be represented by a csc array.
    # Each column corresponds to the contribution of a value cell
    # For each column there are two entries corresponding to the two faces, except for the cells near the boundary
    # The indices of these faces are stored in an array i_f, with i_f.shape = [shape_t[0], shape_t[1]+1 , shape_t[2], 2]
    # Note the shape_t[1]+1, because there is 1 more face than cells.
    # The linear index is: i_f[i,j,k,m] = 2*(shape_t[2]*((shape_t[1]+1)*j + k)) + m
    i_f = (shape_t[1]+1) * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1, 1) + shape_t[2] * np.arange(shape_t[1]).reshape((
        1, -1, 1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1, 1)) + np.array([0, shape_t[2]]).reshape((1, 1, 1, -1))
    
    if (x_c is None):
        x_c = 0.5*(x_f[:-1] + x_f[1:])
    
    dx_inv = np.tile(
        1 / (x_c[1:] - x_c[:-1]).reshape((1, -1, 1)), (shape_t[0], 1, shape_t[2]))
    values = np.empty(i_f.shape)
    values[:,0,:,0] = np.zeros((shape_t[0],shape_t[2]))
    values[:, 1:, :, 0] = dx_inv
    values[:, :-1, :, 1] = -dx_inv
    values[:,-1,:,1] = np.zeros((shape_t[0],shape_t[2]))
    Grad_int = csc_array((values.ravel(), i_f.ravel(), range(0,i_f.size + 1,2)), 
                      shape=(shape_t[0]*(shape_t[1]+1)*shape_t[2], shape_t[0]*shape_t[1]*shape_t[2]))
    return Grad_int

def construct_grad_bc(shape, x_f, x_c=None, bc=None, axis=0):
    """
    Construct the gradient matrix for the boundary faces 

    Args:
        shape (tuple): shape of the domain.
        x_f (ndarray): Face coordinates.
        x_c (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        bc (dict, optional): Boundary conditions. Default is None.
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.

    Returns:
        csc_array: Gradient matrix (Grad).
        csc_array: Contribution of the inhomogeneous BC to the gradient (grad_bc).
    """
    # For the explanation of resizing see construct_grad_int
    if not isinstance(shape, (list, tuple)):
        shape_f = [shape]
    else:
        shape_f = list(shape)
    if (axis<0):
        axis += len(shape)
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

    a, b, d = unwrap_bc(shape, bc) # Get a, b, and d from dictionary

    # Handle special case with one cell in the dimension axis.
    # This is convenient e.g. for flexibility where you can choose not to 
    # spatially discretize a direction, but still use a BC, e.g. with a mass transfer coefficient
    # It is a bit subtle because in this case the two opposite faces influence each other
    if (shape_t[1] == 1):
        if (x_c is None):
            x_c = 0.5*(x_f[0:-1] + x_f[1:])
        i_c = shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1)) + np.array(
            (0, 0)).reshape((1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        values = np.zeros(shape_f_t)
        alpha_1 = (x_f[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2L = (x_c[0] - x_f[0]) / ((x_f[1] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_0L = alpha_1 - alpha_2L
        alpha_2R = -(x_c[0] - x_f[1]) / ((x_f[0] - x_f[1]) * (x_f[0] - x_c[0]))
        alpha_0R = alpha_1 - alpha_2R
        fctr = ((b[0] + alpha_0L * a[0]) * (b[1] +
                    alpha_0R * a[1]) - alpha_2L * alpha_2R * a[0] * a[1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        value = alpha_1 * \
            b[0] * (a[1] * (alpha_0R - alpha_2L) + b[1]) * fctr + np.zeros(shape)
        values[:, 0, :] = np.reshape(value, shape_bc_d)
        value = alpha_1 * \
            b[1] * (a[0] * (-alpha_0L + alpha_2R) - b[0]) * fctr + np.zeros(shape)
        values[:, 1, :] = np.reshape(value, shape_bc_d)

        i_f_bc = shape_f_t[1] * shape_f_t[2] * np.arange(shape_f_t[0]).reshape((-1, 1, 1)) + shape_f_t[2] * np.array(
            [0, shape_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(shape_f_t[2]).reshape((1, 1, -1))
        values_bc = np.zeros((shape_t[0], 2, shape_t[2]))
        value = ((a[1] * (-alpha_0L * alpha_0R + alpha_2L * alpha_2R) - alpha_0L *
                 b[1]) * d[0] - alpha_2L * b[0] * d[1]) * fctr + np.zeros(shape_bc)
        values_bc[:, 0, :] = np.reshape(value, shape_bc_d)
        value = ((a[0] * (+alpha_0L * alpha_0R - alpha_2L * alpha_2R) + alpha_0R *
                 b[0]) * d[1] + alpha_2R * b[1] * d[0]) * fctr + np.zeros(shape_bc)
        values_bc[:, 1, :] = np.reshape(value, shape_bc_d)
    else:
        i_c = shape_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1) + shape_t[2] * np.array([0,1,shape_t[1]-2, shape_t[1]-1]).reshape((
            1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        i_f = shape_f_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1) + shape_t[2] * np.array([0,0,shape_f_t[1]-1, shape_f_t[1]-1]).reshape((
            1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        i_f_bc = shape_f_t[1] * shape_f_t[2] * np.arange(shape_f_t[0]).reshape((-1, 1, 1)) + shape_f_t[2] * np.array(
            [0, shape_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(shape_f_t[2]).reshape((1, 1, -1))
        values_bc = np.zeros((shape_t[0], 2, shape_t[2]))
        values = np.zeros((shape_t[0], 4, shape_t[2]))
        if (x_c is None):
            x_c = 0.5*np.array([x_f[0] + x_f[1], x_f[1] + x_f[2], x_f[-3] + x_f[-2], x_f[-2] + x_f[-1]])
        dx_inv = np.tile(
            1 / (x_c[1:] - x_c[:-1]).reshape((1, -1, 1)), (shape_t[0], 1, shape_t[2]))

        alpha_1 = (x_c[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_2 = (x_c[0] - x_f[0]) / ((x_c[1] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_0 = alpha_1 - alpha_2
        b[0] = b[0] / alpha_0
        fctr = (a[0] + b[0])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        b_fctr = b[0] * fctr
        b_fctr = b_fctr + np.zeros(shape_bc)
        b_fctr = np.reshape(b_fctr, shape_bc_d)
        d_fctr = d[0] * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
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
        b_fctr = b_fctr + np.zeros(shape_bc)
        b_fctr = np.reshape(b_fctr, shape_bc_d)
        d_fctr = d[-1] * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        values[:, -2, :] = b_fctr * alpha_2
        values[:, -1, :] = -b_fctr * alpha_1
        values_bc[:, -1, :] = d_fctr

    Grad = csc_array((values.ravel(), (i_f.ravel(), i_c.ravel())), 
                      shape=(math.prod(shape_f_t), math.prod(shape_t)))
    grad_bc = csc_array((values_bc.ravel(), i_f_bc.ravel(), [
                         0, i_f_bc.size]), shape=(math.prod(shape_f_t),1))
    return Grad, grad_bc

def construct_diff_flux(shape, A_grad, b_grad, param, axis=0):
    """
    Computes the diffusion fluxes in sparse format
    
    Parameters
    ----------  
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.
        shape:  shape (tuple): shape of the domain.
        A_grad: sparse matrix containing the homogeneous terms of the discretized gradient operator
        b_grad: sparse vector containing the heterogeneous terms resulting from the boundary conditions
        param:  a dictionary that should contain the key 'Diff' where the diffusion coefficients
                are stored, that can be a constant, a diagonal vector of length Nc, or an Nc x Nc matrix
    
    Returns
    -------
        diff_flux = A_diff*phi + b_diff
        A_diff: sparse matrix for the homogeneous part of the diffusion fluxes
        b_diff: sparse column vector for the inhomogeneous contributions resulting from the boundary conditions
    
    Example
    -------
        The command to obtain the diffusion fluxes in the x-direction is:
        A_diff_x, b_diff_x = diff_flux(shape, sz, A_grad, b_grad, axis=0, **param)
        
    created by: M. van Sint Annaland
    modified by: M. Galanti
    date: January 2024
    """ 
    
    sf = np.asarray(shape)
    sf[axis] += 1
    nc = shape[-1]
    ntotf = np.prod(sf[0:-1])

    if isinstance(param['Diff'], float):
        A_diff = -param['Diff']*A_grad
        b_diff = -param['Diff']*b_grad
    elif isinstance(param['Diff'][axis-1], float):
        Diff = np.asarray(param['Diff'])*np.eye(nc)
        Diff_mat_x = block_diag([Diff]*ntotf, format='csc')
        A_diff = -Diff_mat_x @ A_grad
        b_diff = -Diff_mat_x @ b_grad
    else:
        Diff_mat_x = block_diag([param['Diff']]*ntotf, format='csc')
        A_diff = -Diff_mat_x @ A_grad
        b_diff = -Diff_mat_x @ b_grad

    return A_diff, b_diff

def construct_div(shape, x_f, nu=0, axis=0):
    """
    Construct the Div matrix based on the given parameters.

    Args:
        shape (tuple): shape of the multi-dimesional array
        x_f (ndarray): Face positions
        nu (int or callable): The integer representing geometry (0: flat, 1: cylindrical, 2: spherical). If it is a function it specifies an area at position x.
        axis (int): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The Div matrix.

    """

    # Trick: Reshape to triplet shape_t, see compute_grad_int for explanation
    if not isinstance(shape, (list, tuple)):
        shape_f = [shape]
    else:
        shape_f = list(shape)
    if (axis<0):
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

    values = np.empty((shape_t[1],2))
    values[:, 0] = -area[:-1] * dvol_inv
    values[:, 1] =  area[1:] * dvol_inv
    values = np.tile(values.reshape((1,-1,1,2)),(shape_t[0],1,shape_t[2]))

    num_cells = np.prod(shape_t, dtype=int);
    Div = csc_array(
        (values.ravel(),(np.repeat(np.arange(num_cells),2) , 
                           i_f.ravel())),
        shape=(num_cells, np.prod(shape_f_t, dtype=int))
    )
    Div.sort_indices()
    return Div

def construct_convflux_upwind(shape, x_f, x_c=None, bc=None, v=1.0, axis=0):
    """
    Construct the Conv and conv_bc matrices based on the given parameters.

    Args:
        shape (tuple): shape of the multi-dimensional array.
        x_f (ndarray): Face positions.
        x_c (ndarray, optional): Cell positions. If not provided, it will be calculated based on the face array.
        bc (list, optional): The boundary conditions. Default is None.
        v (ndarray): Velocities on face positions
        axis (int, optional): The axis along which the convection takes place is performed. Default is 0.

    Returns:
        csc_array: The Conv matrix.
        csc_array: The conv_bc matrix.
    """
    if (x_c is None):
        x_c = 0.5*(x_f[0:-1]+x_f[1:])
    Conv = construct_convflux_upwind_int(shape, v, axis)
    if (bc is None):
        shape_f = shape.copy()
        shape_f[axis] +=1
        conv_bc = csc_array(shape=(math.prod(shape_f),1))
    else:
        Conv_bc, conv_bc = construct_convflux_upwind_bc(shape, x_f, x_c, bc, v, axis)
        Conv += Conv_bc
    return Conv, conv_bc

def construct_convflux_upwind_int(shape, v=1.0, axis=0):
    """
    Construct the Conv matrix for internal faces, e.g. all faces except these on the boundaries

    Args:
        shape (tuple): shape of the ndarrays.
        v (ndarray): The velocity array.
        axis (int, optional): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The Conv matrix.

    """
    if not isinstance(shape, (list, tuple)):
        shape_f = [shape]
    else:
        shape_f = list(shape)
    if (axis<0):
        axis += len(shape)
    shape_t = [math.prod(shape_f[0:axis]), math.prod(
        shape_f[axis:axis+1]), math.prod(shape_f[axis+1:])]

    shape_f[axis] = shape_f[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_t[1] + 1

    v = np.array(v) + np.zeros(shape_f)
    v = v.reshape(shape_f_t)
    fltr_v_pos = (v > 0)
    i_f = (shape_t[1]+1) * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1) + shape_t[2] * np.arange(1,shape_t[1]).reshape((
        1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
    i_c = shape_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1)) + shape_t[2] * np.arange(1,shape_t[1]).reshape((
        1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
    i_c = i_c - shape_t[2] * fltr_v_pos[:, 1:-1, :]
    Conv = csc_array((v[:, 1:-1, :].ravel(), (i_f.ravel(), i_c.ravel())), shape=(
        math.prod(shape_f_t), math.prod(shape_t)))
    Conv.sort_indices()
    return Conv
    
def construct_convflux_upwind_bc(shape, x_f, x_c=None, bc=None, v=1.0, axis=0):
    """
    Construct the Conv and conv_bc matrices for the boundary faces only

    Args:
        shape (tuple): shape of the multidimensional array.
        x_c (ndarray, optional): Cell-centered positions. If not provided, it will be calculated based on the face array.
        x_f (ndarray): Face positions
        bc (list, optional): The boundary conditions. Default is None.
        v (ndarray): The velocity array.
        axis (int, optional): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The Conv matrix.
        csc_array: The conv_bc matrix.

    """

     # Trick: Reshape to triplet shape_t
    if not isinstance(shape, (list, tuple)):
        shape_f = [shape]
    else:
        shape_f = list(shape)
    if (axis<0):
        axis += len(shape)
    shape_t = [math.prod(shape_f[0:axis]), math.prod(shape_f[axis:axis+1]), math.prod(shape_f[axis+1:])]
    
    # Create face arrays  
    shape_f[axis] = shape_f[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_t[1] + 1

    # Create boundary quantity shapes
    shape_bc = shape_f.copy()
    shape_bc[axis] = 1
    shape_bc_d = [shape_t[0], shape_t[2]]

    a, b, d = unwrap_bc(shape, bc)

    v = np.array(v) + np.zeros(shape_f)
    v = v.reshape(shape_f_t)
    fltr_v_pos = (v > 0)

    # Handle special case with one cell in the dimension axis
    if (shape_t[1] == 1):
        if (x_c is None):
            x_c = 0.5*(x_f[0:-1] + x_f[1:])
        i_c = shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1)) + np.array(
            (0, 0)).reshape((1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        values = np.zeros(shape_f_t)
        alpha_1 = (x_f[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2L = (x_c[0] - x_f[0]) / ((x_f[1] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_0L = alpha_1 - alpha_2L
        alpha_2R = -(x_c[0] - x_f[1]) / ((x_f[0] - x_f[1]) * (x_f[0] - x_c[0]))
        alpha_0R = alpha_1 - alpha_2R        
        fctr = ((b[0] + alpha_0L * a[0]) * (b[1] +
                     alpha_0R * a[1]) - alpha_2L * alpha_2R * a[0] * a[1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        values = np.empty((shape_t[0],2,shape_t[2]))
        values[:, 0, :] = ((alpha_1 * a[0] * (a[1] * (alpha_0R - alpha_2L) + b[1])
                           * fctr + np.zeros(shape)).reshape(shape_bc_d))
        values[:, 1, :] = ((alpha_1 * a[1] * (a[0] * (alpha_0L - alpha_2R) + b[0])
                           * fctr + np.zeros(shape)).reshape(shape_bc_d))
        values = values * v
        Conv = csc_array((values.ravel(), i_c.ravel(), np.arange(
            0, i_c.size + 1)), shape=(math.prod(shape_f_t), math.prod(shape_t)))        
        
        i_f_bc = shape_f_t[1] * shape_f_t[2] * np.arange(shape_f_t[0]).reshape((-1, 1, 1)) + shape_f_t[2] * np.array(
            [0, shape_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(shape_f_t[2]).reshape((1, 1, -1))
        values_bc = np.empty((shape_t[0], 2, shape_t[2]))
        values_bc = np.zeros((shape_t[0], 2, shape_t[2]))
        values_bc[:, 0, :] = ((((a[1] * alpha_0R + b[1]) * d[0] - alpha_2L * a[0] * d[1])
                              * fctr + np.zeros(shape_bc)).reshape(shape_bc_d))
        values_bc[:, 1, :] = ((((a[0] * alpha_0L + b[0]) * d[1] - alpha_2R * a[1] * d[0])
                              * fctr + np.zeros(shape_bc)).reshape(shape_bc_d))
        values_bc = values_bc * v
    else:
        i_c = shape_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1) + shape_t[2] * np.array([0,1,shape_t[1]-2, shape_t[1]-1]).reshape((
            1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        i_f = shape_f_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1) + shape_t[2] * np.array([0,0,shape_f_t[1]-1, shape_f_t[1]-1]).reshape((
            1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        i_f_bc = shape_f_t[1] * shape_f_t[2] * np.arange(shape_f_t[0]).reshape((-1, 1, 1)) + shape_f_t[2] * np.array(
            [0, shape_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(shape_f_t[2]).reshape((1, 1, -1))
        values_bc = np.zeros((shape_t[0], 2, shape_t[2]))
        values = np.zeros((shape_t[0], 4, shape_t[2]))
        if (x_c is None):
            x_c = 0.5*np.array([x_f[0] + x_f[1], x_f[1] + x_f[2], x_f[-3] + x_f[-2], x_f[-2] + x_f[-1]])

        alpha_1 = (x_c[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_2 = (x_c[0] - x_f[0]) / ((x_c[1] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a[0] + b[0])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a[0] * fctr
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d[0] * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        values[:, 0, :] = (a_fctr * alpha_1) * v[:, 0, :]
        values[:, 1, :] = -a_fctr * alpha_2 * v[:, 0, :]
        values_bc[:, 0, :] = d_fctr * v[:, 0, :]

        alpha_1 = -(x_c[-2] - x_f[-1]) / ((x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_2 = -(x_c[-1] - x_f[-1]) / ((x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a[-1] + b[-1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a[-1] * fctr
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d[-1] * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        values[:, -1, :] = (a_fctr * alpha_1) * v[:, -1, :]
        values[:, -2, :] = -a_fctr * alpha_2 * v[:, -1, :]
        values_bc[:, -1, :] = d_fctr * v[:, -1, :]
        Conv = csc_array((values.ravel(), (i_f.ravel(), i_c.ravel())), shape=(
            math.prod(shape_f_t), math.prod(shape_t)))
        Conv.sort_indices()

    conv_bc = csc_array((values_bc.ravel(), i_f_bc.ravel(), [
                         0, i_f_bc.size]), shape=(math.prod(shape_f_t),1))
    return Conv, conv_bc

def construct_coefficient_matrix(coeffs, shape=None, axis=None):
    """
    Construct a diagional matrix with coefficients on its diagonal.
    This is useful to create a matrix containing transport coefficients 
    from a field contained in a ndarray. 

    Args:
        coeffs (ndarray): values of the coefficients in a field
        shape (tuple, optional): Shape of the multidimensional field. With this option, some of the dimensions of coeffs can be choosen singleton.
        axis: In case of broadcasting along 'axis' used shape will be shape[axis+1] (can be useful for face-values)

    Returns:
        csc_array: matrix Coeff with coefficients on its diagonal.
    """
    if(shape == None):
        shape = coeffs.shape
        Coeff = csc_array(diags(coeffs.flatten(), format='csc'))
    else:
        shape = list(shape)
        if (axis != None):
            shape[axis] += 1
        coeffs_copy = np.array(coeffs)
        reps = [shape[i] // coeffs_copy.shape[i] if i < len(coeffs_copy.shape) else shape[i] for i in range(len(shape))]
        coeffs_copy = np.tile(coeffs_copy, reps)
        Coeff = csc_array(diags(coeffs_copy.flatten(), format='csc'))
    return Coeff

def numjac_local(f, c, eps_jac=1e-6, axis=-1):
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
        axis (int, optional): The axis along which components are coupled. Default is -1.

    Returns:
        csc_array: The Jacobian matrix.
        ndarray: The function values.

    """
    shape = c.shape
    if (axis<0):
        axis += len(shape)
    shape_t = [math.prod(shape[0:axis]), shape[axis], math.prod(shape[axis+1:len(shape)])]  # [Nx*Ny*Nz, Nc, ???]
    values = np.zeros((*shape_t, shape[axis]))
    i = shape_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1, 1)) + np.zeros((1, shape_t[1], 1, 1)) + np.arange(
    shape_t[2]).reshape((1, 1, -1, 1)) + shape_t[2] * np.arange(shape_t[1]).reshape((1, 1, 1, -1))
    f_value = f(c,).reshape(shape_t)
    c = c.reshape(shape_t)
    dc = -eps_jac * np.abs(c)  # relative deviation
    dc[dc > (-eps_jac)] = eps_jac  # If dc is small use absolute deviation
    dc = (c + dc) - c
    for k in range(shape_t[1]):
        c_perturb = np.copy(c)
        c_perturb[:, k, :] = c_perturb[:, k, :] + dc[:, k, :]
        f_perturb = f(c_perturb.reshape(shape)).reshape(shape_t)
        values[:, k, :, :] = np.transpose((f_perturb - f_value) / dc[:, [k], :],(0,2,1))
    Jac = csc_array((values.flatten(), i.flatten(), np.arange(
        0, i.size + shape_t[1], shape_t[1])), shape=(np.prod(shape_t), np.prod(shape_t)))
    return f_value.reshape(shape), Jac

def newton(func, x0, args=(), tol=1.49012e-08, maxfev=100, solver=None, callback=None):
    """
    Performs a Newton-Raphson iterations to seek the root of the vector valued function func(x0)

    Parameters
    ---------- 
    func : callable `function func(x, *args)`
        function that provides the vector valued function 'g' of which the roots are sought and, as second argument, its Jacobian
    x0 : `numpy.ndarray`
        vector containing the initial guesses for the values of x
    args : `tuple`, extra arguments
        Extra arguments passed to func
    tol : `float`, optional 
        tolerance used for convergence in the Newton-Raphson iteration, default = 1e-6
    maxfev : `int`, optional
        maximum number of iterations used in Newton-Raphson procedure, default = 100
    solver : `str`, optional 
        the method to solve the linearized equations, default = 'bicgstab'
        options available: 'lu', 'cg', 'bicgstab'. For small systems (say Nc*Nx*Ny*Nz<50000) the 
        direct 'lu' is preferred, while for bigger systems 'bicgstab' should be used (because of 
        the asymmetric Jac matrices that typically arise); Use 'cg' only for symmetric Jac matrices.
    callback: callable, optional
        Optional callback function. It is called on every iteration as callback(x, f) where x is the current solution and f the corresponding residual.
        Note that the callback can be used to change x if it is mutable, e.g. to bound the values.

    Returns
    ------- 
    sol : The solution represented as a OptimizeResult object. 
    Important attributes are: x the solution array, success a Boolean flag indicating if the algorithm exited successfully and
    nit the number of iterations used.

    Example
    -------
    sol = newton(numjac_loc(lambda c: f(c, options)), c)

    created by: M. van Sint Annaland, M. Galanti, E.A.J.F. Peters
    date: April 2024
    """

    n = x0.size
    if (solver is None):
        if (n<50000):
            solver = 'spsolve'
        else:
            solver = 'bicgstab'
            
    if (solver == 'spsolve'):
        linsolver = linalg.spsolve
    elif (solver == 'lu'):
        def linsolver(Jac, g):
            Jac_lu = linalg.splu(Jac)
            dx_neg = Jac_lu.solve(g)
            return dx_neg
    elif (solver == 'cg'):
        def linsolver(Jac, g): 
            Jac_iLU = linalg.spilu(Jac)     # determine pre-conditioner M via ILU factorization
            M = linalg.LinearOperator((n,n), Jac_iLU.solve)
            dx_neg, info = linalg.cg(Jac, g, np.zeros(n), tol=1e-9, maxiter=1000, M=M)
            if info!=0:
                print('solution via cg unsuccessful! info = %d' % info)
            return dx_neg
    elif (solver == 'bicgstab'):
        def linsolver(Jac, g):
            Jac_iLU = linalg.spilu(Jac)     # determine pre-conditioner M via ILU factorization
            M = linalg.LinearOperator((n,n), Jac_iLU.solve)
            dx_neg, info = linalg.bicgstab(Jac, g, np.zeros(n), tol=1e-9, maxiter=10, M=M)
            if info!=0:
                print('solution via bicgstab unsuccessful! info = %d' % info)
            return dx_neg
    else:
        linsolver = None
        print("No valid solver selected.")

    converged = False
    it = 0
    x = x0.copy()
    while (not converged) and (it<maxfev):
        it += 1
        g, Jac = func(x, *args)
        g = g.reshape((-1,1))
        dx_neg = linsolver(Jac, g)
        defect = norm(dx_neg[:], ord=np.inf)
        x -= dx_neg.reshape(x.shape)
        converged = (defect<tol)
        if callback:
            callback(x, g)

    if (~converged):
        message = f"Newton stopped after {it} iterations with max. norm {defect}."
    else:
        message = 'The solution converged'

    result = OptimizeResult({
        'x': x,
        'success': converged,
        'message': message,
        'fun': g.reshape(x0.shape),
        'jac': Jac,
        'nit': it
    })
    return result

def clip_approach(x, f, lower_bounds = 0, upper_bounds = None, factor = 0):
    # filter x with lower and upper bounds using an approach factor
    if (factor == 0):
        np.clip(x, lower_bounds, upper_bounds, out = x)
    else:
        if (lower_bounds != None):
            below_lower = (x < lower_bounds)
            if (np.any(below_lower)):
                broadcasted_lower_bounds = np.broadcast_to(lower_bounds, x.shape)
                x[below_lower] = (1.0 + factor)*broadcasted_lower_bounds[below_lower] - factor*x[below_lower]
        if (upper_bounds != None):
            above_upper = (x > upper_bounds)
            if (np.any(above_upper)):
                broadcasted_upper_bounds = np.broadcast_to(upper_bounds, x.shape)
                x[above_upper] = (1.0 + factor)*broadcasted_upper_bounds[above_upper] - factor*x[above_upper]

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
    shape_f = list(c_f.shape)
    if (axis<0):
        axis += len(shape_f)
    shape_f_t = [math.prod(shape_f[:axis]), shape_f[axis], math.prod(shape_f[axis + 1:])]  # reshape as a triplet
    shape = shape_f.copy()
    shape[axis] = shape[axis] - 1
    c_f = np.reshape(c_f, shape_f_t)
    if (x_c is None):
        c_c =  0.5 * (c_f[:, 1:, :] + c_f[:, :-1, :])
    else:
        wght = (x_c - x_f[:-1]) / (x_f[1:] - x_f[:-1])
        c_c = c_f[:, :-1, :] + wght.reshape((1,-1,1)) * (c_f[:, 1:, :] - c_f[:, :-1, :])
    c_c = c_c.reshape(shape)
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
    shape = list(c_c.shape)
    if (axis<0):
        axis += len(shape)
    shape_t = [math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1:])]  # reshape as a triplet
    shape_f = shape.copy()
    shape_f[axis] = shape[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_f[axis]
    if (x_c is None):
        x_c = 0.5*(x_f[:-1]+x_f[1:])
    wght = (x_f[1:-1] - x_c[:-1]) / (x_c[1:] - x_c[:-1])
    c_c = c_c.reshape(shape_t)
    c_f = np.empty(shape_f_t)
    c_f[:,1:-1,:] = c_c[:, :-1, :] + wght.reshape((1,-1,1)) * (c_c[:, 1:, :] - c_c[:, :-1, :])
    c_f[:,0,:] = (c_c[:,0,:]*(x_c[1]-x_f[0]) - c_c[:,1,:]*(x_c[0]-x_f[0]))/(x_c[1]-x_c[0])
    c_f[:,-1,:] = (c_c[:,-1,:]*(x_f[-1]-x_c[-2]) - c_c[:,-2,:]*(x_f[-1]-x_c[-1]))/(x_c[-1]-x_c[-2])
    c_f = c_f.reshape(shape_f)
    return c_f

def interp_cntr_to_stagg_tvd(c_c, x_f, x_c=None, bc=None, v=0, tvd_limiter = None, axis=0):
    """
    Interpolate values at staggered positions to cell-centers using TVD interpolation.

    Args:
        c_c (ndarray): Quantities at cell-centered positions.
        x_f (ndarray): Positions of cell-faces (numel(x_f) = numel(x_c) + 1).
        x_c (ndarray, optional): Cell-centered positions. If not provided, the interpolated values will be averaged between adjacent staggered positions.
        bc  (list, optional): The boundary conditions used to extrapolate to the boundary faces
        v   (ndarray, optional): Velocites on face positions
        tvd_limiter (function, optional): The TVD limiter 
        axis (int, optional): Dimension along which interpolation is performed. Default is 0.

    Returns:
        ndarray: Interpolated concentrations at staggered positions.

    """
    shape = list(c_c.shape)
    if (axis<0):
        axis += len(shape)
    shape_t = [math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1:])]  # reshape as a triplet
    shape_f = shape.copy()
    shape_f[axis] = shape[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_f[axis]
    shape_bc = shape_f.copy()
    shape_bc[axis] = 1
    shape_bc_d = [shape_t[0], shape_t[2]]
    
    if (x_c is None):
        x_c = 0.5*(x_f[:-1]+x_f[1:])
    c_c = c_c.reshape(shape_t)
    c_f = np.empty(shape_f_t)
    
    a, b, d = unwrap_bc(shape, bc)
       
    if (shape_t[1] == 1):
        alpha_1 = (x_f[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_2L = (x_c[0] - x_f[0]) / ((x_f[1] - x_f[0]) * (x_f[1] - x_c[0]))
        alpha_0L = alpha_1 - alpha_2L
        alpha_2R = -(x_c[0] - x_f[1]) / ((x_f[0] - x_f[1]) * (x_f[0] - x_c[0]))
        alpha_0R = alpha_1 - alpha_2R        
        fctr = ((b[0] + alpha_0L * a[0]) * (b[1] +
                     alpha_0R * a[1]) - alpha_2L * alpha_2R * a[0] * a[1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        fctr_m = (alpha_1 * a[0] * (a[1] * (alpha_0R - alpha_2L) + b[1])
                           * fctr) 
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)
        c_f[:,0,:] = fctr_m*c_c;
        fctr_m = (alpha_1 * a[1] * (a[0] * (alpha_0L - alpha_2R) + b[0])
                           * fctr)
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)      
        c_f[:, 1, :] = fctr_m*c_c
        fctr_m = ((a[1] * alpha_0R + b[1]) * d[0] - alpha_2L * a[0] * d[1])* fctr
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)     
        c_f[:, 0, :] += fctr_m
        fctr_m = ((a[0] * alpha_0L + b[0]) * d[1] - alpha_2R * a[1] * d[0])* fctr
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d) 
        c_f[:, 0, :] += fctr_m
        c_f.reshape(shape_f)
        dc_f = np.zeros(shape_f)
    else:
        # bc 0
        alpha_1 = (x_c[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_2 = (x_c[0] - x_f[0]) / ((x_c[1] - x_f[0]) * (x_c[1] - x_c[0]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a[0] + b[0])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a[0] * fctr
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d[0] * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        c_f[:,0,:] = (d_fctr + a_fctr*(alpha_1*c_c[:,0,:] - alpha_2*c_c[:,1,:]))
        # bc 1
        alpha_1 = -(x_c[-2] - x_f[-1]) / ((x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_2 = -(x_c[-1] - x_f[-1]) / ((x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a[-1] + b[-1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a[-1] * fctr
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d[-1] * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        c_f[:,-1,:] = (d_fctr + a_fctr*(alpha_1*c_c[:,-1,:] - alpha_2*c_c[:,-2,:]))
        
        v = np.array(v) + np.zeros(shape_f)
        v = v.reshape(shape_f_t)
        fltr_v_pos = (v > 0)
        
        x_f = x_f.reshape((1,-1,1))
        x_c = x_c.reshape((1,-1,1))
        x_d = x_f[:,1:-1,:]
        x_C = fltr_v_pos[:,1:-1,:]*x_c[:,:-1,:] + ~fltr_v_pos[:,1:-1,:]*x_c[:,1:,:]
        x_U = fltr_v_pos[:,1:-1,:]*np.concatenate((x_f[:,0:1,:],x_c[:,0:-2,:]),axis=1) + ~fltr_v_pos[:,1:-1,:]*np.concatenate((x_c[:,2:,:], x_f[:,-1:,:]),axis=1)
        x_D = fltr_v_pos[:,1:-1,:]*x_c[:,1:,:] + ~fltr_v_pos[:,1:-1,:]*x_c[:,:-1,:]
        x_norm_C = (x_C-x_U)/(x_D-x_U)
        x_norm_d = (x_d-x_U)/(x_D-x_U)
        c_C = fltr_v_pos[:,1:-1,:]*c_c[:,:-1,:] + ~fltr_v_pos[:,1:-1,:]*c_c[:,1:,:]
        c_U = fltr_v_pos[:,1:-1,:]*np.concatenate((c_f[:,0:1,:],c_c[:,0:-2,:]),axis=1) + ~fltr_v_pos[:,1:-1,:]*np.concatenate((c_c[:,2:,:], c_f[:,-1:,:]),axis=1)
        c_D = fltr_v_pos[:,1:-1,:]*c_c[:,1:,:] + ~fltr_v_pos[:,1:-1,:]*c_c[:,:-1,:];
        c_norm_C = np.zeros_like(c_C);
        dc_DU = (c_D-c_U);
        np.divide((c_C-c_U), dc_DU, out=c_norm_C, where=(dc_DU != 0))
        c_f = np.concatenate((c_f[:,0:1,:], c_C, c_f[:,-1:,:]),axis=1)
        if (tvd_limiter == None):
            dc_f = np.zeros(shape_f)    
            c_f = c_f.reshape(shape_f)
        else:
            dc_f = np.zeros(shape_f_t)
            dc_f[:,1:-1,:] = tvd_limiter(c_norm_C, x_norm_C, x_norm_d) * dc_DU
            c_f += dc_f
            dc_f = dc_f.reshape(shape_f)
            c_f = c_f.reshape(shape_f)
        return c_f, dc_f

def upwind(c_norm_C, x_norm_C, x_norm_d):
    """
    Apply the upwind TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        c_norm_C (ndarray): Normalized concentration at cell centers.
        x_norm_C (ndarray): Normalized position of cell centers.
        x_norm_d (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C. (Equals 0 for upwind)
    """
    c_norm_diff = np.zeros_like(c_norm_C)
    return c_norm_diff
               
def minmod(c_norm_C, x_norm_C, x_norm_d):
    """
    Apply the Min-mod TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        c_norm_C (ndarray): Normalized concentration at cell centers.
        x_norm_C (ndarray): Normalized position of cell centers.
        x_norm_d (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    c_norm_diff = np.maximum(0,(x_norm_d-x_norm_C)*np.minimum(c_norm_C/x_norm_C, (1-c_norm_C)/(1-x_norm_C)));
    return c_norm_diff

def osher(c_norm_C, x_norm_C, x_norm_d):
    """
    Apply the Osher TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        c_norm_C (ndarray): Normalized concentration at cell centers.
        x_norm_C (ndarray): Normalized position of cell centers.
        x_norm_d (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    c_norm_diff = np.maximum(0,np.where(c_norm_C<x_norm_C/x_norm_d, (x_norm_d/x_norm_C - 1)*c_norm_C, 1 - c_norm_C))
    return c_norm_diff

def clam(c_norm_C, x_norm_C, x_norm_d):
    """
    Apply the CLAM TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        c_norm_C (ndarray): Normalized concentration at cell centers.
        x_norm_C (ndarray): Normalized position of cell centers.
        x_norm_d (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    c_norm_diff = np.maximum(0,np.where(c_norm_C<x_norm_C/x_norm_d, (x_norm_d/x_norm_C - 1)*c_norm_C, 1 - c_norm_C))
    return c_norm_diff

def muscl(c_norm_C, x_norm_C, x_norm_d):
    """
    Apply the MUSCL limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        c_norm_C (ndarray): Normalized concentration at cell centers.
        x_norm_C (ndarray): Normalized position of cell centers.
        x_norm_d (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    c_norm_diff = np.maximum(0,np.where(c_norm_C<x_norm_C/(2*x_norm_d), ((2*x_norm_d - x_norm_C)/x_norm_C - 1)*c_norm_C, 
                             np.where(c_norm_C<1 + x_norm_C - x_norm_d, x_norm_d - x_norm_C, 1 - c_norm_C)))
    return c_norm_diff

def smart(c_norm_C, x_norm_C, x_norm_d):
    """
    Apply the SMART TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        c_norm_C (ndarray): Normalized concentration at cell centers.
        x_norm_C (ndarray): Normalized position of cell centers.
        x_norm_d (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    c_norm_diff = np.maximum(0,np.where(c_norm_C<x_norm_C/3, (x_norm_d*(1 - 3*x_norm_C + 2*x_norm_d)/(x_norm_C*(1 - x_norm_C)) - 1)*c_norm_C, 
                             np.where(c_norm_C<x_norm_C/x_norm_d*(1 + x_norm_d - x_norm_C), (x_norm_d*(x_norm_d - x_norm_C) + x_norm_d*(1 - x_norm_d)/x_norm_C*c_norm_C)/(1 - x_norm_C) - c_norm_C, 1 - c_norm_C)))
    return c_norm_diff

def stoic(c_norm_C, x_norm_C, x_norm_d):
    """
    Apply the STOIC TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        c_norm_C (ndarray): Normalized concentration at cell centers.
        x_norm_C (ndarray): Normalized position of cell centers.
        x_norm_d (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    c_norm_diff = np.maximum(0,np.where(c_norm_C<x_norm_C*(x_norm_d - x_norm_C)/(x_norm_C + x_norm_d + 2*x_norm_d*x_norm_d - 4*x_norm_d*x_norm_C), x_norm_d*(1 - 3*x_norm_C + 2*x_norm_d)/(x_norm_C*(1 - x_norm_C)) - c_norm_C, 
                             np.where(c_norm_C<x_norm_C, (x_norm_d - x_norm_C + (1 - x_norm_d)*c_norm_C)/(1 - x_norm_C) - c_norm_C,
                             np.where(c_norm_C<x_norm_C/x_norm_d*(1 + x_norm_d - x_norm_C), (x_norm_d*(x_norm_d - x_norm_C) + x_norm_d*(1 - x_norm_d)/x_norm_C*c_norm_C)/(1 - x_norm_C) - c_norm_C, 1 - c_norm_C))))
    return c_norm_diff

def vanleer(c_norm_C, x_norm_C, x_norm_d):
    """
    Apply the van-Leer TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        c_norm_C (ndarray): Normalized concentration at cell centers.
        x_norm_C (ndarray): Normalized position of cell centers.
        x_norm_d (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    c_norm_diff = np.maximum(0, c_norm_C*(1-c_norm_C)*(x_norm_d-x_norm_C)/(x_norm_C*(1-x_norm_C)))   
    return c_norm_diff

def non_uniform_grid(x_L, x_R, n, dx_inf, factor):
    """
    Generate a non-uniform grid of points in the interval [x_L, x_R]
    With factor > 1 the refinement will be at the left wall, 
    with 1/factor you will get the same refinement at the right wall.
    
    Parameters:
        x_L (float): Start point of the interval.
        x_R (float): End point of the interval.
        n (int): Total number of face positions (including x_L and x_R)
        dx_inf (float): Limiting upper-bound grid spacing
        factor (float): Factor used to increase grid spacing

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

def unwrap_bc(shape, bc):
    """
    Unwrap the boundary conditions for a given shape.

    Args:
        shape (tuple): shape of the domain.
        bc (dict): Boundary conditions.

    Returns:
        tuple: Unwrapped boundary conditions (a, b, d).
    """
    if not isinstance(shape, (list,tuple)):
        lgth_shape = 1
    else:
        lgth_shape = len(shape)
      
    a = [None, None]
    b = [None, None]
    d = [None, None]
    
    if (bc is None):
        a[0] = np.zeros((1,) * lgth_shape)
        a[1] = np.zeros((1,) * lgth_shape)
        b[0] = np.zeros((1,) * lgth_shape)
        b[1] = np.zeros((1,) * lgth_shape)
        d[0] = np.zeros((1,) * lgth_shape)
        d[1] = np.zeros((1,) * lgth_shape)
    else:
        a[0] = np.array(bc['a'][0])
        a[0] = a[0][(..., *([np.newaxis]*(lgth_shape-a[0].ndim)))]
        a[1] = np.array(bc['a'][1])
        a[1] = a[1][(..., *([np.newaxis]*(lgth_shape-a[1].ndim)))]
        b[0] = np.array(bc['b'][0])
        b[0] = b[0][(..., *([np.newaxis]*(lgth_shape-b[0].ndim)))]
        b[1] = np.array(bc['b'][1])
        b[1] = b[1][(..., *([np.newaxis]*(lgth_shape-b[1].ndim)))]
        d[0] = np.array(bc['d'][0])
        d[0] = d[0][(..., *([np.newaxis]*(lgth_shape-d[0].ndim)))]
        d[1] = np.array(bc['d'][1])
        d[1] = d[1][(..., *([np.newaxis]*(lgth_shape-d[1].ndim)))]
    return a, b, d
