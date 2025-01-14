"""
Module Name: pymrm
Author: E.A.J.F. Peters, M. Sint Annaland, M. Galanti, D. Rieder
Date: 14/01/2025
License: MIT License
Version: 2.0.0

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

import math
import numpy as np
from scipy.sparse import csc_array, diags, linalg, block_diag
from scipy.linalg import norm
from scipy.optimize import OptimizeResult

def construct_grad(shape, face_positions, cell_centers=None, boundary_conditions=(None, None), axis=0):
    """
    Construct the gradient matrix.

    Args:
        shape (tuple): shape of the domain.
        face_positions (ndarray): Face positions
        cell_centers (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        boundary_conditions (dict, optional): Boundary conditions. Default is None.
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
    face_positions, cell_centers = generate_grid(shape[axis], face_positions, cell_centers, generate_x_c = True)
    
    grad_matrix = construct_grad_int(shape, face_positions, cell_centers, axis)
    
    if (boundary_conditions is None):
        shape_f = shape.copy()
        if (axis<0):
            axis += len(shape)       
        shape_f[axis] +=1
        grad_bc = csc_array(shape=(math.prod(shape_f),1))
    else:
        grad_matrix_bc, grad_bc = construct_grad_bc(shape, face_positions, cell_centers, boundary_conditions, axis)
        grad_matrix += grad_matrix_bc
    return grad_matrix, grad_bc

def construct_grad_int(shape, face_positions,  cell_centers=None, axis=0):
    """
    Construct the gradient matrix for internal faces.

    Args:
        shape (tuple): shape of the domain.
        face_positions (ndarray): Face coordinates.
        cell_centers (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.
    
    Returns:
        csc_array: Gradient matrix (grad_matrix).
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
    
    if (axis<0):
        axis += len(shape)    
    shape_t = [math.prod(shape[0:axis]), math.prod(
        shape[axis:axis+1]), math.prod(shape[axis+1:])] # shape of the reshaped 3 dimenional array   
    # The gradient will be represented by a csc array.
    # Each column corresponds to the contribution of a value cell
    # For each column there are two entries corresponding to the two faces, except for the cells near the boundary
    # The indices of these faces are stored in an array i_f, with i_f.shape = [shape_t[0], shape_t[1]+1 , shape_t[2], 2]
    # Note the shape_t[1]+1, because there is 1 more face than cells.
    # The linear index is: i_f[i,j,k,m] = 2*(shape_t[2]*((shape_t[1]+1)*j + k)) + m
    i_f = (shape_t[1]+1) * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1, 1) + shape_t[2] * np.arange(shape_t[1]).reshape((
        1, -1, 1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1, 1)) + np.array([0, shape_t[2]]).reshape((1, 1, 1, -1))
    
    if (cell_centers is None):
        cell_centers = 0.5*(face_positions[:-1] + face_positions[1:])
    
    dx_inv = np.tile(
        1 / (cell_centers[1:] - cell_centers[:-1]).reshape((1, -1, 1)), (shape_t[0], 1, shape_t[2]))
    values = np.empty(i_f.shape)
    values[:,0,:,0] = np.zeros((shape_t[0],shape_t[2]))
    values[:, 1:, :, 0] = dx_inv
    values[:, :-1, :, 1] = -dx_inv
    values[:,-1,:,1] = np.zeros((shape_t[0],shape_t[2]))
    grad_matrix = csc_array((values.ravel(), i_f.ravel(), range(0,i_f.size + 1,2)), 
                      shape=(shape_t[0]*(shape_t[1]+1)*shape_t[2], shape_t[0]*shape_t[1]*shape_t[2]))
    return grad_matrix

def construct_grad_bc(shape, face_positions, cell_centers=None, boundary_conditions=(None, None), axis=0):
    """
    Construct the gradient matrix for the boundary faces 

    Args:
        shape (tuple): shape of the domain.
        face_positions (ndarray): Face coordinates.
        cell_centers (ndarray, optional): Cell center coordinates. If not provided, it will be calculated as the average of neighboring face coordinates.
        boundary_conditions (dict, optional): Boundary conditions. Default is None.
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.

    Returns:
        csc_array: Gradient matrix (grad_matrix).
        csc_array: Contribution of the inhomogeneous BC to the gradient (grad_bc).
    """
    # For the explanation of resizing see construct_grad_int
    if (axis<0):
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
    if (shape_t[1] == 1):
        a, b, d = [None]*2, [None]*2, [None]*2
        a[0], b[0], d[0] = unwrap_bc(shape, boundary_conditions[0]) # Get a, b, and d for left bc from dictionary
        a[1], b[1], d[1] = unwrap_bc(shape, boundary_conditions[1]) # Get a, b, and d for right bc from dictionary
        if (cell_centers is None):
            cell_centers = 0.5*(face_positions[0:-1] + face_positions[1:])
        i_c = shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1)) + np.array(
            (0, 0)).reshape((1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        i_f = shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1)) + shape_t[2] * np.array([0,1]).reshape((
            1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        values = np.zeros(shape_f_t)
        alpha_1 = (face_positions[1] - face_positions[0]) / ((cell_centers[0] - face_positions[0]) * (face_positions[1] - cell_centers[0]))
        alpha_2L = (cell_centers[0] - face_positions[0]) / ((face_positions[1] - face_positions[0]) * (face_positions[1] - cell_centers[0]))
        alpha_0L = alpha_1 - alpha_2L
        alpha_2R = -(cell_centers[0] - face_positions[1]) / ((face_positions[0] - face_positions[1]) * (face_positions[0] - cell_centers[0]))
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
        if (cell_centers is None):
            cell_centers = 0.5*np.array([face_positions[0] + face_positions[1], face_positions[1] + face_positions[2], face_positions[-3] + face_positions[-2], face_positions[-2] + face_positions[-1]])
        dx_inv = np.tile(
            1 / (cell_centers[1:] - cell_centers[:-1]).reshape((1, -1, 1)), (shape_t[0], 1, shape_t[2]))

        a, b, d = unwrap_bc(shape, boundary_conditions[0]) # Get a, b, and d for left bc from dictionary
        alpha_1 = (cell_centers[1] - face_positions[0]) / ((cell_centers[0] - face_positions[0]) * (cell_centers[1] - cell_centers[0]))
        alpha_2 = (cell_centers[0] - face_positions[0]) / ((cell_centers[1] - face_positions[0]) * (cell_centers[1] - cell_centers[0]))
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

        a, b, d = unwrap_bc(shape, boundary_conditions[1]) # Get a, b, and d for right bc from dictionary
        alpha_1 = -(cell_centers[-2] - face_positions[-1]) / ((cell_centers[-1] - face_positions[-1]) * (cell_centers[-2] - cell_centers[-1]))
        alpha_2 = -(cell_centers[-1] - face_positions[-1]) / ((cell_centers[-2] - face_positions[-1]) * (cell_centers[-2] - cell_centers[-1]))
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
                         0, i_f_bc.size]), shape=(math.prod(shape_f_t),1))
    return grad_matrix, grad_bc

def construct_diff_flux(shape, grad_matrix, grad_bc, parameters, axis=0):
    """
    Computes the diffusion fluxes in sparse format
    
    Parameters
    ----------  
        axis (int, optional): Dimension to construct the gradient matrix for. Default is 0.
        shape:  shape (tuple): shape of the domain.
        grad_matrix: sparse matrix containing the homogeneous terms of the discretized gradient operator
        grad_bc: sparse vector containing the heterogeneous terms resulting from the boundary conditions
        parameters:  a dictionary that should contain the key 'Diff' where the diffusion coefficients
                are stored, that can be a constant, a diagonal vector of length Nc, or an Nc x Nc matrix
    
    Returns
    -------
        diff_flux = A_diff*phi + b_diff
        A_diff: sparse matrix for the homogeneous part of the diffusion fluxes
        b_diff: sparse column vector for the inhomogeneous contributions resulting from the boundary conditions
    
    Example
    -------
        The command to obtain the diffusion fluxes in the x-direction is:
        A_diff_x, b_diff_x = diff_flux(shape, sz, grad_matrix, grad_bc, axis=0, **parameters)
        
    created by: M. van Sint Annaland
    modified by: M. Galanti
    date: January 2024
    """  
    if isinstance(shape, int):
        shape = (shape, )
    else:
        shape = tuple(shape) 
    sf = np.asarray(shape)
    sf[axis] += 1
    nc = shape[-1]
    ntotf = np.prod(sf[0:-1])

    if isinstance(parameters['Diff'], float):
        A_diff = -parameters['Diff']*grad_matrix
        b_diff = -parameters['Diff']*grad_bc
    elif isinstance(parameters['Diff'][axis-1], float):
        Diff = np.asarray(parameters['Diff'])*np.eye(nc)
        Diff_mat_x = block_diag([Diff]*ntotf, format='csc')
        A_diff = -Diff_mat_x @ grad_matrix
        b_diff = -Diff_mat_x @ grad_bc
    else:
        Diff_mat_x = block_diag([parameters['Diff']]*ntotf, format='csc')
        A_diff = -Diff_mat_x @ grad_matrix
        b_diff = -Diff_mat_x @ grad_bc

    return A_diff, b_diff

def construct_div(shape, face_positions, nu=0, axis=0):
    """
    Construct the div_mat matrix based on the given parameters.

    Args:
        shape (tuple): shape of the multi-dimesional array
        face_positions (ndarray): Face positions
        nu (int or callable): The integer representing geometry (0: flat, 1: cylindrical, 2: spherical). If it is a function it specifies an area at position x.
        axis (int): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The div_mat matrix.

    """

    # Trick: Reshape to triplet shape_t, see compute_grad_int for explanation
    if isinstance(shape, int):
        shape_f = [shape]
        shape = (shape, )
    else:
        shape_f = list(shape)
        shape = tuple(shape)
    face_positions = generate_grid(shape[axis], face_positions)
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
        area = nu(face_positions).ravel()
        inv_sqrt3 = 1 / np.sqrt(3)
        face_positions_r = face_positions.ravel()
        dx_f = face_positions_r[1:] - face_positions_r[:-1]
        dvol_inv = 1 / (
            (nu(face_positions_r[:-1] + (0.5 - 0.5 * inv_sqrt3) * dx_f)
             + nu(face_positions_r[:-1] + (0.5 + 0.5 * inv_sqrt3) * dx_f))
            * 0.5 * dx_f
        )
    elif nu == 0:
        area = np.ones(shape_f_t[1])
        dvol_inv = 1 / (face_positions[1:] - face_positions[:-1])
    else:
        area = np.power(face_positions.ravel(), nu)
        vol = area * face_positions.ravel() / (nu + 1)
        dvol_inv = 1 / (vol[1:] - vol[:-1])

    values = np.empty((shape_t[1],2))
    values[:, 0] = -area[:-1] * dvol_inv
    values[:, 1] =  area[1:] * dvol_inv
    values = np.tile(values.reshape((1,-1,1,2)),(shape_t[0],1,shape_t[2]))

    num_cells = np.prod(shape_t, dtype=int);
    div_matrix = csc_array(
        (values.ravel(),(np.repeat(np.arange(num_cells),2) , 
                           i_f.ravel())),
        shape=(num_cells, np.prod(shape_f_t, dtype=int))
    )
    div_matrix.sort_indices()
    return div_matrix

def construct_convflux_upwind(shape, face_positions, cell_centers=None, boundary_conditions=(None,None), velocity=1.0, axis=0):
    """
    Construct the conv_matrix and conv_bc matrices based on the given parameters.

    Args:
        shape (tuple): shape of the multi-dimensional array.
        face_positions (ndarray): Face positions.
        cell_centers (ndarray, optional): Cell positions. If not provided, it will be calculated based on the face array.
        boundary_conditions (list, optional): The boundary conditions. Default is None.
        velocity (ndarray): Velocities on face positions
        axis (int, optional): The axis along which the convection takes place is performed. Default is 0.

    Returns:
        csc_array: The conv_matrix matrix.
        csc_array: The conv_bc matrix.
    """
    if (isinstance(shape, int)):
        shape = (shape,)
    face_positions, cell_centers = generate_grid(shape[axis], face_positions, cell_centers, generate_x_c = True)
    
    conv_matrix = construct_convflux_upwind_int(shape, velocity, axis)
    if (boundary_conditions == None or boundary_conditions == (None,None)):
        shape_f = shape.copy()
        shape_f[axis] +=1
        conv_bc = csc_array(shape=(math.prod(shape_f),1))
    else:
        conv_matrix_bc, conv_bc = construct_convflux_upwind_bc(shape, face_positions, cell_centers, boundary_conditions, velocity, axis)
        conv_matrix += conv_matrix_bc
    return conv_matrix, conv_bc

def construct_convflux_upwind_int(shape, velocity=1.0, axis=0):
    """
    Construct the conv_matrix matrix for internal faces, e.g. all faces except these on the boundaries

    Args:
        shape (tuple): shape of the ndarrays.
        velocity (ndarray): The velocity array.
        axis (int, optional): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The conv_matrix matrix.

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

    velocity = np.array(velocity) + np.zeros(shape_f)
    velocity = velocity.reshape(shape_f_t)
    fltr_v_pos = (velocity > 0)
    i_f = (shape_t[1]+1) * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1) + shape_t[2] * np.arange(1,shape_t[1]).reshape((
        1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
    i_c = shape_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1)) + shape_t[2] * np.arange(1,shape_t[1]).reshape((
        1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
    i_c = i_c - shape_t[2] * fltr_v_pos[:, 1:-1, :]
    conv_matrix = csc_array((velocity[:, 1:-1, :].ravel(), (i_f.ravel(), i_c.ravel())), shape=(
        math.prod(shape_f_t), math.prod(shape_t)))
    conv_matrix.sort_indices()
    return conv_matrix
    
def construct_convflux_upwind_bc(shape, face_positions, cell_centers = None, boundary_conditions=(None,None), velocity=1.0, axis=0):
    """
    Construct the conv_matrix and conv_bc matrices for the boundary faces only

    Args:
        shape (tuple): shape of the multidimensional array.
        cell_centers (ndarray, optional): Cell-centered positions. If not provided, it will be calculated based on the face array.
        face_positions (ndarray): Face positions
        boundary_conditions (tuple, optional): A tuple containting the left and right boundary conditions. Default is None.
        velocity (ndarray): The velocity array.
        axis (int, optional): The axis along which the numerical differentiation is performed. Default is 0.

    Returns:
        csc_array: The conv_matrix matrix.
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

    velocity = np.array(velocity) + np.zeros(shape_f)
    velocity = velocity.reshape(shape_f_t)
    fltr_v_pos = (velocity > 0)

    # Handle special case with one cell in the dimension axis
    if (shape_t[1] == 1):
        a, b, d = [None]*2, [None]*2, [None]*2
        a[0], b[0], d[0] = unwrap_bc(shape, boundary_conditions[0])
        a[1], b[1], d[1] = unwrap_bc(shape, boundary_conditions[1])
        if (cell_centers is None):
            cell_centers = 0.5*(face_positions[0:-1] + face_positions[1:])
        i_c = shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1)) + np.array(
            (0, 0)).reshape((1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        i_f = shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1)) + shape_t[2] * np.array([0,1]).reshape((
            1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        values = np.zeros(shape_f_t)
        alpha_1 = (face_positions[1] - face_positions[0]) / ((cell_centers[0] - face_positions[0]) * (face_positions[1] - cell_centers[0]))
        alpha_2L = (cell_centers[0] - face_positions[0]) / ((face_positions[1] - face_positions[0]) * (face_positions[1] - cell_centers[0]))
        alpha_0L = alpha_1 - alpha_2L
        alpha_2R = -(cell_centers[0] - face_positions[1]) / ((face_positions[0] - face_positions[1]) * (face_positions[0] - cell_centers[0]))
        alpha_0R = alpha_1 - alpha_2R        
        fctr = ((b[0] + alpha_0L * a[0]) * (b[1] +
                     alpha_0R * a[1]) - alpha_2L * alpha_2R * a[0] * a[1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        values = np.empty((shape_t[0],2,shape_t[2]))
        values[:, 0, :] = ((alpha_1 * a[0] * (a[1] * (alpha_0R - alpha_2L) + b[1])
                           * fctr + np.zeros(shape)).reshape(shape_bc_d))
        values[:, 1, :] = ((alpha_1 * a[1] * (a[0] * (alpha_0L - alpha_2R) + b[0])
                           * fctr + np.zeros(shape)).reshape(shape_bc_d))
        values = values * velocity
        conv_matrix = csc_array((values.ravel(), (i_f.ravel(), i_c.ravel())), shape=(math.prod(shape_f_t), math.prod(shape_t)))        
        
        i_f_bc = shape_f_t[1] * shape_f_t[2] * np.arange(shape_f_t[0]).reshape((-1, 1, 1)) + shape_f_t[2] * np.array(
            [0, shape_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(shape_f_t[2]).reshape((1, 1, -1))
        values_bc = np.empty((shape_t[0], 2, shape_t[2]))
        values_bc = np.zeros((shape_t[0], 2, shape_t[2]))
        values_bc[:, 0, :] = ((((a[1] * alpha_0R + b[1]) * d[0] - alpha_2L * a[0] * d[1])
                              * fctr + np.zeros(shape_bc)).reshape(shape_bc_d))
        values_bc[:, 1, :] = ((((a[0] * alpha_0L + b[0]) * d[1] - alpha_2R * a[1] * d[0])
                              * fctr + np.zeros(shape_bc)).reshape(shape_bc_d))
        values_bc = values_bc * velocity
    else:
        i_c = shape_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1) + shape_t[2] * np.array([0,1,shape_t[1]-2, shape_t[1]-1]).reshape((
            1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        i_f = shape_f_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1) + shape_t[2] * np.array([0,0,shape_f_t[1]-1, shape_f_t[1]-1]).reshape((
            1, -1, 1)) + np.arange(shape_t[2]).reshape((1, 1, -1))
        i_f_bc = shape_f_t[1] * shape_f_t[2] * np.arange(shape_f_t[0]).reshape((-1, 1, 1)) + shape_f_t[2] * np.array(
            [0, shape_f_t[1]-1]).reshape((1, -1, 1)) + np.arange(shape_f_t[2]).reshape((1, 1, -1))
        values_bc = np.zeros((shape_t[0], 2, shape_t[2]))
        values = np.zeros((shape_t[0], 4, shape_t[2]))
        if (cell_centers is None):
            cell_centers = 0.5*np.array([face_positions[0] + face_positions[1], face_positions[1] + face_positions[2], face_positions[-3] + face_positions[-2], face_positions[-2] + face_positions[-1]])

        a, b, d = unwrap_bc(shape, boundary_conditions[0]) # Get a, b, and d for left bc from dictionary
        alpha_1 = (cell_centers[1] - face_positions[0]) / ((cell_centers[0] - face_positions[0]) * (cell_centers[1] - cell_centers[0]))
        alpha_2 = (cell_centers[0] - face_positions[0]) / ((cell_centers[1] - face_positions[0]) * (cell_centers[1] - cell_centers[0]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a + b)
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a * fctr
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        values[:, 0, :] = (a_fctr * alpha_1) * velocity[:, 0, :]
        values[:, 1, :] = -a_fctr * alpha_2 * velocity[:, 0, :]
        values_bc[:, 0, :] = d_fctr * velocity[:, 0, :]

        a, b, d = unwrap_bc(shape, boundary_conditions[1]) # Get a, b, and d for right bc from dictionary
        alpha_1 = -(cell_centers[-2] - face_positions[-1]) / ((cell_centers[-1] - face_positions[-1]) * (cell_centers[-2] - cell_centers[-1]))
        alpha_2 = -(cell_centers[-1] - face_positions[-1]) / ((cell_centers[-2] - face_positions[-1]) * (cell_centers[-2] - cell_centers[-1]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a + b)
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a * fctr
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        values[:, -1, :] = (a_fctr * alpha_1) * velocity[:, -1, :]
        values[:, -2, :] = -a_fctr * alpha_2 * velocity[:, -1, :]
        values_bc[:, -1, :] = d_fctr * velocity[:, -1, :]
        conv_matrix = csc_array((values.ravel(), (i_f.ravel(), i_c.ravel())), shape=(
            math.prod(shape_f_t), math.prod(shape_t)))
        conv_matrix.sort_indices()

    conv_bc = csc_array((values_bc.ravel(), i_f_bc.ravel(), [
                         0, i_f_bc.size]), shape=(math.prod(shape_f_t),1))
    return conv_matrix, conv_bc

def construct_coefficient_matrix(coefficients, shape=None, axis=None):
    """
    Construct a diagional matrix with coefficients on its diagonal.
    This is useful to create a matrix containing transport coefficients 
    from a field contained in a ndarray. 

    Args:
        coefficients (ndarray, list): values of the coefficients in a field
        shape (tuple, optional): Shape of the multidimensional field. With this option, some of the dimensions of coefficients can be choosen singleton.
        axis (int, optional): In case of broadcasting along 'axis' used shape will be shape[axis+1] (can be useful for face-values)

    Returns:
        csc_array: matrix Coeff with coefficients on its diagonal.
    """
    if(shape == None):
        shape = coefficients.shape
        Coeff = csc_array(diags(coefficients.flatten(), format='csc'))
    else:
        shape = list(shape)
        if (axis != None):
            shape[axis] += 1
        coefficients_copy = np.array(coefficients)
        reps = [shape[i] // coefficients_copy.shape[i] if i < len(coefficients_copy.shape) else shape[i] for i in range(len(shape))]
        coefficients_copy = np.tile(coefficients_copy, reps)
        Coeff = csc_array(diags(coefficients_copy.flatten(), format='csc'))
    return Coeff

def numjac_local(function, initial_values, epsilon_jac=1e-6, axis=-1):
    """
    Compute the local numerical Jacobian matrix and function values for the given function and initial values.
    
    The function 'function' is assumed to be local, meaning it can be dependent on other components in the array along the 'axis' dimension.
    numjac_local can be used to compute Jacobians of functions like reaction, accumulation, or mass transfer terms, where there 
    is a dependence only on local components in a spatial cell.
    The best choice is to set up the problem such that 'axis' is the last dimension of the multidimensional array, 
    as this will result in a nicely block-structured Jacobian matrix.

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
    axis = [a + len(shape) if a < 0 else a for a in axis]  # Normalize negative indices
    # Calculate the shape tuple for the reshaping operation
    middle_dim = math.prod([shape[a] for a in axis])
    shape_t = [math.prod(shape[:min(axis)]), middle_dim, math.prod(shape[max(axis)+1:])]

    values = np.zeros((*shape_t, shape_t[1]))
    i = shape_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1, 1)) + np.zeros((1, shape_t[1], 1, 1)) + np.arange(
    shape_t[2]).reshape((1, 1, -1, 1)) + shape_t[2] * np.arange(shape_t[1]).reshape((1, 1, 1, -1))
    function_value = function(initial_values,).reshape(shape_t)
    initial_values = initial_values.reshape(shape_t)
    dc = -epsilon_jac * np.abs(initial_values)  # relative deviation
    dc[dc > (-epsilon_jac)] = epsilon_jac  # If dc is small use absolute deviation
    dc = (initial_values + dc) - initial_values
    for k in range(shape_t[1]):
        initial_values_perturb = np.copy(initial_values)
        initial_values_perturb[:, k, :] = initial_values_perturb[:, k, :] + dc[:, k, :]
        function_perturb = function(initial_values_perturb.reshape(shape)).reshape(shape_t)
        values[:, k, :, :] = np.transpose((function_perturb - function_value) / dc[:, [k], :],(0,2,1))
    Jac = csc_array((values.flatten(), i.flatten(), np.arange(
        0, i.size + shape_t[1], shape_t[1])), shape=(np.prod(shape_t), np.prod(shape_t)))
    return function_value.reshape(shape), Jac

def newton(function, initial_guess, args=(), tolerance=1.49012e-08, max_iterations=100, solver=None, callback=None):
    """
    Performs a Newton-Raphson iterations to seek the root of the vector valued function function(initial_guess)

    Parameters
    ---------- 
    function : callable `function function(x, *args)`
        function that provides the vector valued function 'g' of which the roots are sought and, as second argument, its Jacobian
    initial_guess : `numpy.ndarray`
        vector containing the initial guesses for the values of x
    args : `tuple`, extra arguments
        Extra arguments passed to function
    tolerance : `float`, optional 
        tolerance used for convergence in the Newton-Raphson iteration, default = 1e-6
    max_iterations : `int`, optional
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

    n = initial_guess.size
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
    x = initial_guess.copy()
    while (not converged) and (it<max_iterations):
        it += 1
        g, Jac = function(x, *args)
        g = g.reshape((-1,1))
        dx_neg = linsolver(Jac, g)
        defect = norm(dx_neg[:], ord=np.inf)
        x -= dx_neg.reshape(x.shape)
        converged = (defect<tolerance)
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
        'fun': g.reshape(initial_guess.shape),
        'jac': Jac,
        'nit': it
    })
    return result

def clip_approach(values, function, lower_bounds = 0, upper_bounds = None, factor = 0):
    # filter values with lower and upper bounds using an approach factor
    if (factor == 0):
        np.clip(values, lower_bounds, upper_bounds, out = values)
    else:
        if (lower_bounds != None):
            below_lower = (values < lower_bounds)
            if (np.any(below_lower)):
                broadcasted_lower_bounds = np.broadcast_to(lower_bounds, values.shape)
                values[below_lower] = (1.0 + factor)*broadcasted_lower_bounds[below_lower] - factor*values[below_lower]
        if (upper_bounds != None):
            above_upper = (values > upper_bounds)
            if (np.any(above_upper)):
                broadcasted_upper_bounds = np.broadcast_to(upper_bounds, values.shape)
                values[above_upper] = (1.0 + factor)*broadcasted_upper_bounds[above_upper] - factor*values[above_upper]

def interp_stagg_to_cntr(staggered_values, face_positions, cell_centers = None, axis = 0):
    """
    Interpolate values at staggered positions to cell-centers using linear interpolation.

    Args:
        axis (int): Dimension that is interpolated.
        staggered_values (ndarray): Quantities at staggered positions.
        cell_centers (ndarray): Cell-centered positions.
        face_positions (ndarray): Positions of cell-faces (numel(face_positions) = numel(cell_centers) + 1).

    Returns:
        ndarray: Interpolated concentrations at the cell-centered positions.

    """
    shape_f = list(staggered_values.shape)
    if (axis<0):
        axis += len(shape_f)
    shape_f_t = [math.prod(shape_f[:axis]), shape_f[axis], math.prod(shape_f[axis + 1:])]  # reshape as a triplet
    shape = shape_f.copy()
    shape[axis] = shape[axis] - 1
    staggered_values = np.reshape(staggered_values, shape_f_t)
    if (cell_centers is None):
        cell_centered_values =  0.5 * (staggered_values[:, 1:, :] + staggered_values[:, :-1, :])
    else:
        wght = (cell_centers - face_positions[:-1]) / (face_positions[1:] - face_positions[:-1])
        cell_centered_values = staggered_values[:, :-1, :] + wght.reshape((1,-1,1)) * (staggered_values[:, 1:, :] - staggered_values[:, :-1, :])
    cell_centered_values = cell_centered_values.reshape(shape)
    return cell_centered_values

def interp_cntr_to_stagg(cell_centered_values, face_positions, cell_centers=None, axis=0):
    """
    Interpolate values at staggered positions to cell-centers using linear interpolation.

    Args:
        cell_centered_values (ndarray): Quantities at cell-centered positions.
        face_positions (ndarray): Positions of cell-faces (numel(face_positions) = numel(cell_centers) + 1).
        cell_centers (ndarray, optional): Cell-centered positions. If not provided, the interpolated values will be averaged between adjacent staggered positions.
        axis (int, optional): Dimension along which interpolation is performed. Default is 0.

    Returns:
        ndarray: Interpolated concentrations at staggered positions.

    """
    shape = list(cell_centered_values.shape)
    if (axis<0):
        axis += len(shape)
    shape_t = [math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1:])]  # reshape as a triplet
    shape_f = shape.copy()
    shape_f[axis] = shape[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_f[axis]
    if (cell_centers is None):
        cell_centers = 0.5*(face_positions[:-1]+face_positions[1:])
    wght = (face_positions[1:-1] - cell_centers[:-1]) / (cell_centers[1:] - cell_centers[:-1])
    cell_centered_values = cell_centered_values.reshape(shape_t)
    if (shape_t[1]==1):
        staggered_values = np.tile(cell_centered_values, (1,2,1))
    else:
        staggered_values = np.empty(shape_f_t)
        staggered_values[:,1:-1,:] = cell_centered_values[:, :-1, :] + wght.reshape((1,-1,1)) * (cell_centered_values[:, 1:, :] - cell_centered_values[:, :-1, :])
        staggered_values[:,0,:] = (cell_centered_values[:,0,:]*(cell_centers[1]-face_positions[0]) - cell_centered_values[:,1,:]*(cell_centers[0]-face_positions[0]))/(cell_centers[1]-cell_centers[0])
        staggered_values[:,-1,:] = (cell_centered_values[:,-1,:]*(face_positions[-1]-cell_centers[-2]) - cell_centered_values[:,-2,:]*(face_positions[-1]-cell_centers[-1]))/(cell_centers[-1]-cell_centers[-2])
        staggered_values = staggered_values.reshape(shape_f)
    return staggered_values

def interp_cntr_to_stagg_tvd(cell_centered_values, face_positions, cell_centers=None, boundary_conditions=None, velocity=0, tvd_limiter = None, axis=0):
    """
    Interpolate values at staggered positions to cell-centers using TVD interpolation.

    Args:
        cell_centered_values (ndarray): Quantities at cell-centered positions.
        face_positions (ndarray): Positions of cell-faces (numel(face_positions) = numel(cell_centers) + 1).
        cell_centers (ndarray, optional): Cell-centered positions. If not provided, the interpolated values will be averaged between adjacent staggered positions.
        boundary_conditions  (list, optional): The boundary conditions used to extrapolate to the boundary faces
        velocity   (ndarray, optional): Velocites on face positions
        tvd_limiter (function, optional): The TVD limiter 
        axis (int, optional): Dimension along which interpolation is performed. Default is 0.

    Returns:
        ndarray: Interpolated concentrations at staggered positions.

    """
    shape = list(cell_centered_values.shape)
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
    
    if (cell_centers is None):
        cell_centers = 0.5*(face_positions[:-1]+face_positions[1:])
    cell_centered_values = cell_centered_values.reshape(shape_t)
    staggered_values = np.empty(shape_f_t)
       
    if (shape_t[1] == 1):
        a, b, d = [None]*2, [None]*2, [None]*2
        a[0], b[0], d[0] = unwrap_bc(shape, boundary_conditions[0])
        a[1], b[1], d[1] = unwrap_bc(shape, boundary_conditions[1])
        alpha_1 = (face_positions[1] - face_positions[0]) / ((cell_centers[0] - face_positions[0]) * (face_positions[1] - cell_centers[0]))
        alpha_2L = (cell_centers[0] - face_positions[0]) / ((face_positions[1] - face_positions[0]) * (face_positions[1] - cell_centers[0]))
        alpha_0L = alpha_1 - alpha_2L
        alpha_2R = -(cell_centers[0] - face_positions[1]) / ((face_positions[0] - face_positions[1]) * (face_positions[0] - cell_centers[0]))
        alpha_0R = alpha_1 - alpha_2R        
        fctr = ((b[0] + alpha_0L * a[0]) * (b[1] +
                     alpha_0R * a[1]) - alpha_2L * alpha_2R * a[0] * a[1])
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        fctr_m = (alpha_1 * a[0] * (a[1] * (alpha_0R - alpha_2L) + b[1])
                           * fctr) 
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)
        staggered_values[:,0,:] = fctr_m*cell_centered_values[:,0,:];
        fctr_m = (alpha_1 * a[1] * (a[0] * (alpha_0L - alpha_2R) + b[0])
                           * fctr)
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)      
        staggered_values[:, 1, :] = fctr_m*cell_centered_values[:,0,:]
        fctr_m = ((a[1] * alpha_0R + b[1]) * d[0] - alpha_2L * a[0] * d[1])* fctr
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d)     
        staggered_values[:, 0, :] += fctr_m
        fctr_m = ((a[0] * alpha_0L + b[0]) * d[1] - alpha_2R * a[1] * d[0])* fctr
        fctr_m = fctr_m + np.zeros(shape_bc)
        fctr_m = np.reshape(fctr_m, shape_bc_d) 
        staggered_values[:, 1, :] += fctr_m
        staggered_values.reshape(shape_f)
        delta_staggered_values = np.zeros(shape_f)
    else:
        # bc 0
        a, b, d = unwrap_bc(shape, boundary_conditions[0])
        alpha_1 = (cell_centers[1] - face_positions[0]) / ((cell_centers[0] - face_positions[0]) * (cell_centers[1] - cell_centers[0]))
        alpha_2 = (cell_centers[0] - face_positions[0]) / ((cell_centers[1] - face_positions[0]) * (cell_centers[1] - cell_centers[0]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a + b)
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a * fctr
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        staggered_values[:,0,:] = (d_fctr + a_fctr*(alpha_1*cell_centered_values[:,0,:] - alpha_2*cell_centered_values[:,1,:]))
        # bc 1
        a, b, d = unwrap_bc(shape, boundary_conditions[1])
        alpha_1 = -(cell_centers[-2] - face_positions[-1]) / ((cell_centers[-1] - face_positions[-1]) * (cell_centers[-2] - cell_centers[-1]))
        alpha_2 = -(cell_centers[-1] - face_positions[-1]) / ((cell_centers[-2] - face_positions[-1]) * (cell_centers[-2] - cell_centers[-1]))
        alpha_0 = alpha_1 - alpha_2
        fctr = (alpha_0 * a + b)
        np.divide(1, fctr, out=fctr, where=(fctr != 0))
        a_fctr = a * fctr
        a_fctr = a_fctr + np.zeros(shape_bc)
        a_fctr = np.reshape(a_fctr, shape_bc_d)
        d_fctr = d * fctr
        d_fctr = d_fctr + np.zeros(shape_bc)
        d_fctr = np.reshape(d_fctr, shape_bc_d)
        staggered_values[:,-1,:] = (d_fctr + a_fctr*(alpha_1*cell_centered_values[:,-1,:] - alpha_2*cell_centered_values[:,-2,:]))
        
        velocity = np.array(velocity) + np.zeros(shape_f)
        velocity = velocity.reshape(shape_f_t)
        fltr_v_pos = (velocity > 0)
        
        face_positions = face_positions.reshape((1,-1,1))
        cell_centers = cell_centers.reshape((1,-1,1))
        x_d = face_positions[:,1:-1,:]
        x_C = fltr_v_pos[:,1:-1,:]*cell_centers[:,:-1,:] + ~fltr_v_pos[:,1:-1,:]*cell_centers[:,1:,:]
        x_U = fltr_v_pos[:,1:-1,:]*np.concatenate((face_positions[:,0:1,:],cell_centers[:,0:-2,:]),axis=1) + ~fltr_v_pos[:,1:-1,:]*np.concatenate((cell_centers[:,2:,:], face_positions[:,-1:,:]),axis=1)
        x_D = fltr_v_pos[:,1:-1,:]*cell_centers[:,1:,:] + ~fltr_v_pos[:,1:-1,:]*cell_centers[:,:-1,:]
        x_norm_C = (x_C-x_U)/(x_D-x_U)
        x_norm_d = (x_d-x_U)/(x_D-x_U)
        c_C = fltr_v_pos[:,1:-1,:]*cell_centered_values[:,:-1,:] + ~fltr_v_pos[:,1:-1,:]*cell_centered_values[:,1:,:]
        c_U = fltr_v_pos[:,1:-1,:]*np.concatenate((staggered_values[:,0:1,:],cell_centered_values[:,0:-2,:]),axis=1) + ~fltr_v_pos[:,1:-1,:]*np.concatenate((cell_centered_values[:,2:,:], staggered_values[:,-1:,:]),axis=1)
        c_D = fltr_v_pos[:,1:-1,:]*cell_centered_values[:,1:,:] + ~fltr_v_pos[:,1:-1,:]*cell_centered_values[:,:-1,:];
        c_norm_C = np.zeros_like(c_C);
        dc_DU = (c_D-c_U);
        np.divide((c_C-c_U), dc_DU, out=c_norm_C, where=(dc_DU != 0))
        staggered_values = np.concatenate((staggered_values[:,0:1,:], c_C, staggered_values[:,-1:,:]),axis=1)
        if (tvd_limiter == None):
            delta_staggered_values = np.zeros(shape_f)    
            staggered_values = staggered_values.reshape(shape_f)
        else:
            delta_staggered_values = np.zeros(shape_f_t)
            delta_staggered_values[:,1:-1,:] = tvd_limiter(c_norm_C, x_norm_C, x_norm_d) * dc_DU
            staggered_values += delta_staggered_values
            delta_staggered_values = delta_staggered_values.reshape(shape_f)
            staggered_values = staggered_values.reshape(shape_f)
    return staggered_values, delta_staggered_values

def upwind(normalized_concentration_center, normalized_position_center, normalized_position_downwind):
    """
    Apply the upwind TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        normalized_concentration_center (ndarray): Normalized concentration at cell centers.
        normalized_position_center (ndarray): Normalized position of cell centers.
        normalized_position_downwind (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C. (Equals 0 for upwind)
    """
    normalized_concentration_diff = np.zeros_like(normalized_concentration_center)
    return normalized_concentration_diff
               
def minmod(normalized_concentration_center, normalized_position_center, normalized_position_downwind):
    """
    Apply the Min-mod TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        normalized_concentration_center (ndarray): Normalized concentration at cell centers.
        normalized_position_center (ndarray): Normalized position of cell centers.
        normalized_position_downwind (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0,(normalized_position_downwind-normalized_position_center)*np.minimum(normalized_concentration_center/normalized_position_center, (1-normalized_concentration_center)/(1-normalized_position_center)));
    return normalized_concentration_diff

def osher(normalized_concentration_center, normalized_position_center, normalized_position_downwind):
    """
    Apply the Osher TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        normalized_concentration_center (ndarray): Normalized concentration at cell centers.
        normalized_position_center (ndarray): Normalized position of cell centers.
        normalized_position_downwind (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0,np.where(normalized_concentration_center<normalized_position_center/normalized_position_downwind, (normalized_position_downwind/normalized_position_center - 1)*normalized_concentration_center, 1 - normalized_concentration_center))
    return normalized_concentration_diff

def clam(normalized_concentration_center, normalized_position_center, normalized_position_downwind):
    """
    Apply the CLAM TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        normalized_concentration_center (ndarray): Normalized concentration at cell centers.
        normalized_position_center (ndarray): Normalized position of cell centers.
        normalized_position_downwind (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0,np.where(normalized_concentration_center<normalized_position_center/normalized_position_downwind, (normalized_position_downwind/normalized_position_center - 1)*normalized_concentration_center, 1 - normalized_concentration_center))
    return normalized_concentration_diff

def muscl(normalized_concentration_center, normalized_position_center, normalized_position_downwind):
    """
    Apply the MUSCL limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        normalized_concentration_center (ndarray): Normalized concentration at cell centers.
        normalized_position_center (ndarray): Normalized position of cell centers.
        normalized_position_downwind (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0,np.where(normalized_concentration_center<normalized_position_center/(2*normalized_position_downwind), ((2*normalized_position_downwind - normalized_position_center)/normalized_position_center - 1)*normalized_concentration_center, 
                             np.where(normalized_concentration_center<1 + normalized_position_center - normalized_position_downwind, normalized_position_downwind - normalized_position_center, 1 - normalized_concentration_center)))
    return normalized_concentration_diff

def smart(normalized_concentration_center, normalized_position_center, normalized_position_downwind):
    """
    Apply the SMART TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        normalized_concentration_center (ndarray): Normalized concentration at cell centers.
        normalized_position_center (ndarray): Normalized position of cell centers.
        normalized_position_downwind (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0,np.where(normalized_concentration_center<normalized_position_center/3, (normalized_position_downwind*(1 - 3*normalized_position_center + 2*normalized_position_downwind)/(normalized_position_center*(1 - normalized_position_center)) - 1)*normalized_concentration_center, 
                             np.where(normalized_concentration_center<normalized_position_center/normalized_position_downwind*(1 + normalized_position_downwind - normalized_position_center), (normalized_position_downwind*(normalized_position_downwind - normalized_position_center) + normalized_position_downwind*(1 - normalized_position_downwind)/normalized_position_center*normalized_concentration_center)/(1 - normalized_position_center) - normalized_concentration_center, 1 - normalized_concentration_center)))
    return normalized_concentration_diff

def stoic(normalized_concentration_center, normalized_position_center, normalized_position_downwind):
    """
    Apply the STOIC TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        normalized_concentration_center (ndarray): Normalized concentration at cell centers.
        normalized_position_center (ndarray): Normalized position of cell centers.
        normalized_position_downwind (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0,np.where(normalized_concentration_center<normalized_position_center*(normalized_position_downwind - normalized_position_center)/(normalized_position_center + normalized_position_downwind + 2*normalized_position_downwind*normalized_position_downwind - 4*normalized_position_downwind*normalized_position_center), normalized_position_downwind*(1 - 3*normalized_position_center + 2*normalized_position_downwind)/(normalized_position_center*(1 - normalized_position_center)) - normalized_concentration_center, 
                             np.where(normalized_concentration_center<normalized_position_center, (normalized_position_downwind - normalized_position_center + (1 - normalized_position_downwind)*normalized_concentration_center)/(1 - normalized_position_center) - normalized_concentration_center,
                             np.where(normalized_concentration_center<normalized_position_center/normalized_position_downwind*(1 + normalized_position_downwind - normalized_position_center), (normalized_position_downwind*(normalized_position_downwind - normalized_position_center) + normalized_position_downwind*(1 - normalized_position_downwind)/normalized_position_center*normalized_concentration_center)/(1 - normalized_position_center) - normalized_concentration_center, 1 - normalized_concentration_center))))
    return normalized_concentration_diff

def vanleer(normalized_concentration_center, normalized_position_center, normalized_position_downwind):
    """
    Apply the van-Leer TVD limiter to reduce oscillations in numerical schemes.
    Normalized variables NVD are used.

    Args:
        normalized_concentration_center (ndarray): Normalized concentration at cell centers.
        normalized_position_center (ndarray): Normalized position of cell centers.
        normalized_position_downwind (ndarray): Normalized position of down-wind face

    Returns:
        ndarray: Normalized concentration difference c_norm_d-c_norm_C.
    """
    normalized_concentration_diff = np.maximum(0, normalized_concentration_center*(1-normalized_concentration_center)*(normalized_position_downwind-normalized_position_center)/(normalized_position_center*(1-normalized_position_center)))   
    return normalized_concentration_diff

def non_uniform_grid(left_bound, right_bound, num_points, dx_inf, factor):
    """
    Generate a non-uniform grid of points in the interval [left_bound, right_bound]
    With factor > 1 the refinement will be at the left wall, 
    with 1/factor you will get the same refinement at the right wall.
    
    Parameters:
        left_bound (float): Start point of the interval.
        right_bound (float): End point of the interval.
        num_points (int): Total number of face positions (including left_bound and right_bound)
        dx_inf (float): Limiting upper-bound grid spacing
        factor (float): Factor used to increase grid spacing

    Returns:
        numpy.ndarray: Array containing the non-uniform grid points.
    """
    a = np.log(factor)
    unif = np.arange(num_points)
    b = np.exp(-a * unif)
    L = right_bound - left_bound
    c = (np.exp(a * (L / dx_inf - num_points + 1.0)) - b[-1]) / (1 - b[-1])
    face_positions = left_bound + unif * dx_inf + np.log((1 - c) * b + c) * (dx_inf / a)
    return face_positions

def generate_grid(size, face_positions, cell_centers = None, generate_x_c = False):
    if (size+1 == len(face_positions)):
        face_positions = np.array(face_positions)
    else:
        if (size+1 == len(face_positions)):
            face_positions = np.array(face_positions)
        elif (len(face_positions) ==2):
            face_positions = np.linspace(face_positions[0], face_positions[1], size+1)
        elif (face_positions ==None or len(face_positions) ==0):
            face_positions = np.linspace(0.0, 1.0, size+1)
        else:
            raise ValueError("Grid not properly defined")
    if (generate_x_c):
        if (cell_centers is None):
            cell_centers = 0.5*(face_positions[1:] + face_positions[:-1])
        elif (len(cell_centers) == size):
            cell_centers = np.array(cell_centers)
        else:
            raise ValueError("Cell-centered grid not properly defined")
        return face_positions, cell_centers
    else:
        return face_positions

def unwrap_bc(shape, boundary_conditions):
    """
    Unwrap the boundary conditions for a given shape.

    Args:
        shape (tuple): shape of the domain.
        boundary_conditions (dict): Boundary conditions.

    Returns:
        tuple: Unwrapped boundary conditions (a, b, d).
    """
    if not isinstance(shape, (list,tuple)):
        lgth_shape = 1
    else:
        lgth_shape = len(shape)

    if (boundary_conditions is None):
        a = np.zeros((1,) * lgth_shape)
        b = np.zeros((1,) * lgth_shape)
        d = np.zeros((1,) * lgth_shape)
    else:
        a = np.array(boundary_conditions['a'])
        a = a[(..., *([np.newaxis]*(lgth_shape-a.ndim)))]
        b = np.array(boundary_conditions['b'])
        b = b[(..., *([np.newaxis]*(lgth_shape-b.ndim)))]
        d = np.array(boundary_conditions['d'])
        d = d[(..., *([np.newaxis]*(lgth_shape-d.ndim)))]
    return a, b, d
