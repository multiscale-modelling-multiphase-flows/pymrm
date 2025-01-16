"""
Interpolate Submodule for pymrm

This submodule provides functions for interpolating values between staggered 
and cell-centered grids, which is essential in finite-volume and finite-difference 
schemes for solving partial differential equations. It includes standard linear 
interpolation and Total Variation Diminishing (TVD) schemes to prevent numerical 
oscillations in convective transport problems.

Functions:
-----------
- interp_stagg_to_cntr(staggered_values, x_f, x_c=None, axis=0)
    Linearly interpolate staggered grid values to cell-centered values.

- interp_cntr_to_stagg(cell_centered_values, x_f, x_c=None, axis=0)
    Linearly interpolate cell-centered values to staggered grid positions.

- interp_cntr_to_stagg_tvd(cell_centered_values, x_f, x_c=None, bc=None, v=0, tvd_limiter=None, axis=0)
    Perform TVD interpolation from cell-centered values to staggered positions.

Dependencies:
-------------
- numpy: For array manipulations.
- pymrm.helper: For boundary condition handling (`unwrap_bc`).
"""

import numpy as np
from .helper import unwrap_bc


def interp_stagg_to_cntr(staggered_values, x_f, x_c=None, axis=0):
    """
    Interpolate values at staggered positions to cell-centers using linear interpolation.

    Args:
        staggered_values (ndarray): Quantities at staggered positions.
        x_f (ndarray): Positions of cell-faces.
        x_c (ndarray, optional): Cell-centered positions. If None, they are computed as midpoints.
        axis (int, optional): Dimension to interpolate along. Default is 0.

    Returns:
        ndarray: Interpolated values at cell centers.
    """
    shape_f = list(staggered_values.shape)
    if axis < 0:
        axis += len(shape_f)
    shape_f_t = [np.prod(shape_f[:axis]), shape_f[axis], np.prod(shape_f[axis + 1:])]
    shape = shape_f.copy()
    shape[axis] -= 1
    staggered_values = np.reshape(staggered_values, shape_f_t)

    if x_c is None:
        cell_centered_values = 0.5 * (staggered_values[:, 1:, :] + staggered_values[:, :-1, :])
    else:
        wght = (x_c - x_f[:-1]) / (x_f[1:] - x_f[:-1])
        cell_centered_values = staggered_values[:, :-1, :] + wght.reshape((1, -1, 1)) * \
                               (staggered_values[:, 1:, :] - staggered_values[:, :-1, :])

    return cell_centered_values.reshape(shape)


def interp_cntr_to_stagg(cell_centered_values, x_f, x_c=None, axis=0):
    """
    Interpolate values at cell-centers to staggered positions using linear interpolation.

    Args:
        cell_centered_values (ndarray): Quantities at cell-centered positions.
        x_f (ndarray): Positions of cell-faces.
        x_c (ndarray, optional): Cell-centered positions. If None, they are computed as midpoints.
        axis (int, optional): Dimension to interpolate along. Default is 0.

    Returns:
        ndarray: Interpolated values at staggered positions.
    """
    shape = list(cell_centered_values.shape)
    if axis < 0:
        axis += len(shape)
    shape_t = [np.prod(shape[:axis]), shape[axis], np.prod(shape[axis + 1:])]
    shape_f = shape.copy()
    shape_f[axis] += 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] += 1

    if x_c is None:
        x_c = 0.5 * (x_f[:-1] + x_f[1:])

    wght = (x_f[1:-1] - x_c[:-1]) / (x_c[1:] - x_c[:-1])
    cell_centered_values = cell_centered_values.reshape(shape_t)

    staggered_values = np.empty(shape_f_t)
    staggered_values[:, 1:-1, :] = cell_centered_values[:, :-1, :] + \
                                   wght.reshape((1, -1, 1)) * (cell_centered_values[:, 1:, :] - cell_centered_values[:, :-1, :])

    # Extrapolate boundary values
    staggered_values[:, 0, :] = cell_centered_values[:, 0, :]
    staggered_values[:, -1, :] = cell_centered_values[:, -1, :]

    return staggered_values.reshape(shape_f)


def interp_cntr_to_stagg_tvd(cell_centered_values, x_f, x_c=None, bc=None, v=0, tvd_limiter=None, axis=0):
    """
    Interpolate values at cell-centers to staggered positions using a TVD scheme.

    Args:
        cell_centered_values (ndarray): Quantities at cell-centered positions.
        x_f (ndarray): Positions of cell-faces.
        x_c (ndarray, optional): Cell-centered positions. If None, they are computed as midpoints.
        bc (tuple, optional): Boundary conditions. Default is None.
        v (ndarray or float, optional): Velocity field for upwinding. Default is 0.
        tvd_limiter (callable, optional): TVD limiter function. Default is None.
        axis (int, optional): Dimension to interpolate along. Default is 0.

    Returns:
        ndarray: TVD-interpolated values at staggered positions.
    """
    shape = list(cell_centered_values.shape)
    if axis < 0:
        axis += len(shape)
    shape_t = [np.prod(shape[:axis]), shape[axis], np.prod(shape[axis + 1:])]
    shape_f = shape.copy()
    shape_f[axis] += 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] += 1

    if x_c is None:
        x_c = 0.5 * (x_f[:-1] + x_f[1:])

    v = np.broadcast_to(np.asarray(v), shape_f)
    v_t = v.reshape(shape_f_t)
    fltr_v_pos = v_t > 0

    c_C = fltr_v_pos[:, 1:-1, :] * cell_centered_values[:, :-1, :] + \
          ~fltr_v_pos[:, 1:-1, :] * cell_centered_values[:, 1:, :]

    c_U = fltr_v_pos[:, 1:-1, :] * np.concatenate((cell_centered_values[:, :1, :], cell_centered_values[:, :-2, :]), axis=1) + \
          ~fltr_v_pos[:, 1:-1, :] * np.concatenate((cell_centered_values[:, 2:, :], cell_centered_values[:, -1:, :]), axis=1)

    delta_c = c_C - c_U

    if tvd_limiter is not None:
        delta_c = tvd_limiter(delta_c)

    staggered_values = np.empty(shape_f_t)
    staggered_values[:, 1:-1, :] = c_C + delta_c
    staggered_values[:, 0, :] = cell_centered_values[:, 0, :]
    staggered_values[:, -1, :] = cell_centered_values[:, -1, :]

    return staggered_values.reshape(shape_f)
