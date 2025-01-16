"""
Grid Generation Submodule for pymrm
-----------------------------------

This submodule provides essential functions for creating structured grids 
used in multiphase reactor modeling. It includes uniform and non-uniform 
grid generation methods.

Functions:
- non_uniform_grid: Generates a non-uniform grid with refined control over spacing.
- generate_grid: Constructs face-centered and optionally cell-centered grid positions.

Authors:
- E.A.J.F. Peters (TU/e)
"""

import numpy as np

def non_uniform_grid(left_bound, right_bound, num_points, dx_inf, factor):
    """
    Generate a non-uniform grid of points in the interval [left_bound, right_bound].

    This grid allows gradual spacing adjustments, refining or coarsening the mesh
    based on a stretching factor.

    Args:
        left_bound (float): Start of the domain interval.
        right_bound (float): End of the domain interval.
        num_points (int): Number of face positions (including boundaries).
        dx_inf (float): Asymptotic grid spacing for distant points.
        factor (float): Stretching factor (>1 for expansion, <1 for compression).

    Returns:
        np.ndarray: Array of non-uniformly spaced grid points.
    """
    a = np.log(factor)
    unif = np.arange(num_points)
    b = np.exp(-a * unif)
    length = right_bound - left_bound
    c = (np.exp(a * (length / dx_inf - num_points + 1.0)) - b[-1]) / (1 - b[-1])
    x_f = left_bound + unif * dx_inf + np.log((1 - c) * b + c) * (dx_inf / a)
    return x_f


def generate_grid(size, x_f=None, generate_x_c=False, x_c=None):
    """
    Generate a structured grid with face and optional cell-centered positions.

    Args:
        size (int): Number of cells in the grid.
        x_f (np.ndarray, optional): Face-centered grid points. If None, a uniform grid is created.
        generate_x_c (bool, optional): If True, generates cell-centered positions.
        x_c (np.ndarray, optional): User-defined cell-centered positions (optional).

    Returns:
        np.ndarray or tuple[np.ndarray, np.ndarray]: 
            - Face positions (x_f)
            - Cell-centered positions (x_c) if `generate_x_c` is True
    """
    if x_f is None or len(x_f) == 0:
        # Default to a uniform grid between 0 and 1 if x_f is not provided
        x_f = np.linspace(0.0, 1.0, size + 1)
    elif len(x_f) == size + 1:
        x_f = np.asarray(x_f)
    elif len(x_f) == 2:
        # Create uniform grid between specified boundaries
        x_f = np.linspace(x_f[0], x_f[1], size + 1)
    else:
        raise ValueError("Grid cannot be generated: check 'x_f' length.")

    if generate_x_c:
        if x_c is None:
            # Compute midpoints if no cell-centered grid is provided
            x_c = 0.5 * (x_f[1:] + x_f[:-1])
        elif len(x_c) == size:
            x_c = np.asarray(x_c)
        else:
            raise ValueError("Cell-centered grid not properly defined.")
        return x_f, x_c

    return x_f