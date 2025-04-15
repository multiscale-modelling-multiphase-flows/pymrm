# solve.py
"""
The `solve` module provides numerical solvers for nonlinear systems, including
Newton-Raphson methods for efficiently solving systems of equations arising in
multiphase reactor modeling.
"""

import numpy as np
from scipy.sparse import linalg
from scipy.linalg import norm
from scipy.optimize import OptimizeResult

def newton(function, initial_guess, args=(), tol=1.49012e-08, maxfev=100, solver=None, callback=None):
    """
    Perform Newton-Raphson iterations to solve nonlinear systems of equations.

    This method iteratively refines an initial guess to find the root of a system
    of nonlinear equations. It supports various linear solvers for handling the
    Jacobian matrix.

    Args:
        function (callable): A function that takes the current solution estimate
            and additional arguments, and returns a tuple containing the residual
            vector and the Jacobian matrix.
        initial_guess (ndarray): Initial guess for the solution vector.
        args (tuple, optional): Additional arguments to pass to the `function`.
            Default is an empty tuple.
        tol (float, optional): Convergence tolerance for the solution. Iterations
            stop when the infinity norm of the update vector is less than `tol`.
            Default is 1.49012e-08.
        maxfev (int, optional): Maximum number of iterations allowed. Default is 100.
        solver (str, optional): Linear solver to use for solving the Jacobian system.
            Options are 'spsolve', 'cg', or 'bicgstab'. If not specified, 'spsolve'
            is used for small systems (n < 50000), and 'bicgstab' for larger systems.
        callback (callable, optional): A function called after each iteration with
            the current solution estimate and residual vector as arguments.

    Returns:
        OptimizeResult: An object containing the following fields:
            - x (ndarray): The solution vector.
            - success (bool): Whether the solver converged.
            - nit (int): Number of iterations performed.
            - fun (ndarray): The residual vector at the solution.
            - message (str): A message describing the termination status.

    Raises:
        ValueError: If an unsupported solver method is specified.
    """
    n = initial_guess.size
    if solver is None:
        solver = 'spsolve' if n < 50000 else 'bicgstab'

    # Select linear solver
    if solver == 'spsolve':
        linsolver = linalg.spsolve
    elif solver == 'cg':
        def linsolver(jac_matrix, g):
            Jac_iLU = linalg.spilu(jac_matrix)
            M = linalg.LinearOperator((n, n), Jac_iLU.solve)
            dx_neg, info = linalg.cg(jac_matrix, g, tol=1e-9, maxiter=1000, M=M)
            return dx_neg
    elif solver == 'bicgstab':
        def linsolver(jac_matrix, g):
            Jac_iLU = linalg.spilu(jac_matrix)
            M = linalg.LinearOperator((n, n), Jac_iLU.solve)
            dx_neg, info = linalg.bicgstab(jac_matrix, g, tol=1e-9, maxiter=1000, M=M)
            return dx_neg
    else:
        raise ValueError("Unsupported solver method.")

    x = initial_guess.copy()
    for it in range(int(maxfev)):
        g, jac_matrix = function(x, *args)
        dx_neg = linsolver(jac_matrix, g)
        defect = norm(dx_neg, ord=np.inf)
        x -= dx_neg.reshape(x.shape)
        if callback:
            callback(x, g)
        if defect < tol:
            return OptimizeResult({'x': x, 'success': True, 'nit': it + 1, 'fun': g, 'message': 'Converged'})

    return OptimizeResult({'x': x, 'success': False, 'nit': maxfev, 'fun': g, 'message': 'Did not converge'})

def clip_approach(values, function, lower_bounds=0, upper_bounds=None, factor=0):
    """
    Apply bounds and an approach factor to an array of values.

    This function modifies the input array `values` in-place by applying lower
    and upper bounds. If an approach factor is specified, values outside the
    bounds are adjusted proportionally to the factor.

    Args:
        values (ndarray): The array of values to be modified.
        function (callable): A function to apply to the values (not used in the
            current implementation but reserved for future extensions).
        lower_bounds (float or ndarray, optional): The lower bounds for the values.
            Can be a scalar or an array of the same shape as `values`. Default is 0.
        upper_bounds (float or ndarray, optional): The upper bounds for the values.
            Can be a scalar or an array of the same shape as `values`. Default is None.
        factor (float, optional): The approach factor. If 0, values are clipped
            directly to the bounds. If non-zero, values outside the bounds are
            adjusted proportionally. Default is 0.

    Notes:
        - The function modifies the `values` array in-place.
        - If `lower_bounds` or `upper_bounds` are not specified, they are ignored.
    """
    if factor == 0:
        np.clip(values, lower_bounds, upper_bounds, out=values)
    else:
        if lower_bounds is not None:
            below_lower = (values < lower_bounds)
            if np.any(below_lower):
                broadcasted_lower_bounds = np.broadcast_to(lower_bounds, values.shape)
                values[below_lower] = (1.0 + factor) * broadcasted_lower_bounds[below_lower] - factor * values[below_lower]
        if upper_bounds is not None:
            above_upper = (values > upper_bounds)
            if np.any(above_upper):
                broadcasted_upper_bounds = np.broadcast_to(upper_bounds, values.shape)
                values[above_upper] = (1.0 + factor) * broadcasted_upper_bounds[above_upper] - factor * values[above_upper]

