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

def newton(function, initial_guess, args=(), tolerance=1.49012e-08, max_iterations=100, solver=None, callback=None):
    """
    Perform Newton-Raphson iterations to solve nonlinear systems.

    Args:
        function (callable): Function returning the residual and Jacobian.
        initial_guess (ndarray): Initial guess for the solution.
        args (tuple, optional): Additional arguments for the function.
        tolerance (float, optional): Convergence criterion. Default is 1.49012e-08.
        max_iterations (int, optional): Maximum iterations allowed. Default is 100.
        solver (str, optional): Linear solver to use ('spsolve', 'cg', 'bicgstab').
        callback (callable, optional): Function called after each iteration.

    Returns:
        OptimizeResult: Contains the solution, success status, and diagnostic info.
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
    for it in range(max_iterations):
        g, jac_matrix = function(x, *args)
        dx_neg = linsolver(jac_matrix, g)
        defect = norm(dx_neg, ord=np.inf)
        x -= dx_neg.reshape(x.shape)
        if callback:
            callback(x, g)
        if defect < tolerance:
            return OptimizeResult({'x': x, 'success': True, 'nit': it + 1, 'fun': g, 'message': 'Converged'})

    return OptimizeResult({'x': x, 'success': False, 'nit': max_iterations, 'fun': g, 'message': 'Did not converge'})