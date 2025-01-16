"""
pymrm: A Python Package for Multiphase Reactor Modeling

This package provides a comprehensive set of tools for modeling multiphase reactors, 
including grid generation, numerical operators, convection schemes, interpolation methods, 
nonlinear solvers, and utility functions.

**Submodules:**
- `grid`: Functions for generating uniform and non-uniform grids.
- `operator`: Construction of gradient and divergence operators for finite volume methods.
- `convection`: High-resolution convection schemes and TVD limiters.
- `interpolate`: Interpolation techniques between staggered and cell-centered grids.
- `solve`: Nonlinear solvers and numerical approaches.
- `helpers`: Utility functions supporting core operations.
- `numjac`: Numerical Jacobian construction for nonlinear systems.

**Example Usage:**
```python
from pymrm import non_uniform_grid, construct_grad, newton

# Generate a non-uniform grid
x_f = non_uniform_grid(0, 1, 100, dx_inf=0.01, factor=1.05)

# Construct a gradient operator
grad, grad_bc = construct_grad((100,), x_f)

# Use Newton's method to solve a nonlinear problem
solution = newton(lambda x: x**2 - 2, initial_guess=1.0)

Authors:
- E.A.J.F. Peters
- M. van Sint Annaland
- M. Galanti
- D.R. Rieder

License: MIT License
"""

from .grid import generate_grid, non_uniform_grid
from .operators import construct_grad, construct_grad_int, construct_grad_bc, construct_div
from .convect import (
    construct_convflux_upwind, construct_convflux_upwind_int, construct_convflux_upwind_bc,
    upwind, minmod, osher, clam, muscl, smart, stoic, vanleer
)
from .interpolate import interp_stagg_to_cntr, interp_cntr_to_stagg, interp_cntr_to_stagg_tvd, create_staggered_array
from .solve import newton, clip_approach
from .numjac import NumJac, stencil_block_diagonals
from .helpers import unwrap_bc, construct_coefficient_matrix

__all__ = [
    "generate_grid", "non_uniform_grid",
    "construct_grad", "construct_grad_int", "construct_grad_bc", "construct_div",
    "construct_convflux_upwind", "construct_convflux_upwind_int", "construct_convflux_upwind_bc",
    "upwind", "minmod", "osher", "clam", "muscl", "smart", "stoic", "vanleer",
    "interp_stagg_to_cntr", "interp_cntr_to_stagg", "interp_cntr_to_stagg_tvd",
    "newton", "clip_approach",
    "NumJac", "stencil_block_diagonals",
    "construct_coefficient_matrix"
]