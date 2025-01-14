# Import functions and classes from pymrm.py
from .pymrm_core import (
    construct_grad, construct_grad_int, construct_grad_bc, construct_div,
    construct_convflux_upwind, construct_convflux_upwind_int, construct_convflux_upwind_bc,
    construct_coefficient_matrix, newton, clip_approach, non_uniform_grid,
    interp_stagg_to_cntr, interp_cntr_to_stagg, interp_cntr_to_stagg_tvd, upwind,
    minmod, osher, clam, muscl, smart, stoic, vanleer
)

# Import NumJac directly to expose it at the package level
from .numjac import NumJac, stencil_block_diagonals

# Optional: Define what gets imported with `from pymrm import *`
__all__ = [
    'construct_grad', 'construct_grad_int', 'construct_grad_bc', 'construct_div',
    'construct_convflux_upwind', 'construct_convflux_upwind_int', 'construct_convflux_upwind_bc',
    'construct_coefficient_matrix', 'newton', 'clip_approach', 'non_uniform_grid',
    'interp_stagg_to_cntr', 'interp_cntr_to_stagg', 'interp_cntr_to_stagg_tvd', 'upwind',
    'minmod', 'osher', 'clam', 'muscl', 'smart', 'stoic', 'vanleer',
    'NumJac', 'stencil_block_diagonals'
]