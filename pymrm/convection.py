"""
convection.py

This submodule of pymrm provides functions to construct convective flux matrices using
upwind schemes and apply Total Variation Diminishing (TVD) limiters for numerical stability.

Functions:
    - construct_convflux_upwind: Constructs the convective flux matrix using the upwind scheme.
    - construct_convflux_upwind_int: Constructs the internal convective flux matrix.
    - construct_convflux_upwind_bc: Constructs the convective flux matrix for boundary conditions.
    - upwind: Upwind TVD limiter.
    - minmod: Minmod TVD limiter.
    - osher: Osher TVD limiter.
    - clam: CLAM TVD limiter.
    - muscl: MUSCL TVD limiter.
    - smart: SMART TVD limiter.
    - stoic: STOIC TVD limiter.
    - vanleer: Van Leer TVD limiter.
"""

import numpy as np
from scipy.sparse import csc_array
from .interpolation import interp_cntr_to_stagg
from .helper import unwrap_bc, create_staggered_array

def construct_convflux_upwind(shape, x_f, x_c=None, bc=(None, None), v=1.0, axis=0):
    """Construct the convective flux matrix using the upwind scheme."""
    x_f, x_c = interp_cntr_to_stagg(x_f, x_c, axis=axis)
    v_f = create_staggered_array(v, shape, axis, x_f=x_f, x_c=x_c)
    conv_matrix = construct_convflux_upwind_int(shape, v_f, axis)
    
    if bc == (None, None):
        conv_bc = csc_array((np.prod(shape), 1))
    else:
        conv_matrix_bc, conv_bc = construct_convflux_upwind_bc(shape, x_f, x_c, bc, v_f, axis)
        conv_matrix += conv_matrix_bc
    
    return conv_matrix, conv_bc

def construct_convflux_upwind_int(shape, v=1.0, axis=0):
    """Construct the internal convective flux matrix using the upwind scheme."""
    shape_f = list(shape)
    shape_f[axis] += 1
    v_f = np.broadcast_to(v, shape_f)
    fltr_v_pos = (v_f > 0)
    
    i_f = np.arange(np.prod(shape_f)).reshape(shape_f)
    i_c = np.arange(np.prod(shape)).reshape(shape)
    i_c = i_c - fltr_v_pos.astype(int)

    conv_matrix = csc_array((v_f.ravel(), (i_f.ravel(), i_c.ravel())), shape=(np.prod(shape_f), np.prod(shape)))
    return conv_matrix

def construct_convflux_upwind_bc(shape, x_f, x_c, bc, v, axis=0):
    """Construct the convective flux matrix for boundary conditions using the upwind scheme."""
    a_left, b_left, d_left = unwrap_bc(shape, bc[0])
    a_right, b_right, d_right = unwrap_bc(shape, bc[1])

    conv_bc = np.zeros(np.prod(shape) + 1)
    conv_bc[0] = a_left * d_left
    conv_bc[-1] = a_right * d_right
    conv_bc = csc_array(conv_bc.reshape(-1, 1))

    conv_matrix_bc = csc_array((np.zeros(np.prod(shape)), (np.arange(np.prod(shape)), np.arange(np.prod(shape)))))
    return conv_matrix_bc, conv_bc

# TVD Limiters

def upwind(r):
    """Upwind TVD limiter."""
    return np.zeros_like(r)

def minmod(r):
    """Minmod TVD limiter."""
    return np.maximum(0, np.minimum(r, 1))

def osher(r):
    """Osher TVD limiter."""
    return np.maximum(0, np.minimum(2 * r, 1))

def clam(r):
    """CLAM TVD limiter."""
    return np.maximum(0, np.minimum(r, 2))

def muscl(r):
    """MUSCL TVD limiter."""
    return np.maximum(0, np.minimum((1 + r) / 2, 2))

def smart(r):
    """SMART TVD limiter."""
    return np.maximum(0, np.minimum(2 * r, (1 + 3 * r) / 4, 4))

def stoic(r):
    """STOIC TVD limiter."""
    return np.maximum(0, np.minimum((2 + r) / 3, 2))

def vanleer(r):
    """Van Leer TVD limiter."""
    return (r + np.abs(r)) / (1 + np.abs(r))
