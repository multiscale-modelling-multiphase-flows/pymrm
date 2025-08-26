"""pymrm.helpers
=================

Utility helpers used throughout :mod:`pymrm`.

The functions in this module provide small building blocks that are reused in
multiple numerical routines.  They focus on preparing arrays for boundary
conditions and on constructing sparse coefficient matrices that are used in the
finite volume discretisation implemented by the package.

Functions
---------
``unwrap_bc_coeff``
    Expand boundary-condition coefficients to match an arbitrary domain shape.
``construct_coefficient_matrix``
    Create a sparse diagonal matrix from coefficient values.
"""

import math
import numpy as np
from scipy.sparse import diags, csc_array


def unwrap_bc_coeff(shape, bc_coeff, axis=0):
    """Expand boundary-condition coefficients to match a domain shape.

    Parameters
    ----------
    shape : tuple of int
        Target shape of the domain.
    bc_coeff : array_like
        Boundary-condition coefficient (e.g. ``a``, ``b`` or ``d`` terms).
    axis : int, optional
        Axis along which the coefficient applies.  The coefficient is expanded
        along this axis when needed.  Default is ``0``.

    Returns
    -------
    numpy.ndarray
        Array broadcast to ``shape`` containing the boundary-condition
        coefficients.
    """
    if not isinstance(shape, (list, tuple)):
        lgth_shape = 1
    else:
        lgth_shape = len(shape)

    a = np.array(bc_coeff)
    if a.ndim == (lgth_shape - 1):
        a = np.expand_dims(a, axis=axis)
    elif a.ndim != lgth_shape:
        shape_a = (1,) * (lgth_shape - a.ndim) + a.shape
        a = a.reshape(shape_a)
    return a


def construct_coefficient_matrix(coefficients, shape=None, axis=None):
    """
    Build a sparse coefficient matrix with optional broadcasting and (row, col) coupling.

    Modes
    -----
    1. shape is None
       Treat coefficients as a flat sequence placed on the diagonal of an N×N matrix.

    2. shape is a single tuple, e.g. ``(Nz, Nr, ...)``
       Broadcast coefficients to that multidimensional shape (expanding leading size-1
       dimensions as needed) then place all values on the diagonal of a square matrix of
       size ``prod(shape)``. If ``axis`` is given, that dimension is first incremented
       by 1 (staggered / face-centred length) before broadcasting.

    3. shape is a pair of tuples: ``(shape_rows, shape_cols)``
       A dimension in either ``shape_rows`` or ``shape_cols`` can be singular. This can
       create a (possibly rectangular) matrix that couples two fields with different
       (but same-rank) shapes. The working broadcast shape is the element-wise maximum
       of ``shape_rows`` and ``shape_cols`` (adjusted by +1 along ``axis`` if provided;
       matching staggered dims in rows/cols are expanded too).

       Result::

           n_rows = prod(shape_rows)
           n_cols = prod(shape_cols)
           nnz    = prod(working_shape)

    Parameters
    ----------
    coefficients : array_like
        Scalar field to broadcast; shape must be broadcast-compatible with the target.
    shape : None | tuple | (tuple, tuple), optional
        Selects mode (see above).
    axis : int, optional
        Staggered axis: length along this axis is increased by 1 for broadcasting.

    Returns
    -------
    csc_array
        Sparse matrix in CSC format.

    Notes
    -----
    - In mode (3) this is not a diagonal; it is a pointwise coupling pattern.
    - Broadcasting follows NumPy rules after auto-prepending leading 1s.

    Examples
    --------
    Diagonal from flat:
        A = construct_coefficient_matrix(np.ones(20))
    Diagonal from 2D field (staggered in axis 0):
        A = construct_coefficient_matrix(kz, shape=(Nz, Nr), axis=0)
    Rectangular coupling (cell centers -> axial faces):
        A = construct_coefficient_matrix(alpha, shape=((1, Nr), (Nz, Nr)), axis=0)
    """
    if (shape is None):
        coeff_matrix = csc_array(diags(coefficients.ravel(), format='csc'))
    elif all(isinstance(t, tuple) for t in shape):
        shape_rows = shape[0]
        shape_cols = shape[1]
        working_shape = tuple(max(s1, s2) for s1, s2 in zip(shape_rows, shape_cols))
        if axis is not None:
            working_shape = tuple(s if i != axis else s + 1 for i, s in enumerate(working_shape))
            if shape_rows[axis]+1 == working_shape[axis]:
                shape_rows = tuple(s if i != axis else s + 1 for i, s in enumerate(shape_rows))
            if shape_cols[axis]+1 == working_shape[axis]:
                shape_cols = tuple(s if i != axis else s + 1 for i, s in enumerate(shape_cols))
        if coefficients.shape == working_shape:
            coefficients_copy = coefficients
        else:
            coefficients_copy = np.array(coefficients)
            shape_coeff = (1,) * (len(working_shape) - coefficients_copy.ndim) + coefficients_copy.shape
            coefficients_copy = coefficients_copy.reshape(shape_coeff)
            coefficients_copy = np.broadcast_to(coefficients_copy, working_shape)
        num_rows = math.prod(shape_rows)
        rows = np.arange(num_rows).reshape(shape_rows)
        rows = np.broadcast_to(rows, working_shape).ravel()
        num_cols = math.prod(shape_cols)
        cols = np.arange(num_cols).reshape(shape_cols)
        cols = np.broadcast_to(cols, working_shape).ravel()
        coeff_matrix = csc_array((coefficients_copy.ravel(), (rows, cols)), shape=(num_rows, num_cols))
    else:
        if axis is not None:
            shape = tuple(s if i != axis else s + 1 for i, s in enumerate(shape))
        coefficients_copy = np.array(coefficients)
        shape_coeff = (1,) * (len(shape) - coefficients_copy.ndim) + coefficients_copy.shape
        coefficients_copy = coefficients_copy.reshape(shape_coeff)
        coefficients_copy = np.broadcast_to(coefficients_copy, shape)
        coeff_matrix = csc_array(diags(coefficients_copy.ravel(), format='csc'))
    return coeff_matrix
