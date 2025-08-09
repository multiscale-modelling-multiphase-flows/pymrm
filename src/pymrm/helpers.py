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
    """Return a sparse diagonal matrix from coefficient values.

    Parameters
    ----------
    coefficients : array_like
        Values to place on the diagonal of the matrix.
    shape : tuple of int, optional
        Shape of the multidimensional field to which the coefficients should be
        broadcast.  If ``None`` (default), the coefficients are used as
        provided.
    axis : int, optional
        Axis along which coefficients are defined for staggered grids.  When
        given, the size of this axis is incremented by one to account for
        face-centred values.

    Returns
    -------
    scipy.sparse.csc_array
        Sparse diagonal matrix containing the coefficients.
    """
    if shape is None:
        coeff_matrix = csc_array(diags(coefficients.ravel(), format='csc'))
    else:
        if axis is not None:
            shape = tuple(s if i != axis else s + 1 for i, s in enumerate(shape))
        coefficients_copy = np.array(coefficients)
        shape_coeff = (1,) * (len(shape) - coefficients_copy.ndim) + coefficients_copy.shape
        coefficients_copy = coefficients_copy.reshape(shape_coeff)
        coefficients_copy = np.broadcast_to(coefficients_copy, shape)
        coeff_matrix = csc_array(diags(coefficients_copy.ravel(), format='csc'))
    return coeff_matrix

