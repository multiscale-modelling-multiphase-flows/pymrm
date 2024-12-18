"""
Universal Dependency Notation
-----------------------------

This notation provides a consistent format for describing stencil dependencies, 
block dependencies, and range-based dependencies on multidimensional arrays. 
Each dependency is represented as a triple:

    (reference_index, dependent_index, fixed_axes_list, periodic_axes_list)

- `reference_index`:
  A tuple representing the reference position in the array. 
  If the dependency does need a reference (i.e. when fixed_axes_list is empty) this can be set to None.

- `dependent_index`:
  A tuple representing a dependent index.

- `fixed_axes_list`:
  A list of axis indices that are considered "fixed".
  If no axes are fixed, this can be an empty list `[]`.
  
- `periodic_axes_list`:
    A list of axis indices that are considered "periodic".
    If no axes are periodic, this can be an empty list `[]`.

to process simple stencil patterns, block dependencies, and range dependencies 
into a fully expanded set of integer index tuples.

Examples of the uniform dependency notation
--------

(None, (0,0,0), [],[]):      entries with indices (i,j,k) are dependent on the entries with indices (i,j,k).
(None, (0,1,0), [],[]):      entries with indices (i,j,k) are dependent on the entries with indices (i,j+1,k).
(None, (-1,1,0), [],[]):     entries with indices (i,j,k) are dependent on the entries with indices (i-1,j+1,k).
((0,0,0), (-1,1,0), [2],[]): entries with indices (i,j,0) are dependent on the entries with indices (i-1,j+1,0).
((0,0,0), (-1,1,1), [2],[]): entries with indices (i,j,0) are dependent on the entries with indices (i-1,j+1,1).
((0,0,1), (-1,1,0), [2],[]): entries with indices (i,j,1) are dependent on the entries with indices (i-1,j+1,0).

This indicates that there is no specific reference index and no fixed axes.
The pattern is simply a set of dependent offsets. 
This example is for a 3-point dependency along the 0 axis in a 3D array.

Shorthand Dependency Notation
-----------------------------
1. Simple Stencil (No Reference, No Fixed Axes, No periodic axes)

In case of a simple stencil the triple structure can be simplified to a single tuple.

(None, (0,0,0), [],[]) can be written as (0,0,0).
(None, (0,1,0), [],[]) can be written as (0,1,0).
(None, (-1,1,0), [],[]) can be written as (-1,1,0).

2. No periodic axes

If there are no periodic axes the periodic_axes_list can be omitted.

((0,0,0), (0,0,1), [2],[]) can be written as ((0,0,0), (0,0,1), [2]).

2. Expansion of Dependent Indices

If list or slice is provided for the dependent indices this will be expanded to a list of dependencies:

Examples:
([-1,0,1],0,0) expands to [(-1,0,0),(0,0,0),(1,0,0)].
((0,0,0), ([-1,0,1],0,0), [2]) expands to [((0,0,0), (-1,0,0), [2], []), ((0,0,0), (0,0,0), [2], []), ((0,0,0), (1,0,0), [2], [])].
((0,0,0), ([-1,0,1],0,[0,1]), [2]) expands to [((0,0,0), (-1,0,0), [2], []), ((0,0,0), (0,0,0), [2], []), ((0,0,0), (1,0,0), [2], []), ((0,0,0), (-1,0,1), [2], []), ((0,0,0), (0,0,1), [2], []), ((0,0,1), (1,0,1), [2], [])].
For a 7-point stencil in 3D the notation is: [([-1,0,1],0,0),(0,[-1,0,1],0),(0,0,[-1,0,1])]

For block dependencies it can be convenient to specify a range using the slice notation:
((0,0,0), (1,0,slice(None)), [2]) expands to [((0,0,0), (1,0,0), [2], []), ((0,0,0), (1,0,1), [2], []), ..., ((0,0,0), (1,0,shape[2]-1), [2], [])].

3. Expansion of Reference Indices
For the reference indices the expansion is similar to the dependent indices

Example:
((0,0,[0,2,4]), (0,0,[1,3,5]), [2]) expands to [((0,0,0), (0,0,[1,3,5]), [2], []), ((0,0,2), (0,0,[1,3,5]), [2], []), ((0,0,4), (0,0,[1,3,5]), [2], [])]
For each of the entries the dependent indiced can next be further expanded.
"""

import numpy as np
from scipy.sparse import csc_array, csr_array, lil_array, sparray
from scipy.sparse.csgraph import reverse_cuthill_mckee
from numba import njit, jit
import numpy as np
from typing import List, Tuple, Union

def expand_dependencies(shape, dependencies):
    """
    Expand a given set of dependencies into a uniform list of tuples in the 
    universal dependency notation, fully expanded.

    Universal Dependency Notation:
    ------------------------------
    (reference_index, dependent_index, fixed_axes_list, periodic_axes_list)

    - reference_index: A tuple representing the reference position in the array,
      or None if no reference is needed (e.g., simple stencil).
    - dependent_index: A tuple representing a single dependent position in the array.
    - fixed_axes_list: A list of axes considered "fixed" (must remain a list).
      Use [] if none are fixed.
    - periodic_axes_list: A list of axes considered "periodic" (must remain a list).

    Shorthand Notation:
    -------------------
    For simple stencils without reference and fixed axes, you can write just 
    the dependent index tuple (e.g., (0,0,0)) instead of (None, (0,0,0), [], []).

    Expansion Rules:
    ----------------
    Any slice, list, or range in the reference_index or dependent_index should 
    be expanded into a list of integer index tuples.

    Steps Implemented by this Function:
    1. Normalize input: If `dependencies` is a single tuple, convert it into a list.
    2. Ensure each dependency is in triple form:
       - If a dependency is just a single tuple (e.g., (0,1,0)), convert it to 
         (None, (0,1,0), [], []).
       - Otherwise, it should already be in (reference_index, dependent_index, fixed_axes_list) form.
    3. Ensure fixed_axes_list is a list. If None is found, convert it to [].
    4. Expand reference_index and dependent_index:
       - For each dimension, if you encounter:
         * An integer: no change.
         * A slice: expand it into a list of integers [0 ... shape[dim]-1] or the specified subrange.
         * A list of integers: treat it as multiple indices along that dimension.
         * A range object: convert it to a list of integers.
       - Perform a Cartesian product to get all combinations if multiple dimensions have expansions.
    5. After expansion, return a fully expanded list of (reference_index, dependent_index, fixed_axes_list) tuples 
       with only integers in indices.

    Example:
    --------
    shape = (4,4,4)

    Input:
    dependencies = ((0,0,[1,3,5]), ([-1,0,1],0,0), [2])

    After expansion, this yields multiple entries like:
    [((0,0,1), (-1,0,0), [2], []),
     ((0,0,1), (0,0,0), [2], []),
     ((0,0,1), (1,0,0), [2], []),
     ((0,0,3), (-1,0,0), [2], []),
     ... and so forth]

    Returns:
    --------
    A list of tuples in the form (reference_index, dependent_index, fixed_axes_list)
    with all indices fully expanded into integers.
    """

    # Helper functions
    def slice_to_list(slc, dim_size):
        return list(range(*slc.indices(dim_size)))

    def expand_axis(axis_val, dim_size):
        """
        Expand a single axis value which could be:
        - int
        - slice
        - list of ints/slices
        - range
        """
        if isinstance(axis_val, int):
            # Single integer, no expansion needed
            return [axis_val]
        elif isinstance(axis_val, slice):
            # Expand slice
            return slice_to_list(axis_val, dim_size)
        elif isinstance(axis_val, list):
            # Could be a list of integers or slices
            expanded_list = []
            for v in axis_val:
                if isinstance(v, int):
                    expanded_list.append(v)
                elif isinstance(v, slice):
                    expanded_list.extend(slice_to_list(v, dim_size))
                elif isinstance(v, range):
                    expanded_list.extend(list(v))
                else:
                    raise ValueError("Unsupported element in list for axis expansion: {}".format(type(v)))
            return expanded_list
        elif isinstance(axis_val, range):
            return list(axis_val)
        else:
            raise ValueError("Unsupported type in axis specification: {}".format(type(axis_val)))

    def expand_index(idx, shape):
        """
        Expand a single index tuple. The index can contain ints, slices, lists, or ranges.
        Returns a list of fully expanded tuples, or [None] if idx is None.
        """
        if not isinstance(idx, tuple):
            raise ValueError("Index must be a tuple or None.")

        expanded_dims = []
        for i, val in enumerate(idx):
            expanded_dims.append(expand_axis(val, shape[i]))

        # Cartesian product of all expanded dimensions
        from itertools import product
        return list(product(*expanded_dims))

    # Normalize dependencies to a list
    if isinstance(dependencies, tuple):
        dependencies = [dependencies]

    # Convert shorthand notation to full triple form
    normalized_deps = []
    for dep in dependencies:
        if not isinstance(dep, tuple):
            raise ValueError("Each dependency must be a tuple.")

        if len(dep) == 3 and isinstance(dep[1], tuple):
            # Almost in full, assuming no periodic axes
            reference_index, dependent_index, fixed_axes = dep
            periodic_axes = []
        elif len(dep) == 4 and isinstance(dep[1], tuple):
            # Already in full form
            reference_index, dependent_index, fixed_axes, periodic_axes = dep
        else:
            # Shorthand form (e.g., (0,1,0))
            reference_index = (0,)*len(shape)
            dependent_index = dep
            fixed_axes = []
            periodic_axes = []

        # Ensure fixed_axes is a list
        if fixed_axes is None:
            fixed_axes = []
        elif not isinstance(fixed_axes, list):
            raise ValueError("fixed_axes_list must be a list or None.")
        
        # Ensure periodic_axes is a list
        if periodic_axes is None:
            periodic_axes = []
        elif not isinstance(periodic_axes, list):
            raise ValueError("periodic_axes_list must be a list or None.")

        normalized_deps.append((reference_index, dependent_index, fixed_axes, periodic_axes))

    # Now expand reference and dependent indices
    expanded_deps = []
    for (ref_idx, dep_idx, fixed_axes, periodic_axes) in normalized_deps:
        if (ref_idx==None):
            if len(fixed_axes) > 0:
                raise ValueError("Fixed axes are not allowed when reference index is None.")
            ref_idx = (0,)*len(shape)
        ref_expanded = expand_index(ref_idx, shape)    # list of tuples or [None]
        dep_expanded = expand_index(dep_idx, shape)    # list of tuples

        for r in ref_expanded:
            for d in dep_expanded:
                expanded_deps.append((r, d, fixed_axes, periodic_axes))

    return expanded_deps

@njit
def ravel_index_numba(shape, index):
    "Convert a multidimensional index to a flat index."
    lin_idx = 0
    for i in range(len(shape)):
        lin_idx = lin_idx * shape[i] + index[i]
    return lin_idx

@njit
def unravel_index_numba(lin_idx, shape):
    "Convert a flat (linear) index to a multidimensional index."
    idx = np.empty(len(shape), dtype=np.int64)
    for i in range(len(shape) - 1, -1, -1):
        idx[i] = lin_idx % shape[i]
        lin_idx //= shape[i]
    return idx

@njit
def iterate_over_entries(shape, shape_rel, in_pos, out_pos, row_indices, col_indices, start_idx):
    """
    Iterate over all valid relative entries defined by shape_rel and fill row_indices and col_indices.
    This function:
    - Interprets shape_rel as the size of the relative axes.
    - For each combination of relative indices (idx_rel), computes the absolute positions (out_idx, in_idx)
      by adding idx_rel to out_pos and in_pos and applying periodicity.
    - Stores the computed flat indices in row_indices and col_indices.

    Args:
        shape (np.ndarray): The full shape of the multidimensional array.
        shape_rel (np.ndarray): Shape array representing the size of relative axes.
        in_pos (np.ndarray): Input position array (adjusted for relative indexing).
        out_pos (np.ndarray): Output position array (adjusted for relative indexing).
        row_indices (np.ndarray): Preallocated array for row indices.
        col_indices (np.ndarray): Preallocated array for column indices.
        start_idx (int): Starting index in the row_indices/col_indices arrays to place data.

    Returns:
        int: The updated index after filling in entries.
    """
    num_dims = shape.size
    size_lin = 1
    for d in range(num_dims):
        size_lin *= shape_rel[d]

    out_idx = np.empty(num_dims, dtype=np.int64)
    in_idx = np.empty(num_dims, dtype=np.int64)

    current_idx = start_idx
    for idx_lin in range(size_lin):
        idx_rel = unravel_index_numba(idx_lin, shape_rel)

        # Compute absolute indices for output and input
        for d in range(num_dims):
            # out position modulo shape
            out_idx[d] = (out_pos[d] + idx_rel[d]) % shape[d]
            # in position modulo shape
            in_idx[d] = (in_pos[d] + idx_rel[d]) % shape[d]

        # Convert multidimensional indices to linear form
        row_indices[current_idx] = ravel_index_numba(shape, out_idx)
        col_indices[current_idx] = ravel_index_numba(shape, in_idx)
        current_idx += 1

    return current_idx

def generate_sparsity_pattern(shape, dependencies):
    """
    Generate row and column indices for the sparse matrix representation
    of a given stencil pattern. This function attempts to minimize loops
    over large dimensions by focusing on relative axes and using Numba-compiled
    helper functions for speed.

    Parameters:
    - shape: Tuple of ints, the shape of the multidimensional array.
    - dependencies: A list of tuples (out_pos, in_pos, fixed_axes, periodic_axes)
      representing the stencil pattern. Each dependency indicates how output positions
      relate to input positions and which axes are fixed or periodic.

    Returns:
    - row_indices: 1D numpy array of row indices for the sparse pattern.
    - col_indices: 1D numpy array of column indices for the sparse pattern.
    """
    shape = np.array(shape, dtype=np.int64)
    num_dims = len(shape)

    # Estimate total number of non-zero elements needed
    total_elements = 0
    for dep in dependencies:
        out_pos, in_pos, fixed_axes, periodic_axes = dep
        in_pos_arr = np.array(in_pos, dtype=np.int64)
        shape_rel = shape.copy()

        # shape_rel adjusted for non-fixed and periodic axes
        shape_rel -= np.abs(in_pos_arr)
        for d in periodic_axes:
            shape_rel[d] += abs(in_pos_arr[d])
        for d in fixed_axes:
            shape_rel[d] = 1
        total_elements += np.prod(shape_rel)

    # Preallocate arrays for row and col indices
    row_indices = np.empty(total_elements, dtype=np.int64)
    col_indices = np.empty(total_elements, dtype=np.int64)

    current_index = 0
    for dep in dependencies:
        out_pos, in_pos, fixed_axes, periodic_axes = dep
        out_pos_arr = np.array(out_pos, dtype=np.int64)
        in_pos_arr = np.array(in_pos, dtype=np.int64)
        shape_rel = shape.copy()

        # Adjust shape_rel for periodic and fixed axes
        shape_rel -= np.abs(in_pos_arr)
        for d in periodic_axes:
            shape_rel[d] += abs(in_pos_arr[d])
        is_fixed = np.zeros(num_dims, dtype=np.bool_)
        for d in fixed_axes:
            shape_rel[d] = 1
            is_fixed[d] = True

        # Adjust out_pos and in_pos so that one of them starts at zero for relative indexing
        # This reduces complexity when computing final positions.
        for d in range(num_dims):
            if not is_fixed[d]:
                if in_pos_arr[d] < 0:
                    out_pos_arr[d] = -in_pos_arr[d]
                    in_pos_arr[d] = 0
                else:
                    out_pos_arr[d] = 0

        # Fill row and col indices using the helper
        current_index = iterate_over_entries(shape, shape_rel, in_pos_arr, out_pos_arr,
                                             row_indices, col_indices, current_index)

    # Sort the indices by row and then by column for canonical form
    sorted_idx = np.unique(np.concatenate((col_indices.reshape((1, -1)), row_indices.reshape((1, -1))), axis=0), axis=1)
    col_indices = sorted_idx[0, :]
    row_indices = sorted_idx[1, :]
    
    return row_indices, col_indices

@njit
def group_columns_by_non_overlap_numba(indptr, indices):
    n = indptr.size - 1
    g = np.full(n, n, dtype=np.int64)
    groupnum = 0
    J = np.arange(n)
    while len(J) > 0:
        g[J[0]] = groupnum
        col = np.zeros(n, dtype=np.bool_)
        for i in range(indptr[J[0]], indptr[J[0]+1]):
            col[indices[i]] = True
        for k in J:
            if (not col[k]):
                for i in range(indptr[k], indptr[k+1]):
                    col[indices[i]] = True
                g[k] = groupnum
        J = np.where(g == n)[0]
        groupnum += 1
    return g, groupnum

def colgroup(*args, try_reorder=True):
    if isinstance(args[0], sparray):
        S = csc_array(args[0])
        T = csc_array((S.data != 0, S.indices, S.indptr), shape=S.shape)
    elif isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
        rows = args[0]
        cols = args[1]
        data = np.ones(rows.shape, dtype=np.bool_)
        T = csc_array((data, (rows, cols)))
    else:
        raise ValueError("Input should be a sparse array, or two ndarrays containing row and col indices")    

    TT = T.transpose() @ T
    g, num_groups = group_columns_by_non_overlap_numba(TT.indptr, TT.indices)
    
    if try_reorder:   
    # Form the reverse column minimum-degree ordering.
        p = reverse_cuthill_mckee(T)
        p = p[::-1]
        T = T[:, p]
        TT = T.transpose() @ T
        g2, num_groups2 = group_columns_by_non_overlap_numba(TT.indptr, TT.indices)
        # Use whichever packing required fewer groups.
        if num_groups2 < num_groups:
            q = np.argsort(p)
            g = g2[q]
            num_groups = num_groups2
    
    return g, num_groups
class NumJac:
    def __init__(self, shape = None, stencil = None, eps_jac = 1e-6):
        self.shape = shape
        self.eps_jac = eps_jac
        self.init_stencil(stencil)
    
    def init_stencil(self, stencil):
        # Handle direct stencil, string keys, or callable stencils
        if stencil is None:
            raise ValueError("Stencil must be provided as a pattern, function, or predefined key.")

        if isinstance(stencil, str):
            # Use a predefined stencil
            if stencil in stencils_registry:
                stencil = stencils_registry[stencil](self.shape)
            else:
                raise ValueError(f"Unknown stencil key: {stencil}")

        elif callable(stencil):
            # Generate stencil dynamically
            stencil = stencil(self.shape)
        dependencies = expand_dependencies(self.shape, stencil)
        self.rows, self.cols = generate_sparsity_pattern(self.shape, dependencies)
        self.gr, self.num_gr = colgroup(self.rows, self.cols)
        
    def __call__(self, f, c):
        f_value = f(c)
        dc = -self.eps_jac * np.abs(c)
        dc[dc > (-self.eps_jac)] = self.eps_jac
        dc = (c + dc) - c
        c_perturb = np.tile(c[np.newaxis, ...], (self.num_gr,) + (1,) * c.ndim)
        c_perturb.ravel()[c.size*self.gr.ravel() + np.arange(c.size)] += dc.ravel()
        dfdc = np.empty(c_perturb.shape)
        for k in range(self.num_gr):
            dfdc[k,...] = (f(c_perturb[k,...])-f_value) / dc
        values = dfdc.reshape((self.num_gr, -1))[self.gr.ravel()[self.cols], self.rows]
        jac = csc_array((values, (self.rows, self.cols)), shape = (f_value.size, c.size))
        return f_value, jac