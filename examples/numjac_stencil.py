"""
This module provides functionality for expanding dependencies and generating sparsity patterns for operations on multidimensional arrays.

Notation for Sparsity Patterns:
--------------------------------

The sparsity pattern notation allows you to specify dependencies between elements in a multidimensional array. The notation supports shorthand and full formats for specifying these dependencies.

1. **Basic Notation**:
   - Dependencies can be defined using tuples, lists of tuples, or ranges.
   - Each dependency is specified as a tuple, which can have different formats based on the type of dependency.

2. **Dependency Formats**:
   a. **Stencil Shorthand**:
      - A list of tuples specifying relative positions.
      - Example: [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]
      - Indicates dependencies on the previous, current, and next elements along the first axis.

   b. **Block Dependency**:
      - A tuple of the form ((i, j, k), [(i, j, k), ...], fixed_axis), where the first element is a reference index, the second element is a list of dependent indices, and `fixed_axis` specifies the axis along which the dependency is fixed.
      - Example: ((0, 0, 1), [(0, 0, 0), (0, 0, 2)], 2)
      - Indicates that the element at (0, 0, 1) depends on elements at (0, 0, 0) and (0, 0, 2) along the third axis.

   c. **Range Dependencies**:
      - A tuple of the form ((i, j, range), [(i, j, range)], fixed_axis), where `range` specifies a range of indices.
      - Example: ((0, 0, range(0, 3)), [(0, 0, [0, 1, 2])], 2)
      - Indicates that the element at (0, 0, range(0, 3)) depends on elements at (0, 0, 0), (0, 0, 1), and (0, 0, 2) along the third axis.

   d. **All Components Dependencies**:
      - A tuple of the form ((i, j, slice(None)), [(i, j, slice(None))], fixed_axis), where `slice(None)` specifies all indices along that axis.
      - Example: ((0, 0, slice(None)), [(0, 0, slice(None))], 2)
      - Indicates dependencies on all components along the third axis.

3. **Mixed Dependencies**:
   - Mixed dependencies can include various forms of shorthand and full formats combined in a list.
   - Example: [
       [(-1, 0, 0), (0, 0, 0), (1, 0, 0)],  # Stencil shorthand
       ((0, 0, 1), [(0, 0, 0), (0, 0, 2)], 2),  # Block dependency
       ((0, 0, [0, 2]), [(0, 0, [0, 1, 2])], 2),  # Range dependencies
       ((0, 0, slice(None)), [(0, 0, slice(None))], 2)  # All components dependencies
     ]

Functionality:
--------------
1. `is_tuple_or_list_of_tuples(variable)`:
   - Checks if a variable is a tuple or a list of tuples.

2. `slice_to_list(slice_obj: slice, axis_length: int) -> List[int]`:
   - Converts a slice object to a list of indices.

3. `expand_dependencies(shape: Tuple[int, ...], dependencies: Union[List[Union[Tuple, List]], Tuple]) -> List[Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]], Union[int, Tuple[int, ...]]]]`:
   - Expands shorthand notations in dependencies to full format.

4. `expand_indices(indices: Union[List, Tuple], shape: Tuple[int, ...], fixed_axis: Union[int, Tuple[int, ...], None] = None) -> List[Tuple[int, ...]]`:
   - Expands indices with lists and slices to full indices.

5. `product(*args)`:
   - Computes the Cartesian product of input iterables.

Example Usage:
--------------
```python
shape = (100, 200, 6)
dependencies = [
    [(-1, 0, 0), (0, 0, 0), (1, 0, 0)],  # Stencil shorthand
    ((0, 0, 1), [(0, 0, 0), (0, 0, 2)], 2),  # Block dependency
    ((0, 0, [0, 2]), [(0, 0, [0, 1, 2])], 2),  # Range dependencies
    ((0, 0, slice(None)), [(0, 0, slice(None))], 2)  # All components dependencies
]

expanded_dependencies = expand_dependencies(shape, dependencies)
print(expanded_dependencies)
"""

import numpy as np
from scipy.sparse import csc_array, csr_array, lil_array, sparray
from scipy.sparse.csgraph import reverse_cuthill_mckee
from numba import njit, jit
import numpy as np
from typing import List, Tuple, Union

def first_fit_packing(T):
    m, n = T.shape
    TT = T.transpose() @ T 
    g = np.full(n, n, dtype=int)
    groupnum = 0
    J = np.arange(n)
    while len(J) > 0:
        g[J[0]] = groupnum
        col = TT[:, [J[0]]].toarray().ravel()
        for k in J:
            if (not col[k]):
                col |= TT[:, [k]].toarray().ravel()
                g[k] = groupnum
        J = np.where(g == n)[0]
        groupnum += 1
    return g, groupnum

def colgroup(*args):
    if isinstance(args[0], sparray):
        S = csr_array(args[0])
        T = csr_array((S.data != 0, S.indices, S.indptr), shape=S.shape)
    elif isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
        rows = args[0]
        cols = args[1]
        data = np.ones(rows.shape, dtype=bool)
        T = csr_array((data, (rows, cols)))
    else:
        raise ValueError("Input should be a sparse array, or two ndarrays containing row and col indices")    

    g, num_groups = first_fit_packing(T)
    
    # Form the reverse column minimum-degree ordering.
    p = reverse_cuthill_mckee(T)
    p = p[::-1]
    T = T[:, p]
    g2, num_groups2 = first_fit_packing(T)
    
    # Use whichever packing required fewer groups.
    if num_groups <= num_groups2:
        gout = g
    else:
        q = np.argsort(p)
        gout = g2[q]
        num_groups = num_groups2
    return gout, num_groups

def calculate_max_elements(dependencies, shapes_blocked):
    max_elements = 0
    for i in range(len(dependencies)):
        shape_blocked = shapes_blocked[i]
        size = 1
        for dim in shape_blocked:
            size *= dim
        max_elements += size * len(dependencies[i][0]) * len(dependencies[i][1])
    return max_elements


@njit
def flat_index_numba(shape, index):
    """Convert a multidimensional index to a flat index."""
    flat_idx = 0
    for i in range(len(shape)):
        flat_idx = flat_idx * shape[i] + index[i]
    return flat_idx

@njit
def is_within_bounds(index, shape):
    for i in range(len(shape)):
        if not (0 <= index[i] < shape[i]):
            return False
    return True

@njit
def process_dependency(shape, shape_blocked, relative_positions, dependents, rows, cols, start_count):
    count = start_count
    for idx in np.ndindex(shape_blocked):
        flat_idx = flat_index_numba(shape, idx)
        for rel_pos in relative_positions:
            row_idx = np.array(idx) + np.array(rel_pos)
            if is_within_bounds(row_idx, shape):
                flat_row_idx = flat_index_numba(shape, row_idx)
                for dep_idx in dependents:
                    col_idx = np.array(idx) + np.array(dep_idx)
                    if is_within_bounds(col_idx, shape):
                        flat_col_idx = flat_index_numba(shape, col_idx)
                        rows[count] = flat_row_idx
                        cols[count] = flat_col_idx
                        count += 1
    return count

def construct_sparse_matrix_pattern(shape, dependencies):
    shapes_blocked = create_shapes_blocked(shape, dependencies)
    max_elements = calculate_max_elements(dependencies, shapes_blocked)
    rows = np.empty(max_elements, dtype=np.int64)
    cols = np.empty(max_elements, dtype=np.int64)
    
    count = 0
    for i in range(len(dependencies)):
        relative_positions = dependencies[i][0]
        dependents = dependencies[i][1]
        shape_blocked = shapes_blocked[i]
        # Call the Numba-compiled function
        count = process_dependency(shape, shape_blocked, relative_positions, dependents, rows, cols, count)
    
    rows = rows[:count]
    cols = cols[:count]
    return rows, cols

def is_tuple_or_list_of_tuples(variable):
    """
    Check if the variable is a tuple or a list of tuples.
    
    Args:
        variable: The variable to check.
        
    Returns:
        bool: True if the variable is a tuple or a list of tuples, False otherwise.
    """
    if isinstance(variable, tuple):
        return True
    elif isinstance(variable, list) and all(isinstance(item, tuple) for item in variable):
        return True
    return False

def slice_to_list(slice_obj, axis_length):
    """
    Convert a slice object to a list of indices.
    
    Args:
        slice_obj (slice): The slice object.
        axis_length (int): The length of the axis.
        
    Returns:
        List[int]: The list of indices.
    """
    return list(range(*slice_obj.indices(axis_length)))

def expand_dependencies(shape, dependencies):
    """
    Convert shorthand notations in dependencies to full format.
    
    Args:
        shape (Tuple[int, ...]): The shape of the array.
        dependencies (Union[List[Union[Tuple, List]], Tuple]): The dependencies to expand.
        
    Returns:
        List[Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]], Union[int, Tuple[int, ...]]]]: The expanded dependencies.
    """
    expanded_dependencies = []
    if not isinstance(dependencies, list):
        dependencies = [dependencies]
    
    for dep in dependencies:
        if isinstance(dep, tuple) and len(dep) == 3 and is_tuple_or_list_of_tuples(dep[0]) and is_tuple_or_list_of_tuples(dep[1]):
            reference, dependents, fixed_axis = dep
            if isinstance(fixed_axis, int):
                fixed_axis = [fixed_axis]
            elif isinstance(fixed_axis, slice):
                fixed_axis = slice_to_list(fixed_axis, len(shape))
            elif isinstance(fixed_axis, range):
                fixed_axis = list(fixed_axis)
            shape_blocked = tuple(1 if i in fixed_axis else shape[i] for i in range(len(shape)))            
            expanded_reference = expand_indices(reference, shape)
            expanded_dependents = expand_indices(dependents, shape)
        elif isinstance(dep, tuple) and len(dep) == 2 and is_tuple_or_list_of_tuples(dep[0]) and is_tuple_or_list_of_tuples(dep[1]):
            fixed_axis = [-1]
            shape_blocked = shape
            reference, dependents = dep
            expanded_reference = expand_indices(reference, shape)
            expanded_dependents = expand_indices(dependents, shape)
        elif isinstance(dep, tuple) and len(dep) == 1 and is_tuple_or_list_of_tuples(dep[0]) and is_tuple_or_list_of_tuples(dep[1]):
            expanded_reference = [(0, 0, 0)]
            fixed_axis = [-1]
            shape_blocked = shape
            expanded_dependents = expand_indices(dep[0], shape)
        elif isinstance(dep, tuple) and len(dep) == len(shape):
            expanded_reference = [(0, 0, 0)]
            fixed_axis = [-1]
            shape_blocked = shape
            expanded_dependents = expand_indices(dep, shape)
        elif is_tuple_or_list_of_tuples(dep):
            expanded_reference = [(0, 0, 0)]
            fixed_axis = [-1]
            shape_blocked = shape
            expanded_dependents = expand_indices(dep, shape)
        else:
            raise ValueError("Invalid dependency format.")
        expanded_dependencies.append((expanded_reference, expanded_dependents, fixed_axis))
    return expanded_dependencies

def create_shapes_blocked(shape, dependencies):
    shapes_blocked = []
    for dep in dependencies:
        _, _, fixed_axis = dep
        shape_blocked = tuple(1 if i in fixed_axis else dim for i, dim in enumerate(shape))
        shapes_blocked.append(shape_blocked)
    return shapes_blocked

def expand_indices(indices: Union[List, Tuple], shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    Expand indices with lists and slices to full indices.
    
    Args:
        indices (Union[List, Tuple]): The indices to expand.
        shape (Tuple[int, ...]): The shape of the array.
        
    Returns:
        List[Tuple[int, ...]]: The expanded indices.
    """
    expanded_indices = []
    if isinstance(indices, tuple):
        indices = [indices]

    def expand_index(index, axis_length):
        if isinstance(index, slice):
            return slice_to_list(index, axis_length)
        if isinstance(index, range):
            return list(index)
        elif isinstance(index, list):
            expanded_list = []
            for item in index:
                expanded_list.extend(expand_index(item, axis_length))
            return expanded_list
        else:
            return [index]
        
    for index in indices:
        axis_expansion = []
        for i in range(len(shape)):
            expanded_axis_indices = expand_index(index[i], shape[i])
            axis_expansion.append(expanded_axis_indices)
        expanded_indices.extend([tuple(x) for x in product(*axis_expansion)])
    return expanded_indices

def product(*args):
    """
    Cartesian product of input iterables.
    
    Args:
        *args: The input iterables.
        
    Yields:
        Tuple: The next product tuple.
    """
    pools = [tuple(pool) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def axis_is_touched(dependencies):
    axis_touched = np.zeros(len(dependencies[0][0][0]), dtype=bool)
    for dep in dependencies:
        reference, dependent, _ = dep
        for ref in reference:
            axis_touched |= (np.array(ref) != 0)
        for dep2 in dependent:
            axis_touched |= (np.array(dep2) != 0)                           
    return tuple(axis_touched)

class NumJac:
    def __init__(self, shape, stencil):
        self.shape = shape
        dependencies = expand_dependencies(shape, stencil)
        shape_filt = tuple(np.array(axis_is_touched(dependencies),dtype = np.int64) * (np.array(shape, dtype = np.int64)-1) + np.ones(len(shape),dtype = np.int64))
        rows, cols = construct_sparse_matrix_pattern(shape_filt, dependencies)
        gr, self.num_gr = colgroup(rows, cols)
        self.gr = np.broadcast_to(gr.reshape(shape_filt), shape)
        rows, cols = construct_sparse_matrix_pattern(shape, dependencies)
        sorted_indices = np.lexsort((rows, cols))
        rows = rows[sorted_indices]
        cols = cols[sorted_indices]        
        self.rows = rows
        self.cols = cols
        self.eps_jac = 1e-6
        
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