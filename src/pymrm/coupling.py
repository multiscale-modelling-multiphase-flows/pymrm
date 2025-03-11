import math
import numpy as np
from scipy.sparse import csc_array
from pymrm.helpers import unwrap_bc_coeff

def translate_indices_to_larger_array(linear_indices, shape, new_shape, offset=None):
    """
    Translates linear indices from a subarray to their corresponding indices in a larger array.

    Parameters:
    - linear_indices (array-like): Linear indices in the subarray.
    - shape (tuple): Shape of the subarray.
    - new_shape (tuple): Shape of the larger ND array.
    - offset (tuple): Offset of the subarray's top-left corner in the larger array.

    Returns:
    - new_linear_indices (np.ndarray): Linear indices in the larger ND array.
    """

    # Convert linear indices to multi-indices based on the original subarray shape
    multi_indices = np.unravel_index(linear_indices, shape)

    # Shift multi-indices by the offset to their position in the larger array
    if offset is not None:
        adjusted_multi_indices = tuple(m + o for m, o in zip(multi_indices, offset))
    else:
        adjusted_multi_indices = multi_indices
    
    # Convert back to linear indices in the larger ND array
    
    new_linear_indices = np.ravel_multi_index(adjusted_multi_indices, new_shape)

    return new_linear_indices

def update_csc_array_indices(sparse_mat, shape, new_shape, offset=None):
    """
    Updates the row and column indices of a csc_array to match a new larger ND domain,
    while also updating the matrix shape.

    Parameters:
    - sparse_mat (csc_array): The input sparse matrix (in CSC format).
    - shape (tuple) or ((tuple,tuple)): Original ND shape of the subdomain.
    - new_shape (tuple) or ((tuple, tuple)): Target ND shape of the larger domain.
    - offset (tuple) or ((tuple, tuple)): Offset of the subdomain within the larger domain.

    Returns:
    - updated_mat (csc_array): The updated sparse matrix with modified indices, indptr, and correct shape.
    """

    # Extract matrix data and original indices
    data = sparse_mat.data
    row_indices = sparse_mat.indices  # Row indices (modifiable)
    col_pointers = sparse_mat.indptr  # Column pointers
    num_rows = sparse_mat.shape[0]  # Number of rows in the original matrix
    num_cols = sparse_mat.shape[1]  # Number of columns in the original matrix
    

    # Generate original linear row and column indices
    original_linear_rows = row_indices
    original_linear_cols = np.arange(num_cols)  # Columns as a sequential array

    # Translate row and column indices to the larger ND array
    shape = tuple(shape)
    if all(isinstance(dim, int) for dim in shape):
        shape = (shape,shape)
    new_shape = tuple(new_shape)
    if all(isinstance(dim, int) for dim in new_shape):
        new_shape = (new_shape,new_shape)
    if offset is None:
        offset = (None, None)
    else:
        offset = tuple(offset)
        if all(isinstance(dim, int) for dim in offset):
            offset = (offset,offset)
    
    new_row_indices = translate_indices_to_larger_array(original_linear_rows, shape[0], new_shape[0], offset[0])
    new_col_indices = translate_indices_to_larger_array(original_linear_cols, shape[1], new_shape[1], offset[1])

    # Compute new column pointers by shifting to account for empty columns
    num_rows = np.prod(new_shape[0])
    num_cols = math.prod(new_shape[1])
    new_col_pointers = np.zeros(num_cols + 1, dtype=int)
    new_col_pointers[new_col_indices + 1] = np.diff(col_pointers)

    # Convert counts to cumulative sum for proper CSC format
    new_col_pointers = np.cumsum(new_col_pointers)

    # Create a new sparse matrix with the corrected 2D shape
    updated_mat = csc_array((data, new_row_indices, new_col_pointers), shape=(num_rows, num_cols))

    return updated_mat

def construct_interface_matrices(shapes, x_f, x_c=(None,None), ic=({'a0':1, 'a1':1, 'b0':0, 'b1':0}, {'a0':0, 'a1':0, 'b0':1, 'b1':1}), axis=0):
    """
    Construct a sparse matrix representing the interface between two subdomains.

    Parameters:
    - shapes (tuple): Shapes of the two subdomains.
    - x_fs (tuple): Face-centered grid points for the two subdomains.
    - x_cs (tuple, optional): Cell-centered grid points for the two subdomains.
    - ic (tuple, optional): Interface conditions between the two subdomains.
    - axis (int): Axis along which the interface is constructed.

    Returns:
    - two interface_matrices (csc_array): Sparse matrices for constructing interface values
    """

    shape = (min(u,v) for u,v in zip(shapes[0], shapes[1]))
    shape = tuple(s1+s2 if i == axis else min(s1, s2) for i, (s1, s2) in enumerate(zip(shapes[0], shapes[1])))
    shape_i = tuple(1 if i == axis else s for i, s in enumerate(shape))

    # Extract the cell-centered grid points for the two subdomains
    for i in range(2):
        if x_c[i] is None:
            x_c[i] = 0.5 * (x_f[0][1:] + x_f[0][:-1])

    a0, a1, b0, b1 = [[unwrap_bc_coeff(shape, ic_elem[key], axis=axis) if ic_elem else np.zeros((1,)*len(shape)) for bc_element in ic] for key in ['a0', 'a1', 'b0', 'b1']]
    
    alpha_1 = [None, None]
    alpha_1[0] = -(x_c[0][-2] - x_f[0][-1]) / (
        (x_c[0][-1] - x_f[0][-1]) * (x_c[0][-2] - x_c[0][-1]))
    alpha_1[1] = (x_c[1][1] - x_f[1][0]) / (
        (x_c[1][0] - x_f[1][0]) * (x_c[1][1] - x_c[1][0]))
    alpha_2 = [None, None]
    alpha_2[0] = -(x_c[0][-1] - x_f[0][-1]) / (
        (x_c[0][-2] - x_f[0][-1]) * (x_c[0][-2] - x_c[0][-1]))
    alpha_2[1] = (x_c[1][0] - x_f[1][0]) / (
        (x_c[1][1] - x_f[1][0]) * (x_c[1][1] - x_c[1][0]))
    alpha_0 = [alpha_1[0]-alpha_2[0], alpha_1[1]-alpha_2[1]]
    
    



    # Compute the number of cells in each subdomain
    num_cells1 = math.prod(shapes[0])
    num_cells2 = math.prod(shapes[1])

    # Compute the number of faces in each subdomain
    num_faces1 = num_cells1 + 1
    num_faces2 = num_cells2 + 1

    # Compute the number of faces in the interface
    num_faces_interface = 2

    # Compute the number of cells in the interface
    num_cells_interface = 1

    # Compute the total number of faces
    num_faces_total = num_faces1 + num_faces2 + num_faces_interface

    # Compute the total number of cells
    num_cells_total = num_cells1 + num_cells2 + num_cells_interface

    # Compute the total number of grid points
    num_points_total = num_faces_total + num_cells_total

    # Initialize the data, row, and column indices for the sparse matrix
    data = np.ones(num_faces_interface)
    row_indices = np.array([num_faces1, num_faces1 + 1])
    col_indices = np.array([ic1, ic2])

    # Create the sparse matrix representing the interface
    interface_matrix = csc_array((data, row_indices, col_indices), shape=(num_points_total, num_points_total))

    return interface_matrix