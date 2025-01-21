# PyMRM Dependency Notation

## Overview

The **PyMRM Dependency Notation** provides a structured and flexible way to describe dependencies in operations involving discretized fields, such as those used in the **PyMRM simulation package**. These fields are represented as multidimensional arrays, which could model, for example, concentrations of multiple species in a 2D or 3D system.

### Key Concepts
- A **field function** maps an input field (e.g. concentrations) to an output field (e.g. reaction rates). Think of such as a a reaction rate calculation or a Laplace operator.
- A **dependency notation** specifies how values at one position in the output field depend on input values at other positions.
- Dependencies are particularly relevant for:
  - **Spatial axes**: Often involve nearest-neighbor interactions (e.g., stencils).
  - **Non-spatial axes**: Often involve absolute dependencies between indices (e.g., reaction rates between species).

This notation is primarily used internally in PyMRM to construct efficient sparsity patterns for Jacobians. Advanced users can also leverage it to define custom field operations.

---

## Dependency Notation

A single dependency is represented as a **tuple of four elements**:

```text
(in_index, out_index, fixed_axes_list, periodic_axes_list)
```

### Components

1. **`in_index`**:
   - A tuple representing the position in the input (field) array that contributes to the value at `out_index`.
   - 
2. **`out_index`**:
   - A tuple representing the position in the output array (i.e. discretized output field).
   - Only relevant for axes listed in the `fixed_axes_list`.
   - Can be set to `None` if all axes are spatial (no fixed axes).

3. **`fixed_axes_list`**:
   - A list of axes that are considered "fixed" (non-spatial), such as axes representing species or phases.
   - These axes require an explicit reference in `out_index`.

4. **`periodic_axes_list`**:
   - A list of spatial axes that are considered periodic, where values near one boundary depend on values near the opposite boundary.

---

### Key Rules
- **Spatial axes**: Dependencies on spatial axes often follow stencils (e.g., nearest neighbors).
- **Fixed axes**: For the indices along fixed axis the value of the indices are abolute. That is, the value at `out_index` depends on tha value at `in_index`. 
- **Relative axes**: For axes not listed in the `fixed_axes_list`, the values in `out_index` are ignored. In this case the value of `in_index` along this axis should be interpreted relative. This means the value at index `i` depends on the value at `i+in_index`
- If `fixed_axes_list` is empty, all axis are relative `out_index` can be `None`.

---

### Examples

#### Simple Stencils (Spatial Dependencies Only)
1. `((0, 0, 0), None, [], [])`: The value at `(i, j, k)` depends on the field value at `(i, j, k)`.
2. `((0, 1, 0), None, [], [])`: The value at `(i, j, k)` depends on the field value at `(i, j+1, k)`.
3. `((-1, 1, 0), None, [], [])`: The value at `(i, j, k)` depends on the field value at `(i-1, j+1, k)`.

#### Fixed Axes (Non-Spatial Dependencies)
1. `((-1, 1, 0), (0, 0, 0), [2], [])`: The value at `(i, j, 0)` depends on the field value at `(i-1, j+1, 0)`.
2. `((-1, 1, 1), (0, 0, 0), [2], [])`: The value at `(i, j, 0)` depends on the field value at `(i-1, j+1, 1)`.
3. `((-1, 1, 0), (0, 0, 1), [2], [])`: The value at `(i, j, 1)` depends on the field value at `(i-1, j+1, 0)`.

---

## Shorthand Notation

To simplify commonly used patterns, the dependency notation supports the following shorthand conventions:

### 1. **Simple Stencils**
For stencils with no `fixed_axes_list` or `periodic_axes_list`, the structure can be reduced to a single tuple:

- `((0, 0, 0), None, [], [])` → `(0, 0, 0)`
- `((0, 1, 0), None, [], [])` → `(0, 1, 0)`
- `((-1, 1, 0), None, [], [])` → `(-1, 1, 0)`

### 2. **No Periodic Axes**
If `periodic_axes_list` is empty, it can be omitted:

- `((0, 0, 1), (0, 0, 0), [2], [])` → `((0, 0, 1), (0, 0, 0), [2])`

---

## List of Dependencies

A full sparsity pattern is described as a **list of single dependencies**. For example:

1. **1D Stencil** (Three-Point Stencil):
   - `[(-1, 0), (0, 0), (1, 0)]`

2. **3D Stencil** (Seven-Point Stencil):
   - `[(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]`

### Compact Notation for Ranges
Ranges and lists can be expressed using slices or nested lists for convenience:

- `[(-1, 0), (0, 0), (1, 0)]` → `[([-1, 0, 1], 0)]`
- Seven-Point Stencil:
  - `[(0, 0, 0), (-1, 0, 0), ..., (0, 0, 1)]` → `[([-1, 0, 1], 0, 0), (0, [-1, 0, 1], 0), (0, 0, [-1, 0, 1])]`

### Example: Multi-Component Diffusion
In a multi-component diffusion problem, the function value for one species depends on all species in neighboring cells:
```python
dependencies = (([-1, 0, 1], slice(None)), (0, slice(None)), [1])
```

---

## Conclusion

The **PyMRM Dependency Notation** provides a consistent and flexible way to describe dependencies in discretized field operations. While it is primarily used internally, it offers advanced users the ability to define custom operations with precise control over sparsity patterns. This capability is crucial for optimizing Jacobian calculations and other computationally intensive tasks.