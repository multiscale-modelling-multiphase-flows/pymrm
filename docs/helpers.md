# Helper utilities

The `pymrm.helpers` module provides small building blocks that support
several higher-level routines in *pymrm*. It includes functions for
preparing boundary-condition coefficients and for creating sparse
diagonal matrices used in finite volume discretisations.

- `unwrap_bc_coeff` expands boundary-condition coefficient arrays to match a
  specified domain shape.
- `construct_coefficient_matrix` creates a sparse diagonal matrix from a set
  of coefficient values, optionally broadcasting them to a multi-dimensional
  field.

See the docstrings of these functions for detailed parameter descriptions
and usage examples.

