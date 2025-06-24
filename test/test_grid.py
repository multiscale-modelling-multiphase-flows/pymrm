import pytest
import numpy as np
from pymrm.grid import non_uniform_grid, generate_grid

def test_non_uniform_grid():
    grid = non_uniform_grid(0, 1, 10, 0.1, 0.75)
    assert np.all(np.diff(grid) > 0)
    assert grid[0] == 0
    assert np.isclose(grid[-1], 1)

def test_generate_grid():
    size = 10
    x_f = np.linspace(0, 1, size + 1)
    x_f_out, x_c_out = generate_grid(size, x_f, generate_x_c=True)
    assert len(x_f_out) == size + 1
    assert len(x_c_out) == size
