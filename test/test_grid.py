import pytest
import numpy as np
from pymrm import non_uniform_grid, generate_grid

def test_non_uniform_grid():
    grid = non_uniform_grid(0, 1, 10, 0.1, 0.75)
    assert np.all(np.diff(grid) > 0)
    assert grid[0] == 0
    assert grid[-1] == 1

def test_generate_grid():
    faces, centers = generate_grid(10, generate_x_c=True)
    assert len(faces) == 11
    assert len(centers) == 10
