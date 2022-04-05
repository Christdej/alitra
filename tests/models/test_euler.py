import numpy as np
import pytest

from alitra.models import Euler


def test_euler_array(robot_frame):
    expected_array = np.array([0, 0, 0])
    euler: Euler = Euler.from_array(expected_array, frame=robot_frame)
    assert np.allclose(expected_array, euler.to_array())


def test_euler_invalid_array(robot_frame):
    with pytest.raises(ValueError):
        Euler.from_array(np.array([1, 1]), frame=robot_frame)
