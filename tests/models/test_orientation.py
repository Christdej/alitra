import numpy as np
import pytest

from alitra.models import Euler, Orientation


def test_orientation_array(robot_frame):
    expected_array = np.array([1, 1, 1, 1], dtype=float)
    orientation: Orientation = Orientation.from_array(expected_array, robot_frame)
    assert np.allclose(orientation.to_array(), expected_array)


def test_orientation_euler(robot_frame):
    expected_euler = Euler(psi=1, theta=1, phi=1, frame=robot_frame)
    orientation: Orientation = Orientation.from_euler(expected_euler, robot_frame)
    assert np.allclose(orientation.to_euler_array(), expected_euler.to_array())


def test_orientation_invalid_array(robot_frame):
    with pytest.raises(ValueError):
        Orientation.from_array(np.array([1, 1]), frame=robot_frame)
