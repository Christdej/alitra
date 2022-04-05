import numpy as np
import pytest

from alitra.models import Quaternion


def test_quaternion_array(robot_frame):
    expected_array = np.array([1, 1, 1, 1])
    quat = Quaternion.from_array(expected_array, frame=robot_frame)
    assert np.allclose(expected_array, quat.to_array())


def test_quaternion(robot_frame):
    with pytest.raises(ValueError):
        Quaternion.from_array(np.array([1, 2, 3]), frame=robot_frame)
