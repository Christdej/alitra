import numpy as np
import pytest

from alitra.convert import euler_to_quaternion, quaternion_to_euler
from alitra.models import Euler, Frame, Quaternion


@pytest.mark.parametrize(
    "euler, expected",
    [
        (
            Euler(psi=0, phi=0, theta=0, frame=Frame("robot")),
            Quaternion(x=0, y=0, z=0, w=1, frame=Frame("robot")),
        ),
        (
            Euler(
                psi=1.5707963,  # Z
                theta=0.5235988,  # Y
                phi=1.0471976,  # X
                frame=Frame(
                    "robot",
                ),
            ),
            Quaternion(x=0.1830127, y=0.5, z=0.5, w=0.6830127, frame=Frame("robot")),
        ),
    ],
)
def test_euler_to_quaternion(euler, expected):
    quaternion: Quaternion = euler_to_quaternion(euler=euler, seq="ZYX", degrees=False)

    assert np.allclose(quaternion.to_array(), expected.to_array())


@pytest.mark.parametrize(
    "quaternion, expected",
    [
        (
            Quaternion(x=0, y=0, z=0, w=1, frame=Frame("robot")),
            Euler(psi=0, phi=0, theta=0, frame=Frame("robot")),
        ),
        (
            Quaternion(x=0.1830127, y=0.5, z=0.5, w=0.6830127, frame=Frame("robot")),
            Euler(
                psi=1.5707963,  # Z
                theta=0.5235988,  # Y
                phi=1.0471976,  # X
                frame=Frame(
                    "robot",
                ),
            ),
        ),
    ],
)
def test_quaternion_to_euler(quaternion, expected):
    euler_angles: Euler = quaternion_to_euler(
        quaternion=quaternion, sequence="ZYX", degrees=False
    )
    assert np.allclose(euler_angles.to_array(), expected.to_array())
