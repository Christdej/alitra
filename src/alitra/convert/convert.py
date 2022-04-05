from scipy.spatial.transform.rotation import Rotation

from ..models.euler import Euler
from ..models.frame import Frame
from ..models.quaternion import Quaternion


def quaternion_to_euler(
    quaternion: Quaternion, sequence: str = "ZYX", degrees: bool = False
) -> Euler:
    """
    Transform a quaternion into Euler angles.
    :param quaternion: A Quaternion object.
    :param sequence: Rotation sequence for the Euler angles.
    :param degrees: Set to true if the resulting Euler angles should be in degrees. Default is radians.
    :return: Euler object.
    """
    rotation_object: Rotation = Rotation.from_quat(quaternion.to_array())
    euler: Euler = Euler.from_array(
        rotation_object.as_euler(sequence, degrees=degrees), frame=Frame("robot")
    )
    return euler


def euler_to_quaternion(
    euler: Euler, seq: str = "ZYX", degrees: bool = False
) -> Quaternion:
    """
    Transform a quaternion into Euler angles.
    :param euler: An Euler object.
    :param seq: Rotation sequence for the Euler angles.
    :param degrees: Set to true if the provided Euler angles are in degrees. Default is radians.
    :return: Quaternion object.
    """
    rotation_object: Rotation = Rotation.from_euler(
        seq=seq, angles=euler.to_array(), degrees=degrees
    )
    quaternion: Quaternion = Quaternion.from_array(
        rotation_object.as_quat(), frame=Frame("robot")
    )
    return quaternion
