import pytest

from alitra.models import Euler, Transform, Translation


def test_transform(robot_frame, asset_frame):
    with pytest.raises(ValueError):
        euler = Euler(frame=robot_frame)
        translation = Translation(1, 1, from_=robot_frame, to_=asset_frame)
        Transform.from_euler(
            translation, euler=euler, from_=asset_frame, to_=robot_frame
        )


def test_transform_without_rotation(robot_frame, asset_frame):
    with pytest.raises(ValueError):
        translation = Translation(1, 1, from_=robot_frame, to_=asset_frame)
        Transform(translation=translation, from_=asset_frame, to_=robot_frame)
