import pytest

from alitra.models import Transform, Translation


def test_transform(default_rotation, robot_frame, asset_frame):
    with pytest.raises(ValueError):
        translation = Translation(1, 1, from_=robot_frame, to_=asset_frame)
        Transform(
            translation, rotation=default_rotation, from_=asset_frame, to_=robot_frame
        )


def test_transform_without_rotation(robot_frame, asset_frame):
    with pytest.raises(ValueError):
        translation = Translation(1, 1, from_=robot_frame, to_=asset_frame)
        Transform(translation=translation, from_=asset_frame, to_=robot_frame)
