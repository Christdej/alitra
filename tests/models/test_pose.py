from alitra.models import Pose


def test_pose(default_position, default_orientation, robot_frame):
    pose: Pose = Pose(
        position=default_position, orientation=default_orientation, frame=robot_frame
    )
    assert pose.frame == robot_frame
