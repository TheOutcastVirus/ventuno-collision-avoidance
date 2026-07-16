"""Collect free/blocked training images from the OAK camera.

Brings up the OAK-D camera and the keyboard-driven data_collection node. Focus
the terminal and press:
    f  save the current frame as FREE
    b  save the current frame as BLOCKED
    q  quit

Images are written to <output_dir>/{free,blocked}/<uuid>.jpg (default
output_dir=dataset), ready for tools/train_collision_resnet18.py.

    ros2 launch collision_avoider data_collection.launch.py
    ros2 launch collision_avoider data_collection.launch.py output_dir:=/data/run1
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    image_topic_arg = DeclareLaunchArgument(
        "image_topic", default_value="/oak/rgb/image_raw",
        description="RGB image topic to snapshot")

    output_dir_arg = DeclareLaunchArgument(
        "output_dir", default_value="dataset",
        description="Directory to write free/ and blocked/ image folders into")

    image_size_arg = DeclareLaunchArgument(
        "image_size", default_value="224",
        description="Saved image size (square), matching the classifier input")

    enable_camera_arg = DeclareLaunchArgument(
        "enable_camera", default_value="true",
        description="Also start the OAK camera. Set false if it is already running")

    enable_collector_arg = DeclareLaunchArgument(
        "enable_collector", default_value="true",
        description="Start the keyboard collector node. Set false to bring up only the "
                    "camera, then run `ros2 run collision_avoider data_collection` in a "
                    "terminal (needed if launch does not forward an interactive stdin)")

    camera_params = PathJoinSubstitution(
        [FindPackageShare("oak_camera"), "config", "camera.yaml"])

    camera_node = Node(
        package="oak_camera",
        executable="oak_camera_node",
        name="oak_camera",
        parameters=[camera_params],
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_camera")),
    )

    collector_node = Node(
        package="collision_avoider",
        executable="data_collection",
        name="data_collector",
        parameters=[{
            "image_topic": LaunchConfiguration("image_topic"),
            "output_dir": LaunchConfiguration("output_dir"),
            "image_size": ParameterValue(
                LaunchConfiguration("image_size"), value_type=int),
        }],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("enable_collector")),
    )

    return LaunchDescription([
        image_topic_arg,
        output_dir_arg,
        image_size_arg,
        enable_camera_arg,
        enable_collector_arg,
        camera_node,
        collector_node,
    ])
