"""Run the collision-avoidance classifier node on its own.

Subscribes to an RGB image topic and publishes free/blocked probabilities on
/collision/classification. Pair it with a camera (oak_camera) for a live feed,
or with dataset_classifier.launch.py for an offline image-folder test.
"""

from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription


def generate_launch_description():
    backend_arg = DeclareLaunchArgument(
        "backend",
        default_value="npu",
        description="Inference backend: 'cpu' (XNNPACK) or 'npu' (QNN HTP)",
    )

    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="",
        description="Path to .pte model file; empty selects the default for the backend",
    )

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/oak/rgb/image_raw",
        description="RGB image topic to classify",
    )

    qnn_lib_dir_arg = DeclareLaunchArgument(
        "qnn_lib_dir",
        default_value=PathJoinSubstitution([
            EnvironmentVariable("QAIRT_LIB"), "aarch64-oe-linux-gcc11.2"]),
        description="Directory containing QNN runtime libraries; defaults to "
                    "$QAIRT_LIB/aarch64-oe-linux-gcc11.2",
    )

    params_file = PathJoinSubstitution(
        [FindPackageShare("collision_classifier"), "config", "classifier.yaml"]
    )

    classifier_node = Node(
        package="collision_classifier",
        executable="collision_classifier_node",
        name="collision_classifier",
        parameters=[
            params_file,
            {
                "backend": LaunchConfiguration("backend"),
                "model_path": LaunchConfiguration("model_path"),
                "qnn_lib_dir": LaunchConfiguration("qnn_lib_dir"),
                "image_topic": LaunchConfiguration("image_topic"),
            },
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            backend_arg,
            model_path_arg,
            image_topic_arg,
            qnn_lib_dir_arg,
            classifier_node,
        ]
    )
