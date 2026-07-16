"""Offline collision-classifier test: publish images from a folder and classify.

No camera or robot required. Plays the bundled sample images (or any folder you
point it at) onto an image topic and runs the classifier on them, so you can
verify a .pte model loads and produces free/blocked scores:

    ros2 launch collision_classifier dataset_classifier.launch.py backend:=cpu
    ros2 topic echo /collision/classification
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    backend_arg = DeclareLaunchArgument(
        "backend", default_value="cpu",
        description="Inference backend: 'cpu' (XNNPACK) or 'npu' (QNN HTP)")

    model_path_arg = DeclareLaunchArgument(
        "model_path", default_value="",
        description="Path to .pte model file; empty selects the default for the backend")

    dataset_path_arg = DeclareLaunchArgument(
        "dataset_path", default_value="datasets/sample_images",
        description="Directory or image file to publish as classifier input")

    image_topic_arg = DeclareLaunchArgument(
        "image_topic", default_value="/oak/rgb/image_raw",
        description="Image topic shared by dataset publisher and classifier")

    publish_rate_arg = DeclareLaunchArgument(
        "publish_rate", default_value="5.0",
        description="Dataset playback rate in Hz")

    qnn_lib_dir_arg = DeclareLaunchArgument(
        "qnn_lib_dir",
        default_value=PathJoinSubstitution([
            EnvironmentVariable("QAIRT_LIB"), "aarch64-oe-linux-gcc11.2"]),
        description="Directory containing QNN runtime libraries; defaults to "
                    "$QAIRT_LIB/aarch64-oe-linux-gcc11.2")

    params_file = PathJoinSubstitution(
        [FindPackageShare("collision_classifier"), "config", "classifier.yaml"])

    dataset_node = Node(
        package="collision_classifier",
        executable="dataset_image_publisher_node",
        name="dataset_image_publisher",
        parameters=[
            params_file,
            {
                "dataset_path": LaunchConfiguration("dataset_path"),
                "image_topic": LaunchConfiguration("image_topic"),
                "publish_rate": LaunchConfiguration("publish_rate"),
            },
        ],
        output="screen",
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

    return LaunchDescription([
        backend_arg,
        model_path_arg,
        dataset_path_arg,
        image_topic_arg,
        publish_rate_arg,
        qnn_lib_dir_arg,
        dataset_node,
        classifier_node,
    ])
