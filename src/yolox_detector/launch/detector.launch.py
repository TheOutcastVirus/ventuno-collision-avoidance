from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    backend_arg = DeclareLaunchArgument(
        "backend", default_value="cpu",
        description="Inference backend: 'cpu' (XNNPACK) or 'npu' (QNN HTP)")

    model_path_arg = DeclareLaunchArgument(
        "model_path", default_value="models/yolox_tiny_xnnpack.pte",
        description="Path to .pte model file")

    params_file = PathJoinSubstitution(
        [FindPackageShare("yolox_detector"), "config", "detector.yaml"])

    detector_node = Node(
        package="yolox_detector",
        executable="yolox_detector_node",
        name="yolox_detector",
        parameters=[
            params_file,
            {
                "backend": LaunchConfiguration("backend"),
                "model_path": LaunchConfiguration("model_path"),
            },
        ],
        output="screen",
    )

    return LaunchDescription([
        backend_arg,
        model_path_arg,
        detector_node,
    ])
