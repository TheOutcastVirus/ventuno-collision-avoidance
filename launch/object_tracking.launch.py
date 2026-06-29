from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    backend_arg = DeclareLaunchArgument(
        "backend", default_value="cpu",
        description="Inference backend: 'cpu' or 'npu'")

    model_path_arg = DeclareLaunchArgument(
        "model_path", default_value="models/yolox_tiny_xnnpack.pte",
        description="Path to the .pte model file")

    camera_params = PathJoinSubstitution(
        [FindPackageShare("oak_camera"), "config", "camera.yaml"])

    detector_params = PathJoinSubstitution(
        [FindPackageShare("yolox_detector"), "config", "detector.yaml"])

    camera_node = Node(
        package="oak_camera",
        executable="oak_camera_node",
        name="oak_camera",
        parameters=[camera_params],
        output="screen",
    )

    detector_node = Node(
        package="yolox_detector",
        executable="yolox_detector_node",
        name="yolox_detector",
        parameters=[
            detector_params,
            {
                "backend": LaunchConfiguration("backend"),
                "model_path": LaunchConfiguration("model_path"),
            },
        ],
        output="screen",
        # Remap camera topic so both nodes agree on the topic name
        remappings=[("/camera/rgb/image_raw", "/camera/rgb/image_raw")],
    )

    return LaunchDescription([
        backend_arg,
        model_path_arg,
        camera_node,
        detector_node,
    ])
