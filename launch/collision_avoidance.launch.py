"""Full collision-avoidance demo: OAK RGB -> ResNet18 free/blocked -> Create 3.

Starts three nodes:
  * oak_camera_node      — publishes the RGB stream
  * collision_classifier — classifies each frame free/blocked, publishes
                           /collision/classification (and /collision/image debug overlay)
  * collision_avoider    — drives the base (/cmd_vel): forward when free, turn when blocked

The Create 3 runs at the root namespace and subscribes directly to `/cmd_vel`;
no `create3_republisher` or `/_do_not_use` bridge is required. The DDS environment
must match the base: ROS_DOMAIN_ID and RMW_IMPLEMENTATION=rmw_fastrtps_cpp.

Defaults to the NPU (QNN) backend, since models/collision_resnet18_qnn.pte is the
one exported for the Hexagon HTP. Use backend:=cpu with the XNNPACK .pte to run
on CPU.

Examples:
    ros2 launch launch/collision_avoidance.launch.py
    ros2 launch launch/collision_avoidance.launch.py publish_cmd_vel:=false   # dry run
    ros2 launch launch/collision_avoidance.launch.py blocked_threshold:=0.6   # more cautious
    ros2 launch launch/collision_avoidance.launch.py backend:=cpu model_path:=models/collision_resnet18_xnnpack.pte
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    backend_arg = DeclareLaunchArgument(
        "backend", default_value="npu",
        description="Inference backend: 'cpu' (XNNPACK) or 'npu' (QNN HTP)")

    model_path_arg = DeclareLaunchArgument(
        "model_path", default_value="",
        description="Path to the .pte model file; empty selects the default for the backend")

    image_topic_arg = DeclareLaunchArgument(
        "image_topic", default_value="/oak/rgb/image_raw",
        description="RGB image topic shared by the camera and classifier")

    qnn_lib_dir_arg = DeclareLaunchArgument(
        "qnn_lib_dir",
        default_value=PathJoinSubstitution([
            EnvironmentVariable("QAIRT_LIB"), "aarch64-oe-linux-gcc11.2"]),
        description="Directory containing QNN runtime libraries; defaults to "
                    "$QAIRT_LIB/aarch64-oe-linux-gcc11.2 (npu backend)")

    blocked_threshold_arg = DeclareLaunchArgument(
        "blocked_threshold", default_value="0.5",
        description="P(blocked) above which the robot turns instead of driving forward")

    base_speed_arg = DeclareLaunchArgument(
        "base_speed", default_value="0.15",
        description="Forward speed when the path is free (m/s)")

    turn_speed_arg = DeclareLaunchArgument(
        "turn_speed", default_value="0.6",
        description="In-place rotation speed when blocked (rad/s)")

    publish_cmd_vel_arg = DeclareLaunchArgument(
        "publish_cmd_vel", default_value="true",
        description="false = dry run: log Twist commands without moving the robot")

    cmd_vel_topic_arg = DeclareLaunchArgument(
        "cmd_vel_topic", default_value="/cmd_vel",
        description="Velocity topic for the Create 3 base (geometry_msgs/Twist)")



    camera_params = PathJoinSubstitution(
        [FindPackageShare("oak_camera"), "config", "camera.yaml"])
    classifier_params = PathJoinSubstitution(
        [FindPackageShare("collision_classifier"), "config", "classifier.yaml"])
    avoider_params = PathJoinSubstitution(
        [FindPackageShare("collision_avoider"), "config", "avoider.yaml"])

    camera_node = Node(
        package="oak_camera",
        executable="oak_camera_node",
        name="oak_camera",
        parameters=[camera_params],
        output="screen",
    )

    classifier_node = Node(
        package="collision_classifier",
        executable="collision_classifier_node",
        name="collision_classifier",
        parameters=[
            classifier_params,
            {
                "backend": LaunchConfiguration("backend"),
                "model_path": LaunchConfiguration("model_path"),
                "qnn_lib_dir": LaunchConfiguration("qnn_lib_dir"),
                "image_topic": LaunchConfiguration("image_topic"),
            },
        ],
        output="screen",
    )

    avoider_node = Node(
        package="collision_avoider",
        executable="collision_avoider",
        name="collision_avoider",
        parameters=[
            avoider_params,
            {
                "publish_cmd_vel": ParameterValue(
                    LaunchConfiguration("publish_cmd_vel"), value_type=bool),
                "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                "blocked_threshold": ParameterValue(
                    LaunchConfiguration("blocked_threshold"), value_type=float),
                "base_speed": ParameterValue(
                    LaunchConfiguration("base_speed"), value_type=float),
                "turn_speed": ParameterValue(
                    LaunchConfiguration("turn_speed"), value_type=float),
            },
        ],
        output="screen",
    )

    return LaunchDescription([
        backend_arg,
        model_path_arg,
        image_topic_arg,
        qnn_lib_dir_arg,
        blocked_threshold_arg,
        base_speed_arg,
        turn_speed_arg,
        publish_cmd_vel_arg,
        cmd_vel_topic_arg,
        camera_node,
        classifier_node,
        avoider_node,
    ])
