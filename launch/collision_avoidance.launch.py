"""Full collision-avoidance demo: OAK RGB -> ResNet18 free/blocked -> Create 3.

Starts four nodes:
  * create3_repub        — bridges the Create 3 base (its topics live in _do_not_use)
                           to clean root topics, so /cmd_vel actually reaches the base
  * oak_camera_node      — publishes the RGB stream
  * collision_classifier — classifies each frame free/blocked, publishes
                           /collision/classification (and /collision/image debug overlay)
  * collision_avoider    — drives the base (/cmd_vel): forward when free, turn when blocked

Because we replace the TurtleBot 4's Raspberry Pi, we may need to run
create3_republisher ourselves — it relays /cmd_vel down to the Create 3 base and
republishes /odom, /tf, /imu, etc. up to the ROS 2 network. Requires the DDS
environment to match the base: ROS_DOMAIN_ID and RMW_IMPLEMENTATION=rmw_fastrtps_cpp.
Set enable_republisher:=false if the base is already at the root namespace.

Defaults to the NPU (QNN) backend, since models/collision_resnet18_qnn.pte is the
one exported for the Hexagon HTP. Use backend:=cpu with the XNNPACK .pte to run
on CPU.

Examples:
    ros2 launch collision_avoidance.launch.py
    ros2 launch collision_avoidance.launch.py publish_cmd_vel:=false   # dry run
    ros2 launch collision_avoidance.launch.py blocked_threshold:=0.6   # more cautious
    ros2 launch collision_avoidance.launch.py backend:=cpu model_path:=models/collision_resnet18_xnnpack.pte
    ros2 launch collision_avoidance.launch.py enable_republisher:=false  # base at root ns
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
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

    enable_republisher_arg = DeclareLaunchArgument(
        "enable_republisher", default_value="true",
        description="Run create3_republisher to bridge the Create 3 base. Set false "
                    "if the base is bridged elsewhere or configured without _do_not_use")

    robot_ns_arg = DeclareLaunchArgument(
        "robot_ns", default_value="/_do_not_use",
        description="Namespace the Create 3 publishes its raw topics under "
                    "(stock TurtleBot 4 on Jazzy uses /_do_not_use)")

    republisher_ns_arg = DeclareLaunchArgument(
        "republisher_ns", default_value="/",
        description="Namespace the bridged clean topics (/cmd_vel, /odom, ...) appear "
                    "under. Must differ from robot_ns")

    # Bridge the Create 3 base: relays /cmd_vel down to the base and republishes
    # /odom, /tf, /imu, ... up. Reuses the package's own launch file.
    create3_republisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare("create3_republisher"), "bringup",
             "create3_republisher_launch.py"])),
        launch_arguments={
            "robot_ns": LaunchConfiguration("robot_ns"),
            "republisher_ns": LaunchConfiguration("republisher_ns"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("enable_republisher")),
    )

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
        enable_republisher_arg,
        robot_ns_arg,
        republisher_ns_arg,
        create3_republisher,
        camera_node,
        classifier_node,
        avoider_node,
    ])
