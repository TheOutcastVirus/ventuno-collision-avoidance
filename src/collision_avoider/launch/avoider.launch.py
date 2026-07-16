"""Collision-avoidance controller for the Create 3 base.

Runs collision_avoider, which consumes /collision/classification and publishes
geometry_msgs/Twist to /cmd_vel: forward when free, turn in place when blocked.
Our Create 3 runs at the ROOT namespace, so its motion_control node subscribes
to /cmd_vel directly -- no create3_republisher and no /_do_not_use bridge is
needed here. The base is reached over the USB-ethernet gadget link (usb0); see
scripts/create3_usb_gadget.sh.

Requires the DDS environment to match the base (set in ~/.bashrc):
  ROS_DOMAIN_ID=0, RMW_IMPLEMENTATION=rmw_fastrtps_cpp,
  FASTRTPS_DEFAULT_PROFILES_FILE=scripts/fastdds_usb0.xml

Set publish_cmd_vel:=false for a dry run (compute and log Twist without moving).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    publish_cmd_vel_arg = DeclareLaunchArgument(
        "publish_cmd_vel", default_value="true",
        description="false = dry run: compute and log Twist without moving the robot")

    cmd_vel_topic_arg = DeclareLaunchArgument(
        "cmd_vel_topic", default_value="/cmd_vel",
        description="Velocity topic the Create 3 subscribes to (geometry_msgs/Twist)")

    params_file = PathJoinSubstitution(
        [FindPackageShare("collision_avoider"), "config", "avoider.yaml"])

    avoider_node = Node(
        package="collision_avoider",
        executable="collision_avoider",
        name="collision_avoider",
        parameters=[
            params_file,
            {
                "publish_cmd_vel": ParameterValue(
                    LaunchConfiguration("publish_cmd_vel"), value_type=bool),
                "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
            },
        ],
        output="screen",
    )

    return LaunchDescription([
        publish_cmd_vel_arg,
        cmd_vel_topic_arg,
        avoider_node,
    ])
