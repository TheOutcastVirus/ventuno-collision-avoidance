#!/usr/bin/env python3
"""Reactive collision-avoidance controller for the Create 3 base.

Subscribes to ``/collision/classification`` (vision_msgs/Classification, the
free/blocked probabilities from collision_classifier) and publishes
``geometry_msgs/Twist`` velocity commands, mirroring the JetBot collision-
avoidance behavior:

* **free** (P(blocked) < ``blocked_threshold``)  -> drive forward at ``base_speed``
* **blocked** (P(blocked) >= ``blocked_threshold``) -> turn in place at
  ``turn_speed`` toward ``turn_direction`` (+1 = left / CCW) to look for a clear path

The blocked probability is EMA-smoothed (``prob_smoothing``) to avoid twitching
on a single noisy frame.

Safety: velocities are clamped to configurable caps, ``publish_cmd_vel:=false``
runs a no-motion dry run (commands are logged, not sent), the robot stops if no
classification arrives within ``lost_timeout``, and a zero Twist is published on
shutdown.
"""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from vision_msgs.msg import Classification


def clamp(value, limit):
    """Clamp ``value`` to the symmetric range [-limit, limit]."""
    return max(-limit, min(limit, value))


class CollisionAvoider(Node):
    def __init__(self):
        super().__init__('collision_avoider')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('classification_topic', '/collision/classification')
        self.declare_parameter('blocked_label', 'blocked')
        self.declare_parameter('blocked_threshold', 0.5)
        self.declare_parameter('base_speed', 0.15)      # m/s forward when free
        self.declare_parameter('turn_speed', 0.6)       # rad/s in-place when blocked
        self.declare_parameter('turn_direction', 1.0)   # +1 = left (CCW), -1 = right
        self.declare_parameter('prob_smoothing', 0.5)   # EMA weight on new P(blocked)
        self.declare_parameter('max_linear_speed', 0.25)
        self.declare_parameter('max_angular_speed', 1.5)
        self.declare_parameter('lost_timeout', 0.5)     # s w/o classification -> stop
        self.declare_parameter('control_rate', 10.0)    # Hz control loop
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('publish_cmd_vel', True)

        g = self.get_parameter
        classification_topic = g('classification_topic').value
        self.blocked_label = g('blocked_label').value
        self.blocked_threshold = g('blocked_threshold').value
        self.base_speed = g('base_speed').value
        self.turn_speed = g('turn_speed').value
        self.turn_direction = 1.0 if g('turn_direction').value >= 0 else -1.0
        self.prob_smoothing = g('prob_smoothing').value
        self.max_linear_speed = g('max_linear_speed').value
        self.max_angular_speed = g('max_angular_speed').value
        self.lost_timeout = g('lost_timeout').value
        control_rate = g('control_rate').value
        cmd_vel_topic = g('cmd_vel_topic').value
        self.publish_cmd_vel = g('publish_cmd_vel').value

        # ── State ────────────────────────────────────────────────────────────
        self.prob_blocked = None       # EMA-filtered P(blocked)
        self.last_msg_time = None      # ROS time the last classification arrived

        # ── ROS interfaces ───────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.sub = self.create_subscription(
            Classification, classification_topic, self.on_classification, 10)
        self.timer = self.create_timer(1.0 / control_rate, self.control_step)

        self.get_logger().info(
            f"collision_avoider up: classification='{classification_topic}' "
            f"cmd_vel='{cmd_vel_topic}' publish={self.publish_cmd_vel} "
            f"base_speed={self.base_speed} turn_speed={self.turn_speed} "
            f"threshold={self.blocked_threshold}")
        if not self.publish_cmd_vel:
            self.get_logger().warn(
                'publish_cmd_vel is FALSE: computing commands but not moving.')

    def on_classification(self, msg):
        """Extract P(blocked) from the classification and EMA-smooth it."""
        score = None
        for result in msg.results:
            if result.class_id == self.blocked_label:
                score = result.score
                break
        if score is None:
            self.get_logger().warn(
                f"No '{self.blocked_label}' result in classification message",
                throttle_duration_sec=2.0)
            return

        if self.prob_blocked is None:
            self.prob_blocked = score
        else:
            a = self.prob_smoothing
            self.prob_blocked = a * score + (1.0 - a) * self.prob_blocked
        self.last_msg_time = self.get_clock().now()

    def control_step(self):
        twist = Twist()

        age = None
        if self.last_msg_time is not None:
            age = (self.get_clock().now() - self.last_msg_time).nanoseconds * 1e-9

        if self.prob_blocked is None or age is None or age >= self.lost_timeout:
            # No fresh classification: stay stopped (fail safe).
            self._publish(twist)
            return

        if self.prob_blocked < self.blocked_threshold:
            # Path is clear -> drive forward.
            twist.linear.x = clamp(self.base_speed, self.max_linear_speed)
        else:
            # Blocked -> turn in place to find a clear direction.
            twist.angular.z = clamp(
                self.turn_direction * self.turn_speed, self.max_angular_speed)

        self._publish(twist)

    def _publish(self, twist):
        if self.publish_cmd_vel:
            self.cmd_pub.publish(twist)
        else:
            p = self.prob_blocked if self.prob_blocked is not None else float('nan')
            self.get_logger().info(
                f"[dry-run] P(blocked)={p:.2f} "
                f"lin={twist.linear.x:+.3f} ang={twist.angular.z:+.3f}",
                throttle_duration_sec=0.5)

    def stop(self):
        """Publish a single zero Twist so the robot halts on shutdown."""
        try:
            self.cmd_pub.publish(Twist())
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = CollisionAvoider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
