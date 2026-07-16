#!/usr/bin/env python3
"""Keyboard-driven collision-avoidance data collection.

The ROS 2 replacement for the JetBot ``data_collection.ipynb`` camera widget +
free/blocked buttons. Subscribes to the OAK RGB stream and saves the current
frame, labeled by a single keypress, into a torchvision-friendly ImageFolder:

    <output_dir>/
        free/     <uuid>.jpg
        blocked/  <uuid>.jpg

Controls (focus this terminal):
    f  save the current frame as FREE     (safe to drive forward)
    b  save the current frame as BLOCKED  (obstacle / ledge ahead)
    q  quit

Collect varied, roughly balanced data: different orientations, lighting,
obstacle types and floor textures, aiming for at least ~100 images per class.
Then zip <output_dir> and train with tools/train_collision_resnet18.py.

    ros2 run collision_avoider data_collection
    ros2 launch collision_avoider data_collection.launch.py   # also starts the camera
"""

import os
import select
import sys
import termios
import tty
from uuid import uuid1

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')

        self.declare_parameter('image_topic', '/oak/rgb/image_raw')
        self.declare_parameter('output_dir', 'dataset')
        self.declare_parameter('image_size', 224)
        self.declare_parameter('poll_rate', 30.0)

        image_topic = self.get_parameter('image_topic').value
        self.output_dir = self.get_parameter('output_dir').value
        self.image_size = self.get_parameter('image_size').value
        poll_rate = self.get_parameter('poll_rate').value

        self.free_dir = os.path.join(self.output_dir, 'free')
        self.blocked_dir = os.path.join(self.output_dir, 'blocked')
        os.makedirs(self.free_dir, exist_ok=True)
        os.makedirs(self.blocked_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.latest = None      # latest BGR frame as a numpy array

        self.sub = self.create_subscription(
            Image, image_topic, self.on_image, 10)
        self.timer = self.create_timer(1.0 / poll_rate, self.poll_keyboard)

        # Put the terminal in cbreak mode so single keypresses arrive without Enter.
        self._stdin_fd = sys.stdin.fileno()
        self._raw_ok = sys.stdin.isatty()
        if self._raw_ok:
            self._old_term = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
        else:
            # Under `ros2 launch` a node's stdin is usually not an interactive
            # terminal, so keypresses can't be read. Run the collector directly
            # in a terminal instead: `ros2 run collision_avoider data_collection`
            # (start the camera separately).
            self.get_logger().warn(
                "stdin is not a TTY: keyboard capture is disabled. Run "
                "`ros2 run collision_avoider data_collection` in a terminal.")

        self.get_logger().info(
            f"data_collector up: image='{image_topic}' output='{self.output_dir}' "
            f"size={self.image_size}. Keys: [f]=free  [b]=blocked  [q]=quit")
        self._print_counts()

    def on_image(self, msg):
        try:
            self.latest = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'cv_bridge: {e}', throttle_duration_sec=2.0)

    def poll_keyboard(self):
        if not self._raw_ok:
            return
        # Non-blocking: only read if a key is waiting.
        if not select.select([sys.stdin], [], [], 0)[0]:
            return
        key = sys.stdin.read(1).lower()
        if key == 'f':
            self._save(self.free_dir, 'free')
        elif key == 'b':
            self._save(self.blocked_dir, 'blocked')
        elif key == 'q':
            self.get_logger().info('Quitting data collection.')
            raise KeyboardInterrupt

    def _save(self, directory, label):
        if self.latest is None:
            self.get_logger().warn('No camera frame received yet; nothing saved.')
            return
        frame = cv2.resize(self.latest, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_AREA)
        path = os.path.join(directory, f'{uuid1()}.jpg')
        cv2.imwrite(path, frame)
        self._print_counts(saved=label)

    def _print_counts(self, saved=None):
        free_n = len(os.listdir(self.free_dir))
        blocked_n = len(os.listdir(self.blocked_dir))
        prefix = f"saved {saved:>7}  " if saved else ' ' * 14
        self.get_logger().info(f"{prefix}free={free_n}  blocked={blocked_n}")

    def restore_terminal(self):
        if self._raw_ok:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_term)


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.restore_terminal()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
