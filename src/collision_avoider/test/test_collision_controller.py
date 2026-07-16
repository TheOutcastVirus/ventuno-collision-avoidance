import pytest
import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from vision_msgs.msg import Classification, ObjectHypothesis

from collision_avoider.controller import CollisionAvoider


@pytest.fixture
def avoider():
    rclpy.init()
    node = CollisionAvoider()
    # Capture the Twist the control loop would publish instead of sending it.
    node.published = []
    node._publish = node.published.append
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


def classification(blocked_score):
    msg = Classification()
    for label, score in (('blocked', blocked_score), ('free', 1.0 - blocked_score)):
        hyp = ObjectHypothesis()
        hyp.class_id = label
        hyp.score = score
        msg.results.append(hyp)
    return msg


def test_free_drives_forward(avoider):
    avoider.on_classification(classification(0.1))
    avoider.control_step()
    twist = avoider.published[-1]
    assert twist.linear.x > 0.0
    assert twist.angular.z == 0.0


def test_blocked_turns_in_place(avoider):
    avoider.on_classification(classification(0.9))
    avoider.control_step()
    twist = avoider.published[-1]
    assert twist.linear.x == 0.0
    # turn_direction defaults to +1 (left / CCW).
    assert twist.angular.z > 0.0


def test_stale_classification_stops(avoider):
    avoider.on_classification(classification(0.1))
    # Age the last message past lost_timeout.
    avoider.last_msg_time = (
        avoider.get_clock().now() - Duration(seconds=avoider.lost_timeout + 1.0))
    avoider.control_step()
    twist = avoider.published[-1]
    assert twist.linear.x == 0.0
    assert twist.angular.z == 0.0


def test_no_classification_stops(avoider):
    avoider.control_step()
    twist = avoider.published[-1]
    assert twist == Twist()


def test_speed_is_clamped(avoider):
    avoider.base_speed = 10.0          # absurd command
    avoider.max_linear_speed = 0.25
    avoider.on_classification(classification(0.0))
    avoider.control_step()
    twist = avoider.published[-1]
    assert twist.linear.x == pytest.approx(0.25)
