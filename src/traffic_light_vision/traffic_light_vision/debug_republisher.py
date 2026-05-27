#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


class DebugRepublisher(Node):

    def __init__(self):
        super().__init__('debug_republisher')

        self.bridge = CvBridge()

        self.perception_raw_pub = self.create_publisher(
            Image,
            '/perception_debug_raw',
            10
        )

        self.traffic_raw_pub = self.create_publisher(
            Image,
            '/traffic_debug_raw',
            10
        )

        self.create_subscription(
            CompressedImage,
            '/perception/debug/compressed',
            self.perception_callback,
            10
        )

        self.create_subscription(
            CompressedImage,
            '/traffic_debug/compressed',
            self.traffic_callback,
            10
        )

        self.get_logger().info(
            'DebugRepublisher started: compressed debug → raw debug images'
        )

    def decode_compressed(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return None

        return image

    def perception_callback(self, msg):
        image = self.decode_compressed(msg)

        if image is None:
            self.get_logger().warn('Could not decode /perception/debug/compressed')
            return

        out_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = 'perception_debug'
        self.perception_raw_pub.publish(out_msg)

    def traffic_callback(self, msg):
        image = self.decode_compressed(msg)

        if image is None:
            self.get_logger().warn('Could not decode /traffic_debug/compressed')
            return

        out_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = 'traffic_debug'
        self.traffic_raw_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)

    node = DebugRepublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

