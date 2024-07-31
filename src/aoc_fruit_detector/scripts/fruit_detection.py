#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from aoc_fruit_detector.msg import FruitInfoMessage

class FruitDetectionNode(Node):
    def __init__(self):
        super().__init__('fruit_detection')
        # Create a publisher for the custom message type
        self.publisher_ = self.create_publisher(FruitInfoMessage, 'fruit_info', 10)
        # Timer to call the publish_message method periodically
        self.timer = self.create_timer(1.0, self.publish_message)
        self.count = 0

    def publish_message(self):
        # Create an instance of the custom message type
        msg = FruitInfoMessage()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'tomato_plant'
        #msg.header.seq = self.count
        msg.fruit_id = self.count
        msg.fruit_type = 'Tomato'
        msg.fruit_variety = 'Plum'
        msg.data = 'Test message'
        
        # Log and publish the message
        self.get_logger().info(f'Publishing: id={msg.fruit_id}, type={msg.fruit_type}, variety={msg.fruit_variety}, data={msg.data}')
        self.publisher_.publish(msg)
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = FruitDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()