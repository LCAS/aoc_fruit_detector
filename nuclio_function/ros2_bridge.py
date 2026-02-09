"""
ROS2 Bridge for communication with the AOC Fruit Detector node.

This module handles the ROS2 communication for publishing images to the
fruit detector and receiving detection results.
"""

import logging
from typing import Optional, List
import numpy as np
import cv2
import threading
import time

# Try to import ROS2 dependencies
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    
    # Try to import AOC messages, with fallback
    try:
        from aoc_fruit_detector.msg import FruitInfoArray
    except ImportError:
        # Create mock for development/testing
        class FruitInfoArray:
            def __init__(self):
                self.fruits = []
    
    ROS2_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ROS2 not available: {e}")
    ROS2_AVAILABLE = False
    
    # Mock classes for testing without ROS2
    class Node:
        def __init__(self, name): pass
        def create_publisher(self, *args, **kwargs): return None
        def create_subscription(self, *args, **kwargs): return None
        def get_clock(self): return MockClock()
    
    class MockClock:
        def now(self): return MockTime()
    
    class MockTime:
        def to_msg(self): return None
    
    class FruitInfoArray:
        def __init__(self):
            self.fruits = []

logger = logging.getLogger(__name__)

class ROS2Bridge(Node):
    """
    ROS2 bridge for communicating with the fruit detector node.
    """
    
    def __init__(self):
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS2 not available - cannot initialize bridge")
            
        super().__init__('nuclio_fruit_detector_bridge')
        
        self.cv_bridge = CvBridge()
        self.latest_detections = None
        self.detection_received = threading.Event()
        
        # Configure QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        
        # Publisher for camera images
        self.image_publisher = self.create_publisher(
            Image,
            '/camera/image_raw',
            qos_profile
        )
        
        # Subscriber for fruit detection results
        self.fruit_subscriber = self.create_subscription(
            FruitInfoArray,
            '/fruit_info',
            self.fruit_detection_callback,
            qos_profile
        )
        
        # Start ROS2 spinning in a separate thread
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        
        logger.info("ROS2 Bridge initialized")
    
    def fruit_detection_callback(self, msg: FruitInfoArray):
        """
        Callback for receiving fruit detection results.
        """
        logger.info(f"Received fruit detections: {len(msg.fruits)} fruits detected")
        self.latest_detections = msg
        self.detection_received.set()
    
    def process_image(self, cv_image: np.ndarray, confidence_threshold: float = 0.5, timeout: float = 30.0) -> Optional[FruitInfoArray]:
        """
        Process an image through the fruit detector.
        
        Args:
            cv_image: OpenCV image (BGR format)
            confidence_threshold: Minimum confidence for detections
            timeout: Maximum time to wait for results in seconds
            
        Returns:
            FruitInfoArray message with detection results, or None if failed
        """
        try:
            # Reset detection event
            self.detection_received.clear()
            self.latest_detections = None
            
            # Convert OpenCV image to ROS Image message
            ros_image = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'camera_optical_frame'
            
            # Publish image
            logger.info("Publishing image to fruit detector")
            self.image_publisher.publish(ros_image)
            
            # Wait for detection results
            logger.info(f"Waiting for detection results (timeout: {timeout}s)")
            if self.detection_received.wait(timeout=timeout):
                logger.info("Detection results received")
                
                # Filter detections by confidence threshold
                filtered_detections = self._filter_by_confidence(
                    self.latest_detections, 
                    confidence_threshold
                )
                
                return filtered_detections
            else:
                logger.error("Timeout waiting for fruit detection results")
                return None
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None
    
    def _filter_by_confidence(self, detections: FruitInfoArray, threshold: float) -> FruitInfoArray:
        """
        Filter detections by confidence threshold.
        """
        if not detections or not detections.fruits:
            return detections
            
        filtered_fruits = []
        for fruit in detections.fruits:
            if fruit.confidence >= threshold:
                filtered_fruits.append(fruit)
        
        # Create new FruitInfoArray with filtered results
        filtered_array = FruitInfoArray()
        filtered_array.fruits = filtered_fruits
        filtered_array.rgb_image = detections.rgb_image
        filtered_array.depth_image = detections.depth_image
        filtered_array.rgb_image_composed = detections.rgb_image_composed
        
        logger.info(f"Filtered detections: {len(filtered_fruits)}/{len(detections.fruits)} above threshold {threshold}")
        
        return filtered_array
    
    def shutdown(self):
        """
        Shutdown the ROS2 bridge.
        """
        logger.info("Shutting down ROS2 bridge")
        self.executor.shutdown()
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=5.0)


def initialize_ros2():
    """
    Initialize ROS2 if not already initialized.
    """
    if not ROS2_AVAILABLE:
        raise RuntimeError("ROS2 not available")
        
    if not rclpy.ok():
        logger.info("Initializing ROS2")
        rclpy.init()
    else:
        logger.info("ROS2 already initialized")