#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D, Pose
from aoc_fruit_detector.msg import FruitInfoMessage
from predictor import call_predictor

import os, yaml, cv2
from detectron_predictor.detectron_predictor import DetectronPredictor

from cv_bridge import CvBridge, CvBridgeError

from rclpy.qos import QoSProfile, ReliabilityPolicy

import numpy as np

class FruitDetectionNode(Node):
    def __init__(self):
        super().__init__('fruit_detection')
        # Create a publisher for the custom message type
        self.publisher_ = self.create_publisher(FruitInfoMessage, 'fruit_info', 10)
        self.prediction_generator = call_predictor()
        config_path = self.find_data_folder_config()
        if config_path:
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                self.det_predictor = DetectronPredictor(config_data)

                test_image_dir              = config_data['directories']['train_image_dir']
                prediction_json_output_dir  = config_data['directories']['prediction_json_dir']
                self.test_image_dir = test_image_dir
                self.prediction_json_output_dir = prediction_json_output_dir

                # Declare parameters for min_depth and max_depth
                self.declare_parameter('min_depth', 0.1)  # Default value
                self.declare_parameter('max_depth', 15.0)  # Default value
                self.min_depth = self.get_parameter('min_depth').value
                self.max_depth = self.get_parameter('max_depth').value
        else:
            raise FileNotFoundError(f"No config file found in any 'data/config/' folder within {os.getcwd()}")

        self.bridge = CvBridge()
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            qos_profile
        )

        self.depth_sub = self.create_subscription(
            Image,
            'camera/depth',
            self.depth_callback,
            qos_profile
        )

    def find_data_folder_config(self,search_dir='.'):
        for root, dirs, _ in os.walk(search_dir):
            if 'data' in dirs:
                data_folder = os.path.join(root, 'data')
                config_path = os.path.join(data_folder, 'config', 'config.yaml')
                # Check if config.yaml exists in the data/config folder
                if os.path.exists(config_path):
                    return config_path
        return None

    def compute_pose2d(self, mask):
        """Calculate Pose2D from the mask (segmentation coordinates)."""
        x_coords = mask[0::2]  # Every 2nd element starting from index 0 (x coordinates)
        y_coords = mask[1::2]  # Every 2nd element starting from index 1 (y coordinates)
        
        pose2d = Pose2D()
        pose2d.x = sum(x_coords) / len(x_coords)  # Calculate centroid X
        pose2d.y = sum(y_coords) / len(y_coords)  # Calculate centroid Y
        pose2d.theta = 0.0  # Assuming no rotation needed for pose2d
        
        return pose2d
    
    def compute_pose3d(self, pose2d):
        pose3d = Pose()
        pose3d.position.x = pose2d.x
        pose3d.position.y = pose2d.y
        pose3d.position.z = 0.0  # Assuming z = 0 for 2D
        
        # Identity quaternion (no rotation)
        pose3d.orientation.x = 0.0
        pose3d.orientation.y = 0.0
        pose3d.orientation.z = 0.0
        pose3d.orientation.w = 1.0  # No rotation
    
        return pose3d

    def depth_callback(self, msg):
        try:
            # Convert ROS2 depth Image message to OpenCV depth image
            self.cv_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.depth_msg = msg
            # To replace NaN with max_depth value
            # self.cv_depth_image[np.isnan(self.cv_depth_image)] = self.max_depth 
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info("RGB ready.")
            
            # Create image_id as an integer using the timestamp
            image_id = int(f'{msg.header.stamp.sec}{str(msg.header.stamp.nanosec).zfill(9)}')
            self.get_logger().info("image ID is: ", image_id)

            rgb_msg = msg                 
            depth_msg = self.depth_msg

            if self.cv_depth_image is not None:
                # Ensure that the depth image is the same size as the RGB image
                if self.cv_image.shape[:2] != self.cv_depth_image.shape[:2]:
                    self.get_logger().warn("Resizing depth image to match RGB image dimensions.")
                    depth_image = cv2.resize(self.cv_depth_image, (self.cv_image.shape[1], self.cv_image.shape[0]))
                else:
                    depth_image = self.cv_depth_image
            else:
                # If no depth image is available, use the R channel of the RGB image as a substitute
                self.get_logger().warn("No depth image available. Using R channel of RGB as depth substitute.")
                depth_image = self.cv_image[:, :, 2].astype(np.float32)  # Using the R channel
            self.get_logger().info("Depth ready")
            # Combine RGB and depth into a single 4-channel image (3 for RGB + 1 for depth)
            rgbd_image = np.dstack((self.cv_image, depth_image))
            self.get_logger().info("RGBD ready")
            
            json_annotation_message, _, depth_mask = self.det_predictor.get_predictions_message(rgbd_image,image_id)
            annotations = json_annotation_message.get('annotations', [])
            categories = json_annotation_message.get('categories', [])

            for annotation in annotations:
                #self.get_logger().info(f'Annotation: {annotation}')
                fruit_id = annotation.get('id', None)
                image_id = annotation.get('image_id', None)
                category_id = annotation.get('category_id', -1)
                segmentation = annotation.get('segmentation', [])
                segmentation = [point for sublist in segmentation for point in sublist]  # Flatten segmentation
                bbox = annotation.get('bbox', [0.0, 0.0, 0.0, 0.0])
                area = float(annotation.get('area', 0.0))

                category_details = next(
                    (category for category in categories if category.get('id') == category_id),
                    {'name': 'unknown', 'supercategory': 'unknown'}
                )
                ripeness_category = category_details.get('name', 'unknown')

                fruit_msg = FruitInfoMessage()
                fruit_msg.header = Header()
                fruit_msg.header.stamp = self.get_clock().now().to_msg()
                fruit_msg.header.frame_id = rgb_msg.header.frame_id
                fruit_msg.fruit_id = fruit_id
                fruit_msg.image_id = image_id
                ### Fruit Biological Features ####
                fruit_msg.pomological_class = 'Edible Plant'
                fruit_msg.edible_plant_part = 'Culinary Vegetable'
                fruit_msg.fruit_family = 'Solanaceae'
                fruit_msg.fruit_species = 'Solanum lycopersicum'
                fruit_msg.fruit_type = 'Tomato'
                fruit_msg.fruit_variety = 'Plum'
                fruit_msg.fruit_genotype = 'San Marzano'
                #####################################
                fruit_msg.fruit_quality = 'High'
                fruit_msg.ripeness_category = ripeness_category
                if fruit_msg.ripeness_category == 'fruit_ripe':
                    fruit_msg.ripeness_level = 0.95
                else:
                    fruit_msg._ripeness_level = 0.15
                fruit_msg.area = area
                fruit_msg.volume = area*2
                fruit_msg.bbox = bbox
                fruit_msg.bvol = bbox
                fruit_msg.mask2d = segmentation
                fruit_msg.pose2d = self.compute_pose2d(segmentation)
                fruit_msg.mask3d = segmentation
                fruit_msg.pose3d = self.compute_pose3d(fruit_msg.pose2d)
                fruit_msg.confidence = 0.93
                fruit_msg.occlusion_level = 0.88

                fruit_msg.rgb_image = rgb_msg        # Assign the current RGB image
                fruit_msg.depth_image = depth_msg    # Assign the stored depth image

                # Log and publish the message
                self.get_logger().info(f'Publishing: image_id={fruit_msg.image_id}, fruit_id={fruit_msg.fruit_id}, type={fruit_msg.fruit_type}, variety={fruit_msg.fruit_variety}, ripeness={fruit_msg.ripeness_category}')
                self.get_logger().info(f'Publishing pose of fruit: {fruit_msg.pose2d}')
                self.publisher_.publish(fruit_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = FruitDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()