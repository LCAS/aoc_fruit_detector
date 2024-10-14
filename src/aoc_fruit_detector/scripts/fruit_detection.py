#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D, Pose, PoseStamped
from aoc_fruit_detector.msg import FruitInfoMessage, FruitInfoArray
#from predictor import call_predictor

import os, yaml, cv2
from detectron_predictor.detectron_predictor import DetectronPredictor

from cv_bridge import CvBridge, CvBridgeError

from rclpy.qos import QoSProfile, ReliabilityPolicy

import numpy as np

import image_geometry

class FruitDetectionNode(Node):
    def __init__(self):
        super().__init__('fruit_detection')
        # Create a publisher for the custom message type
        self.publisher_fruit = self.create_publisher(FruitInfoArray, 'fruit_info', 5)
        self.publisher_comp = self.create_publisher(Image, 'image_composed', 5)
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
        self.camera_model = image_geometry.PinholeCameraModel()
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.declare_parameter('constant_depth_value', 1.0)  # Default depth value is 1.0
        self.constant_depth_value = self.get_parameter('constant_depth_value').value
        self.tomato = False

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

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            'camera/camera_info',
            self.camera_info_callback,
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

    def compute_pose2d(self, mask, swap_xy=False):
        """Calculate Pose2D from the mask (segmentation coordinates)."""
        
        if swap_xy:
            # Swap x and y if needed
            x_coords = np.array(mask[1::2])  # Treat every 2nd element starting from index 1 as x coordinates
            y_coords = np.array(mask[0::2])  # Treat every 2nd element starting from index 0 as y coordinates
        else:
            x_coords = np.array(mask[0::2])  # Every 2nd element starting from index 0 (x coordinates)
            y_coords = np.array(mask[1::2])  # Every 2nd element starting from index 1 (y coordinates)

        # Filter out outliers using a simple statistical approach (remove points too far from the median)
        x_median = np.median(x_coords)
        y_median = np.median(y_coords)

        # Calculate distances from the median and define a threshold (e.g., 1.5 * interquartile range)
        x_distances = np.abs(x_coords - x_median)
        y_distances = np.abs(y_coords - y_median)

        x_iqr = np.percentile(x_coords, 75) - np.percentile(x_coords, 25)
        y_iqr = np.percentile(y_coords, 75) - np.percentile(y_coords, 25)

        x_threshold = 1.5 * x_iqr
        y_threshold = 1.5 * y_iqr

        # Filter points that are within the threshold (outliers removed)
        filtered_x_coords = x_coords[x_distances < x_threshold]
        filtered_y_coords = y_coords[y_distances < y_threshold]

        # Calculate the centroid from the filtered points
        pose2d = Pose2D()
        pose2d.x = np.mean(filtered_x_coords) if len(filtered_x_coords) > 0 else x_median  # Centroid X
        pose2d.y = np.mean(filtered_y_coords) if len(filtered_y_coords) > 0 else y_median  # Centroid Y
        pose2d.theta = 0.0  # Assuming rotation is unknown

        return pose2d
    
    def compute_pose3d(self, pose2d, depth_mask):
        pose3d = PoseStamped()

        height, width, _ = depth_mask.shape
        x = int(pose2d.x)
        y = int(pose2d.y)

        if 0 <= x < height and 0 <= y < width:
            depth_values_at_pose = depth_mask[x, y, :]
            non_zero_depth_values = depth_values_at_pose[depth_values_at_pose > 0]

            if non_zero_depth_values.size > 0:
                nearest_depth_value = np.min(non_zero_depth_values)
            else:
                nearest_depth_value = 0.0 
        else:
            nearest_depth_value = 0.0
            self.get_logger().warn(f'Out of size x:{x}, height:{height}, y:{y} and width:{width}')
        
        ray = self.back_project_2d_to_3d_ray(pose2d.x, pose2d.y)
        p_3d_camera_frame = self.compute_3d_point_from_depth(ray, nearest_depth_value)
        self.get_logger().info(f'3D point at depth {nearest_depth_value}: [{p_3d_camera_frame[0]:.2f}, {p_3d_camera_frame[1]:.2f}, {p_3d_camera_frame[2]:.2f}]')

        pose3d.pose.position.x = p_3d_camera_frame[0]
        pose3d.pose.position.y = p_3d_camera_frame[1]
        pose3d.pose.position.z = p_3d_camera_frame[2]
        
        # Identity quaternion (no rotation)
        pose3d.pose.orientation.x = 0.0
        pose3d.pose.orientation.y = 0.0
        pose3d.pose.orientation.z = 0.0
        pose3d.pose.orientation.w = 1.0  # No rotation

        pose3d.header.frame_id = self.pose3d_frame
        pose3d.header.stamp = self.get_clock().now().to_msg()
    
        return pose3d

    def camera_info_callback(self, msg):
        
        self.from_camera_info(msg)
        #self.camera_model, self.distortion_coeffs = self.from_camera_info(msg)

        self.pose3d_frame = msg.header.frame_id

        self.get_logger().info('Camera model initialized')
    
    def from_camera_info(self, msg):
        self.camera_model.fromCameraInfo(msg)
        #camera_matrix = np.array(msg.k).reshape(3, 3)
        #distortion_coeffs = np.array(msg.d)
        #return camera_matrix, distortion_coeffs

    def back_project_2d_to_3d_ray(self, u, v):
        ray = self.camera_model.projectPixelTo3dRay((u, v))
        return ray
        #pixel = np.array([[u, v]], dtype=np.float32)
        #pixel = np.expand_dims(pixel, axis=0)
        #undistorted_point = cv2.undistortPoints(pixel, self.camera_model, self.distortion_coeffs)
        #ray = [undistorted_point[0][0][0], undistorted_point[0][0][1], 1.0]
        
        #return ray
    
    def compute_3d_point_from_depth(self, ray, depth):
        # Compute the 3D point by scaling the ray direction with the depth
        return [ray[0] * depth, ray[1] * depth, ray[2] * depth]

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
            self.get_logger().info("Image captured.")
            # Convert ROS Image message to OpenCV image
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info("RGB ready.")
            
            # Create image_id as an integer using the timestamp
            image_id = int(f'{msg.header.stamp.sec}{str(msg.header.stamp.nanosec).zfill(9)}')
            self.get_logger().info(f"Image ID is: {image_id}")

            rgb_msg = msg                 
            if hasattr(self, 'depth_msg') and self.depth_msg is not None:
                depth_msg = self.depth_msg
            else:
                depth_msg = Image()

            if hasattr(self, 'cv_depth_image') and self.cv_depth_image is not None:
                # Ensure that the depth image is the same size as the RGB image
                if self.cv_image.shape[:2] != self.cv_depth_image.shape[:2]:
                    self.get_logger().warn("Resizing depth image to match RGB image dimensions.")
                    depth_image = cv2.resize(self.cv_depth_image, (self.cv_image.shape[1], self.cv_image.shape[0]))
                else:
                    depth_image = self.cv_depth_image
            else:
                # If no depth image is available, use the constant depth value
                self.get_logger().warn(f"No depth image available. Using constant depth value: {self.constant_depth_value}")
                depth_image = np.full(self.cv_image.shape[:2], self.constant_depth_value, dtype=np.float32)

            self.get_logger().info("Depth ready")
            # Combine RGB and depth into a single 4-channel image (3 for RGB + 1 for depth)
            rgbd_image = np.dstack((self.cv_image, depth_image))
            self.get_logger().info("RGBD ready")
            
            json_annotation_message, _, depth_mask = self.det_predictor.get_predictions_message(rgbd_image,image_id)
            annotations = json_annotation_message.get('annotations', [])
            categories = json_annotation_message.get('categories', [])

            '''if isinstance(annotations, list) and len(annotations) > 0:
                self.get_logger().info("Keys of annotations:")
                for idx, annotation in enumerate(annotations):
                    if isinstance(annotation, dict):  # Ensure that the annotation is a dictionary
                        keys = annotation.keys()  # Get keys of the current annotation
                        self.get_logger().info(f"Annotation {idx} keys: {list(keys)}")  # Convert keys to a list for logging
                    else:
                        self.get_logger().warn(f"Annotation {idx} is not a dictionary: {annotation}")
            else:
                self.get_logger().info("No annotations found.")'''

            fruits_msg = FruitInfoArray()
            fruits_msg.fruits = []

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
                ### Tomato Fruit Biological Features ####
                if self.tomato:
                    fruit_msg.pomological_class = 'Edible Plant'
                    fruit_msg.edible_plant_part = 'Culinary Vegetable'
                    fruit_msg.fruit_family = 'Solanaceae'
                    fruit_msg.fruit_species = 'Solanum lycopersicum'
                    fruit_msg.fruit_type = 'Tomato'
                    fruit_msg.fruit_variety = 'Plum'
                    fruit_msg.fruit_genotype = 'San Marzano'
                else:
                    ### Strawberry Fruit Biological Features ####
                    fruit_msg.pomological_class = 'Aggregate'
                    fruit_msg.edible_plant_part = 'Other'
                    fruit_msg.fruit_family = 'Unknown'
                    fruit_msg.fruit_species = 'Unknown'
                    fruit_msg.fruit_type = 'Strawberry'
                    fruit_msg.fruit_variety = 'Unknown'
                    fruit_msg.fruit_genotype = 'Unknown'

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
                fruit_msg.pose2d = self.compute_pose2d(segmentation, True)
                fruit_msg.mask3d = segmentation
                fruit_msg.pose3d = self.compute_pose3d(fruit_msg.pose2d, depth_mask)
                fruit_msg.confidence = 0.93
                fruit_msg.occlusion_level = 0.88

                # Log and publish the message
                #self.get_logger().info(f'Publishing: image_id={fruit_msg.image_id}, fruit_id={fruit_msg.fruit_id}, type={fruit_msg.fruit_type}, variety={fruit_msg.fruit_variety}, ripeness={fruit_msg.ripeness_category}')
                #self.get_logger().info(f'Publishing pose of fruit: {fruit_msg.pose2d}')
                #self.get_logger().info(f'Publishing pose of fruit: {fruit_msg.pose3d}')
                #self.get_logger().info(f'Depth values: {depth_mask}')
                fruits_msg.fruits.append(fruit_msg)
            fruits_msg.rgb_image = rgb_msg        # Assign the current RGB image
            fruits_msg.depth_image = depth_msg    # Assign the stored depth image

            fruits_msg.rgb_image_composed = self.add_markers_on_image(self.cv_image, fruits_msg)
            self.publisher_fruit.publish(fruits_msg)
            self.publisher_comp.publish(fruits_msg.rgb_image_composed)
            self.get_logger().info("Published")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def add_markers_on_image(self, cv_image, fruits_info):
        height, width, _ = cv_image.shape  # Get image dimensions
        scale_factor = min(width, height) / 750  # Scale the circle size based on image dimensions
        
        for fruit in fruits_info.fruits:
            x = int(fruit.pose2d.x)
            y = int(fruit.pose2d.y)
            
            # Set color based on ripeness
            if fruit.ripeness_level < 0.5:
                color = (0, 255, 0)  # Green for unripe
            else:
                color = (0, 0, 255)  # Red for ripe
            
            # Scale the radius based on the image size (e.g., a base of 10 pixels scaled)
            radius = int(10 * scale_factor)
            
            # Draw the circle marker
            cv2.circle(cv_image, (y, x), radius, color, -1)
            
            # Draw the mask outline
            mask_points = np.array(fruit.mask2d, dtype=np.int32).reshape((-1, 2))  # Convert 1D mask to Nx2 format
            
            # Draw the polygon mask outline
            cv2.polylines(cv_image, [mask_points], isClosed=True, color=color, thickness=3)

        # Convert the modified image back to a ROS image message
        composed_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        return composed_image

def main(args=None):
    rclpy.init(args=args)
    node = FruitDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()