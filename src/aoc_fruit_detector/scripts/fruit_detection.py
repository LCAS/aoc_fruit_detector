#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D, Pose, PoseStamped
from aoc_fruit_detector.msg import FruitInfoMessage, FruitInfoArray
from visualization_msgs.msg import Marker, MarkerArray
#from predictor import call_predictor

import os, yaml, cv2
from detectron_predictor.detectron_predictor import DetectronPredictor

from cv_bridge import CvBridge, CvBridgeError

from rclpy.qos import QoSProfile, ReliabilityPolicy

import numpy as np

import image_geometry

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

class FruitDetectionNode(Node):
    def __init__(self):
        super().__init__('fruit_detection')
        # Create a publisher for the custom message type
        self.publisher_fruit = self.create_publisher(FruitInfoArray, 'fruit_info', 5)
        self.publisher_comp = self.create_publisher(Image, 'image_composed', 5)
        self.publisher_3dmarkers = self.create_publisher(MarkerArray, 'fruit_markers', 5)
        self.package_name = 'aoc_fruit_detector'
        config_path = self.get_parameters_file()
        if config_path:
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                
                for section in ['files', 'directories']:
                    if section in config_data:
                        for key, path in config_data[section].items():
                            if path.startswith('./'):
                                package_share_directory = get_package_share_directory(self.package_name)
                                config_data[section][key] = os.path.join(package_share_directory, path.lstrip('./'))

                self.det_predictor = DetectronPredictor(config_data)

                # Declare parameters for min_depth and max_depth
                self.declare_parameter('min_depth', 0.1)  # Default value
                self.declare_parameter('max_depth', 15.0)  # Default value
                self.min_depth = self.get_parameter('min_depth').value
                self.max_depth = self.get_parameter('max_depth').value
        else:
            raise FileNotFoundError(f"No config file found in any ' {self.package_name}/config/' folder within {os.getcwd()}")

        self.bridge = CvBridge()
        self.camera_model = image_geometry.PinholeCameraModel()
        self.set_default_camera_model() # in case camera_info message not available 
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

        self.declare_parameter('constant_depth_value', 1.0)
        self.constant_depth_value = self.get_parameter('constant_depth_value').value

        self.tomato = False
        self.pose3d_frame = ''

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
            if 'aoc_fruit_detector' in dirs:
                data_folder = os.path.join(root, 'aoc_fruit_detector')
                config_path = os.path.join(data_folder, 'config', 'parameters.yaml')
                # Check if parameters.yaml exists in the aoc_fruit_detector/config folder
                if os.path.exists(config_path):
                    return config_path
        return None
    
    def get_parameters_file(self):
        try:
            package_share_directory = get_package_share_directory(self.package_name)
        except PackageNotFoundError:
            raise FileNotFoundError("Package '{self.package_name}' not found in the workspace.")
        
        config_path = os.path.join(package_share_directory, 'config', 'parameters.yaml')
        
        # Check if parameters.yaml exists at the expected location
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config file found at '{config_path}'")
        
        return config_path

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
    
    def publish_fruit_markers(self, fruits_msg):
        marker_array = MarkerArray()

        # Loop over the detected fruits and create a marker for each one
        for i, fruit_msg in enumerate(fruits_msg.fruits):
            marker = self.create_fruit_marker(fruit_msg, i)  # Create a marker with unique ID
            marker_array.markers.append(marker)

        # Publish the marker array
        self.publisher_3dmarkers.publish(marker_array)

    def create_fruit_marker(self, fruit_msg, marker_id):
        marker = Marker()

        marker.header.frame_id = self.pose3d_frame
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "fruits"
        marker.id = marker_id

        marker.type = Marker.SPHERE  # You can choose other marker types like CUBE, ARROW, etc.
        marker.action = Marker.ADD

        # Set the pose using the fruit's 3D pose
        marker.pose.position.x = fruit_msg.pose3d.pose.position.x
        marker.pose.position.y = fruit_msg.pose3d.pose.position.y
        marker.pose.position.z = fruit_msg.pose3d.pose.position.z

        # Set orientation (same as the pose you computed earlier)
        marker.pose.orientation = fruit_msg.pose3d.pose.orientation

        # Set the scale of the marker (size of the fruit marker)
        marker.scale.x = 0.02
        marker.scale.y = 0.02
        marker.scale.z = 0.02

        # Set the color (RGBA)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0  # Fully opaque

        marker.lifetime = rclpy.time.Duration(seconds=0).to_msg()

        return marker

    def compute_pose3d(self, pose2d, depth_mask):
        pose3d = PoseStamped()

        height, width, _ = depth_mask.shape
        x = int(pose2d.x)
        y = int(pose2d.y)

        if 0 <= x < height and 0 <= y < width:
            depth_values_at_pose = depth_mask[x, y, :]
            non_zero_depth_values = depth_values_at_pose[depth_values_at_pose > 0]

            if non_zero_depth_values.size > 0:
                closest_depth_value = np.min(non_zero_depth_values)
            else:
                closest_depth_value = 0.0 
        else:
            closest_depth_value = 0.0
            self.get_logger().warn(f'Out of size x:{x}, height:{height}, y:{y} and width:{width}')
        
        ray = self.back_project_2d_to_3d_ray(pose2d.x, pose2d.y)
        p_3d_camera_frame = self.compute_3d_point_from_depth(ray, closest_depth_value)
        #self.get_logger().info(f'3D point at depth {closest_depth_value}: [{p_3d_camera_frame[0]:.2f}, {p_3d_camera_frame[1]:.2f}, {p_3d_camera_frame[2]:.2f}]')

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

    def set_default_camera_model(self):
        """
        Sets the camera model to default intrinsic parameters (pinhole model)
        """
        default_fx = 525.0  # Focal length in x (pixels)
        default_fy = 525.0  # Focal length in y (pixels)
        default_cx = 319.5  # Principal point x (image center in pixels)
        default_cy = 239.5  # Principal point y (image center in pixels)
        image_width = 640
        image_height = 480

        # Create a fake CameraInfo message to initialize the camera model
        camera_info = CameraInfo()
        camera_info.width = image_width
        camera_info.height = image_height
        camera_info.distortion_model = "plumb_bob"  # Default distortion model

        # Set the intrinsic camera matrix K (3x3 matrix)
        camera_info.k = [default_fx, 0.0, default_cx, 0.0, default_fy, default_cy, 0.0, 0.0, 1.0]

        # Set the projection matrix P (3x4 matrix)
        camera_info.p = [default_fx, 0.0, default_cx, 0.0, 0.0, default_fy, default_cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        # Set default distortion coefficients (D)
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Initialize the camera model with the default CameraInfo
        self.camera_model.fromCameraInfo(camera_info)

        self.get_logger().info("Default camera model initialised with intrinsic parameters")

    def camera_info_callback(self, msg):
        
        self.from_camera_info(msg)
        #self.camera_model, self.distortion_coeffs = self.from_camera_info(msg)

        self.pose3d_frame = msg.header.frame_id

        self.get_logger().info('Camera model acquired from camera_info message and initialised with intrinsic parameters')
    
    def from_camera_info(self, msg):
        self.camera_model.fromCameraInfo(msg)
        #camera_matrix = np.array(msg.k).reshape(3, 3)
        #distortion_coeffs = np.array(msg.d)
        #return camera_matrix, distortion_coeffs

    def back_project_2d_to_3d_ray(self, u, v):
        #ray = self.camera_model.projectPixelTo3dRay((u, v))
        v_flipped = self.camera_model.height - v
        ray = self.camera_model.projectPixelTo3dRay((u, v_flipped))
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
            # To replace NaN and Inf with max_depth value
            self.cv_depth_image[np.isnan(self.cv_depth_image)] = self.max_depth
            self.cv_depth_image[np.isinf(self.cv_depth_image)] = self.max_depth
            self.depth_msg = msg 
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")
    
    def create_confidence_dict(self, confidence_list):
        # Create a dictionary with annotation_id as the key and confidence as the value
        return {entry['annotation_id']: entry['confidence'] for entry in confidence_list}


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

            #info = json_annotation_message.get('info', [])
            #licenses = json_annotation_message.get('licenses', [])
            #self.get_logger().info(f"Info: {info}")
            #self.get_logger().info(f"License: {licenses}")
            #image_info = json_annotation_message.get('images', [])
            #self.get_logger().info(f"images: {image_info}")
            annotations = json_annotation_message.get('annotations', [])
            confidence_list = json_annotation_message.get('confidence', [])
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
            
            confidence_dict = self.create_confidence_dict(confidence_list)

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
                fruit_msg.confidence = float(confidence_dict.get(fruit_id, '-1.0'))
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
            self.publish_fruit_markers(fruits_msg)
            self.publisher_fruit.publish(fruits_msg)
            self.publisher_comp.publish(fruits_msg.rgb_image_composed)
            self.get_logger().info("Published")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def add_markers_on_image(self, cv_image, fruits_info):
        height, width, _ = cv_image.shape  # Get image dimensions
        scale_factor = min(width, height) / 1500  # Scale the circle size based on image dimensions
        
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
            cv2.polylines(cv_image, [mask_points], isClosed=True, color=color, thickness=5)

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