#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D, Pose, PoseStamped
from aoc_fruit_detector.msg import FruitInfoMessage, FruitInfoArray
from visualization_msgs.msg import Marker, MarkerArray
import argparse
#from predictor import call_predictor

import os, yaml, cv2
from detectron_predictor.detectron_predictor import DetectronPredictor

from cv_bridge import CvBridge, CvBridgeError

from rclpy.qos import QoSProfile, ReliabilityPolicy

import numpy as np

import image_geometry

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from detectron_predictor.json_writer.pycococreator.pycococreatortools.fruit_orientation import FruitTypes

class FruitDetectionNode(Node):
    def __init__(self, non_ros_config_path):
        super().__init__('aoc_fruit_detector')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('min_depth', 0.1),
                ('max_depth', 15.0),
                ('constant_depth_value', 1.0),
                ('fruit_type', "strawberry"),
                ('pose3d_frame', ''),
                ('verbose', [False, False, False, True, True]),
                ('pub_verbose', False),
                ('pub_markers', False),
                ('use_ros', True)
            ]
        )

        self.package_name = 'aoc_fruit_detector'
        if non_ros_config_path:
            with open(non_ros_config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                
                for section in ['files', 'directories']:
                    if section in config_data:
                        for key, path in config_data[section].items():
                            if path.startswith('./'):
                                package_share_directory = get_package_share_directory(self.package_name)
                                config_data[section][key] = os.path.join(package_share_directory, path.lstrip('./'))
                
                self.image_dir = config_data['directories']['test_image_dir']
                self.prediction_json_dir = config_data['directories']['prediction_json_dir']
                self.prediction_output_dir = config_data['directories']['prediction_output_dir']

                self.det_predictor = DetectronPredictor(config_data)
                fruit_type = config_data['settings']['fruit_type']
                if (fruit_type.upper()=="STRAWBERRY"):
                    self.fruit_type=FruitTypes.Strawberry
                elif (fruit_type.upper()=="TOMATO"):
                    self.fruit_type=FruitTypes.Tomato
                else:
                    self.fruit_type=FruitTypes.Strawberry
        else:
            raise FileNotFoundError(f"No config file found in any ' {self.package_name}/config/' folder within {os.getcwd()}")

        self.use_ros = self.get_parameter('use_ros').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value

        if self.use_ros:
            # Get parameter values from ROS2 network
            self.constant_depth_value = self.get_parameter('constant_depth_value').value
            if self.get_parameter('fruit_type').value == "tomato":
                self.tomato = True
            elif self.get_parameter('fruit_type').value == "strawberry":
                self.tomato = False
            self.pose3d_frame = self.get_parameter('pose3d_frame').value

            self.draw_centroid = self.get_parameter('verbose').value[0]
            self.draw_bbox = self.get_parameter('verbose').value[1]
            self.draw_mask = self.get_parameter('verbose').value[2]
            self.draw_cf = self.get_parameter('verbose').value[3]
            self.add_text = self.get_parameter('verbose').value[4]

            self.pub_verbose = self.get_parameter('pub_verbose').value
            self.pub_markers = self.get_parameter('pub_markers').value

            self.bridge = CvBridge()
            self.camera_model = image_geometry.PinholeCameraModel()
            self.set_default_camera_model() # in case camera_info message not available 
            
            # Declare subscribers
            qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=1
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

            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                'camera/camera_info',
                self.camera_info_callback,
                qos_profile
            )

            # Create and declare publishers
            self.publisher_fruit = self.create_publisher(FruitInfoArray, 'fruit_info', 5)
            self.publisher_comp = self.create_publisher(Image, 'image_composed', 5)
            self.publisher_3dmarkers = self.create_publisher(MarkerArray, 'fruit_markers', 5)
        else: 
            all_files = sorted([f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))])

            rgb_files = sorted([f for f in all_files if 'image' in f])
            depth_files = sorted([f for f in all_files if 'depth' in f])

            sample_no = 1
            for rgb_file in rgb_files:
                corr_depth_file = rgb_file.replace('image', 'depth', 1)

                if corr_depth_file in depth_files:
                    image_file_name=os.path.join(self.image_dir, rgb_file)
                    depth_file_name = os.path.join(self.image_dir, corr_depth_file)
                    rgb_image = cv2.imread(image_file_name)  # bgr8
                    depth_image = cv2.imread(depth_file_name, cv2.IMREAD_UNCHANGED)
                    depth_image = np.nan_to_num(depth_image, nan=self.max_depth, posinf=self.max_depth, neginf=self.min_depth)
                    rgbd_image = np.dstack((rgb_image, depth_image))
                    filename, extension = os.path.splitext(rgb_file)
                    if (self.prediction_json_dir!=""):
                        os.makedirs(self.prediction_json_dir, exist_ok=True)
                        prediction_json_output_file = os.path.join(self.prediction_json_dir, filename)+'.json'
                    self.det_predictor.get_predictions_image(rgbd_image, prediction_json_output_file, self.prediction_output_dir, image_file_name, sample_no, self.fruit_type)
                else:
                    self.get_logger().warn(f"Warning: No corresponding depth file: {corr_depth_file} for rgb file: {rgb_file}")
                sample_no += 1

    def compute_pose2d(self, annotation_id, pose_dict):
        """
        Retrieve Pose2D from the pose_dict using the fruit_id.

        Args:
            annotation_id (int): The ID of the fruit (annotation ID).
            pose_dict (dict): Dictionary containing centroids and orientations indexed by annotation ID.

        Returns:
            Pose2D: The 2D pose (x, y, theta) for the given fruit_id.
        """

        pose2d = Pose2D()
        if annotation_id in pose_dict:
            centroid, orientation = pose_dict[annotation_id]
            pose2d.x = float(centroid[0])  # Centroid X
            pose2d.y = float(centroid[1])  # Centroid Y
            pose2d.theta = orientation  # Orientation (theta) in degree
            #self.get_logger().info(f'Fruit orientation (deg): {pose2d.theta}')
        else:
            # Default values
            pose2d.x = float('nan')
            pose2d.y = float('nan')
            pose2d.theta = float('nan')
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

    def compute_pose3d(self, pose2d, depth_image):
        pose3d = PoseStamped()

        height, width = depth_image.shape
        x = int(pose2d.x)
        y = int(pose2d.y)

        if 0 <= x < width and 0 <= y < height:
            depth_values_at_pose = depth_image[y, x]
            non_zero_depth_values = depth_values_at_pose[depth_values_at_pose > 0]

            if non_zero_depth_values.size > 0:
                closest_depth_value = np.min(non_zero_depth_values)
            else:
                closest_depth_value = self.max_depth 
        else:
            closest_depth_value = self.max_depth
            self.get_logger().warn(f'Out of size x:{x}, width:{width}, y:{y} and height:{height}')
        
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

    def create_pose_dict(self, pose_list):
        return {entry['annotation_id']: (entry['centroid'], entry['orientation']) for entry in pose_list}

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
            
            json_annotation_message, _, depth_mask = self.det_predictor.get_predictions_message(rgbd_image, image_id, self.fruit_type)

            #info = json_annotation_message.get('info', [])
            #licenses = json_annotation_message.get('licenses', [])
            #self.get_logger().info(f"Info: {info}")
            #self.get_logger().info(f"License: {licenses}")
            #image_info = json_annotation_message.get('images', [])
            #self.get_logger().info(f"images: {image_info}")
            annotations = json_annotation_message.get('annotations', [])
            confidence_list = json_annotation_message.get('confidence', [])
            pose_list = json_annotation_message.get('orientation', [])
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

            pose_dict = self.create_pose_dict(pose_list)

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
                fruit_msg.pose2d = self.compute_pose2d(fruit_id, pose_dict)
                fruit_msg.mask3d = segmentation
                fruit_msg.pose3d = self.compute_pose3d(fruit_msg.pose2d, depth_image)
                fruit_msg.confidence = float(confidence_dict.get(fruit_id, '-1.0'))
                fruit_msg.occlusion_level = 0.88
                # Log and publish the message
                #self.get_logger().info(f'Publishing: image_id={fruit_msg.image_id}, fruit_id={fruit_msg.fruit_id}, type={fruit_msg.fruit_type}, variety={fruit_msg.fruit_variety}, ripeness={fruit_msg.ripeness_category}')
                #self.get_logger().info(f'Publishing pose of fruit: {fruit_msg.pose2d}')
                #self.get_logger().info(f'Publishing pose of fruit: {fruit_msg.pose3d}')
                #self.get_logger().info(f'Depth values: {depth_mask}')
                fruits_msg.fruits.append(fruit_msg)
            if self.pub_verbose:
                fruits_msg.rgb_image = rgb_msg        # Assign the current RGB image
                fruits_msg.depth_image = depth_msg    # Assign the stored depth image
                fruits_msg.rgb_image_composed = self.add_markers_on_image(self.cv_image, fruits_msg)
                self.publisher_comp.publish(fruits_msg.rgb_image_composed)
            if self.pub_markers:
                self.publish_fruit_markers(fruits_msg)
            self.publisher_fruit.publish(fruits_msg)
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
            theta = np.deg2rad(fruit.pose2d.theta)
            
            # Set color based on ripeness
            if fruit.ripeness_level < 0.5:
                color = (0, 255, 0)  # Green for unripe
            else:
                color = (0, 0, 255)  # Red for ripe
            
            if self.draw_centroid:
                # Point the centroid (origin of the fruit)
                radius = int(10 * scale_factor)
                cv2.circle(cv_image, (x, y), radius, color, -1)
            
            if self.draw_mask:
                # Draw the polygon mask outline
                mask_points = np.array(fruit.mask2d, dtype=np.int32).reshape((-1, 2))  # Convert 1D mask to Nx2 format
                cv2.polylines(cv_image, [mask_points], isClosed=True, color=color, thickness=2)
            
            if self.draw_bbox:
                x_min, y_min, width, height = map(int, fruit.bbox)
        
                x_max = x_min + width
                y_max = y_min + height
                cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), color, thickness=2)

            if self.draw_cf:
                # Draw fruit coordinate frame (cf)
                color_x = (0, 0, 255) # Red for x axis
                color_y = (0, 255, 0)  # Green for y axis
                arrow_length = int(30 * scale_factor)
                end_x_x = int(x + arrow_length * np.cos(theta))  # Calculate endpoint x
                end_y_x = int(y + arrow_length * np.sin(theta))  # Calculate endpoint y
                cv2.arrowedLine(cv_image, (x, y), (end_x_x, end_y_x), color_x, thickness=2, tipLength=0.3)
                end_x_y = int(x + arrow_length * np.cos(theta+np.pi/2))  # Calculate endpoint x
                end_y_y = int(y + arrow_length * np.sin(theta+np.pi/2))  # Calculate endpoint y
                cv2.arrowedLine(cv_image, (x, y), (end_x_y, end_y_y), color_y, thickness=2, tipLength=0.3)

            if self.add_text:
                text_position = (end_x_x + 5, end_y_x - 5)  # Offset text slightly from the arrow tip
                font_scale = 1.5 * scale_factor  # Adjust text size based on image size
                font_thickness = max(1, int(2 * scale_factor))  # Scale text thickness
                cv2.putText(
                    cv_image, 
                    f"{fruit.pose2d.theta:.1f}",  # Format theta to 1 decimal place
                    text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (255, 0, 0),  # blue text
                    thickness=font_thickness, 
                    lineType=cv2.LINE_AA
                )

        # Convert the modified image back to a ROS image message
        composed_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        return composed_image

def main(args=None):
    parser = argparse.ArgumentParser(description='Fruit Detector Node')
    parser.add_argument('--config-file', required=True, help='Path to non-ROS parameters YAML file')
    non_ros_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = FruitDetectionNode(non_ros_config_path=non_ros_args.config_file)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()