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

import matplotlib.pyplot as plt

import tf2_ros
from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation as R

from rclpy.qos import qos_profile_sensor_data

# RGB-D SYNC: pair images by their recorded ROS timestamps, not arrival order.
from message_filters import Subscriber, ApproximateTimeSynchronizer

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
                ('pose3d_tf', False),
                ('verbose', [False, False, False, True, True]),
                ('pub_verbose', False),
                ('pub_markers', False),
                ('use_ros', True),
                # VISUAL ODOMETRY: reconstruct relative camera motion from RGB-D bag frames.
                ('enable_rgbd_odometry', True),
                ('odometry_frame', 'vo_map'),
                ('odometry_min_inliers', 15),
                ('odometry_max_reprojection_error', 3.0)
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
                self.filename_patterns = config_data['settings']['filename_patterns']
        else:
            raise FileNotFoundError(f"No config file found in any ' {self.package_name}/config/' folder within {os.getcwd()}")

        self.use_ros = self.get_parameter('use_ros').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value

        if self.use_ros:
            self.get_logger().info(f"ROS2 pipeline is active")
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            # Get parameter values from ROS2 network
            self.constant_depth_value = self.get_parameter('constant_depth_value').value
            if self.get_parameter('fruit_type').value == "tomato":
                self.tomato = True
            elif self.get_parameter('fruit_type').value == "strawberry":
                self.tomato = False
            self.pose3d_frame = self.get_parameter('pose3d_frame').value

            self.pose3d_tf = self.get_parameter('pose3d_tf').value

            # VISUAL ODOMETRY: accumulated transform from camera optical coordinates
            # to a local, arbitrary-origin world frame (initial camera = origin).
            self.vo_enabled = bool(self.get_parameter('enable_rgbd_odometry').value)
            self.vo_frame = str(self.get_parameter('odometry_frame').value)
            self.vo_min_inliers = int(self.get_parameter('odometry_min_inliers').value)
            self.vo_max_reprojection_error = float(self.get_parameter('odometry_max_reprojection_error').value)
            self.vo_T_world_camera = np.eye(4, dtype=np.float64)
            self.vo_valid = False
            self.vo_reference = None
            self.vo_orb = cv2.ORB_create(nfeatures=2000)
            self.vo_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            self.vo_tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            if self.vo_enabled and self.pose3d_tf:
                raise ValueError('RGB-D odometry and pose3d_tf cannot both be enabled')

            self.draw_centroid = self.get_parameter('verbose').value[0]
            self.draw_bbox = self.get_parameter('verbose').value[1]
            self.draw_mask = self.get_parameter('verbose').value[2]
            self.draw_cf = self.get_parameter('verbose').value[3]
            self.add_text = self.get_parameter('verbose').value[4]

            self.pub_verbose = self.get_parameter('pub_verbose').value
            self.pub_markers = self.get_parameter('pub_markers').value

            self.bridge = CvBridge()
            self.camera_model = image_geometry.PinholeCameraModel()
            self.set_default_camera_model() # in case camera_info calibration message not available

            # RGB-D SYNC: both bag topics use sensor-data QoS. Process only
            # timestamp-matched pairs; never reuse a previous depth frame.
            self.image_sub = Subscriber(
                self, Image, 'camera/image_raw', qos_profile=qos_profile_sensor_data
            )
            self.depth_sub = Subscriber(
                self, Image, 'camera/depth', qos_profile=qos_profile_sensor_data
            )
            self.rgbd_sync = ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub], queue_size=30, slop=0.03
            )
            self.rgbd_sync.registerCallback(self.image_callback)

            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                'camera/camera_info',
                self.camera_info_callback,
                qos_profile=qos_profile_sensor_data
            )

            # Create and declare publishers
            self.publisher_fruit = self.create_publisher(FruitInfoArray, 'fruit_info', 5)
            self.publisher_comp = self.create_publisher(Image, 'image_composed', 5)
            self.publisher_3dmarkers = self.create_publisher(MarkerArray, 'fruit_markers', 5)
            # PERSISTENT MARKERS: Allocate a fresh marker ID for every valid detection.
            # Reusing the per-frame detection index would move/overwrite older dots.
            self.next_fruit_marker_id = 0
        else:
            self.get_logger().info(f"Non-ROS configuration is active")
            all_files = sorted([f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))])

            RGB_PATTERN = self.filename_patterns['rgb']
            DEPTH_PATTERN = self.filename_patterns['depth']

            rgb_files = sorted([f for f in all_files if RGB_PATTERN in f])
            depth_files = sorted([f for f in all_files if DEPTH_PATTERN in f])

            sample_no = 1
            for rgb_file in rgb_files:
                corr_depth_file = rgb_file.replace(RGB_PATTERN, DEPTH_PATTERN, 1)

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
                    self.get_logger().warn(f"Warning: No corresponding depth file: {corr_depth_file} for rgb file: {rgb_file}.\nPredicting using rgb only.")
                    image_file_name=os.path.join(self.image_dir, rgb_file)
                    rgb_image = cv2.imread(image_file_name)  # bgr8
                    filename, extension = os.path.splitext(rgb_file)
                    if (self.prediction_json_dir!=""):
                        os.makedirs(self.prediction_json_dir, exist_ok=True)
                        prediction_json_output_file = os.path.join(self.prediction_json_dir, filename)+'.json'

                    self.det_predictor.get_rgb_predictions_image(rgb_image, prediction_json_output_file, self.prediction_output_dir, image_file_name, sample_no, self.fruit_type)

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

        # PERSISTENT MARKERS: Do not issue Marker.DELETEALL between frames.
        # Keep prior valid strawberry locations visible even when detection or
        # visual odometry temporarily fails to supply new world coordinates.
        for fruit_msg in fruits_msg.fruits:
            position = fruit_msg.pose3d.pose.position
            if not np.all(np.isfinite([position.x, position.y, position.z])):
                continue
            # PERSISTENT MARKERS: Distinct IDs prevent subsequent detections from
            # replacing earlier markers (marker IDs were previously reset to 0).
            marker = self.create_fruit_marker(fruit_msg, self.next_fruit_marker_id)
            self.next_fruit_marker_id += 1
            marker_array.markers.append(marker)

        if marker_array.markers:
            self.publisher_3dmarkers.publish(marker_array)

    def create_fruit_marker(self, fruit_msg, marker_id):
        marker = Marker()

        # VISUAL ODOMETRY: marker and position must use the SAME frame.
        marker.header.frame_id = fruit_msg.pose3d.header.frame_id
        marker.header.stamp = fruit_msg.pose3d.header.stamp

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
        marker.color.a = 1.0

        # PERSISTENT MARKERS: Zero lifetime means RViz retains the marker
        # until explicitly deleted or the display is reset.
        marker.lifetime = rclpy.time.Duration(seconds=0).to_msg()

        return marker

    # DEPTH FIX: Estimate the fruit's depth from the full COCO segmentation polygon,
    # rather than trusting a single centroid pixel. None means no reliable depth.
    def estimate_fruit_depth(self, depth_image, segmentation):
        height, width = depth_image.shape[:2]
        polygon = np.asarray(segmentation, dtype=np.float32)
        if polygon.size < 6 or polygon.size % 2 != 0 or not np.all(np.isfinite(polygon)):
            return None

        polygon = np.rint(polygon.reshape(-1, 2)).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        if cv2.contourArea(polygon) <= 0:
            return None

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        values = depth_image[mask.astype(bool)]
        valid = values[np.isfinite(values) & (values >= self.min_depth) & (values <= self.max_depth)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def compute_pose3d(self, pose2d, depth_image, segmentation=None):
        pose3d = PoseStamped()
        # VISUAL ODOMETRY: world positions are only valid when motion tracking succeeds.
        pose3d.header.frame_id = self.vo_frame if self.vo_enabled else self.pose3d_frame
        pose3d.header.stamp = self.current_image_stamp if hasattr(self, 'current_image_stamp') else self.get_clock().now().to_msg()
        pose3d.pose.orientation.w = 1.0

        # DEPTH FIX: Return NaN coordinates if the centroid is invalid instead
        # of inventing a location at max_depth.
        if not np.isfinite(pose2d.x) or not np.isfinite(pose2d.y):
            self.get_logger().warn('Cannot compute 3D position: invalid fruit centroid')
            pose3d.pose.position.x = float('nan')
            pose3d.pose.position.y = float('nan')
            pose3d.pose.position.z = float('nan')
            return pose3d

        height, width = depth_image.shape[:2]
        x = int(pose2d.x)
        y = int(pose2d.y)
        if not (0 <= x < width and 0 <= y < height):
            self.get_logger().warn(f'Out of size x:{x}, width:{width}, y:{y} and height:{height}')
            pose3d.pose.position.x = float('nan')
            pose3d.pose.position.y = float('nan')
            pose3d.pose.position.z = float('nan')
            return pose3d

        # DEPTH FIX: Prefer median valid depth across the segmented strawberry.
        # If its polygon contains no valid depth, do not substitute 15 m.
        depth = self.estimate_fruit_depth(depth_image, segmentation or [])
        if depth is None:
            self.get_logger().warn(f'No valid depth inside fruit mask at ({x}, {y}); 3D pose unavailable')
            pose3d.pose.position.x = float('nan')
            pose3d.pose.position.y = float('nan')
            pose3d.pose.position.z = float('nan')
            return pose3d

        ray = self.back_project_2d_to_3d_ray(pose2d.x, pose2d.y)
        p_3d_camera_frame = self.compute_3d_point_from_depth(ray, depth)
        if self.vo_enabled:
            if not self.vo_valid:
                # VISUAL ODOMETRY: never publish guessed world coordinates after tracking loss.
                p_3d_camera_frame = [float('nan')] * 3
            else:
                homogeneous = np.array([*p_3d_camera_frame[:3], 1.0], dtype=np.float64)
                p_3d_camera_frame = (self.vo_T_world_camera @ homogeneous)[:3]
        pose3d.pose.position.x = float(p_3d_camera_frame[0])
        pose3d.pose.position.y = float(p_3d_camera_frame[1])
        pose3d.pose.position.z = float(p_3d_camera_frame[2])
        return pose3d

    def set_default_camera_model(self):
        """
        Sets the camera model to default intrinsic parameters (pinhole model)
        """
        default_fx = 699.05  # Focal length in x (pixels)
        default_fy = 699.05  # Focal length in y (pixels)
        default_cx = 639.74  # Principal point x (image center in pixels)
        default_cy = 374.974  # Principal point y (image center in pixels)
        image_width = 1280
        image_height = 720

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
        # VISUAL ODOMETRY: the supplied RGB-D bag contains no CameraInfo.
        # The launch remaps CameraInfo to FLIR, which is a different camera.
        # Do not overwrite the ZED default intrinsics with FLIR calibration.
        if self.vo_enabled:
            return
        self.from_camera_info(msg)
        self.pose3d_frame = msg.header.frame_id
        self.get_logger().info('Camera model acquired from camera_info message and initialised with intrinsic parameters')

    def from_camera_info(self, msg):
        self.camera_model.fromCameraInfo(msg)

    def back_project_2d_to_3d_ray(self, u, v):
        ray = self.camera_model.projectPixelTo3dRay((u, v))
        return ray

    def compute_3d_point_from_depth(self, ray, depth):
        # Compute the 3D point in optical frame
        point_optical = np.array([ray[0] * depth, ray[1] * depth, ray[2] * depth, 1.0])
        if self.pose3d_tf:
            # Apply the transformation to convert to camera frame
            point_camera = np.dot(self.tf_matrix, point_optical)
            return point_camera[:3]
        else:
            return point_optical

    def create_confidence_dict(self, confidence_list):
        # Create a dictionary with annotation_id as the key
        return {entry['annotation_id']: entry['confidence'] for entry in confidence_list}

    def create_pose_dict(self, pose_list):
        return {entry['annotation_id']: (entry['centroid'], entry['orientation']) for entry in pose_list}

    def get_optic_tf(self):
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'zed_camera_link', 'zed_left_camera_optical_frame', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5)
            )
            tr = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
            q = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            self.get_logger().info(f"TF between camera and optical frame: t={tr} and q={q}")
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, TimeoutError):
            tr = [0.0, 0.0, 0.0]
            q = R.from_quat([0.5, -0.5, 0.5, -0.5]).as_quat() # default orientation between camera and optical frames
            self.get_logger().warn(f"Default transform between camera and optical frame used: t={tr} and q={q}")
        rot_matrix = R.from_quat(q).as_matrix()
        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = rot_matrix
        tf_matrix[:3, 3] = tr
        return tf_matrix

    # VISUAL ODOMETRY: estimate current camera pose using background RGB features
    # paired with depth from the previous *successfully tracked* frame.
    # solvePnPRansac estimates previous-camera -> current-camera; invert it
    # before accumulating the world -> camera coordinate conversion.
    def update_rgbd_odometry(self, image, depth, stamp):
        if not self.vo_enabled:
            return
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.vo_orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 20:
            self.vo_valid = False
            self.get_logger().warn('VO: too few image features; no world positions this frame')
            return

        K = np.asarray(self.camera_model.intrinsicMatrix(), dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            self.vo_valid = False
            self.get_logger().warn('VO: invalid camera intrinsics')
            return

        current = (keypoints, descriptors, depth.copy())
        if self.vo_reference is None:
            self.vo_reference = current
            self.vo_valid = True  # Initial frame establishes the local origin.
            self.publish_vo_transform(stamp)
            self.get_logger().info('VO: initial camera frame established as local origin')
            return

        prev_keypoints, prev_descriptors, prev_depth = self.vo_reference
        pairs = self.vo_matcher.knnMatch(prev_descriptors, descriptors, k=2)
        good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.70 * n.distance]
        xyz_prev, uv_curr = [], []
        h, w = prev_depth.shape[:2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        for m in good:
            u, v = prev_keypoints[m.queryIdx].pt
            x, y = int(round(u)), int(round(v))
            if not (0 <= x < w and 0 <= y < h):
                continue
            z = float(prev_depth[y, x])
            if not (np.isfinite(z) and self.min_depth <= z <= self.max_depth):
                continue
            xyz_prev.append([(u - cx) * z / fx, (v - cy) * z / fy, z])
            uv_curr.append(keypoints[m.trainIdx].pt)

        if len(xyz_prev) < max(8, self.vo_min_inliers):
            self.vo_valid = False
            self.get_logger().warn(f'VO: only {len(xyz_prev)} valid RGB-D correspondences; no world positions')
            return
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.asarray(xyz_prev, dtype=np.float64),
            np.asarray(uv_curr, dtype=np.float64),
            K, None, flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=150, reprojectionError=self.vo_max_reprojection_error,
            confidence=0.99)
        if not ok or inliers is None or len(inliers) < self.vo_min_inliers:
            self.vo_valid = False
            self.get_logger().warn('VO: PnP motion estimation failed; no world positions')
            return
        # Refine the transform using only RANSAC inliers.
        idx = inliers.ravel()
        ok, rvec, tvec = cv2.solvePnP(
            np.asarray(xyz_prev, np.float64)[idx],
            np.asarray(uv_curr, np.float64)[idx], K, None,
            rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            self.vo_valid = False
            return
        R_curr_prev, _ = cv2.Rodrigues(rvec)
        T_curr_prev = np.eye(4, dtype=np.float64)
        T_curr_prev[:3, :3] = R_curr_prev
        T_curr_prev[:3, 3] = tvec.reshape(3)
        self.vo_T_world_camera = self.vo_T_world_camera @ np.linalg.inv(T_curr_prev)
        self.vo_reference = current
        self.vo_valid = True
        self.publish_vo_transform(stamp)
        self.get_logger().info(f'VO: tracking, {len(idx)} inliers; camera at {self.vo_T_world_camera[:3, 3].round(3).tolist()} m')

    def publish_vo_transform(self, stamp):
        # VISUAL ODOMETRY: RViz Fixed Frame can be set to 'vo_map'.
        # This is a local drifting frame, NOT surveyed robot/world position.
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.vo_frame
        transform.child_frame_id = self.pose3d_frame
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = (
            float(v) for v in self.vo_T_world_camera[:3, 3])
        q = R.from_matrix(self.vo_T_world_camera[:3, :3]).as_quat()
        transform.transform.rotation.x = float(q[0])
        transform.transform.rotation.y = float(q[1])
        transform.transform.rotation.z = float(q[2])
        transform.transform.rotation.w = float(q[3])
        if transform.child_frame_id:
            self.vo_tf_broadcaster.sendTransform(transform)

    def image_callback(self, msg, depth_msg):
        try:
            self.get_logger().info("Image captured.")
            # VISUAL ODOMETRY: bypass the unrelated camera-link TF lookup.
            self.tf_matrix = np.eye(4) if self.vo_enabled else self.get_optic_tf()
            # Convert ROS Image message to OpenCV image
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info("RGB ready.")
            image_id = int(f'{msg.header.stamp.sec}{str(msg.header.stamp.nanosec).zfill(9)}')
            self.get_logger().info(f"Image ID is: {image_id}")

            rgb_msg = msg
            self.current_image_stamp = msg.header.stamp
            # VISUAL ODOMETRY: camera optical frame is taken from image headers.
            # The bag has no CameraInfo messages, so do not rely on a live FLIR frame.
            if self.vo_enabled:
                self.pose3d_frame = msg.header.frame_id
            # RGB-D SYNC: depth_msg is paired with this RGB frame by its
            # header timestamp. Conversion preserves NaN/Inf for filtering.
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            if self.cv_image.shape[:2] != depth_image.shape[:2]:
                self.get_logger().warn("Resizing paired depth image to RGB dimensions.")
                depth_image = cv2.resize(
                    depth_image, (self.cv_image.shape[1], self.cv_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # RGB-D SYNC: the synchronizer enforces a 30 ms pairing window.
            # Feed this exact matched pair into visual odometry.
            if self.vo_enabled:
                self.update_rgbd_odometry(self.cv_image, depth_image, msg.header.stamp)
            self.get_logger().info("Depth ready")
            # Combine RGB and depth into a single 4-channel image (3 for RGB + 1 for depth)
            rgbd_image = np.dstack((self.cv_image, depth_image))
            self.get_logger().info("RGBD ready")

            json_annotation_message, _, rgb_masks, depth_mask = self.det_predictor.get_predictions_message(rgbd_image, image_id, self.fruit_type)

            annotations = json_annotation_message.get('annotations', [])
            confidence_list = json_annotation_message.get('confidence', [])
            pose_list = json_annotation_message.get('orientation', [])
            categories = json_annotation_message.get('categories', [])

            fruits_msg = FruitInfoArray()
            fruits_msg.fruits = []
            confidence_dict = self.create_confidence_dict(confidence_list)
            pose_dict = self.create_pose_dict(pose_list)

            for annotation in annotations:
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
                fruit_msg.volume = area*2 # AY: To be developed
                fruit_msg.bbox = bbox
                fruit_msg.bvol = bbox # AY: To be developed
                fruit_msg.mask2d = segmentation
                fruit_msg.pose2d = self.compute_pose2d(fruit_id, pose_dict)
                fruit_msg.mask3d = segmentation # AY: To be developed
                # DEPTH FIX: Pass the segmentation polygon so the 3D pose uses
                # a robust median depth across the fruit instead of one pixel.
                fruit_msg.pose3d = self.compute_pose3d(fruit_msg.pose2d, depth_image, segmentation)
                fruit_msg.confidence = float(confidence_dict.get(fruit_id, '-1.0'))
                fruit_msg.occlusion_level = 0.88
                fruits_msg.fruits.append(fruit_msg)
            if self.pub_verbose:
                fruits_msg.rgb_image = rgb_msg
                fruits_msg.depth_image = depth_msg
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
        height, width, _ = cv_image.shape
        scale_factor = min(width, height) / 1500
        for fruit in fruits_info.fruits:
            x = int(fruit.pose2d.x)
            y = int(fruit.pose2d.y)
            theta = np.deg2rad(fruit.pose2d.theta)
            if fruit.ripeness_level < 0.5:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            if self.draw_centroid:
                radius = int(10 * scale_factor)
                cv2.circle(cv_image, (x, y), radius, color, -1)
            if self.draw_mask:
                mask_points = np.array(fruit.mask2d, dtype=np.int32).reshape((-1, 2))
                cv2.polylines(cv_image, [mask_points], isClosed=True, color=color, thickness=2)
            if self.draw_bbox:
                x_min, y_min, width, height = map(int, fruit.bbox)
                x_max = x_min + width
                y_max = y_min + height
                cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), color, thickness=2)
            if self.draw_cf:
                color_x = (0, 0, 255)
                color_y = (0, 255, 0)
                arrow_length = int(30 * scale_factor)
                end_x_x = int(x + arrow_length * np.cos(theta))
                end_y_x = int(y + arrow_length * np.sin(theta))
                cv2.arrowedLine(cv_image, (x, y), (end_x_x, end_y_x), color_x, thickness=2, tipLength=0.3)
                end_x_y = int(x + arrow_length * np.cos(theta+np.pi/2))
                end_y_y = int(y + arrow_length * np.sin(theta+np.pi/2))
                cv2.arrowedLine(cv_image, (x, y), (end_x_y, end_y_y), color_y, thickness=2, tipLength=0.3)
            if self.add_text:
                text_position = (end_x_x + 5, end_y_x - 5)
                font_scale = 1.5 * scale_factor
                font_thickness = max(1, int(2 * scale_factor))
                cv2.putText(cv_image, f"{fruit.pose2d.theta:.1f}", text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0),
                            thickness=font_thickness, lineType=cv2.LINE_AA)
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
