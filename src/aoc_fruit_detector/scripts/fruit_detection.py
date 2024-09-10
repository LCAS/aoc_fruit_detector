#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from aoc_fruit_detector.msg import FruitInfoMessage
from predictor import call_predictor

import os, yaml, cv2
from detectron_predictor.detectron_predictor import DetectronPredictor

from cv_bridge import CvBridge, CvBridgeError

from rclpy.qos import QoSProfile, ReliabilityPolicy

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
                self.rgb_files = [f for f in os.listdir(test_image_dir) if os.path.isfile(os.path.join(test_image_dir, f))]
                self.test_image_dir = test_image_dir
                self.prediction_json_output_dir = prediction_json_output_dir
                self.rgb_files = os.listdir(test_image_dir)
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

        # Timer to call the publish_message method periodically
        #self.timer = self.create_timer(1.0, self.publish_message)
        self.count = 0
        self.process_and_publish()

    def find_data_folder_config(self,search_dir='.'):
        for root, dirs, _ in os.walk(search_dir):
            if 'data' in dirs:
                data_folder = os.path.join(root, 'data')
                config_path = os.path.join(data_folder, 'config', 'config.yaml')
                # Check if config.yaml exists in the data/config folder
                if os.path.exists(config_path):
                    return config_path
        return None

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            prediction_json_output_file = ""
            image_file_name = os.path.join(self.test_image_dir, 'image_from_ros.png')

            json_annotation_message, _ = self.det_predictor.get_predictions(
                self.cv_image, prediction_json_output_file, image_file_name
            )
            annotations = json_annotation_message.get('annotations', [])
            categories = json_annotation_message.get('categories', [])

            for annotation in annotations:
                id = annotation.get('id', None)
                image_id = annotation.get('image_id', self.count)
                category_id = annotation.get('category_id', -1)
                segmentation = annotation.get('segmentation', [])
                segmentation = [point for sublist in segmentation for point in sublist]  # Flatten segmentation
                bbox = annotation.get('bbox', [0.0, 0.0, 0.0, 0.0])
                area = float(annotation.get('area', 0.0))

                category_details = next(
                    (category for category in categories if category.get('id') == category_id),
                    {'name': 'unknown', 'supercategory': 'unknown'}
                )
                ripeness = category_details.get('name', 'unknown')

                msg = FruitInfoMessage()
                msg.header = Header()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'tomato_plant'
                msg.fruit_id = id
                msg.fruit_type = 'Tomato'
                msg.fruit_variety = 'Plum'
                msg.ripeness = ripeness
                msg.bbox = bbox
                msg.mask = segmentation
                msg.area = area

                # Log and publish the message
                self.get_logger().info(f'Publishing: id={msg.fruit_id}, type={msg.fruit_type}, variety={msg.fruit_variety}, ripeness={msg.ripeness}')
                self.publisher_.publish(msg)
                self.count += 1
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_and_publish(self):
        try:
            for rgb_file in self.rgb_files:
                image_file_name = os.path.join(self.test_image_dir, rgb_file)
                rgb_image = cv2.imread(image_file_name)
                filename, _ = os.path.splitext(rgb_file)
                if self.prediction_json_output_dir != "":
                    prediction_json_output_file = os.path.join(self.prediction_json_output_dir, filename) + '.json'
                else:
                    prediction_json_output_file = ""

                json_annotation_message, _ = self.det_predictor.get_predictions(
                        rgb_image, prediction_json_output_file, image_file_name)
                
                annotations = json_annotation_message.get('annotations', [])
                categories = json_annotation_message.get('categories', [])

                if not isinstance(annotations, list) or not isinstance(categories, list):
                    raise TypeError("Expected 'annotations' and 'categories' to be lists")
                
                for annotation in annotations:
                    id = annotation.get('id', None)  
                    image_id = annotation.get('image_id', self.count) 
                    category_id = annotation.get('category_id', -1) 
                    segmentation = annotation.get('segmentation', [])
                    segmentation = [point for sublist in segmentation for point in sublist]  
                    bbox = annotation.get('bbox', [0.0, 0.0, 0.0, 0.0])  
                    area = float(annotation.get('area', 0.0))  

                    category_details = next(
                        (category for category in categories if category.get('id') == category_id),
                        {'name': 'unknown', 'supercategory': 'unknown'}
                    )
                    ripeness = category_details.get('name', 'unknown')
                    #category_supercategory = category_details.get('supercategory', 'unknown')

                    '''print(f"id: {id}")
                    print(f"image_id: {image_id}")
                    print(f"category_id: {category_id}")
                    print(f"bbox: {bbox}")
                    print(f"area: {area}")
                    print(f"segmentation: {segmentation}")'''

                    msg = FruitInfoMessage()
                    msg.header = Header()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = 'tomato_plant'
                    #msg.header.seq = self.count
                    msg.fruit_id = self.count
                    msg.fruit_type = 'Tomato'
                    msg.fruit_variety = 'Plum'
                    msg.ripeness = ripeness
                    msg.bbox = bbox
                    msg.mask = segmentation
                    msg.area = area
                
                    # Log and publish the message
                    #self.get_logger().info(f'Publishing: id={msg.fruit_id}, type={msg.fruit_type}, variety={msg.fruit_variety}, ripeness={msg.ripeness}')
                    #self.publisher_.publish(msg)
                    #self.count += 1
        
        except StopIteration:
            self.get_logger().info('No more predictions available.')
        except TypeError as e:
            self.get_logger().error(f'Type error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing and publishing: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = FruitDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()