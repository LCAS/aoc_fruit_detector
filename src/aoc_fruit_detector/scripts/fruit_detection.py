#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from aoc_fruit_detector.msg import FruitInfoMessage
from predictor import call_predictor

class FruitDetectionNode(Node):
    def __init__(self):
        super().__init__('fruit_detection')
        # Create a publisher for the custom message type
        self.publisher_ = self.create_publisher(FruitInfoMessage, 'fruit_info', 10)
        self.prediction_generator = call_predictor()
        # Timer to call the publish_message method periodically
        #self.timer = self.create_timer(1.0, self.publish_message)
        self.count = 0
        self.process_and_publish()

    def process_and_publish(self):
        try:
            json_annotation_message = next(self.prediction_generator)
            print(f"Received message type: {type(json_annotation_message)}")
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
                self.get_logger().info(f'Publishing: id={msg.fruit_id}, type={msg.fruit_type}, variety={msg.fruit_variety}, ripeness={msg.ripeness}')
                self.publisher_.publish(msg)
                self.count += 1

            self.process_and_publish()
        
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