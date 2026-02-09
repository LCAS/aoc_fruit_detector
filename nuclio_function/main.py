"""
Main nuclio function handler for AOC Fruit Detector integration with CVAT.

This module implements the serverless function interface required by CVAT
for automatic annotation using the AOC fruit detector.
"""

import json
import io
import base64
import logging
from typing import Dict, List, Any
import numpy as np
from PIL import Image
import cv2
import rclpy

from ros2_bridge import ROS2Bridge, initialize_ros2
from cvat_converter import CVATConverter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global ROS2 bridge instance
ros2_bridge = None
cvat_converter = None

def init_context(context):
    """
    Initialize the nuclio function context.
    Called once when the function is loaded.
    """
    global ros2_bridge, cvat_converter
    
    logger.info("Initializing AOC Fruit Detector nuclio function")
    
    try:
        # Initialize ROS2
        initialize_ros2()
        
        # Initialize ROS2 bridge and converter
        ros2_bridge = ROS2Bridge()
        cvat_converter = CVATConverter()
        
        logger.info("AOC Fruit Detector function initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize function: {e}")
        raise

def handler(context, event):
    """
    Main nuclio function handler.
    
    Expected input format from CVAT:
    {
        "image": "<base64_encoded_image>",
        "threshold": 0.5  # optional confidence threshold
    }
    
    Returns CVAT-compatible annotations:
    [
        {
            "confidence": 0.95,
            "label": "strawberry",
            "points": [x1, y1, x2, y2],  # bounding box
            "type": "rectangle"
        }
    ]
    """
    global ros2_bridge, cvat_converter
    
    try:
        # Parse request
        body = event.body
        if isinstance(body, bytes):
            body = body.decode('utf-8')
        
        request_data = json.loads(body)
        
        # Extract image from request
        image_data = request_data.get('image')
        if not image_data:
            return context.Response(
                body=json.dumps({"error": "No image data provided"}),
                status_code=400,
                content_type='application/json'
            )
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            # Convert to OpenCV format (BGR)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return context.Response(
                body=json.dumps({"error": "Invalid image data"}),
                status_code=400,
                content_type='application/json'
            )
        
        # Get optional parameters
        confidence_threshold = request_data.get('threshold', 0.5)
        
        # Process image through ROS2 bridge
        logger.info("Processing image through fruit detector")
        fruit_detections = ros2_bridge.process_image(cv_image, confidence_threshold)
        
        if fruit_detections is None:
            logger.error("Failed to get detections from fruit detector")
            return context.Response(
                body=json.dumps({"error": "Fruit detection failed"}),
                status_code=500,
                content_type='application/json'
            )
        
        # Convert to CVAT format
        cvat_annotations = cvat_converter.convert_to_cvat_format(fruit_detections)
        
        logger.info(f"Returning {len(cvat_annotations)} detections")
        
        return context.Response(
            body=json.dumps(cvat_annotations),
            status_code=200,
            content_type='application/json'
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request: {e}")
        return context.Response(
            body=json.dumps({"error": "Invalid JSON format"}),
            status_code=400,
            content_type='application/json'
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return context.Response(
            body=json.dumps({"error": str(e)}),
            status_code=500,
            content_type='application/json'
        )