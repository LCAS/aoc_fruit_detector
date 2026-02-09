#!/usr/bin/env python3
"""
Test script for the AOC Fruit Detector Nuclio Function.

This script tests the integration between the nuclio function and
the existing fruit detector ROS2 node.
"""

import base64
import json
import requests
import cv2
import numpy as np
import argparse
import time
from pathlib import Path

def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_nuclio_function(function_url, image_path, confidence_threshold=0.5):
    """
    Test the nuclio function with a test image.
    
    Args:
        function_url: URL of the deployed nuclio function
        image_path: Path to test image
        confidence_threshold: Minimum confidence threshold
    """
    print(f"Testing nuclio function at: {function_url}")
    print(f"Using test image: {image_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    try:
        # Encode image
        print("Encoding image...")
        image_b64 = encode_image_to_base64(image_path)
        
        # Prepare request
        request_data = {
            "image": image_b64,
            "threshold": confidence_threshold
        }
        
        # Send request
        print("Sending request to nuclio function...")
        start_time = time.time()
        
        response = requests.post(
            function_url,
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=60  # 60 second timeout
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Response received in {processing_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            annotations = response.json()
            print(f"Success! Received {len(annotations)} detections:")
            
            for i, annotation in enumerate(annotations):
                print(f"  Detection {i+1}:")
                print(f"    Label: {annotation.get('label', 'unknown')}")
                print(f"    Confidence: {annotation.get('confidence', 0):.3f}")
                print(f"    Bounding box: {annotation.get('points', [])}")
                if annotation.get('attributes'):
                    print(f"    Attributes: {annotation['attributes']}")
                print()
                
            return annotations
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("Error: Request timed out")
        return None
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to function")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def visualize_detections(image_path, annotations, output_path=None):
    """
    Visualize detections on the image.
    
    Args:
        image_path: Path to original image
        annotations: List of CVAT annotations
        output_path: Path to save annotated image (optional)
    """
    if not annotations:
        print("No detections to visualize")
        return
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Draw detections
    for annotation in annotations:
        if annotation.get('type') == 'rectangle' and len(annotation.get('points', [])) >= 4:
            x1, y1, x2, y2 = annotation['points'][:4]
            
            # Draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label and confidence
            label = annotation.get('label', 'unknown')
            confidence = annotation.get('confidence', 0)
            text = f"{label}: {confidence:.2f}"
            
            cv2.putText(image, text, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save or display result
    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"Annotated image saved to: {output_path}")
    else:
        cv2.imshow('Fruit Detections', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Test AOC Fruit Detector Nuclio Function')
    parser.add_argument('--url', required=True, help='Nuclio function URL')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', help='Path to save annotated image')
    
    args = parser.parse_args()
    
    # Validate inputs
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Test function
    annotations = test_nuclio_function(args.url, image_path, args.threshold)
    
    # Visualize results
    if annotations:
        output_path = Path(args.output) if args.output else None
        visualize_detections(image_path, annotations, output_path)

if __name__ == '__main__':
    main()