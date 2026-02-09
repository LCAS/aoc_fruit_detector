#!/usr/bin/env python3
"""
Health check script for the AOC Fruit Detector nuclio function.
"""

import requests
import json
import sys
import base64
from PIL import Image
import io
import argparse


def create_test_image():
    """Create a simple test image for health check."""
    # Create a simple colored image
    image = Image.new('RGB', (640, 480), color='green')
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return image_b64


def health_check(url, timeout=30):
    """
    Perform health check on the nuclio function.
    
    Args:
        url: Function endpoint URL
        timeout: Request timeout in seconds
        
    Returns:
        True if healthy, False otherwise
    """
    try:
        print(f"Performing health check on: {url}")
        
        # Create test request
        test_image = create_test_image()
        request_data = {
            "image": test_image,
            "threshold": 0.5
        }
        
        # Send request
        response = requests.post(
            url,
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=timeout
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Response type: {type(result)}")
                if isinstance(result, list):
                    print(f"Received {len(result)} detections")
                    print("✓ Function is healthy")
                    return True
                else:
                    print("✗ Unexpected response format")
                    return False
            except json.JSONDecodeError:
                print("✗ Invalid JSON response")
                return False
        else:
            print(f"✗ Error response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("✗ Connection failed")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Health check for AOC Fruit Detector nuclio function')
    parser.add_argument('url', help='Function endpoint URL')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    
    args = parser.parse_args()
    
    success = health_check(args.url, args.timeout)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()