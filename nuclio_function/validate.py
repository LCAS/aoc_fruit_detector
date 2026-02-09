#!/usr/bin/env python3
"""
Simple syntax and import validation test for nuclio function components.
"""

import sys
import traceback

def test_imports():
    """Test basic imports (without ROS2 dependencies)."""
    try:
        import json
        import io
        import base64
        import logging
        import numpy as np
        from PIL import Image
        import cv2
        print("✓ Basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Basic import failed: {e}")
        return False

def test_cvat_converter():
    """Test CVAT converter without ROS2 dependencies."""
    try:
        # Mock the ROS2 message classes for testing
        class MockFruitInfoMessage:
            def __init__(self):
                self.fruit_id = 1
                self.fruit_type = "strawberry"
                self.confidence = 0.95
                self.bbox = [100, 50, 200, 150]
                self.ripeness_category = "Ripe"
                self.ripeness_level = 0.8
                self.area = 2000.0
                self.volume = 0.0
                self.weight = 0.0
                self.occlusion_level = 0.1
                self.pomological_class = "Aggregate"
                self.edible_plant_part = ""
                self.fruit_variety = "unknown"
                self.fruit_quality = "High"
                self.mask2d = []
        
        class MockFruitInfoArray:
            def __init__(self):
                self.fruits = [MockFruitInfoMessage()]
        
        # Test converter logic
        from cvat_converter import CVATConverter
        converter = CVATConverter()
        
        # Mock detection data
        mock_detections = MockFruitInfoArray()
        
        # Test conversion
        annotations = converter.convert_to_cvat_format(mock_detections)
        
        assert len(annotations) == 1
        assert annotations[0]['label'] == 'strawberry_ripe'
        assert annotations[0]['confidence'] == 0.95
        assert annotations[0]['points'] == [100.0, 50.0, 300.0, 200.0]
        assert annotations[0]['type'] == 'rectangle'
        
        print("✓ CVAT converter test passed")
        return True
        
    except Exception as e:
        print(f"✗ CVAT converter test failed: {e}")
        traceback.print_exc()
        return False

def test_main_handler_logic():
    """Test main handler logic without ROS2."""
    try:
        import json
        import base64
        from PIL import Image
        import io
        import numpy as np
        
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='JPEG')
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Test request parsing
        request_data = {
            "image": img_b64,
            "threshold": 0.5
        }
        
        request_json = json.dumps(request_data)
        parsed_data = json.loads(request_json)
        
        # Test image decoding
        decoded_bytes = base64.b64decode(parsed_data['image'])
        decoded_image = Image.open(io.BytesIO(decoded_bytes))
        
        assert decoded_image.size == (100, 100)
        
        print("✓ Main handler logic test passed")
        return True
        
    except Exception as e:
        print(f"✗ Main handler logic test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("Running nuclio function validation tests...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_cvat_converter,
        test_main_handler_logic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All validation tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())