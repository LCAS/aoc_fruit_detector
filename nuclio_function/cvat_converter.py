"""
CVAT Converter for AOC Fruit Detector results.

This module converts fruit detection results from the AOC format to
the annotation format expected by CVAT.
"""

import logging
from typing import List, Dict, Any

# Try to import ROS2 messages, use mock classes if not available
try:
    from aoc_fruit_detector.msg import FruitInfoArray, FruitInfoMessage
except ImportError:
    # Mock classes for testing without ROS2
    class FruitInfoMessage:
        def __init__(self):
            self.fruit_id = 0
            self.fruit_type = ""
            self.confidence = 0.0
            self.bbox = []
            self.ripeness_category = ""
            self.ripeness_level = 0.0
            self.area = 0.0
            self.volume = 0.0
            self.weight = 0.0
            self.occlusion_level = 0.0
            self.pomological_class = ""
            self.edible_plant_part = ""
            self.fruit_variety = ""
            self.fruit_quality = ""
            self.mask2d = []
    
    class FruitInfoArray:
        def __init__(self):
            self.fruits = []

logger = logging.getLogger(__name__)

class CVATConverter:
    """
    Converter for transforming AOC fruit detection results to CVAT format.
    """
    
    def __init__(self):
        # Mapping from fruit types to CVAT labels
        self.fruit_type_mapping = {
            'strawberry': 'strawberry',
            'tomato': 'tomato',
            'apple': 'apple',
            'pear': 'pear',
            # Add more mappings as needed
        }
        
        # Ripeness mapping for enriched labels
        self.ripeness_mapping = {
            'Unripe': 'unripe',
            'Ripe': 'ripe', 
            'Overripe': 'overripe'
        }
        
        logger.info("CVAT Converter initialized")
    
    def convert_to_cvat_format(self, fruit_detections: FruitInfoArray) -> List[Dict[str, Any]]:
        """
        Convert FruitInfoArray to CVAT annotation format.
        
        Args:
            fruit_detections: AOC fruit detection results
            
        Returns:
            List of CVAT-compatible annotations
        """
        if not fruit_detections or not fruit_detections.fruits:
            logger.info("No fruit detections to convert")
            return []
        
        cvat_annotations = []
        
        for fruit in fruit_detections.fruits:
            annotation = self._convert_single_fruit(fruit)
            if annotation:
                cvat_annotations.append(annotation)
        
        logger.info(f"Converted {len(cvat_annotations)} fruit detections to CVAT format")
        return cvat_annotations
    
    def _convert_single_fruit(self, fruit: FruitInfoMessage) -> Dict[str, Any]:
        """
        Convert a single fruit detection to CVAT format.
        
        Args:
            fruit: Single fruit detection
            
        Returns:
            CVAT annotation dictionary
        """
        try:
            # Extract bounding box coordinates
            if not fruit.bbox or len(fruit.bbox) < 4:
                logger.warning(f"Invalid bounding box for fruit {fruit.fruit_id}")
                return None
            
            x, y, width, height = fruit.bbox[:4]
            
            # Convert to absolute coordinates (x1, y1, x2, y2)
            x1, y1 = float(x), float(y)
            x2, y2 = float(x + width), float(y + height)
            
            # Determine label
            label = self._get_cvat_label(fruit)
            
            # Create CVAT annotation
            annotation = {
                "confidence": float(fruit.confidence),
                "label": label,
                "points": [x1, y1, x2, y2],
                "type": "rectangle",
                "attributes": self._extract_attributes(fruit)
            }
            
            # Add mask if available (for polygon annotations)
            if fruit.mask2d and len(fruit.mask2d) > 0:
                polygon_points = self._mask_to_polygon(fruit.mask2d)
                if polygon_points:
                    annotation["type"] = "polygon"
                    annotation["points"] = polygon_points
            
            return annotation
            
        except Exception as e:
            logger.error(f"Error converting fruit detection: {e}")
            return None
    
    def _get_cvat_label(self, fruit: FruitInfoMessage) -> str:
        """
        Determine the CVAT label for a fruit.
        """
        base_label = self.fruit_type_mapping.get(
            fruit.fruit_type.lower(), 
            fruit.fruit_type.lower()
        )
        
        # Optionally include ripeness in label
        if fruit.ripeness_category and fruit.ripeness_category != 'Unknown':
            ripeness = self.ripeness_mapping.get(
                fruit.ripeness_category,
                fruit.ripeness_category.lower()
            )
            return f"{base_label}_{ripeness}"
        
        return base_label
    
    def _extract_attributes(self, fruit: FruitInfoMessage) -> Dict[str, Any]:
        """
        Extract additional attributes from fruit detection.
        """
        attributes = {}
        
        # Basic fruit information
        if fruit.fruit_variety:
            attributes["variety"] = fruit.fruit_variety
        
        if fruit.fruit_quality:
            attributes["quality"] = fruit.fruit_quality
            
        # Ripeness information
        if fruit.ripeness_category:
            attributes["ripeness_category"] = fruit.ripeness_category
            
        if fruit.ripeness_level > 0:
            attributes["ripeness_level"] = round(float(fruit.ripeness_level), 2)
        
        # Physical properties
        if fruit.area > 0:
            attributes["area"] = round(float(fruit.area), 2)
            
        if fruit.volume > 0:
            attributes["volume"] = round(float(fruit.volume), 2)
            
        if fruit.weight > 0:
            attributes["weight"] = round(float(fruit.weight), 2)
        
        # Occlusion level
        if fruit.occlusion_level > 0:
            attributes["occlusion_level"] = round(float(fruit.occlusion_level), 2)
        
        # Botanical classification
        if fruit.pomological_class:
            attributes["pomological_class"] = fruit.pomological_class
            
        if fruit.edible_plant_part:
            attributes["edible_plant_part"] = fruit.edible_plant_part
        
        return attributes
    
    def _mask_to_polygon(self, mask_data: List[float]) -> List[float]:
        """
        Convert mask data to polygon points.
        
        This is a simplified implementation. In practice, you might need
        more sophisticated mask-to-polygon conversion.
        """
        try:
            # For now, return empty list - this would need proper implementation
            # based on the actual mask format used by the fruit detector
            logger.warning("Mask to polygon conversion not yet implemented")
            return []
        except Exception as e:
            logger.error(f"Error converting mask to polygon: {e}")
            return []