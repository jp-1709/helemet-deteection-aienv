"""
Restricted Area Detector
Integrates YOLO detection with restricted area monitoring
"""

import cv2
import numpy as np
import threading
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any
from ultralytics import YOLO


class RestrictedAreaDetector:
    def __init__(self, model_path: str, conf: float = 0.4):
        """Initialize the restricted area detector"""
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_colors = {}
        self.restricted_area = None
        self.alert_played = False
        self.generate_class_colors()
        
    def generate_class_colors(self):
        """Generate random colors for each class"""
        for class_id in self.model.names:
            self.class_colors[self.model.names[class_id]] = tuple(
                random.randint(0, 255) for _ in range(3)
            )
    
    def set_restricted_area(self, area_coords: List[Tuple[int, int]]):
        """Set restricted area as polygon coordinates"""
        if len(area_coords) < 3:
            raise ValueError("Restricted area must have at least 3 points")
        self.restricted_area = np.array(area_coords, dtype=np.int32)
    
    def set_elliptical_restricted_area(self, center: Tuple[int, int], axes: Tuple[int, int]):
        """Set restricted area as ellipse"""
        self.restricted_area = {"type": "ellipse", "center": center, "axes": axes}
    
    def is_point_in_restricted_area(self, point: Tuple[int, int]) -> bool:
        """Check if a point is within the restricted area"""
        if self.restricted_area is None:
            return False
            
        if isinstance(self.restricted_area, dict) and self.restricted_area.get("type") == "ellipse":
            # Proper ellipse equation: (dx/a)^2 + (dy/b)^2 <= 1
            center = self.restricted_area["center"]
            axes = self.restricted_area["axes"]  # [width, height]
            a = axes[0] / 2.0  # semi-major axis (x-direction)
            b = axes[1] / 2.0  # semi-minor axis (y-direction)
            if a == 0 or b == 0:
                return False
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            return (dx / a) ** 2 + (dy / b) ** 2 <= 1.0
        else:
            # Polygon check
            return cv2.pointPolygonTest(self.restricted_area, (float(point[0]), float(point[1])), False) >= 0
    
    def is_near_restricted_area(self, box: List[int]) -> bool:
        """Check if bounding box overlaps with or is inside restricted area"""
        if self.restricted_area is None:
            return False
            
        x1, y1, x2, y2 = box
        # Check center + all 4 corners + mid-edges so partial overlaps are caught
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        check_points = [
            (cx, cy),          # center
            (x1, y1),          # top-left
            (x2, y1),          # top-right
            (x1, y2),          # bottom-left
            (x2, y2),          # bottom-right
            (cx, y1),          # top-mid
            (cx, y2),          # bottom-mid
            (x1, cy),          # left-mid
            (x2, cy),          # right-mid
        ]
        return any(self.is_point_in_restricted_area(pt) for pt in check_points)
    
    def draw_restricted_area(self, frame: np.ndarray) -> np.ndarray:
        """Draw the restricted area on the frame"""
        if self.restricted_area is None:
            return frame
            
        if isinstance(self.restricted_area, dict) and self.restricted_area.get("type") == "ellipse":
            # Draw ellipse — axes stored as [full_width, full_height], cv2.ellipse needs semi-axes
            center = tuple(int(v) for v in self.restricted_area["center"])
            axes = self.restricted_area["axes"]
            semi_axes = (max(1, int(axes[0] // 2)), max(1, int(axes[1] // 2)))
            cv2.ellipse(frame, center, semi_axes, 0, 0, 360, (0, 0, 255), 2)
        else:
            # Draw polygon
            cv2.polylines(frame, [self.restricted_area], True, (0, 0, 255), 2)
            
        return frame
    
    def detect(self, frame: np.ndarray, selected_classes: List[str] = None, 
               alert_classes: List[str] = None) -> Dict[str, Any]:
        """
        Perform detection on frame with restricted area monitoring
        
        Args:
            frame: Input frame (BGR)
            selected_classes: List of classes to detect (None for all)
            alert_classes: List of classes that trigger alerts
            
        Returns:
            Dictionary containing detection results
        """
        if selected_classes is None:
            selected_classes = list(self.model.names.values())
        if alert_classes is None:
            alert_classes = []
            
        # Run YOLO detection
        results = self.model(frame, conf=self.conf, iou=0.3)
        annotated = frame.copy()
        
        detected_objects = []
        near_restricted = False
        alert_triggered = False
        
        for result in results[0].boxes:
            class_id = int(result.cls)
            class_name = self.model.names[class_id]
            
            if class_name not in selected_classes:
                continue
                
            color = self.class_colors.get(class_name, (0, 255, 0))
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = float(result.conf[0])
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Check restricted area
            if self.is_near_restricted_area([x1, y1, x2, y2]):
                near_restricted = True
                cv2.putText(annotated, "⚠ Near restricted area!",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 2, cv2.LINE_AA)
                
                if class_name in alert_classes:
                    alert_triggered = True
            
            detected_objects.append({
                "class": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "near_restricted": self.is_near_restricted_area([x1, y1, x2, y2])
            })
        
        # Draw restricted area
        annotated = self.draw_restricted_area(annotated)
        
        return {
            "annotated_frame": annotated,
            "detected_objects": detected_objects,
            "near_restricted": near_restricted,
            "alert_triggered": alert_triggered,
            "classes_detected": list(set(obj["class"] for obj in detected_objects))
        }
    
    def get_available_classes(self) -> List[str]:
        """Get list of available classes from the model"""
        return list(self.model.names.values())
