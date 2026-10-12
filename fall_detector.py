"""
Fall Detection using YOLOv11
=====================================
Detects human falls in images and videos using a custom-trained YOLOv11 model.
"""

import cv2
import numpy as np
from pathlib import Path
import time
import argparse
import sys

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Run: pip install ultralytics")


# ─────────────────────────────────────────────
# Color palette & labels
# ─────────────────────────────────────────────
COLORS = {
    "fall": (0, 0, 255),       # Red - fall detected
    "standing": (0, 255, 0),    # Green - normal standing
    "sitting": (255, 255, 0),   # Yellow - sitting
    "lying_down": (255, 0, 255), # Magenta - lying down
}

# Map class names from the trained model to our labels
CLASS_MAP = {
    "fall": "fall",
    "fallen": "fall", 
    "falling": "fall",
    "fallen_person": "fall",
    "person_fall": "fall",
    "standing": "standing",
    "sitting": "sitting",
    "lying_down": "lying_down",
    "person": "standing",  # Default to standing for general person detection
}


class FallDetector:
    """YOLOv11-based fall detector."""

    def __init__(self, model_path: str = "fall_yolov11.pt", conf: float = 0.4):
        """
        Args:
            model_path: Path to custom-trained YOLOv11 weights for fall detection.
            conf:       Confidence threshold (0–1).
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics package required. pip install ultralytics")

        print(f"[INFO] Loading fall detection model: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = self.model.names  # {id: name}
        print(f"[INFO] Model loaded. Classes: {list(self.class_names.values())[:10]}...")
        self.stats = {"fall": 0, "standing": 0, "sitting": 0, "lying_down": 0, "frames": 0}
        self.fall_history = []  # Track recent fall detections for stability

    # ── Core detection ────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run inference on a single frame. Returns list of detection dicts."""
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []
        current_frame_falls = []
        
        for box in results.boxes:
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_name = self.class_names[cls_id].lower()
            label    = CLASS_MAP.get(raw_name, raw_name)
            
            # Only include fall-related detections
            if label in ["fall", "standing", "sitting", "lying_down"]:
                detections.append({
                    "label": label,
                    "raw_name": raw_name,
                    "conf": conf_val,
                    "box": (x1, y1, x2, y2),
                })
                
                # Track fall detections for stability
                if label == "fall":
                    current_frame_falls.append({
                        "box": (x1, y1, x2, y2),
                        "conf": conf_val,
                        "timestamp": time.time()
                    })
        
        # Apply stability filter for fall detection
        if current_frame_falls:
            self.fall_history.extend(current_frame_falls)
            # Keep only recent detections (last 2 seconds)
            current_time = time.time()
            self.fall_history = [f for f in self.fall_history if current_time - f["timestamp"] < 2.0]
            
            # Filter detections to only include stable fall detections
            stable_falls = []
            for det in detections:
                if det["label"] == "fall":
                    # Check if this fall is stable (detected in multiple recent frames)
                    stable_count = sum(1 for f in self.fall_history 
                                     if self._boxes_overlap(det["box"], f["box"]))
                    if stable_count >= 2:  # Require at least 2 recent detections
                        stable_falls.append(det)
                else:
                    stable_falls.append(det)
            
            detections = stable_falls
        
        return detections

    def _boxes_overlap(self, box1: tuple, box2: tuple, threshold: float = 0.5) -> bool:
        """Check if two bounding boxes overlap significantly."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Annotate frame with bounding boxes and labels."""
        overlay = frame.copy()
        fall_count = sum(1 for d in detections if d["label"] == "fall")
        standing_count = sum(1 for d in detections if d["label"] == "standing")
        sitting_count = sum(1 for d in detections if d["label"] == "sitting")
        lying_count = sum(1 for d in detections if d["label"] == "lying_down")

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            color = COLORS.get(det["label"], (200, 200, 200))
            label_text = f"{det['raw_name']} {det['conf']:.0%}"

            # Box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Status banner
        h, w = frame.shape[:2]
        banner_h = 40
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (30, 30, 30), -1)

        status = "🚨 FALL DETECTED!" if fall_count > 0 else \
                 "👥 Standing" if standing_count > 0 else \
                 "🪑 Sitting" if sitting_count > 0 else \
                 "😴 Lying Down" if lying_count > 0 else "🔍 Scanning..."
        color  = (0, 0, 255) if fall_count > 0 else \
                 (0, 255, 0) if standing_count > 0 else \
                 (255, 255, 0) if sitting_count > 0 else \
                 (255, 0, 255) if lying_count > 0 else (200, 200, 200)

        cv2.putText(overlay, f"Falls: {fall_count}  Standing: {standing_count}  |  {status}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # Semi-transparent blend
        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # ── Input modes ───────────────────────────────────────────────────────────

    def process_image(self, path: str, save: bool = True) -> str:
        """Detect falls in a static image."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        dets   = self.detect(img)
        result = self.draw(img, dets)
        out    = Path(path).stem + "_fall_detected.jpg"
        if save:
            cv2.imwrite(out, result)
            print(f"[INFO] Saved: {out}")
        self._print_summary(dets)
        return out

    def process_video(self, path: str, save: bool = True) -> str:
        """Detect falls in a video file."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

        fps  = cap.get(cv2.CAP_PROP_FPS) or 25
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = Path(path).stem + "_fall_detected.mp4"
        writer   = None
        if save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        print("[INFO] Processing video... (press Q to stop preview)")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets   = self.detect(frame)
            result = self.draw(frame, dets)
            if writer:
                writer.write(result)
            cv2.imshow("Fall Detection", result)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        if save:
            print(f"[INFO] Saved: {out_path}")
        return out_path

    def process_webcam(self, cam_id: int = 0):
        """Real-time fall detection from webcam."""
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {cam_id}")

        print("[INFO] Webcam started. Press Q to quit.")
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets   = self.detect(frame)
            result = self.draw(frame, dets)

            # FPS counter
            now  = time.time()
            fps  = 1 / (now - prev_time + 1e-9)
            prev_time = now
            cv2.putText(result, f"FPS: {fps:.1f}", (result.shape[1] - 110, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("Fall Detection – Live", result)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

        cap.release()
        cv2.destroyAllWindows()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _print_summary(self, dets: list[dict]):
        counts = {}
        for d in dets:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        print("[RESULT]", counts if counts else "No falls detected")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fall Detector (YOLOv11)")
    parser.add_argument("--source",  default="0",
                        help="Input source: 0=webcam, path to image/video")
    parser.add_argument("--model",   default="fall_yolov11.pt",
                        help="YOLOv11 weights (default: fall_yolov11.pt)")
    parser.add_argument("--conf",    type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save output file")
    args = parser.parse_args()

    detector = FallDetector(model_path=args.model, conf=args.conf)
    src = args.source

    if src.isdigit():
        detector.process_webcam(int(src))
    elif Path(src).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        detector.process_image(src, save=not args.no_save)
    elif Path(src).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        detector.process_video(src, save=not args.no_save)
    else:
        print(f"[ERROR] Unsupported source: {src}")
        sys.exit(1)


if __name__ == "__main__":
    main()
