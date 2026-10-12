"""
yolov8_detector.py
─────────────────────────────────────────────────────────────
YOLOv8-Pose real-time skeleton extraction.

Architecture rationale (from literature):
  • YOLOv8n-pose : 3ms/frame on CPU, 17 COCO keypoints, mAP@0.5 = 80.0
  • YOLOv8s-pose : 6ms/frame on CPU,  mAP@0.5 = 83.2
  • YOLOv8m-pose : 11ms/frame on CPU, mAP@0.5 = 86.6  ← recommended
  We default to 'n' (nano) for SSH/CPU servers; set MODEL_SIZE='m'
  in config.py if you have a GPU.

COCO 17-keypoint layout (used throughout this system):
  0:Nose  1:L-Eye  2:R-Eye  3:L-Ear  4:R-Ear
  5:L-Shoulder  6:R-Shoulder  7:L-Elbow  8:R-Elbow
  9:L-Wrist  10:R-Wrist  11:L-Hip  12:R-Hip
  13:L-Knee  14:R-Knee  15:L-Ankle  16:R-Ankle

Reference: Moutsis et al. 2022 – YOLOv8 + SVM fall detection
           (96.7% accuracy on UP-Fall dataset)
─────────────────────────────────────────────────────────────
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config_ensemble as config_ensemble as config

log = logging.getLogger("yolov8_detector")

# Lazy import — avoids crash on import if ultralytics not yet installed
try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_OK = True
except ImportError:
    _ULTRALYTICS_OK = False
    log.warning("ultralytics not installed. YOLOv8 detector disabled.")


# ─────────────────────────────────────────────────────────────
# COCO 17-keypoint index constants
# ─────────────────────────────────────────────────────────────
KP_NOSE        = 0
KP_L_EYE       = 1;  KP_R_EYE       = 2
KP_L_EAR       = 3;  KP_R_EAR       = 4
KP_L_SHOULDER  = 5;  KP_R_SHOULDER  = 6
KP_L_ELBOW     = 7;  KP_R_ELBOW     = 8
KP_L_WRIST     = 9;  KP_R_WRIST     = 10
KP_L_HIP       = 11; KP_R_HIP       = 12
KP_L_KNEE      = 13; KP_R_KNEE      = 14
KP_L_ANKLE     = 15; KP_R_ANKLE     = 16

# Skeleton edges for drawing
SKELETON_EDGES = [
    (KP_NOSE, KP_L_EYE),   (KP_NOSE, KP_R_EYE),
    (KP_L_EYE, KP_L_EAR),  (KP_R_EYE, KP_R_EAR),
    (KP_L_EAR, KP_L_SHOULDER), (KP_R_EAR, KP_R_SHOULDER),
    (KP_L_SHOULDER, KP_R_SHOULDER),
    (KP_L_SHOULDER, KP_L_ELBOW), (KP_R_SHOULDER, KP_R_ELBOW),
    (KP_L_ELBOW, KP_L_WRIST),   (KP_R_ELBOW, KP_R_WRIST),
    (KP_L_SHOULDER, KP_L_HIP),  (KP_R_SHOULDER, KP_R_HIP),
    (KP_L_HIP, KP_R_HIP),
    (KP_L_HIP, KP_L_KNEE),  (KP_R_HIP, KP_R_KNEE),
    (KP_L_KNEE, KP_L_ANKLE),(KP_R_KNEE, KP_R_ANKLE),
]


# ─────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────

@dataclass
class Keypoint:
    """Single COCO keypoint with pixel coords and confidence."""
    x:    float   # pixel x (0..W)
    y:    float   # pixel y (0..H)
    conf: float   # detection confidence [0,1]

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def valid(self) -> bool:
        return self.conf >= config.YOLO_KP_CONF_THRESHOLD


@dataclass
class PersonDetection:
    """
    One detected person: bounding box + 17 keypoints.
    All coordinates in pixel space.
    """
    bbox:      Tuple[float, float, float, float]   # x1,y1,x2,y2
    bbox_conf: float
    keypoints: List[Keypoint]   # length 17 (COCO order)
    track_id:  int = -1         # ByteTrack ID (-1 if tracking disabled)

    @property
    def bbox_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def kp(self, idx: int) -> Keypoint:
        """Return keypoint by COCO index."""
        return self.keypoints[idx]

    def mid_point(self, a: int, b: int) -> Optional[Tuple[float, float]]:
        """Midpoint between two keypoints (returns None if either is invalid)."""
        ka, kb = self.keypoints[a], self.keypoints[b]
        if not (ka.valid and kb.valid):
            return None
        return ((ka.x + kb.x) / 2, (ka.y + kb.y) / 2)


@dataclass
class YOLOFrameResult:
    """Per-frame output from YOLOv8Detector."""
    persons:         List[PersonDetection]
    frame_annotated: np.ndarray    # BGR with skeleton drawn
    inference_ms:    float
    available:       bool          # False if model not loaded


# ─────────────────────────────────────────────────────────────
# YOLOv8Detector
# ─────────────────────────────────────────────────────────────

class YOLOv8Detector:
    """
    Wraps Ultralytics YOLOv8-Pose for real-time multi-person
    skeleton extraction.

    Usage
    ─────
    detector = YOLOv8Detector()
    result   = detector.infer(frame)
    for person in result.persons:
        ...
    """

    # Colour palette for up to 8 tracked persons
    _PALETTE = [
        (0, 255, 100), (0, 180, 255), (255, 180, 0),
        (255, 60,  60), (120, 60, 255), (0, 220, 220),
        (255, 120, 180), (180, 255, 60),
    ]

    def __init__(self) -> None:
        self._model = None
        self._available = False

        if not _ULTRALYTICS_OK:
            log.warning("YOLOv8 unavailable – pose features disabled.")
            return

        model_path = config.YOLO_MODEL_PATH
        try:
            log.info("Loading YOLOv8-Pose model: %s", model_path)
            self._model = _YOLO(model_path)
            # Warm-up inference
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False, device=config.YOLO_DEVICE)
            self._available = True
            log.info("YOLOv8-Pose loaded successfully on device=%s", config.YOLO_DEVICE)
        except Exception as exc:
            log.error("Failed to load YOLOv8 model: %s", exc)
            self._model = None

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def infer(self, frame: np.ndarray) -> YOLOFrameResult:
        """
        Run YOLOv8-Pose on one BGR frame.

        Parameters
        ----------
        frame : np.ndarray  (H, W, 3) BGR

        Returns
        -------
        YOLOFrameResult
        """
        annotated = frame.copy()

        if not self._available or self._model is None:
            return YOLOFrameResult(
                persons=[], frame_annotated=annotated,
                inference_ms=0.0, available=False,
            )

        t0 = time.perf_counter()
        results = self._model.predict(
            frame,
            conf=config.YOLO_DETECT_CONF,
            iou=config.YOLO_IOU_THRESHOLD,
            device=config.YOLO_DEVICE,
            verbose=False,
            stream=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        persons: List[PersonDetection] = []

        for result in results:
            if result.keypoints is None:
                continue

            boxes    = result.boxes
            kp_data  = result.keypoints.data   # (N, 17, 3) → x,y,conf

            n = len(boxes)
            for i in range(n):
                # ── Bounding box ──────────────────────────────
                box      = boxes[i]
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                bbox_conf = float(box.conf[0])

                if bbox_conf < config.YOLO_DETECT_CONF:
                    continue

                # ── Keypoints ─────────────────────────────────
                kps_raw = kp_data[i].cpu().numpy()   # (17, 3)
                kps = [
                    Keypoint(x=float(kps_raw[j, 0]),
                             y=float(kps_raw[j, 1]),
                             conf=float(kps_raw[j, 2]))
                    for j in range(17)
                ]

                # ── Track ID ──────────────────────────────────
                track_id = -1
                if boxes[i].id is not None:
                    track_id = int(boxes[i].id[0])

                person = PersonDetection(
                    bbox=(x1, y1, x2, y2),
                    bbox_conf=bbox_conf,
                    keypoints=kps,
                    track_id=track_id,
                )
                persons.append(person)

        # ── Annotate frame ────────────────────────────────────
        annotated = self._draw(annotated, persons)

        # ── Debug HUD ─────────────────────────────────────────
        cv2.putText(annotated,
                    f"YOLO {config.YOLO_MODEL_PATH}  {inference_ms:.1f}ms  "
                    f"{len(persons)} person(s)",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1)

        return YOLOFrameResult(
            persons=persons,
            frame_annotated=annotated,
            inference_ms=inference_ms,
            available=True,
        )

    # ──────────────────────────────────────────────────────────
    # Drawing
    # ──────────────────────────────────────────────────────────

    def _draw(
        self,
        frame: np.ndarray,
        persons: List[PersonDetection],
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        for idx, person in enumerate(persons):
            color = self._PALETTE[idx % len(self._PALETTE)]

            # Bounding box
            x1, y1, x2, y2 = [int(v) for v in person.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            label = (f"ID:{person.track_id} "
                     if person.track_id >= 0
                     else "") + f"{person.bbox_conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

            # Skeleton edges
            for a, b in SKELETON_EDGES:
                ka, kb = person.kp(a), person.kp(b)
                if ka.valid and kb.valid:
                    cv2.line(frame,
                             (int(ka.x), int(ka.y)),
                             (int(kb.x), int(kb.y)),
                             color, 2)

            # Keypoint circles
            for kp in person.keypoints:
                if kp.valid:
                    cv2.circle(frame, (int(kp.x), int(kp.y)), 3, color, -1)

        return frame
