"""
config_ensemble.py
─────────────────────────────────────────────────────────────
Configuration parameters for the fall detection ensemble.
─────────────────────────────────────────────────────────────
"""
import os

# ──────────────────────────────────────────────────────────────
# YOLO v8-Pose
# ──────────────────────────────────────────────────────────────
YOLO_MODEL_PATH:     str   = os.environ.get("YOLO_MODEL", "yolov8n-pose.pt")
YOLO_DEVICE:         str   = os.environ.get("YOLO_DEVICE", "cpu")  # "cuda:0" for GPU
YOLO_DETECT_CONF:    float = 0.45
YOLO_IOU_THRESHOLD:  float = 0.45
YOLO_KP_CONF_THRESHOLD: float = 0.4

# ──────────────────────────────────────────────────────────────
# LSTM Classifier
# ──────────────────────────────────────────────────────────────
LSTM_WEIGHTS_PATH:   str   = os.environ.get("LSTM_WEIGHTS", "models/fall_lstm.pt")
LSTM_SEQUENCE_LENGTH: int  = 30
LSTM_MIN_SEQ_LEN:     int  = 10
LSTM_FALL_THRESHOLD:  float = 0.65

# ──────────────────────────────────────────────────────────────
# Ensemble Decision Fusion
# ──────────────────────────────────────────────────────────────
ENSEMBLE_FALL_THRESHOLD: float = 0.65

# ──────────────────────────────────────────────────────────────
# MOTION HISTORY IMAGE (Classical portion)
# ──────────────────────────────────────────────────────────────
MHI_DURATION_MS:        int   = 500
MHI_MOTION_THRESHOLD:   float = 65.0
MHI_DIFF_THRESHOLD:     int   = 30
TARGET_FPS:             int   = 15      # Expected FPS for time calculations

# ──────────────────────────────────────────────────────────────
# SHAPE / ELLIPSE ANALYSIS
# ──────────────────────────────────────────────────────────────
SHAPE_WINDOW_SECONDS:   float = 1.0
SIGMA_THETA_THRESHOLD:  float = 25.0
SIGMA_RHO_THRESHOLD:    float = 0.9
MIN_BLOB_AREA:          int   = 2000

# ──────────────────────────────────────────────────────────────
# IMMOBILITY CONFIRMATION
# ──────────────────────────────────────────────────────────────
IMMOBILITY_WINDOW_SECONDS: float = 5.0
IMMOBILITY_CMOTION_MAX:    float = 15.0
IMMOBILITY_CENTROID_PX:    float = 4.0
IMMOBILITY_AXIS_PX:        float = 4.0
IMMOBILITY_THETA_DEG:      float = 15.0

# ──────────────────────────────────────────────────────────────
# BACKGROUND SUBTRACTION
# ──────────────────────────────────────────────────────────────
MOG2_HISTORY:        int   = 500
MOG2_VAR_THRESHOLD:  int   = 16
MOG2_DETECT_SHADOWS: bool  = True

# ──────────────────────────────────────────────────────────────
# ALERT / COOLDOWN
# ──────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS: float = 10.0
