"""
keypoint_feature_extractor.py
─────────────────────────────────────────────────────────────
Converts raw YOLOv8-Pose keypoints into a rich feature vector
suitable for the LSTM classifier.

Feature engineering follows:
  • Inturi et al. (2023) – Arab. J. Sci. Eng. keypoint ratios
  • Moutsis et al. (2022) – SVM on keypoint geometry
  • Shu & Shu (2021)      – fall pattern recognition features

Feature vector (36-D) per frame:
  ┌─ Positional (6) ─────────────────────────────────────────┐
  │  nose_y_norm, mid_shoulder_y_norm,                       │
  │  mid_hip_y_norm, mid_ankle_y_norm,                       │
  │  bbox_aspect_ratio, bbox_height_norm                     │
  ├─ Angular (10) ────────────────────────────────────────────┤
  │  torso_angle, torso_horizontal_angle,                    │
  │  left_knee_angle, right_knee_angle,                      │
  │  left_elbow_angle, right_elbow_angle,                    │
  │  left_hip_angle, right_hip_angle,                        │
  │  head_tilt_angle, shoulder_roll_angle                    │
  ├─ Ratio / Distance (8) ────────────────────────────────────┤
  │  head_to_hip_height_ratio,                               │
  │  shoulder_width_to_height_ratio,                         │
  │  hip_width_to_height_ratio,                              │
  │  nose_to_shoulder_dist_norm,                             │
  │  hip_to_ankle_dist_norm,                                 │
  │  body_symmetry_score,                                    │
  │  bbox_area_norm,                                         │
  │  head_below_hip_flag                                     │
  ├─ Velocity (6) ────────────────────────────────────────────┤
  │  centroid_vx, centroid_vy,                               │
  │  head_vx, head_vy,                                       │
  │  bbox_area_change_rate, angle_change_rate                │
  └─ MHI features (6) ───────────────────────────────────────┘
     c_motion, sigma_theta, sigma_rho (classical pipeline),
     mhi_blob_count, mhi_mean, mhi_max
─────────────────────────────────────────────────────────────
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from yolov8_detector import (
    PersonDetection, Keypoint,
    KP_NOSE, KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR,
    KP_L_SHOULDER, KP_R_SHOULDER,
    KP_L_ELBOW, KP_R_ELBOW,
    KP_L_WRIST, KP_R_WRIST,
    KP_L_HIP, KP_R_HIP,
    KP_L_KNEE, KP_R_KNEE,
    KP_L_ANKLE, KP_R_ANKLE,
)

FEATURE_DIM = 36   # exported constant — must match LSTM input_size


@dataclass
class FeatureVector:
    """36-dimensional feature vector for one frame."""
    data:       np.ndarray          # shape (36,) float32
    frame_h:    int
    frame_w:    int
    valid_kps:  int                 # number of confident keypoints


class KeypointFeatureExtractor:
    """
    Stateful extractor: maintains a short history of previous frames
    to compute velocity features.

    Call extract(person, frame_hw, mhi_features) each frame.
    Returns FeatureVector or None if the person has too few visible kps.
    """

    MIN_VALID_KEYPOINTS = 5   # minimum to produce a valid vector

    def __init__(self, history_frames: int = 5) -> None:
        # Rolling buffer of recent (centroid_x, centroid_y, head_y, bbox_area, torso_angle)
        self._history: deque = deque(maxlen=history_frames)

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def extract(
        self,
        person:       PersonDetection,
        frame_hw:     Tuple[int, int],       # (height, width)
        mhi_features: Tuple[float, float, float, float, float, float],
        # (c_motion, sigma_theta, sigma_rho, blob_count, mhi_mean, mhi_max)
    ) -> Optional[FeatureVector]:
        """
        Compute the 36-D feature vector for one detected person.

        Returns None if too few keypoints are visible.
        """
        H, W = frame_hw
        kps  = person.keypoints
        valid_kps = sum(1 for k in kps if k.valid)

        if valid_kps < self.MIN_VALID_KEYPOINTS:
            return None

        # ── 1. Positional features (6-D) ──────────────────────
        pos = self._positional(kps, person, H, W)

        # ── 2. Angular features (10-D) ────────────────────────
        ang = self._angular(kps)

        # ── 3. Ratio / distance features (8-D) ───────────────
        rat = self._ratios(kps, person, H, W)

        # ── 4. Velocity features (6-D) ────────────────────────
        vel = self._velocity(kps, person, ang[0])  # ang[0] = torso_angle

        # ── 5. MHI classical features (6-D) ───────────────────
        mhi = np.array(list(mhi_features), dtype=np.float32)
        mhi = np.clip(mhi / np.array([100.0, 90.0, 2.0, 50.0, 255.0, 255.0]),
                      0.0, 1.0)

        feat = np.concatenate([pos, ang, rat, vel, mhi]).astype(np.float32)
        assert feat.shape[0] == FEATURE_DIM, (
            f"Feature dim mismatch: got {feat.shape[0]}, expected {FEATURE_DIM}"
        )

        return FeatureVector(data=feat, frame_h=H, frame_w=W, valid_kps=valid_kps)

    def reset(self) -> None:
        self._history.clear()

    # ──────────────────────────────────────────────────────────
    # Feature group builders
    # ──────────────────────────────────────────────────────────

    def _positional(
        self,
        kps:    List[Keypoint],
        person: PersonDetection,
        H: int, W: int,
    ) -> np.ndarray:
        """6-D positional features (all y-coords normalised to [0,1])."""
        nose_y       = kps[KP_NOSE].y / H if kps[KP_NOSE].valid else 0.5

        mid_sh_y = self._mid_y(kps, KP_L_SHOULDER, KP_R_SHOULDER, H)
        mid_hp_y = self._mid_y(kps, KP_L_HIP,      KP_R_HIP,      H)
        mid_an_y = self._mid_y(kps, KP_L_ANKLE,     KP_R_ANKLE,    H)

        x1, y1, x2, y2 = person.bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        bbox_aspect = np.clip(bw / bh, 0.0, 5.0) / 5.0   # wide=fall
        bbox_h_norm = np.clip(bh / H, 0.0, 1.0)

        return np.array([nose_y, mid_sh_y, mid_hp_y, mid_an_y,
                         bbox_aspect, bbox_h_norm], dtype=np.float32)

    def _angular(self, kps: List[Keypoint]) -> np.ndarray:
        """10-D angle features in normalised degrees [0,1] (÷180)."""

        def angle3(a_idx, b_idx, c_idx) -> float:
            """Angle at B formed by A-B-C."""
            ka, kb, kc = kps[a_idx], kps[b_idx], kps[c_idx]
            if not (ka.valid and kb.valid and kc.valid):
                return 0.5   # neutral
            v1 = np.array([ka.x - kb.x, ka.y - kb.y])
            v2 = np.array([kc.x - kb.x, kc.y - kb.y])
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))) / 180.0

        def torso_angle_deg() -> float:
            """Angle of mid-shoulder → mid-hip vector from vertical [0,1]."""
            msh = self._mid_xy(kps, KP_L_SHOULDER, KP_R_SHOULDER)
            mhp = self._mid_xy(kps, KP_L_HIP,      KP_R_HIP)
            if msh is None or mhp is None:
                return 0.0
            dx = mhp[0] - msh[0]
            dy = mhp[1] - msh[1]
            deg = abs(math.degrees(math.atan2(dx, dy + 1e-9)))
            return deg / 90.0   # 0=upright, 1=horizontal

        def torso_horizontal() -> float:
            """Is torso leaning forward/backward (horizontal component)?"""
            msh = self._mid_xy(kps, KP_L_SHOULDER, KP_R_SHOULDER)
            mhp = self._mid_xy(kps, KP_L_HIP,      KP_R_HIP)
            if msh is None or mhp is None:
                return 0.0
            dx = abs(mhp[0] - msh[0])
            dy = abs(mhp[1] - msh[1]) + 1e-9
            return np.clip(dx / (dx + dy), 0.0, 1.0)

        def head_tilt() -> float:
            kl, kr = kps[KP_L_EAR], kps[KP_R_EAR]
            if not (kl.valid and kr.valid):
                return 0.0
            dy = abs(kl.y - kr.y)
            dx = abs(kl.x - kr.x) + 1e-9
            return np.clip(math.degrees(math.atan2(dy, dx)) / 90.0, 0.0, 1.0)

        def shoulder_roll() -> float:
            kl, kr = kps[KP_L_SHOULDER], kps[KP_R_SHOULDER]
            if not (kl.valid and kr.valid):
                return 0.0
            dy = abs(kl.y - kr.y)
            dx = abs(kl.x - kr.x) + 1e-9
            return np.clip(math.degrees(math.atan2(dy, dx)) / 90.0, 0.0, 1.0)

        ta  = torso_angle_deg()
        th  = torso_horizontal()
        lk  = angle3(KP_L_HIP,      KP_L_KNEE,     KP_L_ANKLE)
        rk  = angle3(KP_R_HIP,      KP_R_KNEE,     KP_R_ANKLE)
        le  = angle3(KP_L_SHOULDER, KP_L_ELBOW,    KP_L_WRIST)
        re  = angle3(KP_R_SHOULDER, KP_R_ELBOW,    KP_R_WRIST)
        lh  = angle3(KP_L_SHOULDER, KP_L_HIP,      KP_L_KNEE)
        rh  = angle3(KP_R_SHOULDER, KP_R_HIP,      KP_R_KNEE)
        ht  = head_tilt()
        sr  = shoulder_roll()

        return np.array([ta, th, lk, rk, le, re, lh, rh, ht, sr],
                        dtype=np.float32)

    def _ratios(
        self,
        kps:    List[Keypoint],
        person: PersonDetection,
        H: int, W: int,
    ) -> np.ndarray:
        """8-D ratio / distance features."""

        nose_y   = kps[KP_NOSE].y if kps[KP_NOSE].valid else -1
        mid_hp_y = self._mid_xy(kps, KP_L_HIP, KP_R_HIP)

        # head_below_hip_flag: 1.0 if head is below hip centre
        head_below = 0.0
        if nose_y >= 0 and mid_hp_y is not None:
            head_below = 1.0 if nose_y > mid_hp_y[1] else 0.0

        # head_to_hip_height_ratio
        msh = self._mid_xy(kps, KP_L_SHOULDER, KP_R_SHOULDER)
        mhp = self._mid_xy(kps, KP_L_HIP,      KP_R_HIP)
        man = self._mid_xy(kps, KP_L_ANKLE,     KP_R_ANKLE)
        head_hip_ratio = 0.5
        if msh and mhp:
            body_h = max(1.0, abs(mhp[1] - msh[1]))
            head_hip_ratio = np.clip(
                (mhp[1] - msh[1]) / (H + 1e-9), 0.0, 1.0
            )

        # shoulder_width / body_height
        sh_w = self._pair_dist(kps, KP_L_SHOULDER, KP_R_SHOULDER) / (W + 1e-9)
        hp_w = self._pair_dist(kps, KP_L_HIP,      KP_R_HIP)      / (W + 1e-9)

        # nose→shoulder distance norm
        n2s = self._kp_dist(kps[KP_NOSE],
                            type('K', (), {'x': (kps[KP_L_SHOULDER].x + kps[KP_R_SHOULDER].x) / 2,
                                           'y': (kps[KP_L_SHOULDER].y + kps[KP_R_SHOULDER].y) / 2,
                                           'valid': True})()) / (H + 1e-9) \
              if (kps[KP_NOSE].valid and kps[KP_L_SHOULDER].valid and kps[KP_R_SHOULDER].valid) else 0.2

        # hip→ankle
        h2a = 0.0
        if mhp and man:
            h2a = np.clip(abs(man[1] - mhp[1]) / (H + 1e-9), 0.0, 1.0)

        # symmetry: mean |left_y - right_y| over all left-right pairs, normalised
        pairs = [(KP_L_SHOULDER, KP_R_SHOULDER), (KP_L_HIP, KP_R_HIP),
                 (KP_L_KNEE, KP_R_KNEE), (KP_L_ANKLE, KP_R_ANKLE)]
        sym_diffs = []
        for a, b in pairs:
            if kps[a].valid and kps[b].valid:
                sym_diffs.append(abs(kps[a].y - kps[b].y) / (H + 1e-9))
        sym = float(np.mean(sym_diffs)) if sym_diffs else 0.0

        x1, y1, x2, y2 = person.bbox
        bbox_area_norm = np.clip((x2 - x1) * (y2 - y1) / (H * W + 1e-9), 0.0, 1.0)

        return np.array([
            np.clip(head_hip_ratio, 0, 1),
            np.clip(sh_w, 0, 1),
            np.clip(hp_w, 0, 1),
            np.clip(n2s,  0, 1),
            np.clip(h2a,  0, 1),
            np.clip(sym,  0, 1),
            np.clip(bbox_area_norm, 0, 1),
            head_below,
        ], dtype=np.float32)

    def _velocity(
        self,
        kps:         List[Keypoint],
        person:      PersonDetection,
        torso_angle: float,
    ) -> np.ndarray:
        """6-D velocity / change-rate features from frame history."""

        x1, y1, x2, y2 = person.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        head_y = kps[KP_NOSE].y if kps[KP_NOSE].valid else cy

        current = (cx, cy, head_y, area, torso_angle)
        self._history.append(current)

        if len(self._history) < 2:
            return np.zeros(6, dtype=np.float32)

        prev = self._history[-2]
        vx = np.clip((cx       - prev[0]) / 640,   -1, 1)
        vy = np.clip((cy       - prev[1]) / 480,   -1, 1)
        hvx = 0.0
        hvy = np.clip((head_y  - prev[2]) / 480,   -1, 1)
        area_chg = np.clip((area - prev[3]) / (640 * 480), -1, 1)
        ang_chg  = np.clip(torso_angle - prev[4], -1, 1)

        return np.array([vx, vy, hvx, hvy, area_chg, ang_chg], dtype=np.float32)

    # ──────────────────────────────────────────────────────────
    # Geometry helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _mid_y(kps, a, b, H) -> float:
        ka, kb = kps[a], kps[b]
        if ka.valid and kb.valid:
            return ((ka.y + kb.y) / 2) / H
        elif ka.valid:
            return ka.y / H
        elif kb.valid:
            return kb.y / H
        return 0.5

    @staticmethod
    def _mid_xy(kps, a, b) -> Optional[Tuple[float, float]]:
        ka, kb = kps[a], kps[b]
        if ka.valid and kb.valid:
            return ((ka.x + kb.x) / 2, (ka.y + kb.y) / 2)
        return None

    @staticmethod
    def _pair_dist(kps, a, b) -> float:
        ka, kb = kps[a], kps[b]
        if ka.valid and kb.valid:
            return math.hypot(ka.x - kb.x, ka.y - kb.y)
        return 0.0

    @staticmethod
    def _kp_dist(ka, kb) -> float:
        if ka.valid and kb.valid:
            return math.hypot(ka.x - kb.x, ka.y - kb.y)
        return 0.0
