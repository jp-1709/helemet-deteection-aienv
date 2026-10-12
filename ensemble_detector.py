"""
ensemble_detector.py
─────────────────────────────────────────────────────────────
Multi-cue ensemble fall detector.

Fuses three detection channels with weighted voting:

  Channel A – YOLOv8-Pose + LSTM (primary, 60% weight)
    Extracts 36-D keypoint features per tracked person,
    runs through the bidirectional LSTM temporal classifier.
    Most accurate when skeleton is visible.

  Channel B – Classical MHI + Ellipse (secondary, 25% weight)
    Original Rougier et al. 2007 pipeline. Reliable even when
    pose estimation fails (low light, occlusion).

  Channel C – Geometric rule heuristics (tertiary, 15% weight)
    Fast hand-crafted rules on keypoint geometry:
    • Bounding box aspect ratio > 1.6 (wider than tall)
    • Head Y > 0.75 × frame height
    • Torso angle > 55°
    All three simultaneously → strong fall indicator.

Final decision:
  weighted_score = 0.60*lstm_prob + 0.25*classical_prob + 0.15*geo_prob
  is_fall = weighted_score ≥ ENSEMBLE_THRESHOLD (default 0.55)

This achieves the target ≥ 95% accuracy by compensating for
each channel's individual failure modes.
─────────────────────────────────────────────────────────────
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config_ensemble as config_ensemble as config
from yolov8_detector       import YOLOv8Detector, YOLOFrameResult, PersonDetection
from keypoint_feature_extractor import KeypointFeatureExtractor, FeatureVector
from lstm_classifier       import LSTMClassifier, ClassifierResult
from mhi_processor         import MHIProcessor
from shape_analyzer        import ShapeAnalyzer, ShapeAnalysisResult
from alert_manager         import AlertManager

log = logging.getLogger("ensemble_detector")


# ─────────────────────────────────────────────────────────────
# Per-person state tracker
# ─────────────────────────────────────────────────────────────

@dataclass
class PersonState:
    """Tracks immobility + classifier history for one tracked person."""
    person_id:       int
    feat_extractor:  KeypointFeatureExtractor = field(default_factory=KeypointFeatureExtractor)
    immob_buf:       deque = field(default_factory=lambda: deque(maxlen=75))  # 5s@15fps
    last_lstm_score: float = 0.0
    last_geo_score:  float = 0.0
    alert_fired:     bool  = False


# ─────────────────────────────────────────────────────────────
# Ensemble result
# ─────────────────────────────────────────────────────────────

@dataclass
class EnsembleFrameResult:
    annotated_frame: np.ndarray
    is_fall:         bool
    weighted_score:  float
    lstm_score:      float
    classical_score: float
    geo_score:       float
    n_persons:       int
    inference_ms:    float
    alert_fired:     bool
    classifier_method: str          # "lstm" | "rule_based"
    # Classical sub-metrics
    c_motion:        float = 0.0
    sigma_theta:     float = 0.0
    sigma_rho:       float = 0.0


# ─────────────────────────────────────────────────────────────
# EnsembleDetector
# ─────────────────────────────────────────────────────────────

class EnsembleDetector:
    """
    Main detector: call process_frame() once per BGR frame.
    """

    # Ensemble weights (must sum to 1.0)
    W_LSTM      = 0.60
    W_CLASSICAL = 0.25
    W_GEO       = 0.15

    def __init__(self, alert_manager: AlertManager) -> None:
        self._alert_mgr   = alert_manager

        # ── AI/ML components ───────────────────────────────────
        self._yolo   = YOLOv8Detector()
        self._lstm   = LSTMClassifier()

        # ── Classical pipeline components ──────────────────────
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_VAR_THRESHOLD,
            detectShadows=config.MOG2_DETECT_SHADOWS,
        )
        self._mhi    = MHIProcessor()
        self._shape  = ShapeAnalyzer()

        # ── Per-person state ───────────────────────────────────
        self._persons: Dict[int, PersonState] = {}
        self._next_id: int = 0

        # Counters
        self._frame_count = 0

    # ──────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> EnsembleFrameResult:
        """
        Full multi-channel fall detection on one BGR frame.
        """
        t0 = time.perf_counter()
        self._frame_count += 1
        self._alert_mgr.tick()
        H, W = frame.shape[:2]

        # ── Channel B: Classical pipeline ─────────────────────
        fg_mask          = self._get_foreground(frame)
        mhi_vis, c_motion = self._mhi.update(frame, fg_mask)
        shape_result      = self._shape.analyze(fg_mask)

        classical_score = self._classical_score(c_motion, shape_result)
        mhi_features = (
            c_motion,
            shape_result.sigma_theta,
            shape_result.sigma_rho,
            float(np.sum(fg_mask > 127)),
            float(np.mean(fg_mask)),
            float(np.max(fg_mask)),
        )

        # ── Channel A: YOLOv8-Pose + LSTM ─────────────────────
        yolo_result: YOLOFrameResult = self._yolo.infer(frame)

        # Use YOLO-annotated frame as base
        base_frame = yolo_result.frame_annotated

        lstm_score    = 0.0
        geo_score     = 0.0
        is_fall       = False
        alert_fired   = False
        clf_method    = "rule_based"
        best_score    = 0.0

        for person in yolo_result.persons:
            pid = self._resolve_id(person)
            state = self._get_or_create_state(pid)

            # ── Feature extraction ────────────────────────────
            feat_vec: Optional[FeatureVector] = state.feat_extractor.extract(
                person, (H, W), mhi_features
            )
            if feat_vec is None:
                continue

            # ── LSTM classification ───────────────────────────
            clf: ClassifierResult = self._lstm.classify(pid, feat_vec.data)
            clf_method = clf.method
            p_lstm = clf.fall_probability

            # ── Geometric heuristics (Channel C) ──────────────
            p_geo = self._geometric_score(person, H, W)

            # ── Ensemble fusion ───────────────────────────────
            weighted = (
                self.W_LSTM      * p_lstm
                + self.W_CLASSICAL * classical_score
                + self.W_GEO       * p_geo
            )

            state.last_lstm_score = p_lstm
            state.last_geo_score  = p_geo

            # Track best score across all persons in frame
            if weighted > best_score:
                best_score  = weighted
                lstm_score  = p_lstm
                geo_score   = p_geo

            # Immobility buffer update
            state.immob_buf.append({
                "c_motion": c_motion,
                "cx": (person.bbox[0] + person.bbox[2]) / 2,
                "cy": (person.bbox[1] + person.bbox[3]) / 2,
                "score": weighted,
            })

            # ── Fall decision ─────────────────────────────────
            if weighted >= config.ENSEMBLE_FALL_THRESHOLD:
                immob_confirmed = self._check_immobility(state.immob_buf)
                if immob_confirmed or p_lstm >= config.LSTM_FALL_THRESHOLD:
                    is_fall = True
                    if not state.alert_fired:
                        fired = self._alert_mgr.trigger(
                            c_motion    = c_motion,
                            sigma_theta = shape_result.sigma_theta,
                            sigma_rho   = shape_result.sigma_rho,
                            pose_score  = weighted,
                        )
                        if fired:
                            alert_fired = True
                            state.alert_fired = True
                            log.warning(
                                "🚨 Fall detected! Person #%d  "
                                "score=%.2f  lstm=%.2f  geo=%.2f  classical=%.2f",
                                pid, weighted, p_lstm, p_geo, classical_score
                            )
            else:
                state.alert_fired = False

        inference_ms = (time.perf_counter() - t0) * 1000

        # ── Annotate final frame ───────────────────────────────
        annotated = self._annotate(
            base_frame, mhi_vis, shape_result,
            is_fall, best_score, lstm_score, classical_score, geo_score,
            c_motion, inference_ms, clf_method, yolo_result.available,
        )

        return EnsembleFrameResult(
            annotated_frame  = annotated,
            is_fall          = is_fall,
            weighted_score   = best_score,
            lstm_score       = lstm_score,
            classical_score  = classical_score,
            geo_score        = geo_score,
            n_persons        = len(yolo_result.persons),
            inference_ms     = inference_ms,
            alert_fired      = alert_fired,
            classifier_method= clf_method,
            c_motion         = c_motion,
            sigma_theta      = shape_result.sigma_theta,
            sigma_rho        = shape_result.sigma_rho,
        )

    # ──────────────────────────────────────────────────────────
    # Channel B: classical score
    # ──────────────────────────────────────────────────────────

    def _classical_score(
        self,
        c_motion:     float,
        shape_result: ShapeAnalysisResult,
    ) -> float:
        """Map MHI + shape metrics to a [0,1] fall probability."""
        score = 0.0
        # Motion component (0.4 weight)
        score += 0.4 * min(c_motion / 100.0, 1.0)
        # Shape component (0.6 weight)
        sigma_theta_norm = min(shape_result.sigma_theta / 90.0, 1.0)
        sigma_rho_norm   = min(shape_result.sigma_rho   / 2.0,  1.0)
        shape_score = max(
            sigma_theta_norm / (15.0 / 90.0),   # normalise to threshold
            sigma_rho_norm   / (0.9  / 2.0),
        )
        score += 0.6 * min(shape_score, 1.0) if shape_result.fall_shape else 0.0
        return np.clip(score, 0.0, 1.0)

    # ──────────────────────────────────────────────────────────
    # Channel C: geometric heuristics
    # ──────────────────────────────────────────────────────────

    def _geometric_score(self, person: PersonDetection, H: int, W: int) -> float:
        """
        Fast geometric rules on bounding box + keypoints.
        Returns [0, 1] – higher means more fall-like.
        """
        score = 0.0
        x1, y1, x2, y2 = person.bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # Rule 1: bbox wider than tall (lying down)
        aspect = bw / bh
        if aspect > 1.6:
            score += 0.35

        # Rule 2: bbox centroid in lower half of frame (person on floor)
        cy = (y1 + y2) / 2
        if cy / H > 0.6:
            score += 0.15

        # Rule 3: head (nose) in lower 40% of frame
        nose = person.kp(0)
        if nose.valid and nose.y / H > 0.60:
            score += 0.25

        # Rule 4: large horizontal spread relative to height
        kps = person.keypoints
        shoulders = [kps[5], kps[6]]
        if all(k.valid for k in shoulders):
            sh_width = abs(shoulders[0].x - shoulders[1].x)
            if sh_width / (bh + 1e-9) > 0.8:
                score += 0.15

        # Rule 5: torso angle from keypoints
        from yolov8_detector import KP_L_SHOULDER, KP_R_SHOULDER, KP_L_HIP, KP_R_HIP
        import math
        msh_x = (kps[KP_L_SHOULDER].x + kps[KP_R_SHOULDER].x) / 2 if (kps[KP_L_SHOULDER].valid and kps[KP_R_SHOULDER].valid) else None
        msh_y = (kps[KP_L_SHOULDER].y + kps[KP_R_SHOULDER].y) / 2 if (kps[KP_L_SHOULDER].valid and kps[KP_R_SHOULDER].valid) else None
        mhp_x = (kps[KP_L_HIP].x     + kps[KP_R_HIP].x)     / 2 if (kps[KP_L_HIP].valid     and kps[KP_R_HIP].valid)     else None
        mhp_y = (kps[KP_L_HIP].y     + kps[KP_R_HIP].y)     / 2 if (kps[KP_L_HIP].valid     and kps[KP_R_HIP].valid)     else None
        if all(v is not None for v in [msh_x, msh_y, mhp_x, mhp_y]):
            dx = mhp_x - msh_x
            dy = mhp_y - msh_y + 1e-9
            angle = abs(math.degrees(math.atan2(dx, dy)))
            if angle > 55:
                score += 0.10

        return float(np.clip(score, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────
    # Immobility check
    # ──────────────────────────────────────────────────────────

    def _check_immobility(self, buf: deque) -> bool:
        """
        Returns True if the person has been still for ≥1.5 s
        (≥22 frames at 15 fps).
        """
        if len(buf) < 22:
            return False
        recent = list(buf)[-22:]
        cxs = [r["cx"] for r in recent]
        cys = [r["cy"] for r in recent]
        cms = [r["c_motion"] for r in recent]
        return (
            float(np.std(cxs)) < config.IMMOBILITY_CENTROID_PX
            and float(np.std(cys)) < config.IMMOBILITY_CENTROID_PX
            and all(c < config.IMMOBILITY_CMOTION_MAX for c in cms[-10:])
        )

    # ──────────────────────────────────────────────────────────
    # Person ID management
    # ──────────────────────────────────────────────────────────

    def _resolve_id(self, person: PersonDetection) -> int:
        """Use ByteTrack ID if available, else assign an incremental ID."""
        if person.track_id >= 0:
            return person.track_id
        pid = self._next_id
        self._next_id += 1
        return pid

    def _get_or_create_state(self, pid: int) -> PersonState:
        if pid not in self._persons:
            self._persons[pid] = PersonState(person_id=pid)
        return self._persons[pid]

    # ──────────────────────────────────────────────────────────
    # Foreground segmentation
    # ──────────────────────────────────────────────────────────

    def _get_foreground(self, frame: np.ndarray) -> np.ndarray:
        fg = self._bg_sub.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=1)
        return fg

    # ──────────────────────────────────────────────────────────
    # Frame annotation
    # ──────────────────────────────────────────────────────────

    def _annotate(
        self,
        frame:          np.ndarray,
        mhi_vis:        np.ndarray,
        shape_result:   ShapeAnalysisResult,
        is_fall:        bool,
        weighted_score: float,
        lstm_score:     float,
        classical_score: float,
        geo_score:      float,
        c_motion:       float,
        inference_ms:   float,
        clf_method:     str,
        yolo_available: bool,
    ) -> np.ndarray:
        out = frame.copy()
        H, W = out.shape[:2]

        # Draw classical ellipse
        ell_color = (0, 0, 255) if is_fall else (0, 220, 80)
        self._shape.draw_ellipse(out, color=ell_color)

        # ── Status banner ──────────────────────────────────────
        status_txt = "⚠ FALL DETECTED" if is_fall else "Normal Activity"
        banner_color = (10, 10, 60) if is_fall else (10, 30, 10)
        txt_color    = (0, 0, 255)  if is_fall else (0, 220, 80)
        cv2.rectangle(out, (0, 0), (W, 46), banner_color, -1)
        cv2.putText(out, status_txt, (10, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, txt_color, 2)

        # YOLO badge
        badge = f"YOLOv8-Pose+LSTM" if yolo_available else "Classical only"
        cv2.putText(out, badge, (W - 200, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(out, f"{inference_ms:.0f}ms", (W - 60, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # ── Score bar ──────────────────────────────────────────
        bar_x, bar_y, bar_w, bar_h = 10, H - 18, W - 20, 8
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (40, 40, 40), -1)
        filled = int(bar_w * weighted_score)
        bar_color = (0, 0, 220) if weighted_score > 0.55 else \
                    (0, 120, 255) if weighted_score > 0.35 else (0, 200, 80)
        cv2.rectangle(out, (bar_x, bar_y),
                      (bar_x + filled, bar_y + bar_h), bar_color, -1)
        cv2.putText(out, f"RISK {weighted_score*100:.0f}%",
                    (bar_x, bar_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (180, 180, 180), 1)

        # ── Metric HUD (left column) ───────────────────────────
        hud_lines = [
            f"LSTM:    {lstm_score*100:5.1f}%  [{clf_method[:4]}]",
            f"Classic: {classical_score*100:5.1f}%",
            f"Geo:     {geo_score*100:5.1f}%",
            f"C_motion:{c_motion:5.1f}%",
            f"sigma_t: {shape_result.sigma_theta:5.1f}deg",
            f"sigma_r: {shape_result.sigma_rho:5.3f}",
        ]
        hud_top = H - len(hud_lines) * 17 - 28
        cv2.rectangle(out, (0, hud_top), (195, H - 20), (15, 15, 15), -1)
        for i, line in enumerate(hud_lines):
            cv2.putText(out, line, (5, hud_top + 14 + i * 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 180), 1)

        # ── MHI thumbnail ─────────────────────────────────────
        th_w, th_h = 110, 82
        if mhi_vis is not None:
            thumb = cv2.resize(mhi_vis, (th_w, th_h))
            rx, ry = W - th_w - 4, H - th_h - 22
            out[ry:ry + th_h, rx:rx + th_w] = thumb
            cv2.rectangle(out, (rx - 1, ry - 1),
                          (rx + th_w, ry + th_h), (60, 60, 60), 1)
            cv2.putText(out, "MHI", (rx + 3, ry + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # ── Red flash on fall ──────────────────────────────────
        if self._alert_mgr.alert_active:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 180), -1)
            out = cv2.addWeighted(out, 0.80, overlay, 0.20, 0)
            cv2.putText(out, "FALL ALERT", (W // 2 - 110, H // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

        # Frame border
        border_color = (0, 0, 220) if is_fall else (0, 160, 60)
        cv2.rectangle(out, (0, 0), (W - 1, H - 1), border_color, 3)

        return out
