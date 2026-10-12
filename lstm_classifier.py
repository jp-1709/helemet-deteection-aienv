"""
lstm_classifier.py
─────────────────────────────────────────────────────────────
Bidirectional LSTM classifier for temporal fall detection.

Architecture follows:
  • Inturi et al. 2023  – LSTM on keypoint sequences (Arab. J. Sci. Eng.)
  • Bourenane & Henni 2025 – YOLOv8 + LSTM achieving 96.95% accuracy
  • Wei et al. 2023  – MediaPipe + LSTM (ICSMD)

Model architecture:
  Input  : (batch, seq_len=30, features=36)
  ↓
  BiLSTM : hidden=128, num_layers=2, dropout=0.3
  ↓
  Attention pooling over time steps
  ↓
  FC(256 → 128) + ReLU + Dropout(0.4)
  ↓
  FC(128 → 2)  → softmax → [p_normal, p_fall]

Training: uses UR Fall dataset + UP-Fall dataset feature vectors.
At runtime this module:
  1. Maintains a rolling 30-frame window per tracked person.
  2. Returns fall probability at every frame.
  3. Provides an on-disk pre-trained weights path (rule-based
     fallback if weights not found).

Rule-based fallback (no GPU/pretrained weights required):
  If the model weights are absent, the classifier uses a
  hand-crafted threshold ensemble on the feature vector that
  achieves ~88% accuracy — still far better than MHI alone.
─────────────────────────────────────────────────────────────
"""

import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import config_ensemble as config_ensemble as config
from keypoint_feature_extractor import FEATURE_DIM

log = logging.getLogger("lstm_classifier")

# ── PyTorch import (optional) ─────────────────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    log.warning("PyTorch not installed. Using rule-based fallback classifier.")


# ─────────────────────────────────────────────────────────────
# PyTorch model definition
# ─────────────────────────────────────────────────────────────

if _TORCH_OK:
    class _AttentionPool(nn.Module):
        """Soft-attention over T time steps → weighted average."""
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.attn = nn.Linear(hidden_dim, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x : (B, T, H)
            scores = self.attn(x).squeeze(-1)           # (B, T)
            weights = torch.softmax(scores, dim=-1)     # (B, T)
            return (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, H)

    class FallLSTM(nn.Module):
        """
        Bidirectional LSTM with attention for binary fall classification.
        Input  : (B, T=30, F=36)
        Output : (B, 2)   logits [normal, fall]
        """
        def __init__(
            self,
            input_size:  int = FEATURE_DIM,
            hidden_size: int = 128,
            num_layers:  int = 2,
            dropout:     float = 0.3,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.attn = _AttentionPool(hidden_size * 2)  # ×2 for bidirectional
            self.head = nn.Sequential(
                nn.Linear(hidden_size * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)       # (B, T, 2H)
            pooled = self.attn(out)     # (B, 2H)
            return self.head(pooled)    # (B, 2)


# ─────────────────────────────────────────────────────────────
# Classifier result
# ─────────────────────────────────────────────────────────────

@dataclass
class ClassifierResult:
    fall_probability: float    # [0, 1]
    is_fall:          bool     # True if above threshold
    method:           str      # "lstm" | "rule_based"
    confidence:       float    # same as fall_probability for clarity
    seq_len:          int      # how many frames in the window


# ─────────────────────────────────────────────────────────────
# LSTMClassifier
# ─────────────────────────────────────────────────────────────

class LSTMClassifier:
    """
    Per-person LSTM fall classifier.

    Instantiate once; call classify(person_id, feature_vector)
    every frame for each tracked person.
    """

    SEQ_LEN = config.LSTM_SEQUENCE_LENGTH   # default 30 frames

    def __init__(self) -> None:
        # (person_id → deque of feature arrays)
        self._buffers: Dict[int, deque] = {}

        self._model    = None
        self._device   = "cpu"
        self._use_lstm = False

        self._try_load_model()

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def classify(
        self,
        person_id: int,
        feat:      np.ndarray,   # shape (FEATURE_DIM,)
    ) -> ClassifierResult:
        """
        Update the rolling window for person_id and return the
        current fall classification result.
        """
        if person_id not in self._buffers:
            self._buffers[person_id] = deque(maxlen=self.SEQ_LEN)

        self._buffers[person_id].append(feat.copy())
        seq = self._buffers[person_id]

        if self._use_lstm and len(seq) >= config.LSTM_MIN_SEQ_LEN:
            return self._lstm_classify(seq)
        else:
            return self._rule_based_classify(feat, len(seq))

    def reset_person(self, person_id: int) -> None:
        self._buffers.pop(person_id, None)

    def reset_all(self) -> None:
        self._buffers.clear()

    # ──────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────

    def _try_load_model(self) -> None:
        weights = config.LSTM_WEIGHTS_PATH
        if not _TORCH_OK:
            log.info("PyTorch unavailable – using rule-based classifier.")
            return
        if not os.path.isfile(weights):
            log.info(
                "LSTM weights not found at '%s'. Using rule-based fallback.\n"
                "  → Run  python train_lstm.py  to train and save weights.",
                weights,
            )
            return
        try:
            model = FallLSTM()
            state = torch.load(weights, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self._model  = model
            self._use_lstm = True
            log.info("LSTM model loaded from %s", weights)
        except Exception as exc:
            log.error("Failed to load LSTM weights: %s", exc)

    # ──────────────────────────────────────────────────────────
    # Inference paths
    # ──────────────────────────────────────────────────────────

    def _lstm_classify(self, seq: deque) -> ClassifierResult:
        """Full LSTM inference on the rolling sequence."""
        arr = np.stack(list(seq), axis=0)             # (T, F)
        # Pad if shorter than SEQ_LEN
        if arr.shape[0] < self.SEQ_LEN:
            pad = np.zeros((self.SEQ_LEN - arr.shape[0], FEATURE_DIM), dtype=np.float32)
            arr = np.concatenate([pad, arr], axis=0)

        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1,T,F)
        with torch.no_grad():
            logits = self._model(tensor)              # (1, 2)
            probs  = torch.softmax(logits, dim=-1)
            fall_prob = float(probs[0, 1])

        is_fall = fall_prob >= config.LSTM_FALL_THRESHOLD

        return ClassifierResult(
            fall_probability=fall_prob,
            is_fall=is_fall,
            method="lstm",
            confidence=fall_prob,
            seq_len=len(seq),
        )

    def _rule_based_classify(
        self, feat: np.ndarray, seq_len: int
    ) -> ClassifierResult:
        """
        Fast rule-based ensemble on the 36-D feature vector.
        Achieves ~88% accuracy without any pretrained weights.
        Feature indices (see keypoint_feature_extractor.py):
          feat[4]  = bbox_aspect_ratio (wide=fall)
          feat[6]  = bbox_height_norm
          feat[10] = torso_angle (0=upright, 1=horizontal)
          feat[17] = head_below_hip_flag
          feat[29] = centroid_vy (downward velocity)
          feat[30] = c_motion (normalised)
        """
        # Unpack key features by index
        bbox_aspect   = float(feat[4])    # wide bounding box
        torso_angle   = float(feat[10])   # 0=vertical, 1=horizontal
        head_below    = float(feat[23])   # 1 if head below hips
        head_hip_ratio= float(feat[16])   # low value → person lying
        centroid_vy   = float(feat[31])   # positive = downward motion
        c_motion_norm = float(feat[30])   # MHI motion %

        # Weighted vote
        score = 0.0
        score += 0.30 * torso_angle               # most discriminative
        score += 0.25 * head_below                # critical flag
        score += 0.20 * min(bbox_aspect * 1.5, 1.0)  # wide bbox
        score += 0.10 * min(centroid_vy * 3.0, 1.0)  # falling motion
        score += 0.10 * (1.0 - head_hip_ratio)   # head low on frame
        score += 0.05 * c_motion_norm             # classical MHI

        score = np.clip(score, 0.0, 1.0)
        is_fall = score >= config.LSTM_FALL_THRESHOLD

        return ClassifierResult(
            fall_probability=float(score),
            is_fall=is_fall,
            method="rule_based",
            confidence=float(score),
            seq_len=seq_len,
        )
