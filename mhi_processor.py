"""
mhi_processor.py
─────────────────────────────────────────────────────────────
Motion History Image (MHI) engine.

Implements the formulation from Bobick & Davis (2001) as
described in Rougier et al. §2:

    H_τ(x,y,t) = τ          if D(x,y,t) = 1
               = max(0, H_τ(x,y,t-1) - 1)   otherwise

where D(x,y,t) is a binary motion mask and τ is a fixed
duration window (default 500 ms).

C_motion is the mean MHI value within the foreground blob,
scaled to [0, 100] % of full motion.
─────────────────────────────────────────────────────────────
"""

import numpy as np
import cv2
import time
from typing import Optional, Tuple

import config_ensemble as config_ensemble as config


class MHIProcessor:
    """
    Maintains and updates a Motion History Image frame-by-frame.

    Attributes
    ----------
    tau : int
        MHI window in *frames* (converted from ms using the
        configured target FPS).
    mhi : np.ndarray
        Current MHI float32 image, shape (H, W).
    """

    def __init__(self) -> None:
        self.tau: int = max(
            1,
            int(config.MHI_DURATION_MS / 1000.0 * config.TARGET_FPS),
        )
        self.mhi: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        fg_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Update the MHI with a new frame and compute C_motion.

        Parameters
        ----------
        frame : np.ndarray  (H, W, 3) BGR
        fg_mask : np.ndarray  (H, W) uint8, 0/255 foreground mask

        Returns
        -------
        mhi_display : np.ndarray  (H, W, 3) BGR  – colour-mapped MHI for overlay
        c_motion    : float  – motion percentage [0, 100]
        """
        h, w = frame.shape[:2]

        # Lazy initialisation
        if self.mhi is None:
            self.mhi = np.zeros((h, w), dtype=np.float32)

        # Binary motion mask D(x,y,t) — use the foreground mask directly
        # after thresholding to suppress noise
        motion_mask = (fg_mask > 127).astype(np.uint8)

        # Update MHI: brightened where motion occurred, decayed elsewhere
        self.mhi = np.where(
            motion_mask == 1,
            float(self.tau),
            np.maximum(0.0, self.mhi - 1.0),
        ).astype(np.float32)

        # ── C_motion ──────────────────────────────────────────
        c_motion = self._compute_cmotion(fg_mask)

        # ── Colourmap for visualisation ───────────────────────
        mhi_norm = cv2.normalize(
            self.mhi, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        mhi_coloured = cv2.applyColorMap(mhi_norm, cv2.COLORMAP_HOT)

        return mhi_coloured, c_motion

    def get_mhi(self) -> Optional[np.ndarray]:
        """Return the raw float32 MHI array (or None if not yet initialised)."""
        return self.mhi

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _compute_cmotion(self, fg_mask: np.ndarray) -> float:
        """
        C_motion = Σ H_τ(x,y,t) over blob pixels  /  # blob pixels  × 100 / τ

        Returns a value in [0, 100].  Returns 0 if no foreground detected.
        """
        if self.mhi is None:
            return 0.0

        blob_pixels = fg_mask > 127
        n_pixels = int(blob_pixels.sum())
        if n_pixels == 0:
            return 0.0

        mean_mhi = float(self.mhi[blob_pixels].mean())
        # Normalise to [0,100] with τ as the maximum possible value
        c_motion = (mean_mhi / self.tau) * 100.0
        return min(c_motion, 100.0)

    def reset(self) -> None:
        """Clear the MHI state (e.g. when camera is restarted)."""
        self.mhi = None
        self._prev_gray = None
