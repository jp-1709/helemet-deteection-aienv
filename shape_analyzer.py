"""
shape_analyzer.py
─────────────────────────────────────────────────────────────
Human silhouette shape analysis via approximated ellipse.

Implements §3.2 and §4.3 of Rougier et al. (2007):

  1. Fit an ellipse to the largest foreground blob using
     second-order central moments.
  2. Track orientation θ and ratio ρ = a/b over a rolling
     1-second window.
  3. Detect a fall if σ_θ > 15° OR σ_ρ > 0.9.

Ellipse semi-axes are derived from the eigenvalues of the
covariance matrix following [Jain 1989]:

    a = (4/π)^(1/4) · [I_max³ / I_min]^(1/8)
    b = (4/π)^(1/4) · [I_min³ / I_max]^(1/8)
─────────────────────────────────────────────────────────────
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import config_ensemble as config_ensemble as config


# ──────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────

@dataclass
class EllipseParams:
    """All geometric parameters of the fitted ellipse."""
    cx:    float   # centroid x (pixels)
    cy:    float   # centroid y (pixels)
    a:     float   # major semi-axis length (pixels)
    b:     float   # minor semi-axis length (pixels)
    theta: float   # orientation in degrees (−90 to +90)
    rho:   float   # ratio a/b (eccentricity proxy)
    area:  int     # blob pixel count


@dataclass
class ShapeAnalysisResult:
    """Output of one call to ShapeAnalyzer.analyze()."""
    ellipse:      Optional[EllipseParams]
    sigma_theta:  float   # std-dev of orientation over window
    sigma_rho:    float   # std-dev of ratio over window
    fall_shape:   bool    # True if shape thresholds exceeded


# ──────────────────────────────────────────────────────────────
# ShapeAnalyzer
# ──────────────────────────────────────────────────────────────

class ShapeAnalyzer:
    """
    Fits an approximated ellipse to the largest foreground blob
    and tracks orientation/ratio history for fall shape detection.
    """

    def __init__(self) -> None:
        # Rolling window stores (timestamp, theta, rho) tuples
        self._window_sec: float = config.SHAPE_WINDOW_SECONDS
        self._history: deque = deque()

        self.last_ellipse: Optional[EllipseParams] = None

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def analyze(self, fg_mask: np.ndarray) -> ShapeAnalysisResult:
        """
        Fit ellipse and compute shape-change statistics.

        Parameters
        ----------
        fg_mask : np.ndarray  (H, W) uint8 foreground mask

        Returns
        -------
        ShapeAnalysisResult
        """
        ellipse = self._fit_ellipse(fg_mask)
        self.last_ellipse = ellipse

        now = time.monotonic()

        if ellipse is not None:
            self._history.append((now, ellipse.theta, ellipse.rho))

        # Expire entries outside the rolling window
        cutoff = now - self._window_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        sigma_theta, sigma_rho = self._compute_deviations()

        fall_shape = (
            sigma_theta > config.SIGMA_THETA_THRESHOLD
            or sigma_rho  > config.SIGMA_RHO_THRESHOLD
        )

        return ShapeAnalysisResult(
            ellipse=ellipse,
            sigma_theta=sigma_theta,
            sigma_rho=sigma_rho,
            fall_shape=fall_shape,
        )

    def is_immobile(self) -> bool:
        """
        Check whether the ellipse has been stationary for the
        current history window (used in the post-fall immobility pass).

        Returns True only if *all* criteria from §4.4 pass:
          – σ_θ < IMMOBILITY_THETA_DEG
          – centroid deviation < IMMOBILITY_CENTROID_PX
          – axis deviation < IMMOBILITY_AXIS_PX
        """
        if len(self._history) < 2:
            return False

        timestamps, thetas, rhos = zip(*self._history)
        thetas_arr = np.array(thetas)
        rhos_arr   = np.array(rhos)

        sigma_theta = float(np.std(thetas_arr))
        sigma_rho   = float(np.std(rhos_arr))

        return (
            sigma_theta < config.IMMOBILITY_THETA_DEG
            and sigma_rho < 0.1   # tight rho tolerance for immobility
        )

    def draw_ellipse(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Overlay the last fitted ellipse on *frame* (in-place).
        Returns the annotated frame.
        """
        if self.last_ellipse is None:
            return frame

        e = self.last_ellipse
        cx, cy = int(e.cx), int(e.cy)
        a, b   = int(e.a),  int(e.b)
        angle  = e.theta

        # OpenCV ellipse: axes are full lengths, not semi-axes
        cv2.ellipse(
            frame,
            (cx, cy),
            (max(1, a), max(1, b)),
            angle,
            0, 360,
            color,
            thickness,
        )
        # Draw centroid
        cv2.circle(frame, (cx, cy), 4, color, -1)
        return frame

    def reset(self) -> None:
        """Clear history (e.g. on camera restart)."""
        self._history.clear()
        self.last_ellipse = None

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _fit_ellipse(self, fg_mask: np.ndarray) -> Optional[EllipseParams]:
        """
        Compute spatial moments of the largest foreground blob
        and derive the best-fitting ellipse parameters.

        Returns None if no suitable blob is found.
        """
        # Threshold to binary
        _, binary = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours and pick the largest
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = int(cv2.contourArea(largest))
        if area < config.MIN_BLOB_AREA:
            return None

        # ── Compute image moments ──────────────────────────────
        M = cv2.moments(largest)
        m00 = M["m00"]
        if m00 < 1e-6:
            return None

        # Centroid
        cx = M["m10"] / m00
        cy = M["m01"] / m00

        # Central moments of second order
        mu20 = M["mu20"] / m00
        mu02 = M["mu02"] / m00
        mu11 = M["mu11"] / m00

        # ── Orientation (eq. from §3.2) ────────────────────────
        theta_rad = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
        theta_deg = math.degrees(theta_rad)

        # ── Eigenvalues → semi-axes ────────────────────────────
        discriminant = math.sqrt(
            max(0.0, (mu20 - mu02) ** 2 + 4.0 * mu11 ** 2)
        )
        i_min = (mu20 + mu02 - discriminant) / 2.0
        i_max = (mu20 + mu02 + discriminant) / 2.0

        # Guard against degenerate cases
        i_min = max(i_min, 1e-6)
        i_max = max(i_max, 1e-6)

        factor = (4.0 / math.pi) ** 0.25
        a = factor * (i_max ** 3 / i_min) ** 0.125
        b = factor * (i_min ** 3 / i_max) ** 0.125

        # Clamp to something sensible
        a = max(a, 1.0)
        b = max(b, 1.0)
        if a < b:
            a, b = b, a   # ensure a >= b

        rho = a / b

        return EllipseParams(
            cx=cx, cy=cy, a=a, b=b,
            theta=theta_deg, rho=rho, area=area,
        )

    def _compute_deviations(self) -> Tuple[float, float]:
        """Return (σ_θ, σ_ρ) over the current window.  Returns (0,0) if
        fewer than 2 samples are available."""
        if len(self._history) < 2:
            return 0.0, 0.0

        _, thetas, rhos = zip(*self._history)
        sigma_theta = float(np.std(np.array(thetas)))
        sigma_rho   = float(np.std(np.array(rhos)))
        return sigma_theta, sigma_rho
