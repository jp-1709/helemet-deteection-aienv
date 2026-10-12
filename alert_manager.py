"""
alert_manager.py
─────────────────────────────────────────────────────────────
Manages fall alert state, cooldown enforcement, and event log.

The AlertManager is the single source of truth for whether an
alert is currently active.  It also keeps an in-memory log of
the last N events that the REST/WebSocket API exposes to the
frontend dashboard.
─────────────────────────────────────────────────────────────
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

import config_ensemble as config_ensemble as config


# ──────────────────────────────────────────────────────────────
# Event data model
# ──────────────────────────────────────────────────────────────

@dataclass
class FallEvent:
    """Represents a single confirmed fall detection event."""
    event_id:     int
    timestamp:    float    # Unix time
    c_motion:     float
    sigma_theta:  float
    sigma_rho:    float
    pose_score:   float
    method:       str      # "classical" | "classical+pose"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":    self.event_id,
            "timestamp":   self.timestamp,
            "iso_time":    time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)
            ),
            "c_motion":    round(self.c_motion, 1),
            "sigma_theta": round(self.sigma_theta, 2),
            "sigma_rho":   round(self.sigma_rho, 3),
            "pose_score":  round(self.pose_score, 3),
            "method":      self.method,
        }


# ──────────────────────────────────────────────────────────────
# AlertManager
# ──────────────────────────────────────────────────────────────

class AlertManager:
    """
    Thread-safe (GIL-protected for CPython) alert controller.

    Parameters
    ----------
    max_log_size : int
        Maximum number of events retained in the in-memory log.
    on_alert : Callable, optional
        Callback invoked with the new FallEvent when an alert
        fires (e.g. to push over WebSocket).
    """

    def __init__(
        self,
        max_log_size: int = 100,
        on_alert: Optional[Callable[[FallEvent], None]] = None,
    ) -> None:
        self._log:          Deque[FallEvent] = deque(maxlen=max_log_size)
        self._on_alert                       = on_alert
        self._last_alert_ts: float           = 0.0
        self._event_counter: int             = 0

        # ── Live state ─────────────────────────────────────────
        self.alert_active:  bool  = False
        self.alert_since:   float = 0.0   # timestamp when alert started
        self._alert_duration_display: float = 5.0  # show banner for N seconds

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def trigger(
        self,
        c_motion:    float,
        sigma_theta: float,
        sigma_rho:   float,
        pose_score:  float = 0.0,
    ) -> bool:
        """
        Attempt to fire a fall alert.

        Respects the cooldown window defined by
        ALERT_COOLDOWN_SECONDS.  Returns True if the alert was
        actually fired (not suppressed by cooldown).
        """
        now = time.time()

        if now - self._last_alert_ts < config.ALERT_COOLDOWN_SECONDS:
            return False   # still in cooldown

        # ── Build event ────────────────────────────────────────
        self._event_counter += 1
        method = "classical+pose" if pose_score > 0 else "classical"
        event  = FallEvent(
            event_id    = self._event_counter,
            timestamp   = now,
            c_motion    = c_motion,
            sigma_theta = sigma_theta,
            sigma_rho   = sigma_rho,
            pose_score  = pose_score,
            method      = method,
        )
        self._log.appendleft(event)

        # ── Update state ───────────────────────────────────────
        self._last_alert_ts = now
        self.alert_active   = True
        self.alert_since    = now

        # ── Invoke callback (e.g. WebSocket broadcast) ─────────
        if self._on_alert is not None:
            self._on_alert(event)

        return True

    def tick(self) -> None:
        """
        Call once per processed frame to expire the visual
        alert banner after _alert_duration_display seconds.
        """
        if self.alert_active:
            if time.time() - self.alert_since > self._alert_duration_display:
                self.alert_active = False

    def get_log(self) -> List[Dict[str, Any]]:
        """Return the event log as a list of dicts (newest first)."""
        return [e.to_dict() for e in self._log]

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the current alert state."""
        return {
            "alert_active": self.alert_active,
            "total_events": self._event_counter,
            "last_event":   self._log[0].to_dict() if self._log else None,
        }

    def set_callback(self, cb: Callable[[FallEvent], None]) -> None:
        """Register or replace the alert callback at runtime."""
        self._on_alert = cb
