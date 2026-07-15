# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thread-safe live state shared between the pipeline, dashboard, and agent."""

from __future__ import annotations

import collections
import threading
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ProcessedEvent:
    """A discrete event detected by the processing layer (pinch, fixation...)."""

    timestamp_ns: int
    kind: str  # "fixation" | "pinch" | "activity" | "speech" | ...
    message: str


@dataclass
class GazeAnalytics:
    pixel_xy: Optional[Tuple[float, float]] = None  # gaze point on camera-rgb
    is_fixating: bool = False
    fixation_duration_s: float = 0.0
    depth_m: float = 0.0


@dataclass
class HandAnalytics:
    left_visible: bool = False
    right_visible: bool = False
    pinching: bool = False
    pinch_distance_m: float = float("nan")


@dataclass
class MotionAnalytics:
    activity: str = "unknown"  # "still" | "walking" | "moving"
    accel_energy: float = 0.0
    step_count: int = 0
    speed_ms: float = 0.0


@dataclass
class SceneAnalytics:
    brightness: float = 0.0
    sharpness: float = 0.0
    frame_motion: float = 0.0
    speech_active: bool = False


class LiveState:
    """Latest values + short histories from every stream, safe across threads."""

    def __init__(self, history_len: int = 2000):
        # Reentrant so code holding locked() can still call the helpers below.
        self._lock = threading.RLock()
        self.start_ts_ns: Optional[int] = None
        self.latest_ts_ns: int = 0
        self.frame_counts: Dict[str, int] = collections.defaultdict(int)
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_rgb_ts_ns: int = 0
        self.gaze = GazeAnalytics()
        self.hands = HandAnalytics()
        self.motion = MotionAnalytics()
        self.scene = SceneAnalytics()
        self.trajectory: Deque[np.ndarray] = collections.deque(maxlen=history_len)
        self.gaze_history: Deque[Tuple[int, float, float]] = collections.deque(
            maxlen=history_len
        )  # (ts, yaw, pitch)
        self.events: Deque[ProcessedEvent] = collections.deque(maxlen=200)
        self.baro_pressure_pa: float = float("nan")

    # All mutation happens under the lock — single top-level fields via
    # update(), related groups of nested fields via `with state.locked():` —
    # so readers get consistent snapshots.
    def locked(self) -> threading.RLock:
        """Lock to hold while mutating (or reading) related fields atomically."""
        return self._lock

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def tick(self, timestamp_ns: int) -> None:
        with self._lock:
            if self.start_ts_ns is None:
                self.start_ts_ns = timestamp_ns
            self.latest_ts_ns = max(self.latest_ts_ns, timestamp_ns)

    def count_frame(self, label: str) -> None:
        with self._lock:
            self.frame_counts[label] += 1

    def add_event(self, event: ProcessedEvent) -> None:
        with self._lock:
            self.events.append(event)

    def recent_events(self, n: int = 15) -> List[ProcessedEvent]:
        with self._lock:
            return list(self.events)[-n:]

    def gaze_history_snapshot(self) -> List[Tuple[int, float, float]]:
        """Copy of gaze_history safe to read from other threads."""
        with self._lock:
            return list(self.gaze_history)

    def gaze_overlay(self) -> Tuple[Optional[Tuple[float, float]], bool]:
        """(pixel_xy, is_fixating) read atomically, for drawing gaze markers."""
        with self._lock:
            return self.gaze.pixel_xy, self.gaze.is_fixating

    def snapshot_status(self) -> dict:
        """A JSON-friendly summary used by the AI agent's tools."""
        with self._lock:
            elapsed = (
                (self.latest_ts_ns - self.start_ts_ns) / 1e9
                if self.start_ts_ns is not None
                else 0.0
            )
            return {
                "session_elapsed_s": round(elapsed, 1),
                "frames_received": dict(self.frame_counts),
                "gaze": {
                    "on_rgb_pixel": self.gaze.pixel_xy,
                    "is_fixating": self.gaze.is_fixating,
                    "fixation_duration_s": round(self.gaze.fixation_duration_s, 2),
                    "depth_m": round(self.gaze.depth_m, 2),
                },
                "hands": {
                    "left_visible": self.hands.left_visible,
                    "right_visible": self.hands.right_visible,
                    "pinching": self.hands.pinching,
                },
                "motion": {
                    "activity": self.motion.activity,
                    "step_count": self.motion.step_count,
                    "speed_ms": round(self.motion.speed_ms, 2),
                },
                "scene": {
                    "brightness_0_255": round(self.scene.brightness, 1),
                    "sharpness": round(self.scene.sharpness, 2),
                    "speech_active": self.scene.speech_active,
                },
                "barometer_pressure_pa": (
                    None
                    if np.isnan(self.baro_pressure_pa)
                    else round(self.baro_pressure_pa, 1)
                ),
                "recent_events": [
                    {"t_s": round((e.timestamp_ns - (self.start_ts_ns or 0)) / 1e9, 1),
                     "kind": e.kind,
                     "message": e.message}
                    for e in list(self.events)[-10:]
                ],
            }

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self.latest_rgb is None else self.latest_rgb.copy()
