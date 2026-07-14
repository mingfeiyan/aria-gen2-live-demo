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

"""IMU processing: activity classification and step counting."""

from __future__ import annotations

import collections
from typing import Deque, Optional, Tuple

import numpy as np

from ..samples import ImuSample

GRAVITY = 9.81
WINDOW_S = 2.0
STILL_ENERGY = 0.15  # m/s^2 RMS of gravity-removed accel
WALKING_ENERGY = 1.6
STEP_PEAK_THRESHOLD = 0.6  # m/s^2 above baseline
STEP_MIN_INTERVAL_NS = int(0.3 * 1e9)


class MotionProcessor:
    """Consumes one IMU stream (use imu-left) and derives coarse activity."""

    def __init__(self):
        self._window: Deque[Tuple[int, float]] = collections.deque()
        self._last_step_ns: int = 0
        self._prev_mag: float = GRAVITY
        self._rising: bool = False
        self.step_count: int = 0
        self.activity: str = "unknown"

    def update(self, imu: ImuSample) -> Tuple[str, float, int, Optional[str]]:
        """Returns (activity, accel_energy, step_count, activity_change_message)."""
        mag = float(np.linalg.norm(imu.accel_msec2))
        deviation = mag - GRAVITY
        self._window.append((imu.timestamp_ns, deviation))
        cutoff = imu.timestamp_ns - int(WINDOW_S * 1e9)
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

        energy = float(np.sqrt(np.mean([d * d for _, d in self._window])))

        # step detection: rising-edge peaks on accel magnitude
        if mag > self._prev_mag:
            self._rising = True
        elif self._rising:
            self._rising = False
            if (
                self._prev_mag - GRAVITY > STEP_PEAK_THRESHOLD
                and imu.timestamp_ns - self._last_step_ns > STEP_MIN_INTERVAL_NS
                and energy > STILL_ENERGY
            ):
                self.step_count += 1
                self._last_step_ns = imu.timestamp_ns
        self._prev_mag = mag

        if energy < STILL_ENERGY:
            activity = "still"
        elif energy < WALKING_ENERGY:
            activity = "walking"
        else:
            activity = "moving vigorously"

        message = None
        if activity != self.activity and self.activity != "unknown":
            message = f"Activity changed: {self.activity} -> {activity}"
        self.activity = activity
        return activity, energy, self.step_count, message
