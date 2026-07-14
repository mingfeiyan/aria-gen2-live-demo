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

"""Eye-gaze processing: RGB projection, fixation detection, attention map."""

from __future__ import annotations

import collections
import math
from typing import Deque, Optional, Tuple

import numpy as np

from ..samples import EyeGazeSample

# Dispersion-based fixation detection (I-DT)
FIXATION_WINDOW_S = 0.25
FIXATION_DISPERSION_RAD = 0.03  # ~1.7 degrees


class GazeProcessor:
    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        focal_px: float = 400.0,
        camera_calib=None,
        heatmap_size: int = 64,
    ):
        """``camera_calib`` (projectaria_tools CameraCalibration) is used when
        available (VRS mode); otherwise a simple pinhole model with ``focal_px``
        matches the synthetic renderer."""
        self._image_size = image_size
        self._focal = focal_px
        self._calib = camera_calib
        self._window: Deque[EyeGazeSample] = collections.deque()
        self._fixation_start_ns: Optional[int] = None
        self.heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float64)

    def set_image_size(self, width: int, height: int) -> None:
        self._image_size = (width, height)

    def project_to_rgb(self, gaze: EyeGazeSample) -> Optional[Tuple[float, float]]:
        """Gaze direction -> pixel on camera-rgb. Returns None if out of frame."""
        if self._calib is not None and gaze.origin_device is not None:
            try:
                point_device = (
                    gaze.origin_device
                    + gaze.direction_device * max(gaze.depth_m, 0.35)
                )
                point_camera = (
                    self._calib.get_transform_device_camera().inverse() @ point_device
                )
                pixel = self._calib.project(np.asarray(point_camera).reshape(3))
                if pixel is None:
                    return None
                return float(pixel[0]), float(pixel[1])
            except Exception:
                pass  # fall through to pinhole
        w, h = self._image_size
        x = w / 2 + self._focal * math.tan(gaze.yaw_rad)
        y = h / 2 + self._focal * math.tan(gaze.pitch_rad)
        if 0 <= x < w and 0 <= y < h:
            return float(x), float(y)
        return None

    def project_device_point(self, point_device: np.ndarray) -> Tuple[float, float]:
        """Project a 3D point in the device frame onto camera-rgb pixels.

        Returns (nan, nan) when the point is behind the camera / out of frame.
        Used for the hand-skeleton overlay.
        """
        if self._calib is not None:
            try:
                point_camera = (
                    self._calib.get_transform_device_camera().inverse() @ point_device
                )
                pixel = self._calib.project(np.asarray(point_camera).reshape(3))
                if pixel is None:
                    return float("nan"), float("nan")
                return float(pixel[0]), float(pixel[1])
            except Exception:
                pass
        x, y, z = float(point_device[0]), float(point_device[1]), float(point_device[2])
        if z <= 0.05:
            return float("nan"), float("nan")
        w, h = self._image_size
        px = w / 2 + self._focal * x / z
        py = h / 2 + self._focal * y / z
        if 0 <= px < w and 0 <= py < h:
            return px, py
        return float("nan"), float("nan")

    def update(self, gaze: EyeGazeSample) -> Tuple[bool, float]:
        """Feed one sample; returns (is_fixating, fixation_duration_s)."""
        self._window.append(gaze)
        cutoff = gaze.timestamp_ns - int(FIXATION_WINDOW_S * 1e9)
        while self._window and self._window[0].timestamp_ns < cutoff:
            self._window.popleft()

        yaws = [g.yaw_rad for g in self._window]
        pitches = [g.pitch_rad for g in self._window]
        dispersion = (max(yaws) - min(yaws)) + (max(pitches) - min(pitches))
        fixating = len(self._window) >= 3 and dispersion < FIXATION_DISPERSION_RAD

        if fixating:
            if self._fixation_start_ns is None:
                self._fixation_start_ns = self._window[0].timestamp_ns
            duration = (gaze.timestamp_ns - self._fixation_start_ns) / 1e9
        else:
            self._fixation_start_ns = None
            duration = 0.0

        # accumulate attention heatmap in normalized gaze-angle space
        n = self.heatmap.shape[0]
        gx = int(np.clip((gaze.yaw_rad + 0.6) / 1.2 * (n - 1), 0, n - 1))
        gy = int(np.clip((gaze.pitch_rad + 0.5) / 1.0 * (n - 1), 0, n - 1))
        self.heatmap *= 0.999  # slow decay so the map stays current
        self.heatmap[gy, gx] += 1.0

        return fixating, duration

    def heatmap_image(self) -> np.ndarray:
        """Attention heatmap as a viridis-ish uint8 RGB image."""
        hm = self.heatmap
        norm = hm / hm.max() if hm.max() > 0 else hm
        r = np.clip(norm * 3 - 1.2, 0, 1)
        g = np.clip(norm * 2.2, 0, 1) * 0.9
        b = np.clip(1.0 - norm * 1.5, 0.15, 1) * (0.4 + 0.6 * norm)
        img = np.stack([r, g, b], axis=-1)
        return (img * 255).astype(np.uint8)
