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

"""Pipeline orchestrator: source -> processing -> live state -> dashboard."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from .processing.gaze import GazeProcessor
from .processing.hands import HandProcessor
from .processing.motion import MotionProcessor
from .processing.scene import SceneProcessor
from .samples import (
    AudioChunk,
    BaroSample,
    EyeGazeSample,
    HandPoseSample,
    ImageFrame,
    ImuSample,
    MagSample,
    VioSample,
)
from .sources.base import StreamSource
from .state import LiveState, ProcessedEvent
from .viz.dashboard import Dashboard

HEATMAP_LOG_PERIOD_NS = int(1e9)  # refresh the attention map once per second


class Pipeline:
    """Consumes a StreamSource, updates LiveState, and logs to the Dashboard."""

    def __init__(
        self,
        source: StreamSource,
        dashboard: Dashboard,
        state: Optional[LiveState] = None,
    ):
        self.source = source
        self.dashboard = dashboard
        self.state = state or LiveState()
        rgb_calib = getattr(source, "rgb_calibration", None)
        self.gaze_proc = GazeProcessor(camera_calib=rgb_calib)
        self.hand_proc = HandProcessor()
        self.motion_proc = MotionProcessor()
        self.scene_proc = SceneProcessor()
        self._last_gaze: Optional[EyeGazeSample] = None
        self._last_heatmap_ns = 0
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _event(self, timestamp_ns: int, kind: str, message: str) -> None:
        self.state.add_event(ProcessedEvent(timestamp_ns, kind, message))
        self.dashboard.log_event(timestamp_ns, kind, message)

    def run(self) -> None:
        with self.source:
            for sample in self.source.samples():
                if self._stop.is_set():
                    break
                self._dispatch(sample)

    # ---------------------------------------------------------------- dispatch
    def _dispatch(self, sample) -> None:
        state, dash = self.state, self.dashboard

        if isinstance(sample, ImageFrame):
            state.tick(sample.timestamp_ns)
            state.count_frame(sample.stream_label)
            dash.log_image(sample)
            if sample.stream_label == "camera-rgb":
                h, w = sample.image.shape[:2]
                self.gaze_proc.set_image_size(w, h)
                state.update(latest_rgb=sample.image, latest_rgb_ts_ns=sample.timestamp_ns)
                brightness, sharpness, motion = self.scene_proc.update_image(sample)
                state.scene.brightness = brightness
                state.scene.sharpness = sharpness
                state.scene.frame_motion = motion
                dash.log_scene(sample.timestamp_ns, brightness, sharpness, motion)
                if sample.timestamp_ns - self._last_heatmap_ns > HEATMAP_LOG_PERIOD_NS:
                    self._last_heatmap_ns = sample.timestamp_ns
                    dash.log_heatmap(
                        sample.timestamp_ns, self.gaze_proc.heatmap_image()
                    )

        elif isinstance(sample, ImuSample):
            state.tick(sample.timestamp_ns)
            dash.log_imu(sample)
            if sample.stream_label == "imu-left":
                activity, energy, steps, message = self.motion_proc.update(sample)
                state.motion.activity = activity
                state.motion.accel_energy = energy
                state.motion.step_count = steps
                dash.log_motion(sample.timestamp_ns, energy, steps)
                if message:
                    self._event(sample.timestamp_ns, "activity", message)

        elif isinstance(sample, EyeGazeSample):
            state.tick(sample.timestamp_ns)
            self._last_gaze = sample
            fixating, duration = self.gaze_proc.update(sample)
            pixel = self.gaze_proc.project_to_rgb(sample)
            was_fixating = state.gaze.is_fixating
            state.gaze.pixel_xy = pixel
            state.gaze.is_fixating = fixating
            state.gaze.fixation_duration_s = duration
            state.gaze.depth_m = sample.depth_m
            state.gaze_history.append(
                (sample.timestamp_ns, sample.yaw_rad, sample.pitch_rad)
            )
            dash.log_gaze(sample, pixel, fixating)
            if fixating and not was_fixating:
                self._event(sample.timestamp_ns, "fixation", "Gaze fixation started")

        elif isinstance(sample, HandPoseSample):
            state.tick(sample.timestamp_ns)
            pinching, changed, distance, message = self.hand_proc.update(sample)
            state.hands.left_visible = sample.left_landmarks_device is not None
            state.hands.right_visible = sample.right_landmarks_device is not None
            state.hands.pinching = pinching
            state.hands.pinch_distance_m = distance
            pixel_landmarks = []
            for label, lm in (
                ("left", sample.left_landmarks_device),
                ("right", sample.right_landmarks_device),
            ):
                if lm is not None:
                    pixels = np.array(
                        [self.gaze_proc.project_device_point(p) for p in lm]
                    )
                    pixel_landmarks.append((label, pixels))
            dash.log_hands(sample, pixel_landmarks)
            if changed and message:
                self._event(sample.timestamp_ns, "pinch", message)

        elif isinstance(sample, VioSample):
            state.tick(sample.timestamp_ns)
            state.trajectory.append(sample.translation_world)
            if sample.linear_velocity is not None:
                state.motion.speed_ms = float(np.linalg.norm(sample.linear_velocity))
            dash.log_vio(sample, list(state.trajectory))

        elif isinstance(sample, BaroSample):
            state.tick(sample.timestamp_ns)
            state.baro_pressure_pa = sample.pressure_pa
            dash.log_baro(sample)

        elif isinstance(sample, MagSample):
            state.tick(sample.timestamp_ns)
            dash.log_mag(sample)

        elif isinstance(sample, AudioChunk):
            state.tick(sample.timestamp_ns)
            rms, active, message = self.scene_proc.update_audio(sample)
            state.scene.speech_active = active
            dash.log_audio(sample, rms)
            if message:
                self._event(sample.timestamp_ns, "speech", message)
