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

"""Replay a recorded Aria Gen 2 VRS file as a real-time stream.

Uses ``projectaria_tools``' VRS data provider (the same API as the official
``aria_rerun_viewer``) and paces samples to the wall clock so the rest of the
demo behaves exactly as it would with live glasses.

Requires: ``pip install projectaria-tools`` and a Gen 2 VRS recording
(e.g. from https://facebookresearch.github.io/projectaria_tools/gen2/).
"""

from __future__ import annotations

import math
from typing import Iterator, List, Optional

import numpy as np

from ..samples import (
    AudioChunk,
    BaroSample,
    EyeGazeSample,
    HandPoseSample,
    ImageFrame,
    ImuSample,
    MagSample,
    Sample,
    VioSample,
)
from .base import RealTimePacer, StreamSource

# Streams the demo consumes. vio_high_frequency and gps are intentionally
# left out to keep the replay light; add them here if you need them.
DEFAULT_STREAMS = [
    "camera-rgb",
    "slam-front-left",
    "slam-front-right",
    "slam-side-left",
    "slam-side-right",
    "camera-et-left",
    "camera-et-right",
    "imu-left",
    "imu-right",
    "mic",
    "baro0",
    "mag0",
    "eyegaze",
    "handtracking",
    "vio",
]


class VrsReplaySource(StreamSource):
    name = "vrs"

    def __init__(
        self,
        vrs_path: str,
        speedup: float = 1.0,
        enabled_streams: Optional[List[str]] = None,
        imu_subsample: int = 4,
    ):
        try:
            from projectaria_tools.core import data_provider  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "VRS replay requires projectaria_tools. Install with:\n"
                "    pip install projectaria-tools"
            ) from e

        from projectaria_tools.core import data_provider

        self._provider = data_provider.create_vrs_data_provider(vrs_path)
        if not self._provider:
            raise RuntimeError(f"Failed to open VRS file: {vrs_path}")
        self._pacer = RealTimePacer(speedup)
        self._enabled = enabled_streams or DEFAULT_STREAMS
        self._imu_subsample = imu_subsample

        calib = self._provider.get_device_calibration()
        self._T_device_cpf = calib.get_transform_device_cpf()
        try:
            self._rgb_calib = calib.get_camera_calib("camera-rgb")
        except Exception:
            self._rgb_calib = None

    @property
    def provider(self):
        return self._provider

    @property
    def rgb_calibration(self):
        """Camera calibration for gaze projection onto the RGB frame."""
        return self._rgb_calib

    def _deliver_options(self):
        options = self._provider.get_default_deliver_queued_options()
        options.deactivate_stream_all()
        available = options.get_stream_ids()
        for label in self._enabled:
            stream_id = self._provider.get_stream_id_from_label(label)
            if stream_id is not None and stream_id in available:
                options.activate_stream(stream_id)
                if label.startswith("imu") and self._imu_subsample > 1:
                    options.set_subsample_rate(stream_id, self._imu_subsample)
        return options

    def samples(self) -> Iterator[Sample]:
        from projectaria_tools.core.sensor_data import (
            SensorDataType,
            TimeDomain,
        )

        provider = self._provider
        frame_counts: dict = {}
        for data in provider.deliver_queued_sensor_data(self._deliver_options()):
            ts = data.get_time_ns(TimeDomain.DEVICE_TIME)
            self._pacer.wait(ts)
            label = provider.get_label_from_stream_id(data.stream_id())
            dtype = data.sensor_data_type()

            if dtype == SensorDataType.IMAGE:
                image_data, record = data.image_data_and_record()
                idx = frame_counts.get(label, 0)
                frame_counts[label] = idx + 1
                yield ImageFrame(
                    label, record.capture_timestamp_ns, image_data.to_numpy_array(), idx
                )

            elif dtype == SensorDataType.IMU:
                imu = data.imu_data()
                yield ImuSample(
                    label,
                    imu.capture_timestamp_ns,
                    np.asarray(imu.accel_msec2),
                    np.asarray(imu.gyro_radsec),
                )

            elif dtype == SensorDataType.EYE_GAZE:
                gaze = data.eye_gaze_data()
                if not gaze.combined_gaze_valid:
                    continue
                gaze_ts = int(gaze.tracking_timestamp.total_seconds() * 1e9)
                direction_cpf = _unit_vector_from_yaw_pitch(gaze.yaw, gaze.pitch)
                origin_device = (
                    self._T_device_cpf @ np.asarray(gaze.combined_gaze_origin_in_cpf)
                )
                direction_device = self._T_device_cpf.rotation() @ direction_cpf
                yield EyeGazeSample(
                    gaze_ts,
                    yaw_rad=float(gaze.yaw),
                    pitch_rad=float(gaze.pitch),
                    depth_m=float(gaze.depth) if gaze.depth else 0.0,
                    origin_device=np.asarray(origin_device).reshape(3),
                    direction_device=np.asarray(direction_device).reshape(3),
                )

            elif dtype == SensorDataType.HAND_POSE:
                hand = data.hand_pose_data()
                hand_ts = int(hand.tracking_timestamp.total_seconds() * 1e9)
                left = right = None
                left_conf = right_conf = 0.0
                if hand.left_hand is not None:
                    left = np.asarray(hand.left_hand.landmark_positions_device)
                    left_conf = float(getattr(hand.left_hand, "confidence", 1.0))
                if hand.right_hand is not None:
                    right = np.asarray(hand.right_hand.landmark_positions_device)
                    right_conf = float(getattr(hand.right_hand, "confidence", 1.0))
                yield HandPoseSample(hand_ts, left, right, left_conf, right_conf)

            elif dtype == SensorDataType.VIO:
                vio = data.vio_data()
                if not _vio_is_good(vio):
                    continue
                T_world_device = (
                    vio.transform_odometry_bodyimu @ vio.transform_bodyimu_device
                )
                translation = np.asarray(T_world_device.translation()).reshape(3)
                quat = T_world_device.rotation().to_quat()  # scalar-first [w,x,y,z]
                yield VioSample(
                    vio.capture_timestamp_ns,
                    translation,
                    np.asarray(quat).reshape(4),
                    _vio_linear_velocity(vio),
                )

            elif dtype == SensorDataType.BAROMETER:
                baro = data.barometer_data()
                yield BaroSample(
                    baro.capture_timestamp_ns,
                    float(baro.pressure),
                    float(getattr(baro, "temperature", 0.0)),
                )

            elif dtype == SensorDataType.MAGNETOMETER:
                mag = data.magnetometer_data()
                yield MagSample(
                    mag.capture_timestamp_ns, np.asarray(mag.mag_tesla).reshape(3)
                )

            elif dtype == SensorDataType.AUDIO:
                audio_data, record = data.audio_data_and_record()
                raw = np.asarray(audio_data.data)
                timestamps = record.capture_timestamps_ns
                if raw.size == 0 or len(timestamps) == 0:
                    continue
                # Downmix interleaved multichannel audio to mono
                num_channels = max(1, raw.size // max(1, len(timestamps)))
                mono = raw.reshape(-1, num_channels).mean(axis=1).astype(np.float32)
                # Scale integer PCM to [-1, 1] by a FIXED full-scale divisor
                # (Aria mics record 32-bit PCM). Normalizing each chunk by its
                # own peak would erase absolute loudness and make silence
                # trip the speech-activity threshold.
                if np.abs(mono).max() > 1.0:
                    mono /= float(2**31)
                yield AudioChunk(int(timestamps[0]), mono, 48000)


def _unit_vector_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    v = np.array(
        [math.tan(yaw), math.tan(pitch), 1.0],
        dtype=np.float64,
    )
    return v / np.linalg.norm(v)


def _vio_linear_velocity(vio) -> Optional[np.ndarray]:
    """Device linear velocity from a VIO record, across projectaria_tools
    API variants. Consumers only use the norm, so the frame (odometry vs
    device) does not matter."""
    for attr in (
        "device_linear_velocity_odometry",
        "linear_velocity_in_odometry",
        "device_linear_velocity",
    ):
        vel = getattr(vio, attr, None)
        if vel is not None:
            return np.asarray(vel, dtype=np.float64).reshape(3)
    return None


def _vio_is_good(vio) -> bool:
    try:
        from projectaria_tools.core.sensor_data import TrackingQuality, VioStatus

        return (
            vio.status == VioStatus.VALID and vio.pose_quality == TrackingQuality.GOOD
        )
    except Exception:
        return True
