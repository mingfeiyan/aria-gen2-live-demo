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

"""Source-agnostic sample types for Aria Gen 2 streams.

Every stream source (live device, VRS replay, synthetic) converts its native
records into these dataclasses so the processing/visualization/agent layers
never depend on where the data came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Canonical Aria Gen 2 stream labels (matches projectaria_tools' aria_rerun_viewer)
GEN2_CAMERA_STREAMS = [
    "camera-rgb",
    "slam-front-left",
    "slam-front-right",
    "slam-side-left",
    "slam-side-right",
    "camera-et-left",
    "camera-et-right",
]
GEN2_IMU_STREAMS = ["imu-left", "imu-right"]
GEN2_ML_STREAMS = ["eyegaze", "handtracking", "vio"]
GEN2_OTHER_STREAMS = ["mic", "baro0", "mag0", "gps"]
ALL_GEN2_STREAMS = (
    GEN2_CAMERA_STREAMS + GEN2_IMU_STREAMS + GEN2_ML_STREAMS + GEN2_OTHER_STREAMS
)


@dataclass
class ImageFrame:
    stream_label: str  # one of GEN2_CAMERA_STREAMS
    timestamp_ns: int  # device capture time
    image: np.ndarray  # HxW (gray) or HxWx3 (RGB), uint8
    frame_index: int = 0


@dataclass
class ImuSample:
    stream_label: str  # "imu-left" | "imu-right"
    timestamp_ns: int
    accel_msec2: np.ndarray  # (3,)
    gyro_radsec: np.ndarray  # (3,)


@dataclass
class EyeGazeSample:
    timestamp_ns: int
    # CPF (central pupil frame) convention, matching projectaria_tools:
    # x right, y down, z forward — so positive yaw looks toward the wearer's
    # right and positive pitch looks DOWN. All sources must emit this
    # convention; the gaze projector and heatmap assume it.
    yaw_rad: float  # combined gaze yaw in CPF frame (positive = right)
    pitch_rad: float  # combined gaze pitch in CPF frame (positive = down)
    depth_m: float  # estimated vergence depth (0 if unknown)
    # Optional 3D gaze data in device frame
    origin_device: Optional[np.ndarray] = None  # (3,)
    direction_device: Optional[np.ndarray] = None  # (3,) unit vector


# 21 landmarks per hand, following the Aria hand-tracking convention
NUM_HAND_LANDMARKS = 21
# Indices used by gesture processing (thumb tip / index tip / wrist)
WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8

# Bone connectivity for plotting a 21-landmark hand skeleton
HAND_SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]


@dataclass
class HandPoseSample:
    timestamp_ns: int
    # (21, 3) landmark positions in the device frame, or None if hand not tracked
    left_landmarks_device: Optional[np.ndarray] = None
    right_landmarks_device: Optional[np.ndarray] = None
    left_confidence: float = 0.0
    right_confidence: float = 0.0


@dataclass
class VioSample:
    timestamp_ns: int
    translation_world: np.ndarray  # (3,) device position in world/odometry frame
    quaternion_wxyz: np.ndarray  # (4,) device orientation
    linear_velocity: Optional[np.ndarray] = None  # (3,)


@dataclass
class BaroSample:
    timestamp_ns: int
    pressure_pa: float
    temperature_c: float = 0.0


@dataclass
class MagSample:
    timestamp_ns: int
    mag_tesla: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,)


@dataclass
class AudioChunk:
    timestamp_ns: int
    # (num_samples,) mono energy envelope is enough for the dashboard;
    # sources may downmix multi-channel mic data.
    samples: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sample_rate_hz: int = 48000


# Union of everything a source can yield
Sample = object
