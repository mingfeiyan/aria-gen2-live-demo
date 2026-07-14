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

"""Rerun dashboard: raw streams, 3D world view, analytics, and event log."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import rerun as rr

from ..samples import (
    AudioChunk,
    BaroSample,
    EyeGazeSample,
    HAND_SKELETON_CONNECTIONS,
    HandPoseSample,
    ImageFrame,
    ImuSample,
    MagSample,
    VioSample,
)

TIMELINE = "device_time"

GAZE_COLOR = [255, 64, 255]
HAND_COLOR = [255, 200, 40]
TRAJ_COLOR = [80, 200, 255]


def _set_time(timestamp_ns: int) -> None:
    if hasattr(rr, "set_time_nanos"):  # rerun-sdk < 0.23
        rr.set_time_nanos(TIMELINE, timestamp_ns)
    else:
        rr.set_time(TIMELINE, duration=timestamp_ns * 1e-9)


def _scalar(path: str, value: float) -> None:
    if hasattr(rr, "Scalars"):  # rerun-sdk >= 0.23
        rr.log(path, rr.Scalars(value))
    else:
        rr.log(path, rr.Scalar(value))


def _make_blueprint():
    try:
        import rerun.blueprint as rrb
    except ImportError:
        return None
    slam_grid = rrb.Grid(
        rrb.Spatial2DView(origin="slam-front-left", name="SLAM FL"),
        rrb.Spatial2DView(origin="slam-front-right", name="SLAM FR"),
        rrb.Spatial2DView(origin="slam-side-left", name="SLAM SL"),
        rrb.Spatial2DView(origin="slam-side-right", name="SLAM SR"),
        grid_columns=2,
        name="SLAM cameras",
    )
    et_pair = rrb.Horizontal(
        rrb.Spatial2DView(origin="camera-et-left", name="ET left"),
        rrb.Spatial2DView(origin="camera-et-right", name="ET right"),
        name="Eye cameras",
    )
    left = rrb.Vertical(
        rrb.Spatial2DView(origin="camera-rgb", name="RGB + gaze + hands"),
        rrb.Horizontal(slam_grid, et_pair),
        row_shares=[2, 1],
        name="Cameras",
    )
    middle = rrb.Vertical(
        rrb.Spatial3DView(origin="world", name="3D world"),
        rrb.Spatial2DView(origin="analytics/attention_heatmap", name="Attention map"),
        row_shares=[2, 1],
    )
    right = rrb.Vertical(
        rrb.TimeSeriesView(origin="imu", name="IMU"),
        rrb.TimeSeriesView(origin="analytics/plots", name="Analytics"),
        rrb.TextLogView(origin="events", name="Processed events & agent"),
        row_shares=[1, 1, 1],
    )
    return rrb.Blueprint(
        rrb.Horizontal(left, middle, right, column_shares=[2, 2, 1]),
        collapse_panels=True,
    )


class Dashboard:
    def __init__(
        self,
        spawn_viewer: bool = True,
        rrd_output_path: Optional[str] = None,
        app_id: str = "aria_gen2_streaming_demo",
    ):
        rr.init(app_id)
        blueprint = _make_blueprint()
        if rrd_output_path:
            rr.save(rrd_output_path)
        elif spawn_viewer:
            rr.spawn()
        if blueprint is not None:
            try:
                rr.send_blueprint(blueprint)
            except Exception:
                pass  # blueprint is cosmetic; never fail the demo over it
        # world axes hint
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # ------------------------------------------------------------- raw streams
    def log_image(self, frame: ImageFrame) -> None:
        _set_time(frame.timestamp_ns)
        rr.log(frame.stream_label, rr.Image(frame.image))

    def log_imu(self, imu: ImuSample) -> None:
        _set_time(imu.timestamp_ns)
        base = f"imu/{imu.stream_label}"
        _scalar(f"{base}/accel_magnitude", float(np.linalg.norm(imu.accel_msec2)))
        _scalar(f"{base}/gyro_magnitude", float(np.linalg.norm(imu.gyro_radsec)))

    def log_baro(self, baro: BaroSample) -> None:
        _set_time(baro.timestamp_ns)
        _scalar("analytics/plots/pressure_hpa", baro.pressure_pa / 100.0)

    def log_mag(self, mag: MagSample) -> None:
        _set_time(mag.timestamp_ns)
        _scalar(
            "analytics/plots/mag_field_ut", float(np.linalg.norm(mag.mag_tesla)) * 1e6
        )

    def log_audio(self, chunk: AudioChunk, rms: float) -> None:
        _set_time(chunk.timestamp_ns)
        _scalar("analytics/plots/audio_rms", rms)

    # ---------------------------------------------------------------- ML + 3D
    def log_gaze(
        self,
        gaze: EyeGazeSample,
        rgb_pixel: Optional[Tuple[float, float]],
        is_fixating: bool,
    ) -> None:
        _set_time(gaze.timestamp_ns)
        if rgb_pixel is not None:
            radius = 14.0 if is_fixating else 8.0
            rr.log(
                "camera-rgb/gaze_point",
                rr.Points2D([rgb_pixel], colors=[GAZE_COLOR], radii=[radius]),
            )
        else:
            rr.log("camera-rgb/gaze_point", rr.Clear(recursive=False))
        if gaze.origin_device is not None and gaze.direction_device is not None:
            depth = gaze.depth_m if gaze.depth_m > 0 else 1.0
            rr.log(
                "world/device/gaze",
                rr.Arrows3D(
                    origins=[gaze.origin_device],
                    vectors=[gaze.direction_device * depth],
                    colors=[GAZE_COLOR],
                ),
            )

    def log_hands(
        self,
        hands: HandPoseSample,
        pixel_landmarks: Sequence[Tuple[str, np.ndarray]] = (),
    ) -> None:
        """``pixel_landmarks``: (hand_label, (21,2) pixels) pairs for RGB overlay."""
        _set_time(hands.timestamp_ns)
        for label, landmarks in (
            ("left", hands.left_landmarks_device),
            ("right", hands.right_landmarks_device),
        ):
            path = f"world/device/hands/{label}"
            if landmarks is None:
                rr.log(path, rr.Clear(recursive=True))
                continue
            strips = [
                [landmarks[a], landmarks[b]] for a, b in HAND_SKELETON_CONNECTIONS
            ]
            rr.log(
                f"{path}/skeleton",
                rr.LineStrips3D(strips, colors=[HAND_COLOR], radii=0.002),
            )
        rr.log("camera-rgb/hands", rr.Clear(recursive=True))
        for label, pixels in pixel_landmarks:
            strips = [
                [pixels[a], pixels[b]]
                for a, b in HAND_SKELETON_CONNECTIONS
                if not (np.any(np.isnan(pixels[a])) or np.any(np.isnan(pixels[b])))
            ]
            if strips:
                rr.log(
                    f"camera-rgb/hands/{label}",
                    rr.LineStrips2D(strips, colors=[HAND_COLOR], radii=2.0),
                )

    def log_vio(self, vio: VioSample, trajectory: List[np.ndarray]) -> None:
        _set_time(vio.timestamp_ns)
        rr.log(
            "world/device",
            rr.Transform3D(
                translation=vio.translation_world,
                rotation=rr.Quaternion(
                    xyzw=[
                        vio.quaternion_wxyz[1],
                        vio.quaternion_wxyz[2],
                        vio.quaternion_wxyz[3],
                        vio.quaternion_wxyz[0],
                    ]
                ),
            ),
        )
        rr.log(
            "world/device/marker",
            rr.Points3D([[0, 0, 0]], colors=[TRAJ_COLOR], radii=0.03),
        )
        if len(trajectory) >= 2:
            rr.log(
                "world/trajectory",
                rr.LineStrips3D([list(trajectory)], colors=[TRAJ_COLOR], radii=0.005),
            )
        if vio.linear_velocity is not None:
            _scalar(
                "analytics/plots/speed_ms", float(np.linalg.norm(vio.linear_velocity))
            )

    # -------------------------------------------------------------- analytics
    def log_motion(self, timestamp_ns: int, energy: float, step_count: int) -> None:
        _set_time(timestamp_ns)
        _scalar("analytics/plots/accel_energy", energy)
        _scalar("analytics/plots/step_count", float(step_count))

    def log_scene(
        self, timestamp_ns: int, brightness: float, sharpness: float, motion: float
    ) -> None:
        _set_time(timestamp_ns)
        _scalar("analytics/plots/brightness", brightness)
        _scalar("analytics/plots/frame_motion", motion)

    def log_heatmap(self, timestamp_ns: int, heatmap_rgb: np.ndarray) -> None:
        _set_time(timestamp_ns)
        rr.log("analytics/attention_heatmap", rr.Image(heatmap_rgb))

    def log_event(self, timestamp_ns: int, kind: str, message: str) -> None:
        _set_time(timestamp_ns)
        level = "INFO" if kind != "agent" else "DEBUG"
        rr.log(f"events/{kind}", rr.TextLog(message, level=level))
