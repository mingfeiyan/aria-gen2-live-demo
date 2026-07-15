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

"""Synthetic Aria Gen 2 source.

Simulates a wearer walking around a room, looking at three "posters" on the
walls and occasionally pinching with their right hand. Produces every Gen 2
stream the demo consumes (RGB + 4 SLAM + 2 ET cameras, 2 IMUs, eye gaze,
hand tracking, VIO, barometer, magnetometer, mic envelope) with plausible
timing — so the full pipeline runs with zero hardware and zero downloads.
"""

from __future__ import annotations

import heapq
import math
from typing import Iterator, Optional

import numpy as np

from ..samples import (
    AudioChunk,
    BaroSample,
    EyeGazeSample,
    HandPoseSample,
    ImageFrame,
    ImuSample,
    MagSample,
    NUM_HAND_LANDMARKS,
    Sample,
    VioSample,
)
from .base import RealTimePacer, StreamSource

RGB_SIZE = 512
SLAM_SIZE = 256
ET_SIZE = 160

# Posters on the walls: (azimuth_rad, elevation_rad, color, name)
POSTERS = [
    (0.0, 0.05, (220, 70, 60), "red poster"),
    (2.0, -0.05, (60, 160, 220), "blue poster"),
    (4.2, 0.0, (90, 200, 90), "green poster"),
]

RGB_FOCAL = 400.0  # synthetic pinhole focal length in pixels (shared with gaze proj)


def synthetic_pinhole_focal() -> float:
    return RGB_FOCAL


class SyntheticSource(StreamSource):
    name = "synthetic"

    def __init__(
        self,
        duration_s: Optional[float] = None,
        speedup: float = 1.0,
        rgb_fps: float = 10.0,
        imu_hz: float = 50.0,
        seed: int = 7,
    ):
        self._duration_s = duration_s
        self._pacer = RealTimePacer(speedup)
        self._rgb_fps = rgb_fps
        self._imu_hz = imu_hz
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ scene
    def _heading(self, t: float) -> float:
        """Wearer slowly turns around the room, dwelling near each poster."""
        base = 0.35 * t
        wobble = 0.15 * math.sin(0.7 * t)
        return base + wobble

    def _walk_position(self, t: float) -> np.ndarray:
        """Walks a 2 m-radius circle, one lap per ~40 s, with gait bounce."""
        ang = 2 * math.pi * t / 40.0
        bounce = 0.02 * math.sin(2 * math.pi * 1.8 * t)  # ~1.8 steps/s
        return np.array([2.0 * math.cos(ang), 2.0 * math.sin(ang), 1.6 + bounce])

    def _gaze_target(self, t: float):
        """Gaze dwells on the poster nearest the current heading, with saccades.

        Returns (yaw, pitch) in the CPF convention EyeGazeSample requires:
        positive pitch looks DOWN, so a poster at elevation +el gives -el.
        """
        heading = self._heading(t)
        best = min(POSTERS, key=lambda p: abs(_wrap(p[0] - heading)))
        yaw = _wrap(best[0] - heading)
        elevation = best[1]
        # saccade jitter + occasional look-away
        yaw += 0.02 * math.sin(9.0 * t) + (0.25 if (t % 11.0) < 0.8 else 0.0)
        elevation += 0.015 * math.cos(7.0 * t)
        # clamp into camera FOV
        return float(np.clip(yaw, -0.55, 0.55)), float(-np.clip(elevation, -0.4, 0.4))

    def _hand_visible(self, t: float) -> bool:
        return (t % 14.0) < 5.0  # hand raised for 5 s out of every 14 s

    def _pinching(self, t: float) -> bool:
        phase = t % 14.0
        return 2.0 < phase < 3.0  # 1 s pinch while the hand is up

    # ---------------------------------------------------------------- renders
    def _render_rgb(self, t: float) -> np.ndarray:
        h = w = RGB_SIZE
        img = np.empty((h, w, 3), dtype=np.uint8)
        # room: warm floor->wall vertical gradient
        grad = np.linspace(90, 170, h, dtype=np.uint8)[:, None]
        img[..., 0] = grad + 20
        img[..., 1] = grad + 10
        img[..., 2] = grad
        heading = self._heading(t)
        cx, cy = w / 2, h / 2
        for az, el, color, _name in POSTERS:
            rel = _wrap(az - heading)
            if abs(rel) > 0.75:
                continue
            px = int(cx + RGB_FOCAL * math.tan(rel))
            py = int(cy - RGB_FOCAL * math.tan(el))
            half = 70
            x0, x1 = max(0, px - half), min(w, px + half)
            y0, y1 = max(0, py - half), min(h, py + half)
            if x0 < x1 and y0 < y1:
                img[y0:y1, x0:x1] = color
                # simple frame border
                img[y0 : min(y1, y0 + 6), x0:x1] = (30, 30, 30)
                img[max(y0, y1 - 6) : y1, x0:x1] = (30, 30, 30)
        # hand silhouette entering from bottom right when visible
        if self._hand_visible(t):
            hx = int(w * 0.72)
            hy = int(h * 0.86)
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - hx) ** 2 + (yy - hy) ** 2 < 48**2
            img[mask] = (215, 170, 140)
        # sensor noise
        noise = self._rng.integers(-6, 7, size=(h, w, 1), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    def _render_gray(self, t: float, size: int, cam_offset: float) -> np.ndarray:
        heading = self._heading(t) + cam_offset
        x = np.linspace(0, 4 * math.pi, size)
        wall = 100 + 60 * np.sin(x + heading * 3.0)
        img = np.tile(wall, (size, 1))
        img += np.linspace(-25, 25, size)[:, None]
        noise = self._rng.normal(0, 6, size=(size, size))
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def _render_eye(self, t: float, size: int, is_left: bool) -> np.ndarray:
        yaw, pitch = self._gaze_target(t)
        img = np.full((size, size), 40, dtype=np.uint8)
        yy, xx = np.ogrid[:size, :size]
        # sclera
        c = size / 2
        sclera = (xx - c) ** 2 + (yy - c) ** 2 < (size * 0.42) ** 2
        img[sclera] = 170
        # pupil displaced by gaze
        px = c + yaw * size * 0.5 * (1.0 if is_left else 0.95)
        py = c + pitch * size * 0.5
        pupil = (xx - px) ** 2 + (yy - py) ** 2 < (size * 0.12) ** 2
        img[pupil] = 15
        noise = self._rng.normal(0, 5, size=(size, size))
        return np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(
            np.uint8
        )

    def _hand_landmarks(self, t: float) -> np.ndarray:
        """Right hand in device frame (x right, y down, z forward), meters."""
        lm = np.zeros((NUM_HAND_LANDMARKS, 3))
        wrist = np.array([0.12, 0.16, 0.35])
        wave = 0.02 * math.sin(2 * math.pi * 0.8 * t)
        wrist[0] += wave
        lm[0] = wrist
        pinch = self._pinching(t)
        # five fingers fanning forward/up from the wrist
        for f in range(5):
            base_dir = np.array([-0.015 + 0.0125 * f, -0.05, 0.02])
            for j in range(4):
                idx = 1 + f * 4 + j
                lm[idx] = wrist + base_dir * (j + 1) * 0.9
        if pinch:  # bring thumb tip and index tip together
            mid = (lm[4] + lm[8]) / 2
            lm[4] = mid + np.array([0.002, 0, 0])
            lm[8] = mid - np.array([0.002, 0, 0])
        lm += self._rng.normal(0, 0.001, size=lm.shape)
        return lm

    # ------------------------------------------------------------------ merge
    def samples(self) -> Iterator[Sample]:
        rng = self._rng
        streams = []  # (period_s, generator fn(t) -> list[Sample])

        def rgb(t):
            i = int(t * self._rgb_fps)
            return [ImageFrame("camera-rgb", _ns(t), self._render_rgb(t), i)]

        def slam(t):
            out = []
            for k, label in enumerate(
                ["slam-front-left", "slam-front-right", "slam-side-left", "slam-side-right"]
            ):
                offset = [-0.3, 0.3, -1.2, 1.2][k]
                out.append(
                    ImageFrame(label, _ns(t), self._render_gray(t, SLAM_SIZE, offset))
                )
            return out

        def et(t):
            return [
                ImageFrame("camera-et-left", _ns(t), self._render_eye(t, ET_SIZE, True)),
                ImageFrame(
                    "camera-et-right", _ns(t), self._render_eye(t, ET_SIZE, False)
                ),
            ]

        def imu(t):
            out = []
            step = 2 * math.pi * 1.8 * t  # gait frequency
            for label, sign in (("imu-left", 1.0), ("imu-right", -1.0)):
                accel = np.array(
                    [
                        0.4 * math.sin(step) * sign,
                        0.3 * math.cos(step * 0.5),
                        9.81 + 1.1 * abs(math.sin(step)),
                    ]
                ) + rng.normal(0, 0.05, 3)
                gyro = np.array(
                    [0.05 * math.sin(0.7 * t), 0.35, 0.04 * math.cos(0.9 * t)]
                ) + rng.normal(0, 0.01, 3)
                out.append(ImuSample(label, _ns(t), accel, gyro))
            return out

        def gaze(t):
            yaw, pitch = self._gaze_target(t)
            # device frame is x right, y down, z forward, so CPF pitch
            # (down-positive) maps directly onto +y
            direction = np.array(
                [math.sin(yaw), math.sin(pitch), math.cos(yaw) * math.cos(pitch)]
            )
            direction /= np.linalg.norm(direction)
            return [
                EyeGazeSample(
                    _ns(t),
                    yaw_rad=yaw,
                    pitch_rad=pitch,
                    depth_m=1.4 + 0.2 * math.sin(0.3 * t),
                    origin_device=np.array([0.0, 0.0, 0.0]),
                    direction_device=direction,
                )
            ]

        def hands(t):
            if not self._hand_visible(t):
                return [HandPoseSample(_ns(t))]
            return [
                HandPoseSample(
                    _ns(t),
                    right_landmarks_device=self._hand_landmarks(t),
                    right_confidence=0.95,
                )
            ]

        def vio(t):
            pos = self._walk_position(t)
            heading = self._heading(t)
            q = np.array([math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)])
            vel = (self._walk_position(t + 0.01) - pos) / 0.01
            return [VioSample(_ns(t), pos, q, vel)]

        def baro(t):
            return [
                BaroSample(
                    _ns(t),
                    pressure_pa=101_325 + 12 * math.sin(0.05 * t) + rng.normal(0, 1.5),
                    temperature_c=24.0 + 0.3 * math.sin(0.02 * t),
                )
            ]

        def mag(t):
            heading = self._heading(t)
            return [
                MagSample(
                    _ns(t),
                    mag_tesla=np.array(
                        [
                            25e-6 * math.cos(heading),
                            25e-6 * math.sin(heading),
                            -40e-6,
                        ]
                    )
                    + rng.normal(0, 3e-7, 3),
                )
            ]

        def mic(t):
            n = 480
            base = rng.normal(0, 0.01, n)
            if (t % 9.0) < 1.5:  # someone talks every 9 s
                base += 0.2 * np.sin(np.linspace(0, 60 * math.pi, n)) * rng.normal(
                    1, 0.3
                )
            return [AudioChunk(_ns(t), base.astype(np.float32), 48000)]

        streams = [
            (1.0 / self._rgb_fps, rgb),
            (1.0 / self._rgb_fps, slam),
            (1.0 / self._rgb_fps, et),
            (1.0 / self._imu_hz, imu),
            (1.0 / 30.0, gaze),
            (1.0 / 30.0, hands),
            (1.0 / 30.0, vio),
            (1.0 / 5.0, baro),
            (1.0 / 10.0, mag),
            (1.0 / 10.0, mic),
        ]

        # merge streams by next-emit time
        heap = [(0.0, i) for i in range(len(streams))]
        heapq.heapify(heap)
        while heap:
            t, i = heapq.heappop(heap)
            if self._duration_s is not None and t > self._duration_s:
                continue
            period, fn = streams[i]
            self._pacer.wait(_ns(t))
            yield from fn(t)
            heapq.heappush(heap, (t + period, i))


def _ns(t: float) -> int:
    return int(t * 1e9)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
