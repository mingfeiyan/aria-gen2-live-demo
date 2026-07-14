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

"""Scene analytics on the RGB stream + speech activity on the mic stream."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..samples import AudioChunk, ImageFrame

SPEECH_RMS_THRESHOLD = 0.05


class SceneProcessor:
    def __init__(self):
        self._prev_small: Optional[np.ndarray] = None
        self._speech_active = False

    def update_image(self, frame: ImageFrame) -> Tuple[float, float, float]:
        """Returns (brightness, sharpness, frame_motion) for an RGB frame."""
        img = frame.image
        gray = img.mean(axis=2) if img.ndim == 3 else img.astype(np.float64)
        small = gray[::8, ::8].astype(np.float64)

        brightness = float(small.mean())
        gy, gx = np.gradient(small)
        sharpness = float(np.mean(gx**2 + gy**2))

        motion = 0.0
        if self._prev_small is not None and self._prev_small.shape == small.shape:
            motion = float(np.mean(np.abs(small - self._prev_small)))
        self._prev_small = small
        return brightness, sharpness, motion

    def update_audio(self, chunk: AudioChunk) -> Tuple[float, bool, Optional[str]]:
        """Returns (rms, speech_active, event_message)."""
        rms = float(np.sqrt(np.mean(chunk.samples**2))) if chunk.samples.size else 0.0
        active = rms > SPEECH_RMS_THRESHOLD
        message = None
        if active and not self._speech_active:
            message = "Speech/sound activity detected"
        self._speech_active = active
        return rms, active, message
