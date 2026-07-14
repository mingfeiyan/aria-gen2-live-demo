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

"""Hand-tracking processing: pinch gesture detection with hysteresis."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..samples import HandPoseSample, INDEX_TIP, THUMB_TIP

PINCH_ON_M = 0.015  # thumb-index distance to enter pinch
PINCH_OFF_M = 0.030  # distance to leave pinch (hysteresis)


class HandProcessor:
    def __init__(self):
        self._pinching = False

    def update(self, hands: HandPoseSample) -> Tuple[bool, bool, float, Optional[str]]:
        """Returns (pinching, pinch_changed, pinch_distance_m, event_message)."""
        distance = float("nan")
        for lm in (hands.right_landmarks_device, hands.left_landmarks_device):
            if lm is not None and len(lm) > INDEX_TIP:
                d = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]))
                distance = d if np.isnan(distance) else min(distance, d)

        was = self._pinching
        if np.isnan(distance):
            self._pinching = False
        elif not self._pinching and distance < PINCH_ON_M:
            self._pinching = True
        elif self._pinching and distance > PINCH_OFF_M:
            self._pinching = False

        changed = was != self._pinching
        message = None
        if changed:
            message = "Pinch gesture started" if self._pinching else "Pinch released"
        return self._pinching, changed, distance, message
