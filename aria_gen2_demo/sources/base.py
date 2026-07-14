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

"""Stream source abstraction.

A StreamSource yields Aria Gen 2 samples (see ``samples.py``) roughly in
device-time order. Live sources are naturally paced by the hardware; replay
sources pace themselves against the wall clock so the dashboard and agent see
a realistic real-time stream.
"""

from __future__ import annotations

import abc
import time
from typing import Iterator, Optional

from ..samples import Sample


class StreamSource(abc.ABC):
    """Interface implemented by device / VRS-replay / synthetic sources."""

    name: str = "unknown"

    @abc.abstractmethod
    def samples(self) -> Iterator[Sample]:
        """Yield samples in (approximately) device-time order, real-time paced."""

    def close(self) -> None:  # optional cleanup
        pass

    def __enter__(self) -> "StreamSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RealTimePacer:
    """Sleeps so that samples with device timestamps are emitted in real time.

    ``speedup`` > 1 replays faster than real time; 0 disables pacing entirely
    (useful for headless tests).
    """

    def __init__(self, speedup: float = 1.0):
        self._speedup = speedup
        self._wall_start: Optional[float] = None
        self._device_start_ns: Optional[int] = None

    def wait(self, device_timestamp_ns: int) -> None:
        if self._speedup <= 0:
            return
        now = time.monotonic()
        if self._wall_start is None:
            self._wall_start = now
            self._device_start_ns = device_timestamp_ns
            return
        target_elapsed = (
            (device_timestamp_ns - self._device_start_ns) / 1e9 / self._speedup
        )
        sleep_s = target_elapsed - (now - self._wall_start)
        if sleep_s > 0:
            time.sleep(min(sleep_s, 0.5))
