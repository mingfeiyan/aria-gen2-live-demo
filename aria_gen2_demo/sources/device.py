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

"""Live streaming from Aria glasses over Wi-Fi/USB via the Aria Client SDK.

The Aria Research Kit Client SDK delivers streamed sensor data through
subscription callbacks. This source bridges those callbacks into the demo's
pull-based ``StreamSource`` interface with a thread-safe queue.

Requirements (ARK access required — see
https://facebookresearch.github.io/projectaria_tools/):

    pip install projectaria_client_sdk
    aria auth pair            # one-time pairing with your glasses
    # start streaming from the companion app or:  aria streaming start ...

Notes:
- The Gen 1 Client SDK (``aria.sdk``) is the streaming API that is publicly
  documented today; Gen 2 device streaming is delivered through the ARK and
  follows the same subscribe/observer shape. This bridge maps whatever the
  installed SDK delivers (RGB / SLAM / ET images, IMU, audio) into the demo's
  sample types, and the rest of the pipeline is identical to replay mode.
- On-device ML streams (eyegaze / handtracking / vio) are only available where
  the installed SDK exposes them; otherwise the dashboard simply shows the raw
  streams and the processing layer skips gaze/hand analytics.
"""

from __future__ import annotations

import queue
from typing import Iterator, Optional

import numpy as np

from ..samples import ImageFrame, ImuSample, Sample
from .base import StreamSource

_QUEUE_MAX = 512

# Aria camera_id -> demo stream label (Gen 1 ids; Gen 2 adds more SLAM cams)
_CAMERA_ID_TO_LABEL = {
    "rgb": "camera-rgb",
    "slam1": "slam-front-left",
    "slam2": "slam-front-right",
    "et": "camera-et-left",
}


class DeviceStreamSource(StreamSource):
    name = "device"

    def __init__(self, interface: str = "wifi", profile: Optional[str] = None):
        try:
            import aria.sdk as aria  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Live device streaming requires the Aria Client SDK, which is\n"
                "distributed to Aria Research Kit partners:\n"
                "    pip install projectaria_client_sdk\n"
                "    aria auth pair\n"
                "No glasses handy? Run the demo with --source synthetic, or\n"
                "--source vrs --vrs <recording.vrs> for a real recording."
            ) from e

        import aria.sdk as aria

        self._aria = aria
        self._queue: "queue.Queue[Sample]" = queue.Queue(maxsize=_QUEUE_MAX)
        self._interface = interface
        self._profile = profile
        self._client = None
        self._observer = None

    def _put(self, sample: Sample) -> None:
        # Drop oldest on overflow: live viewing should prefer freshness.
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(sample)

    def _start(self) -> None:
        aria = self._aria
        source = self

        class Observer:
            def on_image_received(self, image: np.ndarray, record) -> None:
                camera_id = getattr(record, "camera_id", None)
                label = _CAMERA_ID_TO_LABEL.get(
                    str(camera_id).split(".")[-1].lower(), "camera-rgb"
                )
                source._put(
                    ImageFrame(label, int(record.capture_timestamp_ns), image)
                )

            def on_imu_received(self, samples, imu_idx: int) -> None:
                label = "imu-left" if imu_idx == 0 else "imu-right"
                for s in samples:
                    source._put(
                        ImuSample(
                            label,
                            int(s.capture_timestamp_ns),
                            np.asarray(s.accel_msec2),
                            np.asarray(s.gyro_radsec),
                        )
                    )

        streaming_client = aria.StreamingClient()
        config = streaming_client.subscription_config
        config.subscriber_data_type = (
            aria.StreamingDataType.Rgb
            | aria.StreamingDataType.Slam
            | aria.StreamingDataType.EyeTrack
            | aria.StreamingDataType.Imu
        )
        streaming_client.subscription_config = config
        self._observer = Observer()
        streaming_client.set_streaming_client_observer(self._observer)
        streaming_client.subscribe()
        self._client = streaming_client
        print("[device] Subscribed to Aria stream — waiting for data ...")

    def samples(self) -> Iterator[Sample]:
        self._start()
        while True:
            try:
                yield self._queue.get(timeout=5.0)
            except queue.Empty:
                print(
                    "[device] No data for 5 s — is streaming running on the "
                    "glasses / companion app?"
                )

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.unsubscribe()
            except Exception:
                pass
