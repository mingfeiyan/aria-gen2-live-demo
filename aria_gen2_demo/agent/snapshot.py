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

"""Capture the wearer's current point of view for the AI agent.

Encodes the latest RGB frame (with the gaze point burned in) as a PNG using
only the standard library — no OpenCV/Pillow dependency.
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Optional, Tuple

import numpy as np


def draw_gaze_marker(
    image: np.ndarray, pixel_xy: Optional[Tuple[float, float]], fixating: bool
) -> np.ndarray:
    """Return a copy of the frame with a gaze ring drawn on it."""
    img = image.copy()
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if pixel_xy is None:
        return img
    h, w = img.shape[:2]
    cx, cy = pixel_xy
    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    radius = 16 if fixating else 10
    ring = (dist2 > (radius - 3) ** 2) & (dist2 < (radius + 3) ** 2)
    img[ring] = (255, 64, 255)
    return img


def encode_png(image: np.ndarray) -> bytes:
    """Minimal RGB8/Gray8 PNG encoder (stdlib only)."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    h, w = image.shape[:2]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    # prepend filter byte 0 to each scanline
    raw = np.concatenate(
        [np.zeros((h, 1), dtype=np.uint8), image.reshape(h, -1)], axis=1
    ).tobytes()
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def frame_to_image_block(image: np.ndarray) -> dict:
    """Anthropic Messages API image content block for a numpy frame."""
    data = base64.standard_b64encode(encode_png(image)).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }
