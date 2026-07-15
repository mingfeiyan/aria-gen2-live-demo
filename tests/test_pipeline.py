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

"""End-to-end smoke test: synthetic source -> pipeline -> headless Rerun."""

import numpy as np

from aria_gen2_demo.agent.snapshot import draw_gaze_marker, encode_png
from aria_gen2_demo.app import Pipeline
from aria_gen2_demo.samples import ImageFrame
from aria_gen2_demo.sources.synthetic import SyntheticSource
from aria_gen2_demo.state import LiveState
from aria_gen2_demo.viz.dashboard import Dashboard


def test_synthetic_pipeline_end_to_end(tmp_path):
    rrd = tmp_path / "out.rrd"
    # 20 s of device time, unpaced (speedup=0 disables real-time sleeping)
    source = SyntheticSource(duration_s=20.0, speedup=0)
    dashboard = Dashboard(spawn_viewer=False, rrd_output_path=str(rrd))
    state = LiveState()
    Pipeline(source, dashboard, state).run()

    # every camera stream produced frames
    for label in ["camera-rgb", "slam-front-left", "camera-et-left"]:
        assert state.frame_counts[label] > 0, f"no frames for {label}"

    # processing produced analytics
    assert state.latest_rgb is not None
    assert state.motion.activity != "unknown"
    assert state.motion.step_count > 0, "gait bounce should register steps"
    assert len(state.trajectory) > 10
    assert len(state.gaze_history) > 10

    # events fired (hand pinch happens at t≈2-3s, activity change early)
    kinds = {e.kind for e in state.events}
    assert "pinch" in kinds, f"expected a pinch event, got {kinds}"

    # agent-facing snapshot is JSON-friendly
    status = state.snapshot_status()
    assert status["session_elapsed_s"] > 10
    assert status["motion"]["step_count"] > 0

    # recording file was written
    assert rrd.exists() and rrd.stat().st_size > 1000


def test_png_snapshot_roundtrip():
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    marked = draw_gaze_marker(img, (32, 32), fixating=True)
    assert marked.shape == img.shape
    png = encode_png(marked)
    assert png.startswith(b"\x89PNG")
    assert len(png) > 100


def test_gaze_projection_matches_synthetic_camera():
    from aria_gen2_demo.processing.gaze import GazeProcessor
    from aria_gen2_demo.samples import EyeGazeSample

    proc = GazeProcessor(image_size=(512, 512), focal_px=400.0)
    center = proc.project_to_rgb(
        EyeGazeSample(0, yaw_rad=0.0, pitch_rad=0.0, depth_m=1.0)
    )
    assert center is not None
    assert abs(center[0] - 256) < 1e-6 and abs(center[1] - 256) < 1e-6

    frame = ImageFrame("camera-rgb", 0, np.zeros((512, 512, 3), dtype=np.uint8))
    assert frame.image.shape == (512, 512, 3)


def test_gaze_projection_cpf_convention():
    """CPF convention: +yaw looks right (image right), +pitch looks DOWN."""
    from aria_gen2_demo.processing.gaze import GazeProcessor
    from aria_gen2_demo.samples import EyeGazeSample

    proc = GazeProcessor(image_size=(512, 512), focal_px=400.0)
    right = proc.project_to_rgb(EyeGazeSample(0, yaw_rad=0.2, pitch_rad=0.0, depth_m=1.0))
    down = proc.project_to_rgb(EyeGazeSample(0, yaw_rad=0.0, pitch_rad=0.2, depth_m=1.0))
    assert right is not None and right[0] > 256 and abs(right[1] - 256) < 1e-9
    assert down is not None and down[1] > 256 and abs(down[0] - 256) < 1e-9


def test_synthetic_gaze_angles_agree_with_3d_direction():
    """The 2D projection (from yaw/pitch) and the 3D gaze ray
    (direction_device, which drives the world view) must land on ~the same
    RGB pixel — the vertical axis was mirrored between them once."""
    from aria_gen2_demo.processing.gaze import GazeProcessor
    from aria_gen2_demo.samples import EyeGazeSample
    from aria_gen2_demo.sources.synthetic import RGB_FOCAL, RGB_SIZE, SyntheticSource

    src = SyntheticSource(duration_s=2.0, speedup=0)
    sample = next(s for s in src.samples() if isinstance(s, EyeGazeSample))
    proc = GazeProcessor(image_size=(RGB_SIZE, RGB_SIZE), focal_px=RGB_FOCAL)
    from_angles = proc.project_to_rgb(sample)
    from_ray = proc.project_device_point(sample.direction_device)
    assert from_angles is not None and not np.any(np.isnan(from_ray))
    # small tolerance: tan-per-axis vs true perspective differ slightly
    assert abs(from_angles[0] - from_ray[0]) < 5
    assert abs(from_angles[1] - from_ray[1]) < 5
