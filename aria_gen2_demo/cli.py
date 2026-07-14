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

"""Command-line entry point for the Aria Gen 2 streaming demo."""

from __future__ import annotations

import argparse
import threading

from .app import Pipeline
from .state import LiveState
from .viz.dashboard import Dashboard


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="aria-gen2-demo",
        description=(
            "Stream Aria Gen 2 glasses data to your laptop, visualize every "
            "stream in Rerun, see processed analytics, and (optionally) talk "
            "to a Claude agent about what the wearer sees."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "vrs", "device"],
        default="synthetic",
        help="synthetic: generated data (default, zero setup) | "
        "vrs: replay a Gen 2 recording in real time | "
        "device: live glasses via the Aria Client SDK",
    )
    parser.add_argument("--vrs", type=str, help="path to a Gen 2 VRS file (vrs mode)")
    parser.add_argument(
        "--speedup", type=float, default=1.0, help="replay speed multiplier"
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="stop after N seconds of device time (synthetic mode)",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="start the Claude Q&A REPL alongside the stream "
        "(needs ANTHROPIC_API_KEY and `pip install anthropic`)",
    )
    parser.add_argument(
        "--rrd-output",
        type=str,
        default=None,
        help="save the visualization to a .rrd file instead of spawning the viewer",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="do not spawn the Rerun viewer (headless)",
    )
    return parser.parse_args(argv)


def build_source(args):
    if args.source == "synthetic":
        from .sources.synthetic import SyntheticSource

        return SyntheticSource(duration_s=args.duration_s, speedup=args.speedup)
    if args.source == "vrs":
        if not args.vrs:
            raise SystemExit("--source vrs requires --vrs <path-to-recording.vrs>")
        from .sources.vrs_replay import VrsReplaySource

        return VrsReplaySource(args.vrs, speedup=args.speedup)
    from .sources.device import DeviceStreamSource

    return DeviceStreamSource()


def main(argv=None) -> None:
    args = parse_args(argv)
    source = build_source(args)
    dashboard = Dashboard(
        spawn_viewer=not args.no_viewer and args.rrd_output is None,
        rrd_output_path=args.rrd_output,
    )
    state = LiveState()
    pipeline = Pipeline(source, dashboard, state)

    print(f"[demo] streaming from source: {source.name}")
    if args.agent:
        stream_thread = threading.Thread(
            target=pipeline.run, name="aria-pipeline", daemon=True
        )
        stream_thread.start()
        from .agent.agent import run_agent_repl

        try:
            run_agent_repl(state, dashboard)
        finally:
            pipeline.stop()
    else:
        try:
            pipeline.run()
        except KeyboardInterrupt:
            pipeline.stop()
    print("[demo] done.")


if __name__ == "__main__":
    main()
