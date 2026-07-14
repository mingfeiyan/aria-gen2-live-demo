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

"""Claude-powered agent over the live Aria stream.

Each question the wearer types is answered by Claude with:
  * the current RGB frame (gaze point burned in) as vision input, and
  * tools that query the live processed state (activity, gaze, events).

Requires ``pip install anthropic`` and an ``ANTHROPIC_API_KEY`` (or an
``ant auth login`` profile).
"""

from __future__ import annotations

import json
from typing import Optional

from ..state import LiveState
from .snapshot import draw_gaze_marker, frame_to_image_block

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """\
You are an assistant embedded in a live-streaming demo for Aria Research Kit \
Gen 2 smart glasses. The user is wearing (or simulating) the glasses; their \
sensor streams are processed on a laptop in real time.

On every question you receive the wearer's current first-person RGB frame. A \
magenta ring on the frame marks where the wearer is looking right now (larger \
ring = stable fixation). You also have tools to query live analytics: motion \
activity, step count, gaze fixation state, hand gestures, scene stats, and \
recent processed events.

Answer questions about what the wearer sees, looks at, and does. Ground every \
claim in the frame or tool data; if the streams don't support an answer, say \
so. Keep answers to a few sentences — this is a live conversation.
"""


class StreamAgent:
    def __init__(self, state: LiveState, dashboard=None):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The AI agent requires the Anthropic SDK:\n    pip install anthropic"
            ) from e
        import anthropic
        from anthropic import beta_tool

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self._state = state
        self._dashboard = dashboard
        self._history: list = []

        state_ref = state

        @beta_tool
        def get_live_status() -> str:
            """Get the current processed state of every Aria stream: session
            time, per-stream frame counts, gaze fixation state and depth, hand
            visibility and pinch state, motion activity and step count, scene
            brightness/sharpness, speech activity, barometer, and the 10 most
            recent processed events."""
            return json.dumps(state_ref.snapshot_status())

        @beta_tool
        def get_gaze_history(seconds: float = 10.0) -> str:
            """Get the wearer's recent gaze angles (yaw/pitch, radians) over the
            last N seconds, subsampled to at most 50 points.

            Args:
                seconds: How far back to look, in seconds.
            """
            history = list(state_ref.gaze_history)
            if not history:
                return json.dumps({"samples": []})
            latest = history[-1][0]
            cutoff = latest - int(seconds * 1e9)
            window = [h for h in history if h[0] >= cutoff]
            step = max(1, len(window) // 50)
            return json.dumps(
                {
                    "samples": [
                        {
                            "t_rel_s": round((ts - latest) / 1e9, 2),
                            "yaw_rad": round(yaw, 3),
                            "pitch_rad": round(pitch, 3),
                        }
                        for ts, yaw, pitch in window[::step]
                    ]
                }
            )

        self._tools = [get_live_status, get_gaze_history]

    def ask(self, question: str) -> str:
        """Ask about the live stream; returns (and prints) the answer."""
        content: list = []
        frame = self._state.get_rgb_frame()
        if frame is not None:
            marked = draw_gaze_marker(
                frame, self._state.gaze.pixel_xy, self._state.gaze.is_fixating
            )
            content.append(frame_to_image_block(marked))
            content.append(
                {"type": "text", "text": f"[current first-person frame]\n{question}"}
            )
        else:
            content.append(
                {"type": "text", "text": f"(no RGB frame received yet)\n{question}"}
            )
        self._history.append({"role": "user", "content": content})

        runner = self._client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=self._tools,
            messages=self._history,
        )

        answer_parts: list = []
        for message in runner:
            # Mirror history so multi-turn conversation keeps tool context
            self._history.append(
                {"role": "assistant", "content": message.content}
            )
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                self._history.append(tool_response)
            for block in message.content:
                if block.type == "text":
                    answer_parts.append(block.text)

        answer = "\n".join(answer_parts).strip() or "(no answer)"
        if self._dashboard is not None:
            ts = self._state.latest_ts_ns
            self._dashboard.log_event(ts, "agent", f"Q: {question}")
            self._dashboard.log_event(ts, "agent", f"A: {answer}")
        return answer


def run_agent_repl(state: LiveState, dashboard=None) -> None:
    """Interactive question loop on the main thread while the pipeline streams."""
    try:
        agent = StreamAgent(state, dashboard)
    except ImportError as e:
        print(f"\n[agent] {e}")
        return
    print(
        "\n[agent] Ask about the live stream (e.g. 'what am I looking at?', "
        "'am I walking?'). Ctrl-D or 'quit' to exit.\n"
    )
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            return
        try:
            print(f"\nclaude> {agent.ask(question)}\n")
        except Exception as e:  # keep the REPL alive on API errors
            print(f"[agent] error: {e}")
