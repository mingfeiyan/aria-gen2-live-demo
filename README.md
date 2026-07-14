# Aria Gen 2 Streaming Demo

A demonstration app for **Aria Research Kit (ARK) Gen 2 glasses**: stream data
from the glasses to your laptop, visualize **every stream** live in
[Rerun](https://rerun.io), watch **processed analytics** derived from the
streams in real time, and optionally **talk to a Claude AI agent** about what
the wearer is seeing and doing.

```
┌──────────────────────┐     ┌───────────────────────────┐     ┌──────────────────────────┐
│       SOURCES        │     │      PROCESSING           │     │        OUTPUTS           │
│                      │     │                           │     │                          │
│ device  (live Wi-Fi) │     │ gaze → RGB projection     │     │ Rerun dashboard          │
│ vrs     (replay)     ├────▶│ fixation detection        ├────▶│  · RGB + gaze + hands    │
│ synthetic (built-in) │     │ pinch gesture detection   │     │  · 4 SLAM + 2 ET cams    │
│                      │     │ IMU activity + steps      │     │  · 3D world / trajectory │
│ camera-rgb           │     │ scene stats + attention   │     │  · IMU / baro / audio    │
│ slam-{fl,fr,sl,sr}   │     │ speech activity           │     │  · attention heatmap     │
│ camera-et-{l,r}      │     │        │                  │     │  · event log             │
│ imu-{left,right}     │     │        ▼                  │     │                          │
│ eyegaze/handtracking │     │   LiveState (shared)      │────▶│ Claude agent (Q&A REPL)  │
│ vio/mic/baro0/mag0   │     │                           │     │  vision + live tools     │
└──────────────────────┘     └───────────────────────────┘     └──────────────────────────┘
```

## Quickstart (zero hardware needed)

```bash
pip install -e .          # numpy + rerun-sdk
aria-gen2-demo            # synthetic source, spawns the Rerun viewer
```

The synthetic source simulates a wearer walking around a room, looking at
three posters and periodically pinching — every Gen 2 stream is generated with
realistic timing, so the full dashboard and all analytics light up.

## Replay a real Gen 2 recording

```bash
pip install -e ".[vrs]"   # adds projectaria-tools
aria-gen2-demo --source vrs --vrs /path/to/recording.vrs
```

Replay is paced to the wall clock (use `--speedup 2` to go faster). Sample
Gen 2 recordings and dataset links are in the
[Aria Gen 2 docs](https://facebookresearch.github.io/projectaria_tools/gen2/).
In VRS mode the real device calibration is used to project eye gaze and hand
landmarks onto the RGB frame.

## Stream from live glasses

```bash
pip install projectaria_client_sdk   # ARK partner access required
aria auth pair                       # one-time pairing
aria-gen2-demo --source device       # then start streaming from the companion app
```

The device source bridges the Aria Client SDK's subscription callbacks into
the same pipeline used by replay mode, so the dashboard and agent behave
identically.

## AI agent

```bash
pip install -e ".[agent]"
export ANTHROPIC_API_KEY=sk-ant-...
aria-gen2-demo --agent
```

While streams render in Rerun, a REPL lets you ask Claude about the live
session:

```
you> what am I looking at right now?
claude> You're looking at the blue poster on the wall — your gaze (magenta
        ring) has been fixated on it for about 2 seconds while you walk
        slowly around the room (14 steps so far this session).
```

Each question sends Claude the **current first-person RGB frame with the gaze
point burned in**, plus tools that query live analytics
(`get_live_status`, `get_gaze_history`). Questions and answers are also logged
into the Rerun event panel.

## What's in the dashboard

| Panel | Contents |
|---|---|
| RGB + gaze + hands | `camera-rgb` with the live gaze point (magenta, grows on fixation) and 2D hand skeleton overlay |
| SLAM / ET cameras | 4 SLAM cameras and both eye-tracking cameras |
| 3D world | Device pose + trajectory from VIO, 3D gaze ray, 3D hand skeletons |
| Attention map | Decaying heatmap of gaze direction over the session |
| IMU / analytics | Accel & gyro magnitudes, activity energy, step count, speed, barometric pressure, magnetic field, audio RMS |
| Events | Processed events (fixations, pinches, activity changes, speech) and agent Q&A |

## Processed outcomes

- **Gaze**: projection onto the RGB frame (device calibration in VRS mode,
  pinhole otherwise), I-DT fixation detection, dwell time, attention heatmap
- **Hands**: thumb–index pinch detection with hysteresis, visibility
- **Motion**: gravity-removed accel energy → still / walking / moving
  classification, peak-detection step counter, VIO speed
- **Scene**: brightness / sharpness / frame motion, mic speech activity

## Development

```bash
pip install -e ".[dev]"
pytest tests/            # headless end-to-end test on the synthetic source
```

Useful flags: `--rrd-output out.rrd` (record instead of live viewer, open
later with `rerun out.rrd`), `--no-viewer`, `--duration-s N`, `--speedup X`.

## Repo layout

```
aria_gen2_demo/
├── samples.py            # source-agnostic sample dataclasses
├── sources/              # device (live SDK) / vrs_replay / synthetic
├── processing/           # gaze, hands, motion, scene analytics
├── state.py              # thread-safe LiveState shared with the agent
├── viz/dashboard.py      # Rerun logging + blueprint layout
├── agent/                # Claude agent (vision + live tools) and PNG snapshot
├── app.py                # pipeline orchestrator
└── cli.py                # aria-gen2-demo entry point
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
