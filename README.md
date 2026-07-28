# AutoSort

Autonomous sorter for small VEX hardware. An **SO-ARM101** picks one piece from a
pile and drops it into an enclosure; a camera **classifies** it; a **rotating arm**
turns to the matching bin and lets it fall in. Repeats until the pile is empty.

**Team 11101B** — Henry, Vihaan, Aditya

```
pile ─▶ SO-ARM101 pick ─▶ enclosure ─▶ classify (Arducam) ─▶ rotating-arm router ─▶ bin
             ▲                                                                        │
             └──────────────────────── repeat until empty ◀───────────────────────────┘
```

Components that aren't built yet (enclosure camera, router arm) are disabled in
`config.yaml` — the pipeline runs end-to-end regardless and logs what it *would*
have done. Flip their `enabled` flags as hardware comes online.

## Two pick modes

| `arm.pick_mode` | How it grasps | Needs |
|---|---|---|
| `classical` (default) | Detects the piece from the top camera while the arm is at home (clear view), blends 9 hand-taught poses into a grasp, scripted motion. | 15-min one-time teach session, no training |
| `act` (experimental) | Trained ACT policy drives the grasp. | A trained pick policy on the Hub |

Everything after the grasp (inspect, box drop, home) is scripted from taught
poses in both modes, with interpolated (speed-limited) motion throughout.

## Quickstart — no hardware

```bash
./scripts/install.sh
```

```bash
source .venv/bin/activate
```

```bash
python run.py --dry-run
```

That simulates the entire loop (fake pile drains over ~6 cycles).

## Real hardware, first session

Do these once, in order, with the arm and both cameras plugged in and no other
app (LeLab!) holding them:

1. Find the arm's serial port and put it in `config.yaml` under `arm.port`:

```bash
./scripts/find_ports.sh
```

2. Assign cameras by looking at them (never hand-edit indices — macOS shuffles
   them). A window shows each camera; press T for the top view, W for wrist:

```bash
python tools/select_cameras.py
```

3. Teach the poses by physically moving the arm (torque releases; ~15 min).
   Follow the on-screen key prompts — 9 grid points over the pile zone, hover,
   home, inspect, box drop, gripper open/closed:

```bash
python tools/teach.py
```

4. Set `run.dry_run: false` in `config.yaml`, put ONE piece in the pile zone,
   keep a hand near the power switch, and run:

```bash
python run.py
```

`run.mode: step` (the default) waits for Enter between pieces — right for the
first sessions. Switch to `continuous` when it's boring.

## Configure — everything lives in `config.yaml`

| Section | What you set |
|---|---|
| `run` | `continuous` vs `step` mode, `dry_run`, when to give up |
| `arm` | port, `pick_mode`, taught-pose file, ACT policy id (act mode) |
| `cameras` | resolutions; indices come from `tools/select_cameras.py` |
| `perception` | pile / gripper regions, empty-pile threshold (assumes light surface, darker pieces) |
| `classifier` | `enabled`, model path, class `labels`, confidence cutoff |
| `router` | `enabled`, controller port, each label's bin angle |

## How "pick exactly one" works

1. The pick (classical or ACT) grasps a piece and lifts to the `inspect` pose.
2. **Gripper position** says whether it grabbed anything at all (empty grasp → retry).
3. The **wrist camera** counts pieces in the gripper: `2+` → drop back and retry, `1` → continue.
4. The **top camera** counts pieces left in the zone; several `0` reads in a row → done.

## Plug in trained models (when ready)

- **ACT pick policy** — train one, set `arm.policy`, set `arm.pick_mode: act`.
  The observation/processor plumbing (lerobot 0.6 pipelines, camera-name
  mapping) is already wired in `autosort/arm.py`.
- **Classifier** — drop a TorchScript model at `models/classifier.pt`
  (224×224 RGB → logits over `labels`), set `classifier.enabled: true`, add a
  `box` camera entry. Missing model ⇒ everything is labelled `unknown`.
- **Router firmware** — the Arduino answers `G<angle>\n` with `OK\n`
  (see `autosort/router.py`), then set `router.enabled: true`.

## Layout

```
config.yaml            # the one config
taught.json            # created by tools/teach.py (poses live here, not in code)
run.py                 # entry point
autosort/              # pipeline, arm (two pick backends), perception, classifier, router, motion, taught
tools/                 # select_cameras.py, teach.py
scripts/               # install.sh, find_ports.sh
```
