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

## Quickstart

```bash
./scripts/install.sh          # Python 3.12 venv + all dependencies
source .venv/bin/activate
python run.py --dry-run       # runs the whole pipeline with NO hardware (simulated)
```

Then wire up real hardware:

```bash
./scripts/find_ports.sh       # get USB ports -> put them in config.yaml
# edit config.yaml (ports, poses, bins, camera indices), set run.dry_run: false
python run.py
```

That's it — **edit one file (`config.yaml`), run one file (`run.py`)**.

## Configure — everything lives in `config.yaml`

| Section | What you set |
|---|---|
| `run` | `continuous` vs `step` mode, `dry_run`, when to give up |
| `arm` | USB port, ACT policy id, and the scripted `home` / `inspect` / `box_drop` poses |
| `cameras` | indices for `top`, `wrist`, and the `box` Arducam |
| `perception` | pile / gripper regions and the empty-pile threshold |
| `classifier` | model path, the class `labels`, confidence cutoff |
| `router` | controller port and each label's `bins` angle |

## How "pick exactly one" works

1. The **ACT policy** grasps a piece and lifts to the `inspect` pose.
2. **Gripper position** says whether it grabbed anything at all (empty grasp → retry).
3. The **wrist camera** counts pieces in the gripper: `2+` → drop back and retry, `1` → continue.
4. The **top camera** counts pieces left on the tray; several `0` reads in a row → done.

## Plug in your trained models

- **ACT pick policy** — set `arm.policy` to your Hub id (e.g. `VEXAutoSort/act_pick_v1`).
- **Classifier** — drop a TorchScript model at `models/classifier.pt` (224×224 RGB → logits over `labels`). Missing model ⇒ everything is labelled `unknown` so the loop still runs.
- **Router firmware** — the Arduino answers `G<angle>\n` with `OK\n` (see `autosort/router.py`).

## Layout

```
config.yaml            # the one config
run.py                 # the one entry point
autosort/
  config.py            # load + validate config.yaml
  arm.py               # SO-ARM101: ACT pick + scripted place + gripper feedback
  perception.py        # blob-count checks: single-grasp + empty-pile
  classifier.py        # Arducam piece classification
  router.py            # rotating-arm bin routing (serial)
  pipeline.py          # the loop that ties it together  (also `python -m autosort.pipeline`)
scripts/               # install.sh, find_ports.sh
models/                # trained weights (gitignored)
```

## Status

The **structure, control loop, config, and dry-run are complete and runnable.**
Three integration points are marked as stubs until the trained assets exist: the ACT
policy preprocessing (`arm.pick`), the classifier weights (`classifier.py`), and the
router firmware protocol (`router.py`).
