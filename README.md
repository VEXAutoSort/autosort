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
| `classifier` | catalog path, class `labels`, `px_per_mm`, the two reject thresholds |
| `router` | controller port and each label's `bins` angle |

## How "pick exactly one" works

1. The **ACT policy** grasps a piece and lifts to the `inspect` pose.
2. **Gripper position** says whether it grabbed anything at all (empty grasp → retry).
3. The **wrist camera** counts pieces in the gripper: `2+` → drop back and retry, `1` → continue.
4. The **top camera** counts pieces left on the tray; several `0` reads in a row → done.

## How classification works

No model, no weights, no training set. Every VEX fastener is #8-32 and every gear is
24 diametral pitch, so part dimensions land on a known lattice — identifying a piece
is **measuring it and snapping to the nearest entry** in `vex_parts.yaml`.

```
frame − empty-stage reference → mask → mm measurement → nearest catalog entry
```

Segmentation is a difference against a reference shot of the empty stage, not a
threshold on the frame. A steel screw is as bright as white paper but still differs
from bare paper.

Two ways a piece comes back `unknown`, and they mean different things:

| | meaning | what to do |
|---|---|---|
| `distance > max_distance` | nothing in the catalog is that shape | odd part, or an odd resting pose |
| `margin < min_confidence` | two *different* labels fit equally well | re-present the piece and re-image |

Both go to the reject bin. A slow bin beats a wrong bin. Some pairs are genuinely
the same picture — a 1/4" standoff standing on end is indistinguishable from a
low-profile hex nut — so both are in the catalog on purpose, which collapses the
margin and sends them to reject instead of guessing.

### Set it up

```bash
python scripts/calibrate_scale.py       # px/mm -> paste into config.yaml
python scripts/preview_classify.py      # live view: measurement, match, distance, margin
```

`px_per_mm` is the one number everything depends on, so redo it if the camera moves.
`preview_classify.py` prints a ready-to-paste catalog row when you press `r` — use it
to replace the published dimensions with your own measurements, and to add the parts
the catalog is missing (black plastic spacer ODs are noted in `vex_parts.yaml`).

Catalog rows need only `label` and `name`; every dimension field is optional and a
missing one is not compared. Start with `length_mm`/`width_mm` and add `extent`,
`circularity`, `solidity`, or `holes` only where two parts actually collide.

## Plug in your trained models

- **ACT pick policy** — set `arm.policy` to your Hub id (e.g. `VEXAutoSort/act_pick_v1`).
- **Router firmware** — the Arduino answers `G<angle>\n` with `OK\n` (see `autosort/router.py`).

## Layout

```
config.yaml            # the one config
run.py                 # the one entry point
autosort/
  config.py            # load + validate config.yaml
  arm.py               # SO-ARM101: ACT pick + scripted place + gripper feedback
  perception.py        # blob-count checks: single-grasp + empty-pile
  classification/      # Arducam piece classification (measure -> match, no model)
    camera.py          #   box cam + empty-stage reference
    vision.py          #   mask -> millimetre Shape
    catalog.py         #   VEX part table + nearest-entry match
    classifier.py      #   ties it together, applies the reject gates
    vex_parts.yaml     #   the part table
  router.py            # rotating-arm bin routing (serial)
  pipeline.py          # the loop that ties it together  (also `python -m autosort.pipeline`)
scripts/               # install.sh, find_ports.sh
models/                # trained weights (gitignored)
```

## Status

The **structure, control loop, config, and dry-run are complete and runnable.**
**Classification is implemented** — it needs `px_per_mm` calibrated and the catalog
checked against real parts, but no training. Two integration points remain stubs
until the hardware exists: the ACT policy preprocessing (`arm.pick`) and the router
firmware protocol (`router.py`).
