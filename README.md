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

**There is NO color-tuning step.** Piece detection is automatic (dark pieces on
a light surface). If you remember a slider-based color tuner, that was the old
prototype — superseded, quarantined in `classical_pick_SUPERSEDED/`, do not use.

## Two pick modes

| `arm.pick_mode` | How it grasps | Needs |
|---|---|---|
| `classical` (default) | Detects the piece from the top camera while the arm is at home (clear view), blends 9 hand-taught poses into a grasp, scripted motion. | 15-min one-time teach session, no training |
| `act` (experimental) | Trained ACT policy drives the grasp. | A trained pick policy on the Hub |

Everything after the grasp (inspect, box drop, home) is scripted from taught
poses in both modes, with interpolated (speed-limited) motion throughout.

---

## Running on the lab Mac (already set up — use these exact commands)

No install needed; the LeRobot environment is already on this machine. Every
command works from any folder. Quit LeLab first — it owns the cameras and the
arm's serial port.

### Step 1 — assign cameras  ✅ DONE (2026-07-28)

Already completed. Re-run ONLY if cameras get replugged or a new camera is
added (the tools refuse to run if this was never done, so you can't forget):

```bash
~/.local/share/uv/tools/lelab/bin/python /Users/adityagautham/Documents/Claude/Projects/autosort/autosort-repo/tools/select_cameras.py
```

Press T on the overhead view, W on the wrist view, Q to finish.

### Step 2 — teach the poses (~15 min, one time)  ⬅️ YOU ARE HERE

```bash
~/.local/share/uv/tools/lelab/bin/python /Users/adityagautham/Documents/Claude/Projects/autosort/autosort-repo/tools/teach.py
```

The arm goes limp on purpose — support it with one hand when the script starts,
then move it physically. Click ONCE on the video window; the green
"last key seen" line at the top confirms every key press reaches the tool.

Teach in this order (the yellow status line tracks your progress):

1. **Grid — nine points, each one takes TWO presses of G:**
   - Put a piece somewhere NEW in the zone. Keep the arm OUT of the camera's
     view. Wait for the green circle on the piece.
   - **Press G once** → a magenta circle locks onto the piece ("TARGET LOCKED").
     The camera has now memorized where the piece is, so it no longer matters
     that the arm is about to block the view.
   - NOW move the gripper onto the piece — specifically:
     * Height: all the way DOWN at grabbing height, not hovering. The piece
       should sit between the finger TIPS, with the tips almost at the table.
       Whatever pose you record here is exactly where the arm will go before
       closing — if you record it hovering, it will try to grab air.
     * Tightness: fingers open with a small, even gap around the piece — a few
       millimeters of clearance each side, not touching it. Test: if the
       fingers closed right now, would they trap the piece? If yes, it's right.
     * The green circle disappearing while you do this is normal — the target
       is already locked (magenta).
   - **Press G again** → the pose is recorded and paired with the locked spot.
     The counter goes up by one.
   - Pressed G at the wrong moment? **X** cancels the lock.
   Spread the nine spots out: four corners, four edge midpoints, one center.
2. **H** — from the arm's position after your ninth grid point, lift the arm
   straight up about 10 cm, press H.
3. **M** — move the arm to a resting pose fully OUT of the camera's view of the
   pickup zone, press M. All detection happens at this pose — the arm must not
   cover any part of the zone.
4. **I** — the "show me what you caught" pose: gripper raised 15–20 cm off the
   table, fingers pointing down at an EMPTY patch of the light surface, press I.
   After every grasp the arm returns here and the wrist camera counts what's
   between the fingers — so the background behind the fingers must be plain
   surface, never the pickup zone or the box. No piece needs to be in the
   gripper when you record this; only the arm position is saved.
5. **B** — hold the gripper over the drop box (placeholder until the real
   enclosure exists), press B.
6. **O** — fingers fully open, press O. Then **C** — squeeze the fingers closed
   onto a piece by hand, press C.
7. **S** — saves taught.json. If anything is missing it prints exactly what in
   the terminal. **Q** quits.

### Step 3 — first real run

Edit `config.yaml`: set `dry_run: false` (near the top). Put ONE piece in the
pickup zone. Keep a hand near the arm's power switch. Then:

```bash
~/.local/share/uv/tools/lelab/bin/python /Users/adityagautham/Documents/Claude/Projects/autosort/autosort-repo/run.py
```

It picks the piece, shows it to the wrist camera, drops it in the box, logs the
would-be classification ("unknown" until the classifier exists) and routing,
returns home, and waits for Enter before the next piece (step mode).

Testing ladder: one piece at a time until ~10 clean picks → then 3–4 pieces
spread out, not touching (it clears them largest-first) → piles wait for the
vibrating tray.

### If something misbehaves

- "bus glitch on connect … retrying" lines: normal, ignore (flaky servo cables;
  it retries through them). If it fails all 4 attempts: power-cycle the arm and
  reseat the 3-pin servo cables.
- Arm reaches consistently offset from pieces: grid points were too clustered —
  re-run Step 2 with the nine spots spread wider.
- Right place, bad grip: re-teach O and C, or teach the grid poses slightly lower.
- Camera or lighting moved: re-teach (the pixel→pose map assumes a fixed camera).

---

## Fresh machine (teammates)

```bash
./scripts/install.sh
```

```bash
source .venv/bin/activate
```

```bash
python run.py --dry-run
```

That simulates the whole loop with no hardware. Then follow the same steps as
above, using plain `python` instead of the long interpreter path, and
`./scripts/find_ports.sh` to set `arm.port` in `config.yaml`.

## Configure — everything lives in `config.yaml`

| Section | What you set |
|---|---|
| `run` | `continuous` vs `step` mode, `dry_run`, when to give up |
| `arm` | port, `pick_mode`, taught-pose file, ACT policy id (act mode) |
| `cameras` | resolutions; indices come from `tools/select_cameras.py` |
| `perception` | pile / gripper regions, empty-pile threshold |
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
cameras_override.json  # camera roles (created by tools/select_cameras.py)
taught.json            # poses (created by tools/teach.py)
run.py                 # entry point
autosort/              # pipeline, arm (two pick backends), perception, classifier, router, motion, taught
tools/                 # select_cameras.py, teach.py
scripts/               # install.sh, find_ports.sh
```
