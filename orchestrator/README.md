# AutoSort orchestrator

Runs the whole machine end to end on the Ubuntu PC:

```
bucket tips (ID 7) → table shakes (Kasa plug) → overhead camera counts
  → arm picks one part into the cup (LeRobot ACT) → classify the part in the cup
  → distributor swivels to the bin (ID 8) → cup tips (ID 9) → next part
```

When the overhead camera sees an empty table the bucket tips again; after
`max_doses_per_fill` empty tips in a row the bucket is declared empty and the
run stops. **A new pick never starts until the previous part's dump has finished.**

```
python -m orchestrator.run --sim             # no hardware, realistic delays (~10 s / part)
python -m orchestrator.run --dry-run         # same, zero delays — read the transitions
python -m orchestrator.run                   # the real machine
```

## Layout

| file | what | who touches it |
|---|---|---|
| `orchestrator/config.yaml` | every number, port, pin, threshold | everyone |
| `orchestrator/cycle.py` | the state machine | nobody, ideally |
| `orchestrator/hw.py` | `Rig` (interlocks) + `RealRig` (the hardware) | hardware changes |
| `orchestrator/sim.py` | `SimRig` — all stubs | tests |
| `orchestrator/vision.py` | count parts on the table; classify the part in the cup | vision |
| `orchestrator/arm.py` | the LeRobot policy wrapper | **Henry** — swap the policy here |
| `machine/machine.py` | servo bus (IDs 7/8/9) + shaker, with a CLI | hardware bring-up |
| `vision-tools/2_lock_camera_linux.sh` | pins the overhead camera's exposure/WB/gain | runs automatically in INIT |
| `tests/` | state machine, interlocks, distributor multi-turn math | `python -m pytest tests` |

The cycle only talks to a `Rig`. The policy only lives in `arm.py`. Neither
imports the other, so Henry can replace the policy (or the whole arm wrapper)
without the cycle changing, and the cycle can be tested on a laptop with no
hardware installed.

## PC setup (one time)

Only the project's own venv is touched — nothing system-wide except `apt`
packages, the udev rule and the `dialout` group.

```bash
sudo apt install -y python3-venv v4l-utils
sudo usermod -aG dialout $USER            # serial ports — log out/in afterwards

cd ~/autosort
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r orchestrator/requirements.txt
#   torch for the RTX 5090 (CUDA 12.8 wheels) — only if `python -c "import torch"` fails or has no CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

`orchestrator/requirements.txt` pins **LeRobot 0.5.2** (what Henry trains with),
`feetech-servo-sdk` (the `scservo_sdk` module for the machine bus), `python-kasa`
(the smart plug), `onnxruntime-gpu` (the classifier), `opencv-python`, `pyyaml`.

### udev rules → `/dev/arm` and `/dev/machine`

Both Waveshare adapters look identical, so they are told apart by serial number.
Plug in **one** adapter, find its serial, repeat for the other:

```bash
udevadm info -a -n /dev/ttyACM0 | grep -m1 'ATTRS{serial}'
```

Put the two serials into `scripts/99-autosort.rules`, then:

```bash
sudo cp scripts/99-autosort.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/arm /dev/machine
```

Only **one process** may hold each port. Close any `lerobot-teleoperate`,
`machine.py where`, or notebook before starting the orchestrator.

### Kasa login

The HS103 uses KLAP auth, so the shaker needs the Kasa account login. Never put
it in the YAML — export it and the config reads `${KASA_USER}` / `${KASA_PASS}`:

```bash
export KASA_USER='you@example.com' KASA_PASS='...'
python machine/machine.py shake 1.0        # should buzz the table for 1 s
```

## Filling in every VERIFY value

`config.yaml` ships with the values that can only come from the hardware set to
`null`. A real run refuses to start until they are filled in; `--sim` ignores them.

| key | how to get it |
|---|---|
| `cameras.overhead` / `wrist` | `v4l2-ctl --list-devices` — the Arducam AR0234 is the overhead, the Innomaker is the wrist. Indices can change across reboots; the udev-style `/dev/v4l/by-id/...` path also works where an index is expected if you prefer stability. |
| `cameras.classify_camera` + `cup_roi` | Look at the overhead frame (`python scripts/preview_classify.py` with the box cam index, or any viewer). If the cup is in view: `overhead` and set `cup_roi` to the fraction of the frame containing the cup. Otherwise `wrist` (arm parked at HOME, looking at the cup) or add a third camera (`cup`). |
| `cameras.table_roi` | The fraction of the overhead frame that is table and nothing else (exclude the bucket lip and the cup). |
| `bucket.rest_pos`, `bucket.pour_pos` | CALIBRATION.md — `machine.py torque 7 off`, move by hand, `machine.py where 7`. |
| `cup.hold_pos`, `cup.dump_pos` | same, for ID 9. |
| `distributor.bin0_servo_pos` | same, for ID 8 with the ring at bin 0. |
| `distributor.ratio` | Count the teeth: `ratio = ring_teeth / pinion_teeth` (≈ 3.26 from the 93 / 28.5 mm diameters). A wrong ratio shows up as a growing aim error around the ring. |
| `arm.home_pose` | Read the joints off `lerobot-teleoperate` with the arm parked where the policy episodes start and end. |
| `arm.policy` | HF id or local path. `arm.policy_cameras` must list the same camera names the policy was trained with; if the overhead camera is one of them, the arm owns it and the orchestrator takes counting frames from the robot observation automatically. |
| `arm.policy_cameras` | If Henry's policy was trained with `{wrist, top}`, set `{wrist: wrist, top: overhead}`. |
| `classifier.model` / `labels` | the ONNX export of the YOLOv8n-cls model; `labels` in the model's output order. Put the file under `models/`. |
| `vision.min_part_area`, `dark_parts` | Shake a few parts onto the table, run `python -c` with `vision.count_parts` on a grab, or just run `--skip-arm --max-parts 1` and read the `table count =` lines. |

## Bring-up order (recommended)

1. `python -m pytest tests` — the logic, no hardware.
2. `python -m orchestrator.run --dry-run` — read the transitions once.
3. `python machine/machine.py scan` → expect `[7, 8, 9]`. Calibrate per CALIBRATION.md.
4. `export KASA_USER=... KASA_PASS=...; python machine/machine.py shake 1`.
5. `python -m orchestrator.run --skip-arm --max-parts 3` — feeder, shaker,
   camera count, classifier, distributor and cup, no arm. Picks "succeed" instantly,
   so you will see REJECTs (empty cup) — that is expected; you are watching the
   bucket / shake / count / aim / dump mechanics.
6. `python -m orchestrator.run --manual-pick --max-parts 5` — you move the part
   to the cup when prompted; classification + distribution are real.
7. `python -m orchestrator.run --max-parts 3` — the arm policy, full cycle.
8. `python -m orchestrator.run` — let it drain the bucket.

## Running

```
python -m orchestrator.run [--config PATH] [--max-parts N] [--skip-arm] [--manual-pick]
                           [--sim | --dry-run] [--sim-parts N] [--sim-seed S]
```

Stopping safely: **Ctrl-C**, or from another terminal `touch runs/<timestamp>/ESTOP`.
Either torques off IDs 7/8/9 and the arm, switches the shaker off, and exits 130.
Any exception does the same and exits 1 (`FAULT in <state>` in the log).

## Reading the log

Each run creates `runs/<YYYYMMDD-HHMMSS>/`:

* `run.log` — every state transition with a timestamp (`IMAGE -> PICK`), table
  counts, classifications, servo moves, and the summary line at the end.
* `log.csv` — one row per sorted part:

  | column | meaning |
  |---|---|
  | `time` | wall-clock at DUMP |
  | `class` | the label sent to the distributor (`REJECT` when confidence < threshold or label unknown) |
  | `confidence` | classifier confidence (the raw value, even for REJECT) |
  | `bin` | 0–6 |
  | `pick_attempts` | how many policy attempts this part took |
  | `cycle_time_s` | from the IMAGE that saw the part to the end of DUMP |

Quick stats: `python - <<'EOF'` … or just open it in a spreadsheet. The summary
line in `run.log` gives totals per class.

## State machine

```
INIT ─► IMAGE ─┬─ count==0, doses<MAX ─► DOSE ─► SHAKE ─► SETTLE ─► IMAGE
               ├─ count==0, doses≥MAX ─► EMPTY ─► DONE
               └─ count≥1 ─► PICK ─┬─ ok ─► CLASSIFY ─► AIM ─► DUMP ─► LOG ─► IMAGE
                                   └─ fail ─► PICK_RETRY ─┬─ attempts≤retries ─► PICK
                                                          └─ else ─► SHAKE
```

INIT goes to IMAGE first (not straight to DOSE) because the bucket interlock
requires a zero table count — on an empty table this doses immediately, and on
a table with leftovers from the last run it picks them first.

Interlocks live in `Rig` (hw.py) and raise `InterlockError`:

* shaker never ON while the arm is away from HOME, nor during IMAGE/CLASSIFY
* bucket never moves unless the last count was 0 and the arm is at HOME
* distributor and cup never move while the arm is away from HOME
* a pick never starts with the shaker ON; a pick that does not end at HOME is a fault
