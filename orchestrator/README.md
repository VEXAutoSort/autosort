# AutoSort orchestrator

Runs the whole machine end to end on the Ubuntu PC:

```
bucket tips (ID 7) → table shakes (Kasa plug) → overhead camera counts
  → arm picks one part into the cup (IK grasp) → overhead camera classifies the part in the cup
  → distributor swivels to the bin (ID 8) → cup tips (ID 9) → next part
```

"Cup" = the small box on top of the distributor that the arm drops the part
into, the one the tipper servo empties down the ramp into the aimed bin.

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
| `orchestrator/arm.py` | the arm: IK grasp (`autosort/arm.py` + `autosort/grasp.py`) → drop pose → HOME | arm work |
| `machine/machine.py` | servo bus (IDs 7/8/9) + shaker, with a CLI | hardware bring-up |
| `vision-tools/2_lock_camera_linux.sh` | pins the overhead camera's exposure/WB/gain | runs automatically in INIT |
| `tests/` | state machine, interlocks, distributor multi-turn math, arm config mapping | `python -m pytest tests` |

The cycle only talks to a `Rig`. The arm only lives in `arm.py`, and the IK
pick itself is the pre-existing code in `autosort/` (segment the overhead frame
→ PCA centroid + grasp axis → homography to table mm → top-down IK → lift →
scripted move to the taught `drop` pose over the cup → open → HOME). Swapping
the grasp strategy means touching `arm.py` only.

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

`orchestrator/requirements.txt` pins **LeRobot 0.5.2** (the SO-101 driver
`autosort/arm.py` is written against), `feetech-servo-sdk` (the `scservo_sdk`
module for the machine bus), `python-kasa` (the smart plug), `onnxruntime-gpu`
(the classifier), `opencv-python`, `pyyaml`.

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

Only **one process** may hold each port and each camera. Close any
`lerobot-teleoperate`, `machine.py where`, `preview_grasp.py` or notebook before
starting the orchestrator.

### Kasa login

The HS103 uses KLAP auth, so the shaker needs the Kasa account login. Never put
it in the YAML — export it and the config reads `${KASA_USER}` / `${KASA_PASS}`:

```bash
export KASA_USER='you@example.com' KASA_PASS='...'
python machine/machine.py shake 1.0        # should buzz the table for 1 s
```

## Filling in every VERIFY value

`config.yaml` ships with the values that can only come from the hardware set to
`null` (or, for the homography, identity). A real run refuses to start until
they are filled in and tells you which are missing; `--sim` ignores them.

| key | how to get it |
|---|---|
| `cameras.overhead` / `wrist` | `v4l2-ctl --list-devices` — the Arducam AR0234 is the overhead, the Innomaker is the wrist. Indices can change across reboots. |
| `cameras.sizes` | the resolution each camera must deliver; LeRobot refuses a camera that cannot. Arducam 960×600, Innomaker 640×480 are the defaults. |
| `cameras.cup_roi` | the fraction `[x0,y0,x1,y1]` of the **overhead** frame that contains the cup (the drop box on the distributor). Grab a frame, read the pixel corners, divide by width/height. |
| `cameras.table_roi` | the fraction of the overhead frame that is table and nothing else (exclude the bucket lip and the cup). |
| `bucket.rest_pos`, `bucket.pour_pos` | CALIBRATION.md — `machine.py torque 7 off`, move by hand, `machine.py where 7`. |
| `cup.hold_pos`, `cup.dump_pos` | same, for ID 9. |
| `distributor.bin0_servo_pos` | same, for ID 8 with the ring at bin 0. |
| `distributor.ratio` | Count the teeth: `ratio = ring_teeth / pinion_teeth` (≈ 3.26 from the 93 / 28.5 mm diameters). A wrong ratio shows up as an aim error that grows with distance. |
| `arm.poses.home / inspect / drop` | Read the six joints off `lerobot-teleoperate` with the arm parked (home), lifted holding a part (inspect), and over the cup with the gripper where you want it to open (drop — the taught drop position). |
| `arm.grasp.homography` | `python scripts/calibrate_homography.py --index <overhead index>` — 4 markers at known table-mm positions in the arm's frame, click them, paste the matrix. Redo if the camera or the arm moves. |
| `arm.ik.*` | measure the arm: link lengths, base height, the grasp height; then tune `calib` (sign/offset per joint) until `preview_grasp.py`'s red line lands where the gripper actually closes. |
| `classifier.model` / `labels` | the ONNX export of the YOLOv8n-cls model (coming); `labels` in the model's output order. Put the file under `models/`. Until then `backend: catalog` uses the repo's measure-and-match classifier (add `gear` to `class_to_bin` if you use it). |
| `vision.min_part_area`, `dark_parts` | Shake a few parts onto the table and run `--skip-arm --max-parts 1`; read the `table count =` lines. |

## Bring-up order (recommended)

1. `python -m pytest tests` — the logic, no hardware.
2. `python -m orchestrator.run --dry-run` — read the transitions once.
3. `python machine/machine.py scan` → expect `[7, 8, 9]`. Calibrate per CALIBRATION.md.
4. `export KASA_USER=... KASA_PASS=...; python machine/machine.py shake 1`.
5. `python -m orchestrator.run --skip-arm --max-parts 3` — feeder, shaker,
   camera count, classifier, distributor and cup, no arm (no `/dev/arm` needed).
   Picks "succeed" instantly, so you will see REJECTs (empty cup) — expected;
   you are watching the bucket / shake / count / aim / dump mechanics.
6. `python -m orchestrator.run --manual-pick --max-parts 5` — you move the part
   to the cup when prompted; classification + distribution are real.
7. `python scripts/preview_grasp.py --index <overhead>` — the red line is where
   the jaws will close. Then `python -m orchestrator.run --max-parts 3` — the IK
   grasp, full cycle.
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
  | `pick_attempts` | how many grasp attempts this part took |
  | `cycle_time_s` | from the IMAGE that saw the part to the end of DUMP |

The summary line in `run.log` gives totals per class.

## State machine

```
INIT ─► IMAGE ─┬─ count==0, doses<MAX ─► DOSE ─► SHAKE ─► SETTLE ─► IMAGE
               ├─ count==0, doses≥MAX ─► EMPTY ─► DONE
               └─ count≥1 ─► PICK ─┬─ ok ─► CLASSIFY ─► AIM ─► DUMP ─► LOG ─► IMAGE
                                   └─ fail ─► PICK_RETRY ─┬─ attempts≤retries ─► PICK
                                                          └─ else ─► SHAKE
```

A PICK "fails" when the planner finds no graspable blob, the gripper closed on
nothing, or the arm is not back at HOME in time. INIT goes to IMAGE first (not
straight to DOSE) because the bucket interlock requires a zero table count — on
an empty table this doses immediately, and on a table with leftovers from the
last run it picks them first.

Interlocks live in `Rig` (hw.py) and raise `InterlockError`:

* shaker never ON while the arm is away from HOME, nor during IMAGE/CLASSIFY
* bucket never moves unless the last count was 0 and the arm is at HOME
* distributor and cup never move while the arm is away from HOME
* a pick never starts with the shaker ON; a pick that does not end at HOME is a fault

## How the pieces fit (things that bite if changed)

* **Camera ownership.** LeRobot opens the cameras listed in `arm.cameras`, so
  the orchestrator never opens the overhead itself when the arm is in use; it
  takes frames from the robot observation (RGB) and converts them to BGR once
  in `RealRig._frame()`. With `--skip-arm` it opens the overhead directly (BGR).
  `count_parts` and both classifiers expect BGR; `GeometricGrasper.plan` expects
  RGB and gets the raw LeRobot frame. Keep it that way.
* **Pose names.** `autosort/arm.py` calls the drop pose `box_drop`; the
  orchestrator calls it `drop` and aliases it in `arm.py::legacy_arm_cfg`.
* **`arm.cameras` keys** must be `top` / `wrist` — those are the only names
  `autosort/arm.py` registers with LeRobot. One of them must map to `overhead`.
* **Distributor homing.** The ring must physically sit at bin 0 at start. The
  servo wraps ~3.26× per ring turn, so the reading alone cannot tell bins apart.
* **Two bus adapters, two SDKs.** The arm port goes through LeRobot's
  `FeetechMotorsBus`; the machine port through `scservo_sdk` directly. They
  must be different devices (`/dev/arm` vs `/dev/machine`).
