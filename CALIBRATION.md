# Calibration — the VERIFY-by-experiment values

All five servo positions are read the same way: free the servo, move the
mechanism by hand to the position you want, read the encoder, write the number
into `orchestrator/config.yaml`. Takes ~10 minutes for the whole machine.

Everything below talks to `/dev/machine` (the second Waveshare adapter). Make
sure the orchestrator and any other script that opens the bus are closed first.

```bash
source .venv/bin/activate
python machine/machine.py scan          # expect: servos answering on /dev/machine: [7, 8, 9]
```

## The read loop

Two ways. **Jog** (drive it from the keyboard, works over SSH):

```bash
python machine/machine.py --port /dev/ttyACM1 jog <ID>
#   ←/→ (or ↓/↑) nudge by the step   [ / ] halve / double the step   t  torque off to move by hand
#   SPACE records the current position (printed again on quit)   q  quit
```

or **by hand** (servo limp, read the encoder):

```bash
python machine/machine.py torque <ID> off   # the servo goes limp
python machine/machine.py where  <ID>       # prints the position ~10x/s; Ctrl-C to stop
```

Stop at the position you want and note the number. Positions are 0–4095 per
turn (4096 counts = 360°, so 1° ≈ 11.4 counts). Until the udev rule is
installed pass `--port /dev/ttyACMn` (the port whose `scan` shows `[7, 8, 9]`,
not the one with `[1..6]` — that is the arm).

**Mind the wrap.** If your two positions straddle 4095→0 (e.g. REST reads 4000
and POUR reads 600) the servo in position mode will go the long way round.
Rotate the servo horn one spline on its shaft so the whole travel sits well
inside the range, then re-read.

## ID 7 — bucket tilt → `bucket.rest_pos`, `bucket.pour_pos`

1. `torque 7 off`, run `where 7`.
2. Hold the bucket level / fully back (parts stay in) → **rest_pos**.
3. Tip it until parts clearly slide out onto the table, not so far that the
   whole load dumps at once → **pour_pos**. Expect 45–60° of travel ≈ 500–700 counts.
4. `python machine/machine.py move 7 <rest_pos>` then `move 7 <pour_pos>`:
   watch that nothing collides and the pour is a *dose*, not an avalanche.
   Tune `bucket.dwell_s` (1.5 s) and the pour angle together: smaller angle +
   longer dwell = gentler dose.

## ID 9 — cup tipper → `cup.hold_pos`, `cup.dump_pos`

1. `torque 9 off`, `where 9`.
2. Cup level, a part stays put → **hold_pos**.
3. Tip until a part reliably slides down the chute (try a nut and a standoff —
   the standoff is the one that sticks) → **dump_pos**.
4. `move 9 <hold>` / `move 9 <dump>` with a part in the cup to confirm.
   `cup.dwell_s` (0.8 s) is how long it stays tipped.

## ID 8 — distributor → `distributor.bin0_servo_pos`

The ring turns ~3.26 servo turns per revolution, so the servo reading alone can
not tell you which bin the ring faces — that is why there is a homing contract:

> **The ring is parked at bin 0 whenever the orchestrator starts.**

1. Decide which bin is bin 0 and mark it on the ring and the frame.
2. `torque 8 off`, `where 8`. Turn the ring by hand until the chute is centred
   over bin 0 → **bin0_servo_pos**.
3. `python machine/machine.py aim 3 --bin0 <bin0_servo_pos>` — the ring should
   land over bin 3. Then `aim 6`, `aim 0`, `aim 4`. If every landing is a little
   short or long by an amount that grows with distance, the **ratio** is off:
   count the teeth on the ring gear and the pinion and set
   `distributor.ratio = ring_teeth / pinion_teeth`.
4. Leave the ring at bin 0. If it is moved later (someone bumps it, a jam), put
   it back at bin 0 before the next run. The orchestrator checks the servo
   reading against `bin0_servo_pos` at INIT, which catches a nudge but not a
   ring parked at a different bin that happens to give the same reading.

`distributor.tolerance_counts` (±20 servo counts ≈ ±0.5° of ring) and
`settle_s` (0.3 s) are the "arrived" criterion.

## Shaker — `shaker.burst_s`

Set the inline knob to about 30 %. With a dose on the table:

```bash
export KASA_USER=... KASA_PASS=...
python machine/machine.py shake 1.0
python machine/machine.py rhythm 0.8 0.4 3
```

You want parts spread into a single layer without walking off the edge. Adjust
`burst_s` (1.0 s) first, the knob second. Bursts under 0.5 s are not meaningful
through the smart plug (≈0.2 s latency each way).

## Cameras

* `./vision-tools/2_lock_camera_linux.sh /dev/video<N>` before judging any image.
  The orchestrator does this at INIT; settings reset on unplug/reboot.
* `python vision-tools/3_consistency_test.py` (from the Mac tools) — PASS when
  spread < 1.0 — confirms the lock took.
* `cameras.table_roi` / `cameras.cup_roi`: fractions `[x0, y0, x1, y1]` of the
  frame. Take a grab, open it, read the pixel corners, divide by width/height.
* `vision.min_part_area`: the smallest part (a #8 nut) should be several times
  this many pixels; specks and shadows below it.

## Arm — poses, homography, IK

The pick is scripted geometry, so there are three things to teach it.

**Poses** (`arm.poses`). With `lerobot-teleoperate` running, move the arm by
hand and copy the six joint values (degrees) for:

* `home` — parked, clear of the table and the cup, gripper half open. Every
  cycle starts and ends here; `home_tolerance_deg` (6°) is how close counts.
* `inspect` — lifted, holding a part, out of the overhead camera's table view.
* `drop` — gripper directly over the cup (the box on the distributor) at the
  height you want it to open. This is the taught drop position.

**Homography** (`arm.grasp.homography`). Put four markers on the table at known
positions measured in the arm's frame (mm), edit `TABLE_POINTS` in
`scripts/calibrate_homography.py`, run it with `--index <overhead index>`, click
the four markers in order, paste the 3×3 matrix. Redo it whenever the overhead
camera or the arm base moves.

**IK geometry** (`arm.ik`). Measure `link_1`, `link_2`, `tool_len`,
`base_height` with a ruler; set `surface_z` to the gripper-tip height at grasp
and `approach_clearance` to how high above that it should descend from. Then
fix the per-joint `calib: [offset, sign]` so a commanded geometric angle
matches the arm's calibrated zero and direction — command a known table point
with `python -c` or a notebook, see which joint goes the wrong way, flip its
sign, repeat.

**Check it**: `python scripts/preview_grasp.py --index <overhead index>` shows
the blob, the long axis (green) and where the jaws will close (red). Then
`python -m orchestrator.run --max-parts 1`.
