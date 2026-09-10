# SO-101 overhead-camera pick-and-place: research report (2026-09-09)

Rig recap: SO-101 (STS3215, 1/345), fixed Arducam B0495 overhead, taught 40-point grid -> homography + division-model k + tilted z-plane, DLS IK. LOO error ~4 mm mean / 11 mm max. Assume ~250 mm lever from shoulder to fingertip for the mm conversions below.

## 1. Reported problems with SO-100/101, Koch, and similar servo arms

- Independent bench test of STS3215 (Robonine): backlash 0.87 deg measured vs <=0.5 deg datasheet; firmware dead zone of 10 encoder counts (0.88 deg); repeatability +/-0.3 mm at 10 cm lever (0.17 deg, ~2 counts); under 15 kg.cm load the servo sits 20-30 counts (1.8-2.6 deg) off target. https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/
- Peer-reviewed test stand (HardwareX/PMC): STS3215 backlash 7.0 counts (0.62 deg) unloaded, 14.8 counts (1.30 deg) with ~0.3 kgf counter-torque, i.e. backlash roughly doubles under load. STS3250 is ~5x better (1.5 counts unloaded). https://pmc.ncbi.nlm.nih.gov/articles/PMC13087586/
- Thermal: encoder shifts ~0.02 deg/C above 35 C; servos reach ~48 C after 10 min; advice is to recalibrate zero every ~30 min or apply a linear temperature offset. https://www.aliexpress.com/s/wiki-ssr/article/feetech-sts3215-servo_1005009271910382
- Whole-arm repeatability quoted by vendors: +/-2 mm (Seeed/Robotics Center) to +/-2-4 mm (OpenArm-vs-SO-101 comparison). No one publishes an absolute-accuracy number. https://www.roboticscenter.ai/learn/robot-arms/openarm-vs-so101
- URDF vs real: "using the default URDF leads to systematic pose errors that accumulate along the kinematic chain"; URDF zero differs from LeRobot calibration zero. https://foxglove.dev/blog/visualizing-lerobot-so-100-using-foxglove , https://huggingface.co/blog/zfff/umi-lerobot
- Eye-to-hand on SO-101 via cv2.calibrateHandEye failed for one user by ~0.5 m despite 0.02 px reprojection error (tiny 3x3 board, cardboard mount); they abandoned it for point-cloud ICP. https://forum.opencv.org/t/eye-to-hand-calibration-using-lerobot/24195
- Koch/Dynamixel: the reported fix for jitter/overshoot is lowering P-gain, not hardware. https://huggingface.co/blog/zuoxingdong/mobile-manipulation-lekiwi-pincopen

What this means for this rig: at 250 mm, 0.62-1.30 deg backlash = 2.7-5.7 mm and the 10-count dead zone alone = ~3.8 mm. Your 4 mm mean / 11 mm max is essentially the servo dead zone plus load-dependent backlash on shoulder/elbow, not the camera model. Random repeatability (0.17 deg ~ 0.7 mm) is far below your error, so the residual is systematic and correctable.

## 2. Calibration approaches that beat the hand-taught grid

- ChArUco intrinsics (OpenCV): one-time, works with partial board views; disable marker corner refinement when using homography from ChArUco. https://docs.opencv.org/3.4.20/df/d4a/tutorial_charuco_detection.html
- Automatic workspace calibration by homography from ArUco tags (IEEE CASE 2023, code public): tags at known world coords -> homography -> object world XY; exactly your model but re-solved from tags every frame, so a camera bump costs nothing. https://github.com/testbedCIIRC/Robot-Vision-PickPlace
- Full 6-DoF calibrateHandEye needs many diverse orientations "around a sphere"; a fixed camera with a marker on the gripper works only if the gripper is moved through varied poses. https://robotwiki.cs.lth.se/documentation:vision:opencv_hand_eye_calibration
- Low-cost TCP/touch calibration: 4-point method (touch a fixed point from >=4 orientations) and "constrain TCP, vary orientation, record joints" self-calibration. https://control.com/technical-articles/methods-of-performing-robot-tool-center-point-calibration/ , https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6210369/
- Nothing found for a "tap the tag" scheme on SO-101 specifically; the LeRobot hand-eye issue #718 got no answers. https://github.com/huggingface/lerobot/issues/718

What this means for this rig: do not chase 6-DoF hand-eye; a 5-DOF arm cannot supply the orientation diversity and FK error (backlash + URDF) bounds it anyway. Keep the planar model but change the reference frame: (a) capture the four corner ArUcos now and express the whole taught grid in tag coordinates; (b) after any rig move, re-detect tags (seconds) and hand-touch 3-4 printed crosses on the tag sheet to re-fit only a 2D similarity between FK-XY and tag-XY (~5 min, not 1 h). Corner jitter at 960x600 is sub-pixel, i.e. ~0.2-0.5 mm at your ~0.5 mm/px scale, well below arm error.

## 3. Sources of residual error and fixes

- Dead zone: CW/CCW dead-zone registers 26/27 (default 1 in one reference, 10 counts observed by Robonine) and Min_Startup_Force (24) are writable; smaller dead zone = tighter settle at the cost of hunting. https://github.com/commanderfun/STS3215/blob/main/REGISTER_REFERENCE.md
- Backlash: mechanical fix (dual biased servos) cut 14.8 -> 2 counts but costs a second motor per joint; software fix is consistent approach direction + iterative re-command. https://robonine.com/backlash-compensation-in-sts3215-servo-actuators/
- Gravity/load sag: 20-30 counts at 15 kg.cm; the STS3215 encoder is on the output shaft, so Present_Position already reports the real sagged angle. Compensation = read residual after settle and re-command, or fit residual = k_j * gravity_torque_j(q). https://robonine.com/testing-of-feetech-sts3215-servomotor-backlash-repeatability-and-torque/
- Fingertip-to-roll-axis offset: standard TCP 4-point / circle-fit method; on this rig, rotate wrist roll over a fixed spot with a coloured dot on the fingertip and fit a circle in the overhead image to get the offset vector directly in tool frame. https://control.com/technical-articles/methods-of-performing-robot-tool-center-point-calibration/
- Lens: the B0495 ships a 95 deg "low distortion" M12; a ChArUco fit will tell you whether your single-k division model is leaving anything (probably <1 px at the edges). https://www.arducam.com/arducam-2-3mp-ar0234-color-global-shutter-usb-3-0-camera-module.html
- Thermal drift: 0.02 deg/C above 35 C -> ~0.3 deg (1.3 mm) between cold and warm arm. Warm up 10 min before teaching/testing.

What this means for this rig: try, in order, (1) settle-and-re-command loop (goal minus Present_Position > 3 counts -> re-send; cap 3 iterations), (2) always approach the pick from +Z with the same shoulder/elbow direction, (3) TCP circle-fit for the 7 mm offset, (4) only then a per-joint sag table.

## 4. Touch-down / contact sensing with STS3215 load/current

- Registers: Present_Load addr 60 (bits 0-9 magnitude, 0-1000 = 0-100 %, bit 10 direction), Present_Current addr 69 (x6.5 mA), Torque_Limit 48, Protection_Current 28, Overload_Torque 36 (default 80 %), Protection_Time 35. https://github.com/commanderfun/STS3215/blob/main/REGISTER_REFERENCE.md
- Vendor protection: output is disabled if current > 2 A for 2 s or stall torque > 80 % for 2 s (both customisable). https://www.waveshare.com/wiki/ST3215_Servo
- Present_Load rises when you push on the horn while holding position (tutorial demo); no published noise floor, sign convention or latency. https://github.com/commanderfun/STS3215
- No SO-101 "descend until contact" implementation found with thresholds; LeRobot exposes the registers via FeetechMotorsBus.read but ships no example.

What this means for this rig: use position-tracking error (Goal - Present_Position, output-shaft encoder) as the primary contact signal and Present_Current as a confirm; Present_Load looks like a PWM-duty proxy and its quiet-state value is undocumented. Descend in 1 mm steps at <=5 mm/s with Torque_Limit lowered to ~30 % on the wrist joint, read after each step (single-register read ~1-2 ms at 1 Mbps, so a 30-50 Hz loop is fine); expect ~1 mm height resolution and set the dead-zone consideration (10 counts) as your minimum detectable offset. Keep protection thresholds default so a missed detect just trips protection instead of stripping gears.

## 5. Compliant jaws / fingertips

- Official Compliant_Moving_Jaw_SO101.stl: TPU 95A, 20 % infill, hollowed fin-ray ribs, same external geometry (drop-in), supports needed and hard to remove. https://github.com/EmbodiedAI-Group/SO-ARM101-6DoF/blob/main/Optional/Compliant_Gripper/README.md
- Silicone fingertip mould + TPU pad for SO-100/101 (NekoMaker). https://www.thingiverse.com/thing:7152864 , https://www.thingiverse.com/thing:7153144
- Fin-ray gripper for SO-101 (MakerWorld). https://makerworld.com/en/models/2075813-so101-robot-arm-fin-ray-gripper
- Robonine parallel gripper: 84 mm stroke, 120 N, ~$62, needs its own STS3215 and changes the TCP. https://github.com/roboninecom/SO-ARM100-101-Parallel-Gripper
- 3M GM400 gripper tape is the community fix for glossy parts slipping from PLA jaws. https://huggingface.co/blog/zuoxingdong/mobile-manipulation-lekiwi-pincopen
- Nothing found specifically for 3 mm cylinders on SO-101; the fin-ray jaw conforms around round stock but does not centre it.

What this means for this rig: print the official TPU jaw plus a small TPU tip with a 90 deg V-notch on the fixed jaw so screws/spacers self-centre on the roll axis (that also shrinks the 7 mm offset problem for cylinders). Skip the parallel gripper unless you also redo the TCP calibration.

## 6. White parts on a white table (single RGB, classical CV)

- Backlighting gives instant silhouettes; best for presence/orientation of small parts. https://advancedillumination.com/a-practical-guide-to-machine-vision-lighting/
- Dark-field (low-angle, <45 deg) lighting makes edges and part height reflect toward the camera while flat surfaces stay dark; the recommended first try for white-on-white. https://www.cognex.com/en/tools-and-resources/resource-center/machine-vision-lighting-techniques
- Complementary colour: light/background opposite on the colour wheel maximises contrast; a contrasting (ideally dark) background matters most for small parts that fill few pixels. https://advancedillumination.com/a-practical-guide-to-machine-vision-lighting/
- Cross-polarised light + analyser removes specular hotspots on shiny nylocks/screws. https://advancedillumination.com/application-notes/using-polarizing-filters-in-machine-vision/
- Shadow handling: threshold on low luminance AND low saturation, then morphological open/close. https://opencv.org/shadow-correction-using-opencv/

What this means for this rig: the cheapest fix is a matte dark-blue or black mat under the pick zone (tags re-taped on it); add one low-angle LED strip from the side so thin pieces cast a detectable shadow. Backlighting is off the table unless you replace the table with a light panel.

## Ranked changes

| # | Change | Expected gain | Cost | Risk | Rec |
|---|--------|---------------|------|------|-----|
| 1 | Matte dark mat under pick zone + re-tape ArUcos | White parts detectable; higher contrast for all parts | $10, 15 min | Low | Do now |
| 2 | Capture ArUco reference now; store taught grid in tag frame; rig move = re-detect + touch 3-4 crosses to refit 2D similarity | Rig move 1 h -> ~5 min; removes the 2 deg camera-tilt sensitivity | 2-3 h code | Low; tag corner error ~0.3 mm | Do now |
| 3 | Settle-and-re-command loop on Present_Position (cap 3 iters) + fixed approach direction | 1-3 mm off mean, larger off max (dead zone ~3.8 mm) | 1-2 h | Hunting if dead-zone regs lowered too far | Do now |
| 4 | TCP circle-fit of fingertip-to-roll-axis offset via overhead camera | Kills the ~1 cm rotated-grasp error | 30-60 min | Low | Do now |
| 5 | Official TPU jaw + V-notch TPU tip + GM400 tape | Holds 3 mm cylinders; self-centering | 3-4 h print, TPU needed | TPU print quality/supports | Do now if TPU printer available |
| 6 | Touch-down via tracking error + Present_Current, 1 mm steps, wrist Torque_Limit ~30 % | ~1 mm height sensing for thin parts | 2-3 h | Overload trips, fingertip wear | After baseline |
| 7 | ChArUco intrinsics + undistort, replace single-k model | Probably <1 mm; mainly validates the lens model | 30-45 min | Low | After baseline |
| 8 | Per-joint gravity-sag table from Present_Position residuals vs torque | 1-2 mm at extended reach | 2 h | Overfits if arm warms/cools | After baseline |
| - | Full 6-DoF calibrateHandEye eye-to-hand | none over planar model on 5-DOF arm | days | Failed for another SO-101 user | Skip |
| - | Dual-servo anti-backlash joints | 93 % less backlash | new servos + redesign | High | Skip |

Gaps: no published SO-101 absolute-accuracy figure, no SO-101 contact-sensing thresholds, no SO-101-specific fingertip for small cylinders, no camera-mount rigidity data. The numbers above are single-servo bench results scaled to a 250 mm lever.

---

## Checked on our servos (2026-09-09, read-only register dump of the follower)

| register | arm joints | gripper |
|---|---|---|
| CW/CCW_Dead_Zone | 1 / 1 | 1 / 1 |
| Minimum_Startup_Force | 16 | 16 |
| Overload_Torque | 80 % | 25 % |
| Protection_Current | 310 | 250 |
| Max_Torque_Limit | 1000 | 500 |
| P_Coefficient | 16 | 16 |

So the "10-count dead zone" bench figure does NOT apply to this arm (dead zone is 1 count). The residual on this rig is therefore
load-dependent backlash (0.6-1.3 deg per joint = 2.7-5.7 mm at the fingertip) plus URDF-vs-real geometry, not the controller
dead band. Ranked item 3 (settle-and-re-command) still applies via backlash; do not touch the dead-zone registers.
