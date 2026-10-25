"""AutoSort orchestrator — runs the full bucket -> table -> arm -> cup -> bin cycle.

    python -m orchestrator.run --sim          # no hardware, realistic delays
    python -m orchestrator.run                # the real machine (orchestrator/config.yaml)

Layers (keep them separate so the arm policy can be swapped without touching the cycle):
    cycle.py   the state machine                      (knows nothing about hardware)
    hw.py      Rig: interlocks + the real hardware     (machine/ bus, cameras, arm)
    sim.py     SimRig: every actuator/camera stubbed
    vision.py  count parts on the table, classify the part in the cup
    arm.py     the LeRobot policy wrapper (Henry's layer)
"""
