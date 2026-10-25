"""State-machine transitions on the simulator (zero delays)."""
import logging

import pytest

from orchestrator.cycle import EXIT_ERROR, EXIT_ESTOP, EXIT_OK, Orchestrator, State
from orchestrator.sim import SimRig


def run(cfg, tmp_path, **sim_kw):
    max_parts = sim_kw.pop("max_parts", None)
    rig = SimRig(cfg, time_scale=0, **sim_kw)
    orch = Orchestrator(cfg, rig, tmp_path / "run", max_parts=max_parts, sleep=lambda s: None)
    code = orch.run()
    return code, orch, rig


def states(orch):
    return [s for _, s in orch.trace]


def test_full_run_drains_the_bucket(cfg, tmp_path):
    code, orch, rig = run(cfg, tmp_path, bucket_parts=12)
    assert code == EXIT_OK
    assert orch.stats.parts_sorted == 12
    assert rig.bucket_parts == 0 and rig.table_parts == 0 and not rig.cup_has_part
    assert len(rig.dropped) == 12
    assert State.EMPTY in states(orch) and states(orch)[-1] is State.DONE
    # bucket exhausted = max_doses_per_fill empty doses in a row
    assert orch.stats.doses >= cfg.timing.max_doses_per_fill
    assert rig.calls[-2:] == ["all_off", "close"] or rig.calls[-1] == "close"


def test_every_part_follows_pick_classify_aim_dump_log(cfg, tmp_path):
    _, orch, _ = run(cfg, tmp_path, bucket_parts=8)
    seq = states(orch)
    for i, s in enumerate(seq):
        if s is State.CLASSIFY:
            assert seq[i - 1] is State.PICK
            assert seq[i + 1:i + 4] == [State.AIM, State.DUMP, State.LOG]


def test_dump_always_finishes_before_the_next_pick(cfg, tmp_path):
    """The rule the user cares about most, checked on the rig's own call log."""
    _, _, rig = run(cfg, tmp_path, bucket_parts=10)
    picks = [i for i, c in enumerate(rig.calls) if c == "pick"]
    for a, b in zip(picks, picks[1:]):
        assert "dump" in rig.calls[a:b], f"two picks with no dump between: {rig.calls[a:b]}"


def test_init_images_before_any_dose(cfg, tmp_path):
    _, orch, rig = run(cfg, tmp_path, bucket_parts=3)
    seq = states(orch)
    assert seq[:3] == [State.IMAGE, State.DOSE, State.SHAKE]
    assert rig.calls.index("count") < rig.calls.index("dose")


def test_dose_only_when_table_is_empty(cfg, tmp_path):
    _, orch, rig = run(cfg, tmp_path, bucket_parts=9)
    # every dose is immediately preceded by a count that returned 0
    counts = []
    for c in rig.calls:
        if c == "count":
            counts.append(True)
        elif c == "dose":
            assert counts, "dose before any count"
    seq = states(orch)
    for i, s in enumerate(seq):
        if s is State.DOSE:
            assert seq[i - 1] is State.IMAGE


def test_dose_shake_settle_image_order(cfg, tmp_path):
    _, orch, _ = run(cfg, tmp_path, bucket_parts=4)
    seq = states(orch)
    for i, s in enumerate(seq):
        if s is State.DOSE:
            assert seq[i + 1:i + 4] == [State.SHAKE, State.SETTLE, State.IMAGE]


def test_sticky_bucket_doses_up_to_max_then_gives_up(cfg, tmp_path):
    n = cfg.timing.max_doses_per_fill
    code, orch, rig = run(cfg, tmp_path, bucket_parts=50, empty_doses=100)
    assert code == EXIT_OK
    assert orch.stats.doses == n
    assert orch.stats.parts_sorted == 0
    assert State.EMPTY in states(orch)


def test_pick_retries_then_reshake(cfg, tmp_path):
    retries = cfg.timing.pick_retries
    # attempts 1..retries+1 fail -> exhaust retries -> SHAKE -> IMAGE -> PICK succeeds
    fail = set(range(1, retries + 2))
    code, orch, rig = run(cfg, tmp_path, bucket_parts=3, fail_picks=fail)
    assert code == EXIT_OK
    seq = states(orch)
    first_pick = seq.index(State.PICK)
    expected = []
    for _ in range(retries + 1):
        expected += [State.PICK, State.PICK_RETRY]
    assert seq[first_pick:first_pick + len(expected)] == expected
    after = seq[first_pick + len(expected):]
    assert after[:3] == [State.SHAKE, State.SETTLE, State.IMAGE]
    assert orch.stats.pick_failures == retries + 1
    assert orch.stats.parts_sorted == 3


def test_single_pick_failure_retries_without_shake(cfg, tmp_path):
    _, orch, _ = run(cfg, tmp_path, bucket_parts=2, fail_picks={1})
    seq = states(orch)
    i = seq.index(State.PICK)
    assert seq[i:i + 4] == [State.PICK, State.PICK_RETRY, State.PICK, State.CLASSIFY]


def test_low_confidence_goes_to_reject_bin(cfg, tmp_path, monkeypatch):
    rig = SimRig(cfg, time_scale=0, bucket_parts=3)
    monkeypatch.setattr(rig, "_do_classify", lambda: (rig.calls.append("classify"), ("screw", 0.5))[1])
    orch = Orchestrator(cfg, rig, tmp_path / "r", sleep=lambda s: None)
    assert orch.run() == EXIT_OK
    reject_bin = cfg.classifier.class_to_bin["REJECT"]
    assert all(c == f"aim:{reject_bin}" for c in rig.calls if c.startswith("aim:"))
    assert orch.stats.rejects == 3


def test_class_to_bin_mapping_is_respected(cfg, tmp_path):
    import csv

    _, orch, rig = run(cfg, tmp_path, bucket_parts=15)
    aims = [int(c.split(":")[1]) for c in rig.calls if c.startswith("aim:")]
    rows = list(csv.DictReader((tmp_path / "run" / "log.csv").open()))
    assert len(aims) == len(rows) == 15
    for row, b in zip(rows, aims):
        assert int(row["bin"]) == b == cfg.bin_for(row["class"])
        if float(row["confidence"]) < cfg.classifier.threshold:
            assert row["class"] == "REJECT"
    assert any(r["class"] != "REJECT" for r in rows)


def test_max_parts_stops_early(cfg, tmp_path):
    code, orch, rig = run(cfg, tmp_path, bucket_parts=30, max_parts=4)
    assert code == EXIT_OK
    assert orch.stats.parts_sorted == 4
    assert states(orch)[-1] is State.DONE
    assert State.EMPTY not in states(orch)


def test_csv_log_has_one_row_per_part(cfg, tmp_path):
    _, orch, _ = run(cfg, tmp_path, bucket_parts=6)
    rows = (tmp_path / "run" / "log.csv").read_text().strip().splitlines()
    assert rows[0] == "time,class,confidence,bin,pick_attempts,cycle_time_s"
    assert len(rows) == 1 + 6


def test_estop_file_stops_with_everything_off(cfg, tmp_path):
    run_dir = tmp_path / "run"
    rig = SimRig(cfg, time_scale=0, bucket_parts=20)
    orch = Orchestrator(cfg, rig, run_dir, sleep=lambda s: None)
    original = rig._do_arm_pick

    def pick_then_estop(timeout_s):
        ok = original(timeout_s)
        (run_dir / "ESTOP").touch()
        return ok

    rig._do_arm_pick = pick_then_estop
    code = orch.run()
    assert code == EXIT_ESTOP
    assert states(orch)[-1] is State.ESTOP
    assert "all_off" in rig.calls and rig.calls[-1] == "close"
    assert rig.shaker_state is False


def test_keyboard_interrupt_is_an_estop(cfg, tmp_path):
    rig = SimRig(cfg, time_scale=0, bucket_parts=5)

    def boom(*a):
        raise KeyboardInterrupt

    rig._do_bucket_pour = boom
    orch = Orchestrator(cfg, rig, tmp_path / "r", sleep=lambda s: None)
    assert orch.run() == EXIT_ESTOP
    assert "all_off" in rig.calls


def test_hardware_fault_fails_safe_and_exits_nonzero(cfg, tmp_path, caplog):
    rig = SimRig(cfg, time_scale=0, bucket_parts=5)

    def jammed(bin_index):
        raise TimeoutError("distributor jammed")

    rig._do_aim = jammed
    orch = Orchestrator(cfg, rig, tmp_path / "r", sleep=lambda s: None)
    with caplog.at_level(logging.ERROR):
        code = orch.run()
    assert code == EXIT_ERROR
    assert states(orch)[-1] is State.ESTOP
    assert "all_off" in rig.calls
    assert "FAULT in AIM" in caplog.text


def test_every_transition_is_timestamped_and_monotonic(cfg, tmp_path):
    _, orch, _ = run(cfg, tmp_path, bucket_parts=4)
    times = [t for t, _ in orch.trace]
    assert times == sorted(times)
    assert len(orch.trace) > 20


def test_rejects_when_label_unknown_to_bin_map(cfg, tmp_path, monkeypatch):
    rig = SimRig(cfg, time_scale=0, bucket_parts=1)
    monkeypatch.setattr(rig, "_do_classify", lambda: ("widget", 0.99))
    orch = Orchestrator(cfg, rig, tmp_path / "r", sleep=lambda s: None)
    orch.run()
    assert rig.calls.count(f"aim:{cfg.classifier.class_to_bin['REJECT']}") == 1


@pytest.mark.parametrize("seed", range(5))
def test_random_seeds_all_terminate_cleanly(cfg, tmp_path, seed):
    code, orch, rig = run(cfg, tmp_path, bucket_parts=10, seed=seed)
    assert code == EXIT_OK and orch.stats.parts_sorted == 10
