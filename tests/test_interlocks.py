"""The global interlocks are enforced by Rig's public methods, not by convention."""
import pytest

from orchestrator.hw import InterlockError
from orchestrator.sim import SimRig


@pytest.fixture
def rig(cfg):
    r = SimRig(cfg, time_scale=0, bucket_parts=10)
    r.init()
    return r


def test_init_leaves_arm_home_shaker_off(rig):
    assert rig.arm_at_home and not rig.shaker_on
    assert rig.calls[:2] == ["init", "shaker_off"]


def test_bucket_refuses_without_a_zero_count(rig):
    with pytest.raises(InterlockError):
        rig.dose()  # no count yet
    rig.table_parts = 3
    rig.count_table()
    with pytest.raises(InterlockError):
        rig.dose()  # table not empty
    rig.table_parts = 0
    rig.count_table()
    rig.dose()  # now allowed
    assert "dose" in rig.calls


def test_bucket_refuses_when_arm_away(rig):
    rig.count_table()
    rig.arm_at_home = False
    with pytest.raises(InterlockError):
        rig.dose()


def test_pick_invalidates_the_table_count(rig):
    rig.table_parts = 2
    rig.count_table()
    rig.pick(5)
    assert rig.table_count is None
    with pytest.raises(InterlockError):
        rig.dose()


def test_shaker_refuses_when_arm_away(rig):
    rig.arm_at_home = False
    with pytest.raises(InterlockError):
        rig.shake(1.0)
    assert "shaker_on" not in rig.calls


def test_shaker_is_off_after_burst_even_if_sleep_raises(rig):
    rig._sleep = lambda s: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        rig.shake(1.0)
    assert rig.calls[-1] == "shaker_off"
    assert not rig.shaker_on


def test_image_and_classify_refuse_while_shaker_on(rig):
    rig.shaker_on = True
    with pytest.raises(InterlockError):
        rig.count_table()
    with pytest.raises(InterlockError):
        rig.classify()
    rig.shaker_on = False


def test_pick_refuses_while_shaker_on(rig):
    rig.shaker_on = True
    with pytest.raises(InterlockError):
        rig.pick(5)
    assert "pick" not in rig.calls


def test_distributor_and_cup_refuse_when_arm_away(rig):
    rig.arm_at_home = False
    with pytest.raises(InterlockError):
        rig.aim(3)
    with pytest.raises(InterlockError):
        rig.dump()
    assert not any(c.startswith("aim:") or c == "dump" for c in rig.calls)


def test_pick_sets_arm_away_then_home(rig):
    rig.table_parts = 1
    seen = []
    original = rig._do_arm_pick

    def spy(timeout_s):
        seen.append(rig.arm_at_home)
        return original(timeout_s)

    rig._do_arm_pick = spy
    assert rig.pick(5) is True
    assert seen == [False]
    assert rig.arm_at_home is True


def test_pick_that_does_not_return_home_is_a_fault(rig):
    rig._do_arm_is_home = lambda: False
    with pytest.raises(InterlockError):
        rig.pick(5)
    assert rig.arm_at_home is False


def test_estop_turns_everything_off_and_never_raises(rig):
    rig.shaker_on = True

    def broken():
        raise RuntimeError("bus gone")

    rig._do_all_off = broken
    rig.estop()  # must not raise
    assert rig.calls[-1] == "shaker_off"
    assert not rig.shaker_on


def test_estop_is_idempotent(rig):
    rig.estop()
    rig.estop()
    assert rig.calls.count("all_off") == 2
