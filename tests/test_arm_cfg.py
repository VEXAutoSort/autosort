"""The orchestrator `arm:` section must map cleanly onto the existing autosort/arm.py
(ArmCfg / CameraCfg) so the IK grasp code is reused unchanged."""
import pytest

from orchestrator.arm import LEGACY_DROP_POSE, legacy_arm_cfg, legacy_camera_cfgs, make_arm
from orchestrator.config import IDENTITY_H, Config


def test_shipped_config_is_geometric_and_arm_owns_overhead(cfg):
    assert cfg.arm.mode == "geometric"
    assert cfg.arm.top_camera == "top"
    assert cfg.arm.cameras["top"] == "overhead"


def test_legacy_arm_cfg_aliases_drop_to_box_drop(cfg):
    legacy = legacy_arm_cfg("/dev/arm", cfg.arm)
    assert legacy.port == "/dev/arm"
    assert legacy.id == cfg.arm.id
    assert legacy.poses[LEGACY_DROP_POSE] == cfg.arm.poses["drop"]
    assert legacy.poses["home"] == cfg.arm.poses["home"]
    assert legacy.gripper_empty_pos == cfg.arm.gripper_empty_pos
    for key in ("link_1", "link_2", "tool_len", "base_height", "surface_z", "approach_clearance", "calib"):
        assert key in legacy.ik  # everything autosort.arm.Arm._ik / execute_grasp reads


def test_legacy_arm_cfg_requires_all_three_poses(cfg):
    cfg.arm.poses.pop("inspect")
    with pytest.raises(ValueError, match="inspect"):
        legacy_arm_cfg("/dev/arm", cfg.arm)


def test_legacy_camera_cfgs_resolve_indices_and_sizes(cfg):
    cams = legacy_camera_cfgs(cfg.arm, cfg.cameras)
    assert set(cams) == {"top", "wrist"}
    assert cams["top"].index == cfg.cameras.overhead
    assert cams["wrist"].index == cfg.cameras.wrist
    assert (cams["top"].width, cams["top"].height) == (960, 600)     # Arducam native
    assert (cams["wrist"].width, cams["wrist"].height) == (640, 480)  # Innomaker


def test_arm_camera_keys_must_be_top_or_wrist(cfg):
    cfg.arm.cameras = {"overhead": "overhead"}
    with pytest.raises(ValueError, match="top"):
        cfg.validate_static()


def test_legacy_camera_cfgs_reject_null_camera(cfg):
    cfg.cameras.wrist = None
    with pytest.raises(ValueError, match="wrist"):
        legacy_camera_cfgs(cfg.arm, cfg.cameras)


def test_make_arm_picks_the_class_without_importing_lerobot(cfg):
    from orchestrator.arm import GeometricArm, PolicyArm

    assert isinstance(make_arm("/dev/arm", cfg.arm, cfg.cameras), GeometricArm)
    cfg.arm.mode = "act"
    assert isinstance(make_arm("/dev/arm", cfg.arm, cfg.cameras), PolicyArm)


def test_real_run_guard_lists_the_arm_verify_items(cfg):
    missing = cfg.missing_hardware_values()
    assert any("homography" in m for m in missing)          # identity = uncalibrated
    assert not any(m.startswith("arm.ik") for m in missing)  # defaults shipped
    assert not any(m.startswith("arm.poses") for m in missing)
    cfg.arm.grasp["homography"] = [[1.1, 0, 5], [0, 1.1, 5], [0, 0, 1]]
    assert not any("homography" in m for m in cfg.missing_hardware_values())
    assert not any(m.startswith("arm") for m in cfg.missing_hardware_values(skip_arm=True))


def test_act_mode_needs_a_policy(cfg):
    cfg.arm.mode = "act"
    assert "arm.policy" in cfg.missing_hardware_values()


def test_bad_arm_mode_is_rejected(cfg):
    cfg.arm.mode = "teleop"
    with pytest.raises(ValueError, match="arm.mode"):
        cfg.validate_static()


def test_identity_constant_matches_shipped_default():
    assert Config().arm.grasp["homography"] == IDENTITY_H
