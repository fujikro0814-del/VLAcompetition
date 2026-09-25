"""Tests for teleop.replay (C4): a headless recorded episode must replay
bit-exactly per step, within 5 mm from the 20 Hz actions, and the wrong
action definitions must be detected."""
import os
import socket

import mujoco
import numpy as np
import pytest

from teleop import collect, ledger, recorder, replay
from teleop.dualsense_device import FakeDualSenseInput, RawPadState

SCENE = os.path.join(os.path.dirname(__file__), "..",
                     "assets", "panda", "teleop_scene.xml")


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(os.path.abspath(SCENE))


@pytest.fixture(scope="module")
def renderer(model):
    r = mujoco.Renderer(model, collect.IMAGE_SIZE, collect.IMAGE_SIZE)
    yield r
    collect.close_renderer(r)


@pytest.fixture(scope="module")
def episode(model, renderer, tmp_path_factory):
    raw = tmp_path_factory.mktemp("03_収録") / "raw"
    data = mujoco.MjData(model)
    controller = collect.make_collect_controller(model, data)
    pad = FakeDualSenseInput(profile="collect", deadzone=0.15)
    integrator = collect.make_integrator(pad, model, controller)
    integrator.start()
    start = collect.settle_start_state(model)
    led = ledger.EpisodeLedger.init(raw.parent / "ledger", [(raw, "production")], backup_dir=None,
                                    production_hosts=[socket.gethostname()])
    session = collect.CollectSession(model, data, controller, integrator, start,
                                     renderer, raw, "00", 8, episode_ledger=led, kind="production")

    def run(steps, **inputs):
        buttons = {k: inputs.pop(k) for k in ("circle", "triangle", "square") if k in inputs}
        raw_state = RawPadState(**inputs)
        raw_state.buttons.update(buttons)
        pad.backend.state = raw_state
        session.handle_input(integrator.refresh())
        for _ in range(steps):
            controller.update(integrator)
            mujoco.mj_step(model, data)
            session.after_step()

    run(1, circle=True)
    run(1)
    run(300, ly=1.0, lx=-0.5)       # accelerate / move
    run(137)                          # stop (not on a window boundary)
    run(400, r2=1.0)                  # descend
    run(1, square=True)               # close
    run(212)
    run(350, l2=1.0, lx=1.0)          # lift and move
    run(1, square=True)               # open
    run(240)
    run(collect.SAVE_HOLD_STEPS + 5, triangle=True)
    session.close()
    path = next(p for p in raw.glob("*/ep_*") if not p.name.endswith(".partial"))
    return replay.load_episode(path)


def test_step_replay_is_bit_exact(episode, renderer, model):
    rep = replay.replay(episode, "step", renderer, model)
    r = replay.compare(episode, rep)
    assert r["bit_exact_state"] and r["gripper_mismatch_frames"] == 0
    assert r["x_des_err_max_m"] == 0.0
    assert r["img_overhead_max_diff"] <= replay.IMAGE_TOL
    assert r["img_wrist_max_diff"] <= replay.IMAGE_TOL


def test_window_actions_reproduce_trajectory(episode, renderer, model):
    assert episode.data["gripper_closed"].any()          # the script closed it
    rep = replay.replay(episode, "window", renderer, model)
    r = replay.compare(episode, rep, images=False)
    assert r["ee_err_max_m"] < replay.WINDOW_TOL_M
    assert r["x_des_err_max_m"] < 1e-12                   # frames land on x_des
    assert r["gripper_mismatch_frames"] == 0


@pytest.mark.parametrize("mode", ["last_step", "shifted"])
def test_wrong_action_definitions_are_detected(episode, renderer, model, mode):
    rep = replay.replay(episode, mode, renderer, model)
    r = replay.compare(episode, rep, images=False)
    assert r["ee_err_max_m"] >= replay.WINDOW_TOL_M


def test_episode_directory_is_not_modified(episode, renderer, model):
    before = {p: p.stat().st_mtime_ns for p in episode.path.rglob("*")}
    replay.replay(episode, "window", renderer, model)
    assert {p: p.stat().st_mtime_ns for p in episode.path.rglob("*")} == before


def test_scripted_pad_presses_once_per_toggle(model):
    data = mujoco.MjData(model)
    controller = collect.make_collect_controller(model, data)
    pad = replay.ScriptedPad(controller)
    pad.reset(controller.target_pos)
    controller.gripper_closed = False
    controller.prev_grip = False
    pad.closed_target = True
    controller.update(pad)                  # one rising edge -> closed
    assert controller.gripper_closed is True
    for _ in range(5):                      # target reached: no more presses
        controller.update(pad)
    assert controller.gripper_closed is True
    pad.closed_target = False
    controller.update(pad)
    assert controller.gripper_closed is False
