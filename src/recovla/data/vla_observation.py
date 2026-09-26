"""Evaluation-time entry: raw camera renders -> the observation a trained policy receives.

Images go through vla_image_spec.policy_images, the same function convert_to_lerobot.py used to
write the training dataset, so training and evaluation cannot apply different transforms.

The image spec is checked against the TRAINING run, not against a dataset named at evaluation
time: lerobot-train writes checkpoints to <output_dir>/checkpoints/<step>/pretrained_model, and the
training launcher (train_launcher.py, started by start_training.bat) copies the training dataset's
meta/conversion.json to <output_dir>/conversion.json as soon as lerobot-train has created
<output_dir> (it cannot be done before: lerobot-train refuses an existing output dir). A missing
copy, a version-1 dataset or a different transform table stops evaluation before the policy runs.

observation.state is computed from the raw proprioception of the current frame (the recorder's
fields ee_pos, ee_quat, fingers, joints) by vla_state.policy_state_frame, the same definition
convert_to_lerobot.episode_arrays uses (moved to vla_state on 2026-09-17). No evaluation loop
lives here.

Output format = one LeRobot dataset item as lerobot-train feeds the preprocessor: image features
float32 C x H x W in [0, 1], observation.state float32 (15,) (vla_state), task str (no batch
dimension).
"""
import json
import pathlib

import numpy as np

from recovla.data import vla_image_spec as spec
from recovla.data import vla_state

COPIED_CONVERSION = "conversion.json"     # at <output_dir>/, copied by the training launcher


def training_output_dir(checkpoint_dir) -> pathlib.Path:
    """<output_dir> from either <output_dir> itself or .../checkpoints/<step>/pretrained_model."""
    p = pathlib.Path(checkpoint_dir).resolve()
    if p.name == "pretrained_model" and p.parent.parent.name == "checkpoints":
        return p.parent.parent.parent
    return p


def load_training_spec(checkpoint_dir) -> dict:
    """Read and check the conversion.json copied into the training output directory."""
    out = training_output_dir(checkpoint_dir)
    path = out / COPIED_CONVERSION
    if not path.is_file():
        raise spec.ImageSpecError(
            f"{path} not found: the training launcher (start_training.bat) copies the training dataset's "
            f"meta/conversion.json there; without it the image transforms used in training are unknown")
    conversion = json.loads(path.read_text(encoding="utf-8"))
    spec.check_record(conversion, str(path))
    return conversion


def observation_images(raw_by_view: dict) -> dict:
    """{view: raw H x W x 3 uint8} -> {feature key: float32 C x H x W in [0, 1]} via vla_image_spec."""
    out = {}
    for key, img in spec.policy_images(raw_by_view).items():
        if img.dtype != np.uint8:
            raise spec.ImageSpecError(f"{key}: expected uint8 renders, got {img.dtype}")
        out[key] = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0
    return out


class PolicyObservationBuilder:
    """Construct once per evaluation run with the checkpoint being evaluated; the spec check runs
    in the constructor, i.e. before the policy is ever called.

    Target cue (board 0048): if the training conversion.json has "target_cue", observation.state gets the
    3 cue values (vla_state.CUE_NAMES) from the overhead raw render, computed by the same part and the
    same thresholds as the converter (checked here against configs). Call reset() at the start of every
    trial (the cue keeps the last seen position while the target is hidden)."""

    def __init__(self, checkpoint_dir):
        self.conversion = load_training_spec(checkpoint_dir)
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.cue = None
        self.cue_keep_flag = True
        rec = self.conversion.get("target_cue")
        if rec is not None:
            from recovla.data.convert import cue_tracker
            self.cue = cue_tracker()
            self.cue_keep_flag = "cue_visible" in rec["names"]          # 旗を入力から外した学習（0050 の 2）なら 17 次元
            if (rec["thresholds"] != self.cue.thr.to_json()
                    or rec["names"] != vla_state.cue_names(self.cue_keep_flag)):
                raise spec.ImageSpecError(f"target_cue of the training data {rec['thresholds']} {rec['names']} != "
                                          f"runtime {self.cue.thr.to_json()}")
        self.cue_offset = None                    # 手がかりに足すずれ [m]（雑音の試験＝決裁 0052 の任意。既定はなし）
        self.last_cue = None

    def reset(self) -> None:
        if self.cue is not None:
            self.cue.reset()
        self.last_cue = None

    def build(self, raw_by_view: dict, proprio: dict, task: str, cue_color: str = None) -> dict:
        """raw_by_view: {view: raw render}; proprio: one frame of the recorder's fields
        {ee_pos (3,), ee_quat (4,), fingers (2,), joints (7,)} read from the simulation.
        cue_color: the color whose cue is computed (default: the color word of task; E6 swaps it alone)."""
        state = vla_state.policy_state_frame(proprio)
        if self.cue is not None:
            from recovla.perception.color import color_of_instruction
            self.last_cue = self.cue.update(raw_by_view["overhead"], cue_color or color_of_instruction(task))
            if self.cue_offset is not None:       # 手がかりの (x, y) にだけ足す（保った値は部品の中のまま）
                self.last_cue = self.last_cue.copy()
                self.last_cue[:2] += np.asarray(self.cue_offset, np.float32)
            state = vla_state.with_cue(state, self.last_cue, keep_flag=self.cue_keep_flag)
        return {**observation_images(raw_by_view), "observation.state": state, "task": str(task)}
