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
    in the constructor, i.e. before the policy is ever called."""

    def __init__(self, checkpoint_dir):
        self.conversion = load_training_spec(checkpoint_dir)
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)

    def build(self, raw_by_view: dict, proprio: dict, task: str) -> dict:
        """raw_by_view: {view: raw render}; proprio: one frame of the recorder's fields
        {ee_pos (3,), ee_quat (4,), fingers (2,), joints (7,)} read from the simulation."""
        return {**observation_images(raw_by_view), "observation.state": vla_state.policy_state_frame(proprio),
                "task": str(task)}
