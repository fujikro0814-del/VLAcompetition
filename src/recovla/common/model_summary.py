"""コンパイルした MuJoCo の模型の要約（掲示板 0013 の 4）。

mj_saveModel のバイト列は OS・CPU で変わりうるので、OS によらない比較には数と主な配列を使う
（tests/test_c_port.py、scripts/04_scene_reference.py）。
"""
import hashlib

import mujoco
import numpy as np

COUNTS = ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nmesh", "ncam", "nmocap", "nkey")
ARRAYS = ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia", "body_parentid",
          "jnt_type", "jnt_range", "jnt_qposadr", "dof_damping", "dof_armature", "dof_frictionloss",
          "geom_type", "geom_bodyid", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
          "geom_condim", "geom_friction", "geom_rgba", "geom_group",
          "actuator_trntype", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange",
          "cam_pos", "cam_quat", "cam_fovy", "mesh_vertnum", "mesh_facenum", "key_qpos")
OPTIONS = ("timestep", "gravity")


def mjb_sha256(model) -> str:
    buf = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()


def model_summary(model) -> dict:
    return {
        "mujoco": mujoco.__version__,
        "mjb_sha256_windows": mjb_sha256(model),
        "counts": {k: int(getattr(model, k)) for k in COUNTS},
        "options": {k: np.asarray(getattr(model.opt, k)).tolist() for k in OPTIONS},
        "arrays": {k: np.asarray(getattr(model, k)).tolist() for k in ARRAYS},
    }
