"""場面のコンパイル結果の参照値を書く（掲示板 0013 の 4。OS によらない検査の期待値）。

    .venv\\Scripts\\python.exe scripts\\04_scene_reference.py [--check]

scene_g0.xml をコンパイルし、数（nq・nv・nbody・ngeom など）と主な配列（recovla.common.model_summary）を
tests/fixtures/scene_g0_reference.json に書く。書くのは本線の Windows で、mj_saveModel のバイト列が流用元と
一致することを確かめた状態のときだけにする（tests/test_c_port.py の windows の検査）。--check は書かずに比べる。
"""
import argparse
import json
import sys

import mujoco

from recovla.common import config
from recovla.common.model_summary import model_summary
from recovla.sim import control

OUT = config.ROOT / "tests" / "fixtures" / "scene_g0_reference.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    snap = model_summary(mujoco.MjModel.from_xml_path(control.SCENE_PATH))
    if a.check:
        ref = json.loads(OUT.read_text(encoding="utf-8"))
        same = ref == json.loads(json.dumps(snap))
        print("same as the reference" if same else "DIFFERENT from the reference")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snap, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} (sha256 {snap['mjb_sha256_windows']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
