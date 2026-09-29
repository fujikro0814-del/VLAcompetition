"""G1（実機境界）の静的な検査（目標書 v2、0106）。push の前の検査（check_before_push.ps1）から呼ぶ。

実行時の判断をするコード（方策の入口・上位層・完了判定・安全フィルタ・実行器・再試行）の中に、シミュレーションの
真値や制御器の内部に触れる書き方がないかを、構文木で数える。見る印:
  - import: 世界の模型・真値・評価・生成のモジュール（sim.rig・sim.scene・sim.frames・sim.contact・expert・eval・record.episode）
  - 属性: 世界の状態（qpos・qvel・xpos・xquat・cam_xpos・contact など）、真値の関数、制御器の内部（gripper_closed・integrator など）
  - 文字列の添字: 記録用の辞書の真値の欄（cube_pos・x_des・fingers など）と、設定の箱の位置（"box"）
  - 名前: rig・SimRig
今の実行系は G1 を満たしていない（0105 の調査）。今ある違反は docs/g1_known_violations.json に数で書き、
  (1) どのファイル・印でも、今の数が既知の数以下
  (2) 既知の一覧は、タグ g1-baseline の時点の一覧から増えていない（減らすだけ）
を満たせば合格とする。関門 1（10/12）では、実行系の全体を runtime の入れ物に移し、既知の一覧を空にする。

    .venv\\Scripts\\python.exe scripts\\check_g1_boundary.py [--rev HEAD]       # 検査（--rev は git の版から読む）
    .venv\\Scripts\\python.exe scripts\\check_g1_boundary.py --write-baseline   # 既知の一覧を作り直す（数が減ったときだけ使う）
"""
import argparse
import ast
import collections
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
GIT = ROOT / ".tools" / "git" / "cmd" / "git.exe"
BASELINE = "docs/g1_known_violations.json"
BASELINE_TAG = "g1-baseline"

# 実行時の判断をするコード（0105 の調査と 0106 の設計 1）。実行系を runtime の入れ物に移したら、ここを差し替える
RUNTIME_FILES = (
    "src/recovla/policy/runner.py", "src/recovla/policy/scene_policy.py", "src/recovla/eval/closed_loop.py",
    "src/recovla/data/vla_observation.py", "src/recovla/data/vla_state.py", "src/recovla/data/convert.py",
    "src/recovla/perception/color.py", "src/recovla/planner/detect.py", "src/recovla/planner/judge.py",
    "src/recovla/planner/executor.py", "src/recovla/planner/decompose.py", "src/recovla/sim/safety.py",
    "src/recovla/eval/scene_trial.py",
)
FORBIDDEN_IMPORTS = ("recovla.sim.rig", "recovla.sim.scene", "recovla.sim.frames", "recovla.sim.contact",
                     "recovla.expert", "recovla.eval", "recovla.record.episode")
FORBIDDEN_ATTRS = {"qpos", "qvel", "qacc", "xpos", "xquat", "xmat", "geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat",
                   "contact", "ctrl", "truth", "cube_ids", "cube_vadr", "cubes_in_box", "gripper_closed", "integrator",
                   "controller", "scratch", "forward_scratch", "finger_qadr", "finger_vadr", "meter", "desired_pos",
                   "x_cmd", "mj_geomDistance"}
FORBIDDEN_KEYS = {"cube_pos", "cube_quat", "cube_linvel", "cube_in_box", "phase", "contact_robot", "contact_cube_cube",
                  "min_dist", "x_des", "tracker_target", "gripper_cmd", "gripper_closed", "fingertip", "fingers",
                  "safety_active", "target", "box"}
FORBIDDEN_NAMES = {"rig", "SimRig"}


def read(path: str, rev):
    if rev is None:
        p = ROOT / path
        return p.read_text(encoding="utf-8") if p.is_file() else None
    out = subprocess.run([str(GIT), "-C", str(ROOT), "show", f"{rev}:{path}"], capture_output=True)
    return out.stdout.decode("utf-8") if out.returncode == 0 else None


def scan(src: str) -> collections.Counter:
    c = collections.Counter()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(FORBIDDEN_IMPORTS):
                    c[f"import:{a.name}"] += 1
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for m in mods:
                if m.startswith(FORBIDDEN_IMPORTS):
                    c[f"import:{m}"] += 1
                    break
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            c[f"attr:{node.attr}"] += 1
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value in FORBIDDEN_KEYS:
            c[f"key:{node.slice.value}"] += 1
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            c[f"name:{node.id}"] += 1
    return c


def current(rev) -> dict:
    out = {}
    for f in RUNTIME_FILES:
        src = read(f, rev)
        if src is None:
            continue
        hits = scan(src)
        if hits:
            out[f] = dict(sorted(hits.items()))
    return out


def load_baseline(rev) -> dict:
    src = read(BASELINE, rev)
    return {} if src is None else json.loads(src)["violations"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rev", default=None, help="git の版から読む（既定は作業ツリー）")
    ap.add_argument("--write-baseline", action="store_true")
    a = ap.parse_args(argv)
    now = current(a.rev)
    if a.write_baseline:
        doc = {"note": "G1 の既知の違反（scripts/check_g1_boundary.py が数える）。減らすだけ。関門 1 で空にする（0106）",
               "violations": now}
        (ROOT / BASELINE).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"書いた: {BASELINE}（{sum(sum(v.values()) for v in now.values())} 件）")
        return 0
    base = load_baseline(a.rev)
    problems = []
    for f, hits in now.items():
        for k, n in hits.items():
            known = base.get(f, {}).get(k, 0)
            if n > known:
                problems.append(f"{f}: {k} が {n} 件（既知 {known} 件）")
    tag_ok = subprocess.run([str(GIT), "-C", str(ROOT), "rev-parse", "--verify", "--quiet", f"{BASELINE_TAG}^{{commit}}"],
                            capture_output=True).returncode == 0
    if tag_ok:
        first = load_baseline(BASELINE_TAG)
        for f, hits in base.items():
            for k, n in hits.items():
                if n > first.get(f, {}).get(k, 0):
                    problems.append(f"既知の一覧が {BASELINE_TAG} から増えた: {f}: {k}")
    else:
        problems.append(f"タグ {BASELINE_TAG} がない")
    total = sum(sum(v.values()) for v in now.values())
    if problems:
        print(f"G1 の境界の検査: 不合格（{len(problems)} 件）")
        for p in problems:
            print("  " + p)
        return 1
    print(f"G1 の境界の検査: 合格（新しい違反 0、既知の違反 {total} 件が残る。関門 1 で 0 にする）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
