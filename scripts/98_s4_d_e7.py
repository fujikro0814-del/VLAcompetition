"""段階 4 束 1 の D-E7（3 個の連続タスクの戻し先の腕。担当 B。docs/目標書_段階4.md 8-2 の T・8-5・8-6）。

使い方（作業場所 C:\\PAI\\recovery_vla。run の引数は 96_s4_resume.py task と同じものを後ろに付ける）:
    .venv\\Scripts\\python.exe scripts\\98_s4_d_e7.py defs                       # 定義を outputs\\s4\\d_e7\\definitions.json に書く（回す前に掲示）
    .venv\\Scripts\\python.exe scripts\\98_s4_d_e7.py run --arm EH --experiment S4DE7 --condition EH --model R1v3 --trials 190300:40
    .venv\\Scripts\\python.exe scripts\\98_s4_d_e7.py run --arm E0 ... --condition E0_run1 --trials 190300:40 --dry-run
    .venv\\Scripts\\python.exe scripts\\98_s4_d_e7.py summary --experiment S4DE7 [--conditions E0_run1 E0_run2 EH ES EO] [--out ...]
  腕（--arm）: E0（今の実行器。条件名 E0_run1・E0_run2 で同じ 40 種を 2 回）、EH（home の関節角へ戻す）、ES（待機位置の始めへ戻す）、
    EO（緑→赤→青の順の指示文）、EX（条件つき。8-4 の行 1・行 3 の矛盾の枝のときだけ。手順の切り替えで実行系を作り直す）。
  本番（帯 190300〜190339、40 本 x 5 腕。腕は同じ時期に種の塊ごとに交互に回す＝目標書 第 4 節）は運用役が回す。
  smoke は学習用の帯（44400〜44799）だけ。どちらの帯にも入らない種は回さない（終了コード 3）。
読むもの: scripts\\96_s4_resume.py（importlib で読み込み、Engine・SPEC_KEYS・run_json_text を差し替えて cmd_main で回す。書き換えない）、
  configs\\s4_gates.json（帯）、src\\recovla\\diag\\e7.py、summary は outputs\\v2eval\\<実験>\\<条件>\\ の記録。
書くもの: 96_s4_resume.py task と同じ（outputs\\v2eval\\<実験>\\<条件>\\ の run_NNNN.json・.npz・run_NNNN_runtime.json、run.json、G_AUDIT.json、
  progress.json、resume_spec.json、resume_log.json）。試行の json に "diag"（腕・戻し先・API の回し直し・96 の SHA-256）、
  meta["returns"] に kind "goal"（EH・ES）・"rebuild"（EX）の行、run.json に "diag" を足す。defs・summary は outputs\\s4\\d_e7\\ に書く。
96_s4_resume.py と同じく: 試行ごとに世界を作り直す、環境（ドライバ・torch・CUDA・OS）が記録と違えば止める、同じコマンドで続きから回る。
  1 手順の試み 30 s・やり直し 1 回・全体 200 s・計画役 s4（Haiku 5.5）は 96 の既定のまま（変えない）。

結果を見る前に決めてある項目（defs が書く definitions.json。掲示の後は変えない）:
  - 腕の定義・戻し先の関節角・戻し方の数値・EO の指示文（src\\recovla\\diag\\e7.py の冒頭）。
  - 実行のしかたは段階 3 の E7（V3S3\\E7_R1v3）と同じ: 行動の区切り 6 行（--exec-interval 6）、安全フィルタなし（--no-safety。段階 3 の記録の
    runtime.safety が null）。このスクリプトが両方を既定で入れる（違う値を渡すと止める）。
  - EO の計画役の API が失敗した試行（anthropic の例外）は、同じ種で 1 回だけその場で回し直し、試行の diag.api_retry に残す。2 回目も
    失敗したら止める（終了コード 2。続きから回すと同じ種で回る）。
  - 指標（summary）: 2 番目の手順の 1 回目の試みの最初の閉じの持ち上がり（分母は 2 番目の手順が始まった試行）、+y のずれ（分母は最初の
    閉じが起きた試行）、3 個とも（真値）。gate_T_inputs は E0 の基準・d_E0・要求の幅・K.e7 の食い違いと phi を並べるが、判定はしない。
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTD = ROOT / "outputs" / "s4" / "d_e7"
GATES = ROOT / "configs" / "s4_gates.json"
ALLOC = "D_E7"
DEFAULT_CONDITIONS = ("E0_run1", "E0_run2", "EH", "ES", "EO")


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def band_check(seeds, alloc_id: str) -> str:
    """種が s4_gates.json の割り当て alloc_id の帯か、smoke・データの帯（44400〜44799）に全部入っていれば ""、そうでなければ理由。"""
    g = json.loads(GATES.read_text(encoding="utf-8"))
    al = next(x for x in g["bands"]["allocations"] if x["id"] == alloc_id)["range"]
    sm = next(x for x in g["bands"]["classes"] if x["id"] == "smoke_data")["range"]
    seeds = list(seeds)
    if all(al[0] <= s <= al[1] for s in seeds) or all(sm[0] <= s <= sm[1] for s in seeds):
        return ""
    return f"種 {seeds[0]}〜{seeds[-1]} が帯 {alloc_id} {al} にも smoke の帯 {sm} にも収まらない"


def _is_api_error(e: BaseException) -> bool:
    return any(c.__module__.split(".")[0] == "anthropic" for c in type(e).__mro__)


def make_engine(r96, arm: str, info: dict):
    from recovla.diag import e7 as E7
    from recovla.runtime.perception import Perception

    class DiagE7Engine(r96.Engine):
        """96 の Engine の task を、腕に合わせて実行器を差し替えた版（E0・EO は差し替えない）。"""

        def init_task(self):
            super().init_task()
            orig = self.make_task

            def make(io, setup):
                return E7.install(orig(io, setup), arm, Perception)
            self.make_task = make

        def run_one_task(self, i, seed):
            retry = None
            try:
                meta, arrays, rlog = super().run_one_task(i, seed)
            except Exception as e:                           # noqa: BLE001
                if not _is_api_error(e):
                    raise
                retry = {"n": 1, "error": f"{type(e).__name__}: {e}"[:500], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                print(f"[d_e7] 試行 {i} 種 {seed}: 計画役の API が失敗したので同じ種で 1 回だけ回し直す（{retry['error']}）", flush=True)
                meta, arrays, rlog = super().run_one_task(i, seed)
            meta["diag"] = dict(E7.arm_record(arm), api_retry=retry, **info)
            return meta, arrays, rlog
    return DiagE7Engine


def cmd_run(argv) -> int:
    from recovla.diag import e7 as E7
    ap = argparse.ArgumentParser(prog="98_s4_d_e7.py run", add_help=False)
    ap.add_argument("--arm", required=True, choices=sorted(E7.ARMS))
    mine, rest = ap.parse_known_args(argv)
    r96 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_dE7")
    a = r96.build_parser().parse_args(["task"] + rest)
    arm = mine.arm
    want_text = E7.ARMS[arm]["text"]
    if a.text not in (E7.DEFAULT_TEXT, want_text):
        print(f"--text は腕 {arm} では「{want_text}」に決めてある（渡された「{a.text}」は使わない）", file=sys.stderr)
        return 3
    a.text = want_text
    if a.exec_interval not in (None, 6):
        print("D-E7 は段階 3 の E7 と同じ --exec-interval 6・--no-safety で回す（ほかの値は渡さない）", file=sys.stderr)
        return 3
    a.exec_interval, a.no_safety = 6, True
    base, n = (int(x) for x in a.trials.split(":"))
    why = band_check(range(base, base + n), ALLOC)
    if why:
        print(why, file=sys.stderr)
        return 3
    info = {"script": "98_s4_d_e7.py", "r96_sha256": sha256_file(ROOT / "scripts" / "96_s4_resume.py"),
            "e7_py_sha256": sha256_file(ROOT / "src" / "recovla" / "diag" / "e7.py"), "condition": a.condition}
    a.diag_arm = arm
    a.diag_e7_sha256 = info["e7_py_sha256"]
    r96.SPEC_KEYS = tuple(r96.SPEC_KEYS) + ("diag_arm", "diag_e7_sha256")
    r96.Engine = make_engine(r96, arm, info)
    orig_text = r96.run_json_text

    def run_json_text(a_, kind, rows, *x, **kw):
        d = json.loads(orig_text(a_, kind, rows, *x, **kw))
        d["diag"] = dict(E7.arm_record(arm), **info)
        return json.dumps(d, ensure_ascii=False, indent=1)
    r96.run_json_text = run_json_text
    print(f"[d_e7] 腕 {arm}: {E7.ARMS[arm]['ja']}（指示文「{a.text}」、--exec-interval 6・--no-safety）", flush=True)
    ops = r96.load_ops()
    v82 = r96.load_82(a.allow_82_change)
    try:
        return r96.cmd_main(a, v82, ops)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


def cmd_defs(a) -> int:
    """定義（回す前に掲示するもの）を書く。読むだけの計算（学習データの平均の照合、戻し先の手先の高さ、EO のキャッシュの有無）。"""
    import numpy as np
    from recovla.common import config
    from recovla.diag import e7 as E7
    from recovla.harness.setup import nominal_setup
    from recovla.planner import decompose_s4 as D4
    from recovla.runtime.motion import Motion
    means = E7.train_pose_means(ROOT / "outputs" / "manifests" / "R1v3_20261005-133737.json", ROOT)
    cfg = config.load("sensor_v1")
    mo = Motion(nominal_setup(cfg))
    fk = {k: mo.hand_pose(np.asarray(q, float))[0].tolist() for k, q in
          (("home", E7.HOME_Q), ("standby_start", E7.STANDBY_START_Q), ("standby_end", E7.STANDBY_END_Q))}
    check = {"standby_start_vs_train_mean_max_abs": float(np.max(np.abs(np.asarray(E7.STANDBY_START_Q) - means["start_retreat"]["joints"]))),
             "standby_end_vs_train_mean_max_abs": float(np.max(np.abs(np.asarray(E7.STANDBY_END_Q) - means["end"]["joints"]))),
             "home_vs_train_mean_max_abs": float(np.max(np.abs(np.asarray(E7.HOME_Q) - means["start_home"]["joints"])))}
    msg = D4.user_message(E7.EO_TEXT, ["blue", "green", "red"], [])
    key = D4.cache_key(D4.DEFAULT_MODEL, msg)
    eo_cache = (D4.CACHE / f"{key}.json")
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "by": "担当 B（98_s4_d_e7.py defs）",
           "arms": {k: E7.arm_record(k) for k in E7.ARMS}, "goal_params": E7.GOAL_PARAMS,
           "poses": {"HOME_Q": E7.HOME_Q, "STANDBY_START_Q": E7.STANDBY_START_Q, "STANDBY_END_Q": E7.STANDBY_END_Q,
                     "source": E7.POSE_SOURCE, "train_means": means, "constant_vs_mean": check, "fk_hand_xyz": fk},
           "apply_at": "2 番目以降の手順の 1 回目の試みの前（前の手順の完了の判定の後）。やり直しの前の戻す動きは E0 と同じ",
           "run_settings": {"exec_interval": 6, "no_safety": True, "step_timeout_s": 30, "retry": 1, "task_time_limit_s": 200,
                            "planner": "s4（decompose_s4、claude-haiku-5-5）", "world_per_trial": True, "model": "R1v3",
                            "basis": "段階 3 の E7（V3S3\\E7_R1v3）: 推論の間隔 6 行（runtime.inference の k_obs の差）、runtime.safety が null"},
           "eo": {"text": E7.EO_TEXT, "expected_order": E7.EO_ORDER, "cache_key_all_three_on_table": key,
                  "cache_present": eo_cache.is_file(),
                  "note": "検出した机上の色が 3 個（並べ替えて blue, green, red）、箱が空のときの鍵。無ければ本番の最初の EO の試行で 1 回だけ呼ぶ"},
           "metrics": {"first_close_lift": "2 番目の手順の 1 回目の試みの最初の閉じ（gripper_closed の 0→1）から次に開くまでに目標が CUBE_REST_Z + 0.02 m を超えた。"
                                           "分母は 2 番目の手順が始まった試行",
                       "plus_y_shift": "その閉じのこまでの（指先 − 目標）の y > +1 cm。分母は最初の閉じが起きた試行",
                       "all_three_true": "meta.all_three_in_box（真値。終わりに 3 個とも箱の中）",
                       "same_as": "docs\\local\\strategy_20261008\\work\\upper_skeptic\\e7_first_grasp.py（V3S3 の E7 で 2 番目の 1 回目 5/20・+y 16/19 を再現。"
                                  "tests\\test_s4_diag_B.py）"},
           "bands": {"production": "190300〜190339（D_E7）", "conditions": list(DEFAULT_CONDITIONS),
                     "interleave": "同じ時期に種の塊ごとに交互（目標書 第 4 節）。並びは運用役が決める"},
           "files_sha256": {p: sha256_file(ROOT / p) for p in ("src/recovla/diag/e7.py", "scripts/98_s4_d_e7.py",
                                                                "scripts/96_s4_resume.py", "configs/s4_gates.json")}}
    OUTD.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(a.out) if a.out else OUTD / "definitions.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("goal_params", "eo")}, ensure_ascii=False, indent=1))
    print(json.dumps(out["poses"]["constant_vs_mean"], indent=1), json.dumps(fk, indent=1))
    print(f"[d_e7] 書いた: {p}")
    return 0


def cmd_summary(a) -> int:
    from recovla.diag import e7 as E7
    base = ROOT / "outputs" / "v2eval" / a.experiment
    summ = {}
    for c in a.conditions:
        if (base / c).is_dir() and any((base / c).glob("run_[0-9][0-9][0-9][0-9].json")):
            summ[c] = E7.summarize_condition(base / c)
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": a.experiment, "gate_T_inputs": E7.gate_T_inputs(summ),
           "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "trials" or a.with_trials} for k, v in summ.items()}}
    text = json.dumps(out, ensure_ascii=False, indent=1, default=str)
    OUTD.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(a.out) if a.out else OUTD / f"summary_{a.experiment}.json"
    p.write_text(text, encoding="utf-8")
    print(json.dumps(out["gate_T_inputs"], ensure_ascii=False, indent=1))
    print(f"[d_e7] 書いた: {p}")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        return cmd_run(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="腕を 1 つ回す（--arm と 96_s4_resume.py task の引数）")
    p = sub.add_parser("defs", help="定義を書く（回す前に掲示）")
    p.add_argument("--out", default=None)
    p = sub.add_parser("summary", help="記録から関門 T の材料を出す（判定はしない）")
    p.add_argument("--experiment", required=True)
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    p.add_argument("--out", default=None)
    p.add_argument("--with-trials", action="store_true", help="試行ごとの行も書く")
    a = ap.parse_args(argv)
    return {"defs": cmd_defs, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
