"""行動の再生でのビット一致の確かめ（段階 4 束 0 運用役。目標書_段階4.md 11 節 4・第 5 節の層 (i)）。

問い: 96_s4_resume.py で再開した試行は、新しいプロセスの新しい WorldRig で回る。同じプロセスで前の試行に続けて回した場合と、
  方策を通さない部分（世界・センサの模型・計算の時間の模型・知覚）がビット一致するか。試行の間に残る状態（接触・ソルバの温め始め・
  センサの流れなど）があると一致しない。
やり方: 完全な記録の runtime_NNNN.json の行動（行動の区切りごとの a。誘発の上書きの後の値）を、方策の代わりに開ループで流す。
  方策の推論は呼ばない（GPU を使わない）が、推論と知覚を始める時刻・センサの読み・計算の時間の引き方は PolicyRuntime と同じに保つ
  （ReplayRuntime は PolicyRuntime の区切りの処理を写し、行動だけを記録の値に替える）。試行の枠は harness.loop.run_policy_trial のまま。
  経路 1: 新しいプロセス・新しい WorldRig で試行 T だけを流す。
  経路 2: 新しいプロセス・新しい WorldRig で試行 0..T-1 を先に流してから、同じ WorldRig で試行 T を流す。
  2 つの経路の真値の npz の配列（cube_pos・ee_pos・fingers・min_dist・contact など run_policy_trial の全部の欄）を、バイト列で比べる。
  参考に、元の記録（閉ループで回した npz）とも比べる（方策の行動を記録の値にしたとき、閉ループの記録が再現されるか）。

使い方（作業場所 C:\\PAI\\recovery_vla。Python は .venv\\Scripts\\python.exe。シミュレーションを動かすのは運用役だけ）:
    .venv\\Scripts\\python.exe scripts\\96_s4_replay_check.py check --dir outputs\\v2eval\\S4SMOKE60\\resume_p1 --target 2
    .venv\\Scripts\\python.exe scripts\\96_s4_replay_check.py check ... --world-per-trial   # 経路 2 も試行ごとに世界を作り直す
                                                                                      # （96_s4_resume の既定。プロセスに残る状態だけを見る）
    .venv\\Scripts\\python.exe scripts\\96_s4_replay_check.py replay --dir <条件> --order 0,1,2 --out <置き場所>   # 1 つの経路（check が子として呼ぶ）
    .venv\\Scripts\\python.exe scripts\\96_s4_replay_check.py compare --a <npz> --b <npz>                           # 2 つの npz を比べるだけ
読むもの: <条件>\\resume_spec.json（引数の控え）、trial_NNNN.json・runtime_NNNN.json・trial_NNNN.npz、scripts\\82_v2_eval.py（CFG と
  試行の並び。importlib。書き換えない）、scripts\\96_s4_resume.py（82 のハッシュの照合と制限時間の重ね。importlib）。
書くもの: outputs\\s4\\replay_check\\<実験>_<条件>\\ に path1\\、path2\\（replay_NNNN.npz・replay_NNNN.json）と report.json。
  元の記録・82・configs・src は書き換えない。
制約: 新しい試行を始める前に空きの物理メモリが 12 GB 以上あることを確かめる（足りなければ止まる、終了コード 3）。
  run の記録で、mode naive・安全フィルタなしの条件だけに対応する（RTC・安全フィルタ・task は扱わない。扱えないときは終了コード 3）。
終了コード: 0 経路 1 と 経路 2 がビット一致、1 一致しない、2 エラー、3 前提の食い違い。
"""
import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
S4 = ROOT / "outputs" / "s4"
MIN_FREE_GB = 12.0


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                            # noqa: BLE001
        pass


# ---------------------------------------------------------------- 比べる
def compare_npz(pa, pb) -> dict:
    """2 つの npz の全部の配列を、形・型・バイト列で比べる。"""
    za, zb = np.load(pa), np.load(pb)
    keys = sorted(set(za.files) | set(zb.files))
    res, all_eq = {}, True
    for k in keys:
        if k not in za.files or k not in zb.files:
            res[k] = {"equal": False, "why": "missing_in_" + ("a" if k not in za.files else "b")}
            all_eq = False
            continue
        a, b = za[k], zb[k]
        r = {"shape": [list(a.shape), list(b.shape)], "dtype": [str(a.dtype), str(b.dtype)]}
        eq = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        r["equal"] = bool(eq)
        if not eq:
            all_eq = False
            n = min(len(a), len(b)) if a.ndim and b.ndim else 0
            if n and a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
                aa, bb = a[:n].astype(float).reshape(n, -1), b[:n].astype(float).reshape(n, -1)
                diff = np.abs(np.nan_to_num(aa - bb, nan=0.0))
                bad = np.where((diff > 0).any(axis=1) | (np.isnan(aa) != np.isnan(bb)).any(axis=1))[0]
                r["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
                r["first_diff_frame"] = int(bad[0]) if len(bad) else None
                r["n_diff_frames"] = int(len(bad))
        res[k] = r
    return {"a": str(pa), "b": str(pb), "all_equal": all_eq, "n_keys": len(keys),
            "n_equal": sum(1 for r in res.values() if r.get("equal")), "keys": res}


# ---------------------------------------------------------------- 再生（1 つの経路）
def make_replay_runtime_class():
    from recovla.runtime.runner import PolicyRuntime

    class ReplayRuntime(PolicyRuntime):
        """PolicyRuntime の区切りの処理を写し、行動だけを記録の値に替える（推論は計算の時間の模型だけを通す）。"""

        def __init__(self, rec_actions: dict, *a, **kw):
            super().__init__(*a, **kw)
            self.rec = rec_actions
            self.diag = {"missing_k": 0, "held_mismatch": 0, "hold_branch_action_mismatch": 0}

        def _start_inference(self, k, t):
            sensor = self.io.sense(cameras=True)                     # 元と同じセンサの読み（関節の雑音の引き方を保つ）
            if not {"overhead", "wrist"} <= set(sensor.cameras):
                return
            i = self.i
            self.i += 1
            fut = self.io.compute("policy", lambda: None)            # 計算の時間の引き方を保つ（方策は呼ばない）
            self.pending = {"i": i, "k_obs": k, "t_obs": t, "fut": fut, "raw": lambda: None, "was_moving": self.active is not None}
            self.next_infer_k = k + self.s
            self.log_inf.append({"i": i, "k_obs": k, "t_obs": t, "t_ready": fut.t_ready, "latency_s": fut.latency,
                                 "wall_s": fut.wall_s, "rtc_delay": None,
                                 "img_t": {n: c.t_capture for n, c in sensor.cameras.items()}, "cue": None, "state": None})

        def _action_boundary(self):
            k, t = self.k, self.io.now()
            if self.perception is not None:
                self._perception(t)
                if not self.ready or self.stop_reason:
                    a = np.array([0, 0, 0, 0, 0, 0, 1.0 if self.closed else -1.0])
                    r = self.rec.get(k)
                    if r is None or not r["held"] or np.asarray(r["a"], float).tobytes() != a.tobytes():
                        self.diag["hold_branch_action_mismatch"] += 1
                    self.motion.set_velocity(np.zeros(3))
                    self.log_act.append((k, t, None, True, a))
                    return
            if self.pending is not None and self.pending["fut"].ready(t):
                p = self.pending
                p["fut"].result(t)
                lag = k - p["k_obs"]
                o0 = lag if (self.mode != "sync" and p["was_moving"]) else 0
                self.active = {"i": p["i"], "post": None, "raw": None, "k0": k, "o0": o0, "rows_done": 0}
                self.d_est = max(1, lag)
                self.log_inf[-1].update({"k_act": k, "t_act": t, "offset": o0})
                self.pending = None
            if self.pending is None and (self.active is None or (self.next_infer_k is not None and k >= self.next_infer_k)):
                self._start_inference(k, t)
            r = self.rec.get(k)
            if r is None:
                self.diag["missing_k"] += 1
                a = np.zeros(7)
                a[6] = 1.0 if self.closed else -1.0
                held, chunk = True, None
            else:
                a, held = np.asarray(r["a"], float), bool(r["held"])
                if self.active is None and not held:
                    self.diag["held_mismatch"] += 1
                chunk = None if held else (self.active["i"] if self.active is not None else r["chunk"])
            self._apply(a)
            self.log_act.append((k, t, chunk, held, a.copy()))

    return ReplayRuntime


def cmd_replay(a) -> int:
    _utf8()
    d = pathlib.Path(a.dir).resolve()
    spec = json.loads((d / "resume_spec.json").read_text(encoding="utf-8"))
    if spec.get("cmd") != "run" or spec.get("mode") != "naive" or not spec.get("no_safety") or spec.get("ablate") \
            or spec.get("diag_no_gravcomp") or spec.get("grip_gate") or spec.get("xcmd_leash") or spec.get("cart_margin") \
            or spec.get("no_limiter") or spec.get("diag_ik") not in (None, "commanded"):
        print(f"この確かめは run・naive・安全フィルタなし・既定の動きの条件だけに対応する: {spec}", file=sys.stderr)
        return 3
    res = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_rc")
    ops = res.load_ops()
    v82 = res.load_82(False)
    from recovla.common import config
    from recovla.harness.loop import run_policy_trial
    from recovla.harness.sensors import SensorSuite
    from recovla.harness.world import WorldRig
    from recovla.runtime import cue as C
    from recovla.runtime.motion import Motion
    from recovla.runtime.perception import Params, Perception
    trials = v82.trial_list(spec["trials"])
    order = [int(x) for x in a.order.split(",")]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = ops.memory_gb()
    if m["phys_free_gb"] < MIN_FREE_GB:
        print(f"空きの物理メモリ {m['phys_free_gb']} GB < {MIN_FREE_GB} GB。回さない", file=sys.stderr)
        return 3
    Replay = make_replay_runtime_class()
    world = suite = None
    summary = {"dir": str(d), "order": order, "pid": __import__("os").getpid(), "trials": []}
    try:
        for i in order:
            seed, lay, tgt = trials[i]
            meta0 = json.loads((d / f"trial_{i:04d}.json").read_text(encoding="utf-8"))
            if meta0["seed"] != seed or meta0["target"] != tgt:
                print(f"試行 {i} の記録と並びが合わない", file=sys.stderr)
                return 3
            lim = float(meta0["time_limit_s"])
            CFG = res.overlay_limits(v82.CFG, {"time_limit_s": lim})
            if world is None or a.world_per_trial:                   # 既定は経路の最初に 1 回だけ（82 と同じ使い回し）。
                if suite is not None:                                # --world-per-trial は 96_s4_resume の既定と同じ作り直し
                    suite.close()
                world = WorldRig(render=False, cfg=CFG, gravcomp=None)
                suite = SensorSuite(world.model, CFG)
            rt_log = json.loads((d / f"runtime_{i:04d}.json").read_text(encoding="utf-8"))
            rec = {int(x["k"]): x for x in rt_log["runtime"]["actions"]}
            rt_cfg, act, rtv = CFG["runtime"], CFG["actuation"], CFG["runtime_v2"]
            ex = int(spec.get("exec_interval") or rt_cfg["exec_interval"])
            holder = {}

            class StubPolicy:                                        # 方策の代わり（推論しない）。PolicyRuntime が呼ぶ口だけ
                cue = None

                def start_trial(self, seed_):
                    pass

                def reset_cue(self):
                    pass

                def set_cue_pose(self, R, t):
                    pass

            def make(io, setup, rec=rec, ex=ex, CFG=CFG, rt_cfg=rt_cfg, act=act, rtv=rtv):
                per = Perception(setup, Params.from_config(rtv["perception"]), C.Thresholds.from_dict(config.color_detect(CFG)))
                gate = dict(rtv.get("gripper_gate") or {})
                rt = Replay(rec, io, setup, StubPolicy(), perception=per, safety=None, checks=rtv["checks"], gripper_gate=gate,
                            tip_offset=float(CFG["sim"]["fingertip_offset"]), mode="naive", s=ex,
                            d_init=int(rt_cfg["delay_steps"]), rtc_horizon=int(rt_cfg["rtc_guidance_horizon"]),
                            motion=Motion(setup, limiter_enabled=True, margin=float(act["limiter_margin"]), ik_on="commanded",
                                          xcmd_leash_m=rtv.get("xcmd_leash_m"), cart_margin=rtv.get("cart_margin")))
                holder["rt"] = rt
                return rt

            w0 = time.perf_counter()
            meta, arrays, rlog = run_policy_trial(world, suite, make, lay, tgt, seed, inducer=None, cfg=CFG, time_limit_s=lim)
            wall = round(time.perf_counter() - w0, 2)
            np.savez(out / f"replay_{i:04d}.npz", **arrays)
            rt = holder["rt"]
            rep_acts = {int(kk): aa for kk, _, _, _, aa in rt.log_act}
            act_mismatch = sum(1 for kk, r in rec.items()
                               if kk not in rep_acts or np.asarray(r["a"], float).tobytes() != np.asarray(rep_acts[kk], float).tobytes())
            row = {"i": i, "seed": seed, "target": tgt, "time_limit_s": lim, "wall_s": wall,
                   "success": meta["success"], "t_success": meta["t_success"], "t_end": meta["t_end"],
                   "record": {"success": meta0["success"], "t_success": meta0.get("t_success"), "t_end": meta0.get("t_end")},
                   "n_boundaries": {"replay": len(rt.log_act), "record": len(rec)}, "action_mismatch_k": act_mismatch,
                   "diag": rt.diag, "n_inference": {"replay": len(rt.log_inf), "record": len(rt_log["runtime"]["inference"])},
                   "stop_reason": {"replay": rt.stop_reason, "record": rt_log["runtime"].get("stop_reason")},
                   "g2": meta["audit"]["g2"]}
            (out / f"replay_{i:04d}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            summary["trials"].append(row)
            print(f"[replay] {out.name} 試行 {i} seed {seed} {tgt}: success {meta['success']}（記録 {meta0['success']}）"
                  f" t_end {meta['t_end']:.3f}（記録 {meta0.get('t_end')}） 行動の食い違い {act_mismatch} diag {rt.diag} wall {wall}",
                  flush=True)
    finally:
        if suite is not None:
            suite.close()
    try:
        import psutil
        mi = psutil.Process().memory_info()
        summary["peak_wset_gb"] = round(getattr(mi, "peak_wset", mi.rss) / 1024 ** 3, 2)
    except Exception:                            # noqa: BLE001
        pass
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- 2 つの経路を回して比べる
def cmd_check(a) -> int:
    _utf8()
    d = pathlib.Path(a.dir).resolve()
    T = int(a.target)
    tag = f"{d.parent.name}_{d.name}" + ("_world_per_trial" if a.world_per_trial else "")
    base = pathlib.Path(a.out) if a.out else S4 / "replay_check" / tag
    base.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    paths = {"path1": str(T), "path2": ",".join(str(i) for i in range(T + 1))}
    runs = {}
    for name, order in paths.items():
        t0 = time.perf_counter()
        r = subprocess.run([py, str(pathlib.Path(__file__).resolve()), "replay", "--dir", str(d), "--order", order,
                            "--out", str(base / name)] + (["--world-per-trial"] if a.world_per_trial else []),
                           cwd=str(ROOT), timeout=a.timeout_s)
        runs[name] = {"order": order, "exit": r.returncode, "wall_s": round(time.perf_counter() - t0, 1)}
        if r.returncode != 0:
            print(f"[check] {name} が終了コード {r.returncode} で終わった", file=sys.stderr)
            _write(base / "report.json", {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "dir": str(d), "runs": runs, "result": "error"})
            return 2 if r.returncode != 3 else 3
    nm = f"replay_{T:04d}.npz"
    c12 = compare_npz(base / "path1" / nm, base / "path2" / nm)
    vs_rec = {"path1_vs_record": compare_npz(base / "path1" / nm, d / f"trial_{T:04d}.npz"),
              "path2_vs_record": compare_npz(base / "path2" / nm, d / f"trial_{T:04d}.npz")}
    for i in range(T):
        vs_rec[f"path2_trial{i}_vs_record"] = compare_npz(base / "path2" / f"replay_{i:04d}.npz", d / f"trial_{i:04d}.npz")
    rep = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "dir": str(d), "target_trial": T, "runs": runs,
           "world_per_trial": bool(a.world_per_trial),
           "path1_vs_path2": c12,
           "reference_vs_closed_loop_record": {k: {"all_equal": v["all_equal"], "n_equal": v["n_equal"], "n_keys": v["n_keys"],
                                                   "differing": {kk: vv for kk, vv in v["keys"].items() if not vv.get("equal")}}
                                               for k, v in vs_rec.items()},
           "summaries": {n: json.loads((base / n / "summary.json").read_text(encoding="utf-8")) for n in paths},
           "result": "bit_identical" if c12["all_equal"] else "differs"}
    _write(base / "report.json", rep)
    print(json.dumps({"path1_vs_path2_all_equal": c12["all_equal"], "n_equal": c12["n_equal"], "n_keys": c12["n_keys"],
                      "vs_record": {k: v["all_equal"] for k, v in vs_rec.items()}, "runs": runs}, ensure_ascii=False, indent=1))
    print(f"-> {base / 'report.json'}")
    return 0 if c12["all_equal"] else 1


def _write(p: pathlib.Path, obj) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def cmd_compare(a) -> int:
    _utf8()
    r = compare_npz(a.a, a.b)
    print(json.dumps({k: v for k, v in r.items() if k != "keys"} | {"differing": {k: v for k, v in r["keys"].items()
                                                                                 if not v.get("equal")}}, ensure_ascii=False, indent=1))
    return 0 if r["all_equal"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--dir", required=True, help="outputs\\v2eval\\<実験>\\<条件>")
    p.add_argument("--target", required=True, help="比べる試行の番号 T（経路 2 は 0..T-1 を先に流す）")
    p.add_argument("--out", default=None)
    p.add_argument("--timeout-s", type=float, default=3600.0)
    p.add_argument("--world-per-trial", action="store_true", help="経路 2 でも試行ごとに世界を作り直す（96_s4_resume の既定と同じ）")
    p = sub.add_parser("replay")
    p.add_argument("--dir", required=True)
    p.add_argument("--order", required=True, help="同じプロセスで流す試行の番号のカンマ区切り")
    p.add_argument("--out", required=True)
    p.add_argument("--world-per-trial", action="store_true", help="試行ごとに WorldRig・SensorSuite を作り直す")
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    a = ap.parse_args(argv)
    try:
        return {"check": cmd_check, "replay": cmd_replay, "compare": cmd_compare}[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
