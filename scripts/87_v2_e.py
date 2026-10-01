"""目標書 v2 の段階ごとの最終評価（E の一覧、0107 の 10、0108 で承認）。旧版（50_e_eval.py）と同じ組・主な検定・指標で、
実行系は v2（scripts/82_v2_eval.py）。E9（G1〜G3 の監査）と E10（知覚の精度）を足す。

    .venv\\Scripts\\python.exe scripts\\87_v2_e.py plan --stage s1 [--sets A,B,...]     # 実行の順番（outputs/v2eval/<実験>_queue.json）
    .venv\\Scripts\\python.exe scripts\\87_v2_e.py run --stage s1 [--parallel 3]        # 順番に 82_v2_eval.py を回す（済んだ条件は飛ばす）
    .venv\\Scripts\\python.exe scripts\\87_v2_e.py report --stage s1                    # E ごとの表と検定 → outputs/results/v2_e_report_<stage>.json
    .venv\\Scripts\\python.exe scripts\\87_v2_e.py safety-decide --stage s1             # 安全フィルタを主系に残すかの判定（検証用 199800〜）

組（0107 の 10-1。段階 1 の最終モデルは R2、段階 2 は R1）:
  段階 1: A R1・本線  B N1・本線  C R1・同期（止まって推論）  D N1・同期  E R1・RTC（見積もった遅延）  F R2・本線  G R1+・本線
          H R2・安全フィルタ切
  段階 2: A R1v2・本線  B N1v2・本線  C R1v2・同期  D N1v2・同期  E R1v2・RTC  H R1v2・安全フィルタ切
  本線 = naive・s=10・知覚の安全フィルタあり（主系から外すと判定したときは --main-safety off で、本線をフィルタ切にし、H は作らない）
主な検定（旧版と同じ）: E1 最終モデルの失敗注入なしの成功率、E2 最終モデルの P1 の復帰の率、E3 R1 対 N1 の P1 の復帰（本線・同期、Holm）、
  E4 naive 対 RTC の P1、E5 安全フィルタあり対なしの失敗注入なしの接触、E8 R2 対 R1+ の P1（段階 1 だけ）
新しい評価: E9 全試行の G1（到達検査）・G2（世界の停止・早すぎる使用）・G3（口を通る指令の上限違反）の違反の数（基準 0）、
  E10 知覚の精度（机上の立方体の位置の誤差の中央値・p95、箱の中かの判定と真値の一致率）
種: 段階 1 は 120000〜（自然 120000〜120032 の 33 配置×3 色、P1 121000〜・P2 122000〜・P3 123000〜 各 50）、段階 2 は 130000〜
"""
import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

from recovla.common import config

CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"])
RES = OUT / "results"
PARTS = ("nat", "P1", "P2", "P3")
STAGES = {
    "s1": {"experiment": "V2S1", "base": 120000, "final": "F",
           "sets": {"A": ("R1", "naive", True), "B": ("N1", "naive", True), "C": ("R1", "sync", True),
                    "D": ("N1", "sync", True), "E": ("R1", "rtc", True), "F": ("R2", "naive", True),
                    "G": ("R1plus", "naive", True), "H": ("R2", "naive", False)}},
    "s2": {"experiment": "V2S2", "base": 130000, "final": "A",
           "sets": {"A": ("R1v2", "naive", True), "B": ("N1v2", "naive", True), "C": ("R1v2", "sync", True),
                    "D": ("N1v2", "sync", True), "E": ("R1v2", "rtc", True), "H": ("R1v2", "naive", False)}},
    "dev": {"experiment": "V2DEVE", "base": 59800, "final": "F",
            "sets": {"F": ("R2", "naive", True), "H": ("R2", "naive", False)}},
}


def trials_of(stage: str, part: str) -> str:
    b = STAGES[stage]["base"]
    if stage == "dev":
        return {"nat": f"natural:{b}:2", "P1": f"induced:{b + 100}:3", "P2": f"induced:{b + 200}:3", "P3": f"induced:{b + 300}:3"}[part]
    return {"nat": f"natural:{b}:33", "P1": f"induced:{b + 1000}:50", "P2": f"induced:{b + 2000}:50",
            "P3": f"induced:{b + 3000}:50"}[part]


def _e50():
    spec = importlib.util.spec_from_file_location("e50", config.ROOT / "scripts" / "50_e_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cmd_plan(a) -> list:
    st = STAGES[a.stage]
    main_safety = a.main_safety == "on"
    q = []
    for s in (a.sets.split(",") if a.sets else st["sets"]):
        model, mode, safety = st["sets"][s]
        if s == "H" and not main_safety:
            continue                                    # 本線がフィルタ切なら H は本線と同じ
        safety = safety and main_safety
        for part in PARTS:
            args = ["run", "--experiment", st["experiment"], "--condition", f"{s}_{part}", "--model", model,
                    "--mode", mode, "--trials", trials_of(a.stage, part)]
            if part != "nat":
                args += ["--induce", part]
            if not safety:
                args += ["--no-safety"]
            q.append({"name": f"{s}_{part}", "args": args})
    path = OUT / "v2eval" / f"{st['experiment']}_queue.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": a.stage, "main_safety": a.main_safety, "queue": q}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    for x in q:
        print(x["name"], " ".join(x["args"]))
    return q


def cmd_run(a) -> None:
    st = STAGES[a.stage]
    q = json.loads((OUT / "v2eval" / f"{st['experiment']}_queue.json").read_text(encoding="utf-8"))["queue"]
    logs = OUT / "v2eval" / f"{st['experiment']}_logs"
    logs.mkdir(parents=True, exist_ok=True)
    todo = [x for x in q if not (OUT / "v2eval" / st["experiment"] / x["name"] / "run.json").is_file()]
    running = []
    while todo or running:
        while todo and len(running) < a.parallel:
            x = todo.pop(0)
            d = OUT / "v2eval" / st["experiment"] / x["name"]
            if d.exists() and any(d.glob("trial_*.json")):          # 途中で落ちた条件は消してやり直す
                import shutil
                shutil.rmtree(d)
            f = open(logs / f"{x['name']}.log", "w", encoding="utf-8")
            p = subprocess.Popen([sys.executable, str(config.ROOT / "scripts" / "82_v2_eval.py"), *x["args"]],
                                 stdout=f, stderr=subprocess.STDOUT, cwd=config.ROOT)
            running.append((x, p, f))
            print(time.strftime("%H:%M:%S"), "start", x["name"], flush=True)
        time.sleep(10)
        for item in list(running):
            x, p, f = item
            if p.poll() is not None:
                f.close()
                running.remove(item)
                print(time.strftime("%H:%M:%S"), "done", x["name"], "exit", p.returncode, flush=True)


def _audits(d: pathlib.Path) -> dict:
    out = {"trials": 0, "g1": 0, "g2_stops": 0, "g2_early": 0, "g3": 0, "g3_by_item": {}}
    for p in sorted(d.glob("trial_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        au = m["audit"]
        out["trials"] += 1
        out["g1"] += int(au["g1"]["violations"])
        out["g2_stops"] += int(au["g2"]["world_stops"])
        out["g2_early"] += int(au["g2"]["early_use"])
        out["g3"] += int(au["g3"]["total_violations"])
        for k, v in au["g3"]["violations"].items():
            out["g3_by_item"][k] = out["g3_by_item"].get(k, 0) + int(v)
    return out


def _perception_accuracy(d: pathlib.Path) -> dict:
    """E10: 実行系の知覚の記録（runtime_NNNN.json の perception）と、真値のこま（trial_NNNN.npz）を時刻で合わせる。"""
    from recovla.common.seeds import COLORS
    from recovla.sim import frames
    errs, agree = [], []
    box = np.asarray(CFG["scene"]["box"]["pos"], float)
    for p in sorted(d.glob("trial_*.json")):
        i = p.stem.split("_")[1]
        rp = d / f"runtime_{i}.json"
        if not rp.is_file():
            continue
        z = np.load(p.with_suffix(".npz"))
        t = z["sim_time"]
        for e in json.loads(rp.read_text(encoding="utf-8"))["runtime"].get("perception", []):
            k = int(np.argmin(np.abs(t - e["t_obs"])))
            for c, v in e["cubes"].items():
                tp = z["cube_pos"][k, COLORS.index(c)]
                on_table = tp[2] < 0.03 and not frames.in_box(tp, box)
                if on_table and v["status"] in ("seen", "held"):
                    errs.append(float(np.linalg.norm(np.asarray(v["pos"]) - tp)))
                if v["status"] != "in_hand":
                    agree.append(bool(v["in_box"]) == bool(frames.in_box(tp, box)))
    return {"n": len(errs), "median_m": float(np.median(errs)) if errs else None,
            "p95_m": float(np.percentile(errs, 95)) if errs else None,
            "in_box_agreement": float(np.mean(agree)) if agree else None, "n_in_box": len(agree)}


def cmd_report(a) -> None:
    from recovla.eval import report as R
    e50 = _e50()
    st = STAGES[a.stage]
    base = OUT / "v2eval" / st["experiment"]
    have = [s for s in st["sets"] if (base / f"{s}_nat").is_dir()]
    from recovla.eval import gate
    gate.require([base / f"{s}_{p}" for s in have for p in PARTS if (base / f"{s}_{p}").is_dir()],
                 f"最終評価の集計（{st['experiment']}）")

    def rows(s, part):
        d = base / f"{s}_{part}"
        return {(r["seed"], r["target"]): r for r in R.collect([d])} if d.is_dir() else {}
    S = {s: {p: rows(s, p) for p in PARTS} for s in have}
    fin = st["final"]
    res = {"stage": a.stage, "sets": {s: {p: e50.summary(S[s][p]) for p in PARTS} for s in have},
           "final_set": fin, "final_model": st["sets"][fin][0], "primary": {}, "secondary": {}, "new": {}}
    if fin in S:
        res["primary"]["E1"] = {k: res["sets"][fin]["nat"][k] for k in ("n", "successes", "success_wilson")}
        res["primary"]["E2"] = {k: res["sets"][fin]["P1"][k] for k in ("established", "recovered", "recovery_rate", "recovery_wilson")}
    pairs = {"E3_main": ("A", "B"), "E3_sync": ("C", "D"), "E4_naive_vs_rtc": ("A", "E"), "E4_sync_vs_naive": ("C", "A"),
             "E8": ("F", "G"), "E5": (fin, "H")}
    for name, (x, y) in pairs.items():
        if x in S and y in S:
            res["secondary"][name] = {"x": x, "y": y, **e50.compare(S[x], S[y])}
    sec = res["secondary"]
    if "E3_main" in sec and "E3_sync" in sec:
        ps = [sec["E3_main"]["P1_recovery"]["mcnemar_exact_p"], sec["E3_sync"]["P1_recovery"]["mcnemar_exact_p"]]
        adj = e50._holm(ps)
        res["primary"]["E3"] = {"main_A_vs_B": {**sec["E3_main"]["P1_recovery"], "holm_p": adj[0]},
                                "sync_C_vs_D": {**sec["E3_sync"]["P1_recovery"], "holm_p": adj[1]}}
    for e, key in (("E4", "E4_naive_vs_rtc"), ("E8", "E8")):
        if key in sec:
            res["primary"][e] = sec[key]["P1_recovery"]
    if "E5" in sec:
        res["primary"]["E5"] = sec["E5"]["nat_contact"]
    aud = {f"{s}_{p}": _audits(base / f"{s}_{p}") for s in have for p in PARTS if (base / f"{s}_{p}").is_dir()}
    tot = {k: sum(v[k] for v in aud.values()) for k in ("trials", "g1", "g2_stops", "g2_early", "g3")}
    res["new"]["E9"] = {"criterion": "全試行で違反 0", "total": tot, "met": tot["g1"] == 0 and tot["g2_stops"] == 0
                        and tot["g2_early"] == 0 and tot["g3"] == 0, "by_condition": aud}
    res["new"]["E10"] = {s: _perception_accuracy(base / f"{s}_nat") for s in have}
    res["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"v2_e_report_{a.stage}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"primary": res["primary"], "E9": res["new"]["E9"]["total"], "E10": res["new"]["E10"]},
                     ensure_ascii=False, indent=1, default=float))


def cmd_safety_decide(a) -> None:
    """安全フィルタを主系に残すかの判定（0107 の 5、回す前に固めた決まり）。検証用 199800〜199832 の失敗注入なし 99 試行を、
    最終モデル（段階 1 は R2）でフィルタあり・なしの同じシードの対で回し、あり の成功が なし より 3 ポイント以上低ければ主系から外す。
    接触の減り方は記録だけ。結果は outputs/results/v2_safety_main_<stage>.json（以後、この判定に使った決まりは直さない＝0113 の 4）。"""
    from recovla.eval import report as R
    e50 = _e50()
    model = STAGES[a.stage]["sets"][STAGES[a.stage]["final"]][0]
    exp = a.experiment or f"V2SF_{a.stage}"
    for cond, extra in (("on", []), ("off", ["--no-safety"])):
        d = OUT / "v2eval" / exp / cond
        if not (d / "run.json").is_file():
            subprocess.run([sys.executable, str(config.ROOT / "scripts" / "82_v2_eval.py"), "run", "--experiment", exp,
                            "--condition", cond, "--model", model, "--trials", a.trials, *extra], cwd=config.ROOT, check=True)
    from recovla.eval import gate
    gsum = gate.require([OUT / "v2eval" / exp / c for c in ("on", "off")], f"安全フィルタの判定（{exp}）")
    X = {(r["seed"], r["target"]): r for r in R.collect([OUT / "v2eval" / exp / "on"])}
    Y = {(r["seed"], r["target"]): r for r in R.collect([OUT / "v2eval" / exp / "off"])}
    succ = e50.paired_binary(X, Y, e50.SUCC)
    contact = e50.paired_binary(X, Y, e50.CONTACT)
    drop = (succ["y_rate"] - succ["x_rate"]) * 100
    res = {"stage": a.stage, "model": model, "trials": a.trials, "rule": "あり の成功が なし より 3 ポイント以上低ければ主系から外す（0107 の 5）",
           "success": succ, "contact": contact, "drop_points": drop, "keep_in_main": drop < 3.0,
           "experiment": exp, "g_audit": gsum, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"v2_safety_main_{a.stage}{'' if exp == f'V2SF_{a.stage}' else '_' + exp}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--sets", default=None)
    p.add_argument("--main-safety", default="on", choices=("on", "off"))
    p = sub.add_parser("run")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--parallel", type=int, default=3)
    p = sub.add_parser("report")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p = sub.add_parser("safety-decide")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--trials", default="natural:199800:33")
    p.add_argument("--experiment", default=None, help="出力の実験名（既定 V2SF_<段階>。回し直しで前の判定を上書きしない）")
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "run": cmd_run, "report": cmd_report, "safety-decide": cmd_safety_decide}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
