"""段階 4 束 1 の D-復帰（担当 C）: R1v3・N1v3 × 4 通り（落下そのまま・落下で手を止める版・置き損ね・把持失敗）を同じ種で回し、
関門 C の材料（R の 30 s の復帰、R−N、成立した対の数）を出す。判定はしない（二重集計役が configs/s4_gates.json の C で当てはめる）。

使い方（作業場所 C:\\PAI\\recovery_vla）:
    .venv\\Scripts\\python.exe scripts\\98_s4_d_recovery.py plan                       # 条件・種・試行数・見込みのプロセス時間（読むだけ）
    .venv\\Scripts\\python.exe scripts\\98_s4_d_recovery.py run --dry-run              # 本番の帯（190500〜190549）で何を回すかだけ見る
    .venv\\Scripts\\python.exe scripts\\98_s4_d_recovery.py run                        # 本番（運用役だけ。切り離すなら監視役を付ける）
    .venv\\Scripts\\python.exe scripts\\98_s4_d_recovery.py run --variants fall_with_hold --models R1v3,N1v3   # 3 本並行の 1 本分など
    .venv\\Scripts\\python.exe scripts\\98_s4_d_recovery.py run --experiment S4SMOKE_C --trials induced:44470:3 --variants fall_with_hold --models R1v3
    .venv\\Scripts\\python.exe scripts\\98_s4_d_recovery.py score [--experiment S4DREC]  # 関門 C の材料・30/45/60 s の採点・落下の時刻（読むだけ）
    .venv\\Scripts\\python.exe scripts\\98_s4_d_recovery.py check-s3                   # 段階 3 の記録で集計の関数を確かめる（読むだけ）
  監視（切り離したとき）: 96_s4_ops.py wait --progress outputs\\s4\\d_recovery\\progress_<実験>_<腕>.json（この包みの全体の進捗）

読むもの: scripts\\96_s4_resume.py（importlib で読み込む。書き換えない。cmd_main・Engine・score_condition をそのまま使い、
  Engine.run_one・make_spec・run_json_text だけを、読み込んだ写しの上で包む）、scripts\\82_v2_eval.py（96 経由）、
  src\\recovla\\diag\\recovery.py（誘発の版と集計）、configs\\s4_gates.json（帯の割り当て D_recovery・smoke_data、
  budget_notes の 60 s の倍率）、outputs\\v2eval\\V3S3\\*_P1〜P3\\run.json（見込みの計算）、V3S3\\A_P1・B_P1・A_P2・B_P2（check-s3）。
書くもの: outputs\\v2eval\\<実験>\\<モデル>_<通り>\\ に 96_s4_resume.py run と同じ記録（trial_NNNN.json/.npz・runtime_NNNN.json・run.json・
  G_AUDIT.json・progress.json・resume_spec.json・resume_log.json）。試行の json に "diag"（診断の条件）を、meta["induce"] に
  "variant"・"version" を、手を止める版では induce.info["hold"]（着地の時刻・放した時刻・手を動かした塊の観測の時刻）を足す。
  run.json に "diag" を足す。outputs\\s4\\d_recovery\\ に progress_*.json（全体の進捗）・score_<実験>.json・check_s3.json。

条件（結果を見る前に決めた。10/08。docs/目標書_段階4.md 第 8 節の関門 C、s4_gates の gates.C・bands D_recovery）:
  - 帯 190500〜190549（50 種）を 8 条件で同じ種に使う（--trials induced:190500:50 = 41_results.trial_list の induced、配置は
    scene.sample_layout(種)、目標は choose_targets。段階 3 の誘発の試行と同じ引き方）。50 × 8 = 400 試行。
  - 条件の名前は <モデル>_<通り>（R1v3_fall_as_is、R1v3_fall_with_hold、R1v3_misplace、R1v3_grasp_failure と N1v3 の 4 つ）。
  - 実行の設定は段階 3 の V3S3 の誘発の条件と同じ: naive、塊の実行 6 行、安全フィルタなし、制限時間 60 s（96 の既定。単発の試行）、
    試行ごとに世界を作り直す（96 の既定）。
  - 採点は 30 s が主、60 s が副、45 s は記述だけ。誘発の試行の分母は L 秒以内に成立した試行（induce.t_established <= L）。
  - 比べる組（R と N、落下の 2 版）は、種の塊（既定 10 種）ごとに交互に回す（目標書_段階4.md 第 4 節「種の塊ごとに交互」）。
    塊 b の中で、通りの順（--variants の順）× モデルの順（--models の順）に、その塊の試行だけを回す（96 の --max-new で区切る）。
    止まって再開しても、まだ完全でない試行を同じ順で回す。
  - 落下で手を止める版（fall_with_hold、誘発の版 P2H-v1）は診断にだけ使い、テストでは使わない（目標書_段階4.md 第 10 節 6、
    s4_gates の C.b2）。中身は src\\recovla\\diag\\recovery.py の冒頭。置き損ね・把持失敗・落下そのままは段階 3 の誘発のまま。
  - 本番の帯は s4_gates.json の D_recovery の割り当てと完全に一致する指定だけを受け付ける。smoke は smoke_data の帯（44400〜44799）
    で、実験名は S4SMOKE で始める（本番の実験名 S4DREC と混ぜない）。
終了コード: 0 全部そろった、1 途中で止まった（合図・--max-new・メモリ待ちの時間切れ）、2 エラー、3 引数・前提の食い違い。
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT96 = ROOT / "scripts" / "96_s4_resume.py"
GATES = ROOT / "configs" / "s4_gates.json"
OUT_S4 = ROOT / "outputs" / "s4" / "d_recovery"
V3S3 = ROOT / "outputs" / "v2eval" / "V3S3"
EXPERIMENT = "S4DREC"
MODELS = ("R1v3", "N1v3")
ARM = {"R1v3": "R", "N1v3": "N"}             # R = 復帰デモあり、N = 復帰デモなし（段階 3 の A・B と同じモデル）
FIXED = ["--mode", "naive", "--exec-interval", "6", "--no-safety"]    # V3S3 の誘発の条件と同じ（run.json の mode・exec_interval・safety）
# 段階 3 の記録で確かめる値（掲示板 0144・0145、final.md 2-4・2-7。check-s3）
S3_EXPECT = {"P1": {"recovered": 16, "established": 32, "pairs": 27, "r_only": 13, "n_only": 0},
             "A_P2_stale_v1": {"not_post_land": 34, "established": 38}}


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


def gates() -> dict:
    return json.loads(GATES.read_text(encoding="utf-8"))


def allocation(g: dict, aid: str):
    a = next(x for x in g["bands"]["allocations"] if x["id"] == aid)
    return int(a["range"][0]), int(a["range"][1])


def default_trials(g: dict) -> str:
    lo, hi = allocation(g, "D_recovery")
    return f"induced:{lo}:{hi - lo + 1}"


def check_band(trials: str, experiment: str, g: dict) -> str:
    """帯の確かめ。返り値 "production" か "smoke"。合わなければ SystemExit（3）。"""
    try:
        kind, base, n = trials.split(":")
        base, n = int(base), int(n)
    except ValueError:
        raise SystemExit(f"--trials {trials!r}: induced:<種の先頭>:<数> の形で指定する")
    if kind != "induced" or n <= 0:
        raise SystemExit(f"--trials {trials!r}: D-復帰は induced だけ")
    lo, hi = allocation(g, "D_recovery")
    sm = next(c for c in g["bands"]["classes"] if c["id"] == "smoke_data")["range"]
    if (base, base + n - 1) == (lo, hi):
        if experiment.startswith("S4SMOKE") or not experiment.startswith("S4"):
            raise SystemExit(f"本番の帯 {lo}〜{hi} は S4 で始まる本番の実験名で回す（S4SMOKE* は smoke 専用）: {experiment}")
        return "production"
    if sm[0] <= base and base + n - 1 <= sm[1]:
        if not experiment.startswith("S4SMOKE"):
            raise SystemExit(f"smoke の帯（{sm[0]}〜{sm[1]}）は実験名 S4SMOKE* で回す: {experiment}")
        return "smoke"
    raise SystemExit(f"--trials {trials!r}: 帯が s4_gates.json の D_recovery（{lo}〜{hi} を全部）とも smoke_data（{sm[0]}〜{sm[1]}）とも合わない")


def cond_name(model: str, variant: str) -> str:
    return f"{model}_{variant}"


def schedule(todo: dict, order: list, block: int, cap: int = 0) -> list:
    """種の塊ごとに交互に回す順。todo = {条件: [まだ完全でない試行の番号（小さい順）]}、order = 条件の順。
    返り値 [(条件, その回に回す本数)]。96 の cmd_main は todo を小さい順に回すので、塊 b までの残りを --max-new で区切る。
    cap > 0 なら全体の本数をそこで打ち切る（smoke 用）。"""
    if block <= 0:
        raise ValueError("block は 1 以上")
    last = max((max(v) for v in todo.values() if v), default=-1)
    used = {c: 0 for c in order}
    out, total = [], 0
    for b in range(last // block + 1):
        for c in order:
            k = sum(1 for i in todo.get(c, []) if i < (b + 1) * block) - used[c]
            if cap:
                k = min(k, cap - total)
            if k > 0:
                out.append((c, k))
                used[c] += k
                total += k
        if cap and total >= cap:
            break
    return out


# ---------------------------------------------------------------- 96 の包み
def load_96():
    return _load(SCRIPT96, "s4_resume_dC")


def patch_96(r96, DR):
    """読み込んだ 96 の写しの上で、Engine.run_one・make_spec・run_json_text を包む（ファイルは書き換えない）。"""
    sha96 = sha256_file(SCRIPT96)
    sha_mod = sha256_file(pathlib.Path(DR.__file__))
    sha_me = sha256_file(pathlib.Path(__file__))
    base_engine, base_spec, base_runjson = r96.Engine, r96.make_spec, r96.run_json_text

    def diag_of(a) -> dict:
        v = DR.VARIANTS[a.diag_variant]
        return {"name": "D_recovery", "variant": a.diag_variant, "variant_ja": v["ja"], "induce_kind": v["induce"],
                "induce_version": v["version"], "arm": ARM.get(a.model), "model": a.model, "trials": a.trials,
                "start_state": "scene.sample_layout(種)（41_results.trial_list の induced。段階 3 の誘発の試行と同じ配置の引き方）",
                "diag_only": bool(v["hold"]), "test_use": "使わない（診断専用）" if v["hold"] else "段階 3 と同じ誘発",
                "score": {"primary_s": 30, "secondary_s": 60, "descriptive_s": 45,
                          "denominator": "誘発が L 秒以内に成立した試行（induce.t_established <= L）"},
                "script": "scripts/98_s4_d_recovery.py", "script_sha256": sha_me,
                "module": "src/recovla/diag/recovery.py", "module_sha256": sha_mod, "resume_sha256": sha96}

    class DiagEngine(base_engine):
        def run_one(self, i, seed, lay, tgt):
            var = self.a.diag_variant
            v = DR.VARIANTS[var]
            if not v["hold"]:
                meta, arrays, rlog = super().run_one(i, seed, lay, tgt)
            else:
                meta, arrays, rlog = self._run_hold(i, seed, lay, tgt)
                if (meta.get("induce") or {}).get("version") != DR.FallHoldInducer.VERSION:
                    raise RuntimeError("手を止める版の誘発が使われていない（96 の run_one が誘発を作る方法が変わった？）。"
                                       "記録を使わずに 98 の包みを見直す")
            meta["induce"].setdefault("variant", var)
            meta["induce"].setdefault("version", v["version"])
            meta["diag"] = diag_of(self.a)                       # 96（=82）の形に足す欄
            return meta, arrays, rlog

        def _run_hold(self, i, seed, lay, tgt):
            """96 の run_one をそのまま呼び、その間だけ誘発の作り方を手を止める版に、make_runtime を「作った実行系を誘発に
            つなぐ」ものに差し替える（どちらも呼び終えたら戻す）。"""
            from recovla.eval import induce as I
            holder, orig_mk, orig_ind = {}, self.make_runtime, I.Inducer

            def factory(kind, seed_, lay_, tgt_, rig, cfg=None):
                if kind != "P2":
                    raise ValueError(f"手を止める版は P2 だけ（{kind}）")
                holder["ind"] = DR.FallHoldInducer(seed_, lay_, tgt_, rig, cfg)
                return holder["ind"]

            def mk(io, setup):
                rt = orig_mk(io, setup)
                holder["ind"].bind_runtime(rt)
                return rt

            self.make_runtime, I.Inducer = mk, factory
            try:
                return super().run_one(i, seed, lay, tgt)
            finally:
                self.make_runtime, I.Inducer = orig_mk, orig_ind

    def make_spec(a):
        s = base_spec(a)
        s.update(diag_variant=a.diag_variant, diag_induce_version=DR.VARIANTS[a.diag_variant]["version"])
        return s

    def run_json_text(a, kind, rows, wall_s, exec_interval, lim, env=None, segs=None):
        d = json.loads(base_runjson(a, kind, rows, wall_s, exec_interval, lim, env, segs))
        d["diag"] = diag_of(a)                                   # 96 の形に足す欄
        return json.dumps(d, ensure_ascii=False, indent=1)

    r96.Engine, r96.make_spec, r96.run_json_text = DiagEngine, make_spec, run_json_text
    return r96


def build_96_args(r96, a, model: str, variant: str, max_new: int, dry: bool):
    from recovla.diag import recovery as DR
    argv = ["run", "--experiment", a.experiment, "--condition", cond_name(model, variant), "--model", model,
            "--trials", a.trials, "--induce", DR.VARIANTS[variant]["induce"], "--time-limit-s", "60"] + FIXED
    argv += ["--min-free-gb", str(a.min_free_gb), "--min-commit-free-gb", str(a.min_commit_free_gb),
             "--mem-timeout-min", str(a.mem_timeout_min)]
    if a.quiet_window:
        argv += ["--quiet-window", a.quiet_window]
    if a.accept_env_change:
        argv.append("--accept-env-change")
    if max_new:
        argv += ["--max-new", str(max_new)]
    if dry:
        argv.append("--dry-run")
    ns = r96.build_parser().parse_args(argv)
    ns.diag_variant = variant
    return ns


# ---------------------------------------------------------------- run
def cmd_run(a) -> int:
    from recovla.diag import recovery as DR
    g = gates()
    a.trials = a.trials or default_trials(g)
    kind = check_band(a.trials, a.experiment, g)
    models = [m for m in a.models.split(",") if m]
    variants = [v for v in a.variants.split(",") if v]
    bad = [m for m in models if m not in MODELS] + [v for v in variants if v not in DR.VARIANTS]
    if bad:
        raise SystemExit(f"知らないモデル・通り: {bad}（モデル {MODELS}、通り {DR.VARIANT_ORDER}）")
    r96 = patch_96(load_96(), DR)
    ops = r96.load_ops()
    v82 = r96.load_82(False)
    order = [cond_name(m, v) for v in variants for m in models]
    trials = v82.trial_list(a.trials)
    out_root = v82.OUT / a.experiment
    todo = {}
    for m in models:
        for v in variants:
            c = cond_name(m, v)
            todo[c] = [i for i, (seed, lay, tgt) in enumerate(trials)
                       if not r96.check_complete("run", out_root / c, i, {"seed": seed, "target": tgt})[0]]
    plan = schedule(todo, order, a.block, a.max_new)
    print(f"[d-rec] {kind}: 実験 {a.experiment}、帯 {a.trials}（{len(trials)} 試行/条件）、条件 {order}、塊 {a.block} 種", flush=True)
    print(f"[d-rec] 残り: {{{', '.join(f'{c}: {len(v)}' for c, v in todo.items())}}}、この回の順: {plan}", flush=True)
    if a.dry_run:
        code = 0
        for m in models:
            for v in variants:
                code = max(code, _call_96(r96, v82, ops, build_96_args(r96, a, m, v, 0, True)))
        return code
    if not plan:
        print("[d-rec] 回す試行はない（全部完全）。run.json が無い条件があれば 96 で 1 回回すと書く", flush=True)
    OUT_S4.mkdir(parents=True, exist_ok=True)
    tag = f"{a.experiment}_{'-'.join(models)}_{'-'.join(variants)}"
    prog_path = pathlib.Path(a.progress_file) if a.progress_file else OUT_S4 / f"progress_{tag}.json"
    base = {"experiment": a.experiment, "condition": f"D_recovery {'+'.join(order)}", "cmd": "98_s4_d_recovery run",
            "pid": os.getpid(), "status": "starting", "total": len(trials) * len(order),
            "done": len(trials) * len(order) - sum(len(v) for v in todo.values()), "plan": plan, "trials_spec": a.trials,
            "started": r96._now_s(), "stop_reason": None, "error": None}
    prog = r96.Progress(prog_path, ops, base)
    prog.start_heartbeat()
    status, code, rss = "done", 0, []
    try:
        for step, (c, n) in enumerate(plan):
            m, v = c.split("_", 1)
            prog.update(status="running", current=c, current_n=n, step=step)
            ns = build_96_args(r96, a, m, v, n, False)
            rc = _call_96(r96, v82, ops, ns)
            pj = _read_json(out_root / c / "progress.json") or {}
            reason = str(pj.get("stop_reason") or "")
            if rc == 1 and reason.startswith("max_new"):
                pass                                              # 塊の区切り（96 の --max-new）。次の条件へ
            elif rc != 0:
                status, code = {1: "stopped", 2: "error", 3: "error"}.get(rc, "error"), rc
                prog.update(stop_reason=f"{c}: 96 の終了コード {rc}（{reason or pj.get('error')}）")
                break
            rss.append([c, _rss_gb()])
            prog.update(done=_count_complete(r96, v82, out_root, order, trials), rss_gb_after_call=rss)
    except KeyboardInterrupt:
        status, code = "interrupted", 1
    except BaseException as e:                                    # noqa: BLE001
        status, code = "error", 2
        prog.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        traceback.print_exc()
    prog.stop_heartbeat()
    done = _count_complete(r96, v82, out_root, order, trials, full=True)
    if status == "done" and done < len(trials) * len(order) and not a.max_new:
        status, code = "error", 2
        prog.update(error="全部回したのに完全でない記録が残っている")
    elif status == "done" and done < len(trials) * len(order):
        status, code = "stopped", 1                               # --max-new（smoke）で打ち切った
        prog.update(stop_reason=f"max_new:{a.max_new}")
    prog.update(status=status, done=done)
    print(f"[d-rec] {status}（完全 {done}/{len(trials) * len(order)}）", flush=True)
    return code


def _call_96(r96, v82, ops, ns) -> int:
    try:
        return r96.cmd_main(ns, v82, ops)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise
    finally:
        _release_models()


def _release_models() -> None:
    """96 の Engine（方策の模型を持つ）は make_runtime の閉包と循環するので、呼び終えるたびに循環を回収する。
    注意（smoke、10/08）: 1 つのプロセスで 96 を 2 回以上呼ぶと、作業セットの最大が 3.83 GB から 5.2〜5.4 GB になる
    （回収の有無で変わらなかった。3 回目以降は増えない）。メモリを詰めるなら、モデルごとに別のプロセスにする（--models を 1 つ）。"""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                             # noqa: BLE001
        pass


def _rss_gb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1024 ** 3, 2)
    except Exception:                                             # noqa: BLE001
        return None


def _read_json(p: pathlib.Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                             # noqa: BLE001
        return None


def _count_complete(r96, v82, out_root, order, trials, full: bool = False) -> int:
    """完全な試行の数。途中の進捗はファイルがそろっているかだけで数え（速い）、最後は 96 の check_complete で数える。"""
    if full:
        return sum(r96.check_complete("run", out_root / c, i, {"seed": s, "target": t})[0]
                   for c in order for i, (s, lay, t) in enumerate(trials))
    return sum(all(p.is_file() for p in r96.trial_paths("run", out_root / c, i).values())
               for c in order for i in range(len(trials)))


# ---------------------------------------------------------------- plan（見込み）
def cmd_plan(a) -> int:
    from recovla.diag import recovery as DR
    g = gates()
    trials = a.trials or default_trials(g)
    kind = check_band(trials, a.experiment, g)
    n = int(trials.split(":")[2])
    fac = g["budget_notes"]["factor_60_over_30"]
    # 段階 3 の同じ誘発・同じモデルの条件（V3S3、30 s、旧ドライバ 560.94、3 本並行）の 1 試行の実時間に、60 s の倍率を掛ける。
    # N の落下・置き損ねは A の倍率を使う（目標書_段階4.md 8-6 と同じ置き方）。手を止める版は落下そのままと同じと置く（推測）
    src = {("R1v3", "P1"): ("A_P1", "A_P1"), ("N1v3", "P1"): ("B_P1", "B_P1"), ("R1v3", "P2"): ("A_P2", "A_P2"),
           ("N1v3", "P2"): ("B_P2", "A_P2"), ("R1v3", "P3"): ("A_P3", "A_P3"), ("N1v3", "P3"): ("B_P3", "A_P3")}
    rows, tot = [], 0.0
    for v in DR.VARIANT_ORDER:
        for m in MODELS:
            ind = DR.VARIANTS[v]["induce"]
            c3, fk = src[(m, ind)]
            rj = json.loads((V3S3 / c3 / "run.json").read_text(encoding="utf-8"))
            per = float(rj["wall_s"]) / int(rj["n"])
            h = per * n * float(fac[fk]) / 3600
            tot += h
            rows.append({"condition": cond_name(m, v), "induce": ind, "version": DR.VARIANTS[v]["version"], "trials": n,
                         "s3_condition": c3, "s3_wall_per_trial_s": round(per, 1), "factor_60": fac[fk], "process_h": round(h, 2)})
    out = {"kind": kind, "experiment": a.experiment, "trials": trials, "seeds": f"{trials.split(':')[1]}〜{int(trials.split(':')[1]) + n - 1}",
           "conditions": rows, "total_trials": n * len(rows), "process_hours": round(tot, 1),
           "parallel_3_hours_at_95pct": round(tot / 3 / 0.95, 1),
           "charter_estimate_h": g["budget_notes"]["d_recovery_process_hours"],
           "basis": "V3S3 の run.json の wall_s / n（30 s、旧ドライバ、3 本並行）× s4_gates の factor_60_over_30。推測",
           "commands": [f".venv\\Scripts\\python.exe scripts\\98_s4_d_recovery.py run --experiment {a.experiment} --trials {trials} "
                        f"--variants {v} --models R1v3,N1v3" for v in DR.VARIANT_ORDER]}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------- score（読むだけ）
def cmd_score(a) -> int:
    from recovla.diag import recovery as DR
    r96 = load_96()
    base = ROOT / "outputs" / "v2eval" / a.experiment
    ats = tuple(float(x) for x in a.at.split(","))
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": a.experiment, "at": ats,
           "note": "数値だけ。判定は二重集計役が configs/s4_gates.json の gates.C で当てはめる（主 30 s、副 60 s、45 s は記述）",
           "conditions": {}, "variants": {}}
    for v in DR.VARIANT_ORDER:
        dr, dn = base / cond_name(a.r_model, v), base / cond_name(a.n_model, v)
        if not (dr.is_dir() and dn.is_dir()):
            continue
        recs_r, recs_n = DR.load_condition(dr), DR.load_condition(dn)
        for d in (dr, dn):
            res["conditions"][d.name] = {"complete_run_json": (d / "run.json").is_file(),
                                         "score_96": r96.score_condition(d, list(ats))["at"]}
        ent = {"R": dr.name, "N": dn.name, "n_R": len(recs_r), "n_N": len(recs_n),
               "material": DR.gate_c_material(recs_r, recs_n, ats), "cross_check_time_scoring": DR.cross_check(recs_r, recs_n, ats)}
        if DR.VARIANTS[v]["induce"] == "P2":
            tr, tn = DR.fall_timing_dir(dr), DR.fall_timing_dir(dn)
            ent["fall_timing"] = {"R": DR.summarize_fall_timing(tr), "N": DR.summarize_fall_timing(tn), "rows_R": tr, "rows_N": tn}
        res["variants"][v] = ent
    if "fall_with_hold" in res["variants"]:
        res["gate_C_material"] = {"variant": "fall_with_hold", **res["variants"]["fall_with_hold"]["material"].get("primary_30s", {}),
                                  "secondary_60s": res["variants"]["fall_with_hold"]["material"].get("secondary_60s")}
    if {"fall_as_is", "fall_with_hold"} <= set(res["variants"]):           # C.b2 の材料（報告だけ）
        res["C_b2_material"] = {L: {v: res["variants"][v]["material"]["by_L"][L]["recovery_R"] for v in ("fall_as_is", "fall_with_hold")}
                                for L in res["variants"]["fall_with_hold"]["material"]["by_L"]}
    text = json.dumps(res, ensure_ascii=False, indent=1, default=str)
    out = pathlib.Path(a.out) if a.out else OUT_S4 / f"score_{a.experiment}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    brief = {v: {"primary_30s": e["material"].get("primary_30s"), "agree": e["cross_check_time_scoring"]["agree"]}
             for v, e in res["variants"].items()}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    print(f"[d-rec] {out}")
    return 0 if all(e["cross_check_time_scoring"]["agree"] for e in res["variants"].values()) else 2


# ---------------------------------------------------------------- check-s3（読むだけ）
def cmd_check_s3(a) -> int:
    from recovla.diag import recovery as DR
    recs_r, recs_n = DR.load_condition(V3S3 / "A_P1"), DR.load_condition(V3S3 / "B_P1")
    mat = DR.gate_c_material(recs_r, recs_n, (30.0,))
    p = mat["primary_30s"]
    got = {"recovered": p["recovery_R_k_n"][0], "established": p["recovery_R_k_n"][1], "pairs": p["pairs"],
           "r_only": p["r_only"], "n_only": p["n_only"]}
    rows = DR.fall_timing_dir(V3S3 / "A_P2")
    s = DR.summarize_fall_timing(rows)
    v1 = s["v1_class_t_obs"]
    got_p2 = {"not_post_land": s["n_established"] - v1.get("post_land", 0), "established": s["n_established"]}
    ok = got == S3_EXPECT["P1"] and got_p2 == S3_EXPECT["A_P2_stale_v1"]
    cc = DR.cross_check(recs_r, recs_n, (30.0,))
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "ok": bool(ok and cc["agree"]),
           "P1": {"source": "outputs/v2eval/V3S3/A_P1（R1v3）・B_P1（N1v3）。30 s、旧ドライバ 560.94",
                  "expect": S3_EXPECT["P1"], "got": got, "R_minus_N_30": p["R_minus_N"], "cross_check_time_scoring": cc},
           "A_P2_stale_v1": {"source": "outputs/v2eval/V3S3/A_P2。懐疑役の検算 v1（W\\skeptic_lang_recovery\\v1_p2_stale.py）と同じ定義",
                             "expect": S3_EXPECT["A_P2_stale_v1"], "got": got_p2, "summary": s}}
    OUT_S4.mkdir(parents=True, exist_ok=True)
    (OUT_S4 / "check_s3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("ok",)} | {"P1": got, "R_minus_N_30": p["R_minus_N"], "A_P2": got_p2,
                                                     "cross_check": cc["agree"]}, ensure_ascii=False))
    return 0 if res["ok"] else 1


# ---------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--experiment", default=EXPERIMENT)
        p.add_argument("--trials", default=None, help="既定は s4_gates.json の D_recovery（induced:190500:50）")
        if name == "run":
            p.add_argument("--models", default=",".join(MODELS))
            p.add_argument("--variants", default="fall_with_hold,fall_as_is,misplace,grasp_failure",
                           help="回す順でもある（既定は関門 C の版から）")
            p.add_argument("--block", type=int, default=10, help="交互に回す種の塊の大きさ")
            p.add_argument("--max-new", type=int, default=0, help="この回に回す試行の上限（smoke 用。0 で無制限）")
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--progress-file", default=None)
            p.add_argument("--min-free-gb", type=float, default=12.0)
            p.add_argument("--min-commit-free-gb", type=float, default=6.0)
            p.add_argument("--mem-timeout-min", type=float, default=120.0)
            p.add_argument("--quiet-window", default="")
            p.add_argument("--accept-env-change", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("--experiment", default=EXPERIMENT)
    p.add_argument("--r-model", default="R1v3")
    p.add_argument("--n-model", default="N1v3")
    p.add_argument("--at", default="30,45,60")
    p.add_argument("--out", default=None)
    sub.add_parser("check-s3")
    return ap


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                             # noqa: BLE001
        pass
    a = build_parser().parse_args(argv)
    try:
        return {"run": cmd_run, "plan": cmd_plan, "score": cmd_score, "check-s3": cmd_check_s3}[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
