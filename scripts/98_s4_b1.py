"""段階 4 束 2 の B1（実行器 v3 の確認）。3 個の連続タスクを、96_s4_resume.py task の包みで、今の実行器と v3 を同じ種で回す。

使い方（作業場所 C:\\PAI\\recovery_vla。96_s4_resume.py task の続きの引数（--max-new・--accept-env-change など）は後ろに付ける）:
    .venv\\Scripts\\python.exe scripts\\98_s4_b1.py plan --gate1 outputs\\s4\\gate1_result.json          # 腕・帯・戻し先・コマンド（読むだけ）
    .venv\\Scripts\\python.exe scripts\\98_s4_b1.py run --arm R1v3_v3 --gate1 outputs\\s4\\gate1_result.json --dry-run
    .venv\\Scripts\\python.exe scripts\\98_s4_b1.py rotate --gate1 outputs\\s4\\gate1_result.json [--block 10] [--dry-run]
        3 つの腕（R1v3_cur・R1v3_v3・N1v3_v3）を、同じ種で 10 本の塊ごとに交互に回す（目標書_段階4.md 第 4 節）。止まって再開しても
        同じコマンドで続きから回る（96 が完全な記録を飛ばす）。
  腕（条件名 = 腕の名前、実験名の既定 S4B1）:
    R1v3_cur  R1v3 ＋ 今の実行器（runtime/executor.py）。帯 191100〜191159 の 60 本（--trials 191100:60）
    R1v3_v3   R1v3 ＋ 実行器 v3（runtime/executor_v3.py）。同じ 60 本
    N1v3_v3   N1v3 ＋ 実行器 v3。帯 191100〜191129 の 30 本（--trials 191100:30。N1v3 の腕＝復帰デモの効果と取り違えさせない）
  v3 の戻し先（(a)）: 関門 1 の判定の JSON（98_s4_gate1.py の --out。conclusion.bundle2.v3a）から読む。include が偽なら (a) なし。
    判定が "undetermined" なら止める（終了コード 3）。--v3-goal EH|ES|none で上書きでき、上書きは試行の json の b1.v3_source と
    run.json・resume_spec.json に残る。--v3-off a,b,c,d で v3 の部分を個別に切れる（切り分けの診断。本番では使わない）。
  帯（configs/s4_gates.json の allocations と完全に一致する指定だけ）: R1v3 は bundle2_confirm の 191100〜191159 そのもの、
    N1v3 は目標書 4-2 の「N1v3 は 191100〜191129」そのもの。--allow-smoke のときだけ smoke・データの帯 44400〜44799 も通す
    （ただし X2 の生成の帯 44404〜44423 は拒む。実験名は S4SMOKE で始める）。
  実行のしかた（段階 3 の E7・D-E7 と同じ）: 行動の区切り 6 行（--exec-interval 6）、安全フィルタなし（--no-safety）、
    1 手順の試み 30 s・やり直し 1 回・全体 200 s、計画役 s4（Haiku 5.5）、試行ごとに世界を作り直す。違う値を渡すと止める。
  ■ 鍵の注意: 試すときは必ず --dry-run を付ける（付けないと計画役の API を呼び、帯の種を使う）。
読むもの: scripts\\96_s4_resume.py（importlib。Engine・SPEC_KEYS・run_json_text を読み込んだ写しの上で包む。書き換えない）、
  configs\\s4_gates.json（帯）、--gate1 の JSON、src\\recovla\\runtime\\executor_v3.py。
書くもの: 96_s4_resume.py task と同じ記録（outputs\\v2eval\\<実験>\\<腕>\\）。試行の json に "v3"（設定・出来事・介入の種類別の数）と
  "b1"（腕・モデル・戻し先とその出どころ・使ったファイルの SHA-256・API の回し直し）を、run.json に "b1" を足す。
  rotate の進み具合は outputs\\s4\\b1\\rotate_<実験>.log.json。
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
GATES = ROOT / "configs" / "s4_gates.json"
OUTD = ROOT / "outputs" / "s4" / "b1"
ARMS = {
    "R1v3_cur": {"model": "R1v3", "executor": "current", "trials": "191100:60", "ja": "R1v3 ＋ 今の実行器"},
    "R1v3_v3": {"model": "R1v3", "executor": "v3", "trials": "191100:60", "ja": "R1v3 ＋ 実行器 v3"},
    "N1v3_v3": {"model": "N1v3", "executor": "v3", "trials": "191100:30", "ja": "N1v3 ＋ 実行器 v3（復帰デモなしの腕）"},
}
ROTATE_ORDER = ("R1v3_cur", "R1v3_v3", "N1v3_v3")
FILES = ("src/recovla/runtime/executor_v3.py", "src/recovla/runtime/executor.py", "scripts/98_s4_b1.py", "scripts/96_s4_resume.py",
         "configs/s4_gates.json")
V3_PARTS = {"a": "a_goal", "b": "b_placed_fix", "c": "c_final_wait", "d": "d_early_abort"}
SHARED_CACHE = {}                         # rotate: 方策（GPU の模型）をプロセスの中でモデルごとに使い回す


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


# ---------------------------------------------------------------- 帯
def bands() -> dict:
    g = json.loads(GATES.read_text(encoding="utf-8"))
    al = {a["id"]: tuple(a["range"]) for a in g["bands"]["allocations"]}
    sm = next(x for x in g["bands"]["classes"] if x["id"] == "smoke_data")["range"]
    return {"R1v3": al["bundle2_confirm"], "N1v3": (al["bundle2_confirm"][0], 191129), "smoke": tuple(sm), "x2": al["X2_gen"]}


def check_band(arm: str, trials: str, experiment: str, allow_smoke: bool) -> str:
    """"" なら通す。R1v3・N1v3 の本番は割り当てと完全に一致する指定だけ。smoke は --allow-smoke・S4SMOKE の実験名・X2 の帯の外。"""
    b = bands()
    base, n = (int(x) for x in trials.split(":"))
    lo, hi = b[ARMS[arm]["model"]]
    if (base, base + n - 1) == (lo, hi):
        return ""
    seeds = range(base, base + n)
    if allow_smoke:
        if not experiment.startswith("S4SMOKE"):
            return f"smoke の実験名は S4SMOKE で始める（{experiment}）"
        if not all(b["smoke"][0] <= s <= b["smoke"][1] for s in seeds):
            return f"種 {base}〜{base + n - 1} が smoke の帯 {b['smoke']} に収まらない"
        hit = [s for s in seeds if b["x2"][0] <= s <= b["x2"][1]]
        if hit:
            return f"種 {hit[0]}〜{hit[-1]} は X2 の生成の帯 {b['x2']}（予約）なので smoke でも使わない"
        return ""
    return f"{arm} の本番の帯は {lo}:{hi - lo + 1}（{lo}〜{hi}）と完全に一致する指定だけ（渡された {trials}）。smoke は --allow-smoke"


# ---------------------------------------------------------------- 関門 1 の判定から v3 の設定
def v3_from_gate1(path, override: str = None, off: str = "") -> tuple:
    """(v3 の設定, 出どころの記録)。override は "EH"・"ES"・"none"。"""
    src = {"gate1": None, "override": None}
    goal = None
    if path:
        res = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        v3a = res["conclusion"]["bundle2"]["v3a"]
        src["gate1"] = {"path": str(path), "sha256": sha256_file(path), "v3a": v3a,
                        "conclusion_status": res["conclusion"].get("status")}
        if v3a.get("result") == "undetermined" and override is None:
            raise SystemExit("関門 1 の判定で v3 (a) の戻し先が判定できない（undetermined）。判定を待つか --v3-goal で上書きする")
        if v3a.get("include"):
            goal = v3a.get("return_to")
            if goal not in ("EH", "ES"):
                raise SystemExit(f"関門 1 の判定の戻し先が EH・ES でない: {goal!r}")
    elif override is None:
        raise SystemExit("--gate1（関門 1 の判定の JSON）か --v3-goal が要る")
    if override is not None:
        new = None if override == "none" else override
        src["override"] = {"v3_goal": override, "from_gate1": goal, "note": "引数で上書きした（記録に残す）"}
        goal = new
    v3 = {"a_goal": goal}
    for p in [x for x in (off or "").split(",") if x]:
        if p not in V3_PARTS:
            raise SystemExit(f"--v3-off は a,b,c,d の組み合わせ: {off}")
        v3[V3_PARTS[p]] = None if p == "a" else False
    src["v3_off"] = sorted(x for x in (off or "").split(",") if x)
    return v3, src


# ---------------------------------------------------------------- 96 を包む
def patch96(r96, arm: str, v3: dict, info: dict) -> None:
    """96 の Engine・SPEC_KEYS・run_json_text を、読み込んだ写しの上で包む。呼ぶたびに元から包み直す。"""
    from recovla.runtime import executor_v3 as V3
    orig = getattr(r96, "_b1_orig", None)
    if orig is None:
        orig = {"Engine": r96.Engine, "SPEC_KEYS": tuple(r96.SPEC_KEYS), "run_json_text": r96.run_json_text}
        r96._b1_orig = orig
    r96.SPEC_KEYS = orig["SPEC_KEYS"] + ("b1_arm", "b1_v3", "b1_executor_sha256")
    use_v3 = ARMS[arm]["executor"] == "v3"

    class B1Engine(orig["Engine"]):
        def __init__(self, a, v82, cfg, lim, env=None):
            super().__init__(a, v82, cfg, lim, env)
            self.cache = SHARED_CACHE.setdefault(a.model, {})
            self.b1_last = None

        def init_task(self):
            super().init_task()
            inner = self.make_task

            def make(io, setup):
                ex = inner(io, setup)
                if use_v3:
                    ex = V3.install(ex, v3)
                self.b1_last = ex
                return ex
            self.make_task = make

        def run_one_task(self, i, seed):
            retry = None
            try:
                meta, arrays, rlog = super().run_one_task(i, seed)
            except Exception as e:                           # noqa: BLE001
                if not any(c.__module__.split(".")[0] == "anthropic" for c in type(e).__mro__):
                    raise
                retry = {"n": 1, "error": f"{type(e).__name__}: {e}"[:500], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                print(f"[b1] 試行 {i} 種 {seed}: 計画役の API が失敗したので同じ種で 1 回だけ回し直す（{retry['error']}）", flush=True)
                meta, arrays, rlog = super().run_one_task(i, seed)
            if use_v3:
                meta["v3"] = self.b1_last.record()["v3"]
            meta["b1"] = dict(info, api_retry=retry)
            return meta, arrays, rlog

    def run_json_text(a_, kind, rows, *x, **kw):
        d = json.loads(orig["run_json_text"](a_, kind, rows, *x, **kw))
        d["b1"] = dict(info)
        return json.dumps(d, ensure_ascii=False, indent=1)

    r96.Engine = B1Engine
    r96.run_json_text = run_json_text


FORBIDDEN_EXTRA = ("--model", "--trials", "--experiment", "--condition", "--exec-interval", "--no-safety", "--planner", "--text",
                   "--step-timeout-s", "--task-time-limit-s", "--reuse-world")


def run_arm(r96, ops, v82, a, arm: str, extra: list, max_new: int = None) -> int:
    for f in extra:
        if f.split("=")[0] in FORBIDDEN_EXTRA:
            raise SystemExit(f"{f} は 98_s4_b1.py が決める（モデル・帯・6 行・安全フィルタなし・計画役 s4・30 s・200 s）")
    spec = ARMS[arm]
    trials = a.trials or spec["trials"]
    why = check_band(arm, trials, a.experiment, a.allow_smoke)
    if why:
        raise SystemExit(why)
    v3, src = v3_from_gate1(a.gate1, a.v3_goal, a.v3_off)
    info = {"script": "98_s4_b1.py", "arm": arm, "arm_ja": spec["ja"], "model": spec["model"], "executor": spec["executor"],
            "v3": v3 if spec["executor"] == "v3" else None, "v3_source": src if spec["executor"] == "v3" else None,
            "files_sha256": {p: sha256_file(ROOT / p) for p in FILES}}
    argv = ["task", "--experiment", a.experiment, "--condition", arm, "--model", spec["model"], "--trials", trials,
            "--exec-interval", "6", "--no-safety"]
    if max_new:
        argv += ["--max-new", str(max_new)]
    a96 = r96.build_parser().parse_args(argv + list(extra))
    a96.b1_arm = arm
    a96.b1_v3 = json.dumps(info["v3"], sort_keys=True)
    a96.b1_executor_sha256 = info["files_sha256"]["src/recovla/runtime/executor_v3.py"]
    patch96(r96, arm, v3, info)
    print(f"[b1] {arm}: {spec['ja']}（{a.experiment}\\{arm} {trials}、v3 {info['v3']}）", flush=True)
    try:
        return r96.cmd_main(a96, v82, ops)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


def _env(a):
    r96 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_b1")
    return r96, r96.load_ops(), r96.load_82(False)


def cmd_run(a, extra) -> int:
    r96, ops, v82 = _env(a)
    return run_arm(r96, ops, v82, a, a.arm, extra)


def stop_requested(a, arm) -> str:
    for p in (ROOT / "outputs" / "v2eval" / a.experiment / arm / "STOP", ROOT / "outputs" / "s4" / "STOP"):
        if p.is_file():
            return str(p)
    return ""


def cmd_rotate(a, extra) -> int:
    """腕を、同じ種で --block 本の塊ごとに交互に回す（1 つのプロセス。方策はモデルごとに読み込んだまま）。"""
    arms = [x for x in (a.arms.split(",") if a.arms else ROTATE_ORDER)]
    for x in arms:
        if x not in ARMS:
            raise SystemExit(f"--arms: {sorted(ARMS)} のどれか")
    r96, ops, v82 = _env(a)
    if a.dry_run:
        rc = 0
        for arm in arms:
            rc = max(rc, run_arm(r96, ops, v82, a, arm, extra + ["--dry-run"]))
        return rc
    log = {"experiment": a.experiment, "arms": arms, "block": a.block, "rounds": []}
    OUTD.mkdir(parents=True, exist_ok=True)
    lp = OUTD / f"rotate_{a.experiment}.log.json"
    done = {x: False for x in arms}
    for r in range(10_000):
        row = {"round": r, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "rc": {}}
        for arm in arms:
            if done[arm]:
                continue
            s = stop_requested(a, arm)
            if s:
                print(f"[b1] 止める合図 {s}", flush=True)
                log["stopped"] = s
                lp.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
                return 1
            rc = run_arm(r96, ops, v82, a, arm, extra, max_new=a.block)
            row["rc"][arm] = rc
            if rc == 0:
                done[arm] = True
            elif rc != 1:
                log["rounds"].append(row)
                lp.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
                return rc
        log["rounds"].append(row)
        lp.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        if all(done.values()):
            return 0
    return 1


def cmd_plan(a) -> int:
    v3, src = v3_from_gate1(a.gate1, a.v3_goal, a.v3_off)
    out = {"arms": ARMS, "v3": v3, "v3_source": src, "bands": {k: list(v) for k, v in bands().items()},
           "trials_total": sum(int(x["trials"].split(":")[1]) for x in ARMS.values()),
           "commands": [f"scripts\\98_s4_b1.py run --arm {k} --gate1 {a.gate1} --experiment {a.experiment}" for k in ARMS] +
                       [f"scripts\\98_s4_b1.py rotate --gate1 {a.gate1} --experiment {a.experiment} --block 10"]}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "rotate", "plan"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("--arm", required=True, choices=list(ARMS))
        if name == "rotate":
            p.add_argument("--arms", default=None, help="カンマ区切り（既定 R1v3_cur,R1v3_v3,N1v3_v3）")
            p.add_argument("--block", type=int, default=10, help="交互にする単位の本数（種の塊）")
            p.add_argument("--dry-run", action="store_true")
        p.add_argument("--gate1", default=None, help="関門 1 の判定の JSON（98_s4_gate1.py --out）")
        p.add_argument("--v3-goal", default=None, choices=("EH", "ES", "none"), help="戻し先を引数で上書きする（記録に残る）")
        p.add_argument("--v3-off", default="", help="v3 の部分を切る（a,b,c,d のカンマ区切り。診断だけ）")
        p.add_argument("--experiment", default="S4B1")
        p.add_argument("--trials", default=None, help="既定は腕の帯（本番は変えない）。smoke だけ --allow-smoke と一緒に")
        p.add_argument("--allow-smoke", action="store_true")
    return ap


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    a, extra = build_parser().parse_known_args(argv)
    try:
        if a.cmd == "plan":
            return cmd_plan(a)
        return (cmd_run if a.cmd == "run" else cmd_rotate)(a, extra)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
