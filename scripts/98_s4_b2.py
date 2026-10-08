"""段階 4 束 2 の B2（RTC の設定の確認）。関門 R の上位の設定と naive を、自然の試行の延長と把持失敗（P1）で回す。

使い方（作業場所 C:\\PAI\\recovery_vla。96_s4_resume.py run の続きの引数（--max-new・--accept-env-change など）は後ろに付ける）:
    .venv\\Scripts\\python.exe scripts\\98_s4_b2.py plan --gate1 outputs\\s4\\gate1_result.json           # 設定・帯・コマンド（読むだけ）
    .venv\\Scripts\\python.exe scripts\\98_s4_b2.py run --setting naive --part ext --gate1 outputs\\s4\\gate1_result.json --dry-run
    .venv\\Scripts\\python.exe scripts\\98_s4_b2.py rotate --gate1 outputs\\s4\\gate1_result.json [--dry-run]
        設定（naive と上位 0〜2 個）x 部分（ext・P1）を、同じ種で塊ごとに交互に回す（目標書_段階4.md 第 4 節。ext は 2 種 = 6 試行、
        P1 は 5 試行ごと。--block-ext・--block-p1 で変えられる）。同じコマンドで続きから回る。1 つのプロセス（模型は R1v3 の 1 つ）。
        続けるのは 96 が --max-new で止まったときだけ（Ctrl+C・メモリ待ちの時間切れ・止める合図・エラーでは止まる）。1 巡して完全な
        試行が増えなければエラー。進み具合は outputs\\s4\\b2\\rotate_<実験>.progress.json（pid 付き、1 分ごとに心拍。
        96_s4_ops.py wait --progress で見る）。止める合図: <条件>\\STOP、outputs\\s4\\STOP、outputs\\s4\\b2\\rotate_<実験>.STOP。
  部分（条件名 = <設定>_<部分>、実験名の既定 S4B2）:
    ext  自然の試行の延長: --trials natural:191200:23（69 試行。bundle2_rtc_ext）。束 1 の D-RTC の 30 試行（S4DRTC\\<設定>、
         190200〜190209）と合わせて 99 試行にする（合わせるのは 98_s4_b_decide.py b2）
    P1   把持失敗の誘発: --trials induced:191300:50 --induce P1（50 試行。bundle2_p1）
  設定: 関門 1 の判定の JSON（98_s4_gate1.py の --out。conclusion.bundle2.rtc）の settings。b2 が偽なら B2 は回さない（終了コード 0、
    何もしない）。判定が "undetermined" なら止める（終了コード 3）。--settings で上書きでき、上書きは試行の json の b2.settings_source
    と resume_spec.json に残る。naive は比べる相手として必ず入る。
  帯（configs/s4_gates.json の allocations と完全に一致する指定だけ）。--allow-smoke のときだけ smoke・データの帯 44400〜44799 も通す
    （X2 の生成の帯 44404〜44423 は拒む。実験名は S4SMOKE で始める）。
  実行のしかた（D-RTC と同じ）: R1v3、行動の区切り 6 行、安全フィルタなし、単発の試行の制限時間 60 s（30 s の採点が主）、試行ごとに
    世界を作り直す。RTC の設定の当て方は scripts\\98_s4_d_rtc.py の patch96 をそのまま使う（diag_NNNN.npz も同じ形で書く）。
読むもの: scripts\\98_s4_d_rtc.py・96_s4_resume.py・98_s4_b1.py（rotate の共通部。importlib。書き換えない）、src\\recovla\\diag\\rtc.py、
  configs\\s4_gates.json、--gate1 の JSON。
書くもの: 96_s4_resume.py run と 98_s4_d_rtc.py run と同じ記録（outputs\\v2eval\\<実験>\\<設定>_<部分>\\）。試行の json に "b2"
  （設定・部分・設定の出どころ・使ったファイルの SHA-256）を足し、P1 の試行は diag.induce・diag.start_state を誘発の値に直す。
  run.json・resume_spec.json に "b2" を足す。rotate の進み具合は outputs\\s4\\b2\\rotate_<実験>.progress.json・rotate_<実験>.log.json。
終了コード: 96_s4_resume.py と同じ（0 全部そろった、1 途中で止まった、2 エラー、3 引数・前提の食い違い）。
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
GATES = ROOT / "configs" / "s4_gates.json"
OUTD = ROOT / "outputs" / "s4" / "b2"
PARTS = {"ext": {"alloc": "bundle2_rtc_ext", "kind": "natural", "induce": None, "ja": "自然の試行の延長（D-RTC の 30 試行に足して 99）"},
         "P1": {"alloc": "bundle2_p1", "kind": "induced", "induce": "P1", "ja": "把持失敗の誘発"}}
FILES = ("scripts/98_s4_b2.py", "scripts/98_s4_b1.py", "scripts/98_s4_d_rtc.py", "scripts/96_s4_resume.py", "src/recovla/diag/rtc.py",
         "configs/s4_gates.json")


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
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


def drtc():
    return _load(ROOT / "scripts" / "98_s4_d_rtc.py", "s4_d_rtc_for_b2")


# ---------------------------------------------------------------- 帯
def default_trials(part: str) -> str:
    g = json.loads(GATES.read_text(encoding="utf-8"))
    lo, hi = next(a["range"] for a in g["bands"]["allocations"] if a["id"] == PARTS[part]["alloc"])
    return f"{PARTS[part]['kind']}:{lo}:{hi - lo + 1}"


def check_band(part: str, trials: str, experiment: str, allow_smoke: bool) -> str:
    """"" なら通す。本番は割り当てと完全に一致する指定だけ（種類も）。smoke は --allow-smoke・S4SMOKE・X2 の帯の外。"""
    want = default_trials(part)
    if trials == want:
        return ""
    kind, base, n = trials.split(":")
    if kind != PARTS[part]["kind"]:
        return f"{part} の試行の種類は {PARTS[part]['kind']}（渡された {trials}）"
    if not allow_smoke:
        return f"{part} の本番の帯は {want} と完全に一致する指定だけ（渡された {trials}）。smoke は --allow-smoke"
    g = json.loads(GATES.read_text(encoding="utf-8"))
    sm = next(x for x in g["bands"]["classes"] if x["id"] == "smoke_data")["range"]
    x2 = next(a["range"] for a in g["bands"]["allocations"] if a["id"] == "X2_gen")
    seeds = range(int(base), int(base) + int(n))
    if not experiment.startswith("S4SMOKE"):
        return f"smoke の実験名は S4SMOKE で始める（{experiment}）"
    if not all(sm[0] <= s <= sm[1] for s in seeds):
        return f"種 {seeds[0]}〜{seeds[-1]} が smoke の帯 {sm} に収まらない"
    hit = [s for s in seeds if x2[0] <= s <= x2[1]]
    if hit:
        return f"種 {hit[0]}〜{hit[-1]} は X2 の生成の帯 {x2}（予約）なので smoke でも使わない"
    return ""


# ---------------------------------------------------------------- 関門 1 の判定から設定
def settings_from_gate1(path, override: str = None) -> tuple:
    """(回す設定の列（naive を先頭に）, 出どころ, B2 を回すか)。"""
    D = drtc().D
    src = {"gate1": None, "override": None}
    chosen = None
    run_b2 = True
    if path:
        res = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        rtc = res["conclusion"]["bundle2"]["rtc"]
        src["gate1"] = {"path": str(path), "sha256": sha256_file(path), "rtc": rtc}
        if rtc.get("result") == "undetermined" and override is None:
            raise SystemExit("関門 1 の判定で B2 の設定が判定できない（undetermined）。判定を待つか --settings で上書きする")
        if rtc.get("result") != "undetermined":
            run_b2 = bool(rtc.get("b2"))
            chosen = list(rtc.get("settings") or []) if run_b2 else []
    elif override is None:
        raise SystemExit("--gate1（関門 1 の判定の JSON）か --settings が要る")
    if override is not None:
        new = [x for x in override.split(",") if x]
        src["override"] = {"settings": new, "from_gate1": chosen, "note": "引数で上書きした（記録に残す）"}
        chosen, run_b2 = new, bool(new)
    for s in chosen or []:
        if s not in D.CANDIDATES:
            raise SystemExit(f"B2 の設定は関門 R の候補 {list(D.CANDIDATES)} のどれか: {s}")
    if len(chosen or []) > 2:
        raise SystemExit(f"B2 の設定は 2 つまで（関門 R の上位 2 つ）: {chosen}")
    return (["naive"] + list(chosen or [])) if run_b2 else [], src, run_b2


# ---------------------------------------------------------------- 96 を包む（98_s4_d_rtc.py の patch96 の上に）
def patch(m96, setting: str, part: str, info: dict) -> None:
    d = drtc()
    d.patch96(m96, setting, False, sha256_file(ROOT / "scripts" / "98_s4_d_rtc.py"), sha256_file(pathlib.Path(d.D.__file__)))
    Base, base_spec, base_text = m96.Engine, m96.make_spec, m96.run_json_text
    p1 = PARTS[part]["induce"] is not None

    class B2Engine(Base):
        def run_one(self, i, seed, lay, tgt):
            meta, arrays, rlog = super().run_one(i, seed, lay, tgt)
            if p1:                                              # patch96 は自然の試行の値を書くので、誘発の値に直す
                meta["diag"]["induce"] = (meta.get("induce") or {}).get("kind")
                meta["diag"]["start_state"] = "induced（41_results.trial_list の induced。配置と目標は種から決まる）"
            meta["b2"] = dict(info)
            return meta, arrays, rlog

    def make_spec(a):
        s = base_spec(a)
        s["b2"] = {"setting": setting, "part": part, "settings_source": info["settings_source"]}
        return s

    def run_json_text(a_, kind, rows, *x, **kw):
        dd = json.loads(base_text(a_, kind, rows, *x, **kw))
        dd["b2"] = dict(info)
        return json.dumps(dd, ensure_ascii=False, indent=1)

    m96.Engine, m96.make_spec, m96.run_json_text = B2Engine, make_spec, run_json_text


FORBIDDEN_EXTRA = ("--model", "--mode", "--exec-interval", "--induce", "--ablate", "--time-limit-s", "--reuse-world", "--grip-gate",
                   "--trials", "--experiment", "--condition", "--no-safety")


def run_condition(m96, ops, v82, a, setting: str, part: str, src: dict, extra: list, max_new: int = None) -> int:
    for f in extra:
        if f.split("=")[0] in FORBIDDEN_EXTRA:
            raise SystemExit(f"{f} は 98_s4_b2.py が決める（R1v3・6 行・安全フィルタなし・60 s・帯・誘発）")
    D = drtc().D
    trials = (a.trials_ext if part == "ext" else a.trials_p1) or default_trials(part)
    why = check_band(part, trials, a.experiment, a.allow_smoke)
    if why:
        raise SystemExit(why)
    cond = f"{setting}_{part}"
    info = {"script": "98_s4_b2.py", "setting": setting, "setting_def": D.setting(setting), "part": part,
            "part_ja": PARTS[part]["ja"], "settings_source": src, "files_sha256": {p: sha256_file(ROOT / p) for p in FILES}}
    argv = ["run", "--experiment", a.experiment, "--condition", cond, "--model", "R1v3", "--trials", trials,
            "--mode", D.setting(setting)["mode"], "--exec-interval", "6", "--no-safety"]
    if PARTS[part]["induce"]:
        argv += ["--induce", PARTS[part]["induce"]]
    if max_new:
        argv += ["--max-new", str(max_new)]
    patch(m96, setting, part, info)
    a96 = m96.build_parser().parse_args(argv + list(extra))
    print(f"[b2] {cond}: {D.SETTINGS[setting]['ja']}・{PARTS[part]['ja']}（{a.experiment}\\{cond} {trials}）", flush=True)
    try:
        return m96.cmd_main(a96, v82, ops)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


def _env():
    m96 = drtc().load96()
    return m96, m96.load_ops(), m96.load_82(False)


def cmd_run(a, extra) -> int:
    settings, src, run_b2 = settings_from_gate1(a.gate1, a.settings)
    if not run_b2:
        print("[b2] 関門 R で条件を満たす設定がない（B2 は回さない。R の「やめる枝」）", flush=True)
        return 0
    if a.setting not in settings:
        raise SystemExit(f"--setting {a.setting} は今回の B2 の設定 {settings} にない")
    m96, ops, v82 = _env()
    return run_condition(m96, ops, v82, a, a.setting, a.part, src, extra)


def b1mod():
    """rotate の共通部（RotateProgress・rotate_loop・refuse_if_live・read_progress）は 98_s4_b1.py のものを使う。"""
    return _load(ROOT / "scripts" / "98_s4_b1.py", "s4_b1_for_b2")


def cmd_rotate(a, extra) -> int:
    """設定 x 部分を、同じ種で塊ごとに交互に回す（1 つのプロセス。模型は R1v3 の 1 つだけで、RTC の設定は条件ごとに入れ直す
    ＝98_s4_d_rtc.py の rotate と同じ）。続けるのは 96 が --max-new で止まったときだけ（98_s4_b1.py の rotate_loop）。"""
    settings, src, run_b2 = settings_from_gate1(a.gate1, a.settings)
    if not run_b2:
        print("[b2] 関門 R で条件を満たす設定がない（B2 は回さない。R の「やめる枝」）", flush=True)
        return 0
    m96, ops, v82 = _env()
    conds = [(p, s) for p in PARTS for s in settings]
    if a.dry_run:
        rc = 0
        for p, s in conds:
            rc = max(rc, run_condition(m96, ops, v82, a, s, p, src, extra + ["--dry-run"]))
        return rc
    B1 = b1mod()
    for f in extra:
        if f.split("=")[0] in B1.ROTATE_FORBIDDEN + FORBIDDEN_EXTRA:
            raise SystemExit(f"{f} は rotate では渡せない（rotate・98_s4_b2.py が決める）")
    block = {"ext": a.block_ext, "P1": a.block_p1}
    trials = {p: (a.trials_ext if p == "ext" else a.trials_p1) or default_trials(p) for p in PARTS}
    for p in PARTS:
        why = check_band(p, trials[p], a.experiment, a.allow_smoke)
        if why:
            raise SystemExit(why)
    plan = {p: [(seed, tgt) for seed, _, tgt in v82.trial_list(trials[p])] for p in PARTS}
    name = {f"{s}_{p}": (p, s) for p, s in conds}
    OUTD.mkdir(parents=True, exist_ok=True)
    prog_p, log_p = OUTD / f"rotate_{a.experiment}.progress.json", OUTD / f"rotate_{a.experiment}.log.json"
    B1.refuse_if_live(prog_p, ops)
    d = drtc()
    shas = (sha256_file(ROOT / "scripts" / "98_s4_d_rtc.py"), sha256_file(pathlib.Path(d.D.__file__)))

    def count(cond):
        p, s = name[cond]
        d.patch96(m96, s, False, *shas)                  # check_complete は diag の腕も照らすので、設定ごとに包み直す
        out = v82.OUT / a.experiment / cond
        return sum(int(m96.check_complete("run", out, i, {"seed": seed, "target": tgt})[0]) for i, (seed, tgt) in enumerate(plan[p]))

    def call(cond):
        p, s = name[cond]
        c = run_condition(m96, ops, v82, a, s, p, src, extra, max_new=block[p])
        return c, B1.read_progress(v82.OUT / a.experiment / cond / "progress.json")

    def stop_check(cond):
        for stop in (ROOT / "outputs" / "v2eval" / a.experiment / cond / "STOP", ROOT / "outputs" / "s4" / "STOP",
                     OUTD / f"rotate_{a.experiment}.STOP"):
            if stop.is_file():
                return f"stop_file:{stop}"
        return ""

    prog = B1.RotateProgress(prog_p, {"experiment": a.experiment, "condition": f"rotate:{a.experiment}", "conditions": list(name),
                                      "settings": settings, "trials_spec": trials, "total": sum(len(plan[p]) for p, _ in conds),
                                      "block_trials": block, "isolation": "in_process（模型は R1v3 の 1 つ）",
                                      "child_progress": {c: str(v82.OUT / a.experiment / c / "progress.json") for c in name},
                                      "stop_file": str(OUTD / f"rotate_{a.experiment}.STOP")}, m96._write_atomic)
    return B1.rotate_loop(list(name), {c: len(plan[name[c][0]]) for c in name}, count, call, prog, log_p, stop_check)


def cmd_plan(a) -> int:
    settings, src, run_b2 = settings_from_gate1(a.gate1, a.settings)
    out = {"run_b2": run_b2, "settings": settings, "settings_source": src,
           "parts": {p: dict(v, trials=default_trials(p)) for p, v in PARTS.items()},
           "conditions": [f"{s}_{p}" for p in PARTS for s in settings],
           "trials_total": len(settings) * (69 + 50),
           "combine_with": "自然の 99 試行 = S4DRTC\\<設定>（束 1 の D-RTC、natural:190200:10 の 30 試行）＋ S4B2\\<設定>_ext（69 試行）"}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "rotate", "plan"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("--setting", required=True)
            p.add_argument("--part", required=True, choices=list(PARTS))
        if name == "rotate":
            p.add_argument("--block-ext", type=int, default=6, help="ext を交互にする単位の試行の数（2 種 = 6 試行）")
            p.add_argument("--block-p1", type=int, default=5, help="P1 を交互にする単位の試行の数")
            p.add_argument("--dry-run", action="store_true")
        p.add_argument("--gate1", default=None, help="関門 1 の判定の JSON（98_s4_gate1.py --out）")
        p.add_argument("--settings", default=None, help="設定を引数で上書きする（カンマ区切り。記録に残る）")
        p.add_argument("--experiment", default="S4B2")
        p.add_argument("--trials-ext", default=None, help="既定は帯の割り当て（本番は変えない）")
        p.add_argument("--trials-p1", default=None, help="既定は帯の割り当て（本番は変えない）")
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
