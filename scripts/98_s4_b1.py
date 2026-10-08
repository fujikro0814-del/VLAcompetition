"""段階 4 束 2 の B1（実行器 v3 の確認）。3 個の連続タスクを、96_s4_resume.py task の包みで、今の実行器と v3 を同じ種で回す。

使い方（作業場所 C:\\PAI\\recovery_vla。96_s4_resume.py task の続きの引数（--max-new・--accept-env-change など）は後ろに付ける）:
    .venv\\Scripts\\python.exe scripts\\98_s4_b1.py plan --gate1 outputs\\s4\\gate1_result.json          # 腕・帯・戻し先・コマンド（読むだけ）
    .venv\\Scripts\\python.exe scripts\\98_s4_b1.py run --arm R1v3_v3 --gate1 outputs\\s4\\gate1_result.json --dry-run
    .venv\\Scripts\\python.exe scripts\\98_s4_b1.py rotate --gate1 outputs\\s4\\gate1_result.json [--block 10] [--in-process] [--dry-run]
        3 つの腕（R1v3_cur・R1v3_v3・N1v3_v3）を、同じ種で 10 本の塊ごとに交互に回す（目標書_段階4.md 第 4 節）。止まって再開しても
        同じコマンドで続きから回る（96 が完全な記録を飛ばす）。
        既定は、腕の 1 回の呼び出し（--max-new <block> の 96）ごとに子プロセスを起こす。R1v3 と N1v3 の 2 つの模型を 1 つのプロセスに
        同時に持たないので、メモリの最大は模型 1 つ分（手順書 第 8 節）。--in-process なら 1 つのプロセスで回し、模型が替わるときに
        前の模型を手放す（gc と torch.cuda.empty_cache。解放は保証しない）。
        続けるのは子が --max-new で止まったとき（progress.json の stop_reason が max_new:...）だけ。Ctrl+C・メモリ待ちの時間切れ・
        止める合図・エラーでは止まる。1 巡して完全な試行が 1 本も増えなければエラーで止まる。
        進み具合は outputs\\s4\\b1\\rotate_<実験>.progress.json（pid 付き。96_s4_ops.py wait --progress で見る。1 分ごとに心拍）と
        rotate_<実験>.log.json（呼び出しの並び）。止める合図: <腕>\\STOP、outputs\\s4\\STOP、outputs\\s4\\b1\\rotate_<実験>.STOP
        （今の呼び出しが終わってから止まる）。
  腕（条件名 = 腕の名前、実験名の既定 S4B1）:
    R1v3_cur  R1v3 ＋ 今の実行器（runtime/executor.py）。帯 191100〜191159 の 60 本（--trials 191100:60）
    R1v3_v3   R1v3 ＋ 実行器 v3（runtime/executor_v3.py）。同じ 60 本
    N1v3_v3   N1v3 ＋ 実行器 v3。帯 191100〜191129 の 30 本（--trials 191100:30。N1v3 の腕＝復帰デモの効果と取り違えさせない）
  v3 の戻し先（(a)）: 関門 1 の判定の JSON（98_s4_gate1.py の --out。conclusion.bundle2.v3a）から読む。include が偽なら (a) なし。
    判定が "undetermined" なら止める（終了コード 3）。--v3-goal EH|ES|none で上書きでき、上書きは試行の json の b1.v3_source と
    run.json・resume_spec.json に残る（上書きしたら掲示板に掲示する。手順書 第 11 節）。--v3-off a,b,c,d で v3 の部分を個別に切れる
    （切り分けの診断。本番では使わない）。
  帯（configs/s4_gates.json の allocations と完全に一致する指定だけ）: R1v3 は bundle2_confirm の 191100〜191159 そのもの、
    N1v3 は目標書 4-2 の「N1v3 は 191100〜191129」そのもの。--allow-smoke のときだけ smoke・データの帯 44400〜44799 も通す
    （ただし X2 の生成の帯 44404〜44423 は拒む。実験名は S4SMOKE で始める）。
  実行のしかた（段階 3 の E7・D-E7 と同じ）: 行動の区切り 6 行（--exec-interval 6）、安全フィルタなし（--no-safety）、
    1 手順の試み 30 s・やり直し 1 回・全体 200 s、計画役 s4（Haiku 5.5）、試行ごとに世界を作り直す。違う値を渡すと止める。
  ■ 鍵の注意: 試すときは必ず --dry-run を付ける（付けないと計画役の API を呼び、帯の種を使う）。
読むもの: scripts\\96_s4_resume.py（importlib。Engine・SPEC_KEYS・run_json_text を読み込んだ写しの上で包む。書き換えない）、
  configs\\s4_gates.json（帯）、--gate1 の JSON、src\\recovla\\runtime\\executor_v3.py。
書くもの: 96_s4_resume.py task と同じ記録（outputs\\v2eval\\<実験>\\<腕>\\）。試行の json に "v3"（版・設定・出来事・介入の種類別の数）と
  "b1"（腕・モデル・実行器の版・戻し先とその出どころ・使ったファイルの SHA-256・API の回し直し）を、run.json に "b1" を足す。
  rotate の進み具合は outputs\\s4\\b1\\rotate_<実験>.progress.json・rotate_<実験>.log.json。
終了コード: 96_s4_resume.py と同じ（0 全部そろった、1 途中で止まった、2 エラー、3 引数・前提の食い違い）。
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
GATES = ROOT / "configs" / "s4_gates.json"
OUTD = ROOT / "outputs" / "s4" / "b1"
ARMS = {
    "R1v3_cur": {"model": "R1v3", "executor": "current", "trials": "191100:60", "ja": "R1v3 ＋ 今の実行器"},
    "R1v3_v3": {"model": "R1v3", "executor": "v3", "trials": "191100:60", "ja": "R1v3 ＋ 実行器 v3"},
    "N1v3_v3": {"model": "N1v3", "executor": "v3", "trials": "191100:30", "ja": "N1v3 ＋ 実行器 v3（復帰デモなしの腕）"},
}
ROTATE_ORDER = ("R1v3_cur", "R1v3_v3", "N1v3_v3")
CURRENT_VERSION = "current"               # 今の実行器（runtime/executor.py、凍結）の版の名前（meta.b1.executor_version）
FILES = ("src/recovla/runtime/executor_v3.py", "src/recovla/runtime/executor.py", "scripts/98_s4_b1.py", "scripts/96_s4_resume.py",
         "configs/s4_gates.json")
V3_PARTS = {"a": "a_goal", "b": "b_placed_fix", "c": "c_final_wait", "d": "d_early_abort"}
SHARED_CACHE = {}                         # --in-process の rotate: 方策（GPU の模型）をモデルごとに持つ（模型が替わるときに手放す）


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
        src["override"] = {"v3_goal": override, "from_gate1": goal, "note": "引数で上書きした（記録に残す。掲示板に掲示する）"}
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
    r96.SPEC_KEYS = orig["SPEC_KEYS"] + ("b1_arm", "b1_v3", "b1_executor_sha256", "b1_executor_version")
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
            self.b1_last = None                              # 試行ごとに空にする（前の試行の実行器の記録を写さない）
            try:
                meta, arrays, rlog = super().run_one_task(i, seed)
            except Exception as e:                           # noqa: BLE001
                if not any(c.__module__.split(".")[0] == "anthropic" for c in type(e).__mro__):
                    raise
                retry = {"n": 1, "error": f"{type(e).__name__}: {e}"[:500], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                print(f"[b1] 試行 {i} 種 {seed}: 計画役の API が失敗したので同じ種で 1 回だけ回し直す（{retry['error']}）", flush=True)
                self.b1_last = None
                meta, arrays, rlog = super().run_one_task(i, seed)
            if use_v3:
                if not isinstance(self.b1_last, V3.V3TaskRuntime):
                    raise RuntimeError(f"試行 {i}: この試行の実行器 v3 の記録がない（make が呼ばれていない）")
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
ROTATE_FORBIDDEN = ("--max-new", "--progress-file", "--stop-file")      # rotate が決める（塊の本数・子の progress.json・止める合図）


def run_arm(r96, ops, v82, a, arm: str, extra: list, max_new: int = None) -> int:
    from recovla.runtime import executor_v3 as V3
    for f in extra:
        if f.split("=")[0] in FORBIDDEN_EXTRA:
            raise SystemExit(f"{f} は 98_s4_b1.py が決める（モデル・帯・6 行・安全フィルタなし・計画役 s4・30 s・200 s）")
    spec = ARMS[arm]
    trials = a.trials or spec["trials"]
    why = check_band(arm, trials, a.experiment, a.allow_smoke)
    if why:
        raise SystemExit(why)
    v3, src = v3_from_gate1(a.gate1, a.v3_goal, a.v3_off)
    is_v3 = spec["executor"] == "v3"
    info = {"script": "98_s4_b1.py", "arm": arm, "arm_ja": spec["ja"], "model": spec["model"], "executor": spec["executor"],
            "executor_version": V3.VERSION if is_v3 else CURRENT_VERSION,
            "v3": v3 if is_v3 else None, "v3_source": src if is_v3 else None,
            "files_sha256": {p: sha256_file(ROOT / p) for p in FILES}}
    argv = ["task", "--experiment", a.experiment, "--condition", arm, "--model", spec["model"], "--trials", trials,
            "--exec-interval", "6", "--no-safety"]
    if max_new:
        argv += ["--max-new", str(max_new)]
    a96 = r96.build_parser().parse_args(argv + list(extra))
    a96.b1_arm = arm
    a96.b1_v3 = json.dumps(info["v3"], sort_keys=True)
    a96.b1_executor_sha256 = info["files_sha256"]["src/recovla/runtime/executor_v3.py"] if is_v3 else None   # v3 の腕だけ固定する
    a96.b1_executor_version = info["executor_version"]
    patch96(r96, arm, v3, info)
    if is_v3 and src.get("override"):
        print(f"[b1] 注意: 関門 1 の判定を --v3-goal で上書きした（{src['override']}）。回すなら掲示板に掲示する", flush=True)
    print(f"[b1] {arm}: {spec['ja']}（{a.experiment}\\{arm} {trials}、実行器 {info['executor_version']}、v3 {info['v3']}）", flush=True)
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
    for p in (ROOT / "outputs" / "v2eval" / a.experiment / arm / "STOP", ROOT / "outputs" / "s4" / "STOP",
              OUTD / f"rotate_{a.experiment}.STOP"):
        if p.is_file():
            return f"stop_file:{p}"
    return ""


# ---------------------------------------------------------------- rotate の共通部（98_s4_b2.py も使う）
class RotateProgress:
    """rotate の progress.json（98_s4_d_rtc.py の rotate と同じ形。pid・proc_create_time・status・done・total・calls）。
    心拍の別スレッドが every_s ごとに updated を書く（塊 1 つが 30 分を超えても 96_s4_ops.py wait が「止まった」とみなさない）。"""

    def __init__(self, path: pathlib.Path, base: dict, write):
        self.path, self.write = path, write
        self.lock = threading.RLock()
        self.evt = threading.Event()
        self.th = None
        try:
            import psutil
            ct = psutil.Process().create_time()
        except Exception:                                # noqa: BLE001
            ct = None
        self.d = dict({"pid": os.getpid(), "proc_create_time": ct, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "status": "running", "done": 0, "successes": None, "calls": []}, **base)

    def put(self, **kw) -> None:
        with self.lock:
            self.d.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
            self.write(self.path, json.dumps(self.d, ensure_ascii=False, indent=1, default=str))

    def start(self, every_s: float = 60.0) -> None:
        def beat():
            while not self.evt.wait(every_s):
                self.put()
        self.th = threading.Thread(target=beat, daemon=True)
        self.th.start()

    def stop(self) -> None:
        self.evt.set()
        if self.th is not None:
            self.th.join(timeout=5)


def refuse_if_live(path: pathlib.Path, ops) -> None:
    """同じ rotate を 2 つのプロセスで回さない（前の progress.json の pid が生きていて、状態が最終でなければ止める）。"""
    if not path.is_file():
        return
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return
    if old.get("status") in ops.LIVE_STATUS and old.get("pid") and old.get("pid") != os.getpid() \
            and ops._alive(int(old["pid"]), old.get("proc_create_time")):
        raise SystemExit(f"{path}: pid {old['pid']} がこの rotate をまだ回している（status={old.get('status')}）。二重に回さない")


def read_progress(p: pathlib.Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def rotate_loop(names: list, totals: dict, count, call, prog: RotateProgress, log_path: pathlib.Path, stop_check) -> int:
    """names を順に、1 回の呼び出しで塊 1 つずつ回す（98_s4_d_rtc.py の rotate と同じ決まり）。
    count(名前) = 完全な試行の数、call(名前) = (終了コード, 子の progress.json の中身)、stop_check(名前) = 止める合図（"" ならなし）。
    続けるのは、終了コード 0 か、1 で stop_reason が max_new:... のときだけ。1 巡して完全な試行が増えなければエラー。"""
    status, code = "done", 0
    last = None
    prog.start()
    prog.put(status="running")
    try:
        while True:
            counts = {n: count(n) for n in names}
            prog.put(done=sum(counts.values()), counts=counts)
            left = [n for n in names if counts[n] < totals[n]]
            if not left:
                break
            if last == counts:
                raise RuntimeError(f"1 巡しても完全な試行が増えない: {counts}")
            last = dict(counts)                          # 写し（counts は下で呼び出しごとに更新する）
            for n in left:
                s = stop_check(n)
                if s:
                    status, code = "stopped", 1
                    prog.put(stop_reason=s)
                    print(f"[rotate] 止める合図 {s}", flush=True)
                    return code
                t0 = time.time()
                prog.put(current=n, current_started=time.strftime("%Y-%m-%d %H:%M:%S"))
                c, pr = call(n)
                with prog.lock:
                    prog.d["calls"].append({"name": n, "code": c, "status": pr.get("status"), "stop_reason": pr.get("stop_reason"),
                                            "done": pr.get("done"), "wall_s": round(time.time() - t0, 1)})
                    prog.write(log_path, json.dumps(prog.d["calls"], ensure_ascii=False, indent=1))
                counts[n] = count(n)
                prog.put(done=sum(counts.values()), counts=counts)
                reason = str(pr.get("stop_reason") or "")
                if c == 0 or (c == 1 and reason.startswith("max_new")):
                    continue
                status, code = ("stopped", 1) if c == 1 else ("error", c if c in (2, 3) else 2)
                prog.put(stop_reason=f"{n}: {reason or pr.get('error') or f'終了コード {c}'}")
                return code
    except KeyboardInterrupt:
        status, code = "interrupted", 1
        prog.put(stop_reason="KeyboardInterrupt")
    except BaseException as e:                           # noqa: BLE001
        status, code = "error", 2
        prog.put(error=f"{type(e).__name__}: {e}")
        raise
    finally:
        prog.stop()
        prog.put(status=status)
    return code


# ---------------------------------------------------------------- B1 の rotate
def release_models(keep: str) -> list:
    """--in-process: keep 以外の模型を手放す（Engine が持つ辞書も空にする）。手放したモデルの名前の列。"""
    gone = [k for k in list(SHARED_CACHE) if k != keep]
    for k in gone:
        SHARED_CACHE.pop(k).clear()
    if gone:
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[b1] 模型を手放した: {gone}（次は {keep}）", flush=True)
    return gone


def child_argv(a, arm: str, extra: list, block: int) -> list:
    """子プロセスで腕を 1 回（--max-new block）回すコマンド。"""
    argv = [sys.executable, str(pathlib.Path(__file__).resolve()), "run", "--arm", arm, "--experiment", a.experiment]
    if a.gate1:
        argv += ["--gate1", str(a.gate1)]
    if a.v3_goal:
        argv += ["--v3-goal", a.v3_goal]
    if a.v3_off:
        argv += ["--v3-off", a.v3_off]
    if a.trials:
        argv += ["--trials", a.trials]
    if a.allow_smoke:
        argv.append("--allow-smoke")
    return argv + list(extra) + ["--max-new", str(int(block))]


def run_child(argv: list) -> int:
    p = subprocess.Popen(argv, cwd=str(ROOT))
    try:
        return p.wait()
    except KeyboardInterrupt:                           # 子にも Ctrl+C が届く。子が記録を閉じるのを待つ
        try:
            p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            p.kill()
        raise


def cmd_rotate(a, extra) -> int:
    """腕を、同じ種で --block 本の塊ごとに交互に回す。既定は呼び出しごとに子プロセス（模型は 1 つだけ）。"""
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
    for f in extra:
        if f.split("=")[0] in ROTATE_FORBIDDEN + FORBIDDEN_EXTRA:
            raise SystemExit(f"{f} は rotate では渡せない（rotate・98_s4_b1.py が決める）")
    for arm in arms:                                     # 子を起こす前に、帯と関門 1 の判定を確かめる
        why = check_band(arm, a.trials or ARMS[arm]["trials"], a.experiment, a.allow_smoke)
        if why:
            raise SystemExit(why)
    v3_from_gate1(a.gate1, a.v3_goal, a.v3_off)
    OUTD.mkdir(parents=True, exist_ok=True)
    prog_p, log_p = OUTD / f"rotate_{a.experiment}.progress.json", OUTD / f"rotate_{a.experiment}.log.json"
    refuse_if_live(prog_p, ops)
    seeds = {}
    for arm in arms:
        base, n = (int(x) for x in (a.trials or ARMS[arm]["trials"]).split(":"))
        seeds[arm] = list(range(base, base + n))
    totals = {arm: len(seeds[arm]) for arm in arms}

    def count(arm):
        out = v82.OUT / a.experiment / arm
        return sum(int(r96.check_complete("task", out, i, {"seed": s})[0]) for i, s in enumerate(seeds[arm]))

    def call(arm):
        if a.in_process:
            release_models(ARMS[arm]["model"])
            c = run_arm(r96, ops, v82, a, arm, extra, max_new=a.block)
        else:
            c = run_child(child_argv(a, arm, extra, a.block))
        return c, read_progress(v82.OUT / a.experiment / arm / "progress.json")

    prog = RotateProgress(prog_p, {"experiment": a.experiment, "condition": f"rotate:{a.experiment}", "arms": arms,
                                   "trials_spec": {arm: a.trials or ARMS[arm]["trials"] for arm in arms}, "total": sum(totals.values()),
                                   "block_trials": a.block, "isolation": "in_process" if a.in_process else "subprocess",
                                   "child_progress": {arm: str(v82.OUT / a.experiment / arm / "progress.json") for arm in arms},
                                   "stop_file": str(OUTD / f"rotate_{a.experiment}.STOP")}, r96._write_atomic)
    return rotate_loop(arms, totals, count, call, prog, log_p, lambda arm: stop_requested(a, arm))


def cmd_plan(a) -> int:
    v3, src = v3_from_gate1(a.gate1, a.v3_goal, a.v3_off)
    out = {"arms": ARMS, "v3": v3, "v3_source": src, "bands": {k: list(v) for k, v in bands().items()},
           "trials_total": sum(int(x["trials"].split(":")[1]) for x in ARMS.values()),
           "commands": [f"scripts\\98_s4_b1.py run --arm {k} --gate1 {a.gate1} --experiment {a.experiment}" for k in ARMS] +
                       [f"scripts\\98_s4_b1.py rotate --gate1 {a.gate1} --experiment {a.experiment} --block 10"],
           "watch": f"scripts\\96_s4_ops.py wait --progress outputs\\s4\\b1\\rotate_{a.experiment}.progress.json"}
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
            p.add_argument("--in-process", action="store_true",
                           help="子プロセスを起こさず 1 つのプロセスで回す（模型が替わるときに前の模型を手放す）")
            p.add_argument("--dry-run", action="store_true")
        p.add_argument("--gate1", default=None, help="関門 1 の判定の JSON（98_s4_gate1.py --out）")
        p.add_argument("--v3-goal", default=None, choices=("EH", "ES", "none"), help="戻し先を引数で上書きする（記録に残る。掲示する）")
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
