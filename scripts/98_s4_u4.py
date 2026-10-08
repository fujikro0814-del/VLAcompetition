"""段階 4 束 6 (ii) の U4（LLM による立て直しの計画と日本語の状況報告）。3 個の連続タスクを、96_s4_resume.py task の包みで回す。

使い方（作業場所 C:\\PAI\\recovery_vla。run の引数は 96_s4_resume.py task と同じものを後ろに付ける）:
    .venv\\Scripts\\python.exe scripts\\98_s4_u4.py run --arm U4 --experiment S4U4 --condition U4 --model R1v3 --trials 191400:40 --dry-run
    .venv\\Scripts\\python.exe scripts\\98_s4_u4.py run --arm U0 --experiment S4U4 --condition U0_run1 --model R1v3 --trials 191400:40
    .venv\\Scripts\\python.exe scripts\\98_s4_u4.py summary --experiment S4U4 [--conditions U0_run1 U0_run2 U4] [--out ...]
  腕（--arm）: U0（今の実行器。立て直しなし。条件名 U0_run1・U0_run2 で同じ 40 種を 2 回＝束 1 の D-E7 の E0 と同じ扱い）、
    U4（runtime/executor_u4.py の包み＋planner/replan_s4.py）。手順書は docs/stage4/u4_protocol.md（結果を見る前に固定）。
  帯は 191400〜191439（s4_gates.json の bundle6_u4）だけ。ほかの帯の種は回さない（終了コード 3）。--allow-smoke を付けたときだけ
    smoke・データの帯（44400〜44799）も通す（動作の確かめ。評価の数字にしない）。
  実行のしかたは D-E7 と同じ: 行動の区切り 6 行（--exec-interval 6）、安全フィルタなし（--no-safety）。このスクリプトが既定で入れる
    （違う値を渡すと止める）。1 手順の試み 30 s・やり直し 1 回・全体 200 s・計画役 s4（Haiku 5.5）は 96 の既定のまま。
  U4 を --dry-run なしで回すときは ANTHROPIC_API_KEY が要る（無ければ止める。黙って「止まって知らせる」に倒れ続けるのを防ぐ）。
読むもの: scripts\\96_s4_resume.py（importlib。Engine・SPEC_KEYS・run_json_text を差し替えて cmd_main で回す。書き換えない）、
  configs\\s4_gates.json（帯）、summary は outputs\\v2eval\\<実験>\\<条件>\\ の記録。
書くもの: 96_s4_resume.py task と同じ（outputs\\v2eval\\<実験>\\<条件>\\ の run_NNNN.json・.npz・run_NNNN_runtime.json、run.json、
  G_AUDIT.json、progress.json、resume_spec.json、resume_log.json）。試行の json に "replans"（立て直しの記録）と "u4"（腕・介入の
  種類別の数・終わり方・使ったファイルの SHA-256・API の回し直し）、run.json に "u4" を足す。summary は outputs\\s4\\u4\\ に書く。
  計画役・立て直しの応答のキャッシュは outputs\\llm_cache\\。
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTD = ROOT / "outputs" / "s4" / "u4"
GATES = ROOT / "configs" / "s4_gates.json"
ALLOC = "bundle6_u4"
ARMS = {"U0": "今の実行器（立て直しなし）", "U4": "立て直しの計画（replan_s4、claude-haiku-5-5）"}
DEFAULT_CONDITIONS = ("U0_run1", "U0_run2", "U4")
FILES = ("src/recovla/planner/replan_s4.py", "src/recovla/runtime/executor_u4.py", "scripts/98_s4_u4.py",
         "scripts/96_s4_resume.py", "src/recovla/runtime/executor.py", "configs/s4_gates.json")
# 料金（米ドル / 100 万トークン。claude-haiku-5-5、入力 10 万トークン以下）。回す日に Anthropic の料金のページで確かめ、違えば
# summary の --price-in・--price-out で渡す（記録の usage から計算し直せる）
PRICE_PER_MTOK = {"claude-haiku-5-5": (0.10, 0.50)}


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


def band_check(seeds, allow_smoke: bool = False) -> str:
    """種が帯 bundle6_u4 に全部入っていれば ""（--allow-smoke のときは smoke・データの帯に全部入っていてもよい）、そうでなければ理由。"""
    g = json.loads(GATES.read_text(encoding="utf-8"))
    al = next(x for x in g["bands"]["allocations"] if x["id"] == ALLOC)["range"]
    sm = next(x for x in g["bands"]["classes"] if x["id"] == "smoke_data")["range"]
    seeds = list(seeds)
    if seeds and all(al[0] <= s <= al[1] for s in seeds):
        return ""
    if allow_smoke and seeds and all(sm[0] <= s <= sm[1] for s in seeds):
        return ""
    extra = f"にも smoke の帯 {sm}" if allow_smoke else "（smoke の帯は --allow-smoke のときだけ）"
    return f"種 {seeds[0] if seeds else '-'}〜{seeds[-1] if seeds else '-'} が帯 {ALLOC} {al} {extra}に収まらない"


def _is_api_error(e: BaseException) -> bool:
    return any(c.__module__.split(".")[0] == "anthropic" for c in type(e).__mro__)


def make_engine(r96, arm: str, info: dict):
    from recovla.planner import replan_s4 as R4
    from recovla.runtime import executor_u4 as U4

    class U4Engine(r96.Engine):
        """96 の Engine の task を、腕 U4 のときだけ立て直しの包みに差し替えた版（U0 は差し替えない）。"""

        def init_task(self):
            super().init_task()
            orig = self.make_task
            self.u4_last = None

            def make(io, setup):
                ex = orig(io, setup)
                if arm == "U4":
                    ex = U4.install(ex, R4.replan_step, max_replans=R4.MAX_REPLANS)
                self.u4_last = ex
                return ex
            self.make_task = make

        def run_one_task(self, i, seed):
            retry = None
            try:
                meta, arrays, rlog = super().run_one_task(i, seed)
            except Exception as e:                           # noqa: BLE001
                if not _is_api_error(e):
                    raise
                retry = {"n": 1, "error": f"{type(e).__name__}: {e}"[:500], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                print(f"[u4] 試行 {i} 種 {seed}: 計画役の API が失敗したので同じ種で 1 回だけ回し直す（{retry['error']}）", flush=True)
                meta, arrays, rlog = super().run_one_task(i, seed)
            ex = self.u4_last
            rec = ex.record() if ex is not None else {}
            meta["replans"] = rec.get("replans", [])
            meta["u4"] = dict(info, arm=arm, api_retry=retry, ended=rec.get("ended"),
                              interventions=rec.get("interventions", {"plan_change": 0, "replan_requests": 0, "llm_calls": 0}),
                              max_replans=R4.MAX_REPLANS if arm == "U4" else 0,
                              max_runs_per_color=R4.MAX_RUNS_PER_COLOR if arm == "U4" else None)
            return meta, arrays, rlog
    return U4Engine


def cmd_run(argv) -> int:
    ap = argparse.ArgumentParser(prog="98_s4_u4.py run", add_help=False)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--allow-smoke", action="store_true")
    mine, rest = ap.parse_known_args(argv)
    r96 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_u4")
    a = r96.build_parser().parse_args(["task"] + rest)
    arm = mine.arm
    if a.exec_interval not in (None, 6):
        print("U4 は D-E7 と同じ --exec-interval 6・--no-safety で回す（ほかの値は渡さない）", file=sys.stderr)
        return 3
    a.exec_interval, a.no_safety = 6, True
    if a.planner != "s4":
        print("U4 の計画役は s4（decompose_s4、claude-haiku-5-5）だけ", file=sys.stderr)
        return 3
    base, n = (int(x) for x in a.trials.split(":"))
    why = band_check(range(base, base + n), mine.allow_smoke)
    if why:
        print(why, file=sys.stderr)
        return 3
    if arm == "U4" and not a.dry_run:
        from recovla.planner import decompose as D
        if not D._api_key():
            print("U4 は ANTHROPIC_API_KEY が要る（無いと立て直しが全部「止まって知らせる」に倒れる）。--dry-run なら要らない",
                  file=sys.stderr)
            return 3
    info = {"script": "98_s4_u4.py", "condition": a.condition,
            "files_sha256": {p: sha256_file(ROOT / p) for p in FILES}}
    a.u4_arm = arm
    a.u4_replan_sha256 = info["files_sha256"]["src/recovla/planner/replan_s4.py"]
    a.u4_executor_sha256 = info["files_sha256"]["src/recovla/runtime/executor_u4.py"]
    r96.SPEC_KEYS = tuple(r96.SPEC_KEYS) + ("u4_arm", "u4_replan_sha256", "u4_executor_sha256")
    r96.Engine = make_engine(r96, arm, info)
    orig_text = r96.run_json_text

    def run_json_text(a_, kind, rows, *x, **kw):
        d = json.loads(orig_text(a_, kind, rows, *x, **kw))
        d["u4"] = dict(info, arm=arm, arm_ja=ARMS[arm])
        return json.dumps(d, ensure_ascii=False, indent=1)
    r96.run_json_text = run_json_text
    print(f"[u4] 腕 {arm}: {ARMS[arm]}（指示文「{a.text}」、--exec-interval 6・--no-safety、帯 {ALLOC}）", flush=True)
    ops = r96.load_ops()
    v82 = r96.load_82(a.allow_82_change)
    try:
        return r96.cmd_main(a, v82, ops)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


# ---------------------------------------------------------------- summary（記録だけを読む。判定はしない）
def truth_box_at(meta: dict, t: float) -> list:
    """時刻 t までに真値で箱に収まった色（task_loop の truth_success_t。採点の照らし合わせだけに使う）。"""
    ts = meta.get("truth_success_t") or {}
    return [c for c in ("red", "green", "blue") if c in ts and float(ts[c]) <= float(t) + 1e-9]


def llm_usage(meta: dict) -> dict:
    """計画役（起動時）と立て直しの呼び出し数・トークン。キャッシュから読んだ応答は呼び出しに数えない。"""
    out = {"decompose_calls": 0, "replan_calls": 0, "input_tokens": 0, "output_tokens": 0}
    plan = meta.get("plan") or {}
    if plan and not plan.get("from_cache") and plan.get("usage"):
        out["decompose_calls"] += 1
        out["input_tokens"] += int(plan["usage"]["input_tokens"])
        out["output_tokens"] += int(plan["usage"]["output_tokens"])
    for r in meta.get("replans") or []:
        llm = ((r.get("out") or {}).get("llm") or {})
        out["replan_calls"] += int(llm.get("calls", 0))
        for u in llm.get("usage") or []:
            out["input_tokens"] += int(u["input_tokens"])
            out["output_tokens"] += int(u["output_tokens"])
    return out


def grading_rows(meta: dict) -> list:
    """日本語の報告の採点の表（docs/stage4/u4_protocol.md 第 5 節の規則で、人が正誤を付ける）。真値の欄は採点のためだけ。"""
    rows = []
    for r in meta.get("replans") or []:
        out = r.get("out") or {}
        rows.append({"run": meta.get("run"), "seed": meta.get("seed"), "n": r.get("n"), "t_request": r.get("t_request"),
                     "failed_color": r.get("failed_color"), "action": r.get("action"),
                     "applied": (r.get("applied") or {}).get("kind"), "fallback": bool(out.get("fallback")),
                     "rejected": out.get("rejected"), "report": r.get("report"), "report_source": out.get("report_source"),
                     "report_rejected": out.get("report_rejected"),
                     "perceived_box": (r.get("perception") or {}).get("in_box"),
                     "truth_box_at_request": truth_box_at(meta, r.get("t_request") or 0.0),
                     "truth_final_in_box": [c for c, v in (meta.get("final_in_box") or {}).items() if v],
                     "grade": None, "grade_note": None})
    return rows


def summarize_condition(d: pathlib.Path, price) -> dict:
    metas = []
    for p in sorted(d.glob("run_[0-9][0-9][0-9][0-9].json")):
        metas.append(json.loads(p.read_text(encoding="utf-8")))
    n = len(metas)
    stored = [sum(bool(v) for v in (m.get("final_in_box") or {}).values()) for m in metas]
    use = [llm_usage(m) for m in metas]
    tok_in, tok_out = sum(u["input_tokens"] for u in use), sum(u["output_tokens"] for u in use)
    iv = [((m.get("u4") or {}).get("interventions") or {}) for m in metas]
    actions = {}
    for m in metas:
        for r in m.get("replans") or []:
            k = f"{r.get('action')}{'(fallback)' if (r.get('out') or {}).get('fallback') else ''}"
            actions[k] = actions.get(k, 0) + 1
    rows = [g for m in metas for g in grading_rows(m)]
    return {"n": n, "all_three_true": sum(bool(m.get("all_three_in_box")) for m in metas),
            "mean_stored_count": (sum(stored) / n) if n else None,
            "stopped": sum(1 for m in metas if m.get("stopped")), "ended_finish": sum(1 for m in metas if (m.get("u4") or {}).get("ended")),
            "timed_out": sum(1 for m in metas if m.get("timed_out")),
            "interventions": {"plan_change": sum(int(x.get("plan_change", 0)) for x in iv),
                              "replan_requests": sum(int(x.get("replan_requests", 0)) for x in iv),
                              "stop_notify": sum(1 for m in metas if m.get("stopped"))},
            "actions": actions,
            "llm": {"decompose_calls": sum(u["decompose_calls"] for u in use), "replan_calls": sum(u["replan_calls"] for u in use),
                    "input_tokens": tok_in, "output_tokens": tok_out,
                    "cost_usd": round(tok_in / 1e6 * price[0] + tok_out / 1e6 * price[1], 6), "price_per_mtok": list(price)},
            "g1_violations": sum(int(((m.get("audit") or {}).get("g1") or {}).get("violations", 0)) for m in metas),
            "reports": rows}


def cmd_summary(a) -> int:
    base = ROOT / "outputs" / "v2eval" / a.experiment
    price = (a.price_in, a.price_out) if a.price_in is not None else PRICE_PER_MTOK["claude-haiku-5-5"]
    summ = {}
    for c in a.conditions:
        if (base / c).is_dir() and any((base / c).glob("run_[0-9][0-9][0-9][0-9].json")):
            summ[c] = summarize_condition(base / c, price)
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": a.experiment, "note": "記述だけ（検定の族に入れない）",
           "conditions": summ}
    OUTD.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(a.out) if a.out else OUTD / f"summary_{a.experiment}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for c, s in summ.items():
        print(f"{c}: n={s['n']} 3 個とも {s['all_three_true']} 平均の収納数 {s['mean_stored_count']} 計画の変更 "
              f"{s['interventions']['plan_change']} LLM {s['llm']['decompose_calls']}+{s['llm']['replan_calls']} 回 "
              f"${s['llm']['cost_usd']}")
    print(f"[u4] 書いた: {p}")
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
    p = sub.add_parser("summary", help="記録から指標を出す（記述だけ。判定はしない）")
    p.add_argument("--experiment", required=True)
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    p.add_argument("--price-in", type=float, default=None, help="入力の料金（米ドル / 100 万トークン）")
    p.add_argument("--price-out", type=float, default=None, help="出力の料金（米ドル / 100 万トークン）")
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if (a.price_in is None) != (a.price_out is None):
        ap.error("--price-in と --price-out は両方渡す")
    return {"summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
