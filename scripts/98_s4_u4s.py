"""段階 4 束 6 (ii) の U4S（U4 と同じで、立て直しの計画役だけを claude-sonnet-5-5 にした腕）の入口。98_s4_u4.py を包む。

使い方（作業場所 C:\\PAI\\recovery_vla。run の引数は 96_s4_resume.py task と同じものを後ろに付ける）:
    .venv\\Scripts\\python.exe scripts\\98_s4_u4s.py run --experiment S4U4 --condition U4S --model R1v3 --trials 191400:100 --max-new 10 --dry-run
    .venv\\Scripts\\python.exe scripts\\98_s4_u4s.py summary --experiment S4U4 [--conditions U0_run1 U0_run2 U4 U4S] [--price <モデル> <入力> <出力>]
  足した根拠: docs/stage4/u4_protocol.md 第 3 節の規則（立て直しの試験台 docs/stage4/replan_bench_protocol.md 第 8 節）。
    10/08 23:56 に規則が成り立った（報告の正しさ Haiku 27.5%・Sonnet 40.0%、手の妥当さ どちらも 100%）。
  腕 U4S: 起動時の計画役は U4 と同じ s4（decompose_s4、claude-haiku-5-5）。立て直しだけを planner/replan_multi.py の
    make_replan_step("claude-sonnet-5-5") に替える（SYSTEM・SCHEMA・弾く規則・報告の検査・上限は replan_s4 と同じ）。
    実行器（runtime/executor_u4.py）・帯・--exec-interval 6・--no-safety・API の回し直し・鍵の確かめは 98_s4_u4.py と同じ。
  帯は 98_s4_u4.py の band_check をそのまま使う（s4_gates.json の bundle6_u4。改訂 3 で 191400〜191499 になるまでは
    191400:100 を拒む。終了コード 3）。--allow-smoke も同じ（X2 の生成の帯 44404〜44423 は拒む）。
  ■ 鍵の注意: 試すときは必ず --dry-run を付ける。--dry-run なしで起動すると、環境変数 ANTHROPIC_API_KEY を空にしても
    ユーザー環境変数（Windows のレジストリ）の鍵を読み、本物の API を呼んで帯の種を使う。
  取り違えの防ぎ: 試行の json の "u4" に arm（U4S）・replanner_model・replanner_variant・planner_model と使ったファイルの
    SHA-256 を残す。条件のフォルダに U4S 以外（U4・U0・腕の記録なし）の記録・控えがあれば、続きから回さない（終了コード 3。
    --accept-spec-change でも通さない）。条件名 U0_run1・U0_run2・U4 も使わない。
  summary は 98_s4_u4.py の summarize_condition（二重集計 cross_check を含む）で数え、費用だけを呼び出しのモデルごとに
    計算し直す（計画役は claude-haiku-5-5、立て直しは記録の requested_model。料金は src/recovla/vlm/cost.py の PRICES）。
    二重集計が一致しない、または 1 つの条件に腕・立て直しのモデルが混ざっていれば終了コード 1。
読むもの: scripts\\98_s4_u4.py・scripts\\96_s4_resume.py（importlib。書き換えない）、configs\\s4_gates.json（帯）、
  summary は outputs\\v2eval\\<実験>\\<条件>\\ の記録。
書くもの: 98_s4_u4.py run と同じ（outputs\\v2eval\\<実験>\\<条件>\\）。summary は outputs\\s4\\u4\\summary_<実験>_u4s.json。
  立て直しの応答のキャッシュは outputs\\llm_cache\\（replan_multi の鍵。U4 の鍵とは混ざらない）。
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARM = "U4S"
ARM_JA = "立て直しの計画（replan_multi、claude-sonnet-5-5）"
REPLAN_MODEL = "claude-sonnet-5-5"
PLANNER_MODEL = "claude-haiku-5-5"
DEFAULT_CONDITIONS = ("U0_run1", "U0_run2", "U4", "U4S")


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


s98 = _load(ROOT / "scripts" / "98_s4_u4.py", "s4_u4_for_u4s")
FILES = tuple(s98.FILES) + ("scripts/98_s4_u4s.py", "src/recovla/planner/replan_multi.py")
OUTD = s98.OUTD
band_check = s98.band_check


def load_r96():
    """96_s4_resume.py を読む（テストで差し替える口）。"""
    return _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_u4s")


def make_step():
    """executor_u4.install に渡す立て直しの関数（Sonnet 5.5 に固定）。"""
    from recovla.planner import replan_multi as RM
    return RM.make_replan_step(REPLAN_MODEL)


def make_engine(r96, info: dict):
    """98 の U4 の Engine を土台に、立て直しの関数だけを Sonnet 5.5 に差し替えた Engine。"""
    from recovla.planner import replan_s4 as R4
    from recovla.runtime import executor_u4 as U4
    step = make_step()
    Base = s98.make_engine(r96, "U4", info)

    class U4SEngine(Base):
        def init_task(self):
            super().init_task()
            orig = self.make_task                       # 98 の make（replan_s4 を入れ、u4_last を残す）

            def make(io, setup):
                ex = orig(io, setup)
                U4.install(ex, step, max_replans=R4.MAX_REPLANS)      # 同じ物の立て直しの関数だけ替える
                if getattr(ex.replan, "model", None) != REPLAN_MODEL:
                    raise RuntimeError(f"立て直しのモデルが {REPLAN_MODEL} でない")
                return ex
            self.make_task = make

        def run_one_task(self, i, seed):
            meta, arrays, rlog = super().run_one_task(i, seed)
            meta["u4"].update(arm=ARM, replanner_model=REPLAN_MODEL, replanner_variant=info["replanner_variant"],
                              planner_model=PLANNER_MODEL)
            return meta, arrays, rlog
    U4SEngine.u4s_step = step
    return U4SEngine


def _arm_of(d: dict) -> tuple:
    return d.get("arm"), d.get("replanner_model")


def foreign_records(out: pathlib.Path) -> list:
    """条件のフォルダにある、U4S 以外が作った記録・控え（理由の並び）。空なら続きから回してよい。"""
    out = pathlib.Path(out)
    if not out.is_dir():
        return []
    why = []
    ok = (ARM, REPLAN_MODEL)
    for p in sorted(out.glob("resume_spec*.json")) + [out / "run.json"] + sorted(out.glob("run_[0-9][0-9][0-9][0-9].json")):
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            why.append(f"{p.name}: 読めない（{type(e).__name__}）")
            continue
        if p.name.startswith("resume_spec"):
            got = (d.get("u4_arm"), d.get("u4_replanner_model"))
        else:
            got = _arm_of(d.get("u4") or {})
        if got != ok:
            why.append(f"{p.name}: 腕 {got[0]}・立て直し {got[1]}")
    return why


def cmd_run(argv) -> int:
    ap = argparse.ArgumentParser(prog="98_s4_u4s.py run", add_help=False)
    ap.add_argument("--arm", default=ARM, choices=[ARM])
    ap.add_argument("--allow-smoke", action="store_true")
    mine, rest = ap.parse_known_args(argv)
    r96 = load_r96()
    a = r96.build_parser().parse_args(["task"] + rest)
    if a.exec_interval not in (None, 6):
        print("U4S は U4 と同じ --exec-interval 6・--no-safety で回す（ほかの値は渡さない）", file=sys.stderr)
        return 3
    a.exec_interval, a.no_safety = 6, True
    if a.planner != "s4" or r96.PLANNERS["s4"]["model"] != PLANNER_MODEL:
        print(f"U4S の起動時の計画役は s4（decompose_s4、{PLANNER_MODEL}）だけ", file=sys.stderr)
        return 3
    if a.condition in s98.DEFAULT_CONDITIONS:
        print(f"条件名 {a.condition} は 98_s4_u4.py の腕の名なので U4S では使わない（例 U4S）", file=sys.stderr)
        return 3
    base, n = (int(x) for x in a.trials.split(":"))
    why = band_check(range(base, base + n), mine.allow_smoke)
    if why:
        print(why, file=sys.stderr)
        return 3
    if not a.dry_run:
        from recovla.planner import decompose as D
        if not D._api_key():
            print("U4S は ANTHROPIC_API_KEY が要る（無いと立て直しが全部「止まって知らせる」に倒れる）。--dry-run なら要らない",
                  file=sys.stderr)
            return 3
        print("[u4s] --dry-run なしで起動した: 帯の種を使い、計画役（Haiku 5.5）・立て直し（Sonnet 5.5）の API を呼ぶ"
              "（環境変数が空でもユーザー環境変数の鍵を読む）。試すだけなら Ctrl+C で止めて --dry-run を付ける", flush=True)
    from recovla.planner import replan_multi as RM
    info = {"script": "98_s4_u4s.py", "base_script": "98_s4_u4.py", "condition": a.condition,
            "replanner_model": REPLAN_MODEL, "replanner_variant": RM.variant(REPLAN_MODEL), "planner_model": PLANNER_MODEL,
            "files_sha256": {p: s98.sha256_file(ROOT / p) for p in FILES}}
    a.u4_arm = ARM
    a.u4_replan_sha256 = info["files_sha256"]["src/recovla/planner/replan_s4.py"]
    a.u4_executor_sha256 = info["files_sha256"]["src/recovla/runtime/executor_u4.py"]
    a.u4_replanner_model = REPLAN_MODEL
    a.u4_replan_multi_sha256 = info["files_sha256"]["src/recovla/planner/replan_multi.py"]
    r96.SPEC_KEYS = tuple(r96.SPEC_KEYS) + ("u4_arm", "u4_replan_sha256", "u4_executor_sha256", "u4_replanner_model",
                                            "u4_replan_multi_sha256")
    r96.Engine = make_engine(r96, info)
    orig_text = r96.run_json_text

    def run_json_text(a_, kind, rows, *x, **kw):
        d = json.loads(orig_text(a_, kind, rows, *x, **kw))
        d["u4"] = dict(info, arm=ARM, arm_ja=ARM_JA)
        return json.dumps(d, ensure_ascii=False, indent=1)
    r96.run_json_text = run_json_text
    ops = r96.load_ops()
    v82 = r96.load_82(a.allow_82_change)
    bad = foreign_records(pathlib.Path(v82.OUT) / a.experiment / a.condition)
    if bad:
        print(f"[u4s] {a.experiment}/{a.condition} に U4S 以外の記録・控えがあるので続きから回さない（別の条件名にする）: "
              f"{bad[:5]}{' ほか' if len(bad) > 5 else ''}", file=sys.stderr)
        return 3
    print(f"[u4s] 腕 {ARM}: {ARM_JA}（起動時の計画役 {PLANNER_MODEL}、指示文「{a.text}」、--exec-interval 6・--no-safety、"
          f"帯 {s98.ALLOC}）", flush=True)
    try:
        return r96.cmd_main(a, v82, ops)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


# ---------------------------------------------------------------- summary（記録だけを読む。判定はしない）
def llm_usage_by_model(meta: dict) -> dict:
    """{モデル: {decompose_calls, replan_calls, input_tokens, output_tokens}}。計画役は Haiku 5.5、立て直しは記録の
    requested_model（無ければ試行の u4.replanner_model、それも無ければ Haiku 5.5＝98 の U4）。キャッシュの応答は数えない。"""
    out = {}

    def add(model, kind, n, tin, tout):
        r = out.setdefault(model, {"decompose_calls": 0, "replan_calls": 0, "input_tokens": 0, "output_tokens": 0})
        r[kind] += n
        r["input_tokens"] += tin
        r["output_tokens"] += tout
    plan = meta.get("plan") or {}
    if plan and not plan.get("from_cache") and plan.get("usage"):
        add(PLANNER_MODEL, "decompose_calls", 1, int(plan["usage"]["input_tokens"]), int(plan["usage"]["output_tokens"]))
    default = (meta.get("u4") or {}).get("replanner_model") or PLANNER_MODEL
    for r in meta.get("replans") or []:
        llm = ((r.get("out") or {}).get("llm") or {})
        n = int(llm.get("calls", 0))
        us = llm.get("usage") or []
        if n or us:
            add(llm.get("requested_model") or default, "replan_calls", n, sum(int(u["input_tokens"]) for u in us),
                sum(int(u["output_tokens"]) for u in us))
    return out


def cost_of(by_model: dict, prices: dict) -> dict:
    rows, total = {}, 0.0
    for m, u in sorted(by_model.items()):
        if m not in prices:
            raise KeyError(f"料金の無いモデル: {m}（--price {m} <入力> <出力> で渡す）")
        pi, po = prices[m]
        c = u["input_tokens"] / 1e6 * pi + u["output_tokens"] / 1e6 * po
        rows[m] = dict(u, price_per_mtok=list(prices[m]), cost_usd=round(c, 6))
        total += c
    return {"by_model": rows, "cost_usd": round(total, 6)}


def arms_of_condition(metas: list) -> dict:
    """{"<腕>/<立て直しのモデル>": 本数}。U0・98 の U4 は立て直しのモデルの欄が無い。"""
    out = {}
    for m in metas:
        u = m.get("u4") or {}
        k = f"{u.get('arm')}/{u.get('replanner_model') or (PLANNER_MODEL if u.get('arm') == 'U4' else '-')}"
        out[k] = out.get(k, 0) + 1
    return out


def summarize_condition(d: pathlib.Path, prices: dict) -> dict:
    s = s98.summarize_condition(d, prices[PLANNER_MODEL])     # 二重集計を含む。費用だけ下で計算し直す
    metas, _ = s98.load_condition(d)
    by = {}
    for m in metas:
        for k, u in llm_usage_by_model(m).items():
            r = by.setdefault(k, {"decompose_calls": 0, "replan_calls": 0, "input_tokens": 0, "output_tokens": 0})
            for f in r:
                r[f] += u[f]
    c = cost_of(by, prices)
    s["llm"] = dict({k: v for k, v in s["llm"].items() if k != "price_per_mtok"}, cost_usd=c["cost_usd"],
                    by_model=c["by_model"], note="費用は呼び出しのモデルごとの料金で計算（計画役 Haiku 5.5、立て直しは記録のモデル）")
    s["arms"] = arms_of_condition(metas)
    return s


def cmd_summary(a) -> int:
    from recovla.vlm import cost
    prices = {k: tuple(v) for k, v in cost.PRICES.items()}
    for m, pi, po in a.price or []:
        prices[m] = (float(pi), float(po))
    base = ROOT / "outputs" / "v2eval" / a.experiment
    summ = {}
    for c in a.conditions:
        if (base / c).is_dir() and any((base / c).glob("run_[0-9][0-9][0-9][0-9].json")):
            summ[c] = summarize_condition(base / c, prices)
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": a.experiment, "note": "記述だけ（検定の族に入れない）",
           "script": "98_s4_u4s.py", "prices_per_mtok": {k: list(v) for k, v in sorted(prices.items())}, "conditions": summ}
    OUTD.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(a.out) if a.out else OUTD / f"summary_{a.experiment}_u4s.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for c, s in summ.items():
        print(f"{c}: n={s['n']} 腕 {s['arms']} 3 個とも {s['all_three_true']} 平均の収納数 {s['mean_stored_count']} "
              f"計画の変更 {s['interventions']['plan_change']} LLM {s['llm']['decompose_calls']}+{s['llm']['replan_calls']} 回 "
              f"${s['llm']['cost_usd']}")
    print(f"[u4s] 書いた: {p}")
    rc = 0
    bad = {c: [k for k, v in s_["cross_check"]["items"].items() if not v["match"]] for c, s_ in summ.items()
           if not s_["cross_check"]["all_match"]}
    if bad:
        print(f"[u4s] 二重集計が一致しない（98 と 56 の数え方）: {bad}。数字を使わずに原因を調べる", file=sys.stderr)
        rc = 1
    mixed = {c: s_["arms"] for c, s_ in summ.items() if len(s_["arms"]) > 1}
    if mixed:
        print(f"[u4s] 1 つの条件に腕・立て直しのモデルが混ざっている: {mixed}。数字を使わずに原因を調べる", file=sys.stderr)
        rc = 1
    return rc


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
    sub.add_parser("run", help="U4S を回す（96_s4_resume.py task の引数）。試すときは必ず --dry-run を付ける"
                               "（付けないと、環境変数を空にしてもユーザー環境変数の鍵を読んで本物の API を呼ぶ）")
    p = sub.add_parser("summary", help="記録から指標を出す（記述だけ。判定はしない）")
    p.add_argument("--experiment", required=True)
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    p.add_argument("--price", nargs=3, action="append", metavar=("MODEL", "IN", "OUT"),
                   help="モデルの料金（米ドル / 100 万トークン）を上書きする。何度でも渡せる")
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return {"summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
