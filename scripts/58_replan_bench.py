"""段階 4: 立て直しの計画役のオフラインの試験台（R-bench。研究の道・記述だけ）。手順書 docs/stage4/replan_bench_protocol.md。

E7（3 個の連続タスク）の記録で実行器が止まった場面を取り出し、U4 の立て直しの計画役が受け取るはずだった入力をそのまま作って、
claude-haiku-5-5・claude-sonnet-5-5・claude-opus-5-5 に答えさせ、手の正しさと日本語の報告の正しさを比べる。GPU・シミュレーションは使わない。

使い方（作業場所 C:\\PAI\\recovery_vla、Python は .venv\\Scripts\\python.exe）:
    58_replan_bench.py items [--dirs DIR ...] [--tag s3] [--allow-bundle1] [--write]
        止まった場面の項目を作る（既定は outputs/v2eval/V3S3/E7_* だけ）。--write のときだけ outputs/s4/replan_bench/items_<tag>.json に書く
    58_replan_bench.py cost  [--tag s3] [--models haiku,sonnet,opus] [--repeats 5] [--budget 10] [--write]
        費用の見込み（同期と Batch）。--write のときだけ cost_estimate_<tag>.json に書く
    58_replan_bench.py ask   --tag s3 [--models ...] [--repeats 5] [--batch [--no-wait] | --collect [BATCH_ID]] [--limit N] [--budget 10] --execute
        API に問う。**--execute が無ければ、何を何回呼ぶかと見込みの費用を出すだけで、クライアントを作らない（鍵も読まない）**
    58_replan_bench.py score --tag s3 [--models ...] [--repeats 5] [--manual FILE] [--allow-missing] [--write]
        u4_protocol.md 第 5 節の規則で報告を機械で採点し、手の妥当さ・倒した率・手の分布・モデル間の一致を出す（記述だけ）
  ask・score は、items --write で書いた items_<tag>.json だけを使う（材料を固定する。無ければ止める）。
■ 鍵の注意: decompose._api_key は、環境変数 ANTHROPIC_API_KEY を空にしてもユーザー環境変数（Windows のレジストリ）の鍵を読む。
  そのため ask は --execute を付けたときだけクライアントを作る。試すときは --execute を付けない。
書くもの: outputs/s4/replan_bench/ 以下だけ（items_*.json・cost_estimate_*.json・cache/・batches/・spend_log.jsonl・failures.jsonl・
  score_*.json・sheet_*.json）。既存のファイルは書き換えない。記録（outputs/v2eval/）は読むだけ。
束 1 の記録（S4DE7・S4DRTC・S4DSTART・S4XPL・outputs/s4/d_*）は、関門 1 の表の後に --allow-bundle1 を付けたときだけ読む。
終了コード: 0 正常（--execute なしの ask は見込みだけ出して 0）、2 エラー、3 前提の食い違い（予算超え・答えが欠けている など）。
"""
import argparse
import collections
import fractions
import hashlib
import json
import math
import pathlib
import re
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recovla.common.seeds import COLORS          # noqa: E402
from recovla.planner import replan_multi as M    # noqa: E402
from recovla.planner import replan_s4 as R4      # noqa: E402
from recovla.vlm import cost as C                # noqa: E402

BENCH_VERSION = "replan-bench-v1"
OUT = ROOT / "outputs" / "s4" / "replan_bench"
V2EVAL = ROOT / "outputs" / "v2eval"
DEFAULT_GLOB = ("V3S3", "E7_*")
BUNDLE1_EXPERIMENTS = ("S4DE7", "S4DRTC", "S4DSTART", "S4XPL")
DEFAULT_REPEATS = 5
DEFAULT_BUDGET_USD = 10.0
TAG_RE = re.compile(r"^[A-Za-z0-9_]+$")
EXECUTOR_KEYS = ("text", "plan", "steps", "stopped", "detected")     # 実行器が自分で持っている記録だけ（真値の欄は入れない）
TRUTH_KEYS = ("truth_success_t", "final_in_box", "all_three_in_box", "all_planned_in_box")
# 費用の見込みの仮定（ask の実測があれば置き換わる）。文字数からのトークンの見込み: 日本語 1 字 ≒ 1 トークン、
# 英数字・記号 4 字 ≒ 1 トークン、構造化出力の形の説明の分 SCHEMA_OVERHEAD
JP_TOKENS_PER_CHAR = 1.0
ASCII_CHARS_PER_TOKEN = 4.0
SCHEMA_OVERHEAD = 300
OUTPUT_TOKENS = {"claude-haiku-5-5": 200, "claude-sonnet-5-5": 250, "claude-opus-5-5": 800}
MARGIN_IN, MARGIN_OUT = 1.1, 1.5


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rel(p) -> str:
    try:
        return pathlib.Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


# ================================================================ items（止まった場面と、そのときの実行器の入力）
def step_result(s: dict) -> str:
    """executor_u4.ReplanTaskRuntime.step_result と同じ（tests が照らす）。"""
    if s.get("judged_complete") is True:
        return "success"
    if s.get("skipped"):
        return "skipped"
    if s.get("judged_complete") is False:
        return "timeout"
    return "pending"


def executor_view(meta: dict) -> dict:
    """試行の json のうち、実行器が自分で持っている欄だけの写し（真値の欄はここで落とす）。"""
    return json.loads(json.dumps({k: meta.get(k) for k in EXECUTOR_KEYS}))


def perception_from_entry(entry: dict) -> dict:
    """runtime の知覚の記録 1 件（WorldModel の写し）から、executor_u4.perception_summary と同じ色の状態を作る。"""
    out = {"table": [], "in_box": [], "in_hand": [], "lost": []}
    for c, e in sorted((entry.get("cubes") or {}).items()):
        st = e.get("status")
        if st in ("seen", "held"):
            out["in_box" if e.get("in_box") else "table"].append(c)
        elif st == "in_hand":
            out["in_hand"].append(c)
        elif st == "lost":
            out["lost"].append(c)
    return out


def perception_at(runtime_doc, t: float):
    """時刻 t に実行器が持っていた WorldModel（t 以前に取り込んだ最後の知覚の記録。同じ時刻は含める）。無ければ (None, 理由)。"""
    if not runtime_doc:
        return None, "no_runtime_file"
    per = ((runtime_doc.get("runtime") or {}).get("perception")) or []
    before = [e for e in per if float(e.get("t", 1e18)) <= float(t) + 1e-9]
    if not before:
        return None, "no_perception_before_stop"
    e = before[-1]
    return {"summary": perception_from_entry(e), "t_entry": e.get("t"), "t_obs": e.get("t_obs"),
            "age_s": round(float(t) - float(e.get("t")), 4)}, ""


def perception_reconstructed(view: dict) -> dict:
    """知覚の記録が無いときの代わり（実行器の記録だけ）: 起動時の検出の机・箱に、判定で成功した手順の色を箱へ動かす。
    手の中・見失いは分からないので空にする（限界。手順書 2-3）。"""
    det = view.get("detected") or {}
    done = [s["color"] for s in (view.get("steps") or []) if step_result(s) == "success"]
    box = list(dict.fromkeys(list(det.get("box") or []) + done))
    table = [c for c in (det.get("table") or []) if c not in box]
    return {"table": table, "in_box": box, "in_hand": [], "lost": []}


def build_input(view: dict, runtime_doc):
    """止まった場面の、立て直しの計画役への入力（replan_s4.replan_step の引数と同じ形）。返り値 (input, 付記, 除外の理由)。
    view は executor_view の戻り値（真値の欄を持たない）。"""
    st, steps = view.get("stopped"), view.get("steps") or []
    if not st:
        return None, None, "no_stop"
    if not steps or int(st.get("step", -1)) != len(steps) - 1 or step_result(steps[-1]) != "timeout":
        return None, None, "stop_not_at_failed_last_step"
    t = float(st["t"])
    per, why = perception_at(runtime_doc, t)
    if per is not None:
        perception, note = per["summary"], {"source": "runtime_log", "t_entry": per["t_entry"], "t_obs": per["t_obs"],
                                            "age_s": per["age_s"]}
    else:
        perception, note = perception_reconstructed(view), {"source": "reconstructed_from_record", "why": why}
    inp = {"text": str(view.get("text") or ""), "plan_order": list((view.get("plan") or {}).get("steps") or []),
           "steps": [{"color": s["color"], "result": step_result(s), "attempts": len(s.get("attempts") or [])} for s in steps],
           "perception": perception, "failed_step": int(st["step"]), "failed_color": steps[-1]["color"], "replans_done": 0}
    return inp, dict(note, t_request=t), ""


def truth_box_at(meta: dict, t: float) -> list:
    """時刻 t までに真値で箱に収まった色（98_s4_u4.py の truth_box_at と同じ。採点だけに使う）。"""
    ts = meta.get("truth_success_t") or {}
    return [c for c in COLORS if c in ts and float(ts[c]) <= float(t) + 1e-9]


def truth_of(meta: dict, t: float) -> dict:
    return {"box_at_request": truth_box_at(meta, t),
            "final_in_box": [c for c in COLORS if (meta.get("final_in_box") or {}).get(c)],
            "truth_success_t": meta.get("truth_success_t") or {}, "all_three_in_box": bool(meta.get("all_three_in_box"))}


def is_bundle1(d: pathlib.Path) -> bool:
    r = rel(d)
    return any(f"/{x}/" in f"/{r}/" for x in BUNDLE1_EXPERIMENTS) or r.startswith("outputs/s4/d_")


def default_dirs() -> list:
    exp, pat = DEFAULT_GLOB
    return sorted(p for p in (V2EVAL / exp).glob(pat) if p.is_dir())


def resolve_dirs(dirs, allow_bundle1: bool) -> list:
    """読むフォルダ（既定は V3S3 の E7_*）。束 1 の記録は --allow-bundle1 のときだけ。"""
    ds = [pathlib.Path(x) for x in dirs] if dirs else default_dirs()
    bad = [rel(d) for d in ds if is_bundle1(d)]
    if bad and not allow_bundle1:
        raise SystemExit(f"束 1 の記録 {bad} は関門 1 の表の後に --allow-bundle1 を付けたときだけ読む")
    return ds


def build_items(dirs, tag: str) -> dict:
    items, excluded, sources = [], [], []
    for d in dirs:
        d = pathlib.Path(d)
        runs = sorted(d.glob("run_[0-9][0-9][0-9][0-9].json"))
        n_items = 0
        for p in runs:
            b = p.read_bytes()
            meta = json.loads(b.decode("utf-8"))
            if "steps" not in meta or "plan" not in meta:
                excluded.append({"file": rel(p), "why": "not_a_task_record"})
                continue
            rt = p.with_name(p.stem + "_runtime.json")
            rt_b = rt.read_bytes() if rt.is_file() else None
            runtime_doc = json.loads(rt_b.decode("utf-8")) if rt_b else None
            inp, note, why = build_input(executor_view(meta), runtime_doc)
            item_id = f"{rel(d).replace('outputs/v2eval/', '')}/{p.stem}"
            if inp is None:
                excluded.append({"item_id": item_id, "why": why, "timed_out": bool(meta.get("timed_out"))})
                continue
            ctx = R4.make_context(**inp)
            items.append({"item_id": item_id, "input": inp, "candidates": R4.candidates(ctx),
                          "perception_note": note, "t_request": note["t_request"],
                          "source": {"file": rel(p), "sha256": sha256_bytes(b), "runtime": rel(rt) if rt_b else None,
                                     "runtime_sha256": sha256_bytes(rt_b) if rt_b else None, "run": meta.get("run"),
                                     "seed": meta.get("seed"), "model": meta.get("model")},
                          "truth": truth_of(meta, note["t_request"])})
            n_items += 1
        sources.append({"dir": rel(d), "n_runs": len(runs), "n_items": n_items})
    return {"version": BENCH_VERSION, "tag": tag, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "sources": sources,
            "items": items, "excluded": excluded,
            "note": "input は実行器の記録と知覚の記録だけから作った（真値は truth に分けて、採点だけに使う）"}


def summarize_items(doc: dict) -> dict:
    its = doc["items"]
    return {"tag": doc["tag"], "sources": doc["sources"], "items": len(its),
            "perception_source": dict(collections.Counter(x["perception_note"]["source"] for x in its)),
            "perception_age_s_max": max((x["perception_note"].get("age_s") or 0.0 for x in its), default=None),
            "failed_step": dict(collections.Counter(str(x["input"]["failed_step"] + 1) for x in its)),
            "failed_color": dict(collections.Counter(x["input"]["failed_color"] for x in its)),
            "n_candidates": dict(collections.Counter(str(len(x["candidates"])) for x in its)),
            "perceived_in_box_vs_truth": dict(collections.Counter(
                "same" if sorted(x["input"]["perception"]["in_box"]) == sorted(x["truth"]["box_at_request"]) else "differs"
                for x in its)),
            "failed_color_truly_in_box": sum(1 for x in its if x["input"]["failed_color"] in x["truth"]["box_at_request"]),
            "excluded": dict(collections.Counter(e["why"] for e in doc["excluded"]))}


def items_path(tag: str) -> pathlib.Path:
    return OUT / f"items_{tag}.json"


def load_items(tag: str, dirs=None, allow_bundle1=False) -> dict:
    """items_<tag>.json があれば読む。無ければ記録から作る（書かない）。"""
    p = items_path(tag)
    if p.is_file() and not dirs:
        return json.loads(p.read_text(encoding="utf-8"))
    if not dirs and tag != "s3":
        raise SystemExit(f"{rel(p)} が無い（--tag {tag} は先に items --dirs ... --write）")
    return build_items(resolve_dirs(dirs, allow_bundle1), tag)


def load_fixed_items(tag: str) -> dict:
    """ask・score は書いた items_<tag>.json だけを使う（材料を固定する）。"""
    p = items_path(tag)
    if not p.is_file():
        raise SystemExit(f"{rel(p)} が無い（先に items --tag {tag} --write）")
    return json.loads(p.read_text(encoding="utf-8"))


# ================================================================ 問い（項目 x モデル x 回）
def models_of(s: str) -> list:
    out = []
    for x in s.split(","):
        x = x.strip()
        m = M.MODELS.get(x, x)
        M.settings(m)
        out.append(m)
    return out


def key_extra(repeat: int) -> dict:
    return {"bench": BENCH_VERSION, "repeat": int(repeat)}


def questions(doc: dict, models: list, repeats: int) -> list:
    qs = []
    for it in doc["items"]:
        ctx = R4.make_context(**it["input"])
        msg = R4.user_message(ctx)
        for m in models:
            for r in range(repeats):
                qs.append({"item_id": it["item_id"], "model": m, "repeat": r, "ctx": ctx, "msg": msg,
                           "key": M.cache_key(m, msg, key_extra(r))})
    return qs


def cache_dir() -> pathlib.Path:
    return OUT / "cache"


def read_jsonl(p: pathlib.Path) -> list:
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def append_jsonl(p: pathlib.Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def spent_usd() -> float:
    return round(sum(float(x.get("usd") or 0.0) for x in read_jsonl(OUT / "spend_log.jsonl")), 6)


def failures() -> dict:
    out = collections.defaultdict(lambda: {"attempts": 0, "errors": []})
    for x in read_jsonl(OUT / "failures.jsonl"):
        out[x["key"]]["attempts"] += int(x.get("attempts", 1))
        out[x["key"]]["errors"] += list(x.get("errors") or [])
    return out


def has_cache(key: str) -> bool:
    return (cache_dir() / f"{key}.json").is_file()


def todo_of(qs: list) -> list:
    fail = failures()
    seen, out = set(), []
    for q in qs:
        if q["key"] in seen or has_cache(q["key"]) or fail[q["key"]]["attempts"] >= 1 + R4.API_RETRIES:
            continue
        seen.add(q["key"])
        out.append(q)
    return out


# ================================================================ 費用
def text_tokens(s: str) -> float:
    jp = sum(1 for ch in s if ord(ch) > 127)
    return jp * JP_TOKENS_PER_CHAR + (len(s) - jp) / ASCII_CHARS_PER_TOKEN


def calib_from_cache() -> dict:
    """キャッシュの実測（モデルごとの入力・出力のトークンの中央値）。"""
    by = collections.defaultdict(lambda: {"input_tokens": [], "output_tokens": []})
    d = cache_dir()
    if d.is_dir():
        for p in d.glob("*.json"):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            m, u = (rec.get("request") or {}).get("requested_model"), rec.get("usage") or {}
            if m and u.get("input_tokens") is not None:
                by[m]["input_tokens"].append(int(u["input_tokens"]))
                by[m]["output_tokens"].append(int(u.get("output_tokens") or 0))
    return {m: {"input_tokens": statistics.median(v["input_tokens"]), "output_tokens": statistics.median(v["output_tokens"]),
                "n": len(v["input_tokens"])} for m, v in by.items() if v["input_tokens"]}


def question_tokens(q: dict, calib: dict) -> dict:
    c = calib.get(q["model"])
    if c:
        return {"input_tokens": int(math.ceil(c["input_tokens"] * MARGIN_IN)),
                "output_tokens": int(math.ceil(c["output_tokens"] * MARGIN_OUT)), "source": "measured"}
    tin = text_tokens(R4.SYSTEM) + text_tokens(q["msg"]) + SCHEMA_OVERHEAD
    return {"input_tokens": int(math.ceil(tin * MARGIN_IN)),
            "output_tokens": int(math.ceil(OUTPUT_TOKENS[q["model"]] * MARGIN_OUT)), "source": "formula"}


def estimate(qs: list, calib: dict = None) -> dict:
    calib = calib if calib is not None else calib_from_cache()
    rows = {}
    for q in qs:
        tok = question_tokens(q, calib)
        r = rows.setdefault(q["model"], {"model": q["model"], "questions": 0, "input_tokens": 0, "output_tokens": 0,
                                         "token_source": tok["source"]})
        r["questions"] += 1
        r["input_tokens"] += tok["input_tokens"]
        r["output_tokens"] += tok["output_tokens"]
    for r in rows.values():
        u = {"input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]}
        r["usd_sync"] = round(C.cost_usd(r["model"], u, batch=False), 4)
        r["usd_batch"] = round(C.cost_usd(r["model"], u, batch=True), 4)
        r["usd_per_question_sync"] = round(r["usd_sync"] / r["questions"], 6) if r["questions"] else 0.0
    return {"rows": list(rows.values()), "total_usd_sync": round(sum(r["usd_sync"] for r in rows.values()), 4),
            "total_usd_batch": round(sum(r["usd_batch"] for r in rows.values()), 4),
            "prices_per_mtok": {m: C.PRICES[m] for m in rows}, "batch_factor": C.BATCH_FACTOR,
            "assumptions": {"jp_tokens_per_char": JP_TOKENS_PER_CHAR, "ascii_chars_per_token": ASCII_CHARS_PER_TOKEN,
                            "schema_overhead": SCHEMA_OVERHEAD, "output_tokens": OUTPUT_TOKENS,
                            "margin_in": MARGIN_IN, "margin_out": MARGIN_OUT}}


# ================================================================ ask
def make_client():
    """本物のクライアント（decompose.client。鍵はレジストリからも読む）。ask --execute のときだけ呼ぶ。"""
    return R4.client()


def log_spend(key: str, model: str, usage: dict, mode: str, item_id=None) -> float:
    usd = round(C.cost_usd(model, usage or {}, batch=(mode == "batch")), 8)
    append_jsonl(OUT / "spend_log.jsonl", {"key": key, "item_id": item_id, "model": model, "mode": mode, "usage": usage,
                                           "usd": usd, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    return usd


def ask_sync(qs: list, cli, budget: float) -> dict:
    n_called, n_fail = 0, 0
    for q in qs:
        if spent_usd() > budget:
            log(f"これまでの費用 ${spent_usd()} が上限 ${budget} を超えたので止める")
            return {"called": n_called, "failed": n_fail, "stopped_by_budget": True}
        out = M.replan(q["ctx"], model=q["model"], cli=cli, use_cache=True, cache_dir=cache_dir(),
                       key_extra=key_extra(q["repeat"]))
        llm = out["llm"]
        for u in llm.get("usage") or []:
            log_spend(q["key"], q["model"], u, "sync", q["item_id"])
        n_called += int(llm.get("calls", 0))
        if out["rejected"][:1] == ["llm_failed"]:
            n_fail += 1
            append_jsonl(OUT / "failures.jsonl", {"key": q["key"], "item_id": q["item_id"], "model": q["model"],
                                                  "repeat": q["repeat"], "mode": "sync", "attempts": int(llm.get("calls", 0)),
                                                  "errors": llm.get("errors")})
        log(f"[ask] {q['model']} {q['item_id']} r{q['repeat']} {out['decision']['action']} "
            f"{'受理' if out['accepted'] else '倒した'} 呼び出し {llm.get('calls', 0)}")
    return {"called": n_called, "failed": n_fail, "stopped_by_budget": False}


def custom_id(key: str) -> str:
    return "k" + key[:48]


def submit_batch(qs: list, cli) -> dict:
    reqs = [{"custom_id": custom_id(q["key"]), "params": M.build_params(q["model"], q["msg"])} for q in qs]
    b = cli.messages.batches.create(requests=reqs)
    rec = {"batch_id": b.id, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "t_created": time.time(), "n": len(qs),
           "status": getattr(b, "processing_status", None),
           "map": {custom_id(q["key"]): {"key": q["key"], "item_id": q["item_id"], "model": q["model"], "repeat": q["repeat"],
                                         "msg": q["msg"]} for q in qs}}
    d = OUT / "batches"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{b.id}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[batch] 出した {b.id}: {len(qs)} 件")
    return rec


def collect_batch(batch_id: str, cli, wait: bool = True, poll_s: float = 30.0, sleep=time.sleep) -> dict:
    path = OUT / "batches" / f"{batch_id}.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    while True:
        b = cli.messages.batches.retrieve(batch_id)
        st = getattr(b, "processing_status", None)
        if st == "ended" or not wait:
            break
        log(f"[batch] {batch_id} {st}")
        sleep(poll_s)
    rec["status"] = st
    if st != "ended":
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec
    counts = collections.Counter()
    for r in cli.messages.batches.results(batch_id):
        m = rec["map"].get(r.custom_id)
        if m is None:
            counts["unknown_custom_id"] += 1
            continue
        typ = r.result.type
        counts[typ] += 1
        errs = []
        if typ == "succeeded":
            fields = M.message_fields(r.result.message)
            log_spend(m["key"], m["model"], fields["usage"], "batch", m["item_id"])
            data, why = M.parse_message(fields)
            if data is not None:
                M.write_cache(cache_dir() / f"{m['key']}.json",
                              M.cache_record(m["key"], m["model"], m["msg"], fields, 0, "batch", key_extra(m["repeat"]),
                                             {"batch_id": batch_id, "custom_id": r.custom_id}))
                continue
            errs.append(f"parse:{why}")
        else:
            err = getattr(r.result, "error", None)
            errs.append(f"batch:{typ}:{str(err)[:200] if err is not None else ''}")
        append_jsonl(OUT / "failures.jsonl", {"key": m["key"], "item_id": m["item_id"], "model": m["model"],
                                              "repeat": m["repeat"], "mode": "batch", "attempts": 1, "errors": errs})
    rec.update(ended=time.strftime("%Y-%m-%d %H:%M:%S"), counts=dict(counts))
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[batch] {batch_id} 終わり {dict(counts)}")
    return rec


# ================================================================ 採点（u4_protocol.md 第 5 節の規則を機械で当てる）
JA2C = {"赤": "red", "緑": "green", "青": "blue"}
_C = "[赤緑青]"
_CL = rf"(?P<c>{_C}(?:の立方体)?(?:(?:と|・|、|，|および|や){_C}(?:の立方体)?)*)"
FAIL_RE = re.compile(r"入れられ(?:ませんでした|なかった|ず|ない)|入り(?:ませんでした|きらなかった)|入らなかった|"
                     r"終えられ(?:ませんでした|なかった|ず)|終わら(?:なかった|ず)|でき(?:ませんでした|なかった|ず)|失敗|"
                     r"つかめ(?:ませんでした|なかった|ず)|うまくいか|間に合わ|運べ(?:ませんでした|なかった|ず)|"
                     r"置け(?:ませんでした|なかった|ず)")
EMPTY_RE = re.compile(r"箱(?:の中)?(?:に|には|は)(?:まだ)?(?:何も|なにも)(?:入っ|あり|ない)|箱(?:の中)?は(?:まだ)?空")
_NEG = r"(?!ておらず|ていな|てない|ていませ)"
_IN = rf"(?:入っ{_NEG}|入りまし|収まっ{_NEG}|あり(?!ませ)|ある)"
BOX_RES = [
    re.compile(rf"箱(?:の中)?(?:に|には)(?:すでに|もう|既に)?{_CL}(?:の立方体)?(?:が|も|だけが)?(?:すでに|もう|既に)?{_IN}"),
    re.compile(rf"{_CL}(?:の立方体)?(?:は|が|も)(?:すでに|もう|既に)?箱(?:の中)?に(?:{_IN}|入れ(?:まし|終え|た|てあ))"),
    re.compile(rf"{_CL}(?:の立方体)?を(?:すでに|もう|既に)?箱(?:の中)?に入れ(?:まし|終え|た|てあ)"),
    re.compile(rf"箱(?:の中)?に(?:入っている|ある)の(?:は|が){_CL}"),
]
STOP_RE = re.compile(r"止ま|止め|停止|中断|知らせ|待ち|指示を|判断を")
FINISH_RE = re.compile(r"終え|終了|終わり|終わら|完了")
SKIP_RE = re.compile(r"飛ばし|あきらめ|諦め|見送|スキップ|除いて|外して")
RETRY_RE = re.compile(r"もう一度|もう1度|もう 1 度|再度|やり直|再び|改めて")
CONT_RE = re.compile(r"入れます|入れ直|試します|続けます|運びます|進めます|取りかか|取り掛か|向かいます|入れていきます|行います|入れて")
LATER_RE = re.compile(r"後に回|後回し|最後|後で")


def colors_in(s: str) -> list:
    return list(dict.fromkeys(JA2C[ch] for ch in s if ch in JA2C))


def _sentences(t: str) -> list:
    """(始め, 終わり) の並び（。！？ で区切る）。"""
    out, s = [], 0
    for m in re.finditer(r"[。！？!?]", t):
        out.append((s, m.end()))
        s = m.end()
    if s < len(t):
        out.append((s, len(t)))
    return out


def _clauses(t: str, a: int, b: int) -> list:
    out, s = [], a
    for m in re.finditer(r"[、，,]", t[a:b]):
        out.append((s, a + m.end()))
        s = a + m.end()
    if s < b:
        out.append((s, b))
    return out


def parse_report(text: str) -> dict:
    """報告の中の 3 種類の主張を取り出す（採点の前の段）。曖昧なものは ambiguous の印を付ける。"""
    t = (text or "").strip()
    named, fail_spans = [], []
    for a, b in _sentences(t):
        cls = _clauses(t, a, b)
        for i, (ca, cb) in enumerate(cls):
            if not FAIL_RE.search(t[ca:cb]):
                continue
            cs = colors_in(t[ca:cb])
            k = i
            while not cs and k > 0:                       # 色が無ければ、同じ文の前の節で最も近い色
                k -= 1
                cs = colors_in(t[cls[k][0]:cls[k][1]])
            named += cs
            fail_spans.append((a, cb))                    # 文の始めから失敗の節の終わりまで
    box_cols, box_spans, empty = [], [], False
    for m in EMPTY_RE.finditer(t):
        empty = True
        box_spans.append(m.span())
    for rx in BOX_RES:
        for m in rx.finditer(t):
            if any(a <= m.start() < b for a, b in fail_spans) and FAIL_RE.search(m.group(0)):
                continue
            box_cols += colors_in(m.group("c"))
            box_spans.append(m.span())
    keep = [True] * len(t)
    for a, b in fail_spans + box_spans:
        for i in range(a, min(b, len(t))):
            keep[i] = False
    rest = "".join(ch if k else "　" for ch, k in zip(t, keep))
    cats = set()
    if SKIP_RE.search(rest):
        cats.add("skip")
    if STOP_RE.search(rest):
        cats.add("stop")
    if FINISH_RE.search(rest) and "skip" not in cats:
        cats.add("finish")
    seq = colors_in(rest)
    if (RETRY_RE.search(rest) or CONT_RE.search(rest)) and seq and "skip" not in cats:
        cats.add("continue")
    later = []
    for a, b in _sentences(rest):
        for ca, cb in _clauses(rest, a, b):
            if LATER_RE.search(rest[ca:cb]):
                later += colors_in(rest[ca:cb])
    skip_cols = []
    if "skip" in cats:
        for a, b in _sentences(rest):
            for ca, cb in _clauses(rest, a, b):
                if SKIP_RE.search(rest[ca:cb]):
                    skip_cols += colors_in(rest[ca:cb])
    return {"failed_named": list(dict.fromkeys(named)), "has_fail_claim": bool(fail_spans),
            "box_empty": empty, "box_colors": list(dict.fromkeys(box_cols)), "has_box_claim": bool(box_spans),
            "next_kinds": sorted(cats), "next_seq": seq, "later": list(dict.fromkeys(later)),
            "skip_colors": list(dict.fromkeys(skip_cols))}


def grade_report(text: str, failed: str, truth_box: list, decision: dict, applied_order: list) -> dict:
    """u4_protocol.md 第 5 節の 3 種類の主張に correct・wrong・none・ambiguous を付け、報告の正しさを出す。"""
    p = parse_report(text)
    tb = set(truth_box)
    # 1 終わらなかった色
    if not p["failed_named"]:
        c1 = "none"
    elif failed in tb:
        c1 = "wrong"                                       # 真値では既に箱の中（判定の取りこぼしに由来。別に数える）
    else:
        c1 = "correct" if set(p["failed_named"]) == {failed} else "wrong"
    # 2 箱の中の色
    if p["box_empty"] and p["box_colors"]:
        c2 = "ambiguous"
    elif p["box_empty"]:
        c2 = "correct" if not tb else "wrong"
    elif p["box_colors"]:
        c2 = "correct" if set(p["box_colors"]) == tb else "wrong"
    else:
        c2 = "none"
    # 3 次にすること（実際に通した手と比べる）
    kinds, a = set(p["next_kinds"]), decision["action"]
    if not kinds:
        c3 = "none"
    elif len(kinds) > 1:
        c3 = "ambiguous"
    else:
        k = next(iter(kinds))
        if a in ("stop", "finish"):
            c3 = "correct" if k == a else "wrong"
        elif a == "skip":
            c3 = "correct" if k == "skip" and (not p["skip_colors"] or set(p["skip_colors"]) == {decision["color"]}) else "wrong"
        else:                                              # next・reorder: 述べた色の順が、これから実行する並びの先頭と合う
            seq = p["next_seq"]
            ok = k == "continue" and bool(seq) and seq == list(applied_order[:len(seq)])
            if not ok and k == "continue" and p["later"]:   # 「後に回す」「最後に」と述べた色は、並びの後ろに置いて照らす
                moved = [c for c in seq if c not in p["later"]] + [c for c in seq if c in p["later"]]
                ok = moved == list(applied_order[:len(moved)])
            c3 = "correct" if ok else "wrong"
    correct = c1 == "correct" and c2 not in ("wrong", "ambiguous") and c3 not in ("wrong", "ambiguous")
    return {"c1": c1, "c2": c2, "c3": c3, "correct": correct,
            "needs_manual": "ambiguous" in (c2, c3), "truth_failed_in_box": failed in tb, "parsed": p}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list:
    if n <= 0:
        return [None, None]
    ph = k / n
    den = 1 + z * z / n
    cen = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, cen - half), 4), round(min(1.0, cen + half), 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


class _NoCall:
    """採点はキャッシュだけを読む（呼んだら失敗）。"""

    def __init__(self):
        import types
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        raise AssertionError("採点で API を呼ぼうとした")


def answer_of(q: dict, fail: dict):
    """(out, 状態)。状態は cached・llm_failed・missing。"""
    if has_cache(q["key"]):
        out = M.replan(q["ctx"], model=q["model"], cli=_NoCall(), use_cache=True, cache_dir=cache_dir(),
                       key_extra=key_extra(q["repeat"]))
        if out["llm"].get("from_cache"):
            return M.finish_step(q["ctx"], out), "cached"
    f = fail.get(q["key"])
    if f and f["attempts"] >= 1 + R4.API_RETRIES:
        llm = {"called": True, "calls": f["attempts"], "errors": f["errors"], "from_cache": False}
        return M.finish_step(q["ctx"], M.fallback_out(q["ctx"], q["model"], llm, f["errors"])), "llm_failed"
    return None, "missing"


def row_id(key: str) -> str:
    return hashlib.sha256(("row:" + key).encode()).hexdigest()[:12]


def score_rows(doc: dict, qs: list, manual: dict = None) -> tuple:
    items = {x["item_id"]: x for x in doc["items"]}
    fail = failures()
    rows, missing = [], []
    for q in qs:
        out, state = answer_of(q, fail)
        if out is None:
            missing.append({"item_id": q["item_id"], "model": q["model"], "repeat": q["repeat"]})
            continue
        it = items[q["item_id"]]
        g = grade_report(out["report"], it["input"]["failed_color"], it["truth"]["box_at_request"], out["decision"],
                         out["next_order"])
        rid = row_id(q["key"])
        man = (manual or {}).get(rid)
        if man:
            g.update({k: man[k] for k in ("c1", "c2", "c3") if k in man}, manual=True)
            g["correct"] = g["c1"] == "correct" and g["c2"] not in ("wrong", "ambiguous") and g["c3"] not in ("wrong", "ambiguous")
            g["needs_manual"] = False
        raw = (out.get("llm_output") or {})
        rows.append({"row_id": rid, "item_id": q["item_id"], "model": q["model"], "repeat": q["repeat"], "state": state,
                     "decision": out["decision"], "next_order": out["next_order"], "raw_action": raw.get("action"),
                     "accepted": bool(out["accepted"]), "fallback": bool(out["fallback"]),
                     "fallback_kind": ("llm_failed" if out["rejected"][:1] == ["llm_failed"] else
                                       "rejected" if out["fallback"] else None),
                     "rejected": out["rejected"], "report": out["report"], "report_source": out["report_source"],
                     "report_rejected": out.get("report_rejected"), "grade": g})
    return rows, missing


def per_model(rows: list) -> dict:
    out = {}
    for m in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == m]
        n = len(rs)
        claims = {c: dict(collections.Counter(r["grade"][c] for r in rs)) for c in ("c1", "c2", "c3")}
        by_src = {s: rate(sum(r["grade"]["correct"] for r in rs if r["report_source"] == s),
                          sum(1 for r in rs if r["report_source"] == s)) for s in ("llm", "template")}
        out[m] = {"n": n, "valid_action": rate(sum(r["accepted"] for r in rs), n),
                  "fallback": rate(sum(r["fallback"] for r in rs), n),
                  "fallback_kind": dict(collections.Counter(r["fallback_kind"] for r in rs if r["fallback_kind"])),
                  "rejected_reasons": dict(collections.Counter(":".join(x.split(":")[:2]) for r in rs
                                                               if r["fallback_kind"] == "rejected" for x in r["rejected"])),
                  "actions_final": dict(collections.Counter(r["decision"]["action"] for r in rs)),
                  "actions_raw": dict(collections.Counter(str(r["raw_action"]) for r in rs)),
                  "report_source": dict(collections.Counter(r["report_source"] for r in rs)),
                  "report_correct": rate(sum(r["grade"]["correct"] for r in rs), n),
                  "report_correct_by_source": by_src, "claims": claims,
                  "needs_manual": sum(1 for r in rs if r["grade"]["needs_manual"]),
                  "truth_failed_in_box": sum(1 for r in rs if r["grade"]["truth_failed_in_box"]),
                  "stable_items": stable(rs)}
    return out


def stable(rs: list) -> dict:
    by = collections.defaultdict(list)
    for r in rs:
        by[r["item_id"]].append(json.dumps(r["decision"], sort_keys=True))
    multi = [v for v in by.values() if len(v) > 1]
    return rate(sum(1 for v in multi if len(set(v)) == 1), len(multi))


def agreement(rows: list) -> dict:
    idx = {(r["model"], r["item_id"], r["repeat"]): r for r in rows}
    models = list(dict.fromkeys(r["model"] for r in rows))
    out = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            pairs = [(idx[k], idx[(b,) + k[1:]]) for k in idx if k[0] == a and (b,) + k[1:] in idx]
            out[f"{a}|{b}"] = {"action": rate(sum(x["decision"]["action"] == y["decision"]["action"] for x, y in pairs), len(pairs)),
                               "decision": rate(sum(x["decision"] == y["decision"] for x, y in pairs), len(pairs))}
    return out


def sonnet_rule(pm: dict) -> dict:
    """手順書の事前に決めた規則: Sonnet 5.5 の報告の正しさ ≥ Haiku 5.5 ＋ 10 ポイント かつ 手の妥当さ（受理の率）≥ Haiku なら U4S を足す。"""
    h, s = pm.get("claude-haiku-5-5"), pm.get("claude-sonnet-5-5")
    if not h or not s or not h["n"] or not s["n"]:
        return {"decidable": False, "why": "Haiku と Sonnet の両方の答えが要る"}
    rc_h = fractions.Fraction(h["report_correct"]["k"], h["n"])
    rc_s = fractions.Fraction(s["report_correct"]["k"], s["n"])
    va_h = fractions.Fraction(h["valid_action"]["k"], h["n"])
    va_s = fractions.Fraction(s["valid_action"]["k"], s["n"])
    c1 = rc_s >= rc_h + fractions.Fraction(1, 10)
    c2 = va_s >= va_h
    return {"decidable": True, "report_correct": {"haiku": float(rc_h), "sonnet": float(rc_s),
                                                  "diff_pp": round(float(rc_s - rc_h) * 100, 2)},
            "valid_action": {"haiku": float(va_h), "sonnet": float(va_s)},
            "cond_report_plus10": c1, "cond_valid_not_lower": c2, "add_U4S": bool(c1 and c2),
            "needs_manual_left": {"haiku": h["needs_manual"], "sonnet": s["needs_manual"]}}


# ================================================================ コマンド
def cmd_items(a) -> int:
    doc = build_items(resolve_dirs(a.dirs, a.allow_bundle1), a.tag)       # いつも記録から作り直す（同じ記録なら同じ結果）
    s = summarize_items(doc)
    print(json.dumps(s, ensure_ascii=False, indent=1))
    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        p = items_path(a.tag)
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"書いた: {rel(p)}（SHA-256 {sha256_bytes(p.read_bytes())[:12]}…）")
    else:
        log("書いていない（--write で items_<tag>.json に書く）")
    return 0


def cmd_cost(a) -> int:
    doc = load_items(a.tag, a.dirs or None, a.allow_bundle1)
    qs = questions(doc, models_of(a.models), a.repeats)
    est = estimate(qs)
    todo = todo_of(qs)
    est_todo = estimate(todo)
    res = {"tag": a.tag, "items": len(doc["items"]), "models": models_of(a.models), "repeats": a.repeats,
           "questions": len(qs), "todo": len(todo), "estimate_all": est, "estimate_todo": est_todo,
           "spent_usd": spent_usd(), "budget_usd": a.budget,
           "within_budget_sync": spent_usd() + est_todo["total_usd_sync"] <= a.budget + 1e-12,
           "within_budget_batch": spent_usd() + est_todo["total_usd_batch"] <= a.budget + 1e-12}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"cost_estimate_{a.tag}.json").write_text(json.dumps(dict(res, written=time.strftime("%Y-%m-%d %H:%M:%S")),
                                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if res["within_budget_batch"] else 3


def cmd_ask(a) -> int:
    doc = load_fixed_items(a.tag)
    qs = questions(doc, models_of(a.models), a.repeats)
    todo = todo_of(qs)
    if a.limit:
        todo = todo[:a.limit]
    est = estimate(todo)
    cost_now = est["total_usd_batch"] if a.batch else est["total_usd_sync"]
    log(f"問い {len(qs)}（まだのもの {len(todo)}）。見込み ${cost_now}（これまで ${spent_usd()}、上限 ${a.budget}）")
    if not a.execute:
        log("--execute が無いので呼ばない（クライアントも作らない・鍵も読まない）。見込みだけ出した")
        return 0
    if a.collect is not None:
        ids = [a.collect] if a.collect != "ALL" else [
            p.stem for p in sorted((OUT / "batches").glob("*.json"))
            if json.loads(p.read_text(encoding="utf-8")).get("status") != "ended"]
        if ids:
            cli = make_client()
            for bid in ids:
                collect_batch(bid, cli, wait=not a.no_wait)
        return 0
    if spent_usd() + cost_now > a.budget + 1e-12:
        log("予算の上限を超える見込みなので止める（--budget）")
        return 3
    if not todo:
        log("問うものが無い（全部キャッシュにあるか、2 回失敗して倒した）")
        return 0
    cli = make_client()
    if a.batch:
        rec = submit_batch(todo, cli)
        if not a.no_wait:
            collect_batch(rec["batch_id"], cli, wait=True)
        return 0
    res = ask_sync(todo, cli, a.budget)
    log(f"同期: 呼び出し {res['called']}、2 回とも失敗 {res['failed']}、これまで ${spent_usd()}")
    return 3 if res["stopped_by_budget"] else 0


def cmd_score(a) -> int:
    doc = load_fixed_items(a.tag)
    qs = questions(doc, models_of(a.models), a.repeats)
    manual = json.loads(pathlib.Path(a.manual).read_text(encoding="utf-8")) if a.manual else None
    rows, missing = score_rows(doc, qs, manual)
    if missing and not a.allow_missing:
        log(f"答えの無い問いが {len(missing)} ある（ask の後に採点する。--allow-missing で欠けたまま出す）")
        return 3
    pm = per_model(rows)
    res = {"version": BENCH_VERSION, "tag": a.tag, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "note": "記述だけ。検定の族に入れない。提出物の顔の切り替えの判定（掲示板 0157）に使わない",
           "items": len(doc["items"]), "repeats": a.repeats, "missing": missing, "per_model": pm,
           "agreement": agreement(rows), "sonnet_rule": sonnet_rule(pm) if a.tag == "s3" else {"decidable": False,
                                                                                                  "why": "規則は s3 の項目だけで決める"},
           "rows": rows}
    for m, s in pm.items():
        print(f"{m}: n={s['n']} 受理 {s['valid_action']['rate']} 報告の正しさ {s['report_correct']['rate']} "
              f"{s['report_correct']['wilson95']} 手 {s['actions_final']} 要確認 {s['needs_manual']}")
    print(json.dumps(res["sonnet_rule"], ensure_ascii=False))
    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"score_{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        sheet = [{"row_id": r["row_id"], "report": r["report"], "failed_color": doc_item(doc, r["item_id"])["input"]["failed_color"],
                  "truth_box_at_request": doc_item(doc, r["item_id"])["truth"]["box_at_request"],
                  "action": r["decision"], "next_order": r["next_order"], "auto": {k: r["grade"][k] for k in ("c1", "c2", "c3")}}
                 for r in sorted(rows, key=lambda r: r["row_id"]) if r["grade"]["needs_manual"]]
        (OUT / f"sheet_{a.tag}.json").write_text(json.dumps(sheet, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"書いた: score_{a.tag}.json・sheet_{a.tag}.json（人が見る行 {len(sheet)}。モデル名は伏せた）")
    return 0


def doc_item(doc: dict, item_id: str) -> dict:
    return next(x for x in doc["items"] if x["item_id"] == item_id)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, models=True):
        p.add_argument("--tag", default="s3")
        if models:
            p.add_argument("--models", default="haiku,sonnet,opus")
            p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p = sub.add_parser("items", help="止まった場面の項目を作る（--write で書く）")
    common(p, models=False)
    p.add_argument("--dirs", nargs="*", default=None, help="E7 の条件のフォルダ（既定は outputs/v2eval/V3S3/E7_*）")
    p.add_argument("--allow-bundle1", action="store_true")
    p.add_argument("--write", action="store_true")
    p = sub.add_parser("cost", help="費用の見込み（--write で書く）")
    common(p)
    p.add_argument("--dirs", nargs="*", default=None)
    p.add_argument("--allow-bundle1", action="store_true")
    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD)
    p.add_argument("--write", action="store_true")
    p = sub.add_parser("ask", help="API に問う（--execute のときだけ）")
    common(p)
    p.add_argument("--execute", action="store_true", help="これが無ければ呼ばない（見込みだけ）")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--collect", nargs="?", const="ALL", default=None, help="Batch を回収する（ID なしは終わっていない全部）")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD)
    p = sub.add_parser("score", help="採点（--write で書く）")
    common(p)
    p.add_argument("--manual", default=None, help="人が付けた採点（{row_id: {c1, c2, c3}}）")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--write", action="store_true")
    return ap


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    a = build_parser().parse_args(argv)
    if not TAG_RE.match(a.tag):
        print("--tag は英数字と _ だけ", file=sys.stderr)
        return 2
    if a.tag == "s3" and getattr(a, "allow_bundle1", False):
        print("--tag s3 は段階 3 の記録だけ（束 1 の記録は別の tag にする。例: --tag d_e7）", file=sys.stderr)
        return 2
    try:
        return {"items": cmd_items, "cost": cmd_cost, "ask": cmd_ask, "score": cmd_score}[a.cmd](a)
    except SystemExit as e:                              # 前提の食い違い（項目が無い・束 1 の記録など）は終了コード 3
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
