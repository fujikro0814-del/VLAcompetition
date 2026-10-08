"""段階 4 束 6 (iii): VLM のオフラインの試験台（研究の道・記述だけ）。手順書 docs/stage4/vlm_bench_protocol.md。

使い方（作業場所 C:\\PAI\\recovery_vla、Python は .venv\\Scripts\\python.exe）:
    57_vlm_bench.py items  [--task all|completion|failure] [--include-s4 S4DREC] [--dry-run]
        記録から問いの項目と正解を作り outputs/s4/vlm/items_<課題>.json に書く（--dry-run は数だけ）
    57_vlm_bench.py render --task completion --view overhead256 [--limit N] [--min-free-gb 14] [--dry-run]
        除外でない項目を描き、outputs/s4/vlm/images/<課題>/<視点>/ に PNG と manifest.json（SHA-256）を書く
    57_vlm_bench.py render --smoke        1 つの記録（E7_R1v3 の run_0015）から 3 枚だけ（outputs/s4/vlm/smoke/）
    57_vlm_bench.py cost  [--budget 30]   本番の見込み（問いの数 x 1 問のトークン x 単価。Batch は半額）
    57_vlm_bench.py ask   --task completion --view overhead256 --models haiku,sonnet,opus
                          (--sync [--limit N] | --latency-sample 20 | --batch [--no-wait] | --collect [BATCH_ID]) [--budget 30] [--dry-run]
    57_vlm_bench.py ask   --smoke --models haiku     smoke の画像で 1 問（同期）
    57_vlm_bench.py score [--task all]    outputs/s4/vlm/score_<課題>.json
本番の順（手順書 9 節）: items → render（4 通り）→ cost → ask --latency-sample 20（主の視点）→ ask --batch（全部）→ score。
既存のファイルは書き換えない。書くのは outputs/s4/vlm/ 以下だけ。束 1 の記録（S4DREC など）は --include-s4 を付けたときだけ読む。
終了コード: 0 正常、2 エラー、3 前提の食い違い（メモリ不足・予算超え・画像が無いなど）。
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recovla.vlm import ask as A          # noqa: E402
from recovla.vlm import cost as C         # noqa: E402
from recovla.vlm import labels as L       # noqa: E402

OUT = A.OUT
PLAN = (("completion", "overhead256", True), ("completion", "overhead512", False),
        ("failure", "presentation", True), ("failure", "overhead512", False))     # (課題, 視点, 主か)
N_IMAGES = {"completion": 1, "failure": len(L.FRAME_OFFSETS)}
LATENCY_N = 20
SMOKE_ITEM = "V3S3_E7_R1v3_r0015_s2_ref"       # 取りこぼしの場面（青が箱の中、実行器は未完了）
SMOKE_VIEWS = ("overhead256", "overhead512", "presentation")
FINAL_MD_FN_RUNS = (6, 10, 13, 15)             # 掲示板 0144〜0145・final.md の「取りこぼし 4 件」


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def tasks_of(a) -> list:
    return ["completion", "failure"] if a.task == "all" else [a.task]


def items_path(task: str, out=None) -> pathlib.Path:
    return pathlib.Path(out or OUT) / f"items_{task}.json"


def load_items(task: str, out=None) -> dict:
    p = items_path(task, out)
    if not p.is_file():
        raise SystemExit(f"{p} が無い（先に items）")
    return json.loads(p.read_text(encoding="utf-8"))


def build(task: str, include_s4=()) -> dict:
    return L.build_completion() if task == "completion" else L.build_failure(include_s4=tuple(include_s4))


def summarize(doc: dict) -> dict:
    import collections
    inc = L.included(doc["items"])
    s = {"task": doc["task"], "sources": doc["sources"], "items": len(doc["items"]), "included": len(inc),
         "labels": dict(collections.Counter(str(x["label"]) for x in inc)),
         "excluded": dict(collections.Counter(e for x in doc["items"] for e in x["exclude"])),
         "frames": sum(len(x["frames"]) for x in inc),
         "joints_source": dict(collections.Counter(f.get("joints_source") for x in inc for f in x["frames"]))}
    if doc["task"] == "completion":
        s["fn_steps"] = [(x["run"], x["step"], x["color"]) for x in doc["steps"] if x["fn_scene"]]
        s["fp_steps"] = [(x["run"], x["step"], x["color"]) for x in doc["steps"] if x["fp_scene"]]
        s["fn_points"] = [x["item_id"] for x in inc if x["scene"]["fn_point"]]
    else:
        by = collections.defaultdict(collections.Counter)
        for x in inc:
            by[x["source"]][x["label"]] += 1
        s["by_source"] = {k: dict(v) for k, v in sorted(by.items())}
        s["label_source"] = dict(collections.Counter(x["label_source"] for x in inc))
    return s


# ---------------------------------------------------------------- items
def cmd_items(a) -> int:
    for task in tasks_of(a):
        doc = build(task, a.include_s4)
        s = summarize(doc)
        print(json.dumps(s, ensure_ascii=False, indent=1))
        if not a.dry_run:
            OUT.mkdir(parents=True, exist_ok=True)
            doc["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
            doc["summary"] = s
            items_path(task).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            log(f"書いた {items_path(task)}")
    return 0


# ---------------------------------------------------------------- render
def memory_ok(min_gb: float) -> bool:
    from recovla.vlm.render import free_phys_gb
    g = free_phys_gb()
    log(f"空きの物理メモリ {g if g is None else round(g, 2)} GB（下限 {min_gb} GB）")
    return g is None or g >= min_gb


def cmd_render(a) -> int:
    if a.smoke:
        doc = L.build_completion()
        doc["items"] = [x for x in doc["items"] if x["item_id"] == SMOKE_ITEM]
        if a.dry_run:
            print(json.dumps(doc["items"], ensure_ascii=False, indent=1)[:4000])
            return 0
        if not memory_ok(a.min_free_gb):
            return 3
        from recovla.vlm import render as R
        res = {}
        for v in SMOKE_VIEWS:
            t0 = time.perf_counter()
            m = R.render_items(doc, v, OUT / "smoke" / "images" / "completion" / v, log=log)
            res[v] = {"s": round(time.perf_counter() - t0, 2), "files": m["items"]}
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    doc = load_items(a.task)
    inc = L.included(doc["items"])
    n = len(inc) if a.limit is None else min(a.limit, len(inc))
    log(f"{a.task} {a.view}: 項目 {n}、画像 {sum(len(x['frames']) for x in inc[:n])} 枚")
    if a.dry_run:
        return 0
    if not memory_ok(a.min_free_gb):
        return 3
    from recovla.vlm import render as R
    t0 = time.perf_counter()
    R.render_items(doc, a.view, OUT / "images" / a.task / a.view, limit=a.limit, log=log)
    log(f"描き終わり {round(time.perf_counter() - t0, 1)} s")
    return 0


# ---------------------------------------------------------------- 問いの集め方
def model_ids(a) -> list:
    return [A.MODELS.get(m, m) for m in a.models.split(",")]


def gather(task: str, view: str, model: str, out_root=None) -> list:
    root = out_root or OUT
    man = A.load_manifest(task, view, root)
    if out_root is not None and str(out_root).endswith("smoke"):
        doc = L.build_completion()
    else:
        doc = load_items(task)
    return A.questions(task, view, model, L.included(doc["items"]), man, root)


def latency_sample(qs: list, n: int) -> list:
    return sorted(qs, key=lambda q: hashlib.sha256(q["item"]["item_id"].encode()).hexdigest())[:n]


def calib_from_cache() -> dict:
    """キャッシュの実測（同期・Batch とも）から (モデル, 課題, 視点の大きさ) ごとの入力・出力のトークンの中央値。"""
    import numpy as np
    from recovla.vlm.render import VIEWS
    acc = {}
    if not A.CACHE.is_dir():
        return {}
    for p in A.CACHE.glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        u = r.get("usage") or {}
        if r.get("view") not in VIEWS or "input_tokens" not in u:
            continue
        k = (r["model_requested"], r["task"], C.view_out_key(VIEWS[r["view"]]["out"]))
        acc.setdefault(k, []).append((u["input_tokens"], u.get("output_tokens", 0)))
    return {k: {"input_tokens": float(np.median([x[0] for x in v])), "output_tokens": float(np.median([x[1] for x in v])),
                "n": len(v)} for k, v in acc.items()}


def cmd_ask(a) -> int:
    models = model_ids(a)
    out_root = OUT / "smoke" if a.smoke else None
    task, view = ("completion", a.view or "overhead256") if a.smoke else (a.task, a.view)
    from recovla.vlm.render import VIEWS
    try:
        allq = {m: gather(task, view, m, out_root) for m in models}
    except FileNotFoundError as e:
        log(f"前提が無い: {e}")
        return 3
    if a.latency_sample:
        allq = {m: latency_sample(q, a.latency_sample) for m, q in allq.items()}
    elif a.limit is not None:
        allq = {m: q[:a.limit] for m, q in allq.items()}
    todo = {m: [q for q in qs if A.cache_get(q["key"]) is None] for m, qs in allq.items()}
    batch = bool(a.batch)
    calib = calib_from_cache()
    plan = [{"task": task, "view": view, "view_out": tuple(VIEWS[view]["out"]), "n_images": N_IMAGES[task] if not a.smoke else 1,
             "model": m, "n_questions": len(todo[m]), "batch": batch} for m in models]
    est = C.estimate(plan, calib, batch=batch, budget=a.budget, spent=A.spent_usd())
    log(f"{task} {view}: 問い {sum(len(q) for q in allq.values())}、キャッシュに無い {sum(len(q) for q in todo.values())}、"
        f"見込み ${est['total_usd']}（これまで ${est['spent_usd']}、上限 ${a.budget}）")
    if a.dry_run:
        print(json.dumps(est, ensure_ascii=False, indent=1))
        return 0
    if not est["within_budget"]:
        log("予算の上限を超える見込みなので止める（--budget）")
        return 3
    from recovla.planner.decompose import client
    cli = client().with_options(timeout=A.SYNC_TIMEOUT_S, max_retries=2)
    if a.collect is not None:
        ids = [a.collect] if a.collect else [p.stem for p in A.BATCHES.glob("*.json")
                                            if json.loads(p.read_text(encoding="utf-8")).get("status") != "ended"]
        for bid in ids:
            A.collect_batch(bid, cli, wait=not a.no_wait, log=log)
        return 0
    if batch:
        recs = []
        for m in models:
            recs += A.submit_batches(todo[m], cli, out=out_root, log=log)
        if not a.no_wait:
            for r in recs:
                A.collect_batch(r["batch_id"], cli, log=log)
        return 0
    for m in models:
        A.ask_sync(allq[m], cli, out=out_root, log=log)
    return 0


# ---------------------------------------------------------------- cost
def cmd_cost(a) -> int:
    from recovla.vlm.render import VIEWS
    counts = {}
    for task in ("completion", "failure"):
        p = items_path(task)
        doc = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else build(task, a.include_s4)
        counts[task] = len(L.included(doc["items"]))
    calib = calib_from_cache()
    plan = []
    for task, view, main in PLAN:
        for m in A.MODELS.values():
            plan.append({"task": task, "view": view, "view_out": tuple(VIEWS[view]["out"]), "n_images": N_IMAGES[task],
                         "model": m, "n_questions": counts[task], "batch": True})
            if main:
                plan.append({"task": task, "view": view, "view_out": tuple(VIEWS[view]["out"]), "n_images": N_IMAGES[task],
                             "model": m, "n_questions": min(LATENCY_N, counts[task]), "batch": False, "latency_sample": True})
    est = C.estimate(plan, calib, budget=a.budget, spent=A.spent_usd())
    est["counts"] = counts
    est["calib"] = {"|".join(k): v for k, v in calib.items()}
    est["n_questions_total"] = sum(p["n_questions"] for p in plan)
    est["n_images_to_render"] = {f"{t}/{v}": counts[t] * N_IMAGES[t] for t, v, _ in PLAN}
    est["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost_estimate.json").write_text(json.dumps(est, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: est[k] for k in ("counts", "n_questions_total", "n_images_to_render", "total_usd", "spent_usd",
                                          "budget_usd", "within_budget")}, ensure_ascii=False, indent=1))
    for r in est["rows"]:
        print(f"  {r['task']:10s} {r['view']:12s} {r['model']:18s} {'sync ' if not r['batch'] else 'batch'} "
              f"{r['n_questions']:5d} 問 x ${r['usd_per_question']:.5f} = ${r['usd']:.3f}（{r['tokens_per_question']['source']}）")
    return 0 if est["within_budget"] else 3


# ---------------------------------------------------------------- score
def cmd_score(a) -> int:
    from recovla.vlm import score as S
    for task in tasks_of(a):
        res = {"task": task, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "results": {}}
        for t, view, main in PLAN:
            if t != task:
                continue
            for m in A.MODELS.values():
                try:
                    qs = gather(task, view, m)
                except FileNotFoundError:
                    continue
                rows = S.answer_rows(qs, A.cache_get)
                fn = S.score_completion if task == "completion" else S.score_failure
                r = fn(rows)
                if task == "failure":
                    r["by_source"] = S.by_group(rows, "condition", S.score_failure)
                else:                                   # final.md の取りこぼし 4 本（6・10・13・15）の場面だけの数え方も出す
                    sub = {k: v for k, v in r["fn_points"]["items"].items()
                           if any(f"_r{i:04d}_" in k for i in FINAL_MD_FN_RUNS)}
                    r["fn_points_final_md"] = {"runs": list(FINAL_MD_FN_RUNS), "n": len(sub),
                                               "vlm_yes": sum(1 for v in sub.values() if v == "yes")}
                res["results"][f"{view}|{m}"] = dict(r, main_view=main)
        (OUT / f"score_{task}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        for k, r in res["results"].items():
            print(task, k, "一致率", r["accuracy"], "遅れ中央値", r["latency_s_median"], "1問 $", r["usd_per_question"],
                  "照合", r["dual_check"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("items")
    p.add_argument("--task", default="all", choices=("all", "completion", "failure"))
    p.add_argument("--include-s4", nargs="*", default=[])
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("render")
    p.add_argument("--task", default="completion", choices=("completion", "failure"))
    p.add_argument("--view", default="overhead256", choices=("overhead256", "overhead512", "presentation"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-free-gb", type=float, default=14.0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("cost")
    p.add_argument("--budget", type=float, default=C.DEFAULT_BUDGET_USD)
    p.add_argument("--include-s4", nargs="*", default=[])
    p = sub.add_parser("ask")
    p.add_argument("--task", default="completion", choices=("completion", "failure"))
    p.add_argument("--view", default=None, choices=("overhead256", "overhead512", "presentation"))
    p.add_argument("--models", default="haiku,sonnet,opus")
    p.add_argument("--sync", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--latency-sample", type=int, default=None)
    p.add_argument("--batch", action="store_true")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--collect", nargs="?", const="", default=None)
    p.add_argument("--budget", type=float, default=C.DEFAULT_BUDGET_USD)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("--task", default="all", choices=("all", "completion", "failure"))
    a = ap.parse_args(argv)
    if a.cmd == "ask" and not a.smoke and a.view is None:
        ap.error("ask には --view が要る（--smoke を除く）")
    try:
        return {"items": cmd_items, "render": cmd_render, "cost": cmd_cost, "ask": cmd_ask, "score": cmd_score}[a.cmd](a)
    except SystemExit:
        raise
    except Exception as e:                      # noqa: BLE001
        import traceback
        traceback.print_exc()
        log(f"エラー: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
