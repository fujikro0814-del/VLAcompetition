"""VLM の試験台の採点（記述だけ。検定はしない。段階 3・段階 4 の主な結果・検定の族には入れない。手順書 7 節）。

    rows = answer_rows(questions, cache_get)     # 問い 1 つ = 1 行（正解・答え・usage・費用・時間）
    s = score_completion(rows)  /  score_failure(rows)
    s["dual_check"]                               # 独立な 2 つ目の実装（_counts_alt）と件数が一致したか

完了の判定（正解 = 箱の中に置かれている。答え yes / no / unsure、形の崩れた答えは invalid）:
  一致率 = (正解が真で yes + 正解が偽で no) / 問いの数（unsure・invalid は外れに数える。率も別に出す）
  取りこぼし = 正解が真なのに yes でない（no・unsure・invalid）、誤検出（誤った完了）= 正解が偽なのに yes
  実行器が決めた時刻（executor.decision が真偽）だけの部分集合で、実行器と VLM の取りこぼし・誤検出を並べる
  取りこぼしの場面（scene.fn_point）の VLM の yes の数
失敗の種類の分類（5 種類。答えが 5 つ以外は invalid）:
  一致率、種類ごとの再現率・適合率、マクロ平均の再現率、混同行列（正解 x 答え＋invalid）
共通: 遅れの中央値（同期の呼び出しの wall_s だけ。Batch は 1 問ごとの時間が無い）、1 問あたりの費用（実際の usage と、
  呼んだ形 sync・batch の単価）、トークンの中央値、拒否（stop_reason "refusal"）の数。
"""
import collections

import numpy as np

from recovla.vlm import labels as L

COMPLETION_ANSWERS = ("yes", "no", "unsure", "invalid")
FAILURE_ANSWERS = L.CLASSES + ("invalid",)


def answer_rows(qs: list, get) -> list:
    """qs は ask.questions の出力。get(鍵) はキャッシュの記録か None。答えの無い問いは answered=False。"""
    rows = []
    for q in qs:
        rec = get(q["key"])
        it = q["item"]
        ans = None if rec is None or rec.get("parsed") is None else rec["parsed"]["answer"]
        rows.append({"item_id": it["item_id"], "task": q["task"], "view": q["view"], "model": q["model"],
                     "label": it["label"], "answered": rec is not None,
                     "answer": "invalid" if rec is not None and ans is None else ans,
                     "executor_decision": (it.get("executor") or {}).get("decision"),
                     "fn_point": (it.get("scene") or {}).get("fn_point", False),
                     "condition": it.get("condition"), "label_source": it.get("label_source"),
                     "usage": (rec or {}).get("usage") or {}, "usd": (rec or {}).get("usd"), "mode": (rec or {}).get("mode"),
                     "wall_s": (rec or {}).get("wall_s"), "stop_reason": (rec or {}).get("stop_reason"),
                     "model_returned": (rec or {}).get("model")})
    return rows


def _frac(k, n):
    return {"k": int(k), "n": int(n), "rate": None if n == 0 else round(k / n, 4)}


def _common(rows: list) -> dict:
    ans = [r for r in rows if r["answered"]]
    walls = [float(r["wall_s"]) for r in ans if r["mode"] == "sync" and r["wall_s"] is not None]
    usd = [float(r["usd"]) for r in ans if r["usd"] is not None]
    tin = [r["usage"].get("input_tokens") for r in ans if r["usage"].get("input_tokens") is not None]
    tout = [r["usage"].get("output_tokens") for r in ans if r["usage"].get("output_tokens") is not None]
    return {"n_questions": len(rows), "n_answered": len(ans), "n_unanswered": len(rows) - len(ans),
            "modes": dict(collections.Counter(r["mode"] for r in ans)),
            "latency_s_median": None if not walls else round(float(np.median(walls)), 3), "latency_n": len(walls),
            "latency_s_p90": None if not walls else round(float(np.percentile(walls, 90)), 3),
            "usd_total": round(sum(usd), 6), "usd_per_question": None if not usd else round(sum(usd) / len(usd), 6),
            "input_tokens_median": None if not tin else float(np.median(tin)),
            "output_tokens_median": None if not tout else float(np.median(tout)),
            "refusals": sum(1 for r in ans if r["stop_reason"] == "refusal"),
            "models_returned": sorted({str(r["model_returned"]) for r in ans})}


# ---------------------------------------------------------------- 完了の判定
def _completion_counts(rows: list) -> dict:
    """実装 1: 行を 1 つずつ見て数える。"""
    c = {"tp": 0, "tn": 0, "fn": 0, "fp": 0, "unsure": 0, "invalid": 0, "n": 0}
    conf = {str(lab): {a: 0 for a in COMPLETION_ANSWERS} for lab in (True, False)}
    for r in rows:
        if not r["answered"]:
            continue
        c["n"] += 1
        a = r["answer"]
        conf[str(bool(r["label"]))][a] += 1
        if a in ("unsure", "invalid"):
            c[a] += 1
        if r["label"]:
            c["tp" if a == "yes" else "fn"] += 1
        else:
            c["fp" if a == "yes" else "tn"] += 1
    return {"counts": c, "confusion": conf}


def _completion_counts_alt(rows: list) -> dict:
    """実装 2（照合用。配列で数える）。"""
    ans = [r for r in rows if r["answered"]]
    lab = np.array([bool(r["label"]) for r in ans], bool)
    yes = np.array([r["answer"] == "yes" for r in ans], bool)
    a = np.array([r["answer"] for r in ans], dtype=object)
    return {"tp": int((lab & yes).sum()), "fn": int((lab & ~yes).sum()), "fp": int((~lab & yes).sum()),
            "tn": int((~lab & ~yes).sum()), "unsure": int((a == "unsure").sum()), "invalid": int((a == "invalid").sum()),
            "n": len(ans)}


def score_completion(rows: list) -> dict:
    k = _completion_counts(rows)
    c = k["counts"]
    alt = _completion_counts_alt(rows)
    pos, neg = c["tp"] + c["fn"], c["tn"] + c["fp"]
    out = {"accuracy": _frac(c["tp"] + c["tn"], c["n"]), "miss_rate": _frac(c["fn"], pos),
           "false_complete_rate": _frac(c["fp"], neg), "unsure": _frac(c["unsure"], c["n"]),
           "invalid": _frac(c["invalid"], c["n"]), "counts": c, "confusion": k["confusion"],
           "dual_check": alt == c, "dual_alt": alt}
    dec = [r for r in rows if r["answered"] and r["executor_decision"] is not None]
    ex_fn = sum(1 for r in dec if r["label"] and r["executor_decision"] is False)
    ex_fp = sum(1 for r in dec if not r["label"] and r["executor_decision"] is True)
    vk = _completion_counts(dec)["counts"]
    out["decision_points"] = {"n": len(dec), "positives": sum(1 for r in dec if r["label"]),
                              "executor": {"miss": ex_fn, "false_complete": ex_fp},
                              "vlm": {"miss": vk["fn"], "false_complete": vk["fp"], "unsure": vk["unsure"]}}
    fnp = [r for r in rows if r["answered"] and r["fn_point"]]
    out["fn_points"] = {"n": len(fnp), "vlm_yes": sum(1 for r in fnp if r["answer"] == "yes"),
                        "items": {r["item_id"]: r["answer"] for r in fnp}}
    out.update(_common(rows))
    return out


# ---------------------------------------------------------------- 失敗の種類の分類
def _failure_counts(rows: list) -> dict:
    conf = {lab: {a: 0 for a in FAILURE_ANSWERS} for lab in L.CLASSES}
    for r in rows:
        if r["answered"]:
            conf[r["label"]][r["answer"]] += 1
    return conf


def _failure_counts_alt(rows: list) -> dict:
    cnt = collections.Counter((r["label"], r["answer"]) for r in rows if r["answered"])
    return {lab: {a: cnt.get((lab, a), 0) for a in FAILURE_ANSWERS} for lab in L.CLASSES}


def score_failure(rows: list) -> dict:
    conf = _failure_counts(rows)
    alt = _failure_counts_alt(rows)
    n = sum(sum(v.values()) for v in conf.values())
    correct = sum(conf[c][c] for c in L.CLASSES)
    per = {}
    for c in L.CLASSES:
        support = sum(conf[c].values())
        predicted = sum(conf[x][c] for x in L.CLASSES)
        per[c] = {"recall": _frac(conf[c][c], support), "precision": _frac(conf[c][c], predicted), "support": support}
    rec = [per[c]["recall"]["rate"] for c in L.CLASSES if per[c]["support"] > 0]
    out = {"accuracy": _frac(correct, n), "per_class": per,
           "macro_recall": None if not rec else round(float(np.mean(rec)), 4),
           "failure_vs_success": _frac(sum(conf[c][a] for c in L.FAILURE_CLASSES for a in L.FAILURE_CLASSES)
                                       + conf["success"]["success"], n),
           "invalid": _frac(sum(conf[c]["invalid"] for c in L.CLASSES), n),
           "confusion": conf, "dual_check": alt == conf}
    out.update(_common(rows))
    return out


def by_group(rows: list, key: str, fn) -> dict:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {str(k): fn(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}
