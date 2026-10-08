"""介入率（束 6 (i)）: 段階 3 の 3 個の連続タスクの記録から、1 本ごとに「仕組みが手助けした回数」を種類別に数える。数字は手で書かない。

    .venv\\Scripts\\python.exe scripts\\56_intervention_s3.py          # → outputs/results/intervention_s3.json

読むもの: outputs/v2eval/V3S3/E7_R1v3/run_NNNN.json（試行の記録。harness/task_loop.py の meta）と run_NNNN_runtime.json の latency
（計算の口ごとの応答時間。llm の要素の数＝LLM を呼んだ回数）。段階 3 の確定した記録だけを読む（段階 4 の記録は読まない）。

■ 介入の定義（実行器 src/recovla/runtime/executor.py と harness/task_loop.py を読んで決めた）
  介入＝方策（VLA）に任せたままにせず、方策の外の仕組みが割り込んだ回数。4 種類に分け、LLM の呼び出しは介入とは別に数える。
  1. 決まった戻す動き（scripted_return）: 置いた後の戻す動き（executor.py 140〜142 行、_begin_return(t, "placed")）。
     俯瞰で箱の中に目標の色・指が開・手が待機位置から離れている、が return_to_retreat.trigger_s 続くと、方策を止めて
     決まった速さの動き（ReturnMotion）で待機位置へ戻す。記録は meta["returns"] の kind == "placed" の 1 件＝1 回。
  2. 出し直し（retry）: 手順の持ち時間 planner.step_timeout_s を超えたとき、待機位置へ戻してから同じ手順をやり直す
     （executor.py 143〜148 行、_begin_return(t, "retry") → 236〜237 行の _begin_step(attempt+1)）。
     数え方は steps[].attempts の数 − 1 の和（return_to_retreat が無効でも数えられる）。戻す動きが有効なときは
     meta["returns"] の kind == "retry" の数と一致することを確かめる（一致しなければ止める）。出し直しに含まれる戻す動きは
     1. に重ねて数えない。
     内訳として、時間切れの時点（その戻す動きの t_begin。無ければ次の試みの t_begin）で、目標の色がすでに箱の中に
     1 s 静止していた（truth_success_t がその時刻以前）ものを「判定だけのやり直し」（judge_only）、それ以外を
     「本当のやり直し」（genuine）とする。真値は評価の道具で、実行器は使っていない。
  3. 計画の変更（replan）: 段階 3 の実行器は、起動時の知覚の後に LLM で手順を 1 度だけ作り（executor.py 117〜135 行）、
     途中で作り直す道がない。記録の plan は 1 つ（dict）なので 0。plan が列（複数の計画）なら、その数 − 1 を数える。
     段階 4 の束 6 (ii)（runtime/executor_u4.py）の記録は plan を dict のまま書き換え、立て直しを meta["replans"] に残す。
     その行のうち intervention_kind == "plan_change"（next・reorder・skip・finish を通したもの）を 1 件＝1 回として足す
     （stop に倒したものは今の実行器と同じ止まり方なので数えない）。立て直しの後の戻す動き（returns の kind == "replan"）は
     計画の変更に含め、1. に重ねて数えない。その数は、続ける手を通した立て直しの数（applied.kind == "continue"）を超えない
     ことを確かめる（超えたら止める。戻す動きを切った場合・打ち切りで戻す途中だった場合は少なくてよい）。
  4. 判定の上書き（judge_override）: 完了判定（runtime.judge.JudgeV2）以外が手順の完了を決めた回数。段階 3 の実行器では、
     手順が完了になるのは判定が真になった時刻 done_t があるときだけ（executor.py 138〜139・238〜240 行）なので、
     記録で judged_complete が真なのに t_judge が無い手順を数える（段階 3 では 0 のはず）。
  別に数えるもの:
  - LLM の呼び出し（llm_calls）: runtime の latency["llm"] の要素の数（無ければ plan があれば 1）。キャッシュから返した数・
    トークン・応答時間も出す。
  - 止まって知らせた（stopped）: 出し直しても手順を終えられず、実行器が止まって返答で知らせた本数（executor.py 166〜181 行）。
    現場なら人が見に行く場面。うち 3 個とも実は箱に入っていた本数（完了判定の取りこぼし）も出す。
■ 成功の定義: 終わりに 3 個とも箱の中（meta["all_three_in_box"]。run.json の all_three・60_paper.py の e7_summary と同じ）。
  参考に、実行器が自分で全部終えたと判断した本数（止まらず、全手順の judged_complete が真）も出す。
■ success@k: 介入の合計が k 回以下で、かつ成功した本数 ÷ 全本数。k = 0, 1, …, 観測した最大（少なくとも 3）。
  参考に、判定だけのやり直しを数えない版（genuine_only）も出す。
■ 時間あたりの成功: シミュレーションの中の時間（meta["t_end"]。起動時の確かめ・LLM の待ちを含む。試行の間の並べ直しは含まない）
  の合計で割る。成功 1 本あたりの時間、1 時間あたりの成功・箱に入った立方体の数、時間 T までに 3 個とも入った割合（3 色の
  truth_success_t の最大 ≤ T）も出す。
■ 基準値（docs/local/strategy_20261008/final.md の束 6）: 1.05 回/本、LLM 1 回/本、@0 1/20・@1 5/20・@2 6/20。再現できたかを出す。
"""
import argparse
import collections
import importlib.util
import json
import math
import pathlib
import re
import sys
import time

from recovla.common import config
from recovla.common.seeds import COLORS

ROOT = config.ROOT
CFG = config.load_v2()
OUTPUTS = config.path(CFG["paths"]["outputs"])
E7_DIR = OUTPUTS / "v2eval" / "V3S3" / "E7_R1v3"
OUT = OUTPUTS / "results" / "intervention_s3.json"
KINDS = ("scripted_return", "retry", "replan", "judge_override")
LABELS = {"scripted_return": "決まった戻す動き", "retry": "出し直し（やり直し）", "replan": "計画の変更",
          "judge_override": "判定の上書き"}
RETURN_KIND = {"placed": "scripted_return", "retry": "retry",    # executor.py の _begin_return の kind → 介入の種類
               "replan": "replan"}                                # executor_u4.py（束 6 (ii)）の立て直しの後の戻す動き
BASELINE = {"source": "docs/local/strategy_20261008/final.md 束 6 (i)・diag_upper.md 1-1（work/upper/e7_interventions.json）",
            "interventions_per_run": 1.05, "llm_calls_per_run": 1.0, "success_at_k": {"0": 1, "1": 5, "2": 6}, "n": 20}
TIME_POINTS_S = (60.0, 90.0, 120.0, 150.0, 200.0)
EPS = 1e-9


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson の 95% 信頼区間（scripts/50_e_eval.py の _wilson と同じ式。読み込みを軽くするためここに置く）。"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ------------------------------------------------------------------ 1 本ごと

def count_run(meta: dict, latency: dict = None) -> dict:
    """1 本の記録（task_loop の meta）から介入を種類別に数える。latency は runtime の latency（無ければ None）。"""
    steps = meta.get("steps") or []
    plan = meta.get("plan")
    truth = meta.get("truth_success_t") or {}
    returns = meta.get("returns") or []
    kinds = collections.Counter(r.get("kind") for r in returns)
    unknown = sorted(set(kinds) - set(RETURN_KIND), key=str)
    if unknown:
        raise ValueError(f"run {meta.get('run')}: 知らない戻す動きの種類 {unknown}（定義を足してから数える）")
    retries = []
    for s in steps:
        att = s.get("attempts") or []
        for prev, nxt in zip(att, att[1:]):
            ret = [r for r in returns if r.get("kind") == "retry" and r.get("step") == s["step"]
                   and r.get("attempt") == prev["attempt"]]
            t_fire = float(ret[0]["t_begin"]) if ret else float(nxt["t_begin"])
            tt = truth.get(s["color"])
            judge_only = tt is not None and float(tt) <= t_fire + EPS
            retries.append({"step": int(s["step"]), "color": s["color"], "from_attempt": int(prev["attempt"]),
                            "t_fire": round(t_fire, 3), "judge_only": bool(judge_only)})
    if kinds["retry"] and kinds["retry"] != len(retries):
        raise ValueError(f"run {meta.get('run')}: 出し直しの数が合わない（returns {kinds['retry']}、attempts {len(retries)}）")
    replan = (len(plan) - 1) if isinstance(plan, list) else 0
    replans = meta.get("replans") or []
    plan_change = sum(1 for r in replans if r.get("intervention_kind") == "plan_change")
    n_continue = sum(1 for r in replans if (r.get("applied") or {}).get("kind") == "continue")
    if kinds["replan"] > n_continue:
        raise ValueError(f"run {meta.get('run')}: 立て直しの後の戻す動き {kinds['replan']} 回が、続ける手を通した立て直し "
                         f"{n_continue} 回より多い")
    replan += plan_change
    override = sum(1 for s in steps if s.get("judged_complete") and s.get("t_judge") is None)
    counts = {"scripted_return": int(kinds["placed"]), "retry": len(retries), "replan": int(max(0, replan)),
              "judge_override": int(override)}
    plan0 = plan[0] if isinstance(plan, list) and plan else plan
    plan_steps = (plan0 or {}).get("steps") or []
    if latency is not None and "llm" in latency:
        llm_calls = len(latency["llm"])
        llm_lat = [float(x) for x in latency["llm"]]
    else:
        llm_calls = 1 if plan0 else 0
        llm_lat = []
    usage = (plan0 or {}).get("usage") or {}
    in_box = meta.get("final_in_box") or {}
    t3 = [truth.get(c) for c in COLORS]
    stopped = meta.get("stopped")
    judged_all = (stopped is None and not meta.get("timed_out") and len(steps) == len(plan_steps) > 0
                  and all(s.get("judged_complete") for s in steps))
    return {
        "run": meta.get("run"), "seed": meta.get("seed"),
        "interventions": counts, "total": sum(counts.values()),
        "retry_judge_only": sum(r["judge_only"] for r in retries),
        "retry_genuine": sum(not r["judge_only"] for r in retries),
        "retries": retries,
        "llm_calls": llm_calls, "llm_from_cache": bool((plan0 or {}).get("from_cache")) if plan0 else False,
        "llm_latency_s": llm_lat, "llm_tokens": {"input": usage.get("input_tokens"), "output": usage.get("output_tokens")},
        "success": bool(meta.get("all_three_in_box")), "judged_all_done": bool(judged_all),
        "stopped": stopped is not None, "stopped_step": None if stopped is None else stopped.get("step"),
        "timed_out": bool(meta.get("timed_out")),
        "n_in_box": sum(bool(in_box.get(c)) for c in COLORS),
        "t_end": float(meta["t_end"]),
        "t_all_three": max(float(x) for x in t3) if all(x is not None for x in t3) else None,
    }


# ------------------------------------------------------------------ まとめ

def success_at_k(rows: list, key: str = "total", k_max: int = None) -> dict:
    """介入の合計（key）が k 以下で成功した本数。k = 0..k_max（既定は観測した最大と 3 の大きい方）。"""
    if k_max is None:
        k_max = max([3] + [int(r[key]) for r in rows])
    n = len(rows)
    out = {}
    for k in range(k_max + 1):
        s = sum(1 for r in rows if r["success"] and r[key] <= k)
        lo, hi = wilson(s, n)
        out[str(k)] = {"k": s, "n": n, "rate": round(s / n, 4) if n else None, "wilson95": [round(lo, 4), round(hi, 4)]}
    return out


def aggregate(rows: list, planner: dict = None) -> dict:
    n = len(rows)
    if n == 0:
        raise ValueError("記録が 0 本")
    tot = {k: sum(r["interventions"][k] for r in rows) for k in KINDS}
    total = sum(r["total"] for r in rows)
    for r in rows:
        r["genuine_total"] = r["total"] - r["retry_judge_only"]
    succ = sum(r["success"] for r in rows)
    t_sum = sum(r["t_end"] for r in rows)
    lat = [x for r in rows for x in r["llm_latency_s"]]
    tok_in = [r["llm_tokens"]["input"] for r in rows if r["llm_tokens"]["input"] is not None]
    tok_out = [r["llm_tokens"]["output"] for r in rows if r["llm_tokens"]["output"] is not None]
    t_succ = sorted(r["t_all_three"] for r in rows if r["success"] and r["t_all_three"] is not None)
    res = {
        "n": n,
        "settings": planner or {},
        "per_type": {k: {"label": LABELS[k], "total": tot[k], "per_run": round(tot[k] / n, 4)} for k in KINDS},
        "interventions_total": total, "interventions_per_run": round(total / n, 4),
        "distribution": {str(k): v for k, v in sorted(collections.Counter(r["total"] for r in rows).items())},
        "retry_breakdown": {"genuine": sum(r["retry_genuine"] for r in rows), "judge_only": sum(r["retry_judge_only"] for r in rows),
                            "judge_only_runs": [r["run"] for r in rows if r["retry_judge_only"]],
                            "by_color": dict(collections.Counter(x["color"] for r in rows for x in r["retries"]))},
        "llm": {"calls_total": sum(r["llm_calls"] for r in rows), "calls_per_run": round(sum(r["llm_calls"] for r in rows) / n, 4),
                "from_cache": sum(r["llm_from_cache"] for r in rows),
                "tokens_mean": {"input": round(sum(tok_in) / len(tok_in), 1) if tok_in else None,
                                "output": round(sum(tok_out) / len(tok_out), 1) if tok_out else None},
                "latency_s_mean": round(sum(lat) / len(lat), 3) if lat else None, "latency_n": len(lat)},
        "success": {"all_three_in_box": succ, "n": n, "wilson95": [round(x, 4) for x in wilson(succ, n)],
                    "judged_all_done": sum(r["judged_all_done"] for r in rows)},
        "success_at_k": success_at_k(rows, "total"),
        "success_at_k_genuine_only": success_at_k(rows, "genuine_total"),
        "stopped": {"n": sum(r["stopped"] for r in rows), "all_three_in_box": sum(r["stopped"] and r["success"] for r in rows),
                    "at_step": {str(k): v for k, v in sorted(collections.Counter(r["stopped_step"] for r in rows if r["stopped"]).items())},
                    "timed_out": sum(r["timed_out"] for r in rows)},
        "time": {
            "basis": "シミュレーションの中の時間（t_end の合計。起動時の確かめと LLM の待ちを含み、試行の間の並べ直しは含まない）",
            "sim_s_total": round(t_sum, 1), "sim_s_mean": round(t_sum / n, 1),
            "s_per_success": round(t_sum / succ, 1) if succ else None,
            "success_per_hour": round(succ / (t_sum / 3600.0), 2) if t_sum > 0 else None,
            "cubes_in_box_mean": round(sum(r["n_in_box"] for r in rows) / n, 3),
            "cubes_per_hour": round(sum(r["n_in_box"] for r in rows) / (t_sum / 3600.0), 1) if t_sum > 0 else None,
            "all_three_time_s": t_succ and {"min": round(t_succ[0], 1), "max": round(t_succ[-1], 1),
                                            "median": round(t_succ[len(t_succ) // 2] if len(t_succ) % 2 else
                                                            (t_succ[len(t_succ) // 2 - 1] + t_succ[len(t_succ) // 2]) / 2, 1)},
            "success_by_time": {f"{int(T)}": sum(1 for x in t_succ if x <= T + EPS) for T in TIME_POINTS_S},
        },
    }
    return res


def baseline_check(res: dict, base: dict = BASELINE) -> dict:
    sk = res["success_at_k"]
    items = {
        "interventions_per_run": (res["interventions_per_run"], base["interventions_per_run"]),
        "llm_calls_per_run": (res["llm"]["calls_per_run"], base["llm_calls_per_run"]),
        **{f"success_at_{k}": (sk[k]["k"] if k in sk else None, v) for k, v in base["success_at_k"].items()},
        "n": (res["n"], base["n"]),
    }
    out = {k: {"now": a, "baseline": b, "match": a is not None and abs(float(a) - float(b)) < 1e-6} for k, (a, b) in items.items()}
    return {"source": base["source"], "items": out, "all_match": all(v["match"] for v in out.values())}


def phrases(res: dict) -> dict:
    """提出物で使える言い方の案。値は res から差し込む（手で書かない）。開発中の記号・使わない語は入れない。"""
    n, sk = res["n"], res["success_at_k"]
    k0, k1, k2 = sk["0"]["k"], sk["1"]["k"], sk["2"]["k"]
    st, tm, rb = res["stopped"], res["time"], res["retry_breakdown"]
    other0 = all(res["per_type"][k]["total"] == 0 for k in ("scripted_return", "replan", "judge_override"))
    p = {
        "success_at_k": f"「全部片付けて」の {n} 回の作業で、やり直しなしで 3 個とも箱に入れたのは {k0} 回、"
                        f"仕組みが自動でやり直すのを 1 回まで許すと {k1} 回、2 回までで {k2} 回でした。",
        "per_run": f"手助けの回数は 1 回の作業あたり平均 {res['interventions_per_run']:.2f} 回"
                   + ("で、すべてが持ち時間を超えた後のやり直しでした。" if other0 else "でした。")
                   + f"そのうち {rb['judge_only']} 回は、立方体はもう箱に入っていて、完了の判定が出なかっただけのやり直しでした。",
        "llm": f"言葉の指示を手順に分ける LLM の呼び出しは、1 回の作業につき {res['llm']['calls_per_run']:g} 回（最初だけ）でした。",
        "per_100": f"100 回の作業に直すと、やり直しなしで 3 個とも片付くのは約 {round(100 * k0 / n)} 回、"
                   f"自動のやり直しを含めると約 {round(100 * res['success']['all_three_in_box'] / n)} 回で、"
                   f"残りの約 {100 - round(100 * res['success']['all_three_in_box'] / n)} 回は人が仕上げる必要があります"
                   f"（{n} 回の作業からの見積もりで、幅は大きい）。",
        "stopped": f"ロボットが止まって知らせたのは {n} 回中 {st['n']} 回で、そのうち {st['all_three_in_box']} 回は"
                   f"実は 3 個とも箱に入っていました（完了の判定の取りこぼし）。",
        "per_hour": f"シミュレーションの中の時間で 1 時間動かすと、3 個とも片付く作業は約 {tm['success_per_hour']:.0f} 回、"
                    f"箱に入る立方体は約 {tm['cubes_per_hour']:.0f} 個の割合でした。",
    }
    return p


def forbidden_hits(texts: dict) -> list:
    """60_paper.py の FORBIDDEN（使わない語）に当たる所。"""
    pm = _mod("paper", ROOT / "scripts" / "60_paper.py")
    return [{"key": k, "pattern": pat, "hit": m.group(0)} for k, t in texts.items() for pat in pm.FORBIDDEN
            for m in re.finditer(pat, t)]


def load_runs(d: pathlib.Path) -> list:
    rows = []
    for p in sorted(d.glob("run_[0-9][0-9][0-9][0-9].json")):
        meta = json.loads(p.read_text(encoding="utf-8"))
        rt = p.with_name(p.stem + "_runtime.json")
        lat = json.loads(rt.read_text(encoding="utf-8")).get("latency") if rt.is_file() else None
        rows.append(count_run(meta, lat))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(E7_DIR))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    d = pathlib.Path(a.dir)
    rows = load_runs(d)
    pl = CFG.get("planner", {})
    planner = {"step_timeout_s": pl.get("step_timeout_s"), "retry": pl.get("retry"),
               "return_to_retreat": pl.get("return_to_retreat"), "source": "configs/default.yaml の planner"}
    res = aggregate(rows, planner)
    res["definitions"] = {k: LABELS[k] for k in KINDS}
    res["definitions_note"] = "定義はこのスクリプトの冒頭。LLM の呼び出しは介入に数えない。成功は終わりに 3 個とも箱の中"
    res["baseline_check"] = baseline_check(res)
    res["phrases"] = phrases(res)
    res["phrases_forbidden_hits"] = forbidden_hits(res["phrases"])
    res["per_run"] = [{k: v for k, v in r.items() if k != "retries"} | {"retries": r["retries"]} for r in rows]
    res["read_from"] = str(d)
    res["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    show = {k: res[k] for k in ("n", "per_type", "interventions_per_run", "distribution", "retry_breakdown", "llm", "success",
                                "success_at_k", "success_at_k_genuine_only", "stopped", "time")}
    print(json.dumps(show, ensure_ascii=False, indent=1))
    print(json.dumps(res["baseline_check"], ensure_ascii=False, indent=1))
    print(json.dumps(res["phrases"], ensure_ascii=False, indent=1))
    print("使わない語:", res["phrases_forbidden_hits"] or "なし")
    return 0


if __name__ == "__main__":
    sys.exit(main())
