"""種の台帳（docs/種の台帳.md）と、outputs 以下の記録から抜き出した「使った種」を照らす（段階 4・束 0・台帳役）。

使い方（作業場所 C:\\PAI\\recovery_vla。CPU だけ。シミュレーション・GPU は使わない）:
    .venv\\Scripts\\python.exe scripts\\97_s4_ledger_check.py
        全体の照合。結果を outputs\\s4\\ledger_check.json に書く。
    .venv\\Scripts\\python.exe scripts\\97_s4_ledger_check.py --bands 160000-169999 170000-179999
        任意の帯が未使用か（記録にも台帳の「使用済み」にも無いか）を確かめ、結果の "bands" に入れる。
        全体の照合も同時に行う。帯の確認だけを終了コードにしたいときは --bands-only を付ける。
    .venv\\Scripts\\python.exe scripts\\97_s4_ledger_check.py --self-test
        抜き出し・照合の部品の単体検査（記録を読まない）。
    オプション: --ledger <台帳のパス>  --out <出力のパス>  --root <リポジトリの根>

終了コード: 0 = 問題なし／1 = 問題あり（出力の "problems" を見る）／2 = 台帳の読み取りの失敗など。

読むもの（すべて読み取りだけ。記録は書き換えない）:
  - docs/種の台帳.md の末尾の節「照合用の対応表」（機械が読む表）。本文の表との食い違いは
    「prose_not_in_table」に出す（本文にだけある範囲の検出。警告）。
  - outputs 以下の記録（outputs/sealed は開かない。名前の数だけ数える）と docs/results の json:
      trial_*.json / task_*.json / run_*.json（run_*_runtime は除く）の seed、
      run.json の trials・trials_spec・seeds、E6 の json（trials・rows）、
      gen の meta.json・generation.jsonl・handover.jsonl・r2_generation.jsonl・エピソードの
      フォルダ名、manifests のキー、datasets / train の conversion.json の layout_seed、
      demo_v2 / demo の meta.json、planner・g・k1・hand・perception・v2judge・results などの
      seed / seeds / layout_seed、results の csv の seed 列、
      queue の json（--trials の指定。弱い証拠として別に扱う）、
      train_run.json などの学習の種（配置の種とは別に集計し、台帳との照合には使わない）。
書くもの: outputs/s4/ledger_check.json だけ（--out で変更可）。

照合の中身:
  not_in_ledger              記録にあるのに、台帳の「使用済み」に入っていない種（台帳にない使用）
  ledger_used_without_record 台帳が「使用済み」で記録を期待しているのに、記録が 1 件も無い範囲
  planned_with_use           台帳が「予定」「予約」の範囲の中に、台帳の「使用済み」に載っていない使用の記録がある
                             （段階 4 の予定を含む。載っている使用は警告 planned_with_registered_use に並べる）
  banned_with_use            台帳が「使わない」「取り消し」の範囲の中に使用の記録がある
  ledger_inconsistency       対応表の中の矛盾（使用済み同士の重なり、予定の帯からはみ出す使用済みの行、
                             取り消し・使わないとの重なり、表の書式の誤り）
警告（終了コードに入れない）: prose_not_in_table（本文にだけある範囲）、ledger_used_partial_record（使用済みの
範囲の一部にしか記録がない）、status_mismatch_note（『記録なし』と注記したのに記録がある）、
planned_with_registered_use（予定の帯の中の、台帳に載せた使用）。
結果を見る前に決めた項目: 上の区分と終了コードの規則、記録の読み方（どのキー・ファイル名を使用とみなすか）。
見た後に決めた項目: 対応表の中身（記録から確かめた範囲を台帳に書いた。『なし』と注記した範囲の理由）、
クリップのフォルダ名の読み取り（demo_v2・V3DEMO の名前に種がある形を見て足した）、
使用済みの行が予定の帯に収まる場合を矛盾から外す規則（S4SMOKE が 44400 を使ったのを見て決めた）。
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = ROOT / "docs" / "種の台帳.md"
DEFAULT_OUT = ROOT / "outputs" / "s4" / "ledger_check.json"

# 配置の種として数える最小値。これ未満の seed（学習の種 1000 など）は low_seeds に分けて報告するだけ。
MIN_LAYOUT_SEED = 10000
MAX_LAYOUT_SEED = 999999      # これを超える seed（日付の乱数の種 20261005 など）は high_seeds に分けて報告するだけ。
MAX_RANGE_LEN = 5000         # [a, b] の 2 要素を範囲とみなすときの上限（超えたら 2 つの種として扱う）

SEED_INT_KEYS = {"seed", "layout_seed", "trial_seed", "last_seed_used", "replay_order_seed"}
SEED_LIST_KEYS = {"seeds", "layout_seeds", "seeds_used", "placed_seeds"}
SPEC_KEYS = {"trials", "trials_spec"}
SKIP_FILE_RE = re.compile(
    r"^(trial_\d+_runtime|run_\d+_runtime|runtime_\d+|G_AUDIT|config|train_config|"
    r"policy_preprocessor|policy_postprocessor|optimizer_param_groups|scheduler_state|"
    r"training_step|info|stats)\.json$"
)
TRIAL_FILE_RE = re.compile(r"^(trial|task|run)_\d+\.json$")
TRAIN_FILE_RE = re.compile(r"(^train_|train_run\.json$|train_launch_config\.json$|g0_train\d+\.json$)")
EPISODE_DIR_RE = re.compile(r"(?:^|_)(\d{5,6})_(?:red|green|blue)_r\d+$|^R2P\d_(\d{5,6})_")
# 動画・描き直しのクリップのフォルダ名（例: demo_v2/nat_140011_R、V3DEMO/p1_141004_N、demo/task_115000）
CLIP_DIR_RE = re.compile(r"^(?:nat|p1|p2|p3|task)_(\d{6})(?:_|$)")
CLIP_PARENTS = {"demo", "demo_v2", "V3DEMO"}
MANIFEST_KEY_RE = re.compile(r"(?:^|_)(\d{5,6})_(?:red|green|blue)_r\d+$|^R2P\d_(\d{5,6})_")
SPEC_RE = re.compile(r"^(?:[A-Za-z0-9_\-]+:)?(\d{3,7}):(\d+)$")
# ファイル名に種が入っている記録（npz・mp4 など。例: expert_n_58000_red.npz、policy_R2_P1_000_58200_red.npz）
FILENAME_SEED_RE = re.compile(r"(?:^|_)(\d{5,6})_(?:red|green|blue)(?:_|\.|$)")
# docs/results の注記の文章にある種の範囲（例: 学習用のシード 59010〜59029）
NOTE_RANGE_RE = re.compile(r"(?:シード|種)\s*(\d{5,6})\s*〜\s*(\d{5,6})")

TIER_TRIAL, TIER_RUN, TIER_DERIVED, TIER_QUEUE = "trial", "run", "derived", "queue"
USED_TIERS = {TIER_TRIAL, TIER_RUN, TIER_DERIVED}

STATUS_JA = {
    "使用済み": "used",
    "予定": "planned",
    "予約": "reserved",
    "取り消し": "void",
    "使わない": "banned",
}


# ----------------------------------------------------------------------------- 範囲の部品
def merge_ranges(ranges):
    out = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def seeds_to_ranges(seeds):
    return merge_ranges((s, s) for s in seeds)


def in_ranges(x, merged):
    starts = [r[0] for r in merged]
    i = bisect.bisect_right(starts, x) - 1
    return i >= 0 and merged[i][0] <= x <= merged[i][1]


def parse_spec(text):
    """'natural:199230:33' / '115000:20' / '59000:10' → (開始, 個数)。複数は , で区切る。"""
    out = []
    if not isinstance(text, str):
        return out
    for part in re.split(r"[,\s]+", text.strip()):
        m = SPEC_RE.match(part)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def seeds_from_listlike(v):
    """seeds の値（文字列の指定・[a,b]・種の列・[[種, 色]…]）から整数の集合を返す。"""
    out = set()
    if isinstance(v, str):
        for s, n in parse_spec(v):
            out.update(range(s, s + n))
    elif isinstance(v, list):
        if len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            a, b = v
            if 0 <= b - a <= MAX_RANGE_LEN:
                out.update(range(a, b + 1))
            else:
                out.update(v)
        else:
            for x in v:
                if isinstance(x, int) and not isinstance(x, bool):
                    out.add(x)
                elif isinstance(x, list) and x and isinstance(x[0], int) and not isinstance(x[0], bool):
                    out.add(x[0])
    return out


# ----------------------------------------------------------------------------- 証拠の集め方
class Evidence:
    def __init__(self):
        self.seed_info = defaultdict(lambda: {"tiers": set(), "labels": set(), "sample": None})
        self.train_seeds = defaultdict(set)      # 値 → ファイルの例
        self.low_seeds = defaultdict(set)        # 値 → ラベル（MIN_LAYOUT_SEED 未満）
        self.high_seeds = defaultdict(set)       # 値 → ラベル（MAX_LAYOUT_SEED 超）
        self.queue_only = defaultdict(lambda: {"labels": set(), "sample": None})
        self.spec_runs = []                      # run.json の指定と、その実行の中の試行記録の突き合わせ用
        self.counts = defaultdict(int)
        self.parse_errors = []
        self.empty_clip_dirs = []                # 中身のないクリップのフォルダ（使用に数えない）
        self.sealed = {"entries": 0, "files": 0, "opened": 0}

    def add(self, seed, tier, label, path):
        if not isinstance(seed, int) or isinstance(seed, bool):
            return
        if seed < MIN_LAYOUT_SEED:
            self.low_seeds[seed].add(label)
            return
        if seed > MAX_LAYOUT_SEED:
            self.high_seeds[seed].add(label)
            return
        info = self.seed_info[seed]
        info["tiers"].add(tier)
        info["labels"].add(label)
        if info["sample"] is None:
            info["sample"] = path

    def add_many(self, seeds, tier, label, path):
        for s in seeds:
            self.add(s, tier, label, path)


def label_of(rel_parts, filename):
    """ラベル: outputs/<top>/<実験>。時刻の入ったフォルダ名は時刻を落とす。"""
    parts = list(rel_parts)
    if not parts:
        return "outputs/" + filename
    top = parts[0]
    if len(parts) == 1:
        return f"{top}/{filename}"
    sub = re.sub(r"_\d{8}-\d{6}(_\d{8}-\d{6})?", "", parts[1])
    sub = re.sub(r"_s\d+of\d+$", "", sub)
    return f"{top}/{sub}"


def walk_generic(obj, ev, label, path, tier, in_train, depth=0):
    if depth > 12:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SEED_INT_KEYS and isinstance(v, int) and not isinstance(v, bool):
                if in_train and k == "seed":
                    ev.train_seeds[v].add(path)
                else:
                    ev.add(v, tier, label, path)
            elif k in SEED_LIST_KEYS:
                ev.add_many(seeds_from_listlike(v), tier, label, path)
            elif k in SPEC_KEYS and isinstance(v, str):
                ev.add_many(seeds_from_listlike(v), tier, label, path)
            elif k == "seed_ranges" and isinstance(v, dict):
                for name, r in v.items():
                    ev.add_many(seeds_from_listlike(r), tier, label, path)
            elif k == "note" and isinstance(v, str):
                for m in NOTE_RANGE_RE.finditer(v):
                    a, b = int(m.group(1)), int(m.group(2))
                    if 0 <= b - a <= MAX_RANGE_LEN:
                        ev.add_many(range(a, b + 1), tier, label, path)
            if isinstance(v, (dict, list)):
                walk_generic(v, ev, label, path, tier, in_train, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                walk_generic(v, ev, label, path, tier, in_train, depth + 1)


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def scan_queue(obj, ev, label, path):
    for item in obj.get("queue", []) if isinstance(obj, dict) else []:
        args = item.get("args", [])
        for i, a in enumerate(args):
            if a in ("--trials", "--seeds") and i + 1 < len(args):
                seeds = seeds_from_listlike(str(args[i + 1]))
                for s in seeds:
                    if s >= MIN_LAYOUT_SEED:
                        q = ev.queue_only[s]
                        q["labels"].add(label)
                        if q["sample"] is None:
                            q["sample"] = path


def collect(root: Path) -> Evidence:
    ev = Evidence()
    out_root = root / "outputs"
    sealed_root = out_root / "sealed"
    for dp, dns, fns in os.walk(out_root):
        dpp = Path(dp)
        if dpp == sealed_root or sealed_root in dpp.parents:
            ev.sealed["entries"] += len(dns) + len(fns)
            ev.sealed["files"] += len(fns)
            dns[:] = []
            continue
        rel_parts = dpp.relative_to(out_root).parts
        top = rel_parts[0] if rel_parts else ""
        if top == "llm_cache":
            ev.counts["llm_cache_files(not parsed: hash-keyed, no seeds)"] += len(fns)
            dns[:] = []
            continue
        # gen: エピソードのフォルダ名
        if top == "gen" and len(rel_parts) == 2:
            m = EPISODE_DIR_RE.search(dpp.name)
            if m:
                s = int(m.group(1) or m.group(2))
                ev.add(s, TIER_TRIAL, label_of(rel_parts[:2], ""), str(dpp.relative_to(root)))
                ev.counts["gen_episode_dirs"] += 1
        # demo・demo_v2・V3DEMO: クリップのフォルダ名（中身が空のフォルダは使用に数えず、情報に出す）
        if dpp.parent.name in CLIP_PARENTS:
            m = CLIP_DIR_RE.search(dpp.name)
            if m:
                if fns:
                    ev.add(int(m.group(1)), TIER_TRIAL, label_of(rel_parts[:2], ""), str(dpp.relative_to(root)))
                    ev.counts["clip_dirs"] += 1
                else:
                    ev.empty_clip_dirs.append(str(dpp.relative_to(root)))
        for fn in fns:
            p = dpp / fn
            relp = str(p.relative_to(root))
            label = label_of(rel_parts, fn)
            if relp.replace("\\", "/") == "outputs/s4/ledger_check.json":
                continue  # このスクリプト自身の出力は読まない
            mfn = FILENAME_SEED_RE.search(fn)
            if mfn and not fn.endswith(".json"):
                ev.add(int(mfn.group(1)), TIER_TRIAL, label, relp)
                ev.counts["filename_seed_files"] += 1
            try:
                if fn.endswith(".jsonl"):
                    n = 0
                    with open(p, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                j = json.loads(line)
                            except Exception:
                                continue
                            if isinstance(j, dict):
                                for k in ("seed", "layout_seed"):
                                    if isinstance(j.get(k), int):
                                        ev.add(j[k], TIER_TRIAL, label, relp)
                                        n += 1
                                        break
                    ev.counts["jsonl_files"] += 1
                    continue
                if fn.endswith(".csv") and top == "results":
                    with open(p, encoding="utf-8", errors="replace", newline="") as f:
                        rd = csv.DictReader(f)
                        if rd.fieldnames and "seed" in rd.fieldnames:
                            for row in rd:
                                try:
                                    ev.add(int(row["seed"]), TIER_DERIVED, label, relp)
                                except (ValueError, TypeError):
                                    pass
                            ev.counts["csv_files"] += 1
                    continue
                if not fn.endswith(".json"):
                    continue
                if SKIP_FILE_RE.match(fn):
                    ev.counts["json_skipped_by_name"] += 1
                    continue
                if fn == "plan.json" and top == "f":
                    ev.counts["plan_json_skipped(計画。使用の記録ではない)"] += 1
                    continue
                if TRIAL_FILE_RE.match(fn):
                    with open(p, encoding="utf-8") as f:
                        head = f.read()
                    j = json.loads(head)
                    if isinstance(j, dict) and isinstance(j.get("seed"), int):
                        ev.add(j["seed"], TIER_TRIAL, label, relp)
                    elif isinstance(j, dict) and isinstance(j.get("layout"), dict) and isinstance(j["layout"].get("seed"), int):
                        ev.add(j["layout"]["seed"], TIER_TRIAL, label, relp)
                    ev.counts["trial_like_files"] += 1
                    continue
                if os.path.getsize(p) > 40_000_000:
                    ev.counts["json_too_large_skipped"] += 1
                    continue
                j = read_json(p)
                if fn == "meta.json" and top == "gen":
                    if isinstance(j, dict):
                        for k in ("layout_seed", "seed"):
                            if isinstance(j.get(k), int):
                                ev.add(j[k], TIER_TRIAL, label, relp)
                                break
                    ev.counts["gen_meta_files"] += 1
                    continue
                if fn == "run.json":
                    tier = TIER_RUN
                elif "queue" in fn.lower():
                    scan_queue(j, ev, label, relp)
                    ev.counts["queue_files"] += 1
                    continue
                elif top == "manifests":
                    tier = TIER_DERIVED
                    for e in j.get("entries", []) if isinstance(j, dict) else []:
                        m = MANIFEST_KEY_RE.search(str(e.get("key", "")))
                        if m:
                            ev.add(int(m.group(1) or m.group(2)), tier, label, relp)
                    ev.counts["manifest_files"] += 1
                    continue
                else:
                    tier = TIER_DERIVED
                in_train = bool(TRAIN_FILE_RE.search(fn))
                walk_generic(j, ev, label, relp, tier, in_train)
                ev.counts["json_walked"] += 1
            except Exception as e:  # 読めない記録は止まらず報告する
                ev.parse_errors.append({"path": relp, "error": f"{type(e).__name__}: {e}"})
    # docs/results の json（結果の写し）
    dres = root / "docs" / "results"
    if dres.is_dir():
        for p in sorted(dres.glob("*.json")):
            relp = str(p.relative_to(root))
            try:
                walk_generic(read_json(p), ev, f"docs/results/{p.name}", relp, TIER_DERIVED, bool(TRAIN_FILE_RE.search(p.name)))
                ev.counts["docs_results_json"] += 1
            except Exception as e:
                ev.parse_errors.append({"path": relp, "error": f"{type(e).__name__}: {e}"})
    return ev


def spec_vs_trials(root: Path):
    """run.json の指定が作る種の集合と、同じフォルダの試行の記録の種の集合を突き合わせる（情報用）。"""
    rows = []
    out_root = root / "outputs"
    sealed_root = out_root / "sealed"
    for dp, dns, fns in os.walk(out_root):
        dpp = Path(dp)
        if dpp == sealed_root or sealed_root in dpp.parents:
            dns[:] = []
            continue
        if "run.json" not in fns:
            continue
        try:
            j = read_json(dpp / "run.json")
        except Exception:
            continue
        if not isinstance(j, dict):
            continue
        spec_seeds = set()
        for k in ("trials", "trials_spec", "seeds"):
            if isinstance(j.get(k), str):
                spec_seeds |= seeds_from_listlike(j[k])
        if not spec_seeds:
            continue
        trial_seeds = set()
        for fn in fns:
            if re.match(r"^(trial|task|run)_\d+\.json$", fn):
                try:
                    with open(dpp / fn, encoding="utf-8") as f:
                        jj = json.load(f)
                    if isinstance(jj, dict) and isinstance(jj.get("seed"), int):
                        trial_seeds.add(jj["seed"])
                except Exception:
                    pass
        if trial_seeds != spec_seeds:
            rows.append({
                "dir": str(dpp.relative_to(root)),
                "spec_seeds": len(spec_seeds), "trial_seeds": len(trial_seeds),
                "spec_not_in_trials": len(spec_seeds - trial_seeds),
                "trials_not_in_spec": len(trial_seeds - spec_seeds),
                "example_trials_not_in_spec": sorted(trial_seeds - spec_seeds)[:3],
            })
    return rows


# ----------------------------------------------------------------------------- 台帳の読み取り
TABLE_HEADING_RE = re.compile(r"^##\s*照合用の対応表")


def parse_ledger(text: str):
    """末尾の『照合用の対応表』から機械が読む行を取り出す。戻り値: (entries, errors, prose_ranges)"""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if TABLE_HEADING_RE.match(ln):
            start = i
            break
    entries, errors = [], []
    if start is None:
        errors.append("台帳に『## 照合用の対応表』の節がない")
        prose = lines
    else:
        prose = lines[:start]
        for ln in lines[start + 1:]:
            if ln.startswith("## "):
                break
            if not ln.startswith("|"):
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) < 6 or not re.match(r"^\d+$", cells[0]):
                continue
            a, b, status_ja, kind, rec, note = cells[0], cells[1], cells[2], cells[3], cells[4], "|".join(cells[5:])
            if status_ja not in STATUS_JA:
                errors.append(f"状態が不明: {ln}")
                continue
            if not re.match(r"^\d+$", b) or int(b) < int(a):
                errors.append(f"範囲が不正: {ln}")
                continue
            record_expected = rec.startswith("あり")
            if not (record_expected or rec.startswith("なし")):
                errors.append(f"記録の列は『あり』か『なし:理由』: {ln}")
                continue
            entries.append({
                "start": int(a), "end": int(b), "status": STATUS_JA[status_ja], "kind": kind,
                "record_expected": record_expected, "record_note": rec, "note": note,
            })
    prose_ranges = []
    for ln in prose:
        if not ln.startswith("|"):
            continue
        for m in re.finditer(r"(?<!\d)(\d{5,6})\s*〜\s*(\d{5,6})(?!\d)", ln):
            prose_ranges.append((int(m.group(1)), int(m.group(2))))
    return entries, errors, prose_ranges


# ----------------------------------------------------------------------------- 照合
def group_runs(seeds_sorted):
    return seeds_to_ranges(seeds_sorted)


def describe_range(ev: Evidence, a, b):
    labs, tiers = set(), set()
    n = 0
    sample = None
    for s in range(a, b + 1):
        info = ev.seed_info.get(s)
        if info:
            n += 1
            labs |= info["labels"]
            tiers |= info["tiers"]
            sample = sample or info["sample"]
    return {"range": [a, b], "n_seeds_with_record": n, "labels": sorted(labs), "tiers": sorted(tiers), "example_path": sample}


def check(ev: Evidence, entries, ledger_errors, prose_ranges):
    problems = {k: [] for k in (
        "not_in_ledger", "ledger_used_without_record", "planned_with_use",
        "banned_with_use", "ledger_inconsistency")}
    warnings = {"prose_not_in_table": [], "ledger_used_partial_record": [], "status_mismatch_note": [],
                "planned_with_registered_use": []}
    for e in ledger_errors:
        problems["ledger_inconsistency"].append({"kind": "table_format", "detail": e})

    used_seeds = sorted(s for s, info in ev.seed_info.items() if info["tiers"] & USED_TIERS)
    by_status = defaultdict(list)
    for e in entries:
        by_status[e["status"]].append([e["start"], e["end"]])
    used_merged = merge_ranges(by_status["used"])

    # 1. 台帳にない使用
    not_in = [s for s in used_seeds if not in_ranges(s, used_merged)]
    for a, b in group_runs(not_in):
        d = describe_range(ev, a, b)
        cover = [f"{e['status']}:{e['start']}-{e['end']}" for e in entries if e["start"] <= b and a <= e["end"]]
        d["ledger_status_here"] = cover or ["台帳に項目なし"]
        problems["not_in_ledger"].append(d)

    # 2. 使用済みなのに記録がない
    for e in entries:
        if e["status"] != "used":
            continue
        a, b = e["start"], e["end"]
        have = [s for s in range(a, b + 1) if s in ev.seed_info and (ev.seed_info[s]["tiers"] & USED_TIERS)]
        if e["record_expected"]:
            if not have:
                problems["ledger_used_without_record"].append({
                    "range": [a, b], "kind": e["kind"], "note": e["note"][:160]})
            elif len(have) < (b - a + 1):
                warnings["ledger_used_partial_record"].append({
                    "range": [a, b], "seeds_with_record": len(have), "size": b - a + 1,
                    "missing_ranges": seeds_to_ranges(set(range(a, b + 1)) - set(have))[:8],
                    "kind": e["kind"]})
        else:
            if have:
                warnings["status_mismatch_note"].append({
                    "range": [a, b], "detail": "『記録なし』と注記したが記録が見つかった",
                    "seeds_with_record": len(have), "record_note": e["record_note"][:100]})

    # 3. 予定・予約の中の使用。台帳に「使用済み」として載っていない使用は問題、
    #    載っている使用（予定の帯の中で、その帯の使い道として使った分）は警告に並べて見えるようにする。
    for e in entries:
        if e["status"] not in ("planned", "reserved"):
            continue
        a, b = e["start"], e["end"]
        hit = [s for s in used_seeds if a <= s <= b]
        for registered, key in ((False, None), (True, "planned_with_registered_use")):
            sub = [s for s in hit if in_ranges(s, used_merged) == registered]
            for ra, rb in group_runs(sub):
                d = describe_range(ev, ra, rb)
                d["ledger_entry"] = {"status": e["status"], "range": [a, b], "kind": e["kind"], "note": e["note"][:120]}
                if registered:
                    warnings[key].append(d)
                else:
                    problems["planned_with_use"].append(d)

    # 4. 使わない・取り消しの中の使用
    for e in entries:
        if e["status"] in ("banned", "void"):
            a, b = e["start"], e["end"]
            hit = [s for s in used_seeds if a <= s <= b]
            for ra, rb in group_runs(hit):
                d = describe_range(ev, ra, rb)
                d["ledger_entry"] = {"status": e["status"], "range": [a, b], "kind": e["kind"], "note": e["note"][:120]}
                problems["banned_with_use"].append(d)

    # 5. 対応表の中の矛盾
    es = sorted(entries, key=lambda x: (x["start"], x["end"]))
    for i, e in enumerate(es):
        for f in es[i + 1:]:
            if f["start"] > e["end"]:
                break
            pair = {e["status"], f["status"]}
            soft = {"planned", "reserved"}
            if pair <= soft:
                continue  # 予定の帯とその使い道の入れ子は許す
            if "used" in pair and len(pair) == 2 and pair & soft:
                # 使用済みが予定・予約の帯の中に収まっていれば許す（帯の中で使った分）
                u, p = (e, f) if e["status"] == "used" else (f, e)
                if p["start"] <= u["start"] and u["end"] <= p["end"]:
                    continue
            if pair & {"used"} or pair & {"banned", "void"}:
                problems["ledger_inconsistency"].append({
                    "kind": "overlap",
                    "a": [e["start"], e["end"], e["status"], e["kind"]],
                    "b": [f["start"], f["end"], f["status"], f["kind"]]})

    # 6. 本文にだけある範囲（警告）
    table_all = merge_ranges((e["start"], e["end"]) for e in entries)
    seen = set()
    for a, b in prose_ranges:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        if not (in_ranges(a, table_all) and in_ranges(b, table_all)):
            warnings["prose_not_in_table"].append([a, b])
    return problems, warnings


def check_bands(ev: Evidence, entries, bands):
    used_seed_set = {s for s, info in ev.seed_info.items() if info["tiers"] & USED_TIERS}
    out = []
    for a, b in bands:
        hit = sorted(s for s in used_seed_set if a <= s <= b)
        led = [e for e in entries if e["start"] <= b and a <= e["end"]]
        led_used = [e for e in led if e["status"] == "used"]
        led_block = [e for e in led if e["status"] in ("banned", "void")]
        q = sorted(s for s in ev.queue_only if a <= s <= b and s not in used_seed_set)
        out.append({
            "band": [a, b],
            "unused": not hit and not led_used,
            "record_uses": [describe_range(ev, x, y) for x, y in group_runs(hit)],
            "ledger_used_overlaps": [[e["start"], e["end"], e["kind"]] for e in led_used],
            "ledger_planned_overlaps": [[e["start"], e["end"], e["status"], e["kind"]] for e in led if e["status"] in ("planned", "reserved")],
            "ledger_blocked_overlaps": [[e["start"], e["end"], e["status"], e["kind"]] for e in led_block],
            "queue_declared_only": group_runs(q)[:20],
        })
    return out


def parse_band_arg(s):
    m = re.match(r"^(\d+)\s*[-〜~:]\s*(\d+)$", s)
    if not m:
        raise SystemExit(f"--bands の書き方は 160000-169999: {s}")
    a, b = int(m.group(1)), int(m.group(2))
    if b < a:
        raise SystemExit(f"--bands の終わりが始めより小さい: {s}")
    return a, b


# ----------------------------------------------------------------------------- 単体検査
def self_test():
    assert merge_ranges([(1, 3), (4, 6), (9, 9)]) == [[1, 6], [9, 9]]
    assert parse_spec("natural:199230:33") == [(199230, 33)]
    assert parse_spec("115000:20") == [(115000, 20)]
    assert parse_spec("selection:194000:99,induced:195000:50") == [(194000, 99), (195000, 50)]
    assert seeds_from_listlike([49000, 49009]) == set(range(49000, 49010))
    assert seeds_from_listlike("59000:10") == set(range(59000, 59010))
    assert seeds_from_listlike([[140001, "blue"], [140003, "green"]]) == {140001, 140003}
    assert seeds_from_listlike([192000, 192001, 192002]) == {192000, 192001, 192002}
    assert EPISODE_DIR_RE.search("H_44000_red_r0").group(1) == "44000"
    assert EPISODE_DIR_RE.search("R2P1_40000_at_est").group(2) == "40000"
    assert not EPISODE_DIR_RE.search("S3H_logs")
    ev = Evidence()
    ev.add(160001, TIER_TRIAL, "x", "p")
    ev.add(1000, TIER_TRIAL, "x", "p")
    assert 1000 in ev.low_seeds and 160001 in ev.seed_info
    text = (
        "# t\n| 帯 | a |\n|---|---|\n| 160000〜160010 | b |\n\n## 照合用の対応表\n\n"
        "| 開始 | 終了 | 状態 | 層 | 記録 | 備考 |\n|---|---|---|---|---|---|\n"
        "| 160000 | 169999 | 予定 | 帯 | あり | テスト |\n"
        "| 170000 | 170009 | 使用済み | 実績 | なし:単体検査 | x |\n"
    )
    entries, errors, prose = parse_ledger(text)
    assert not errors and len(entries) == 2 and entries[1]["record_expected"] is False, (entries, errors)
    assert prose == [(160000, 160010)]
    problems, warnings = check(ev, entries, errors, prose)
    assert len(problems["not_in_ledger"]) == 1 and problems["not_in_ledger"][0]["range"] == [160001, 160001]
    assert len(problems["planned_with_use"]) == 1
    # 使用済みの範囲に入れば not_in_ledger は出ない
    entries2 = entries + [{"start": 160000, "end": 160005, "status": "used", "kind": "k", "record_expected": True, "record_note": "あり", "note": ""}]
    problems2, warnings2 = check(ev, entries2, [], [])
    assert not problems2["not_in_ledger"]
    # 予定の帯の中に収まる使用済みの行は矛盾ではなく、使用は「登録済みの使用」として警告に出る
    assert not problems2["ledger_inconsistency"], problems2["ledger_inconsistency"]
    assert not problems2["planned_with_use"] and len(warnings2["planned_with_registered_use"]) == 1
    # はみ出す使用済みの行・使用済み同士の重なり・取り消しとの重なりは矛盾
    for extra in ({"start": 159990, "end": 160005, "status": "used"},
                  {"start": 170005, "end": 170006, "status": "used"},
                  {"start": 160003, "end": 160003, "status": "void"}):
        x = dict(kind="k", record_expected=True, record_note="あり", note="", **extra)
        p3, _ = check(ev, entries2 + [x], [], [])
        assert any(p["kind"] == "overlap" for p in p3["ledger_inconsistency"]), extra
    assert CLIP_DIR_RE.search("nat_140011_R").group(1) == "140011"
    assert CLIP_DIR_RE.search("task_145003").group(1) == "145003"
    assert CLIP_DIR_RE.search("p1_141004_N").group(1) == "141004"
    assert not CLIP_DIR_RE.search("_logs")
    # 帯の確認
    b = check_bands(ev, entries, [(160000, 160009), (161000, 161009)])
    assert b[0]["unused"] is False and b[1]["unused"] is True
    # 抜き出し（疑似的な構造）
    ev2 = Evidence()
    walk_generic({"seeds": "59000:3", "rows": [{"seed": 59000}, {"seed": 59002}], "trials": "natural:199230:2",
                  "seed_ranges": {"A": [52000, 52002]}, "layout": {"seed": 20000}}, ev2, "L", "p", TIER_DERIVED, False)
    got = set(ev2.seed_info)
    assert got == {59000, 59001, 59002, 199230, 199231, 52000, 52001, 52002, 20000}, got
    ev3 = Evidence()
    walk_generic({"config": {"seed": 1000}}, ev3, "L", "train_run.json", TIER_DERIVED, True)
    assert 1000 in ev3.train_seeds and not ev3.seed_info and not ev3.low_seeds
    assert FILENAME_SEED_RE.search("expert_n_58000_red.npz").group(1) == "58000"
    assert FILENAME_SEED_RE.search("policy_R2_P1_000_58200_red.npz").group(1) == "58200"
    assert not FILENAME_SEED_RE.search("trial_0000.npz")
    ev4 = Evidence()
    walk_generic({"note": "学習用のシード 59010〜59029（配置の種類は乱数列のまま）"}, ev4, "L", "p", TIER_DERIVED, False)
    assert set(ev4.seed_info) == set(range(59010, 59030))
    ev5 = Evidence()
    ev5.add(20261005, TIER_DERIVED, "L", "p")
    assert 20261005 in ev5.high_seeds and not ev5.seed_info
    print("self-test OK")


# ----------------------------------------------------------------------------- 本体
def sha256_of(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_report(root: Path, ledger_path: Path, bands):
    text = ledger_path.read_text(encoding="utf-8")
    entries, errors, prose_ranges = parse_ledger(text)
    ev = collect(root)
    problems, warnings = check(ev, entries, errors, prose_ranges)
    used_seeds = sorted(s for s, info in ev.seed_info.items() if info["tiers"] & USED_TIERS)

    by_label = defaultdict(set)
    for s, info in ev.seed_info.items():
        for lab in info["labels"]:
            by_label[lab].add(s)
    used_by_label = {lab: seeds_to_ranges(sorted(ss)) for lab, ss in sorted(by_label.items())}

    queue_only = sorted(s for s in ev.queue_only if s not in ev.seed_info)
    queue_only_ranges = []
    for a, b in group_runs(queue_only):
        labs = set()
        for s in range(a, b + 1):
            if s in ev.queue_only:
                labs |= ev.queue_only[s]["labels"]
        queue_only_ranges.append({"range": [a, b], "labels": sorted(labs)})

    report = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": "scripts/97_s4_ledger_check.py",
        "ledger": {"path": str(ledger_path.relative_to(root)) if ledger_path.is_relative_to(root) else str(ledger_path),
                   "sha256": sha256_of(ledger_path), "table_entries": len(entries)},
        "scan": {
            "counts": dict(ev.counts),
            "seeds_with_record": len(used_seeds),
            "labels": len(used_by_label),
            "parse_errors": ev.parse_errors,
            "sealed_not_opened": ev.sealed,
            "rules": {
                "min_layout_seed": MIN_LAYOUT_SEED,
                "used_tiers": sorted(USED_TIERS),
                "queue_is_weak_evidence": True,
                "sealed": "outputs/sealed は開かない。台帳の該当範囲は『なし』と注記する",
            },
        },
        "problems": problems,
        "warnings": warnings,
        "info": {
            "train_seeds(配置の種とは別。照合に使わない)": {str(k): sorted(v)[:3] for k, v in sorted(ev.train_seeds.items())},
            "low_seeds(10000 未満。報告のみ)": {str(k): sorted(v)[:5] for k, v in sorted(ev.low_seeds.items())},
            "high_seeds(1000000 以上。日付の乱数の種など。報告のみ)": {str(k): sorted(v)[:5] for k, v in sorted(ev.high_seeds.items())},
            "queue_declared_only(queue にだけある種。実行の記録なし)": queue_only_ranges,
            "empty_clip_dirs(中身のないクリップのフォルダ。使用に数えない)": ev.empty_clip_dirs,
            "run_spec_vs_trial_files": spec_vs_trials(root),
        },
        "used_ranges": [describe_range(ev, a, b) for a, b in seeds_to_ranges(used_seeds)],
        "used_ranges_by_label": used_by_label,
    }
    band_rows = check_bands(ev, entries, bands) if bands else []
    report["bands"] = band_rows
    return report, problems, band_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="種の台帳と記録の照合")
    ap.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--bands", nargs="*", default=[], help="例: 160000-169999 170000-179999")
    ap.add_argument("--bands-only", action="store_true", help="終了コードを --bands の結果だけで決める")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    root = Path(args.root)
    ledger = Path(args.ledger)
    if not ledger.is_file():
        print(f"台帳が無い: {ledger}", file=sys.stderr)
        return 2
    bands = [parse_band_arg(b) for b in args.bands]
    report, problems, band_rows = build_report(root, ledger, bands)
    n_problem = sum(len(v) for v in problems.values())
    bands_ok = all(b["unused"] for b in band_rows)
    ok = bands_ok if args.bands_only else (n_problem == 0 and bands_ok)
    report["ok"] = ok
    report["summary"] = {k: len(v) for k, v in problems.items()}
    report["summary"].update({"warnings_" + k: len(v) for k, v in report["warnings"].items()})
    report["summary"]["bands_all_unused"] = bands_ok if band_rows else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    for b in band_rows:
        print(f"帯 {b['band'][0]}-{b['band'][1]}: {'未使用' if b['unused'] else '使用あり'}"
              f"（記録 {len(b['record_uses'])} 件、台帳の使用済みと重なり {len(b['ledger_used_overlaps'])} 件、"
              f"予定と重なり {len(b['ledger_planned_overlaps'])} 件）")
    print("OK（問題なし）" if ok else "問題あり: " + str(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
