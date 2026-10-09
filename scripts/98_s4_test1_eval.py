"""段階 4 テスト 1 の起動の入口。96_s4_resume.py の run・task を包み、登録版の事前登録 v1（docs/stage4/prereg_test1_v1.md、掲示板 0168。
第 4・5・13-1・15 節）の枝 1〜4 の条件を、同じ種で種の塊ごとに交互に回す。98_s4_b4_eval.py と同じ作り（掲示板 0166 の直しを含む）。

使い方（作業場所 C:\\PAI\\recovery_vla。96_s4_resume.py の続きの引数（--accept-env-change・--min-free-gb など）は後ろに付ける）:
    .venv\\Scripts\\python.exe scripts\\98_s4_test1_eval.py plan --branch 1 --rtc-setting ZEROS          # 条件・帯・試行数・時間（読むだけ）
    .venv\\Scripts\\python.exe scripts\\98_s4_test1_eval.py ckpt                                         # 保存点 6 本の対応と SHA-256
    .venv\\Scripts\\python.exe scripts\\98_s4_test1_eval.py rotate --branch 1 --rtc-setting ZEROS --dry-run
    .venv\\Scripts\\python.exe scripts\\98_s4_test1_eval.py rotate --branch 1 --rtc-setting ZEROS --worker 1 --expect R1v3s1001=<SHA-256> ...
        （3 本並行は、同じコマンドを --worker 1・2・3 で 3 つ起こす。下の「rotate」）
    .venv\\Scripts\\python.exe scripts\\98_s4_test1_eval.py run --branch 1 --cond P1_R1v3 --max-new 10 [--dry-run]   # 1 条件だけ
    .venv\\Scripts\\python.exe scripts\\98_s4_test1_eval.py health --worker 1 [--dry-run]               # 健全性の確認（190600〜190632）
    .venv\\Scripts\\python.exe scripts\\98_s4_test1_eval.py layout --branch 1 --rtc-setting ZEROS --out outputs\\s4\\test1\\   # 二重集計の入力
  ■ 試すときは必ず --dry-run を付ける（付けないと GPU で評価を回し、計画役の API を呼び、帯の種を使う）。

枝（登録版 第 13-1 節。束 2 の判定の「枝の確定」の掲示で 1 つに決まる。引数で束 2 の判定を上書きしない）:
  枝 1（B1・B2 を採る、案 B）2,240 試行、枝 2（B1 だけ、案 A）1,842、枝 3（B2 だけ、案 B）1,790、枝 4（どちらも採らない）1,392。
  E7（枝 1・2）: 160000:150 を 3 腕（E7_R1v3_cur＝R1v3＋今の実行器、E7_R1v3_v3＝R1v3＋v3、E7_N1v3_v3＝N1v3＋v3）。96 の task、
    実行器 v3 は v3.1 で (a) なし（関門 T 不合格。0163）、計画役は 3 腕とも s4（Haiku 5.5）、1 手順 30 s・やり直し 1 回・全体 200 s、
    行動の区切り 6 行・安全フィルタなし。v3 の包みは 98_s4_b1.py の patch96 をそのまま使う（試行の json に "v3"・"b1" が入る）。
  P1（全部の枝）: induced:162000:100 を 6 モデル（P1_R1v3・P1_N1v3・P1_R1v3s1001・P1_N1v3s1001・P1_R1v3s1002・P1_N1v3s1002）。
  自然（全部の枝）: natural:161000:66（198 試行）を nat_R1v3・nat_N1v3、natural:161000:33（99 試行）を種 1001・1002 の 4 モデル。
  RTC の腕（枝 1・3、案 B）: nat_R1v3_rtc（natural:161000:66）、P1_R1v3_rtc・P1_N1v3_rtc（induced:162000:100）。設定は --rtc-setting
    （束 2 の判定の掲示で決まる。関門 R の候補のどれか）。RTC の当て方は 98_s4_d_rtc.py の patch96 をそのまま使う（diag_NNNN.npz も同じ形）。
  単発の試行は naive（RTC の腕は設定の mode）・6 行・安全フィルタなし・60 s（主な採点は 30 s）。試行ごとに世界を作り直す（96 の既定）。
  条件名は 98_s4_test1.py example の layout と同じ（RTC の N1v3 の P1 は P1_N1v3_rtc）。実験名は S4T1。
帯: s4_gates.json の test1_E7（160000〜160149）・test1_natural（161000〜161065。種 1001・1002 は 161000〜161032）・test1_P1
  （162000〜162099）と完全に一致する指定だけ（e7_band_extended は偽。n＝150）。健全性の確認は seed_copy_health（190600〜190632）。
  smoke は --allow-smoke・実験名 S4SMOKE*・smoke・データの帯 44400〜44799 の中で X2（44404〜44423）と台帳の使用済みの外だけ。
保存点（82_v2_eval.py の CKPT の写しに足す。82 は書き換えない）: R1v3・N1v3 は 82 の CKPT（登録版 第 5 節の実行と同じことを確かめる）、
  R1v3s1001・N1v3s1001・R1v3s1002・N1v3s1002 は束 3 の outputs\\s4\\train\\<登録版 第 5 節の実行名>\\checkpoints\\020000\\pretrained_model
  （outputs\\s4\\seed_wrap\\postcheck_<実行名>.json が通っていること。smoke の実行は使わない）。--run 名前=<実行名> で変えられる（記録に残る。
  登録版と違う実行は解析の版の照合 (b-4) で未完になる）。checkpoints\\last があれば 020000 と同じ中身かを確かめ、違えば警告する。
  SHA-256 は 98_s4_b4_eval.py と同じ ckpt_digest。--expect 名前=<SHA-256> と違えば起動しない。名前は 6 本のモデルのほか、
  executor_v3（src/recovla/runtime/executor_v3.py）・rtc_module（src/recovla/diag/rtc.py）・entry（このファイル）。
記録: 試行の json と run.json に "t1"（枝・条件・役・モデル・保存点のパスと SHA-256・RTC の設定・実行器・使ったファイルの SHA-256
  （s4_gates.json は LF にそろえた値））。resume_spec.json に t1_ckpt_sha256・t1_rtc・t1_entry_sha256（再開のとき違えば 96 が止める）。
rotate（98_s4_b1.py と同じ止め方・progress・腕ごとの子プロセス）: 1 回の呼び出しで塊 1 つ（E7 5 種、P1 10 種、自然 11 種＝33 試行。
  登録版 第 5 節）を子プロセスで回す。続けるのは子が 0 で終わるか --max-new で止まったときだけ。Ctrl+C・止める合図・エラーでは止まる。
  呼び出しで完全な試行が増えなければエラー。3 本並行は同じコマンドを --worker 1・2・3 で起こす: 各ワーカーは、ほかのワーカーが回して
  いない条件のうち、塊の番号がいちばん若いもの（同じなら E7 → P1 → 自然、条件の順）を取る（条件ごとの占有のファイルで 2 重に回さない）。
  同じ比べる組（E7・P1・自然）の中で 1 つの条件だけが 1 塊より先に進むことはしない（登録版 第 8 節「交互の塊が崩れた」）。
  進み具合は outputs\\s4\\test1_eval\\rotate_<実験>_b<枝>_w<ワーカー>.progress.json（pid 付き、1 分ごとに心拍）。止める合図:
  <条件>\\STOP、outputs\\s4\\STOP、outputs\\s4\\test1_eval\\rotate_<実験>.STOP（全ワーカー）。
引数の拒み方: モデル・帯・実行のしかた・誘発・制限時間・計画役の引数は、省略形も前方一致で拒む（0166）。
ドライバ: 610.88 と違っても止めない。警告を出し、版は試行の env に残る（0166）。記録と今の環境（ドライバ・torch・CUDA・OS）が違えば
  止める（96 の --accept-env-change のときだけ進め、96 が環境の区切りを書く）。
終了コード: 96_s4_resume.py と同じ（0 全部そろった、1 途中で止まった、2 エラー、3 引数・前提の食い違い）。
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
GATES = ROOT / "configs" / "s4_gates.json"
OUTD = ROOT / "outputs" / "s4" / "test1_eval"
LEDGER = ROOT / "docs" / "種の台帳.md"
EXPERIMENT = "S4T1"
HEALTH_EXPERIMENT = "S4T1HC"
POSTED_GATES_SHA256 = "7f2f651cafa9cf97b5548324d3fb8ea0bec5e891cca9c8859c7dd9c0347646e9"   # 掲示板 0165（改訂 4。LF）
EXPECTED_DRIVER = "610.88"
V3_VERSION = "v3.1"
ENV_KEYS = ("driver", "torch", "torch_cuda", "os_build")          # 96_s4_resume.ENV_STOP_KEYS と同じ
CKPT_SUB = ("checkpoints", "020000", "pretrained_model")
E7_N = 150
# 登録版 第 5 節のチェックポイント（(根, 実行名, 後の点検の置き場)）
REGISTERED = {
    "R1v3": ("outputs/train", "train_R1v3_20261005-180404_20261005-180404", None),
    "N1v3": ("outputs/train", "train_N1v3_20261005-202158_20261005-202158", None),
    "R1v3s1001": ("outputs/s4/train", "train_R1v3s1001_20261009-002457_20261009-002457", "outputs/s4/seed_wrap"),
    "N1v3s1001": ("outputs/s4/train", "train_N1v3s1001_20261009-024259_20261009-024259", "outputs/s4/seed_wrap"),
    "R1v3s1002": ("outputs/s4/train", "train_R1v3s1002_20261009-050112_20261009-050112", "outputs/s4/seed_wrap"),
    "N1v3s1002": ("outputs/s4/train", "train_N1v3s1002_20261009-071920_20261009-071920", "outputs/s4/seed_wrap"),
}
BRANCHES = {1: {"b1": True, "b2": True, "plan": "B", "trials": 2240, "process_h": 72.2, "parallel3_h": 25.3},
            2: {"b1": True, "b2": False, "plan": "A", "trials": 1842, "process_h": 63.9, "parallel3_h": 22.4},
            3: {"b1": False, "b2": True, "plan": "B", "trials": 1790, "process_h": 31.3, "parallel3_h": 11.0},
            4: {"b1": False, "b2": False, "plan": "A", "trials": 1392, "process_h": 22.9, "parallel3_h": 8.0}}
BLOCK_SEEDS = {"e7": 5, "p1": 10, "nat": 11, "health": 11}        # 登録版 第 5 節「組の並べ方」（健全性の確認は自然と同じ）
GROUP_RANK = {"e7": 0, "p1": 1, "nat": 2, "health": 3}
PER_TRIAL_H = {"e7": 40.9 / 450, "p1_R": 0.019, "p1_N": 0.023, "nat": 10.3 / 792, "rtc_nat": 4.2 / 198, "rtc_p1_R": 0.018,
               "rtc_p1_N": 0.023, "health": 1.6 / 132}            # 登録版 第 5 節の表から（推測）
HEALTH_SPECS = ("selection:190600:33", "natural:190600:33")       # 色の扱いは回す前に掲示（登録版 第 4 節）。既定は 1 種 1 色
FILES = ("scripts/98_s4_test1_eval.py", "scripts/96_s4_resume.py", "scripts/82_v2_eval.py", "scripts/98_s4_b1.py",
         "scripts/98_s4_d_rtc.py", "src/recovla/runtime/executor_v3.py", "src/recovla/runtime/executor.py", "src/recovla/diag/rtc.py",
         "configs/s4_gates.json")
FORBIDDEN_EXTRA = ("--model", "--mode", "--exec-interval", "--induce", "--ablate", "--time-limit-s", "--reuse-world", "--grip-gate",
                   "--trials", "--experiment", "--condition", "--no-safety", "--no-limiter", "--xcmd-leash", "--cart-margin",
                   "--diag-ik", "--diag-no-gravcomp", "--planner", "--text", "--step-timeout-s", "--task-time-limit-s")
ROTATE_FORBIDDEN = ("--max-new", "--progress-file", "--stop-file")


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


def sha256_lf(p) -> str:
    """改行を LF にそろえた SHA-256（掲示の値の計算と同じ。gate1.py・98_s4_b4_check.py と同じ）。"""
    return hashlib.sha256(pathlib.Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def files_sha256(root: pathlib.Path = ROOT) -> dict:
    return {p: (sha256_lf(root / p) if p == "configs/s4_gates.json" else sha256_file(root / p)) for p in FILES}


def gates(path: pathlib.Path = GATES) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 条件
def conditions(branch, rtc_setting: str = None) -> list:
    """枝の条件の並び。[{cond, key（98_s4_test1 の layout の鍵）, group, kind（task|run）, model, trials, induce, rtc, arm}]。
    branch="health" なら健全性の確認の 4 条件。"""
    if branch == "health":
        return [{"cond": f"health_{m}", "key": f"health.{m}", "group": "health", "kind": "run", "model": m, "trials": HEALTH_SPECS[0],
                 "induce": None, "rtc": None, "arm": None} for m in ("R1v3s1001", "N1v3s1001", "R1v3s1002", "N1v3s1002")]
    if branch not in BRANCHES:
        raise SystemExit(f"--branch は 1〜4（渡された {branch}）")
    br = BRANCHES[branch]
    if br["b2"] and not rtc_setting:
        raise SystemExit(f"枝 {branch} は RTC の腕を回す（案 B）。--rtc-setting（束 2 の判定の掲示の設定）が要る")
    if not br["b2"] and rtc_setting:
        raise SystemExit(f"枝 {branch} は RTC の腕を回さない（案 A）。--rtc-setting は渡さない")
    out = []
    if br["b1"]:
        for arm, key, model in (("R1v3_cur", "e7.cur", "R1v3"), ("R1v3_v3", "e7.v3", "R1v3"), ("N1v3_v3", "e7.n1v3_v3", "N1v3")):
            out.append({"cond": f"E7_{arm}", "key": key, "group": "e7", "kind": "task", "model": model, "trials": f"160000:{E7_N}",
                        "induce": None, "rtc": None, "arm": arm})
    for layer, sfx in (("1000", ""), ("1001", "s1001"), ("1002", "s1002")):
        for side, base in (("R", "R1v3"), ("N", "N1v3")):
            out.append({"cond": f"P1_{base}{sfx}", "key": f"p1.{layer}.{side}", "group": "p1", "kind": "run", "model": base + sfx,
                        "trials": "induced:162000:100", "induce": "P1", "rtc": None, "arm": None})
    for layer, sfx, n in (("1000", "", 66), ("1001", "s1001", 33), ("1002", "s1002", 33)):
        for side, base in (("R", "R1v3"), ("N", "N1v3")):
            out.append({"cond": f"nat_{base}{sfx}", "key": f"natural.{layer}.{side}", "group": "nat", "kind": "run", "model": base + sfx,
                        "trials": f"natural:161000:{n}", "induce": None, "rtc": None, "arm": None})
    if br["b2"]:
        out.append({"cond": "nat_R1v3_rtc", "key": "rtc.natural", "group": "nat", "kind": "run", "model": "R1v3",
                    "trials": "natural:161000:66", "induce": None, "rtc": rtc_setting, "arm": None})
        out.append({"cond": "P1_R1v3_rtc", "key": "rtc.p1", "group": "p1", "kind": "run", "model": "R1v3",
                    "trials": "induced:162000:100", "induce": "P1", "rtc": rtc_setting, "arm": None})
        out.append({"cond": "P1_N1v3_rtc", "key": "rtc.p1_n", "group": "p1", "kind": "run", "model": "N1v3",
                    "trials": "induced:162000:100", "induce": "P1", "rtc": rtc_setting, "arm": None})
    return out


def n_trials(spec: str) -> int:
    parts = spec.split(":")
    if len(parts) == 2:
        return int(parts[1])
    return int(parts[2]) * (3 if parts[0] == "natural" else 1)


def per_seed(c: dict) -> int:
    return 3 if c["trials"].startswith("natural:") else 1


def block_trials(c: dict) -> int:
    return BLOCK_SEEDS[c["group"]] * per_seed(c)


def find(conds: list, name: str) -> dict:
    for c in conds:
        if c["cond"] == name:
            return c
    raise SystemExit(f"条件は {[c['cond'] for c in conds]} のどれか（渡された {name}）")


# ---------------------------------------------------------------- 帯
def check_tables(g: dict = None) -> list:
    """条件の表の帯が s4_gates.json の test1_*・seed_copy_health の割り当てと合うか。"""
    g = g or gates()
    al = {a["id"]: tuple(a["range"]) for a in g["bands"]["allocations"]}
    want = {"test1_E7": (160000, 160000 + E7_N - 1), "test1_natural": (161000, 161065), "test1_P1": (162000, 162099),
            "seed_copy_health": (190600, 190632)}
    return [f"{k}: s4_gates.json は {al.get(k)}、入口の表は {v}" for k, v in want.items() if al.get(k) != v]


def ledger_used(text: str) -> list:
    """台帳の表の「| 下 | 上 | 使用済み |」の行（97_s4_b4_data.py と同じ読み方）。"""
    rx = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*使用済み\s*\|", re.M)
    return [(int(a), int(b)) for a, b in rx.findall(text)]


def check_band(c: dict, trials: str, experiment: str, allow_smoke: bool, g: dict = None, ledger_text: str = None) -> str:
    """"" なら通す。本番は条件の表と完全に一致する指定（種類も）と本番の実験名だけ。smoke は --allow-smoke・S4SMOKE*・
    smoke・データの帯の中で X2・台帳の使用済みの外。"""
    g = g or gates()
    probs = check_tables(g)
    if probs:
        return "帯の表が s4_gates.json と合わない: " + "; ".join(probs)
    parts = trials.split(":")
    want_parts = c["trials"].split(":")
    try:
        nums = [int(x) for x in parts[-2:]]
    except ValueError:
        return f"--trials {trials!r} の形が違う（{c['trials']} の形）"
    if len(parts) != len(want_parts) or (len(parts) == 3 and parts[0] != want_parts[0] and c["group"] != "health"):
        return f"{c['cond']} の試行の種類・形は {c['trials']}（渡された {trials}）"
    want_exp = HEALTH_EXPERIMENT if c["group"] == "health" else EXPERIMENT
    prod = (trials in HEALTH_SPECS) if c["group"] == "health" else (trials == c["trials"])
    if prod:
        if experiment != want_exp:
            return f"本番の帯 {trials} は実験名 {want_exp} だけで回す（渡された {experiment}）"
        return ""
    if not allow_smoke:
        return f"{c['cond']} の本番の帯は {c['trials']} と完全に一致する指定だけ（渡された {trials}）。smoke は --allow-smoke"
    if not experiment.startswith("S4SMOKE"):
        return f"smoke の実験名は S4SMOKE で始める（{experiment}）"
    base, n = nums
    seeds = range(base, base + n)
    sm = next(x for x in g["bands"]["classes"] if x["id"] == "smoke_data")["range"]
    x2 = next(a["range"] for a in g["bands"]["allocations"] if a["id"] == "X2_gen")
    if n <= 0 or not all(sm[0] <= s <= sm[1] for s in seeds):
        return f"smoke の種 {base}〜{base + n - 1} が smoke・データの帯 {sm} に収まらない"
    if any(x2[0] <= s <= x2[1] for s in seeds):
        return f"smoke の種が X2 の生成の帯 {x2}（予約）に入る"
    text = ledger_text if ledger_text is not None else (LEDGER.read_text(encoding="utf-8-sig") if LEDGER.is_file() else "")
    hit = [(lo, hi) for lo, hi in ledger_used(text) if not (hi < base or base + n - 1 < lo)]
    if hit:
        return f"smoke の種 {base}〜{base + n - 1} は台帳で使用済み {hit[0]}"
    return ""


# ---------------------------------------------------------------- 保存点
def ckpt_digest(d: pathlib.Path) -> dict:
    """98_s4_b4_eval.py と同じ: 相対パスの順に「相対パス<TAB>SHA-256<LF>」をつないだ UTF-8 の SHA-256。"""
    files = sorted(p for p in pathlib.Path(d).rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"保存点 {d} にファイルがない")
    each = {p.relative_to(d).as_posix(): sha256_file(p) for p in files}
    text = "".join(f"{k}\t{v}\n" for k, v in sorted(each.items()))
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "files": each,
            "rule": "相対パスの順に「相対パス<TAB>ファイルの SHA-256<LF>」をつないだ UTF-8 の SHA-256"}


def registered_path(name: str) -> str:
    base, run, _ = REGISTERED[name]
    return "/".join((base, run) + CKPT_SUB)


def _postcheck_ok(d: dict) -> bool:
    return (bool(d.get("train_config_only_seed_and_names")) and bool(d.get("same_dataset_fingerprint"))
            and d.get("exit_code") == 0 and bool(d.get("checkpoints_ok")))


def resolve_ckpt(name: str, v82=None, choose: dict = None, root: pathlib.Path = ROOT) -> dict:
    if name not in REGISTERED:
        raise SystemExit(f"知らないモデル {name}（{sorted(REGISTERED)}）")
    base, run, wrap = REGISTERED[name]
    run = (choose or {}).get(name) or run
    if "_smoke" in run:
        raise SystemExit(f"{name}: smoke の実行 {run} は使わない")
    rel = "/".join((base, run) + CKPT_SUB)
    out = {"name": name, "path": rel, "run": run, "registered": rel == registered_path(name), "postcheck": None, "warnings": []}
    if wrap is None:                                    # 段階 3 の R1v3・N1v3: 82 の CKPT と同じこと
        have = None if v82 is None else v82.CKPT.get(name)
        if have is None or pathlib.PurePath(have).as_posix() != rel:
            raise SystemExit(f"{name}: 82 の CKPT {have} が登録版 第 5 節の {rel} と違う（新しい学習の出力がある？）")
        out["source"] = "82_v2_eval.CKPT（段階 3、2 万手）"
    else:
        pc = root / wrap / f"postcheck_{run}.json"
        try:
            ok = _postcheck_ok(json.loads(pc.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            ok = False
        if not ok:
            raise SystemExit(f"{name}: 後の点検 {pc.relative_to(root).as_posix()} が無いか通っていない")
        out["postcheck"] = pc.relative_to(root).as_posix()
        out["source"] = "束 3（99_s4_train_seed.py --post-check を通った実行。掲示板 0167）"
    if not (root / rel).is_dir():
        raise SystemExit(f"{name}: 2 万手の保存点 {rel} がない")
    last = root / base / run / "checkpoints" / "last" / "pretrained_model"
    if last.is_dir():
        try:
            same = last.resolve() == (root / rel).resolve() or ckpt_digest(last)["sha256"] == ckpt_digest(root / rel)["sha256"]
        except SystemExit:
            same = False
        if not same:
            out["warnings"].append("checkpoints/last が 020000 と違う中身（登録版 第 5 節の 020000 を使う）")
    return out


def ckpt_info(name: str, v82=None, choose: dict = None, expect: dict = None, root: pathlib.Path = ROOT) -> dict:
    r = resolve_ckpt(name, v82, choose, root)
    dg = ckpt_digest(root / r["path"])
    r.update(sha256=dg["sha256"], files_sha256=dg["files"], digest_rule=dg["rule"])
    want = (expect or {}).get(name)
    if want and want != r["sha256"]:
        raise SystemExit(f"{name}: 保存点の SHA-256 {r['sha256']} が掲示した値 {want} と違う（起動しない）")
    r["expected_sha256"] = want
    return r


EXPECT_KEYS = tuple(REGISTERED) + ("executor_v3", "rtc_module", "entry")
EXPECT_FILES = {"executor_v3": "src/recovla/runtime/executor_v3.py", "rtc_module": "src/recovla/diag/rtc.py",
                "entry": "scripts/98_s4_test1_eval.py"}


def parse_pairs(items: list, what: str, keys=EXPECT_KEYS) -> dict:
    out = {}
    for x in items or []:
        if "=" not in x:
            raise SystemExit(f"{what} は 名前=値 の形: {x}")
        k, v = x.split("=", 1)
        if k not in keys:
            raise SystemExit(f"{what}: 知らない名前 {k}（{list(keys)}）")
        out[k] = v
    return out


def check_expect_files(expect: dict, fsha: dict) -> None:
    for k, rel in EXPECT_FILES.items():
        if expect.get(k) and expect[k] != fsha[rel]:
            raise SystemExit(f"{rel} の SHA-256 {fsha[rel]} が掲示した値 {expect[k]} と違う（起動しない）")


def inject_ckpt(v82, info: dict) -> None:
    """読み込んだ 82 の写しの CKPT に足す（82 のファイルは書き換えない）。"""
    have = v82.CKPT.get(info["name"])
    if have is not None and pathlib.PurePath(have).as_posix() != info["path"]:
        raise SystemExit(f"{info['name']}: 82 の CKPT に別の保存点 {have} がある（{info['path']} と違う）")
    v82.CKPT[info["name"]] = info["path"]


# ---------------------------------------------------------------- 引数・環境
def forbidden_flag(f: str, names) -> bool:
    """f（--名前 か --名前=値）が names のどれかに当たるか。argparse の省略形も前方一致で拒む（0166）。"""
    if not f.startswith("--"):
        return False
    name = f.split("=", 1)[0]
    return len(name) > 2 and any(x.startswith(name) for x in names)


def has_flag(extra: list, flag: str, min_len: int) -> bool:
    return any(x.startswith("--") and len(x.split("=", 1)[0]) >= min_len and flag.startswith(x.split("=", 1)[0]) for x in extra)


def record_envs(dirs: list) -> dict:
    out = {}
    for d in dirs:
        seen = []
        if d.is_dir():
            for p in sorted(list(d.glob("trial_[0-9][0-9][0-9][0-9].json")) + list(d.glob("run_[0-9][0-9][0-9][0-9].json"))):
                try:
                    e = json.loads(p.read_text(encoding="utf-8")).get("env")
                except (OSError, json.JSONDecodeError):
                    continue
                k = None if not isinstance(e, dict) else tuple((x, e.get(x)) for x in ENV_KEYS)
                if k not in seen:
                    seen.append(k)
        out[d.name] = seen
    return out


def env_mismatch(cur: dict, rec: dict, expected_driver: str = EXPECTED_DRIVER) -> dict:
    out = {}
    if (cur or {}).get("driver") != expected_driver:
        out["driver"] = {"now": (cur or {}).get("driver"), "expected": expected_driver}
    now = tuple((x, (cur or {}).get(x)) for x in ENV_KEYS)
    bad = {c: [("env なし" if e is None else dict(e)) for e in envs if e != now] for c, envs in rec.items()}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        out["records"] = bad
    return out


# ---------------------------------------------------------------- 96 を包む
def restore96(r96) -> None:
    """96 の写しを最初の形に戻す（同じプロセスで別の条件を包み直すとき、前の包みを残さない）。"""
    keys = ("Engine", "SPEC_KEYS", "make_spec", "run_json_text", "overlay_limits", "trial_paths", "check_complete", "write_trial_files")
    pr = getattr(r96, "_t1_pristine", None)
    if pr is None:
        r96._t1_pristine = {k: getattr(r96, k) for k in keys if hasattr(r96, k)}
        return
    for k, v in pr.items():
        setattr(r96, k, v)


def wrap_t1(r96, info: dict) -> None:
    """今の 96 の写し（家族の包みの後）の上に "t1" を足す。"""
    Base, base_spec, base_text = r96.Engine, r96.make_spec, r96.run_json_text
    role = info["role"]

    class T1Engine(Base):
        def run_one(self, i, seed, lay, tgt):
            meta, arrays, rlog = super().run_one(i, seed, lay, tgt)
            if info["rtc"] and info["induce"] and isinstance(meta.get("diag"), dict):   # d_rtc は自然の値を書くので誘発に直す（B2 と同じ）
                meta["diag"]["induce"] = (meta.get("induce") or {}).get("kind")
                meta["diag"]["start_state"] = "induced（41_results.trial_list の induced。配置と目標は種から決まる）"
            meta["t1"] = dict(info)
            return meta, arrays, rlog

        def run_one_task(self, i, seed):
            meta, arrays, rlog = super().run_one_task(i, seed)
            meta["t1"] = dict(info)
            return meta, arrays, rlog

    def make_spec(a):
        s = base_spec(a)
        s.update(t1_branch=info["branch"], t1_role=role, t1_ckpt_path=info["ckpt"]["path"], t1_ckpt_sha256=info["ckpt"]["sha256"],
                 t1_rtc=(info["rtc"] or {}).get("setting"), t1_entry_sha256=info["files_sha256"]["scripts/98_s4_test1_eval.py"])
        return s

    def run_json_text(a_, kind, rows, *x, **kw):
        d = json.loads(base_text(a_, kind, rows, *x, **kw))
        d["t1"] = dict(info)
        return json.dumps(d, ensure_ascii=False, indent=1)

    r96.Engine, r96.make_spec, r96.run_json_text = T1Engine, make_spec, run_json_text


def rtc_info(setting: str) -> dict:
    from recovla.diag import rtc as D
    if setting not in D.CANDIDATES:
        raise SystemExit(f"--rtc-setting は関門 R の候補 {list(D.CANDIDATES)} のどれか（渡された {setting}）")
    s = D.setting(setting)
    return {"setting": setting, "mode": s["mode"], "setting_def": s,
            "setting_sha256": hashlib.sha256(json.dumps(s, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
            "module_sha256": sha256_file(ROOT / "src" / "recovla" / "diag" / "rtc.py")}


def setup_condition(r96, c: dict, info: dict) -> None:
    """96 の写しを戻し、家族の包み（E7 は 98_s4_b1.py、RTC は 98_s4_d_rtc.py）を当て、"t1" を足す。"""
    restore96(r96)
    if c["kind"] == "task":
        b1 = _load(ROOT / "scripts" / "98_s4_b1.py", "s4_b1_for_t1")
        b1.patch96(r96, c["arm"], info["v3"] or {"a_goal": None}, info["b1"])
    elif c["rtc"]:
        dr = _load(ROOT / "scripts" / "98_s4_d_rtc.py", "s4_d_rtc_for_t1")
        dr.patch96(r96, c["rtc"], False, info["files_sha256"]["scripts/98_s4_d_rtc.py"], info["rtc"]["module_sha256"])
    wrap_t1(r96, info)


def build_96_argv(c: dict, experiment: str, trials: str, max_new: int = None) -> list:
    if c["kind"] == "task":
        argv = ["task", "--experiment", experiment, "--condition", c["cond"], "--model", c["model"], "--trials", trials,
                "--exec-interval", "6", "--no-safety", "--planner", "s4", "--step-timeout-s", "30", "--task-time-limit-s", "200"]
    else:
        mode = "naive"
        if c["rtc"]:
            from recovla.diag import rtc as D
            mode = D.setting(c["rtc"])["mode"]
        argv = ["run", "--experiment", experiment, "--condition", c["cond"], "--model", c["model"], "--trials", trials,
                "--mode", mode, "--exec-interval", "6", "--no-safety", "--time-limit-s", "60"]
        if c["induce"]:
            argv += ["--induce", c["induce"]]
    if max_new:
        argv += ["--max-new", str(int(max_new))]
    return argv


def make_info(c: dict, branch, ck: dict, fsha: dict) -> dict:
    v3 = {"a_goal": None} if c["arm"] in ("R1v3_v3", "N1v3_v3") else None
    info = {"script": "scripts/98_s4_test1_eval.py", "branch": branch, "cond": c["cond"], "role": c["key"], "group": c["group"],
            "model": c["model"], "induce": c["induce"], "executor": (None if c["kind"] != "task" else ("v3" if v3 else "current")),
            "ckpt": {k: ck.get(k) for k in ("path", "run", "registered", "postcheck", "source", "sha256", "expected_sha256", "warnings")},
            "rtc": rtc_info(c["rtc"]) if c["rtc"] else None, "files_sha256": dict(fsha), "driver_expected": EXPECTED_DRIVER,
            "prereg": "docs/stage4/prereg_test1_v1.md（掲示板 0168）", "v3": v3, "b1": None}
    if c["kind"] == "task":
        info["b1"] = {"script": "98_s4_test1_eval.py", "arm": c["arm"], "model": c["model"],
                      "executor": "v3" if v3 else "current", "executor_version": V3_VERSION if v3 else "current", "v3": v3,
                      "v3_source": {"gate_T": "不合格（掲示板 0163）。(a) は入れない"} if v3 else None,
                      "files_sha256": {p: fsha[p] for p in ("src/recovla/runtime/executor_v3.py", "src/recovla/runtime/executor.py",
                                                            "scripts/98_s4_b1.py", "scripts/96_s4_resume.py", "configs/s4_gates.json")}}
    return info


def run_condition(r96, ops, v82, a, conds: list, name: str, extra: list, max_new: int = None) -> int:
    for f in extra:
        if forbidden_flag(f, FORBIDDEN_EXTRA):
            raise SystemExit(f"{f} は 98_s4_test1_eval.py が決める（モデル・帯・実行のしかた・誘発・制限時間・計画役）")
    c = find(conds, name)
    trials = a.trials or c["trials"]
    if c["group"] == "health" and not a.trials:
        trials = a.health_spec
    why = check_band(c, trials, a.experiment, a.allow_smoke)
    if why:
        raise SystemExit(why)
    expect = parse_pairs(a.expect, "--expect")
    fsha = files_sha256()
    check_expect_files(expect, fsha)
    if fsha["configs/s4_gates.json"] != POSTED_GATES_SHA256:
        raise SystemExit(f"configs/s4_gates.json の SHA-256（LF）{fsha['configs/s4_gates.json']} が掲示の値（0165）と違う")
    ck = ckpt_info(c["model"], v82, parse_pairs(a.run, "--run", tuple(REGISTERED)), expect)
    for w in ck["warnings"]:
        print(f"[t1] 警告 {c['model']}: {w}", flush=True)
    inject_ckpt(v82, ck)
    info = make_info(c, a.branch, ck, fsha)
    setup_condition(r96, c, info)
    a96 = r96.build_parser().parse_args(build_96_argv(c, a.experiment, trials, max_new) + list(extra))
    if c["kind"] == "task":                            # 98_s4_b1.py の run_arm と同じ控えの鍵
        a96.b1_arm = c["arm"]
        a96.b1_v3 = json.dumps(info["b1"]["v3"], sort_keys=True)
        a96.b1_executor_sha256 = fsha["src/recovla/runtime/executor_v3.py"] if info["v3"] else None
        a96.b1_executor_version = info["b1"]["executor_version"]
    print(f"[t1] 枝 {a.branch} {name}: {c['model']}・{c['group']}{'・RTC ' + c['rtc'] if c['rtc'] else ''}"
          f"{'・実行器 ' + info['executor'] if info['executor'] else ''}（{a.experiment}\\{name} {trials}、保存点 {ck['path']} "
          f"SHA-256 {ck['sha256'][:12]}…）", flush=True)
    try:
        return r96.cmd_main(a96, v82, ops)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


def _env():
    r96 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_t1")
    return r96, r96.load_ops(), r96.load_82(False)


def env_gate(r96, ops, v82, a, names: list, extra: list) -> int:
    cur = r96.read_env(ops)
    mm = env_mismatch(cur, record_envs([v82.OUT / a.experiment / n for n in names]))
    print(f"[t1] 環境: {r96.env_brief(cur)}", flush=True)
    if "driver" in mm:
        print(f"[t1] 警告: ドライバが {mm['driver']['now']}（決めた版 {EXPECTED_DRIVER} と違う）。止めずに続ける（0166）。版は試行の env に残る",
              flush=True)
    if "records" in mm:
        print(f"[t1] 環境の食い違い（記録と今）: {json.dumps(mm['records'], ensure_ascii=False)}", flush=True)
        if not has_flag(extra, "--accept-env-change", len("--accept-e")):
            print("[t1] 本番なら止める（dry-run は続けて見せる）。続けるなら --accept-env-change（96 が環境の区切りを書く。報告で分ける）",
                  file=sys.stderr, flush=True)
            return 3
    return 0


def _conds_of(a) -> list:
    conds = conditions("health" if a.cmd == "health" else a.branch, getattr(a, "rtc_setting", None))
    if getattr(a, "groups", None):
        keep = set(a.groups.split(","))
        conds = [c for c in conds if c["group"] in keep]
    return conds


def cmd_run(a, extra) -> int:
    conds = _conds_of(a)
    r96, ops, v82 = _env()
    rc = env_gate(r96, ops, v82, a, [a.cond], extra)
    if rc and not a.dry_run:
        return rc
    return max(rc, run_condition(r96, ops, v82, a, conds, a.cond, extra + (["--dry-run"] if a.dry_run else []), a.max_new))


# ---------------------------------------------------------------- rotate（ワーカーで分け合う）
def seeds_done(c: dict, done: int) -> int:
    return done // per_seed(c)


def pick_next(conds: list, done: dict, live: set) -> dict:
    """次に回す条件。そろっていない・ほかで回っていない・同じ組で 1 塊より先に進んでいないもののうち、塊の番号が若いもの
    （同じなら E7 → P1 → 自然、条件の順）。無ければ None。"""
    best = None
    for idx, c in enumerate(conds):
        tot = n_trials_of(c)
        if done[c["cond"]] >= tot or c["cond"] in live:
            continue
        grp = [x for x in conds if x["group"] == c["group"] and done[x["cond"]] < n_trials_of(x)]
        low = min(seeds_done(x, done[x["cond"]]) for x in grp)
        mine = seeds_done(c, done[c["cond"]])
        if mine >= low + BLOCK_SEEDS[c["group"]]:
            continue                                      # 1 塊より先に進まない（登録版 第 8 節）
        key = (mine // BLOCK_SEEDS[c["group"]], GROUP_RANK[c["group"]], idx)
        if best is None or key < best[0]:
            best = (key, c)
    return None if best is None else best[1]


def n_trials_of(c: dict) -> int:
    return n_trials(c.get("trials_run") or c["trials"])


class Claims:
    """条件ごとの占有（ワーカーの間で 2 重に回さない）。outputs\\s4\\test1_eval\\claims\\<実験>\\<条件>.claim。"""

    def __init__(self, d: pathlib.Path, alive):
        self.d, self.alive = d, alive
        d.mkdir(parents=True, exist_ok=True)

    def _p(self, name):
        return self.d / f"{name}.claim"

    def live(self, name) -> bool:
        p = self._p(name)
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if x.get("pid") == os.getpid():
            return False
        if self.alive(int(x.get("pid", -1)), x.get("proc_create_time")):
            return True
        try:
            p.unlink()                                    # 死んだワーカーの占有は外す
        except OSError:
            pass
        return False

    def take(self, name, worker) -> bool:
        if self.live(name):
            return False
        try:
            fd = os.open(str(self._p(name)), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        try:
            import psutil
            ct = psutil.Process().create_time()
        except Exception:                                # noqa: BLE001
            ct = None
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "proc_create_time": ct, "worker": worker, "at": time.strftime("%Y-%m-%d %H:%M:%S")}))
        return True

    def drop(self, name) -> None:
        try:
            self._p(name).unlink()
        except OSError:
            pass


def shared_loop(conds: list, count, call, is_live, claims: Claims, worker, prog, log_path: pathlib.Path, stop_check,
                wait_s: float = 30.0, max_idle_s: float = 6 * 3600.0, sleep=time.sleep) -> int:
    """98_s4_b1.py の rotate_loop と同じ止め方で、ワーカーの間で条件を分け合う。count(条件)＝完全な試行の数、call(条件)＝(終了コード,
    子の progress.json)、is_live(条件)＝ほかで回っているか、stop_check()＝止める合図（"" ならなし）。"""
    status, code, idle = "done", 0, 0.0
    prog.start()
    prog.put(status="running")
    try:
        while True:
            done = {c["cond"]: count(c["cond"]) for c in conds}
            prog.put(done=sum(done.values()), counts=done)
            left = [c for c in conds if done[c["cond"]] < n_trials_of(c)]
            if not left:
                break
            s = stop_check()
            if s:
                status, code = "stopped", 1
                prog.put(stop_reason=s)
                print(f"[rotate] 止める合図 {s}", flush=True)
                return code
            live = {c["cond"] for c in left if is_live(c["cond"])}
            c = pick_next(conds, done, live)
            if c is None or not claims.take(c["cond"], worker):
                if idle >= max_idle_s:
                    raise RuntimeError(f"{max_idle_s:.0f} s 待っても回せる条件がない（ほかのワーカー: {sorted(live)}）")
                prog.put(waiting={"live_elsewhere": sorted(live), "since_s": idle})
                sleep(wait_s)
                idle += wait_s
                continue
            idle = 0.0
            name = c["cond"]
            t0 = time.time()
            prog.put(current=name, current_started=time.strftime("%Y-%m-%d %H:%M:%S"), waiting=None)
            try:
                rc, pr = call(name)
            finally:
                claims.drop(name)
            new = count(name)
            with prog.lock:
                prog.d["calls"].append({"name": name, "code": rc, "status": pr.get("status"), "stop_reason": pr.get("stop_reason"),
                                        "done_before": done[name], "done_after": new, "wall_s": round(time.time() - t0, 1)})
                prog.write(log_path, json.dumps(prog.d["calls"], ensure_ascii=False, indent=1))
            reason = str(pr.get("stop_reason") or "")
            if rc == 0 or (rc == 1 and reason.startswith("max_new")):
                if new <= done[name] and new < n_trials_of(c):
                    raise RuntimeError(f"{name}: 呼び出しで完全な試行が増えない（{done[name]} → {new}）")
                continue
            status, code = ("stopped", 1) if rc == 1 else ("error", rc if rc in (2, 3) else 2)
            prog.put(stop_reason=f"{name}: {reason or pr.get('error') or f'終了コード {rc}'}")
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


def child_argv(a, name: str, extra: list, block: int) -> list:
    sub = "health" if a.cmd == "health" else "run"
    argv = [sys.executable, str(pathlib.Path(__file__).resolve()), sub]
    if sub == "run":
        argv += ["--branch", str(a.branch)]
        if a.rtc_setting:
            argv += ["--rtc-setting", a.rtc_setting]
    else:
        argv += ["--health-spec", a.health_spec]
    argv += ["--one", name, "--experiment", a.experiment, "--max-new", str(int(block))]
    for x in a.run or []:
        argv += ["--run", x]
    for x in a.expect or []:
        argv += ["--expect", x]
    if a.allow_smoke:
        argv += ["--allow-smoke"]
    if a.trials:
        argv += ["--trials", a.trials]
    return argv + list(extra)


def cmd_rotate(a, extra) -> int:
    conds = _conds_of(a)
    if not conds:
        raise SystemExit("回す条件がない（--groups）")
    r96, ops, v82 = _env()
    for c in conds:
        c["trials_run"] = a.trials or (a.health_spec if c["group"] == "health" else c["trials"])
        why = check_band(c, c["trials_run"], a.experiment, a.allow_smoke)
        if why:
            raise SystemExit(why)
    expect = parse_pairs(a.expect, "--expect")
    fsha = files_sha256()
    check_expect_files(expect, fsha)
    for m in sorted({c["model"] for c in conds}):           # 子を起こす前に、保存点がそろうことを確かめる
        ck = ckpt_info(m, v82, parse_pairs(a.run, "--run", tuple(REGISTERED)), expect)
        for w in ck["warnings"]:
            print(f"[t1] 警告 {m}: {w}", flush=True)
    rc = env_gate(r96, ops, v82, a, [c["cond"] for c in conds], extra)
    total = sum(n_trials_of(c) for c in conds)
    print(f"[t1] {'健全性の確認' if a.cmd == 'health' else f'枝 {a.branch}'}: {len(conds)} 条件、{total} 試行", flush=True)
    if a.dry_run:
        for c in conds:
            sub = argparse.Namespace(**dict(vars(a), trials=c["trials_run"] if (a.allow_smoke or c["group"] == "health") else None))
            rc = max(rc, run_condition(r96, ops, v82, sub, conds, c["cond"], extra + ["--dry-run"]))
        return rc
    if rc:
        return rc
    for f in extra:
        if forbidden_flag(f, ROTATE_FORBIDDEN + FORBIDDEN_EXTRA):
            raise SystemExit(f"{f} は rotate では渡せない（rotate・98_s4_test1_eval.py が決める）")
    B1 = _load(ROOT / "scripts" / "98_s4_b1.py", "s4_b1_for_t1")
    plans = {}
    for c in conds:
        if c["kind"] == "task":
            base, n = (int(x) for x in c["trials_run"].split(":"))
            plans[c["cond"]] = [(s, None) for s in range(base, base + n)]
        else:
            plans[c["cond"]] = [(s, t) for s, _, t in v82.trial_list(c["trials_run"])]
    byname = {c["cond"]: c for c in conds}
    tag = f"{a.experiment}_{'health' if a.cmd == 'health' else 'b' + str(a.branch)}_w{a.worker}"
    OUTD.mkdir(parents=True, exist_ok=True)
    prog_p, log_p = OUTD / f"rotate_{tag}.progress.json", OUTD / f"rotate_{tag}.log.json"
    B1.refuse_if_live(prog_p, ops)
    info_cache = {}

    def count(name):
        c = byname[name]
        if c["rtc"]:                                       # d_rtc の check_complete は diag も照らす
            if name not in info_cache:
                info_cache[name] = {"rtc": rtc_info(c["rtc"]), "files_sha256": fsha}
            restore96(r96)
            _load(ROOT / "scripts" / "98_s4_d_rtc.py", "s4_d_rtc_for_t1").patch96(
                r96, c["rtc"], False, fsha["scripts/98_s4_d_rtc.py"], info_cache[name]["rtc"]["module_sha256"])
        else:
            restore96(r96)
        out = v82.OUT / a.experiment / name
        kind = c["kind"]
        n = sum(int(r96.check_complete(kind, out, i, {"seed": s} if kind == "task" else {"seed": s, "target": t})[0])
                for i, (s, t) in enumerate(plans[name]))
        restore96(r96)
        return n

    def call(name):
        code = B1.run_child(child_argv(a, name, extra, block_trials(byname[name])))
        return code, B1.read_progress(v82.OUT / a.experiment / name / "progress.json")

    def is_live(name):
        p = v82.OUT / a.experiment / name / "progress.json"
        pr = B1.read_progress(p)
        return bool(pr.get("status") in ops.LIVE_STATUS and pr.get("pid") and pr.get("pid") != os.getpid()
                    and ops._alive(int(pr["pid"]), pr.get("proc_create_time")))

    claims = Claims(OUTD / "claims" / a.experiment, ops._alive)

    def stop_check():
        for stop in [ROOT / "outputs" / "s4" / "STOP", OUTD / f"rotate_{a.experiment}.STOP", OUTD / f"rotate_{tag}.STOP"] + \
                    [v82.OUT / a.experiment / c["cond"] / "STOP" for c in conds]:
            if stop.is_file():
                return f"stop_file:{stop}"
        return ""

    prog = B1.RotateProgress(prog_p, {"experiment": a.experiment, "condition": f"rotate:{tag}", "branch": a.branch, "worker": a.worker,
                                      "conditions": [c["cond"] for c in conds], "trials_spec": {c["cond"]: c["trials_run"] for c in conds},
                                      "total": total, "block_seeds": BLOCK_SEEDS, "isolation": "subprocess（腕ごとの子プロセス）",
                                      "rtc_setting": a.rtc_setting if a.cmd != "health" else None,
                                      "child_progress": {c["cond"]: str(v82.OUT / a.experiment / c["cond"] / "progress.json") for c in conds},
                                      "stop_file": str(OUTD / f"rotate_{a.experiment}.STOP")}, r96._write_atomic)
    return shared_loop(conds, count, call, lambda n: is_live(n) or claims.live(n), claims, a.worker, prog, log_p, stop_check)


# ---------------------------------------------------------------- 読むだけのもの
def estimate(branch, rtc_setting: str = None) -> dict:
    rows, tot = [], 0.0
    for c in conditions(branch, rtc_setting):
        n = n_trials(c["trials"])
        if c["group"] == "e7":
            per = PER_TRIAL_H["e7"]
        elif c["group"] == "health":
            per = PER_TRIAL_H["health"]
        elif c["rtc"]:
            per = PER_TRIAL_H["rtc_nat"] if c["group"] == "nat" else PER_TRIAL_H["rtc_p1_" + c["model"][0]]
        elif c["group"] == "p1":
            per = PER_TRIAL_H["p1_" + c["model"][0]]
        else:
            per = PER_TRIAL_H["nat"]
        rows.append({"cond": c["cond"], "trials": n, "process_h": round(n * per, 2)})
        tot += n * per
    return {"rows": rows, "trials": sum(r["trials"] for r in rows), "process_h": round(tot, 1), "parallel3_h": round(tot / 2.85, 1),
            "registered": None if branch == "health" else {k: BRANCHES[branch][k] for k in ("trials", "process_h", "parallel3_h")},
            "note": "推測。1 試行の値は登録版 第 5 節の表から割り戻した"}


def cmd_plan(a) -> int:
    branch = "health" if a.health else a.branch
    rs = None if branch == "health" else a.rtc_setting
    conds = conditions(branch, rs)
    out = {"branch": branch, "rtc_setting": rs, "experiment": HEALTH_EXPERIMENT if branch == "health" else EXPERIMENT,
           "band_problems": check_tables(),
           "conditions": [{k: c[k] for k in ("cond", "key", "group", "kind", "model", "trials", "induce", "rtc", "arm")} for c in conds],
           "estimate": estimate(branch, rs), "block_seeds": BLOCK_SEEDS}
    if branch != "health" and out["estimate"]["trials"] != BRANCHES[branch]["trials"]:
        out["band_problems"].append(f"試行数 {out['estimate']['trials']} が登録版の {BRANCHES[branch]['trials']} と違う")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if not out["band_problems"] else 3


def cmd_ckpt(a) -> int:
    v82 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_t1").load_82(False)
    expect = parse_pairs(a.expect, "--expect")
    res, bad = {}, {}
    for m in REGISTERED:
        try:
            res[m] = ckpt_info(m, v82, parse_pairs(a.run, "--run", tuple(REGISTERED)), expect)
        except SystemExit as e:
            bad[m] = str(e.code)
    fsha = files_sha256()
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "ckpt.json").write_text(json.dumps({"models": res, "problems": bad, "files_sha256": fsha}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    for m, r in res.items():
        print(f"{m}: {r['path']}  SHA-256 {r['sha256']}{'  警告 ' + '; '.join(r['warnings']) if r['warnings'] else ''}")
    for m, why in bad.items():
        print(f"{m}: 決められない — {why}")
    for k, rel in EXPECT_FILES.items():
        print(f"{k}: {rel}  SHA-256 {fsha[rel]}")
    print(f"configs/s4_gates.json（LF）: {fsha['configs/s4_gates.json']}（掲示 0165: {'一致' if fsha['configs/s4_gates.json'] == POSTED_GATES_SHA256 else '違う'}）")
    return 0 if not bad else 3


def layout_params(branch, rtc_setting: str = None, versions: dict = None) -> tuple:
    """98_s4_test1.py check の layout・params（登録版 第 13-1 節の値）。"""
    br = BRANCHES[branch]
    lay = {"root": "outputs/v2eval", "experiment": EXPERIMENT}
    for c in conditions(branch, rtc_setting):
        k = c["key"].split(".")
        ent = {"cond": c["cond"], "model": c["model"]}
        if k[0] == "e7":
            lay.setdefault("e7", {})[k[1]] = ent
        elif k[0] == "rtc":
            lay.setdefault("rtc", {})[k[1]] = ent
        else:
            lay.setdefault(k[0], {}).setdefault(k[1], {})[k[2]] = ent
    params = {"h1": br["b1"], "e7_n": E7_N, "rtc_arm": br["b2"], "plan": br["plan"], "guard_mode": "point", "ni_margin": 0.10,
              "h2_layers": ["1001", "1002"], "rtc_p1_n": 100 if br["plan"] == "B" else 50, "e7_band_extended": False,
              "c4_on_time": None, "p_fill": {"branch": branch, "note": "P-n の値と出どころは「枝の確定」の掲示から写す"},
              "versions": versions}
    return lay, params


def versions_now(branch, rtc_setting: str, v82=None, expect: dict = None) -> dict:
    fsha = files_sha256()
    br = BRANCHES[branch]
    ck = {}
    for m in REGISTERED:
        try:
            ck[m] = ckpt_info(m, v82, None, expect)["sha256"]
        except SystemExit:
            ck[m] = (expect or {}).get(m) or "<枝の確定の掲示の値>"
    return {"entry_sha256": fsha["scripts/98_s4_test1_eval.py"],
            "executor_v3_sha256": fsha["src/recovla/runtime/executor_v3.py"] if br["b1"] else None,
            "rtc_setting": rtc_setting if br["b2"] else None,
            "rtc_module_sha256": fsha["src/recovla/diag/rtc.py"] if br["b2"] else None, "ckpt_sha256": ck}


def cmd_layout(a) -> int:
    try:
        v82 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_t1").load_82(False)
    except SystemExit:
        v82 = None
    ver = versions_now(a.branch, a.rtc_setting, v82, parse_pairs(a.expect, "--expect"))
    lay, params = layout_params(a.branch, a.rtc_setting, ver)
    d = pathlib.Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "layout.json").write_text(json.dumps(lay, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"書いた: {d / 'layout.json'}、{d / 'params.json'}（versions は今のファイルから計算した値。「枝の確定」の掲示の値と見比べる）")
    return 0


# ---------------------------------------------------------------- 引数
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "ckpt", "run", "rotate", "health", "layout"):
        p = sub.add_parser(name)
        if name in ("plan", "run", "rotate", "layout"):
            p.add_argument("--branch", type=int, required=name != "plan", choices=sorted(BRANCHES))
            p.add_argument("--rtc-setting", default=None, help="枝 1・3 だけ（束 2 の判定の掲示の設定）")
        if name == "plan":
            p.add_argument("--health", action="store_true", help="健全性の確認の条件を出す")
        if name in ("ckpt", "run", "rotate", "health", "layout"):
            p.add_argument("--run", action="append", default=[], help="保存点の実行を変える（R1v3s1001=<実行名>。記録に残る）")
            p.add_argument("--expect", action="append", default=[],
                           help="掲示した SHA-256（R1v3s1001=…、executor_v3=…、rtc_module=…、entry=…）。違えば起動しない")
        if name in ("run", "rotate", "health"):
            p.add_argument("--experiment", default=None, help="既定は S4T1（健全性の確認は S4T1HC）")
            p.add_argument("--trials", default=None, help="smoke だけ（--allow-smoke と一緒に）。本番は表の帯")
            p.add_argument("--allow-smoke", action="store_true")
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--health-spec", default=HEALTH_SPECS[0], choices=HEALTH_SPECS, help="健全性の確認の帯の指定")
        if name in ("run", "health"):
            p.add_argument("--one", dest="cond", default=None, help="1 条件だけ回す（rotate の子が使う）")
            p.add_argument("--max-new", type=int, default=None)
        if name == "run":
            p.add_argument("--cond", dest="cond", default=None, help="条件名（plan の conditions）")
        if name in ("rotate", "health"):
            p.add_argument("--worker", type=int, default=1, help="3 本並行のときのワーカーの番号（1〜3）")
            p.add_argument("--groups", default=None, help="カンマ区切り（e7,p1,nat）。既定は全部")
        if name == "layout":
            p.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    a, extra = build_parser().parse_known_args(argv)
    if a.cmd in ("run", "rotate", "health") and a.experiment is None:
        a.experiment = HEALTH_EXPERIMENT if a.cmd == "health" else EXPERIMENT
    if a.cmd == "health":
        a.branch, a.rtc_setting = "health", None
    if a.cmd not in ("run", "rotate", "health") and extra:
        print(f"知らない引数: {extra}", file=sys.stderr)
        return 3
    try:
        if a.cmd == "plan":
            if not a.health and a.branch is None:
                raise SystemExit("--branch（1〜4）か --health が要る")
            return cmd_plan(a)
        if a.cmd == "ckpt":
            return cmd_ckpt(a)
        if a.cmd == "layout":
            return cmd_layout(a)
        if a.cmd in ("run", "health") and a.cond:
            return cmd_run(a, extra)
        if a.cmd == "run":
            raise SystemExit("run には --cond が要る（全部を回すなら rotate）")
        return cmd_rotate(a, extra)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
