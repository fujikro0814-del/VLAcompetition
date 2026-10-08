"""段階 4 束 4 の評価の入口（関門 2・関門 3・テスト 2）。96_s4_resume.py run を包み、R4・N4（種 1000・1001）と R1v3（種 1000・1001）を、
同じ種で種の塊ごとに交互に回す（事前登録 v2 の草案 第 4・5 節・12-2 節、掲示板 0165）。

使い方（作業場所 C:\\PAI\\recovery_vla。96_s4_resume.py run の続きの引数（--accept-env-change・--min-free-gb など）は後ろに付ける）:
    .venv\\Scripts\\python.exe scripts\\98_s4_b4_eval.py plan --phase G2                      # 条件・帯・試行数・見込みの時間（読むだけ）
    .venv\\Scripts\\python.exe scripts\\98_s4_b4_eval.py ckpt                                 # 保存点の対応と SHA-256（読むだけ）→ outputs\\s4\\b4_eval\\ckpt.json
    .venv\\Scripts\\python.exe scripts\\98_s4_b4_eval.py run --phase G2 --cond P1_R4 --dry-run  # 1 条件だけ（何を回すかを見る）
    .venv\\Scripts\\python.exe scripts\\98_s4_b4_eval.py rotate --phase G2 [--block 10] [--dry-run]
        段階の全条件（部分 x モデル）を、同じ種で 10 種の塊ごとに交互に回す（自然は 10 種 = 30 試行）。腕の 1 回の呼び出し
        （--max-new <塊>）ごとに子プロセスを起こすので、模型は 1 つずつしか持たない。止まったら同じコマンドで続きから回る
        （96 が完全な記録を飛ばす）。止め方・進み具合は 98_s4_b1.py の rotate_loop と同じ: 続けるのは子が --max-new で止まったとき
        だけ。1 巡して完全な試行が増えなければエラー。進み具合は outputs\\s4\\b4_eval\\rotate_<実験>.progress.json（pid 付き、1 分ごとに
        心拍。96_s4_ops.py wait --progress で見る）。止める合図: <条件>\\STOP、outputs\\s4\\STOP、outputs\\s4\\b4_eval\\rotate_<実験>.STOP。
        3 本並行は --parts で分けたプロセスにする（進み具合は rotate_<実験>_<部分>.progress.json。分けた組ごとに 1 つだけ回る）。
    .venv\\Scripts\\python.exe scripts\\98_s4_b4_eval.py layout --phase T2 --out outputs\\s4\\b4_eval\\layout_T2.json   # 二重集計の入力
    smoke: --allow-smoke --experiment S4SMOKE_B4 --trials induced:44680:2（種 44680〜44683 だけ。本番の帯は割り当てそのものだけ）
  ■ 試すときは必ず --dry-run を付ける（付けないと GPU で評価を回し、帯の種を使う）。

段階と条件（条件名 = <部分>_<モデル>。実験名の既定 G2 = S4B4G2、G3 = S4B4G3、T2 = S4T2）:
  G2（関門 2。bundle4_gate2 192100〜192999 の中の、事前登録 v2 の草案 12-2 の割り当て）:
    nat  natural:192100:66（198 試行）  R4・R1v3            守り 1
    P1   induced:192200:100 P1           R4・N4・R1v3        守り 2・3
    P2   induced:192300:100 P2           R4・R1v3            狙い
    P3   induced:192600:100 P3           R4・N4・R1v3        記述だけ（掲示板 0165）
  G3（関門 3）:
    P1   induced:192400:100 P1           R4s1001・N4s1001
    P2   induced:192500:100 P2           R4s1001・R1v3s1001
  T2（テスト 2。s4_gates.json の test2_*）:
    nat  natural:165000:66（test2_natural）  R4・R1v3
    P1   induced:166000:100（test2_P1）      R4・N4・R1v3
    P2   induced:167000:100（test2_drop）    R4・N4・R1v3      H1・H2
    P3   induced:168000:100（test2_misplace）R4・N4・R1v3      副次・族の外
  実行のしかた（段階 3 と同じ。事前登録 v2 の草案 第 5 節）: naive、行動の区切り 6 行、安全フィルタなし、単発の試行 60 s（30 s の採点が
    主）、試行ごとに世界を作り直す（96 の既定）。誘発は recovla.eval.induce.Inducer のまま（P1・P2・P3。手を止める版は使わない）。
保存点（82_v2_eval.py の CKPT にないものは、読み込んだ 82 の写しの CKPT に足す。82・96 は書き換えない）:
  R4・N4（種 1000）、R4s1001・N4s1001: outputs\\s4\\train_b4\\train_<R4|N4>s<種>_<日時>_*\\checkpoints\\020000\\pretrained_model。
    outputs\\s4\\b4_wrap\\postcheck_<実行名>.json（99_s4_train_b4.py --post-check）が通った（train_config_ok・exit_code 0・
    checkpoints_ok）実行だけ。通った実行が 2 つ以上なら止める（--run R4=<実行名> で選ぶ。記録に残る）。
  R1v3s1001: outputs\\s4\\train\\train_R1v3s1001_*（束 3）。outputs\\s4\\seed_wrap\\postcheck_<実行名>.json が通ったものだけ。
  R1v3: 82 の CKPT["R1v3"]（段階 3 の 2 万手）。事前登録 v2 の草案 第 5 節の実行（train_R1v3_20261005-180404_...）と同じことを確かめる。
  保存点の SHA-256: pretrained_model の中のファイルを相対パスの順に並べ、「相対パス<TAB>ファイルの SHA-256」の行をつないだものの
    SHA-256（ckpt_digest）。試行の json の "b4"、run.json の "b4"、resume_spec.json の b4_ckpt_sha256 に残す（再開のとき保存点が
    違えば 96 の控えの照合で止まる）。--expect R4=<SHA-256>（掲示した値。事前登録 v2 の P-2）を付ければ、違うときに起動しない。
環境の照合（96 の条件ごとの照合に足す）: 回の始めに、段階の全条件の完全な記録の env（ドライバ・torch・CUDA・OS）と今の環境、
  ドライバの版 610.88（事前登録 v2 の草案 第 5 節）を照らす。違えば止める（終了コード 3。--accept-env-change のときだけ進め、
  96 が環境の区切りを書く。報告で分ける）。
読むもの: scripts\\96_s4_resume.py・82_v2_eval.py（importlib。96 経由）、98_s4_b1.py（rotate の共通部）、configs\\s4_gates.json、
  outputs\\s4\\b4_wrap\\・seed_wrap\\ の postcheck_*.json、保存点のファイル。
書くもの: 96_s4_resume.py run と同じ記録（outputs\\v2eval\\<実験>\\<条件>\\）。試行の json と run.json に "b4"（段階・部分・モデル・保存点の
  パスと SHA-256・使ったファイルの SHA-256）。outputs\\s4\\b4_eval\\ に ckpt.json・rotate_<実験>.progress.json・.log.json・layout_*.json。
終了コード: 96_s4_resume.py と同じ（0 全部そろった、1 途中で止まった、2 エラー、3 引数・前提の食い違い）。
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
GATES = ROOT / "configs" / "s4_gates.json"
OUTD = ROOT / "outputs" / "s4" / "b4_eval"
B4_WRAP = ROOT / "outputs" / "s4" / "b4_wrap"
B4_TRAIN = ROOT / "outputs" / "s4" / "train_b4"
B3_WRAP = ROOT / "outputs" / "s4" / "seed_wrap"
B3_TRAIN = ROOT / "outputs" / "s4" / "train"
CKPT_SUB = ("checkpoints", "020000", "pretrained_model")
R1V3_RUN = "train_R1v3_20261005-180404_20261005-180404"           # 事前登録 v2 の草案 第 5 節・99_s4_train_seed.EXISTING_RUN
EXPECTED_DRIVER = "610.88"                                        # 事前登録 v2 の草案 第 5 節
ENV_KEYS = ("driver", "torch", "torch_cuda", "os_build")          # 96_s4_resume.ENV_STOP_KEYS と同じ
SMOKE_SEEDS = (44680, 44683)                                      # 依頼 12: smoke は 44680〜44683 だけ
FIXED = ["--mode", "naive", "--exec-interval", "6", "--no-safety", "--time-limit-s", "60"]

# 保存点の出どころ。kind: b4 = 99_s4_train_b4.py、b3 = 99_s4_train_seed.py（束 3）、s3 = 82 の CKPT（段階 3）
MODELS = {
    "R4": {"kind": "b4", "base": "R4", "seed": 1000, "ja": "R4（R1v3 ＋ 滑りの復帰デモ 50 本）、種 1000"},
    "N4": {"kind": "b4", "base": "N4", "seed": 1000, "ja": "N4（N1v3 ＋ 同じ配置の通常デモ 50 本）、種 1000"},
    "R4s1001": {"kind": "b4", "base": "R4", "seed": 1001, "ja": "R4、種 1001（関門 3）"},
    "N4s1001": {"kind": "b4", "base": "N4", "seed": 1001, "ja": "N4、種 1001（関門 3）"},
    "R1v3": {"kind": "s3", "base": "R1v3", "seed": 1000, "ja": "R1v3（段階 3、種 1000）"},
    "R1v3s1001": {"kind": "b3", "base": "R1v3", "seed": 1001, "ja": "R1v3、種 1001（束 3）"},
}
PART_INDUCE = {"nat": None, "P1": "P1", "P2": "P2", "P3": "P3"}
PART_PER_TRIAL_H = {"nat": 0.0104, "P1": 0.019, "P2": 0.027, "P3": 0.030}   # 事前登録 v2 の草案 第 5 節（P1 の N 系は 0.023）
# 段階 → 実験名の既定・部分ごとの帯（[下, 上]、自然は種の帯）・割り当ての id（test2_* は s4_gates.json と照らす）・モデル
PHASES = {
    "G2": {"experiment": "S4B4G2", "ja": "関門 2（検証）", "alloc_within": "bundle4_gate2", "parts": {
        "nat": {"band": (192100, 192165), "models": ("R4", "R1v3")},
        "P1": {"band": (192200, 192299), "models": ("R4", "N4", "R1v3")},
        "P2": {"band": (192300, 192399), "models": ("R4", "R1v3")},
        "P3": {"band": (192600, 192699), "models": ("R4", "N4", "R1v3")}}},
    "G3": {"experiment": "S4B4G3", "ja": "関門 3（種 1001）", "alloc_within": "bundle4_gate2", "parts": {
        "P1": {"band": (192400, 192499), "models": ("R4s1001", "N4s1001")},
        "P2": {"band": (192500, 192599), "models": ("R4s1001", "R1v3s1001")}}},
    "T2": {"experiment": "S4T2", "ja": "テスト 2", "alloc_within": None, "parts": {
        "nat": {"band": (165000, 165065), "alloc": "test2_natural", "models": ("R4", "R1v3")},
        "P1": {"band": (166000, 166099), "alloc": "test2_P1", "models": ("R4", "N4", "R1v3")},
        "P2": {"band": (167000, 167099), "alloc": "test2_drop", "models": ("R4", "N4", "R1v3")},
        "P3": {"band": (168000, 168099), "alloc": "test2_misplace", "models": ("R4", "N4", "R1v3")}}},
}
FILES = ("scripts/98_s4_b4_eval.py", "scripts/96_s4_resume.py", "scripts/82_v2_eval.py", "scripts/98_s4_b1.py", "configs/s4_gates.json")


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


def gates() -> dict:
    return json.loads(GATES.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 条件と帯
def conditions(phase: str) -> list:
    """[(条件名, 部分, モデル)]。並びは 部分の順 x モデルの順（rotate の順）。"""
    return [(f"{part}_{m}", part, m) for part, v in PHASES[phase]["parts"].items() for m in v["models"]]


def default_trials(phase: str, part: str) -> str:
    lo, hi = PHASES[phase]["parts"][part]["band"]
    return f"{'natural' if part == 'nat' else 'induced'}:{lo}:{hi - lo + 1}"


def check_phase_bands(phase: str, g: dict = None) -> list:
    """この段階の帯の表が s4_gates.json と合うか（test2_* は範囲が同じ、関門 2・3 は bundle4_gate2 の中で互いに重ならない）。"""
    g = g or gates()
    al = {a["id"]: tuple(a["range"]) for a in g["bands"]["allocations"]}
    probs, seen = [], []
    for part, v in PHASES[phase]["parts"].items():
        lo, hi = v["band"]
        if v.get("alloc"):
            if al.get(v["alloc"]) != (lo, hi):
                probs.append(f"{phase}.{part}: 帯 {lo}〜{hi} が s4_gates.json の {v['alloc']} {al.get(v['alloc'])} と違う")
        w = PHASES[phase]["alloc_within"]
        if w and not (al[w][0] <= lo and hi <= al[w][1]):
            probs.append(f"{phase}.{part}: 帯 {lo}〜{hi} が {w} {al[w]} の外")
        seen.append((lo, hi, part))
    seen.sort()
    for (a0, a1, pa), (b0, b1, pb) in zip(seen, seen[1:]):
        if b0 <= a1:
            probs.append(f"{phase}: {pa} と {pb} の帯が重なる")
    return probs


def check_band(phase: str, part: str, trials: str, experiment: str, allow_smoke: bool, g: dict = None) -> str:
    """"" なら通す。本番は割り当てと完全に一致する指定（種類も）と、その段階の本番の実験名だけ。smoke は --allow-smoke・
    S4SMOKE で始まる実験名・種 44680〜44683 だけ。"""
    try:
        kind, base, n = trials.split(":")
        base, n = int(base), int(n)
    except ValueError:
        return f"--trials {trials!r}: <natural|induced>:<種の先頭>:<数> の形で指定する"
    want_kind = "natural" if part == "nat" else "induced"
    if kind != want_kind:
        return f"{part} の試行の種類は {want_kind}（渡された {trials}）"
    probs = check_phase_bands(phase, g)
    if probs:
        return "帯の表が s4_gates.json と合わない: " + "; ".join(probs)
    if trials == default_trials(phase, part):
        if experiment.startswith("S4SMOKE") or not experiment.startswith("S4"):
            return f"本番の帯 {trials} は S4 で始まる本番の実験名で回す（S4SMOKE* は smoke 専用）: {experiment}"
        if experiment != PHASES[phase]["experiment"]:
            return f"{phase} の本番の実験名は {PHASES[phase]['experiment']}（渡された {experiment}）"
        return ""
    if not allow_smoke:
        return f"{phase}.{part} の本番の帯は {default_trials(phase, part)} と完全に一致する指定だけ（渡された {trials}）。smoke は --allow-smoke"
    if not experiment.startswith("S4SMOKE"):
        return f"smoke の実験名は S4SMOKE で始める（{experiment}）"
    if n <= 0 or not (SMOKE_SEEDS[0] <= base and base + n - 1 <= SMOKE_SEEDS[1]):
        return f"smoke の種は {SMOKE_SEEDS[0]}〜{SMOKE_SEEDS[1]} だけ（渡された {base}〜{base + n - 1}）"
    return ""


# ---------------------------------------------------------------- 保存点
def ckpt_digest(d: pathlib.Path) -> dict:
    """保存点のフォルダの SHA-256（相対パスの順に「相対パス<TAB>SHA-256」の行をつないだものの SHA-256）とファイルごとの値。"""
    files = sorted(p for p in d.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"保存点 {d} にファイルがない")
    each = {p.relative_to(d).as_posix(): sha256_file(p) for p in files}
    text = "".join(f"{k}\t{v}\n" for k, v in sorted(each.items()))
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "files": each,
            "rule": "相対パスの順に「相対パス<TAB>ファイルの SHA-256<LF>」をつないだ UTF-8 の SHA-256"}


def _postcheck_ok(kind: str, d: dict) -> bool:
    if kind == "b4":
        return bool(d.get("train_config_ok")) and d.get("exit_code") == 0 and bool(d.get("checkpoints_ok"))
    return (bool(d.get("train_config_only_seed_and_names")) and bool(d.get("same_dataset_fingerprint"))
            and d.get("exit_code") == 0 and bool(d.get("checkpoints_ok")))


def candidate_runs(name: str, root: pathlib.Path = ROOT) -> list:
    """[(実行名, 後の点検の JSON のパス, 通ったか)]。実行名は train_<元>s<種>_<日時>_*（smoke は名前に _smoke が付くので入らない）。"""
    spec = MODELS[name]
    wrap = root / "outputs" / "s4" / ("b4_wrap" if spec["kind"] == "b4" else "seed_wrap")
    train = root / "outputs" / "s4" / ("train_b4" if spec["kind"] == "b4" else "train")
    rx = re.compile(rf"^train_{re.escape(spec['base'])}s{spec['seed']}_\d{{8}}-\d{{6}}_")
    out = []
    if not train.is_dir():
        return out
    for d in sorted(p for p in train.iterdir() if p.is_dir() and rx.match(p.name)):
        pc = wrap / f"postcheck_{d.name}.json"
        ok = False
        if pc.is_file():
            try:
                ok = _postcheck_ok(spec["kind"], json.loads(pc.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                ok = False
        out.append((d.name, pc, ok))
    return out


def resolve_ckpt(name: str, v82=None, choose: dict = None, root: pathlib.Path = ROOT) -> dict:
    """{"name", "path"（ROOT からの相対）, "run", "postcheck", "source"}。決められなければ SystemExit。"""
    spec = MODELS.get(name)
    if spec is None:
        raise SystemExit(f"知らないモデル {name}（{sorted(MODELS)}）")
    if spec["kind"] == "s3":
        if v82 is None or name not in v82.CKPT:
            raise SystemExit(f"{name}: 82 の CKPT に保存点がない（段階 3 の学習の出力が見つからない）")
        rel = pathlib.PurePath(v82.CKPT[name]).as_posix()
        if f"/{R1V3_RUN}/" not in f"/{rel}/":
            raise SystemExit(f"{name}: 82 の CKPT が {rel}。事前登録 v2 の草案 第 5 節の実行 {R1V3_RUN} と違う（新しい学習の出力がある？）")
        return {"name": name, "path": rel, "run": R1V3_RUN, "postcheck": None, "source": "82_v2_eval.CKPT（段階 3、2 万手）"}
    runs = candidate_runs(name, root)
    passed = [r for r in runs if r[2]]
    pick = (choose or {}).get(name)
    if pick:
        hit = [r for r in passed if r[0] == pick]
        if not hit:
            raise SystemExit(f"{name}: --run の {pick} は後の点検を通った実行にない（通った実行: {[r[0] for r in passed]}）")
        passed = hit
    if not passed:
        raise SystemExit(f"{name}: 後の点検（--post-check）を通った学習がない（候補 {[r[0] for r in runs]}）")
    if len(passed) > 1:
        raise SystemExit(f"{name}: 後の点検を通った学習が 2 つ以上ある {[r[0] for r in passed]}。--run {name}=<実行名> で選ぶ")
    run, pc, _ = passed[0]
    base = "outputs/s4/train_b4" if spec["kind"] == "b4" else "outputs/s4/train"
    rel = "/".join((base, run) + CKPT_SUB)
    if not (root / rel).is_dir():
        raise SystemExit(f"{name}: 2 万手の保存点 {rel} がない")
    return {"name": name, "path": rel, "run": run, "postcheck": pc.relative_to(root).as_posix(),
            "source": "99_s4_train_b4.py --post-check を通った実行" if spec["kind"] == "b4" else "束 3（99_s4_train_seed.py --post-check を通った実行）"}


def ckpt_info(name: str, v82=None, choose: dict = None, expect: dict = None, root: pathlib.Path = ROOT) -> dict:
    """保存点の対応と SHA-256。expect（掲示した値）と違えば SystemExit。"""
    r = resolve_ckpt(name, v82, choose, root)
    dg = ckpt_digest(root / r["path"])
    r.update(sha256=dg["sha256"], files_sha256=dg["files"], digest_rule=dg["rule"])
    want = (expect or {}).get(name)
    if want and want != r["sha256"]:
        raise SystemExit(f"{name}: 保存点の SHA-256 {r['sha256']} が掲示した値 {want} と違う（起動しない）")
    r["expected_sha256"] = want
    return r


def parse_pairs(items: list, what: str) -> dict:
    out = {}
    for x in items or []:
        if "=" not in x:
            raise SystemExit(f"{what} は 名前=値 の形: {x}")
        k, v = x.split("=", 1)
        if k not in MODELS:
            raise SystemExit(f"{what}: 知らないモデル {k}")
        out[k] = v
    return out


def inject_ckpt(v82, info: dict) -> None:
    """読み込んだ 82 の写しの CKPT に足す（82 のファイルは書き換えない）。96 は config.path(v82.CKPT[--model]) で読む。"""
    have = v82.CKPT.get(info["name"])
    if have is not None and pathlib.PurePath(have).as_posix() != info["path"]:
        raise SystemExit(f"{info['name']}: 82 の CKPT に別の保存点 {have} がある（{info['path']} と違う）")
    v82.CKPT[info["name"]] = info["path"]


# ---------------------------------------------------------------- 環境の照合
def record_envs(dirs: list) -> dict:
    """{条件のフォルダの名前: [完全な試行の env の ENV_KEYS の組（重なりなし）]}。env の無い試行は None。"""
    out = {}
    for d in dirs:
        seen = []
        if d.is_dir():
            for p in sorted(d.glob("trial_[0-9][0-9][0-9][0-9].json")):
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
    """今の環境と段階の全条件の記録の env、決めたドライバの版を照らす。空なら食い違いなし。"""
    out = {}
    if (cur or {}).get("driver") != expected_driver:
        out["driver"] = {"now": (cur or {}).get("driver"), "expected": expected_driver}
    now = tuple((x, (cur or {}).get(x)) for x in ENV_KEYS)
    bad = {}
    for c, envs in rec.items():
        diff = [("env なし" if e is None else dict(e)) for e in envs if e != now]
        if diff:
            bad[c] = diff
    if bad:
        out["records"] = bad
    return out


# ---------------------------------------------------------------- 96 を包む
def patch96(r96, info: dict) -> None:
    """96 の Engine・make_spec・run_json_text を、読み込んだ写しの上で包む（呼ぶたびに元から包み直す）。"""
    orig = getattr(r96, "_b4_orig", None)
    if orig is None:
        orig = {"Engine": r96.Engine, "make_spec": r96.make_spec, "run_json_text": r96.run_json_text}
        r96._b4_orig = orig

    class B4Engine(orig["Engine"]):
        def run_one(self, i, seed, lay, tgt):
            meta, arrays, rlog = super().run_one(i, seed, lay, tgt)
            meta["b4"] = dict(info)                                  # 96（=82）の形に足す欄
            return meta, arrays, rlog

    def make_spec(a):
        s = orig["make_spec"](a)
        s.update(b4_phase=info["phase"], b4_part=info["part"], b4_ckpt_path=info["ckpt"]["path"],
                 b4_ckpt_sha256=info["ckpt"]["sha256"])
        return s

    def run_json_text(a_, kind, rows, *x, **kw):
        d = json.loads(orig["run_json_text"](a_, kind, rows, *x, **kw))
        d["b4"] = dict(info)
        return json.dumps(d, ensure_ascii=False, indent=1)

    r96.Engine, r96.make_spec, r96.run_json_text = B4Engine, make_spec, run_json_text


FORBIDDEN_EXTRA = ("--model", "--mode", "--exec-interval", "--induce", "--ablate", "--time-limit-s", "--reuse-world", "--grip-gate",
                   "--trials", "--experiment", "--condition", "--no-safety", "--no-limiter", "--xcmd-leash", "--cart-margin",
                   "--diag-ik", "--diag-no-gravcomp")


def split_cond(phase: str, cond: str) -> tuple:
    for c, part, m in conditions(phase):
        if c == cond:
            return part, m
    raise SystemExit(f"{phase} の条件は {[c for c, _, _ in conditions(phase)]} のどれか（渡された {cond}）")


def build_96_argv(a, phase: str, part: str, model: str, trials: str, max_new: int = None) -> list:
    argv = ["run", "--experiment", a.experiment, "--condition", f"{part}_{model}", "--model", model, "--trials", trials] + FIXED
    if PART_INDUCE[part]:
        argv += ["--induce", PART_INDUCE[part]]
    if max_new:
        argv += ["--max-new", str(int(max_new))]
    return argv


def run_condition(r96, ops, v82, a, phase: str, cond: str, extra: list, max_new: int = None) -> int:
    for f in extra:
        if f.split("=")[0] in FORBIDDEN_EXTRA:
            raise SystemExit(f"{f} は 98_s4_b4_eval.py が決める（モデル・帯・naive・6 行・安全フィルタなし・60 s・誘発）")
    part, model = split_cond(phase, cond)
    trials = a.trials or default_trials(phase, part)
    why = check_band(phase, part, trials, a.experiment, a.allow_smoke)
    if why:
        raise SystemExit(why)
    ck = ckpt_info(model, v82, parse_pairs(a.run, "--run"), parse_pairs(a.expect, "--expect"))
    inject_ckpt(v82, ck)
    info = {"script": "98_s4_b4_eval.py", "phase": phase, "phase_ja": PHASES[phase]["ja"], "part": part, "model": model,
            "model_ja": MODELS[model]["ja"], "induce": PART_INDUCE[part],
            "ckpt": {k: ck[k] for k in ("path", "run", "postcheck", "source", "sha256", "expected_sha256")},
            "files_sha256": {p: sha256_file(ROOT / p) for p in FILES}}
    patch96(r96, info)
    a96 = r96.build_parser().parse_args(build_96_argv(a, phase, part, model, trials, max_new) + list(extra))
    print(f"[b4] {phase} {cond}: {MODELS[model]['ja']}・{part}（{a.experiment}\\{cond} {trials}、保存点 {ck['path']} "
          f"SHA-256 {ck['sha256'][:12]}…）", flush=True)
    try:
        return r96.cmd_main(a96, v82, ops)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


def _env(a):
    r96 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_b4")
    return r96, r96.load_ops(), r96.load_82(False)


def phase_env_gate(r96, ops, v82, a, phase: str, conds: list) -> int:
    """段階の全条件の記録と今の環境を照らす。食い違いがあり --accept-env-change が無ければ 3。"""
    cur = r96.read_env(ops)
    mm = env_mismatch(cur, record_envs([v82.OUT / a.experiment / c for c in conds]))
    print(f"[b4] 環境: {r96.env_brief(cur)}", flush=True)
    if mm:
        print(f"[b4] 環境の食い違い（段階の全条件・ドライバ {EXPECTED_DRIVER}）: {json.dumps(mm, ensure_ascii=False)}", flush=True)
        if "--accept-env-change" not in a.extra:
            print("[b4] 本番なら止める（dry-run は続けて見せる）。続けるなら --accept-env-change（96 が環境の区切りを書く。報告で分ける）",
                  file=sys.stderr, flush=True)
            return 3
    return 0


def cmd_run(a, extra) -> int:
    r96, ops, v82 = _env(a)
    a.extra = extra
    rc = phase_env_gate(r96, ops, v82, a, a.phase, [a.cond])
    if rc and not a.dry_run:
        return rc
    code = run_condition(r96, ops, v82, a, a.phase, a.cond, extra + (["--dry-run"] if a.dry_run else []), a.max_new)
    return max(code, rc)


def b1mod():
    """rotate の共通部（RotateProgress・rotate_loop・refuse_if_live・read_progress・run_child）は 98_s4_b1.py のものを使う。"""
    return _load(ROOT / "scripts" / "98_s4_b1.py", "s4_b1_for_b4")


def child_argv(a, cond: str, extra: list, block: int) -> list:
    argv = [sys.executable, str(pathlib.Path(__file__).resolve()), "run", "--phase", a.phase, "--cond", cond,
            "--experiment", a.experiment, "--max-new", str(int(block))]
    for x in a.run or []:
        argv += ["--run", x]
    for x in a.expect or []:
        argv += ["--expect", x]
    if a.allow_smoke:
        argv.append("--allow-smoke")
        part = split_cond(a.phase, cond)[0]
        tr = (a.trials_nat if part == "nat" else a.trials) or default_trials(a.phase, part)
        argv += ["--trials", tr]
    return argv + list(extra)


def block_of(part: str, block: int) -> int:
    """塊の試行の数。自然は 1 種 = 3 試行（3 色）なので 3 倍。"""
    return block * (3 if part == "nat" else 1)


def rotate_tag(a) -> str:
    """進み具合のファイルの名前。--parts・--models で分けた rotate（3 本並行）は別の名前にする（同じ組だけを 2 重に回さない）。"""
    tag = a.experiment
    if a.parts:
        tag += "_" + "-".join(sorted(a.parts.split(",")))
    if a.models:
        tag += "_" + "-".join(sorted(a.models.split(",")))
    return tag


def cmd_rotate(a, extra) -> int:
    conds = conditions(a.phase)
    if a.parts:
        keep = set(a.parts.split(","))
        conds = [c for c in conds if c[1] in keep]
    if a.models:
        keep = set(a.models.split(","))
        conds = [c for c in conds if c[2] in keep]
    if not conds:
        raise SystemExit("回す条件がない（--parts・--models）")
    r96, ops, v82 = _env(a)
    a.extra = extra
    trials = {}
    for c, part, m in conds:
        trials[c] = ((a.trials_nat if part == "nat" else a.trials) if a.allow_smoke else None) or default_trials(a.phase, part)
        why = check_band(a.phase, part, trials[c], a.experiment, a.allow_smoke)
        if why:
            raise SystemExit(why)
    for m in sorted({m for _, _, m in conds}):                    # 子を起こす前に、保存点がそろうことを確かめる
        ckpt_info(m, v82, parse_pairs(a.run, "--run"), parse_pairs(a.expect, "--expect"))
    rc = phase_env_gate(r96, ops, v82, a, a.phase, [c for c, _, _ in conds])
    if a.dry_run:
        for c, _, _ in conds:
            sub = argparse.Namespace(**dict(vars(a), trials=trials[c] if a.allow_smoke else None))
            rc = max(rc, run_condition(r96, ops, v82, sub, a.phase, c, extra + ["--dry-run"]))
        return rc
    if rc:
        return rc
    B1 = b1mod()
    for f in extra:
        if f.split("=")[0] in B1.ROTATE_FORBIDDEN + FORBIDDEN_EXTRA:
            raise SystemExit(f"{f} は rotate では渡せない（rotate・98_s4_b4_eval.py が決める）")
    plan = {c: [(seed, tgt) for seed, _, tgt in v82.trial_list(trials[c])] for c, _, _ in conds}
    part_of = {c: p for c, p, _ in conds}
    OUTD.mkdir(parents=True, exist_ok=True)
    tag = rotate_tag(a)
    prog_p, log_p = OUTD / f"rotate_{tag}.progress.json", OUTD / f"rotate_{tag}.log.json"
    B1.refuse_if_live(prog_p, ops)

    def count(c):
        out = v82.OUT / a.experiment / c
        return sum(int(r96.check_complete("run", out, i, {"seed": s, "target": t})[0]) for i, (s, t) in enumerate(plan[c]))

    def call(c):
        code = B1.run_child(child_argv(a, c, extra, block_of(part_of[c], a.block)))
        return code, B1.read_progress(v82.OUT / a.experiment / c / "progress.json")

    def stop_check(c):
        for stop in (v82.OUT / a.experiment / c / "STOP", ROOT / "outputs" / "s4" / "STOP", OUTD / f"rotate_{a.experiment}.STOP",
                     OUTD / f"rotate_{tag}.STOP"):
            if stop.is_file():
                return f"stop_file:{stop}"
        return ""

    names = [c for c, _, _ in conds]
    prog = B1.RotateProgress(prog_p, {"experiment": a.experiment, "condition": f"rotate:{a.experiment}", "phase": a.phase,
                                      "conditions": names, "trials_spec": trials, "total": sum(len(plan[c]) for c in names),
                                      "block_seeds": a.block, "isolation": "subprocess（模型は 1 つずつ）",
                                      "child_progress": {c: str(v82.OUT / a.experiment / c / "progress.json") for c in names},
                                      "stop_file": str(OUTD / f"rotate_{tag}.STOP")}, r96._write_atomic)
    return B1.rotate_loop(names, {c: len(plan[c]) for c in names}, count, call, prog, log_p, stop_check)


def estimate(phase: str) -> dict:
    rows, tot = [], 0.0
    for c, part, m in conditions(phase):
        lo, hi = PHASES[phase]["parts"][part]["band"]
        n = (hi - lo + 1) * (3 if part == "nat" else 1)
        per = 0.023 if (part == "P1" and m.startswith("N")) else PART_PER_TRIAL_H[part]
        rows.append({"cond": c, "trials": n, "process_h": round(n * per, 2)})
        tot += n * per
    return {"rows": rows, "trials": sum(r["trials"] for r in rows), "process_h": round(tot, 1),
            "parallel3_h": round(tot / (3 * 0.95), 1), "note": "推測。1 試行の値は事前登録 v2 の草案 第 5 節（P3 は 0165 の 0.030）"}


def cmd_plan(a) -> int:
    out = {"phase": a.phase, "ja": PHASES[a.phase]["ja"], "experiment": PHASES[a.phase]["experiment"],
           "band_problems": check_phase_bands(a.phase),
           "conditions": [{"cond": c, "part": p, "model": m, "trials": default_trials(a.phase, p), "induce": PART_INDUCE[p]}
                          for c, p, m in conditions(a.phase)],
           "estimate": estimate(a.phase), "fixed": FIXED,
           "commands": [f"scripts\\98_s4_b4_eval.py rotate --phase {a.phase} --dry-run",
                        f"scripts\\98_s4_b4_eval.py rotate --phase {a.phase}",
                        f"scripts\\98_s4_b4_eval.py layout --phase {a.phase} --out outputs\\s4\\b4_eval\\layout_{a.phase}.json",
                        f"scripts\\98_s4_b4_check.py check --layout outputs\\s4\\b4_eval\\layout_{a.phase}.json --params <params.json>"]}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if not out["band_problems"] else 3


def cmd_ckpt(a) -> int:
    v82 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_b4").load_82(False)
    res, bad = {}, {}
    for m in (a.models.split(",") if a.models else MODELS):
        try:
            res[m] = ckpt_info(m, v82, parse_pairs(a.run, "--run"), parse_pairs(a.expect, "--expect"))
        except SystemExit as e:
            bad[m] = str(e.code)
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "ckpt.json").write_text(json.dumps({"models": res, "problems": bad}, ensure_ascii=False, indent=1), encoding="utf-8")
    for m, r in res.items():
        print(f"{m}: {r['path']}  SHA-256 {r['sha256']}")
    for m, why in bad.items():
        print(f"{m}: 決められない — {why}")
    return 0 if not bad else 3


def layout_of(phase: str, experiment: str = None, root: str = "outputs/v2eval") -> dict:
    """二重集計（98_s4_b4_check.py）の入力。条件名は "<部分>.<モデル>" → フォルダ名。"""
    return {"root": root, "phase": phase, "experiment": experiment or PHASES[phase]["experiment"],
            "conds": {f"{p}.{m}": {"cond": c, "model": m} for c, p, m in conditions(phase)}}


def cmd_layout(a) -> int:
    lay = layout_of(a.phase, a.experiment)
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(lay, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"書いた: {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "ckpt", "run", "rotate", "layout"):
        p = sub.add_parser(name)
        if name != "ckpt":
            p.add_argument("--phase", required=True, choices=list(PHASES))
            p.add_argument("--experiment", default=None, help="既定は段階の本番の実験名（S4B4G2・S4B4G3・S4T2）")
        if name in ("ckpt", "run", "rotate"):
            p.add_argument("--run", action="append", default=[], help="保存点の実行を選ぶ（R4=<実行名>。後の点検が通ったものだけ）")
            p.add_argument("--expect", action="append", default=[], help="掲示した保存点の SHA-256（R4=<SHA-256>）。違えば起動しない")
        if name == "ckpt":
            p.add_argument("--models", default=None)
        if name in ("run", "rotate"):
            p.add_argument("--trials", default=None, help="smoke だけ（--allow-smoke と一緒に）。本番は帯の割り当て")
            p.add_argument("--allow-smoke", action="store_true")
            p.add_argument("--dry-run", action="store_true")
        if name == "run":
            p.add_argument("--cond", required=True, help="<部分>_<モデル>（plan の conditions）")
            p.add_argument("--max-new", type=int, default=None)
        if name == "rotate":
            p.add_argument("--block", type=int, default=10, help="交互にする単位の種の数（自然は 1 種 = 3 試行）")
            p.add_argument("--parts", default=None, help="カンマ区切り（既定は全部）")
            p.add_argument("--models", default=None, help="カンマ区切り（既定は全部）")
            p.add_argument("--trials-nat", default=None, help="smoke の自然の試行（--allow-smoke と一緒に。例 natural:44680:1）")
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
    if getattr(a, "phase", None) and a.experiment is None:
        a.experiment = PHASES[a.phase]["experiment"]
    if a.cmd not in ("run", "rotate") and extra:
        print(f"知らない引数: {extra}", file=sys.stderr)
        return 3
    try:
        if a.cmd == "plan":
            return cmd_plan(a)
        if a.cmd == "ckpt":
            return cmd_ckpt(a)
        if a.cmd == "layout":
            return cmd_layout(a)
        return (cmd_run if a.cmd == "run" else cmd_rotate)(a, extra)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
