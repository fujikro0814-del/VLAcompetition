"""段階 4 束 1 の解析の入口の点検（掲示板 0155 の 2 節、査読 docs/stage4/review_cloud.md の重要 1・3）。記録を読むだけ。

使い方（作業場所 C:\\PAI\\recovery_vla。束 1 が終わってから、解析の前に全条件で回す。GPU・シミュレーションは使わない）:
    .venv\\Scripts\\python.exe scripts\\98_s4_d_audit.py [--plan docs\\stage4\\bundle1_defs\\bundle1_plan_queue.json] ^
        [--out outputs\\s4\\audit\\bundle1_audit.json] [--only RTC_naive E7_EH ...] [--no-git] ^
        [--ledger-json <97 の結果> | --no-ledger] [--gate1-input outputs\\s4\\gate1_input.json] [--require-gate1]
  計画（bundle1_plan_queue.json の conditions。enabled が偽のもの＝EX は、フォルダがあるときだけ点検する）と、各条件の
  フォルダ（conditions の out_dir）を照らす。1 つでも欠ければ終了コード 1、欠けた条件ごとに「96 を呼び直す」コマンドを出す。
  満たさない条件の結果は報告に使わない（0155 の 2 節）。

確かめる項目（96 の条件。trial_NNNN.json・run_NNNN.json を、ほかの集計と別にこのスクリプトの正規表現で列挙する）:
  1 run_json・g_audit  run.json が読め、G_AUDIT.json の met が真
  2 trials             試行の本数・番号（0〜N−1）・種の集合が計画どおり（natural は種ごとに 3 色で、試行 i の種は 先頭 + i // 3、
                       ほかは 先頭 + i）。読めない記録・数字 4 桁でない名前の記録があれば欠け
  3 diag               全試行の diag（D-RTC の設定と影、D-E7 の腕、単発の開始と先客、移植の腕と回、D-復帰の通りとモデル）と
                       実験・条件・モデルが、フォルダ名（計画の行）と一致する
  4 time_limits        試行の制限時間（time_limits から source を除いたもの）が 1 種類で、計画の値と同じ。run.json も同じ
  5 env_segments       run.json の env_segments が 1 つ、試行の env（ドライバ・torch・CUDA・OS）も 1 種類。2 つ以上なら区切りごとの
                       試行の番号を出す（区切りごとに分けて出す。目標書_段階4.md 11 節 5）
  6 versions           試行の diag の *_sha256（96・診断のモジュール・包み・定義の SHA-256）がそれぞれ 1 種類。git の HEAD が 2 つ以上
                       なら、それぞれの HEAD で「子が読み込むファイル」（下の CHILD_SCRIPTS、src/recovla の全部、CHILD_CONFIGS の
                       設定のファイル）の中身の SHA-256 を git show から計算して比べ、既存のファイルが 1 つも違わなければ同じ版と
                       みなす（0155 の 2-5。後から足されただけのファイルは、既存のファイルが変わらない限り子が読めないので数えない）。
                       configs は子が読み込むものだけ（configs/demo/**（動画の場面）と学習の種の設定 s4_seed_*.yaml は読まない）。
                       --no-git では HEAD が 2 つ以上なら欠けとする
  7 double_count       このスクリプトで数えた件数（run: 30・60 s の成功と分母。誘発は t_established < T の試行が分母。
                       task: all_three_in_box と timed_out）と、96_s4_resume.py の score_condition（別の列挙・別の実装）が一致する
  X2（outputs\\s4\\x2）は 96 の条件でないので、gen_summary.json・summary.json があること、種の集合（gen の used_seeds・
  eval の episodes の seed）、env_segments が 1 つ、script_sha256 があることだけを見る（1・4 は対象外）。
  参考（欠けにしない）: resume_log.json の回ごとの git_dirty_n（作業場所に未コミットの変更があったか）。

条件をまたぐ項目（0155 の 2 節の 6・7。結果の JSON の ledger・gate1_cross_check）:
  6 ledger             種の台帳の照合（scripts/97_s4_ledger_check.py の build_report をこの場で呼ぶ。--ledger-json で既にある
                       結果を渡してもよいが、点検する記録のどれよりも新しいこと）で parse_errors（読めず種も拾えない記録）が 0。
                       台帳のほかの問題の件数は並べるだけ（台帳役が直す）。--no-ledger で飛ばす（結果に skipped と書く）
  7 gate1_cross_check  src/recovla/eval/gate1.py が判定に使う入力（98_s4_gate1.py --make-input の JSON。既定
                       outputs\\s4\\gate1_input.json）のうち、試行の json から数えられる件数を、このスクリプトの列挙と数え方で
                       照らす: 関門 R の natural_success_30・n_trials（設定ごと）、関門 T の all_three_true の k・n（腕ごと）、
                       関門 C の recovery_R・recovery_N の k・n と paired の pairs・r_only・n_only（版と L ごと。対の鍵は種と目標の色）。
                       npz の解析が要る値（R の半径・移動の比・速度の跳び・影・X2、T の first_close_lift、S・XPL の plus_y_shift、
                       K の d0）は、ここでは照らせない（not_compared に並べる）。入力がまだ無ければ status=not_available
                       （終了コードに入れない。--require-gate1 で欠けにする）。標準出力には食い違った項目の名前だけを出し、
                       件数は出さない（結果を見る前の点検のため。件数は JSON にだけ書く）

書くもの: --out の JSON（条件ごとの項目・詳細・呼び直すコマンド、ledger、gate1_cross_check）。標準出力に表。
終了コード: 0 全部そろった、1 欠けがある（条件、台帳の parse_errors、照らした gate1 の食い違い）、2 計画が読めない・引数の誤り。
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLAN = pathlib.Path("docs") / "stage4" / "bundle1_defs" / "bundle1_plan_queue.json"
OUT = pathlib.Path("outputs") / "s4" / "audit" / "bundle1_audit.json"
PY = r".venv\Scripts\python.exe"
ENV_KEYS = ("driver", "torch", "torch_cuda", "os_build")         # 96_s4_resume.ENV_STOP_KEYS と同じ
EPS = 1e-9
_TRIAL = re.compile(r"trial_(\d+)\.json")                         # 96 の score・time_scoring・diag とは別の列挙
_RUN = re.compile(r"run_(\d+)\.json")
# 子（96 の写しを読み込む包み）が読み込むスクリプト。src/recovla は全部を照らす
CHILD_SCRIPTS = ("scripts/96_s4_resume.py", "scripts/96_s4_ops.py", "scripts/82_v2_eval.py", "scripts/41_results.py")
# 子が読み込む設定のファイル（明示の一覧）。default.yaml: recovla.common.config.load（全部の子）。g0.yaml: record・
# eval.closed_loop。sensor_v1・runtime_v2*.yaml: config.load_v2・harness.world。expert_v3.yaml: expert.generate の v3
# （X2 の生成）。latency_v1.json: harness.robot_io。s4_gates.json: 98_s4_d_* の帯の確認。
# configs/demo/**（動画・説明資料の場面）と s4_seed_*.yaml（学習の種）は子が読まないので入れない
CHILD_CONFIGS = ("configs/default.yaml", "configs/g0.yaml", "configs/sensor_v1.yaml", "configs/runtime_v2.yaml",
                 "configs/runtime_v2_color.yaml", "configs/runtime_v2_derived.yaml", "configs/runtime_v2_judge.yaml",
                 "configs/expert_v3.yaml", "configs/latency_v1.json", "configs/s4_gates.json")
LEDGER = pathlib.Path("docs") / "種の台帳.md"
GATE1_INPUT = pathlib.Path("outputs") / "s4" / "gate1_input.json"
FAMILY_SCRIPT = {"RTC": "scripts/98_s4_d_rtc.py", "E7": "scripts/98_s4_d_e7.py", "ST": "scripts/98_s4_d_start.py",
                 "XPL": "scripts/98_s4_d_start.py", "RC": "scripts/98_s4_d_recovery.py", "X2": "scripts/98_s4_x2.py"}
CHECKS = ("run_json", "g_audit", "trials", "diag", "time_limits", "env_segments", "versions", "double_count")


def _read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def family(cond: dict) -> str:
    return cond["id"].split("_", 1)[0]


def _ok(ok: bool, detail=None) -> dict:
    return {"ok": bool(ok), "detail": detail}


def _na(why: str) -> dict:
    return {"ok": True, "na": True, "detail": why}


# ---------------------------------------------------------------- 計画
def expected_trials(cond: dict) -> list:
    """[(試行の番号, 種)]。natural は種ごとに 3 色（41_results.trial_list と同じ並び）。"""
    spec = cond["trials_spec"]
    parts = spec.split(":")
    if cond["kind"] == "task":
        base, n = int(parts[0]), int(parts[1])
        return [(i, base + i) for i in range(n)]
    kind, base, n = parts[0], int(parts[1]), int(parts[2])
    if kind == "natural":
        return [(i, base + i // 3) for i in range(3 * n)]
    return [(i, base + i) for i in range(n)]


def expected_limits(cond: dict) -> dict:
    tl = cond.get("time_limits") or {}
    if cond["kind"] == "task":
        return {k: float(tl[k]) for k in ("step_timeout_s", "retry", "task_time_limit_s")}
    return {"time_limit_s": float(tl["time_limit_s"])}


def recall_command(cond: dict, plan: dict) -> dict:
    """欠けた条件の「96 を呼び直す」コマンド（同じコマンドで続きから回り、試行が全部そろっていれば run.json と G_AUDIT だけを書く）。"""
    fam = family(cond)
    jobs = {j["id"]: j for j in plan.get("jobs", [])}
    job = jobs.get(cond.get("job")) or {}
    note = None
    if fam == "RTC" and cond["id"] != "RTC_shadow":
        cmd = f"{PY} scripts\\98_s4_d_rtc.py run --setting {cond['record_condition']}"
        note = "rotate は完全な本数がそろった設定を回さないので、設定ごとの run で呼ぶ（run は 96 の cmd_main をそのまま呼ぶ）"
    elif fam == "RC":
        cmd = (f"{PY} scripts\\98_s4_d_recovery.py run --experiment {cond['experiment']} --trials {cond['trials_spec']} "
               f"--variants {cond['variant']} --models {cond['model']}")
        note = ("98_s4_d_recovery.py run が、試行が全部そろった条件にも 96 を 1 回呼ぶ版であること（査読の重要 3。依頼 5 の直し）。"
                "前の版では 96 を呼ばずに終わるので、呼んだ後にこの点検をもう一度回して確かめる")
    elif len(job.get("covers") or []) == 1 and job.get("cmd"):
        cmd = job["cmd"]
    else:
        cmd, note = None, f"計画に 1 条件だけの仕事が無い（仕事 {cond.get('job')}）。包みの使い方を見て呼ぶ"
    return {"cmd": cmd, "note": note}


# ---------------------------------------------------------------- 記録の列挙と読み込み（ほかの集計と別に）
def enumerate_records(d: pathlib.Path, kind: str) -> tuple:
    """(番号 → (ファイル名, meta), 読めない・名前の崩れた記録の並び)。"""
    pat = _TRIAL if kind == "run" else _RUN
    recs, bad = {}, []
    for p in sorted(d.iterdir()) if d.is_dir() else []:
        m = pat.fullmatch(p.name)
        if not m or not p.is_file():
            continue
        if len(m.group(1)) != 4:
            bad.append({"file": p.name, "why": "番号が 4 桁でない（96 の記録の名前と違う）"})
            continue
        try:
            meta = _read_json(p)
        except Exception as e:                       # noqa: BLE001
            bad.append({"file": p.name, "why": f"読めない: {type(e).__name__}"})
            continue
        if not isinstance(meta, dict):
            bad.append({"file": p.name, "why": "json の形が違う"})
            continue
        recs[int(m.group(1))] = (p.name, meta)
    return recs, bad


# ---------------------------------------------------------------- 項目
def check_run_json(d: pathlib.Path) -> tuple:
    p = d / "run.json"
    if not p.is_file():
        return _ok(False, "run.json が無い"), None
    try:
        return _ok(True, "run.json あり"), _read_json(p)
    except Exception as e:                           # noqa: BLE001
        return _ok(False, f"run.json が読めない: {type(e).__name__}"), None


def check_g_audit(d: pathlib.Path) -> dict:
    p = d / "G_AUDIT.json"
    if not p.is_file():
        return _ok(False, "G_AUDIT.json が無い")
    try:
        g = _read_json(p)
    except Exception as e:                           # noqa: BLE001
        return _ok(False, f"G_AUDIT.json が読めない: {type(e).__name__}")
    return _ok(g.get("met") is True, {"met": g.get("met"), "trials": g.get("trials")})


def check_trials(cond: dict, recs: dict, bad: list) -> dict:
    exp = expected_trials(cond)
    got_idx = sorted(recs)
    want_idx = [i for i, _ in exp]
    seed_of = dict(exp)
    key = "run" if cond["kind"] == "task" else "trial"
    wrong = []
    for i, (name, m) in sorted(recs.items()):
        if i not in seed_of:
            continue
        if m.get(key) != i or m.get("seed") != seed_of[i]:
            wrong.append({"file": name, key: m.get(key), "seed": m.get("seed"), "want_seed": seed_of[i]})
    tg_bad = []
    if cond["kind"] == "run" and cond["trials_spec"].startswith("natural:"):
        by_seed = {}
        for i, (_, m) in recs.items():
            by_seed.setdefault(m.get("seed"), []).append(m.get("target"))
        tg_bad = [{"seed": s, "targets": t} for s, t in sorted(by_seed.items(), key=lambda x: str(x[0])) if len(set(t)) != len(t)]
    seeds_got = sorted({m.get("seed") for _, m in recs.values()}, key=str)
    seeds_want = sorted(set(seed_of.values()))
    det = {"n": len(recs), "n_plan": int(cond["trials"]), "missing": sorted(set(want_idx) - set(got_idx)),
           "extra": sorted(set(got_idx) - set(want_idx)), "wrong_seed_or_index": wrong, "duplicate_targets": tg_bad,
           "unreadable": bad, "seeds_match_plan": seeds_got == seeds_want,
           "plan_seed_range": cond.get("seeds")}
    ok = (len(recs) == int(cond["trials"]) == len(want_idx) and not det["missing"] and not det["extra"] and not wrong
          and not tg_bad and not bad and det["seeds_match_plan"]
          and (not cond.get("seeds") or (seeds_want[0], seeds_want[-1]) == tuple(cond["seeds"])))
    return _ok(ok, det)


def _diag_expect(cond: dict) -> list:
    """[(説明, 取り出す関数, 期待値)]。全試行に当てる。"""
    fam = family(cond)
    dg = lambda m: m.get("diag") or {}                                          # noqa: E731
    out = []
    if cond["kind"] == "run":
        out += [("experiment", lambda m: m.get("experiment"), cond["experiment"]),
                ("condition", lambda m: m.get("condition"), cond["record_condition"]),
                ("model", lambda m: (m.get("model") or {}).get("name") if isinstance(m.get("model"), dict) else m.get("model"),
                 cond["model"])]
    else:
        out += [("model", lambda m: m.get("model"), cond["model"])]
    if fam == "RTC":
        shadow = cond["record_condition"].startswith("shadow_")
        out += [("diag.diag", lambda m: dg(m).get("diag"), "D-RTC-shadow" if shadow else "D-RTC"),
                ("diag.arm（設定）", lambda m: dg(m).get("arm"), cond["record_condition"].replace("shadow_", "", 1)),
                ("diag.shadow", lambda m: bool(dg(m).get("shadow")), shadow)]
    elif fam == "E7":
        out += [("diag.diag", lambda m: dg(m).get("diag"), "D-E7"), ("diag.arm（腕）", lambda m: dg(m).get("arm"), cond["arm"]),
                ("diag.condition", lambda m: dg(m).get("condition"), cond["record_condition"])]
    elif fam == "ST":
        out += [("diag.diag", lambda m: dg(m).get("diag"), "D-single-start"),
                ("diag.start（開始）", lambda m: dg(m).get("start"), cond["start"]),
                ("diag.prior（先客）", lambda m: dg(m).get("prior"), cond["prior"])]
    elif fam == "XPL":
        out += [("diag.diag", lambda m: dg(m).get("diag"), "XPL"), ("diag.arm（腕）", lambda m: dg(m).get("arm"), cond["arm"]),
                ("diag.rep（回）", lambda m: dg(m).get("rep"), cond["rep"])]
    elif fam == "RC":
        out += [("diag.name", lambda m: dg(m).get("name"), "D_recovery"),
                ("diag.variant（通り）", lambda m: dg(m).get("variant"), cond["variant"]),
                ("diag.model", lambda m: dg(m).get("model"), cond["model"]),
                ("induce.variant", lambda m: (m.get("induce") or {}).get("variant"), cond["variant"]),
                ("induce.kind = diag.induce_kind", lambda m: (m.get("induce") or {}).get("kind") == dg(m).get("induce_kind"), True)]
    return out


def check_diag(cond: dict, recs: dict) -> dict:
    bad = []
    for i, (name, m) in sorted(recs.items()):
        for what, get, want in _diag_expect(cond):
            got = get(m)
            if got != want:
                bad.append({"file": name, "item": what, "got": got, "want": want})
    return _ok(bool(recs) and not bad, {"mismatch": bad[:50], "n_mismatch": len(bad)} if bad else "全試行で一致")


def _lim_key(lim: dict) -> str:
    return json.dumps({k: float(v) for k, v in (lim or {}).items() if k != "source" and v is not None}, sort_keys=True)


def check_time_limits(cond: dict, recs: dict, run_json) -> dict:
    seen = {}
    for i, (_, m) in recs.items():
        lim = m.get("time_limits") or ({"time_limit_s": m.get("time_limit_s")} if cond["kind"] == "run" else {})
        seen.setdefault(_lim_key(lim), []).append(i)
    want = _lim_key(expected_limits(cond))
    rj = None if not run_json else _lim_key(run_json.get("time_limits"))
    ok = list(seen) == [want] and (rj is None or rj == want)
    return _ok(ok, {"seen": {k: len(v) for k, v in seen.items()}, "plan": want, "run_json": rj})


def check_env(recs: dict, run_json) -> dict:
    segs = {}
    for i, (_, m) in sorted(recs.items()):
        e = m.get("env")
        key = json.dumps({k: e.get(k) for k in ENV_KEYS}, sort_keys=True) if isinstance(e, dict) else "null"
        segs.setdefault(key, []).append(i)
    rj = None if not run_json else run_json.get("env_segments")
    n_rj = None if rj is None else len(rj)
    det = {"trial_env_kinds": len(segs), "run_json_env_segments": n_rj,
           "segments": [{"env": json.loads(k), "trials": v} for k, v in segs.items()]}
    ok = len(segs) == 1 and "null" not in segs and (run_json is None or n_rj == 1)
    if len(segs) > 1 or (n_rj or 0) > 1:
        det["note"] = "環境の区切りが 2 つ以上。報告は区切りごとに分けて出す（segments の試行の番号で分ける）"
    return _ok(ok, det)


def _git(root: pathlib.Path, *args) -> bytes:
    """96_s4_ops._git と同じく、作業場所の .tools\\git\\cmd\\git.exe があればそれを、無ければ PATH の git を使う。"""
    exe = root / ".tools" / "git" / "cmd" / "git.exe"
    r = subprocess.run([str(exe) if exe.is_file() else "git", "-C", str(root), *args], capture_output=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return r.stdout


def child_files_sha(root: pathlib.Path, head: str, fam: str) -> dict:
    """HEAD での「子が読み込むファイル」の中身（git の blob）の SHA-256。{パス: SHA-256}。"""
    paths = set(CHILD_SCRIPTS) | set(CHILD_CONFIGS) | ({FAMILY_SCRIPT[fam]} if fam in FAMILY_SCRIPT else set())
    tree = _git(root, "ls-tree", "-r", "--name-only", head, "--", "src/recovla").decode("utf-8").splitlines()
    paths |= {p for p in tree if p.strip()}
    out = {}
    for p in sorted(paths):
        try:
            out[p] = hashlib.sha256(_git(root, "show", f"{head}:{p}")).hexdigest()
        except RuntimeError:
            out[p] = None                                       # その HEAD に無い
    return out


def compare_heads(root: pathlib.Path, heads: list, fam: str) -> dict:
    """HEAD ごとの子が読み込むファイルを比べる。changed: 2 つ以上の HEAD にあって中身が違う・どこかで消えたファイル。
    added_only: 後の HEAD で足されただけのファイル（既存のファイルが変わらない限り子は読めないので、版の違いに数えない）。"""
    shas = {h: child_files_sha(root, h, fam) for h in heads}
    paths = sorted(set().union(*(set(s) for s in shas.values())))
    changed, added = [], []
    for p in paths:
        vals = [shas[h].get(p) for h in heads]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        if len(set(present)) > 1:
            changed.append(p)
        elif len(present) < len(vals):
            first = next(k for k, v in enumerate(vals) if v is not None)
            (added if all(v is not None for v in vals[first:]) and first > 0 else changed).append(p)
    return {"heads": heads, "n_files": len(paths), "changed": changed, "added_only": added}


def check_versions(cond: dict, recs: dict, root: pathlib.Path, use_git: bool) -> dict:
    heads, shas = {}, {}
    for i, (_, m) in sorted(recs.items()):
        h = (m.get("env") or {}).get("git_head")
        heads.setdefault(str(h), []).append(i)
        for k, v in (m.get("diag") or {}).items():
            if k.endswith("sha256"):
                shas.setdefault(k, {}).setdefault(str(v), []).append(i)
    det = {"git_heads": {h: len(v) for h, v in heads.items()},
           "sha256": {k: {v: len(ix) for v, ix in vs.items()} for k, vs in shas.items()}}
    ok = bool(recs) and all(len(vs) == 1 for vs in shas.values())
    if "None" in heads:
        ok = False
        det["note"] = "env.git_head の無い試行がある"
    elif len(heads) > 1:
        if not use_git:
            ok = False
            det["note"] = "HEAD が 2 つ以上（--no-git なので、子が読み込むファイルを照らせない）"
        else:
            # 古い順（最初に現れた試行の番号の順）に並べて比べる
            order = sorted(heads, key=lambda h: min(heads[h]))
            try:
                cmp = compare_heads(root, order, family(cond))
                det["heads_compare"] = cmp
                ok = ok and not cmp["changed"]
                det["note"] = ("HEAD は違うが、子が読み込むファイルの中身は同じ（同じ版とみなす）" if not cmp["changed"] else
                               "HEAD の間で子が読み込むファイルが違う（新旧の版が混ざっている）")
            except Exception as e:                   # noqa: BLE001
                ok = False
                det["note"] = f"git で照らせない: {type(e).__name__}: {e}"
    return _ok(ok, det)


def _git_dirty(d: pathlib.Path) -> list:
    p = d / "resume_log.json"
    try:
        log = _read_json(p)
    except Exception:                                # noqa: BLE001
        return []
    return [{"start": s.get("start"), "git_dirty_n": (s.get("env") or {}).get("git_dirty_n")} for s in log.get("sessions", [])]


def tally(cond: dict, recs: dict, ats=(30.0, 60.0)) -> dict:
    """このスクリプトの数え方（96 の score・time_scoring・diag/recovery と別の実装。0155 の 1-1・1-2 の定義）。"""
    if cond["kind"] == "task":
        return {"n": len(recs), "all_three_in_box": sum(bool(m.get("all_three_in_box")) for _, m in recs.values()),
                "timed_out": sum(bool(m.get("timed_out")) for _, m in recs.values())}
    out = {"n": len(recs), "at": {}}
    for T in ats:
        k = n = 0
        for _, m in recs.values():
            ind = m.get("induce") or {}
            if ind.get("kind"):
                te = ind.get("t_established")
                if not (ind.get("established") and te is not None and float(te) < T - EPS):
                    continue
            n += 1
            ts = m.get("t_success")
            k += bool(m.get("success") and ts is not None and float(ts) <= T + EPS)
        out["at"][f"{T:g}"] = {"successes": k, "n": n}
    return out


def load_96():
    spec = importlib.util.spec_from_file_location("s4_resume_audit", ROOT / "scripts" / "96_s4_resume.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_double(cond: dict, d: pathlib.Path, recs: dict, r96) -> dict:
    mine = tally(cond, recs)
    if r96 is None:
        return _ok(False, {"mine": mine, "note": "96_s4_resume.py を読み込めない"})
    try:
        s = r96.score_condition(d, [30.0, 60.0])
    except Exception as e:                           # noqa: BLE001
        return _ok(False, {"mine": mine, "note": f"96 の score が読めない: {type(e).__name__}: {e}"})
    if cond["kind"] == "task":
        theirs = {"n": s["n"], "all_three_in_box": s.get("all_three_in_box"), "timed_out": s.get("timed_out")}
    else:
        theirs = {"n": s["n"], "at": {k: {"successes": v["successes"], "n": v["n"]} for k, v in s["at"].items()}}
    return _ok(mine == theirs, {"mine": mine, "score_96": theirs})


def _diff_keys(a, b, pre="") -> list:
    """2 つの辞書で値の違う鍵の名前（値は出さない。標準出力に件数を出さないため）。"""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            out += _diff_keys(a.get(k), b.get(k), f"{pre}.{k}" if pre else str(k))
        return out
    return [] if a == b else [pre or "(全体)"]


# ---------------------------------------------------------------- 6 種の台帳の照合（条件をまたぐ）
def load_script(name: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, ROOT / "scripts" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _newest_record_mtime(dirs: list) -> float:
    t = 0.0
    for d in dirs:
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() == ".json":
                    t = max(t, p.stat().st_mtime)
    return t


def check_ledger(root: pathlib.Path, ledger_json, dirs: list) -> dict:
    """0155 の 2-6: 種の台帳の照合で parse_errors が 0。ledger_json が無ければ 97 の build_report をこの場で呼ぶ（書かない）。"""
    if ledger_json:
        p = pathlib.Path(ledger_json)
        try:
            rep = _read_json(p)
        except Exception as e:                       # noqa: BLE001
            return {"ok": False, "source": str(p), "detail": f"台帳の照合の結果が読めない: {type(e).__name__}"}
        if p.stat().st_mtime < _newest_record_mtime(dirs):
            return {"ok": False, "source": str(p), "detail": "台帳の照合の結果が、点検する記録より古い（97 を回し直す）"}
        src = str(p)
    else:
        ledger = root / LEDGER
        if not ledger.is_file():
            return {"ok": False, "source": None, "detail": f"台帳が無い: {ledger}"}
        try:
            m = load_script("97_s4_ledger_check.py", "s4_ledger_for_audit")
            rep, _, _ = m.build_report(root, ledger, [])
        except Exception as e:                       # noqa: BLE001
            return {"ok": False, "source": "97_s4_ledger_check.build_report", "detail": f"照合が回らない: {type(e).__name__}: {e}"}
        src = "97_s4_ledger_check.build_report（この場で）"
    probs = rep.get("problems") or {}
    unread = probs.get("unreadable_records")
    n_parse = len(unread) if isinstance(unread, list) else (rep.get("summary") or {}).get("parse_errors")
    others = {k: len(v) for k, v in probs.items() if k != "unreadable_records" and isinstance(v, list)}
    return {"ok": n_parse == 0, "source": src, "parse_errors": n_parse,
            "parse_error_paths": [e.get("path") for e in (unread or [])][:20],
            "other_problems": others, "ledger_sha256": (rep.get("ledger") or {}).get("sha256"),
            "note": "0155 の 2-6 は parse_errors が 0 であること。ほかの問題の件数は並べるだけ（台帳役が直す）"}


# ---------------------------------------------------------------- 7 gate1.py の入力との照らし合わせ（条件をまたぐ）
def compact(recs: dict) -> list:
    """照らし合わせに要る欄だけの写し（メモリを抑える）。"""
    out = []
    for _, m in recs.values():
        ind = m.get("induce") or {}
        out.append({"seed": m.get("seed"), "target": m.get("target"), "success": m.get("success"), "t_success": m.get("t_success"),
                    "all_three_in_box": m.get("all_three_in_box"),
                    "induce": {"kind": ind.get("kind"), "established": ind.get("established"), "t_established": ind.get("t_established")}})
    return out


def _est_before(m: dict, L: float) -> bool:
    ind = m.get("induce") or {}
    te = ind.get("t_established")
    return bool(ind.get("kind") and ind.get("established") and te is not None and float(te) < L - EPS)


def _succ_by(m: dict, L: float) -> bool:
    ts = m.get("t_success")
    return bool(m.get("success") and ts is not None and float(ts) <= L + EPS)


def gate1_cross_check(gin: dict, plan_conds: list, kept: dict) -> dict:
    """gate1.py の入力の、試行の json から数えられる件数を、このスクリプトの数え方で照らす。kept: {条件の id: compact の並び}。"""
    by_id = {c["id"]: c for c in plan_conds}
    rows, not_compared = [], []

    def add(item, mine, theirs, why=None):
        rows.append({"item": item, "ok": mine is not None and mine == theirs, "mine": mine, "gate1_input": theirs,
                     **({"note": why} if why else {})})

    R = gin.get("R") if isinstance(gin.get("R"), dict) else None
    if R is not None:
        for name, s in (R.get("settings") or {}).items():
            cid = f"RTC_{name}"
            recs = kept.get(cid)
            if recs is None:
                add(f"R.{name}", None, {"natural_success_30": s.get("natural_success_30"), "n_trials": s.get("n_trials")},
                    f"条件 {cid} の記録を点検していない")
                continue
            mine = {"natural_success_30": sum(_succ_by(m, 30.0) for m in recs), "n_trials": len(recs)}
            add(f"R.{name}", mine, {"natural_success_30": s.get("natural_success_30"), "n_trials": s.get("n_trials")})
        not_compared += [f"R.{k}" for k in ("radial_gap_mm", "move_ratio", "seam_jump_mps")]
        not_compared += [k for k in ("shadow_plan_shorter_mm", "x2_shortfall_mm") if k in R]
    T = gin.get("T") if isinstance(gin.get("T"), dict) else None
    if T is not None:
        e7 = {pathlib.PureWindowsPath(c["out_dir"]).name: c["id"] for c in plan_conds if family(c) == "E7"}
        for arm, a in (T.get("arms") or {}).items():
            theirs = (a or {}).get("all_three_true")
            cid = e7.get(arm)
            recs = kept.get(cid) if cid else None
            if recs is None:
                add(f"T.{arm}.all_three_true", None, theirs, f"腕 {arm} の条件（{cid}）の記録を点検していない")
                continue
            add(f"T.{arm}.all_three_true", {"k": sum(bool(m.get("all_three_in_box")) for m in recs), "n": len(recs)},
                {"k": (theirs or {}).get("k"), "n": (theirs or {}).get("n")})
        not_compared.append("T.*.first_close_lift（npz の解析）")
    C = gin.get("C") if isinstance(gin.get("C"), dict) else None
    if C is not None:
        for variant, ent in C.items():
            rc = [c for c in plan_conds if family(c) == "RC" and c.get("variant") == variant]
            rid = next((c["id"] for c in rc if str(c.get("model", "")).startswith("R")), None)
            nid = next((c["id"] for c in rc if str(c.get("model", "")).startswith("N")), None)
            rr, nn = kept.get(rid), kept.get(nid)
            for L, x in ((ent or {}).get("by_L") or {}).items():
                Lf = float(L)
                theirs = {"recovery_R": x.get("recovery_R"), "recovery_N": x.get("recovery_N"), "paired": x.get("paired")}
                if rr is None or nn is None:
                    add(f"C.{variant}.{L}", None, theirs, f"条件 {rid}・{nid} の記録を点検していない")
                    continue
                mine = {}
                for key, recs in (("recovery_R", rr), ("recovery_N", nn)):
                    den = [m for m in recs if _est_before(m, Lf)]
                    mine[key] = {"k": sum(_succ_by(m, Lf) for m in den), "n": len(den)}
                br = {(m["seed"], m.get("target")): m for m in rr}
                bn = {(m["seed"], m.get("target")): m for m in nn}
                keys = [k for k in br if k in bn and _est_before(br[k], Lf) and _est_before(bn[k], Lf)]
                ro = sum(_succ_by(br[k], Lf) and not _succ_by(bn[k], Lf) for k in keys)
                no = sum(_succ_by(bn[k], Lf) and not _succ_by(br[k], Lf) for k in keys)
                mine["paired"] = {"pairs": len(keys), "r_only": ro, "n_only": no}
                theirs = {"recovery_R": {k: (x.get("recovery_R") or {}).get(k) for k in ("k", "n")},
                          "recovery_N": {k: (x.get("recovery_N") or {}).get(k) for k in ("k", "n")},
                          "paired": {k: (x.get("paired") or {}).get(k) for k in ("pairs", "r_only", "n_only")}}
                add(f"C.{variant}.{L}", mine, theirs)
    for key, why in (("S", "plus_y_shift（npz の解析）"), ("XPL", "plus_y_shift（npz の解析）"), ("K", "d0（K の回し直しの記録）")):
        if key in gin:
            not_compared.append(f"{key}: {why}")
    missing = [g for g in ("R", "T", "C") if g not in gin]
    return {"status": "done", "ok": bool(rows) and all(r["ok"] for r in rows) and not missing, "rows": rows,
            "gates_missing_in_input": missing, "not_compared": not_compared,
            "mismatch": [{"item": r["item"], "keys": _diff_keys(r["mine"], r["gate1_input"]) if r["mine"] is not None
                          else [r.get("note")]} for r in rows if not r["ok"]]}


# ---------------------------------------------------------------- X2
def audit_x2(cond: dict, d: pathlib.Path) -> dict:
    gen = cond["id"] == "X2_gen"
    p = d / ("gen_summary.json" if gen else "summary.json")
    res = {c: _na("X2 は 96 の条件でない") for c in ("g_audit", "diag", "time_limits", "double_count")}
    if not p.is_file():
        res.update(run_json=_ok(False, f"{p.name} が無い"), trials=_ok(False, "要約が無い"), env_segments=_ok(False, "要約が無い"),
                   versions=_ok(False, "要約が無い"))
        return res
    try:
        s = _read_json(p)
    except Exception as e:                           # noqa: BLE001
        res.update({c: _ok(False, f"{p.name} が読めない: {type(e).__name__}") for c in ("run_json", "trials", "env_segments", "versions")})
        return res
    base, n = (int(x) for x in cond["trials_spec"].split(":"))
    want = list(range(base, base + n))
    got = sorted(s.get("used_seeds") or []) if gen else sorted(e.get("seed") for e in s.get("episodes") or [])
    res["run_json"] = _ok(True, f"{p.name} あり")
    res["trials"] = _ok(got == want if gen else (set(got) <= set(want) and len(got) == len(set(got)) and len(got) > 0),
                        {"seeds": got, "plan": [base, base + n - 1],
                         "note": None if gen else "測定は生成で保存できたエピソードだけ（失敗した種は無い）"})
    segs = {json.dumps(x.get("env"), sort_keys=True) for x in s.get("env_segments") or []}
    res["env_segments"] = _ok(len(segs) == 1, {"kinds": len(segs), "segments": s.get("env_segments")})
    res["versions"] = _ok(bool(s.get("script_sha256")), {"script_sha256": s.get("script_sha256"),
                                                         "module_sha256": s.get("module_sha256")})
    return res


# ---------------------------------------------------------------- 本体
def audit_condition(cond: dict, plan: dict, root: pathlib.Path, use_git: bool, r96, kept: dict = None) -> dict:
    d = root / pathlib.Path(cond["out_dir"].replace("\\", "/"))
    row = {"id": cond["id"], "dir": str(d), "kind": cond["kind"], "enabled": bool(cond.get("enabled", True))}
    if cond["kind"] == "x2":
        checks = audit_x2(cond, d)
    elif not d.is_dir():
        checks = {c: _ok(False, "条件のフォルダが無い") for c in CHECKS}
    else:
        recs, bad = enumerate_records(d, cond["kind"])
        c_rj, run_json = check_run_json(d)
        checks = {"run_json": c_rj, "g_audit": check_g_audit(d), "trials": check_trials(cond, recs, bad),
                  "diag": check_diag(cond, recs), "time_limits": check_time_limits(cond, recs, run_json),
                  "env_segments": check_env(recs, run_json), "versions": check_versions(cond, recs, root, use_git),
                  "double_count": check_double(cond, d, recs, r96)}
        row["git_dirty_n"] = _git_dirty(d)
        if kept is not None:
            kept[cond["id"]] = compact(recs)
    row["checks"] = checks
    row["ok"] = all(c["ok"] for c in checks.values())
    row["recall"] = None if row["ok"] else recall_command(cond, plan)
    return row


def table(rows: list) -> str:
    head = "| 条件 | " + " | ".join(CHECKS) + " | 判定 |"
    lines = [head, "|" + "---|" * (len(CHECKS) + 2)]
    for r in rows:
        cells = []
        for c in CHECKS:
            x = r["checks"].get(c) or {}
            cells.append("—" if x.get("na") else ("○" if x.get("ok") else "×"))
        lines.append(f"| {r['id']} | " + " | ".join(cells) + f" | {'そろった' if r['ok'] else '欠け'} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default=None, help="計画（既定 docs\\stage4\\bundle1_defs\\bundle1_plan_queue.json）")
    ap.add_argument("--root", default=None, help="作業場所（out_dir の起点と git の場所。既定はこのスクリプトのリポジトリ）")
    ap.add_argument("--only", nargs="+", default=None, help="点検する条件の id（計画の conditions の id）")
    ap.add_argument("--out", default=None, help="結果の JSON（既定 outputs\\s4\\audit\\bundle1_audit.json）")
    ap.add_argument("--no-git", action="store_true", help="git で HEAD の間のファイルを照らさない（HEAD が 2 つ以上なら欠け）")
    ap.add_argument("--ledger-json", default=None, help="既にある 97_s4_ledger_check.py の結果（無ければこの場で照合する）")
    ap.add_argument("--no-ledger", action="store_true", help="種の台帳の照合（0155 の 2-6）を飛ばす（結果に skipped と書く）")
    ap.add_argument("--gate1-input", default=None, help="gate1.py の入力（既定 outputs\\s4\\gate1_input.json。無ければ not_available）")
    ap.add_argument("--require-gate1", action="store_true", help="gate1.py の入力が無いときも欠けにする")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root) if a.root else ROOT
    plan_p = pathlib.Path(a.plan) if a.plan else root / PLAN
    try:
        plan = _read_json(plan_p)
        conds = plan["conditions"]
    except Exception as e:                           # noqa: BLE001
        print(f"計画が読めない: {plan_p}: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if a.only:
        unknown = sorted(set(a.only) - {c["id"] for c in conds})
        if unknown:
            print(f"計画に無い条件: {unknown}", file=sys.stderr)
            return 2
        conds = [c for c in conds if c["id"] in a.only]
    try:
        r96 = load_96()
    except Exception as e:                           # noqa: BLE001
        print(f"96_s4_resume.py を読み込めない（double_count は欠けにする）: {type(e).__name__}: {e}", file=sys.stderr)
        r96 = None
    rows, kept, dirs = [], {}, []
    for c in conds:
        d = root / pathlib.Path(c["out_dir"].replace("\\", "/"))
        if not c.get("enabled", True) and not d.exists():
            continue                                 # 条件つきの腕（EX）で、回していないもの
        rows.append(audit_condition(c, plan, root, not a.no_git, r96, kept))
        dirs.append(d)
    bad = [r for r in rows if not r["ok"]]
    # 6 種の台帳の照合（条件をまたぐ）
    if a.no_ledger:
        ledger = {"ok": None, "skipped": True, "detail": "--no-ledger"}
    else:
        ledger = check_ledger(root, a.ledger_json, dirs)
    # 7 gate1.py の入力との照らし合わせ（条件をまたぐ）
    gp = pathlib.Path(a.gate1_input) if a.gate1_input else root / GATE1_INPUT
    if not gp.is_file():
        g1 = {"status": "not_available", "ok": None, "path": str(gp),
              "detail": "gate1.py の入力がまだ無い（98_s4_gate1.py --make-input で組んでから、この点検を回し直す）"}
    else:
        try:
            raw = gp.read_bytes()
            g1 = gate1_cross_check(json.loads(raw.decode("utf-8-sig")), plan["conditions"], kept)
            g1.update(path=str(gp), sha256=hashlib.sha256(raw).hexdigest())
        except Exception as e:                       # noqa: BLE001
            g1 = {"status": "error", "ok": False, "path": str(gp), "detail": f"{type(e).__name__}: {e}"}
    g1_bad = g1["ok"] is False or (a.require_gate1 and g1["ok"] is not True)
    all_ok = not bad and ledger["ok"] is not False and not g1_bad
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "plan": str(plan_p), "root": str(root),
           "rule": "掲示板 0155 の 2 節。満たさない条件の結果は報告に使わない。欠けた条件は recall のコマンドで 96 を呼び直す",
           "n_conditions": len(rows), "n_bad": len(bad), "ok": all_ok, "conditions_ok": not bad,
           "ledger": ledger, "gate1_cross_check": g1, "conditions": rows}
    text = json.dumps(out, ensure_ascii=False, indent=1, default=str)
    op = pathlib.Path(a.out) if a.out else root / OUT
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(text, encoding="utf-8")
    print(table(rows))
    for r in bad:
        why = []
        for c, x in r["checks"].items():
            if x["ok"]:
                continue
            det = x.get("detail")
            if c == "double_count" and isinstance(det, dict) and "mine" in det and "score_96" in det:
                det = {"食い違う項目": _diff_keys(det["mine"], det["score_96"])}     # 件数は出さない（JSON にだけ）
            why.append(f"{c}: {json.dumps(det, ensure_ascii=False, default=str)[:300]}")
        print(f"\n[audit] {r['id']}（{r['dir']}）が欠けている")
        for w in why:
            print(f"  - {w}")
        if r["recall"]:
            print(f"  呼び直す: {r['recall']['cmd']}")
            if r["recall"].get("note"):
                print(f"  注: {r['recall']['note']}")
    if ledger.get("skipped"):
        print("\n[audit] 種の台帳の照合（0155 の 2-6）: 飛ばした（--no-ledger）")
    else:
        print(f"\n[audit] 種の台帳の照合（0155 の 2-6）: {'parse_errors 0' if ledger['ok'] else '欠け'}"
              f"（{ledger.get('detail') or ledger.get('parse_errors')}。ほかの問題: {ledger.get('other_problems')}）")
    if g1["status"] == "done":
        print(f"[audit] gate1.py の入力との照らし合わせ（0155 の 2-7）: {len(g1['rows'])} 項目、"
              f"{'一致' if g1['ok'] else '食い違い・欠けあり'}。食い違い: {json.dumps(g1['mismatch'], ensure_ascii=False)[:600]}。"
              f"照らせない値: {len(g1['not_compared'])} 件")
    else:
        print(f"[audit] gate1.py の入力との照らし合わせ（0155 の 2-7）: {g1['status']}（{g1.get('detail')}）")
    print(f"\n[audit] {len(rows) - len(bad)}/{len(rows)} 条件がそろった。書いた: {op}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
