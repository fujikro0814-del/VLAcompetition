"""段階 4 束 1 の解析の入口の点検（掲示板 0155 の 2 節、査読 docs/stage4/review_cloud.md の重要 1・3）。記録を読むだけ。

使い方（作業場所 C:\\PAI\\recovery_vla。束 1 が終わってから、解析の前に全条件で回す。GPU・シミュレーションは使わない）:
    .venv\\Scripts\\python.exe scripts\\98_s4_d_audit.py [--plan docs\\stage4\\bundle1_defs\\bundle1_plan_queue.json] ^
        [--out outputs\\s4\\audit\\bundle1_audit.json] [--only RTC_naive E7_EH ...] [--no-git]
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
                       なら、それぞれの HEAD で「子が読み込むファイル」（下の CHILD_SCRIPTS と src/recovla・configs の全部）の中身の
                       SHA-256 を git show から計算して比べ、既存のファイルが 1 つも違わなければ同じ版とみなす（0155 の 2-5。後から
                       足されただけのファイルは、既存のファイルが変わらない限り子が読めないので数えない）。--no-git では HEAD が
                       2 つ以上なら欠けとする
  7 double_count       このスクリプトで数えた件数（run: 30・60 s の成功と分母。誘発は t_established < T の試行が分母。
                       task: all_three_in_box と timed_out）と、96_s4_resume.py の score_condition（別の列挙・別の実装）が一致する
  X2（outputs\\s4\\x2）は 96 の条件でないので、gen_summary.json・summary.json があること、種の集合（gen の used_seeds・
  eval の episodes の seed）、env_segments が 1 つ、script_sha256 があることだけを見る（1・4 は対象外）。
  参考（欠けにしない）: resume_log.json の回ごとの git_dirty_n（作業場所に未コミットの変更があったか）。

書くもの: --out の JSON（条件ごとの項目・詳細・呼び直すコマンド）。標準出力に表。
終了コード: 0 全部そろった、1 欠けがある、2 計画が読めない・引数の誤り。
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
# 子（96 の写しを読み込む包み）が読み込むスクリプト。src/recovla と configs は全部を照らす
CHILD_SCRIPTS = ("scripts/96_s4_resume.py", "scripts/96_s4_ops.py", "scripts/82_v2_eval.py", "scripts/41_results.py")
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
    paths = set(CHILD_SCRIPTS) | ({FAMILY_SCRIPT[fam]} if fam in FAMILY_SCRIPT else set())
    tree = _git(root, "ls-tree", "-r", "--name-only", head, "--", "src/recovla", "configs").decode("utf-8").splitlines()
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
def audit_condition(cond: dict, plan: dict, root: pathlib.Path, use_git: bool, r96) -> dict:
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
    rows = []
    for c in conds:
        d = root / pathlib.Path(c["out_dir"].replace("\\", "/"))
        if not c.get("enabled", True) and not d.exists():
            continue                                 # 条件つきの腕（EX）で、回していないもの
        rows.append(audit_condition(c, plan, root, not a.no_git, r96))
    bad = [r for r in rows if not r["ok"]]
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "plan": str(plan_p), "root": str(root),
           "rule": "掲示板 0155 の 2 節。満たさない条件の結果は報告に使わない。欠けた条件は recall のコマンドで 96 を呼び直す",
           "n_conditions": len(rows), "n_bad": len(bad), "ok": not bad, "conditions": rows}
    text = json.dumps(out, ensure_ascii=False, indent=1, default=str)
    op = pathlib.Path(a.out) if a.out else root / OUT
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(text, encoding="utf-8")
    print(table(rows))
    for r in bad:
        why = [f"{c}: {json.dumps(x.get('detail'), ensure_ascii=False, default=str)[:300]}"
               for c, x in r["checks"].items() if not x["ok"]]
        print(f"\n[audit] {r['id']}（{r['dir']}）が欠けている")
        for w in why:
            print(f"  - {w}")
        if r["recall"]:
            print(f"  呼び直す: {r['recall']['cmd']}")
            if r["recall"].get("note"):
                print(f"  注: {r['recall']['note']}")
    print(f"\n[audit] {len(rows) - len(bad)}/{len(rows)} 条件がそろった。書いた: {op}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
