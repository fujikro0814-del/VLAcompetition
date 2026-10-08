"""段階 4 束 4（滑りのデータ）: 落下・置き損ねの復帰デモと、同じ配置の通常デモを作る（目標書_段階4.md 8-2 の C・8-4、
configs/s4_gates.json の bundle4_gates・candidate_selection の slip_rule）。

使い方（作業場所 C:\\PAI\\recovery_vla。gen は --dry-run を付けたときだけ生成しない。試すときは必ず --dry-run）:
    .venv\\Scripts\\python.exe scripts\\97_s4_b4_data.py plan                     # 指定（種・色・配置の種類）を outputs\\s4\\b4\\plan.json に書く
    .venv\\Scripts\\python.exe scripts\\97_s4_b4_data.py gen --dry-run            # 何を回し何を飛ばすか、見込みの時間を出す
    .venv\\Scripts\\python.exe scripts\\97_s4_b4_data.py gen --workers 8          # 生成（描画あり。運用役だけ）。止まったら同じコマンドで続きから
    .venv\\Scripts\\python.exe scripts\\97_s4_b4_data.py gen --smoke --workers 2  # smoke（種 44680〜44683、各種類 1 本ぶん）
    .venv\\Scripts\\python.exe scripts\\97_s4_b4_data.py verify                   # 採る本数の確かめ → outputs\\s4\\b4\\data.json
    .venv\\Scripts\\python.exe scripts\\97_s4_b4_data.py gen --top-up             # 足りない枠だけ、予備の種で候補を足して回す（verify が足りないと言ったとき）

中身（段階 3 の R1v3 のデータ＝scripts\\30_f.py gen-data --rig v3 と同じ関数・同じ決まり。違いは本数と誘発の種類だけ）:
  - 足す本数: 落下（種類 B）30 本・置き損ね（種類 C）20 本。段階 3 の B 30・C 20 と合わせて 60・40 になる（final.md 束 4、
    s4_gates の C.continue.action「30/20 本から 60/40 本、R にだけ」）。掴み損ね（A）は足さない。
  - 誘発: recovery.expert.inject の B・C（段階 3 と同じ版。関門 C の診断専用の「手を止める版」＝diag/recovery.py の
    FallHoldInducer は使わない）。生成は recovla.expert.generate.generate（rig v3＝configs/expert_v3.yaml、描画あり）。
  - 枠: 30_f.py の split_counts（配置の種類 empty・prefilled_1・prefilled_2 を 0.75・0.125・0.125 で最大剰余法）、候補は枠ごとに
    必要数の CANDIDATE_FACTOR（2.0）倍、色は配置の机上の色を順に回す（30_f.py の cmd_plan と同じ）。
  - 同じ配置の通常デモ（N 用の相手）: 復帰の候補と同じ（種、色、配置の種類）で種類 n を作る。復帰と相手の両方が（作り直しを
    含めて）成功した候補を、枠ごとに候補の順に必要数だけ採る（30_f.py の cmd_gen_data と同じ。片方が失敗したら両方から外す）。
  - 止める決まり: 種類ごとの捨てた割合が 30_f.py の DROPPED_STOP（10%）を超えたら、verify が止めて諮る（決裁 0042 と同じ）。
  - 同じ仕組みの確かめ（stage3_identity）: 生成の前に、data_v3.json に記録した設定（inject・expert・scene・sim）とコードの
    SHA-256 の束が今と同じで、記録したコミットから生成・誘発・世界・センサ・設定のファイルが変わっていないことを確かめる
    （違えば gen は止まる。--dry-run は結果を出すだけ）。
種の帯（学習用の smoke・データの帯 44400〜44799 の中。回す前に決めた。docs\\stage4\\bundle4_protocol.md 第 2 節）:
  B の候補 44600〜44659（60）、C の候補 44700〜44739（40）、足りないときの予備 B 44660〜44679・C 44740〜44779、
  smoke 44680〜44683。docs\\種の台帳.md の「使用済み」の行、X2 の 44404〜44423、掲示板 0164 の smoke（44500〜44522、
  44500〜44599 を予約とみなす）とは重ならないことを、plan・gen のたびに台帳を読んで確かめる（重なれば止める）。
続きから回す: 枠（種類 × 配置の種類）ごとに 1 回の生成（outputs\\gen\\S4B4_<種類>_<配置の種類>_<日時>）にし、
  outputs\\s4\\b4\\gen_state.json に完了した枠を残す。完了（generation.jsonl の行数が指定の数と同じで timing.json がある）の
  枠は飛ばし、書きかけの枠は _incomplete_<日時> に名前を変えて退避し、同じ指定で回し直す。
読むもの: scripts\\30_f.py（importlib。関数と定数）、docs\\種の台帳.md、configs（config.load）、outputs\\f\\data_v3.json（rig の照合と
  見込みの時間）。書くもの: outputs\\gen\\S4B4_*（生成）、outputs\\s4\\b4\\plan.json・gen_state.json・data.json。
"""
import argparse
import collections
import importlib.util
import json
import pathlib
import re
import sys
import time

from recovla.common import config

ROOT = config.ROOT
CFG = config.load()
OUTPUTS = config.path(CFG["paths"]["outputs"])
B4 = OUTPUTS / "s4" / "b4"
GEN = OUTPUTS / "gen"
LEDGER = ROOT / "docs" / "種の台帳.md"
RIG = "v3"
NEED = {"B": 30, "C": 20}                         # 足す本数（段階 3 の B 30・C 20 と合わせて 60・40）
SEED_BASE = {"B": 44600, "C": 44700}
TOPUP_BASE = {"B": 44660, "C": 44740}
TOPUP_LIMIT = {"B": 44679, "C": 44779}
SMOKE_BASE = {"B": 44680, "C": 44682}
SMOKE_NEED = {"B": 1, "C": 1}
DATA_BAND = (44400, 44799)
X2_BAND = (44404, 44423)
BOARD_0164_RESERVED = (44500, 44599)              # 掲示板 0164 の smoke（44500〜44522）。100 番台ごと予約とみなす


def load_f30():
    spec = importlib.util.spec_from_file_location("s3_f30", ROOT / "scripts" / "30_f.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 帯
def ledger_used(text: str) -> list:
    """台帳の照合用の表の「| 下 | 上 | 状態 |」の行から、使えない範囲を読む。使用済み・予約は全部、予定は帯そのもの
    （44400〜44799 の行）を除いて数える（帯の中に後から足された予定・予約の行も避ける）。"""
    out = []
    for ln in text.splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(使用済み|予約|予定)\s*\|", ln)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if m.group(3) == "予定" and (lo, hi) == DATA_BAND:
                continue
            out.append((lo, hi))
    return out


def band_problems(seeds, ledger_text: str = None) -> list:
    """種が学習用の帯の中で、X2・台帳の使用済み・0164 の予約と重ならないこと。問題の一覧（空なら通す）。"""
    text = LEDGER.read_text(encoding="utf-8") if ledger_text is None else ledger_text
    used = ledger_used(text) + [X2_BAND, BOARD_0164_RESERVED]
    prob = []
    for s in sorted(set(seeds)):
        if not DATA_BAND[0] <= s <= DATA_BAND[1]:
            prob.append(f"種 {s} が学習用の帯 {DATA_BAND} の外")
        for lo, hi in used:
            if lo <= s <= hi:
                prob.append(f"種 {s} が使用済み・予約の範囲 {lo}〜{hi} と重なる")
                break
    return prob


# ---------------------------------------------------------------- 段階 3 と同じ生成の仕組みか
# 生成・誘発・世界・センサ・設定の場所。段階 3 の R1v3 のデータを作ったコミット（data_v3.json の code_version.git_commit）から、
# ここにある既存のファイルが 1 つでも変わっていたら止める（新しく足したファイルは、既存のファイルが読まない限り効かないので数えない）
STAGE3_PATHS = ("src/recovla/expert", "src/recovla/harness", "src/recovla/sim", "src/recovla/record", "src/recovla/common",
                "src/recovla/data", "configs/default.yaml", "configs/g0.yaml", "configs/expert_v3.yaml", "configs/sensor_v1.yaml",
                "configs/runtime_v2.yaml", "configs/runtime_v2_color.yaml", "configs/runtime_v2_derived.yaml",
                "configs/runtime_v2_judge.yaml", "assets")


def stage3_identity(v3: dict = None, git_diff=None, cfg_now: dict = None) -> dict:
    """今の生成の仕組みが、段階 3 の R1v3 のデータ（outputs\\f\\data_v3.json）を作ったときと同じか。
    (1) 記録した設定（inject・expert・scene・sim）が今の値と同じ、(2) 記録したファイルの SHA-256 の束（code_sha256）が同じ、
    (3) 記録したコミットからの git の差分で、上の場所の既存のファイルが変わっていない（作業場所の書きかけも含む）。"""
    from recovla.common import code_version
    if v3 is None:
        p = OUTPUTS / "f" / "data_v3.json"
        if not p.is_file():
            return {"ok": False, "why": "outputs\\f\\data_v3.json がない"}
        v3 = json.loads(p.read_text(encoding="utf-8"))
    if cfg_now is None:
        from recovla.expert import generate as G
        cfg_now = {"inject": CFG["inject"], "expert": (G.rig_config(RIG) or CFG)["expert"], "scene": CFG["scene"], "sim": CFG["sim"]}
    used = v3.get("config_used") or {}
    cfg_same = {k: used.get(k) == v for k, v in cfg_now.items()}
    rec = v3.get("code_version") or {}
    code_same = rec.get("code_sha256") == code_version.code_version()["code_sha256"]
    commit = rec.get("git_commit")
    if git_diff is None:
        def git_diff(c):
            return code_version._git(ROOT, "diff", "--name-only", "--diff-filter=MDRTC", c, "--", *STAGE3_PATHS)
    out = git_diff(commit) if commit else None
    changed = None if out is None else [ln.strip() for ln in out.splitlines() if ln.strip()]
    ok = v3.get("rig") == RIG and all(cfg_same.values()) and code_same and changed == []
    return {"ok": ok, "stage3_commit": commit, "rig": v3.get("rig"), "config_same": cfg_same, "code_sha256_same": code_same,
            "changed_files_since_stage3": changed,
            "why": None if ok else "段階 3 の R1v3 のデータを作ったときと、生成の仕組み（設定・コード）が違う（または git で確かめられない）"}


# ---------------------------------------------------------------- 指定
def build_plan(need: dict, seed_base: dict, f30, sample_layout, factor: float = None) -> dict:
    """30_f.py の cmd_plan と同じ作り方で、種類ごと・配置の種類ごとの候補を並べる。"""
    factor = f30.CANDIDATE_FACTOR if factor is None else factor
    rec = {}
    for kind, n in need.items():
        cells, seed = {}, seed_base[kind]
        for lk, k in f30.split_counts(n).items():
            cand = []
            for i in range(int(round(k * factor))):
                lay = sample_layout(seed, lk)
                cols = lay.table_colors
                cand.append({"seed": seed, "color": cols[i % len(cols)], "layout_kind": lk, "start": lay.start})
                seed += 1
            cells[lk] = {"need": k, "candidates": cand}
        rec[kind] = cells
    return rec


def plan_seeds(rec: dict) -> list:
    return [c["seed"] for cells in rec.values() for cell in cells.values() for c in cell["candidates"]]


def chunks_of(rec: dict) -> list:
    """[(枠の名前, [(種類, 種, 色, 配置の種類)])]。枠の中は 復帰の候補 → 同じ（種、色）の通常 の順（30_f.py と同じ並べ方）。"""
    out = []
    for kind, cells in rec.items():
        for lk, cell in cells.items():
            specs = [(kind, c["seed"], c["color"], lk) for c in cell["candidates"]]
            specs += [("n", c["seed"], c["color"], lk) for c in cell["candidates"]]
            if specs:
                out.append((f"{kind}_{lk}", specs))
    return out


# ---------------------------------------------------------------- 状態（続きから）
def load_state(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"chunks": {}}


def chunk_complete(run_dir: pathlib.Path, n_specs: int) -> bool:
    log = run_dir / "generation.jsonl"
    if not (run_dir / "timing.json").is_file() or not log.is_file():
        return False
    return sum(1 for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()) == n_specs


def run_chunks(chunks: list, state_path: pathlib.Path, gen_root: pathlib.Path, generate, spec_cls, workers: int,
               dry_run: bool, prefix: str = "S4B4") -> dict:
    """枠ごとに生成する。完了した枠は飛ばし、書きかけは退避して回し直す。generate・spec_cls は差し替えられる（テスト）。"""
    st = load_state(state_path)
    todo, skipped = [], []
    for name, specs in chunks:
        rec = st["chunks"].get(name)
        if rec and rec.get("complete") and chunk_complete(ROOT / rec["run"] if not pathlib.Path(rec["run"]).is_absolute()
                                                          else pathlib.Path(rec["run"]), len(specs)):
            skipped.append(name)
        else:
            todo.append((name, specs))
    plan = {"skip": skipped, "run": [n for n, _ in todo], "specs_to_run": sum(len(s) for _, s in todo)}
    if dry_run:
        return plan
    for name, specs in todo:
        old = st["chunks"].get(name)
        if old:
            p = ROOT / old["run"] if not pathlib.Path(old["run"]).is_absolute() else pathlib.Path(old["run"])
            if p.is_dir():
                p.rename(p.with_name(p.name + f"_incomplete_{time.strftime('%Y%m%d-%H%M%S')}"))
        run = gen_root / f"{prefix}_{name}_{time.strftime('%Y%m%d-%H%M%S')}"
        st["chunks"][name] = {"run": _rel(run), "n_specs": len(specs), "complete": False,
                              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        _write(state_path, st)
        res = generate([spec_cls(s, c, lk, k) for k, s, c, lk in specs], run, workers=workers, render=True, rig_kind=RIG)
        st["chunks"][name].update(complete=chunk_complete(run, len(specs)), ended=time.strftime("%Y-%m-%d %H:%M:%S"),
                                  saved=sum(bool(r["success"]) for r in res))
        _write(state_path, st)
    return plan


def _rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def _write(p: pathlib.Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------- 採る（30_f.py cmd_gen_data と同じ規則）
def read_results(state: dict) -> dict:
    """{(種類, 種, 色): generation.jsonl の行}。"""
    by = {}
    for rec in state["chunks"].values():
        p = ROOT / rec["run"] if not pathlib.Path(rec["run"]).is_absolute() else pathlib.Path(rec["run"])
        log = p / "generation.jsonl"
        if not log.is_file():
            continue
        for ln in log.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                r["_run"] = rec["run"]
                by[(r["kind"], int(r["layout_seed"]), r["color"])] = r
    return by


def choose(rec_plan: dict, by: dict, drop_stop: float) -> dict:
    def saved(r):
        return r["attempts"][-1]["name"] if r["success"] else None
    chosen, cells = [], []
    for kind, lk_cells in rec_plan.items():
        for lk, cell in lk_cells.items():
            got, examined, dropped = [], 0, 0
            for c in cell["candidates"]:
                rk, tk = (kind, c["seed"], c["color"]), ("n", c["seed"], c["color"])
                if rk not in by or tk not in by:
                    continue
                if len(got) >= cell["need"]:
                    break
                examined += 1
                if by[rk]["success"] and by[tk]["success"]:
                    got.append({"kind": kind, "layout_kind": lk, "seed": c["seed"], "color": c["color"],
                                "recovery": {"run": by[rk]["_run"], "key": saved(by[rk])},
                                "twin": {"run": by[tk]["_run"], "key": saved(by[tk])}, "start": c["start"]})
                else:
                    dropped += 1
            chosen += got
            cells.append({"kind": kind, "layout_kind": lk, "need": cell["need"], "got": len(got), "examined": examined,
                          "dropped": dropped, "short": cell["need"] - len(got)})
    by_kind = {}
    for k in rec_plan:
        rows = [c for c in cells if c["kind"] == k]
        ex, dr = sum(r["examined"] for r in rows), sum(r["dropped"] for r in rows)
        by_kind[k] = {"need": sum(r["need"] for r in rows), "got": sum(r["got"] for r in rows), "examined": ex, "dropped": dr,
                      "drop_rate": dr / ex if ex else None}
    stop = any(v["drop_rate"] is not None and v["drop_rate"] > drop_stop for v in by_kind.values())
    short = [c for c in cells if c["short"] > 0]
    return {"chosen": chosen, "cells": cells, "by_kind": by_kind, "stop_drop_rule": stop, "short_cells": short,
            "count_ok": not short and all(v["got"] == v["need"] for v in by_kind.values())}


# ---------------------------------------------------------------- 本体
def cmd_plan(a) -> int:
    from recovla.sim import scene
    f30 = load_f30()
    need, base = (SMOKE_NEED, SMOKE_BASE) if a.smoke else (NEED, SEED_BASE)
    rec = build_plan(need, base, f30, scene.sample_layout)
    prob = band_problems(plan_seeds(rec))
    if prob:
        print("止める（帯）: " + " / ".join(prob[:5]), file=sys.stderr)
        return 3
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": a.smoke, "rig": RIG, "need": need, "seed_base": base,
           "candidate_factor": f30.CANDIDATE_FACTOR, "kind_share": f30.RECOVERY_KIND_SHARE, "recovery": rec,
           "seeds": [min(plan_seeds(rec)), max(plan_seeds(rec))], "n_specs": 2 * len(plan_seeds(rec)),
           "rule": "30_f.py の cmd_plan・cmd_gen_data と同じ（split_counts・候補の倍数・色の回し方・両方成功した候補を順に採る）"}
    _write(B4 / ("plan_smoke.json" if a.smoke else "plan.json"), out)
    print(json.dumps({k: out[k] for k in ("need", "seeds", "n_specs")}, ensure_ascii=False))
    return 0


def _plan(a) -> dict:
    p = B4 / ("plan_smoke.json" if a.smoke else "plan.json")
    if not p.is_file():
        raise SystemExit(f"{p} がない（先に plan）")
    return json.loads(p.read_text(encoding="utf-8"))


def topup_plan(plan: dict, state: dict, f30, sample_layout) -> dict:
    """足りない枠だけ、予備の種で候補を足した指定（足りない本数の CANDIDATE_FACTOR 倍）。"""
    res = choose(plan["recovery"], read_results(state), f30.DROPPED_STOP)
    need = collections.defaultdict(dict)
    for c in res["short_cells"]:
        need[c["kind"]][c["layout_kind"]] = c["short"]
    rec = {}
    for kind, cells in need.items():
        seed, rec[kind] = TOPUP_BASE[kind], {}
        used = {c["seed"] for cl in plan["recovery"][kind].values() for c in cl["candidates"]}
        while seed in used:
            seed += 1
        for lk, k in cells.items():
            cand = []
            for i in range(int(round(k * f30.CANDIDATE_FACTOR))):
                lay = sample_layout(seed, lk)
                cols = lay.table_colors
                cand.append({"seed": seed, "color": cols[i % len(cols)], "layout_kind": lk, "start": lay.start})
                seed += 1
            if seed - 1 > TOPUP_LIMIT[kind]:
                raise SystemExit(f"{kind} の予備の種が足りない（{TOPUP_BASE[kind]}〜{TOPUP_LIMIT[kind]}）。手順書の帯を足して諮る")
            rec[kind][lk] = {"need": k, "candidates": cand}
    return rec


def cmd_gen(a) -> int:
    from recovla.expert import generate as G
    from recovla.sim import scene
    plan = _plan(a)
    state_path = B4 / ("gen_state_smoke.json" if a.smoke else "gen_state.json")
    rec = plan["recovery"]
    if a.top_up:
        st = load_state(state_path)
        add = topup_plan(plan, st, load_f30(), scene.sample_layout)
        if not add:
            print("足りない枠はない（top-up は要らない）")
            return 0
        for kind, cells in add.items():                   # 指定に足す（同じ枠の候補の後ろへ）
            for lk, cell in cells.items():
                rec[kind][lk]["candidates"] += cell["candidates"]
                rec[kind][lk].setdefault("topup", []).extend(c["seed"] for c in cell["candidates"])
        plan["recovery"] = rec
        if not a.dry_run:
            _write(B4 / ("plan_smoke.json" if a.smoke else "plan.json"), plan)
        chunks = [(f"{n}_topup", s) for n, s in chunks_of(add)]
    else:
        chunks = chunks_of(rec)
    prob = band_problems([s for _, specs in chunks for _, s, _, _ in specs])
    if prob:
        print("止める（帯）: " + " / ".join(prob[:5]), file=sys.stderr)
        return 3
    ident = stage3_identity()
    if not ident["ok"] and not a.dry_run:
        print("止める（段階 3 と同じ生成の仕組みでない）: " + json.dumps(ident, ensure_ascii=False), file=sys.stderr)
        return 3
    res = run_chunks(chunks, state_path, GEN, G.generate, G.EpisodeSpec, a.workers, a.dry_run,
                     prefix="S4SMOKE_B4" if a.smoke else "S4B4")
    est = estimate_hours(res["specs_to_run"], a.workers)
    print(json.dumps(dict(res, estimate=est, stage3_identity=ident, dry_run=a.dry_run), ensure_ascii=False, indent=1))
    return 0 if ident["ok"] else 3


def estimate_hours(n_specs: int, workers: int) -> dict:
    """段階 3 の R1v3 のデータの生成の実時間（outputs\\f\\data_v3.json の generation_wall_s と指定の数）から比で見込む（推測）。"""
    p = OUTPUTS / "f" / "data_v3.json"
    if not p.is_file():
        return {"hours": None, "note": "outputs\\f\\data_v3.json がないので見込めない（作者の PC で出る）"}
    d = json.loads(p.read_text(encoding="utf-8"))
    n_ref = None
    wall = float(d.get("generation_wall_s") or 0.0)
    plan_ref = OUTPUTS / "f" / "plan.json"
    if plan_ref.is_file():
        pr = json.loads(plan_ref.read_text(encoding="utf-8"))
        n_ref = len(pr["normal"]["specs"]) + 2 * sum(len(c["candidates"]) for cells in pr["recovery"].values() for c in cells.values())
    if not n_ref or not wall:
        return {"hours": None, "note": "段階 3 の指定の数か実時間が読めない"}
    h = wall / 3600.0 * n_specs / n_ref * (int(d.get("workers") or workers) / max(1, workers))
    return {"hours": round(h, 2), "ref_specs": n_ref, "ref_wall_h": round(wall / 3600.0, 2), "ref_workers": d.get("workers"),
            "note": "比での見込み（推測）。復帰の候補は作り直しが多いと延びる"}


def cmd_verify(a) -> int:
    f30 = load_f30()
    plan = _plan(a)
    st = load_state(B4 / ("gen_state_smoke.json" if a.smoke else "gen_state.json"))
    res = choose(plan["recovery"], read_results(st), f30.DROPPED_STOP)
    missing = []
    for c in res["chosen"]:
        for side in ("recovery", "twin"):
            p = config.path(c[side]["run"]) / c[side]["key"]
            if not ((p / "meta.json").is_file() and (p / "data.npz").is_file()):
                missing.append(str(p))
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": a.smoke, "need": plan["need"], **res,
           "stage3_identity": stage3_identity(),
           "missing_files": missing, "ok": res["count_ok"] and not res["stop_drop_rule"] and not missing,
           "drop_stop_rule": f"種類ごとの捨てた割合が {f30.DROPPED_STOP:.0%} を超えたら止めて諮る（30_f.py・決裁 0042）"}
    _write(B4 / ("data_smoke.json" if a.smoke else "data.json"), out)
    print(json.dumps({k: out[k] for k in ("by_kind", "short_cells", "stop_drop_rule", "ok")}, ensure_ascii=False, indent=1))
    if not out["ok"]:
        print("本数がそろわない・捨てた割合が多い・ファイルが欠けている。足りない枠は gen --top-up", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "gen", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--smoke", action="store_true")
        if name == "gen":
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--top-up", action="store_true")
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "gen": cmd_gen, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
