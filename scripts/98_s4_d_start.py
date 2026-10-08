"""段階 4 束 1 の D-単発の開始と「移植」の腕（担当 B。docs/目標書_段階4.md 8-2 の S・8-3・8-4・8-6）。

使い方（作業場所 C:\\PAI\\recovery_vla。start・xpl の引数は 96_s4_resume.py run と同じものを後ろに付ける）:
    .venv\\Scripts\\python.exe scripts\\98_s4_d_start.py defs           # 移植の状態・壁際・格子の定義を outputs\\s4\\d_start\\definitions.json に書く（回す前に掲示）
    .venv\\Scripts\\python.exe scripts\\98_s4_d_start.py start --start standby_end --prior wall_side \\
        --experiment S4DSTART --condition send_wall --model R1v3 --trials natural:190400:11
    .venv\\Scripts\\python.exe scripts\\98_s4_d_start.py xpl --arm as --rep 1 --experiment S4XPL --condition as_r1 --model R1v3 \\
        [--states 0:20] [--seed-base 190420]
    .venv\\Scripts\\python.exe scripts\\98_s4_d_start.py summary --experiment S4DSTART|S4XPL [--out ...]
  start: --start home|standby_start|standby_end、--prior on_grid|wall_side（6 条件。同じ種 190400〜190410 x 3 色 = 33 試行/条件）。
  xpl: --arm as|home|grid（XPL_as・XPL_home・XPL_grid）、--rep 1|2（同じ 20 状態を 2 回）。種は --seed-base + 状態の番号（本番は
    190420 + i が E7_R1v3 の run_{i:04d}）。--trials は自動（"xpl:<種の先頭>:<本数>"。続きから回すときの照合に使う）。
  本番は運用役が回す。smoke は学習用の帯（44400〜44799）だけ。どちらの帯にも入らない種は回さない（終了コード 3）。
  --mode naive --exec-interval 6 --no-safety（物差し K と同じ、最終評価の A と同じ）をこのスクリプトが既定で入れる。制限時間は 96 の既定 60 s。
読むもの: scripts\\96_s4_resume.py（importlib で読み込み、Engine・plan_trials・SPEC_KEYS・run_json_text を差し替えて cmd_main で回す。書き換えない）、
  configs\\s4_gates.json（帯）、outputs\\v2eval\\V3S3\\E7_R1v3 の run_0000〜0019 の json・npz・_runtime.json（移植の状態と壁際。読むだけ。
  目標書 8-3・第 10 節 7 が決める例外）、outputs\\s4\\d_start\\definitions.json（start・xpl は必ずこれを読み、無ければ止める）。
書くもの: 96_s4_resume.py run と同じ（outputs\\v2eval\\<実験>\\<条件>\\ の trial_NNNN.json・.npz・runtime_NNNN.json、run.json、G_AUDIT.json、
  progress.json、resume_spec.json、resume_log.json）。試行の json に "diag"（条件・置き直しの中身・置き直した後の世界の状態・定義の SHA-256）、
  run.json に "diag" を足す。layout.start は placed_home・placed_standby_start・standby_end・xpl（どれも待機位置の始めの標準の開始から
  同じ置き直しの道で置く＝src\\recovla\\diag\\start.py の START_RULE。10/08 に、standby_end だけが置き直しを通っていたのを 3 通りとも
  同じ道にそろえた）。defs・summary は outputs\\s4\\d_start\\。

結果を見る前に決めてある項目（src\\recovla\\diag\\start.py の冒頭と definitions.json。掲示の後は変えない）:
  開始の 3 通り・先客の 2 通りの置き方、壁際の並びと当て方、格子の置き場所の決め方、移植の「2 番目の始め」の添字、XPL_grid の空きの決め方、
  置き直しの落ち着かせ方。指標は単発の最初の閉じ（試行全体の最初の gripper_closed の 0→1）の +y のずれ（主）と持ち上がり（副）、
  分母は最初の閉じが起きた試行。summary は関門 S・移植の材料を並べるが判定はしない。
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTD = ROOT / "outputs" / "s4" / "d_start"
DEFS = OUTD / "definitions.json"
GATES = ROOT / "configs" / "s4_gates.json"
E7_DIR = ROOT / "outputs" / "v2eval" / "V3S3" / "E7_R1v3"
START_SHORT = {"home": "home", "standby_start": "sstart", "standby_end": "send"}
PRIOR_SHORT = {"on_grid": "grid", "wall_side": "wall"}


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


def band_check(seeds, alloc_id: str) -> str:
    g = json.loads(GATES.read_text(encoding="utf-8"))
    al = next(x for x in g["bands"]["allocations"] if x["id"] == alloc_id)["range"]
    sm = next(x for x in g["bands"]["classes"] if x["id"] == "smoke_data")["range"]
    seeds = list(seeds)
    if all(al[0] <= s <= al[1] for s in seeds) or all(sm[0] <= s <= sm[1] for s in seeds):
        return ""
    return f"種 {seeds[0]}〜{seeds[-1]} が帯 {alloc_id} {al} にも smoke の帯 {sm} にも収まらない"


def read_defs() -> dict:
    if not DEFS.is_file():
        raise SystemExit(f"{DEFS} が無い。先に defs を回して掲示する")
    return json.loads(DEFS.read_text(encoding="utf-8"))


def make_engine(r96, spec_fn, info: dict):
    from recovla.diag import start as S
    from recovla.harness.sensors import SensorSuite

    class DiagRunEngine(r96.Engine):
        """96 の Engine の run を、世界だけ DiagWorldRig に替え、試行ごとに置き直しを渡す版（実行系は 96 の写しのまま）。"""
        _pending = None

        def new_world(self):                                 # 96 の Engine.new_world と同じ（WorldRig を DiagWorldRig に替えただけ）
            if self.suite is not None:
                self.suite.close()
            gc = False if getattr(self.a, "diag_no_gravcomp", False) else None
            self.world = S.DiagWorldRig(render=False, cfg=self.cfg, gravcomp=gc)
            self.suite = SensorSuite(self.world.model, self.cfg)

        def _before_trial(self):
            super()._before_trial()
            self.world.diag_override = self._pending

        def run_one(self, i, seed, lay, tgt):
            lay2, ov, rec = spec_fn(i, seed, lay, tgt)
            self._pending = ov
            meta, arrays, rlog = super().run_one(i, seed, lay2, tgt)
            meta["diag"] = dict(rec, applied=self.world.diag_applied, **info)
            return meta, arrays, rlog
    return DiagRunEngine


def _run(r96, a, spec_fn, info: dict, diag_keys: dict, plan=None) -> int:
    for k, v in diag_keys.items():
        setattr(a, k, v)
    r96.SPEC_KEYS = tuple(r96.SPEC_KEYS) + tuple(diag_keys)
    r96.Engine = make_engine(r96, spec_fn, info)
    if plan is not None:
        r96.plan_trials = lambda a_, v82: plan
    orig_text = r96.run_json_text

    def run_json_text(a_, kind, rows, *x, **kw):
        d = json.loads(orig_text(a_, kind, rows, *x, **kw))
        d["diag"] = dict(diag_keys, **info)
        return json.dumps(d, ensure_ascii=False, indent=1)
    r96.run_json_text = run_json_text
    ops = r96.load_ops()
    v82 = r96.load_82(a.allow_82_change)
    try:
        return r96.cmd_main(a, v82, ops)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


def _fixed_settings(a, rest) -> str:
    """K・最終評価の A と同じ実行のしかた（naive・6 行・安全フィルタなし）に揃える。違う値が渡されたら理由を返す。"""
    if a.mode != "naive" or a.exec_interval not in (None, 6) or a.induce or a.ablate:
        return "D-単発の開始・移植は --mode naive --exec-interval 6 --no-safety（誘発・ablate なし）で回す"
    a.exec_interval, a.no_safety = 6, True
    return ""


def cmd_start(argv) -> int:
    from recovla.diag import start as S
    ap = argparse.ArgumentParser(prog="98_s4_d_start.py start", add_help=False)
    ap.add_argument("--start", required=True, choices=sorted(S.START_KEYS))
    ap.add_argument("--prior", required=True, choices=S.PRIORS)
    mine, rest = ap.parse_known_args(argv)
    r96 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_dstart")
    a = r96.build_parser().parse_args(["run"] + rest)
    why = _fixed_settings(a, rest)
    kind, base, n = a.trials.split(":")
    if kind != "natural":
        why = why or "--trials は natural:<種の先頭>:<種の数>（82 の natural と同じ並び。種ごとに 3 色）"
    why = why or band_check(range(int(base), int(base) + int(n)), "D_single_start")
    if why:
        print(why, file=sys.stderr)
        return 3
    defs = read_defs()
    pool = defs["wall_pool"]
    band_base = int(base)

    def spec_fn(i, seed, lay, tgt):
        return S.start_trial_spec(seed, lay, tgt, mine.start, mine.prior, pool, band_base)
    info = {"script": "98_s4_d_start.py start", "definitions_sha256": sha256_file(DEFS),
            "r96_sha256": sha256_file(ROOT / "scripts" / "96_s4_resume.py"),
            "start_py_sha256": sha256_file(ROOT / "src" / "recovla" / "diag" / "start.py")}
    print(f"[d_start] 開始 {mine.start}・先客 {mine.prior}（壁際の並び {len(pool)} 個、種の先頭 {band_base}）", flush=True)
    return _run(r96, a, spec_fn, info, {"diag_start": mine.start, "diag_prior": mine.prior,
                                        "diag_defs_sha256": info["definitions_sha256"]})


def cmd_xpl(argv) -> int:
    from recovla.diag import start as S
    ap = argparse.ArgumentParser(prog="98_s4_d_start.py xpl", add_help=False)
    ap.add_argument("--arm", required=True, choices=S.XPL_ARMS)
    ap.add_argument("--rep", required=True, type=int, choices=(1, 2))
    ap.add_argument("--states", default="0:20", help="<最初の状態の番号>:<個数>（既定 0:20）")
    ap.add_argument("--seed-base", type=int, default=190420, help="状態 i の種 = この値 + i（本番 190420。smoke は学習用の帯）")
    mine, rest = ap.parse_known_args(argv)
    s0, ns = (int(x) for x in mine.states.split(":"))
    rest = [x for x in rest]
    if "--trials" in rest:
        print("xpl の --trials は自動で決める（渡さない）", file=sys.stderr)
        return 3
    rest += ["--trials", f"xpl:{mine.seed_base + s0}:{ns}"]
    r96 = _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume_dxpl")
    a = r96.build_parser().parse_args(["run"] + rest)
    why = _fixed_settings(a, rest) or band_check(range(mine.seed_base + s0, mine.seed_base + s0 + ns), "D_transplant")
    if why:
        print(why, file=sys.stderr)
        return 3
    defs = read_defs()
    states = defs["xpl_states"][s0:s0 + ns]
    if len(states) != ns:
        print(f"状態が {len(states)} 個しかない（--states {mine.states}）", file=sys.stderr)
        return 3
    plan = [(k, mine.seed_base + s0 + k, None, st["target"]) for k, st in enumerate(states)]

    def spec_fn(i, seed, lay, tgt):
        return S.xpl_trial_spec(seed, states[i], mine.arm, mine.rep)
    info = {"script": "98_s4_d_start.py xpl", "definitions_sha256": sha256_file(DEFS),
            "r96_sha256": sha256_file(ROOT / "scripts" / "96_s4_resume.py"),
            "start_py_sha256": sha256_file(ROOT / "src" / "recovla" / "diag" / "start.py"),
            "seed_base": mine.seed_base, "state_first": s0}
    print(f"[d_start] 移植 XPL_{mine.arm}・{mine.rep} 回目（状態 {s0}〜{s0 + ns - 1}、種 {plan[0][1]}〜{plan[-1][1]}）", flush=True)
    return _run(r96, a, spec_fn, info, {"diag_xpl_arm": mine.arm, "diag_xpl_rep": mine.rep, "diag_xpl_states": mine.states,
                                        "diag_seed_base": mine.seed_base, "diag_defs_sha256": info["definitions_sha256"]},
                plan=plan)


def cmd_defs(a) -> int:
    """移植の 20 状態・壁際の並び・決まりの文言を書く（E7_R1v3 の記録を読むだけ）。"""
    from recovla.diag import e7 as E7
    from recovla.diag import start as S
    states = S.extract_xpl_states(E7_DIR, 20)
    pool = S.wall_pool(states)
    for st in states:                                          # XPL_grid の行き先も先に決めて載せる
        ri = 0
        others = [st["cube_pos"][k] for k in range(3) if k != ri]
        slot, dist = S.nearest_empty_slot(st["cube_pos"][ri], others)
        st["xpl_grid"] = {"slot": slot, "move_m": round(dist, 4)}
    src = {}
    for i in range(20):
        for k, p in S._e7_paths(E7_DIR, i).items():
            src[str(p.relative_to(ROOT))] = sha256_file(p)
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "by": "担当 B（98_s4_d_start.py defs）",
           "rules": {"xpl_index": S.XPL_INDEX_RULE, "wall": S.WALL_RULE, "grid": S.GRID_RULE, "prior_of": S.PRIOR_OF,
                     "start_keys": S.START_KEYS, "grid_occupied_m": S.GRID_OCCUPIED_M, "wall_thresh_m": S.WALL_THRESH_M,
                     "start_rule": S.START_RULE, "base_start": S.BASE_START, "start_q": {k: list(v) for k, v in S.START_Q.items()},
                     "settle": "置き直した後、腕を置き直した関節角のまま位置のサーボで保ち（1/3・2/3 の時点で測った関節角の差だけ指令を直す）、"
                               "ハンドは開いたまま、configs の scene.settle_prefilled_s（1.0 s）落ち着かせて時刻 0。開始の 3 通り・先客の 2 通りの"
                               "6 条件と移植の腕のすべてがこの道を通る",
                     "standby_end_q": E7.STANDBY_END_Q, "standby_start_q": E7.STANDBY_START_Q, "home_q": E7.HOME_Q,
                     "pose_source": E7.POSE_SOURCE,
                     "xpl_seed": "種 190420 + i が run_{i:04d}（目標書 8-3）。目標は steps[1] の色（20 本とも green）",
                     "metrics": "単発の最初の閉じ（試行全体で最初の gripper_closed の 0→1）。+y のずれ＝その時の（指先 − 目標）の y > +1 cm、"
                                "持ち上がり＝次に開くまでに目標が CUBE_REST_Z + 0.02 m を超えた。分母は最初の閉じが起きた試行"},
           "xpl_states": states, "wall_pool": pool,
           "source_note": "outputs\\v2eval\\V3S3\\E7_R1v3（段階 3 の最終評価の E7。読むだけ。目標書 8-3・第 10 節 7 が決める例外）",
           "source_sha256": src,
           "bands": {"D_single_start": "190400〜190410（natural:190400:11、6 条件で同じ種）", "D_transplant": "190420〜190439（--seed-base 190420）"},
           "files_sha256": {p: sha256_file(ROOT / p) for p in ("src/recovla/diag/start.py", "src/recovla/diag/e7.py",
                                                                "scripts/98_s4_d_start.py", "scripts/96_s4_resume.py")}}
    OUTD.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(a.out) if a.out else DEFS
    if p.is_file() and not a.force:
        old = json.loads(p.read_text(encoding="utf-8"))
        same = old.get("xpl_states") == json.loads(json.dumps(states)) and old.get("wall_pool") == json.loads(json.dumps(pool))
        print(f"[d_start] {p} は既にある（状態・壁際は {'同じ' if same else '違う'}）。書き直すなら --force（掲示の後は書き直さない）")
        return 0 if same else 3
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[d_start] 状態 {len(states)} 個（目標 {sorted({s['target'] for s in states})}）、壁際の並び {len(pool)} 個: "
          f"{[w['from_index'] for w in pool]}、XPL_grid の行き先 {[s['xpl_grid']['slot'] for s in states]}")
    print(f"[d_start] 書いた: {p}（SHA-256 {sha256_file(p)}）")
    return 0


def summary_dirs(base: pathlib.Path) -> tuple:
    """summary に使うフォルダを、全試行の diag から見分ける（mujoco を読まない。査読の重要 1）。
    返り値 (starts {"<start>|<prior>": フォルダ}, xpl {腕: {回: フォルダ}}, 見分けられないフォルダ名, 問題の文の並び)。
    同じ鍵のフォルダが 2 つ（前の版では名前順で後のほうが黙って上書きした）、1 つのフォルダの中で diag の鍵が混ざっている、
    は問題として返す（cmd_summary は止める）。"""
    starts, xpl, unknown, probs, seen = {}, {}, [], [], {}
    for d in sorted(x for x in base.iterdir() if x.is_dir()):
        tj = sorted(d.glob("trial_[0-9][0-9][0-9][0-9].json"))
        if not tj:
            continue
        keys = set()
        for p in tj:
            dg = json.loads(p.read_text(encoding="utf-8")).get("diag") or {}
            if dg.get("diag") == "D-single-start":
                keys.add(("start", f"{dg.get('start')}|{dg.get('prior')}"))
            elif dg.get("diag") == "XPL":
                keys.add(("xpl", f"{str(dg.get('arm')).replace('XPL_', '')}|{dg.get('rep')}"))
            else:
                keys.add(("unknown", str(dg.get("diag"))))
        if len(keys) != 1:
            probs.append(f"{d.name}: 試行の diag の鍵が {len(keys)} 種類ある {sorted(keys)}（1 つのフォルダに別の条件が混ざっている）")
            continue
        key = next(iter(keys))
        if key[0] == "unknown":
            unknown.append(d.name)
            continue
        if key in seen:
            probs.append(f"同じ鍵 {key[1]}（{key[0]}）のフォルダが 2 つある: {seen[key]} と {d.name}（どちらを使うか決めてから回す）")
            continue
        seen[key] = d.name
        if key[0] == "start":
            starts[key[1]] = d
        else:
            arm, rep = key[1].split("|")
            xpl.setdefault(arm, {})[int(rep)] = d
    return starts, xpl, unknown, probs


def cmd_summary(a) -> int:
    from recovla.diag import start as S
    base = ROOT / "outputs" / "v2eval" / a.experiment
    sd, xd, unknown, probs = summary_dirs(base)
    if probs:
        for p in probs:
            print(f"[d_start] {p}", file=sys.stderr)
        print("[d_start] summary を書かずに止めた（同じ鍵のフォルダ・混ざったフォルダ）", file=sys.stderr)
        return 3
    starts = {k: S.summarize_single(d) for k, d in sd.items()}
    xpl = {arm: {rep: S.summarize_single(d) for rep, d in reps.items()} for arm, reps in xd.items()}
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": a.experiment, "skipped_dirs": unknown,
           "dirs": {k: d.name for k, d in sd.items()} | {f"XPL_{arm}_r{rep}": d.name for arm, reps in xd.items()
                                                          for rep, d in reps.items()}}
    if starts:
        out["gate_S_inputs"] = S.gate_S_inputs(starts)
    if xpl:
        out["xpl_inputs"] = S.xpl_inputs(xpl)
    if a.with_trials:
        out["rows"] = {k: v["rows"] for k, v in starts.items()} | {f"XPL_{k}_r{r}": v["rows"] for k, reps in xpl.items()
                                                                   for r, v in reps.items()}
    text = json.dumps(out, ensure_ascii=False, indent=1, default=str)
    OUTD.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(a.out) if a.out else OUTD / f"summary_{a.experiment}.json"
    p.write_text(text, encoding="utf-8")
    print(text[:6000])
    print(f"[d_start] 書いた: {p}")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "start":
        return cmd_start(argv[1:])
    if argv and argv[0] == "xpl":
        return cmd_xpl(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start", help="D-単発の開始の 1 条件を回す（--start・--prior と 96_s4_resume.py run の引数）")
    sub.add_parser("xpl", help="移植の腕を 1 つ回す（--arm・--rep と 96_s4_resume.py run の引数）")
    p = sub.add_parser("defs", help="移植の状態・壁際・格子の定義を書く（回す前に掲示）")
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("summary", help="記録から関門 S・移植の材料を出す（判定はしない）")
    p.add_argument("--experiment", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--with-trials", action="store_true")
    a = ap.parse_args(argv)
    return {"defs": cmd_defs, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
