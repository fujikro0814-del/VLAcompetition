"""段階 4 束 1 の X2（担当 A）: R1v3 を、学習に使っていないエキスパートの記録に開ループで当て、10・20・30・40 行先の予測の不足を測る。

使い方（作業場所 C:\\PAI\\recovery_vla。gen・eval とも同じコマンドをもう一度打てば続きから回る）:
    .venv\\Scripts\\python.exe scripts\\98_s4_x2.py gen  [--seeds 44404:20] [--name X2_gen] [--dry-run]   # エキスパートの記録の生成（数分）
    .venv\\Scripts\\python.exe scripts\\98_s4_x2.py eval [--name X2_gen] [--model R1v3] [--stride 1] [--dry-run]   # 開ループの推論と不足（GPU で数分）
    .venv\\Scripts\\python.exe scripts\\98_s4_x2.py summary [--name X2_gen] [--model R1v3]                 # 書いた結果を読むだけ
  smoke（担当 A の小帯 44430〜44449 だけ）: gen --seeds 44440:1 --name SMOKE_A_x2 → eval --name SMOKE_A_x2
  gen・eval の共通の引数（96_s4_resume.py と同じ名前・同じ意味）: --dry-run（何を飛ばし何を回すか・環境の照合を見るだけ。何も書かない）、
    --accept-env-change、--max-new N（新しい種・エピソードを N 本回したら止まる。終了コード 1、stop_reason "max_new:N"）。
  監視: outputs\\s4\\x2\\gen\\<名前>\\progress.json・outputs\\s4\\x2\\eval_<名前>_<モデル>\\progress.json（96_s4_resume.py と同じ鍵:
    status・pid・proc_create_time・updated・done・total・stop_reason・error・env など。96_s4_ops.py wait --progress がそのまま読む）。

環境の照合（96_s4_resume.py read_env と同じ。目標書_段階4.md 5-1・11 節 5）: 回の始めに nvidia-smi のドライバ、torch・CUDA、OS
  （CurrentBuildNumber.UBR）、git の HEAD を読み、ドライバ・torch・CUDA・OS（96 の ENV_STOP_KEYS）が記録と違えば止める（終了コード 3）。
  照らす記録: gen は、この名前の前の回（gen_log.json の sessions）。eval は、生成の要約（gen_summary.json の env。生成と測定を
  別の環境で混ぜない）と、この出力の前の回（eval_log.json の sessions。記録の無いまま残っている途中のエピソードの npz は
  「環境の分からない記録」として食い違いに数える）。ドライバの版が読めなければ回さない。--accept-env-change を付けたときだけ進め、
  その回の session に "env_segment" を書く（報告で環境ごとに分ける）。

途中で切れたとき（再起動・Ctrl+C・エラー）の退避と回し直し（96_s4_resume.py と同じ考え方。同じコマンドをもう一度打つ）:
  gen: 種ごとに別の小フォルダ outputs\\s4\\x2\\gen\\<名前>\\part_<種>\\ に generate を 1 回ずつ呼ぶ（generate は新しいフォルダにしか
    書かないので、種ごとに分けて「完全な種」を飛ばせるようにした。世界は種ごとに作り直す＝96 の既定と同じく、試行の順と再開の有無に
    よらない）。part_<種> は timing.json があり、generation.jsonl がその種の 1 行で、成功ならエピソードの meta.json と data.npz（zip が
    壊れていない）がそろっていれば「完全」（失敗の種も完全。作り直しの上限まで試した結果）。完全でない part_<種>、名前のフォルダの直下の
    ほかのもの（この直しの前の形で途中で切れた run.json・generation.jsonl・エピソードのフォルダなど）、*.tmp は
    <名前>\\_incomplete_<時刻>\\ へ退避して、同じ種で回し直す（完全な part には触れない）。全部の種が完全になったら gen_summary.json を
    書く（これがあれば「済み」で何もしない）。
  eval: エピソードごとに <エピソード>.npz を一時ファイルから置き換えて書く。npz が読めて k_index・pred_post・f_close がそろえば
    「完全」とみなして推論を飛ばし（不足は npz と生成の記録から計算し直す。GPU なし）、*.tmp は _incomplete_<時刻>\\ へ退避する。
    全部そろったら summary.json を書く（あれば「済み」。--force で前の出力を _superseded_<時刻>\\ へ移して最初から）。
  止める合図: outputs\\s4\\STOP か、出力のフォルダの STOP を見たら、今の種・エピソードを終えてから止まる（終了コード 1）。

生成の手段（既存の生成の道を読んで決めた）: R1v3 の学習データ（outputs/train/train_R1v3_*/conversion.json の sources は全部
  outputs/gen/F_data_v3_20261005-133737）は scripts/30_f.py gen-data --rig v3 で作った。その本体は src/recovla/expert/generate.py の
  generate(specs, run_dir, rig_kind="v3")（harness/gen_v2.py の SensedDrivenRig に configs/expert_v3.yaml を重ねた世界。エキスパートは
  真値で動き、指令は実行系と同じ口、画像・状態はセンサの模型を通したもの）。ここでも同じ関数を同じ引数（rig_kind="v3"、描画あり、
  作り直しは configs の expert.slip_retry_max まで）で、種ごとに呼ぶ。新しい生成のコードは書かない（指定の組み立てと記録の要約だけ）。
  指定: 種ごとに 1 本。配置は scene.sample_layout(種, "empty")（自然の評価と同じ空の箱。開始の姿勢は配置の乱数列から＝学習データと同じ）、
  色は机上の色の (種 mod 色の数) 番目（30_f.py の check_specs と同じ選び方）、種類は通常（kind "n"）。
  学習に使っていない: 種 44404〜44423 は学習用の帯の smoke・データの区画（目標書 4-2 の X2_gen）で、学習データの種は 20000〜32xxx。
変換（eval）: 学習の変換（src/recovla/data/convert.py の convert）と同じ関数で観測を作る。10 fps のこま k = 生のこま 2k、状態は
  episode_arrays の state、画像は vla_observation.observation_images（= vla_image_spec.policy_images）、手がかりは episode_tracker
  （エピソードに記録した信じている較正。閾値は保存点の conversion.json の target_cue と照合）を k の順に全部のこまで更新し、旗の
  扱いは保存点に合わせる（R1v3 は xy＝旗なし 17 次元）。雑音は実行系と同じ noise_generator(配置の種, k)。RTC は切る。
  推論するのは k + 10 <= f1 // 2（f1 はエキスパートが最初に閉じた生のこま）の k だけ（--stride ごと）。

読むもの: outputs\\s4\\x2\\gen\\<名前>\\（gen が書いたエピソード）、保存点（82_v2_eval.py の CKPT。96_s4_resume.load_82 でハッシュ照合）、configs。
書くもの: gen: outputs\\s4\\x2\\gen\\<名前>\\part_<種>\\（generate の run.json・generation.jsonl・timing.json・エピソードのフォルダ）、
    gen_log.json（回ごとの環境・回した種・退避）、progress.json、gen_summary.json（種・指定・成否・作り直し・各エピソードの meta.json・
    data.npz の SHA-256・環境・スクリプトの SHA-256）。
  eval: outputs\\s4\\x2\\eval_<名前>_<モデル>\\<エピソード>.npz（k_index、pred_post (n, 50, 7)、f_close）、eval_log.json、progress.json、
    summary.json（h ごとの不足、件数、環境、保存点、生成の要約の SHA-256）。summary は読むだけ。
結果を見る前に決めてある項目: 帯 44404〜44423（20 本。smoke は 44430〜44449）、指定の作り方（上）、変換と雑音（上）、
  不足の定義（src\\recovla\\diag\\rtc.py の冒頭の X2: (e_h − p_h)·u、u は x_des → 目標の立方体、|·| >= 30 mm、k + h <= f1 // 2、
  h = 10・20・30・40、集計は全こまの中央値、どれかの h で 15 mm 以上なら「不足が 15 mm 以上」= R.b1 の X2 の側）、--stride 1。
終了コード: 96_s4_resume.py と同じ（0 済んだ、1 途中で止まった（合図・--max-new・Ctrl+C）、2 エラー、
  3 前提の食い違い（帯・メモリ・環境・gen の要約が無い・二重に回す））。
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recovla.diag import rtc as D            # noqa: E402

X2 = ROOT / "outputs" / "s4" / "x2"
GATES = ROOT / "configs" / "s4_gates.json"
GLOBAL_STOP = ROOT / "outputs" / "s4" / "STOP"
SMOKE_BAND = (44400, 44799)
MIN_FREE_GB = 12.0
EXIT = {"done": 0, "stopped": 1, "interrupted": 1, "error": 2}       # 96_s4_resume.EXIT と同じ（memory_timeout は使わない）


def _load(path: pathlib.Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load96():
    return _load(ROOT / "scripts" / "96_s4_resume.py", "s4_resume")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write(p: pathlib.Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


def _rel(p: pathlib.Path) -> str:
    try:
        return str(pathlib.Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def x2_band() -> tuple:
    g = json.loads(GATES.read_text(encoding="utf-8"))
    for a in g["bands"]["allocations"]:
        if a["id"] == "X2_gen":
            return tuple(a["range"])
    raise SystemExit(f"{GATES}: allocations に X2_gen がない")


def parse_seeds(s: str) -> list:
    base, n = (int(x) for x in s.split(":"))
    return list(range(base, base + n))


# ---------------------------------------------------------------- 環境・メモリ・二重起動（96_s4_resume.py と同じ規則）
def env_conflicts(m96, cur: dict, recorded: list, sessions: list) -> dict:
    """recorded: [(ラベル, env or None)]。env の無い記録は「環境の分からない記録」として数える（96 の unknown_records と同じ）。
    前に --accept-env-change で今の環境へ切り替えた回があれば、それより前の記録は分けてあるので数えない（96 と同じ）。
    返り値は {ラベル: 差 or "unknown"}（差は {鍵: (記録, 今)}）。空なら食い違いなし。"""
    keys = m96.ENV_STOP_KEYS
    cur_key = json.dumps({k: cur.get(k) for k in keys}, sort_keys=True)
    accepted = any(s.get("env_segment") and s.get("env")
                   and json.dumps({k: s["env"].get(k) for k in keys}, sort_keys=True) == cur_key for s in sessions)
    out = {}
    for label, e in recorded:
        if accepted and not label.startswith("gen_summary"):
            continue
        if not e:
            out[label] = "unknown"
            continue
        d = m96._env_diff(e, cur)
        if d:
            out[label] = d
    prev = [s for s in sessions if s.get("env")]
    if prev:
        d = m96._env_diff(prev[-1]["env"], cur)
        if d:
            out["previous_session"] = d
    return out


def check_env(m96, cur: dict, conf: dict, accept: bool, where: str) -> None:
    if not cur.get("driver"):
        raise SystemExit("nvidia-smi からドライバの版を読めない。環境の記録のない実行は報告に使えないので回さない")
    if conf and not accept:
        raise SystemExit(f"{where}: 環境が記録と違う: {conf}\n（(記録, 今)。unknown は環境の記録がない途中の記録）。黙って混ぜない。"
                         f"続けるなら --accept-env-change（この回を env_segment として記録に書き、報告で分ける）")


def check_memory(ops) -> dict:
    mem = ops.memory_gb()
    if mem["phys_free_gb"] < MIN_FREE_GB:
        raise SystemExit(f"空きの物理メモリ {mem['phys_free_gb']} GB < {MIN_FREE_GB} GB（運用の決まり）。回さない")
    return mem


def check_not_running(ops, prog_path: pathlib.Path) -> None:
    """同じ出力を 2 つのプロセスで回さない（96 と同じ。前の progress.json の pid が生きていて、状態が最終でなければ止める）。"""
    if not prog_path.is_file():
        return
    try:
        old = json.loads(prog_path.read_text(encoding="utf-8"))
    except Exception:                            # noqa: BLE001
        return
    if old.get("status") in ops.LIVE_STATUS and old.get("pid") and old.get("pid") != os.getpid() \
            and ops._alive(int(old["pid"]), old.get("proc_create_time")):
        raise SystemExit(f"{prog_path}: pid {old['pid']} がまだ回している（status={old.get('status')}）。二重に回さない")


def read_log(p: pathlib.Path) -> dict:
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:                        # noqa: BLE001
            pass
    return {"sessions": []}


def stop_requested(out_dir: pathlib.Path) -> str:
    for p in (out_dir / "STOP", GLOBAL_STOP):
        if p.exists():
            return f"stop_file:{_rel(p)}"
    return ""


def quarantine(out_dir: pathlib.Path, names: list, stamp: str) -> list:
    moved = []
    if not names:
        return moved
    d = out_dir / f"_incomplete_{stamp}"
    d.mkdir(exist_ok=True)
    for n in names:
        src = out_dir / n
        if src.exists():
            shutil.move(str(src), str(d / n))
            moved.append(n)
    return moved


class _Stop(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------- gen
def specs_for(seeds: list) -> list:
    from recovla.expert import generate as G
    from recovla.sim import scene
    out = []
    for seed in seeds:
        lay = scene.sample_layout(seed, "empty")
        color = lay.table_colors[seed % len(lay.table_colors)]
        out.append(G.EpisodeSpec(int(seed), color, "empty", "n"))
    return out


def _npz_ok(p: pathlib.Path, need=()) -> bool:
    try:
        if not p.is_file() or p.stat().st_size == 0 or not zipfile.is_zipfile(p):
            return False
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:
                return False
            names = {n[:-4] if n.endswith(".npy") else n for n in z.namelist()}
        return bool(names) and all(k in names for k in need)
    except Exception:                            # noqa: BLE001
        return False


def part_state(run_dir: pathlib.Path, seed: int) -> tuple:
    """(完全か, 理由, generation.jsonl の 1 行 or None)。part_<種> の完全さ（冒頭の「途中で切れたとき」）。"""
    d = run_dir / f"part_{seed}"
    if not d.is_dir():
        return False, "missing", None
    if not (d / "timing.json").is_file():
        return False, "no_timing", None
    try:
        lines = [json.loads(ln) for ln in (d / "generation.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception as e:                       # noqa: BLE001
        return False, f"jsonl:{type(e).__name__}", None
    if len(lines) != 1 or lines[0].get("layout_seed") != seed:
        return False, "jsonl:mismatch", None
    r = lines[0]
    if r.get("success"):
        ep = d / pathlib.Path(str(r["attempts"][-1].get("path") or r["attempts"][-1]["name"])).name
        if not (ep / "meta.json").is_file() or not _npz_ok(ep / "data.npz"):
            return False, "episode_files", None
    return True, "ok", r


def episode_row(r: dict, part: pathlib.Path) -> dict:
    last = r["attempts"][-1]
    row = {"seed": r["layout_seed"], "color": r["color"], "success": r["success"], "retries": r["retries"],
           "failures": [t.get("failure") for t in r["attempts"]], "duration_s": last.get("duration_s"),
           "start_pose": (last.get("layout") or {}).get("start"), "name": last.get("name"), "part": _rel(part)}
    if r["success"]:
        p = part / pathlib.Path(str(last.get("path") or last["name"])).name
        row.update(path=_rel(p), meta_sha256=sha256_file(p / "meta.json"), data_sha256=sha256_file(p / "data.npz"))
    return row


GEN_KEEP = {"progress.json", "gen_log.json", "STOP"}


def cmd_gen(a) -> int:
    seeds = parse_seeds(a.seeds)
    lo, hi = x2_band()
    if not (all(lo <= s <= hi for s in seeds) or all(SMOKE_BAND[0] <= s <= SMOKE_BAND[1] for s in seeds)):
        raise SystemExit(f"種 {seeds[0]}〜{seeds[-1]} は X2 の帯 {lo}〜{hi} にも学習用の帯 {SMOKE_BAND} にもない")
    run_dir = X2 / "gen" / a.name
    specs = specs_for(seeds)
    print(f"[x2] 生成 {len(specs)} 本 → {_rel(run_dir)}", flush=True)
    if (run_dir / "gen_summary.json").is_file():
        print(f"[x2] {_rel(run_dir)}\\gen_summary.json がある（済み）。何もしない", flush=True)
        if a.dry_run:
            print(f"[dry-run] {_rel(run_dir)}: 済み（gen_summary.json あり）。回す 0 本")
        return 0
    states = {s.layout_seed: part_state(run_dir, s.layout_seed) for s in specs}
    todo = [s for s in specs if not states[s.layout_seed][0]]
    n_done = len(specs) - len(todo)
    stray = sorted(p.name for p in run_dir.iterdir()) if run_dir.is_dir() else []
    good_parts = {f"part_{sd}" for sd, st in states.items() if st[0]}
    stray = [n for n in stray if n not in good_parts and n not in GEN_KEEP and not n.startswith("_incomplete_")]
    m96 = load96()
    ops = m96.load_ops()
    log_path = run_dir / "gen_log.json"
    log = read_log(log_path)
    # 完全な part の環境: part の generation.jsonl には環境が無いので、その part を書いた回（gen_log の session）の環境で照らす
    by_seed_env = {}
    for s_ in log.get("sessions", []):
        for sd in s_.get("ran_seeds", []):
            by_seed_env[sd] = s_.get("env")
    recorded = [(f"part_{sd}", by_seed_env.get(sd)) for sd, st in states.items() if st[0]]
    env = m96.read_env(ops)
    conf = env_conflicts(m96, env, recorded, log.get("sessions", []))
    print(f"[x2] 環境: {m96.env_brief(env)}", flush=True)
    if a.dry_run:
        print(f"[dry-run] {_rel(run_dir)}: 全 {len(specs)} 本、完全 {n_done} 本（飛ばす）、回す {len(todo)} 本")
        for s in specs:
            ok, why, _ = states[s.layout_seed]
            print(f"  種 {s.layout_seed} {s.color:6s} {'完全' if ok else '回す(' + why + ')'}")
        if stray:
            print(f"[dry-run] 退避するもの（_incomplete_<時刻>\\ へ）: {stray}")
        if conf:
            print(f"[dry-run] 環境の食い違い: {conf}{'（--accept-env-change で分けて進める）' if a.accept_env_change else ''}")
        try:
            print(f"[dry-run] 空きメモリ {ops.memory_gb()['phys_free_gb']} GB（{MIN_FREE_GB} 未満なら回さない）")
        except Exception:                        # noqa: BLE001
            pass
        return 3 if (not env.get("driver") or (conf and not a.accept_env_change)) else 0
    check_env(m96, env, conf, a.accept_env_change, _rel(run_dir))
    mem = check_memory(ops)
    run_dir.mkdir(parents=True, exist_ok=True)
    prog_path = run_dir / "progress.json"
    check_not_running(ops, prog_path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    moved = quarantine(run_dir, stray, stamp)
    if moved:
        print(f"[x2] 途中で切れた書きかけを退避: {_rel(run_dir)}\\_incomplete_{stamp}\\ {moved}", flush=True)
    sess = {"start": _now_s(), "pid": os.getpid(), "skipped_complete": n_done, "to_run": [s.layout_seed for s in todo],
            "ran_seeds": [], "quarantined": moved, "end": None, "status": "running", "env": env, "memory_at_start": mem}
    if conf:
        sess["env_segment"] = {"accepted": "--accept-env-change", "conflicts": conf,
                               "note": "この回で生成した種は、前の環境の種と別の環境として分けて報告する"}
    log.setdefault("sessions", []).append(sess)
    _write(log_path, log)
    n_succ = sum(1 for st in states.values() if st[0] and st[2]["success"])
    base = {"experiment": "outputs\\s4\\x2", "condition": f"gen\\{a.name}", "cmd": "gen", "model": None, "trials_spec": a.seeds,
            "pid": os.getpid(), "host": os.environ.get("COMPUTERNAME"), "started": _now_s(), "status": "starting",
            "total": len(specs), "done": n_done, "skipped_complete": n_done, "ran_this_session": 0, "successes": n_succ,
            "last_trial_wall_s": None, "stop_reason": None, "error": None, "quiet_window": None, "min_free_gb": MIN_FREE_GB,
            "time_limits": None, "env": env, "env_conflicts": conf or None}
    prog = m96.Progress(prog_path, ops, base)
    prog.start_heartbeat()
    from recovla.expert import generate as G
    status, ran = "done", 0
    try:
        prog.update(status="running", todo=[s.layout_seed for s in todo])
        for s in todo:
            if a.max_new and ran >= a.max_new:
                raise _Stop(f"max_new:{a.max_new}")
            r = stop_requested(run_dir)
            if r:
                raise _Stop(r)
            prog.update(current_seed=s.layout_seed, current_started=_now_s())
            t0 = time.perf_counter()
            res = G.generate([s], run_dir / f"part_{s.layout_seed}", workers=1, render=True, rig_kind="v3")
            wall = round(time.perf_counter() - t0, 1)
            ran += 1
            n_succ += int(bool(res[0]["success"]))
            sess["ran_seeds"].append(s.layout_seed)
            _write(log_path, log)
            prog.update(done=n_done + ran, ran_this_session=ran, last_trial_wall_s=wall, successes=n_succ)
            print(f"[x2] 種 {s.layout_seed} {s.color}: 成功 {res[0]['success']}、作り直し {res[0]['retries']}、{wall} s", flush=True)
    except _Stop as e:
        status = "stopped"
        prog.update(stop_reason=e.reason)
        print(f"[x2] 止める合図: {e.reason}", flush=True)
    except KeyboardInterrupt:
        status = "interrupted"
        prog.update(stop_reason="KeyboardInterrupt")
    except BaseException as e:                   # noqa: BLE001
        status = "error"
        prog.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
        traceback.print_exc()
    # 全部の種が完全なら gen_summary.json
    states = {s.layout_seed: part_state(run_dir, s.layout_seed) for s in specs}
    complete = all(st[0] for st in states.values())
    if status == "done" and not complete:
        status = "error"
        prog.update(error="全部回したのに、完全でない種が残っている: "
                          f"{[(sd, st[1]) for sd, st in states.items() if not st[0]]}")
    if status == "done":
        eps = [episode_row(states[s.layout_seed][2], run_dir / f"part_{s.layout_seed}") for s in specs]
        wall_total = sum(float(json.loads((run_dir / f"part_{s.layout_seed}" / "timing.json").read_text(encoding="utf-8"))["wall_s"])
                         for s in specs)
        segs = [{"env": {k: (x.get("env") or {}).get(k) for k in m96.ENV_STOP_KEYS}, "seeds": x.get("ran_seeds", [])}
                for x in log["sessions"] if x.get("ran_seeds")]
        summ = {"written": _now_s(), "name": a.name, "seeds": a.seeds, "rig_kind": "v3",
                "generator": "recovla.expert.generate.generate([spec], run_dir/part_<種>, workers=1, render=True, rig_kind='v3')（種ごと）",
                "spec_rule": "scene.sample_layout(種, 'empty')、色は table_colors[種 mod 色の数]、kind n", "n_specs": len(specs),
                "n_saved": sum(e["success"] for e in eps), "episodes": eps, "wall_s": round(wall_total, 1),
                "env": env, "env_segments": segs, "memory_at_start": mem, "sessions": len(log["sessions"]),
                "script_sha256": sha256_file(pathlib.Path(__file__)), "used_seeds": seeds}
        _write(run_dir / "gen_summary.json", summ)
        print(f"[x2] 保存 {summ['n_saved']}/{len(specs)} 本、生成の時間の和 {wall_total:.0f} s", flush=True)
    sess.update({"end": _now_s(), "status": status, "stop_reason": prog.d.get("stop_reason")})
    _write(log_path, log)
    prog.stop_heartbeat()
    prog.update(status=status, done=sum(st[0] for st in states.values()), complete=complete)
    return EXIT[status]


# ---------------------------------------------------------------- eval
def first_close_raw(data: dict) -> int | None:
    idx = np.flatnonzero(np.asarray(data["gripper_closed"], bool))
    return int(idx[0]) if idx.size else None


def build_observations(meta, data, ep_dir, tracker, keep_flag, k_last):
    """学習の変換と同じ作り方で、10 fps のこま 0..k_last の観測を順に出す（手がかりは全部のこまで更新する）。"""
    from recovla.data import convert as C
    from recovla.data import vla_observation, vla_state
    from recovla.data import vla_image_spec as spec
    arr = C.episode_arrays(meta, data)
    etr = C.episode_tracker(meta, tracker)
    etr.reset()
    for k, i in enumerate(arr["raw_index"]):
        if k > k_last:
            break
        raw = {view: C.read_raw_image(ep_dir / view / f"{int(i):06d}.png") for view in spec.CAMERAS}
        c = etr.update(raw["overhead"], meta["target"])
        state = vla_state.with_cue(arr["state"][k], c, keep_flag=keep_flag)
        yield k, {**vla_observation.observation_images(raw), "observation.state": state, "task": meta["instruction"]}


EVAL_NPZ_KEYS = ("k_index", "pred_post", "f_close")


class _Model:
    """方策の読み込み（最初に推論が要るときだけ）。"""

    def __init__(self, a, m96):
        self.a, self.m96, self.pol, self.tracker, self.load_s, self.ckpt = a, m96, None, None, None, None

    def get(self):
        if self.pol is not None:
            return self.pol, self.tracker
        from recovla.common import config
        from recovla.data import convert as C
        from recovla.harness.setup import nominal_setup
        from recovla.harness.sensors import SensorSuite
        from recovla.harness.world import WorldRig
        from recovla.runtime.motion import Motion
        from recovla.runtime.policy import SensorPolicy
        from recovla.runtime.runner import disable_rtc_for
        v82 = self.m96.load_82(False)
        cfg = config.load_v2()
        # SensorPolicy は自分の手がかり（実行系の口）のために setup.cameras を要る。評価の枠と同じ作り方で設置情報を作る
        # （ここでは方策の observe も手がかりも使わない。観測は学習の変換と同じ関数で作る＝build_observations）
        world = WorldRig(render=False, cfg=cfg)
        suite = SensorSuite(world.model, cfg)
        setup = suite.start_trial(0, world.data, nominal_setup(cfg))
        suite.close()
        self.ckpt = config.path(v82.CKPT[self.a.model])
        t = time.perf_counter()
        pol = SensorPolicy(self.ckpt, setup, Motion(setup).hand_pose)
        disable_rtc_for(pol)
        self.load_s = time.perf_counter() - t
        tc = pol.conversion.get("target_cue")
        tracker = None
        if tc is not None:
            tracker = C.cue_tracker(tc.get("thresholds_source", "training"))
            if tracker.thr.to_json() != tc["thresholds"]:
                raise SystemExit(f"手がかりの閾値が保存点の学習データと違う: {tracker.thr.to_json()} != {tc['thresholds']}")
        self.pol, self.tracker = pol, tracker
        return pol, tracker


def cmd_eval(a) -> int:
    gen_dir = X2 / "gen" / a.name
    gp = gen_dir / "gen_summary.json"
    out_dir = X2 / f"eval_{a.name}_{a.model}"
    m96 = load96()
    ops = m96.load_ops()
    if not gp.is_file():
        if a.dry_run:
            env = m96.read_env(ops)
            print(f"[x2] 環境: {m96.env_brief(env)}", flush=True)
            print(f"[dry-run] {_rel(gp)} がまだ無い（gen の後に回る）。回す本数は gen の後に決まる。出力 {_rel(out_dir)}")
            return 0 if env.get("driver") else 3
        raise SystemExit(f"{gp} がない（gen を先に）")
    if a.model not in m96.load_82(False).CKPT:
        raise SystemExit(f"--model {a.model}: 82_v2_eval.py の CKPT にない")
    gs = json.loads(gp.read_text(encoding="utf-8"))
    if (out_dir / "summary.json").is_file() and not a.force:
        print(f"[x2] {_rel(out_dir)}\\summary.json がある（済み）。--force で作り直す", flush=True)
        if a.dry_run:
            print(f"[dry-run] {_rel(out_dir)}: 済み（summary.json あり）。回す 0 本")
        return 0
    eps_all = [e for e in gs["episodes"] if e["success"]]
    log_path = out_dir / "eval_log.json"
    force_move = a.force and out_dir.is_dir()
    log = {"sessions": []} if force_move else read_log(log_path)
    done_names = set() if force_move else {e["name"] for e in eps_all if _npz_ok(out_dir / f"{e['name']}.npz", EVAL_NPZ_KEYS)}
    todo = [e for e in eps_all if e["name"] not in done_names]
    by_name_env = {}
    for s_ in log.get("sessions", []):
        for nm in s_.get("ran", []):
            by_name_env[nm] = s_.get("env")
    recorded = [("gen_summary.json", gs.get("env"))] + [(f"episode:{nm}", by_name_env.get(nm)) for nm in sorted(done_names)]
    env = m96.read_env(ops)
    conf = env_conflicts(m96, env, recorded, log.get("sessions", []))
    print(f"[x2] 環境: {m96.env_brief(env)}", flush=True)
    if a.dry_run:
        print(f"[dry-run] {_rel(out_dir)}: 生成 {_rel(gen_dir)} の成功 {len(eps_all)} 本、完全 {len(done_names)} 本（推論を飛ばす）、"
              f"回す {len(todo)} 本{'（--force: 前の出力を _superseded_<時刻> へ移して最初から）' if force_move else ''}")
        for e in eps_all:
            print(f"  {e['name']:24s} {'完全' if e['name'] in done_names else '回す'}")
        if conf:
            print(f"[dry-run] 環境の食い違い: {conf}{'（--accept-env-change で分けて進める）' if a.accept_env_change else ''}")
        try:
            print(f"[dry-run] 空きメモリ {ops.memory_gb()['phys_free_gb']} GB（{MIN_FREE_GB} 未満なら回さない）")
        except Exception:                        # noqa: BLE001
            pass
        return 3 if (not env.get("driver") or (conf and not a.accept_env_change)) else 0
    check_env(m96, env, conf, a.accept_env_change, _rel(out_dir))
    mem = check_memory(ops)
    prog_path = out_dir / "progress.json"
    check_not_running(ops, prog_path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if force_move:                                  # 前の出力は消さずに退避（progress.json は今の回が書き直す）
        d = out_dir / f"_superseded_{stamp}"
        d.mkdir(parents=True, exist_ok=True)
        for p in list(out_dir.iterdir()):
            if p.name != d.name and not p.name.startswith(("_superseded_", "_incomplete_")) and p.name != "progress.json":
                shutil.move(str(p), str(d / p.name))
    out_dir.mkdir(parents=True, exist_ok=True)
    moved = quarantine(out_dir, sorted(p.name for p in out_dir.glob("*.tmp")), stamp)
    sess = {"start": _now_s(), "pid": os.getpid(), "skipped_complete": len(done_names), "to_run": [e["name"] for e in todo],
            "ran": [], "quarantined": moved, "end": None, "status": "running", "env": env, "memory_at_start": mem}
    if conf:
        sess["env_segment"] = {"accepted": "--accept-env-change", "conflicts": conf,
                               "note": "この回で推論したエピソードは、前の環境のものと別の環境として分けて報告する"}
    log.setdefault("sessions", []).append(sess)
    _write(log_path, log)
    base = {"experiment": "outputs\\s4\\x2", "condition": f"eval_{a.name}_{a.model}", "cmd": "eval", "model": a.model,
            "trials_spec": f"gen:{a.name}", "pid": os.getpid(), "host": os.environ.get("COMPUTERNAME"), "started": _now_s(),
            "status": "starting", "total": len(eps_all), "done": len(done_names), "skipped_complete": len(done_names),
            "ran_this_session": 0, "successes": None, "last_trial_wall_s": None, "stop_reason": None, "error": None,
            "quiet_window": None, "min_free_gb": MIN_FREE_GB, "time_limits": None, "env": env, "env_conflicts": conf or None}
    prog = m96.Progress(prog_path, ops, base)
    prog.start_heartbeat()
    from recovla.data import convert as C
    from recovla.policy.schedule import noise_generator
    model = _Model(a, m96)
    rows_min = min(D.ROWS)
    status, ran, t0 = "done", 0, time.perf_counter()
    try:
        prog.update(status="running")
        for e in todo:
            if a.max_new and ran >= a.max_new:
                raise _Stop(f"max_new:{a.max_new}")
            r = stop_requested(out_dir)
            if r:
                raise _Stop(r)
            if model.pol is None:
                prog.update(status="loading")
                model.get()
                prog.update(status="running")
            pol, tracker = model.get()
            prog.update(current=e["name"], current_seed=e["seed"], current_started=_now_s())
            t1 = time.perf_counter()
            ep_dir = ROOT / e["path"]
            meta, data = C.load_raw(ep_dir)
            f1 = first_close_raw(data)
            k_close = (int(meta["n_frames"]) - 1) // C.STRIDE if f1 is None else f1 // C.STRIDE
            k_last = k_close - rows_min
            ks, preds = [], []
            for k, obs in build_observations(meta, data, ep_dir, tracker, pol.cue_keep_flag, k_last):
                if k % a.stride:
                    continue
                post = pol.infer(obs, noise_generator(int(meta["layout_seed"]), int(k)), None)
                ks.append(k)
                preds.append(np.asarray(post, np.float32)[:, :7])
            p = out_dir / f"{meta['name']}.npz"
            tmp = p.with_name(p.name + ".tmp")
            with open(tmp, "wb") as f:
                np.savez(f, k_index=np.array(ks, np.int32), pred_post=np.array(preds, np.float32).reshape(len(ks), -1, 7),
                         f_close=np.int32(-1 if f1 is None else f1))
            os.replace(tmp, p)
            ran += 1
            sess["ran"].append(meta["name"])
            _write(log_path, log)
            prog.update(done=len(done_names) + ran, ran_this_session=ran, last_trial_wall_s=round(time.perf_counter() - t1, 1))
            print(f"[x2] {meta['name']}: 推論 {len(ks)} 本", flush=True)
    except _Stop as ex:
        status = "stopped"
        prog.update(stop_reason=ex.reason)
        print(f"[x2] 止める合図: {ex.reason}", flush=True)
    except KeyboardInterrupt:
        status = "interrupted"
        prog.update(stop_reason="KeyboardInterrupt")
    except BaseException as ex:                  # noqa: BLE001
        status = "error"
        prog.update(error=f"{type(ex).__name__}: {ex}", traceback=traceback.format_exc()[-3000:])
        traceback.print_exc()
    wall = time.perf_counter() - t0
    complete = all(_npz_ok(out_dir / f"{e['name']}.npz", EVAL_NPZ_KEYS) for e in eps_all)
    if status == "done" and not complete:
        status = "error"
        prog.update(error="全部回したのに、完全でないエピソードが残っている")
    if status == "done":
        # 不足は npz（推論の結果）と生成の記録から計算する（飛ばしたエピソードも同じ道。GPU は使わない）
        eps, per_ep, n_inf = [], [], 0
        for e in eps_all:
            meta, data = C.load_raw(ROOT / e["path"])
            with np.load(out_dir / f"{e['name']}.npz", allow_pickle=False) as z:
                ks, preds, fc = np.asarray(z["k_index"], int), np.asarray(z["pred_post"], np.float32), int(z["f_close"])
            f1 = None if fc < 0 else fc
            n_inf += len(ks)
            ti = D.COLORS.index(meta["target"])
            sf = D.x2_episode_shortfall(preds.reshape(len(ks), -1, 7), ks, np.asarray(data["x_des"], float),
                                        np.asarray(data["cube_pos"], float)[:, ti, :2], f1, stride=C.STRIDE)
            row = {"episode": meta["name"], "seed": meta["layout_seed"], "target": meta["target"], "start_pose": meta.get("start_pose"),
                   "f_close_raw": f1, "n_inferences": int(len(ks)), "skipped_inference": e["name"] in done_names,
                   "median_mm": {h: (None if not v else round(float(np.median(v)), 2)) for h, v in sf.items()},
                   "n": {h: len(v) for h, v in sf.items()}}
            eps.append(sf)
            per_ep.append(row)
            print(f"[x2] {meta['name']}: 推論 {len(ks)} 本、h ごとの中央値 {row['median_mm']}（件数 {row['n']}）", flush=True)
        segs = [{"env": {k: (x.get("env") or {}).get(k) for k in m96.ENV_STOP_KEYS}, "episodes": x.get("ran", [])}
                for x in log["sessions"] if x.get("ran")]
        summ = {"written": _now_s(), "gen": a.name, "gen_summary_sha256": sha256_file(gp), "model": a.model,
                "checkpoint": _rel(model.ckpt) if model.ckpt else None, "stride": a.stride, "rows": list(D.ROWS),
                "definition": "src/recovla/diag/rtc.py の冒頭の X2（正は予測が短い）", "summary": D.x2_summary(eps),
                "n": {str(h): sum(len(ep[str(h)]) for ep in eps) for h in D.ROWS}, "episodes": per_ep,
                "n_inferences": n_inf, "wall_s_this_session": round(wall, 1),
                "model_load_s": None if model.load_s is None else round(model.load_s, 1),
                "env": env, "env_segments": segs, "memory_at_start": mem, "sessions": len(log["sessions"]),
                "noise": "noise_generator(配置の種, k)", "rtc": "切る（disable_rtc_for）",
                "script_sha256": sha256_file(pathlib.Path(__file__)), "module_sha256": sha256_file(pathlib.Path(D.__file__))}
        try:
            import torch
            summ["gpu_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        except Exception:                        # noqa: BLE001
            pass
        _write(out_dir / "summary.json", summ)
        print(f"[x2] 要約: {json.dumps(summ['summary'], ensure_ascii=False)}、件数 {summ['n']}、{wall:.0f} s", flush=True)
    sess.update({"end": _now_s(), "status": status, "stop_reason": prog.d.get("stop_reason")})
    _write(log_path, log)
    prog.stop_heartbeat()
    prog.update(status=status, complete=complete)
    return EXIT[status]


def cmd_summary(a) -> int:
    p = X2 / f"eval_{a.name}_{a.model}" / "summary.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    print(json.dumps({k: s.get(k) for k in ("gen", "model", "summary", "n", "n_inferences", "wall_s", "wall_s_this_session")},
                     ensure_ascii=False, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gen")
    p.add_argument("--seeds", default="44404:20", help="<先頭>:<数>（既定は X2 の帯 44404〜44423）")
    p.add_argument("--name", default="X2_gen")
    p = sub.add_parser("eval")
    p.add_argument("--name", default="X2_gen")
    p.add_argument("--model", default="R1v3")
    p.add_argument("--stride", type=int, default=1, help="推論する 10 fps のこまの間隔（既定 1 = 全部）")
    p.add_argument("--force", action="store_true", help="前の出力を _superseded_<時刻>\\ へ移して最初から")
    for name in ("gen", "eval"):
        p = sub.choices[name]
        p.add_argument("--dry-run", action="store_true", help="何を飛ばし何を回すか・環境の照合を見るだけ（何も書かない）")
        p.add_argument("--accept-env-change", action="store_true",
                       help="ドライバ・torch・CUDA・OS の版が記録と違っても進める（この回を env_segment として分ける）")
        p.add_argument("--max-new", type=int, default=0, help="新しい種・エピソードをこの本数回したら止まる（0 で無制限）")
    p = sub.add_parser("summary")
    p.add_argument("--name", default="X2_gen")
    p.add_argument("--model", default="R1v3")
    return ap


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                            # noqa: BLE001
        pass
    a = build_parser().parse_args(argv)
    try:
        return {"gen": cmd_gen, "eval": cmd_eval, "summary": cmd_summary}[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
