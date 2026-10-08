"""段階 4 束 1 の D-RTC（RTC の 6 設定）と影の推論（担当 A）。関門 R の材料を作る。

使い方（作業場所 C:\\PAI\\recovery_vla。続きから回す仕組み・環境の照合・制限時間は 96_s4_resume.py run と同じ。同じコマンドで続きから回る）:
    .venv\\Scripts\\python.exe scripts\\98_s4_d_rtc.py settings [--out outputs\\s4\\d_rtc\\settings.json]   # 6 設定の中身（回す前に掲示する）
    .venv\\Scripts\\python.exe scripts\\98_s4_d_rtc.py plan                                               # 本番のコマンドと見込み
    .venv\\Scripts\\python.exe scripts\\98_s4_d_rtc.py run --setting range10_cap5 [--dry-run] [96 の続きの引数 ...]
        既定: --experiment S4DRTC --condition <設定名> --trials natural:190200:10（30 試行）。R1v3・6 行・安全フィルタなし・60 s
    .venv\\Scripts\\python.exe scripts\\98_s4_d_rtc.py run --setting current_repro --shadow [--dry-run]
        影の推論。既定: --condition shadow_current_repro --trials natural:190220:4（12 試行）。実行は誘導した側、影は記録だけ
    .venv\\Scripts\\python.exe scripts\\98_s4_d_rtc.py rotate [--settings all] [--block-seeds 1] [--dry-run]
        6 設定を同じ種で、種の塊ごとに交互に回す（目標書_段階4.md 第 4 節「比べる組は同じ種・同じ時期に、種の塊ごとに交互に」）。
        1 つのプロセスで方策を読み込んだまま設定を切り替える（RTC の設定だけを入れ直す）。監視役は
        outputs\\s4\\d_rtc\\rotate_<タグ>.progress.json を見る（96_s4_ops.py wait --progress ...）。
        smoke 用に --block-trials N（交互の単位を試行の数で）と --rounds N（N 巡で止める）がある。
        全設定の試行がそろっても run.json か G_AUDIT.json が無い設定（再起動の隙間で切れた）は、96 を 1 回呼んで書かせてから終わる
    .venv\\Scripts\\python.exe scripts\\98_s4_d_rtc.py metrics --experiment S4K --condition K1 [--out ...]        # 関門 R の指標（読むだけ）
    .venv\\Scripts\\python.exe scripts\\98_s4_d_rtc.py shadow-metrics --experiment S4DRTC --condition shadow_current_repro
  smoke（学習用の帯の担当 A の小帯 44430〜44449 だけ）: run --setting ZEROS --experiment S4SMOKE_A --condition ZEROS --trials natural:44430:1

読むもの: scripts\\96_s4_resume.py（importlib で読み込み、関数を包む・差し替える。書き換えない）、82_v2_eval.py（96 経由、ハッシュ照合）、
  configs（読んだ辞書の写しに RTC の値を重ねる。ファイルは書き換えない）、configs\\s4_gates.json（帯の確認）、src\\recovla\\diag\\rtc.py。
書くもの: outputs\\v2eval\\<実験>\\<条件>\\ に 96 と同じ記録（trial_NNNN.json/.npz、runtime_NNNN.json、run.json、G_AUDIT.json、
  progress.json、resume_spec.json、resume_log.json）と diag_NNNN.npz（推論ごとの誘導した塊・影の塊。下）。
  trial の json に "diag"（診断の名前・腕＝設定・設定の中身・実行系の値・影の有無・開始状態・誘発の版・lerobot に入った RTC の値・
  推論の本数）を足し、"runtime" に "rtc"（実行系の値）を足す。run.json に "diag" を足す。resume_spec.json に "diag_rtc" を足す
  （同じ条件名で別の設定を混ぜない）。rotate は outputs\\s4\\d_rtc\\rotate_<タグ>.progress.json・rotate_<タグ>.log.json。
  metrics・shadow-metrics は --out を指定したときだけ書く。
  diag_NNNN.npz: inf_index・has_rtc・inference_delay・left_over_rows・guided_post/guided_raw（推論ごとの (50, 7)。raw は後処理の前の
  正規化された空間の先頭 7 次元）・has_shadow・shadow_post/shadow_raw（影がなければ NaN）。並びは runtime_NNNN.json の
  inference[] と同じ（本数を照合して記録する）。試行の 3 つのファイルより先に書き、続きから回すときの「完全」にこのファイルも要る。

6 設定が実行系の何の値に当たるか（src\\recovla\\diag\\rtc.py の SETTINGS。結果を見る前に固定し、回す前に掲示する＝目標書 9 節）:
  共通: PolicyRuntime（src\\recovla\\runtime\\runner.py）の s = 6、d_init = 4（configs runtime.delay_steps）、方策に渡す
    inference_delay = min(直前の推論で実際にかかった行の数, s − 1)（runner.py 255〜259 行）、前の塊の残りは後処理の前の値を
    normalize_left_over で範囲 E 行に揃える（runner.py 257〜259 行）、流れの積分 10 段（SmolVLA num_steps）、
    補正にヤコビアンなし（lerobot 0.6.1 modeling_rtc.py 212〜219 行）、R1v3、安全フィルタなし、60 s
  current_repro              mode rtc、E = 40、EXP、β = 10（configs/default.yaml 155〜157 行＝組 E・V3S3/E_nat と同じ）
  paper_formula_range44_cap5 mode rtc、E = 44、EXP、β = 5
  range40_cap5               mode rtc、E = 40、EXP、β = 5
  range10_cap5               mode rtc、E = 10、EXP、β = 5
  ZEROS                      mode rtc、E = 44、ZEROS（重みは前の min(d, E) = d 行だけ 1）、β = 5（実装役の決定。下）
  naive                      mode naive（RTC なし。物差し K と同じ）
  ZEROS の範囲と上限は文書に書かれていない。範囲は ZEROS では効かない（重みは d 行だけ。lerobot get_prefix_weights）。上限は
  ほかの 3 候補と同じ 5 にし、論文の式どおりの設定と「減衰の形だけ」が違うようにした（W\\rtc の F1「重みを前の d 行だけ」）。

「論文の式どおり」の根拠（arXiv 2506.07339 Real-Time Execution of Action Chunking Flow Policies、Kinetix の参照実装）:
  式 5（soft masking）: W_i = 1（i < d）、c_i (e^{c_i} − 1)/(e − 1)（d <= i < H − s、c_i = (H − s − i)/(H − s − d + 1)）、0（i >= H − s）。
  誘導の重み: min(β, (1 − τ)/τ · 1/r_τ²)、r_τ² = (1 − τ)²/(τ² + (1 − τ)²)、既定 β = 5。参照実装 eval_flow.py は範囲 = H − s、段数 5。
  lerobot の EXP（get_prefix_weights(start = d, end = E, total = H)）は c = (E − i)/(E − d + 1) なので、E = H − s = 50 − 6 = 44 で
  式 5 と一致する（tests\\test_s4_diag_A.py で数値の一致を確かめる）。誘導の重みの式は lerobot も同じ（modeling_rtc.py 221〜227 行）。
  src・lerobot が論文と違う所と、この設定で直したか:
    (1) 範囲 E: configs/default.yaml 155 行は 40（9/26 に 10 行ごとの実行で滑らかさで選んだ値）→ この設定で 44 に直す
    (2) 上限 β: default.yaml 157 行は 10（lerobot の既定）→ この設定で 5 に直す
    (3) 補正のヤコビアン: 論文・参照実装は ∂Â₁/∂A^τ（jax.vjp）を掛けるが、lerobot 0.6.1 は v_t を requires_grad の前に計算する
        ので恒等（modeling_rtc.py 212〜219 行。W\\rtc\\a9_vjp_check.py）→ 直さない（ライブラリの書き換えになる。F2 は後回しの決定）
    (4) 段数: SmolVLA は 10 段、論文は 5 段 → 直さない（方策の推論の設定。β = 5・10 段の最終段の引き戻しは 0.50、論文の 5・5 段は
        0.85。W\\rtc_skeptic\\s2_out.json）
    (5) d: runner.py は直前の実測の遅れ（最大 s − 1 = 5）を渡す。論文も遅れの見込みを使う → 同じ考え方なので直さない
    (6) s/H: ここは 6/50、論文の実機は 25/50 → 実行の設定なので直さない（線形の模型で最新の観測の割合は 0.32 対 0.82。s2_out.json）
  つまり「論文の式どおり」は「重みの式と β を論文に合わせた設定」であり、論文の実装の再現ではない（(3)(4)(6) が残る）。

影の推論（目標書 8-2 の R.b1、final.md の束 1。結果を見る前に固定）:
  誘導して実行する設定は current_repro（崩れた組 E の設定。仮説 1「固定」と 2「先の計画が短い」を分けるのが目的＝W\\rtc の X1）。
  推論ごとに、同じ観測・同じ雑音（雑音の生成器の状態を戻して引き直す）で RTC の引数を渡さない推論（＝naive と同じ）を足して
  記録する。戻り値・次の推論の前の塊は誘導した側のまま。計算の時間はシミュレーションの上では分布から引くので（harness/robot_io.py）、
  影を足しても世界の進みは変わらない（実時間だけ延びる）。指標は src\\recovla\\diag\\rtc.py の冒頭（h = 10・20・30・40 行の変位の差の
  中央値。どれかが 15 mm 以上なら「先の計画が短い」）。

結果を見る前に決めてある項目: 6 設定の中身（上）、各設定の帯（190200〜190209、30 試行。影は 190220〜190223、12 試行）、
  60 s で回し 30 s を主に採点、R1v3・6 行・安全フィルタなし・試行ごとに世界を作り直す（96 の既定）、影の誘導側の設定、
  関門 R の指標の定義（src\\recovla\\diag\\rtc.py の冒頭。移動の比の 30 s 版・主の集計（中央値）・影の推論の比べる相手と h の規則・
  ZEROS の中身は settings の "definitions_before_run" に文で出す＝回す前に掲示する）、
  rotate の交互の単位（既定 1 種 = 3 試行ごとに設定を替える）。
終了コード: 96_s4_resume.py と同じ（0 全部そろった、1 途中で止まった、2 エラー、3 引数・前提の食い違い）。
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recovla.diag import rtc as D            # noqa: E402

S4 = ROOT / "outputs" / "s4"
OUT_DIAG = S4 / "d_rtc"
GATES = ROOT / "configs" / "s4_gates.json"
MODEL = "R1v3"
EXEC_INTERVAL = 6
DEFAULTS = {"experiment": "S4DRTC", "trials": "natural:190200:10", "shadow_trials": "natural:190220:4", "shadow_setting": "current_repro"}
SMOKE_BAND = (44400, 44799)                  # 学習用の帯（smoke・データ。目標書 4-1）


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


# ---------------------------------------------------------------- 帯の確認（s4_gates.json の allocations）
def allowed_band(shadow: bool) -> tuple:
    g = json.loads(GATES.read_text(encoding="utf-8"))
    want = "D_RTC_shadow" if shadow else "D_RTC"
    for a in g["bands"]["allocations"]:
        if a["id"] == want:
            return tuple(a["range"])
    raise SystemExit(f"{GATES}: allocations に {want} がない")


def trial_seeds(trials: str) -> list:
    kind, base, n = trials.split(":")
    return list(range(int(base), int(base) + int(n)))


def check_band(trials: str, shadow: bool, allow_other: bool) -> None:
    """種が割り当ての帯か、smoke の学習用の帯の中にあること（違えば止める。--allow-other-band で外せる）。"""
    kind = trials.split(":")[0]
    if kind != "natural":
        raise SystemExit(f"D-RTC は自然の試行だけ（--trials natural:<先頭>:<数>）: {trials}")
    seeds = trial_seeds(trials)
    lo, hi = allowed_band(shadow)
    in_main = all(lo <= s <= hi for s in seeds)
    in_smoke = all(SMOKE_BAND[0] <= s <= SMOKE_BAND[1] for s in seeds)
    if not (in_main or in_smoke) and not allow_other:
        raise SystemExit(f"種 {seeds[0]}〜{seeds[-1]} は割り当ての帯 {lo}〜{hi} にも smoke の帯 {SMOKE_BAND} にもない")


# ---------------------------------------------------------------- 96 を包む
SHARED_CACHE = {}                            # rotate: 方策（GPU の模型）をプロセスの中で使い回す（RTC の設定は条件ごとに入れ直す）
PENDING_DIAG = {}                            # 試行の番号 → diag の配列（write_trial_files が書く）


def rtc_config_seen(pol) -> dict:
    rc = pol.policy.config.rtc_config
    proc = getattr(getattr(pol.policy, "model", None), "rtc_processor", None)
    if rc is None:
        return {"enabled": False, "processor": proc is not None}
    return {"enabled": bool(rc.enabled), "prefix_attention_schedule": str(getattr(rc.prefix_attention_schedule, "name", rc.prefix_attention_schedule)),
            "max_guidance_weight": float(rc.max_guidance_weight), "execution_horizon": int(rc.execution_horizon),
            "processor": proc is not None}


def check_seen(seen: dict, name: str) -> None:
    s = D.setting(name)
    if s["mode"] != "rtc":
        if seen["enabled"] or seen["processor"]:
            raise RuntimeError(f"{name}: RTC を切ったはずが lerobot に残っている {seen}")
        return
    want = {"enabled": True, "prefix_attention_schedule": s["schedule"], "max_guidance_weight": float(s["max_guidance_weight"]),
            "execution_horizon": int(s["horizon"]), "processor": True}
    if seen != want:
        raise RuntimeError(f"{name}: lerobot に入った RTC の値 {seen} が設定 {want} と違う")


def patch96(m, name: str, shadow: bool, script_sha: str, module_sha: str) -> None:
    """96 の関数を包む・差し替える（96 のファイルは書き換えない）。同じプロセスで呼ぶたびに、元の関数から包み直す。"""
    orig = getattr(m, "_diag_orig", None)
    if orig is None:
        orig = {k: getattr(m, k) for k in ("overlay_limits", "make_spec", "trial_paths", "check_complete", "write_trial_files",
                                            "run_json_text", "Engine")}
        m._diag_orig = orig
    setting_rec = D.setting(name)
    diag_name = "D-RTC-shadow" if shadow else "D-RTC"

    def overlay_limits(cfg, lim):
        return D.overlay_rtc(orig["overlay_limits"](cfg, lim), name)

    def make_spec(a):
        s = orig["make_spec"](a)
        s["diag_rtc"] = {"diag": diag_name, "setting": setting_rec, "shadow": shadow}
        return s

    def trial_paths(kind, out, i):
        ps = orig["trial_paths"](kind, out, i)
        if kind == "run":
            ps["diag"] = out / f"diag_{i:04d}.npz"
        return ps

    def check_complete(kind, out, i, expect):
        ok, why, meta = orig["check_complete"](kind, out, i, expect)
        if not ok or kind != "run":
            return ok, why, meta
        dg = (meta or {}).get("diag") or {}
        if dg.get("arm") != name or bool(dg.get("shadow")) != shadow:
            return False, f"diag_meta:{dg.get('arm')}/{dg.get('shadow')}", None
        import zipfile
        p = out / f"diag_{i:04d}.npz"
        try:
            with zipfile.ZipFile(p) as z:
                if z.testzip() is not None or not z.namelist():
                    return False, "diag:crc", None
        except Exception as e:                   # noqa: BLE001
            return False, f"diag:{type(e).__name__}", None
        return ok, why, meta

    def write_trial_files(kind, out, i, meta, arrays, rlog, v82):
        da = PENDING_DIAG.pop(i)
        p = out / f"diag_{i:04d}.npz"
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **da)
        os.replace(tmp, p)                       # 試行の 3 つより先に書く（3 つがそろっていれば diag もある）
        orig["write_trial_files"](kind, out, i, meta, arrays, rlog, v82)

    def run_json_text(a, kind, rows, wall_s, exec_interval, lim, env=None, segs=None):
        d = json.loads(orig["run_json_text"](a, kind, rows, wall_s, exec_interval, lim, env, segs))
        d["diag"] = {"diag": diag_name, "arm": name, "setting": setting_rec, "shadow": shadow, "script": "scripts/98_s4_d_rtc.py",
                     "script_sha256": script_sha, "module_sha256": module_sha}
        return json.dumps(d, ensure_ascii=False, indent=1)

    class DiagEngine(orig["Engine"]):
        def __init__(self, a, v82, cfg, lim, env=None):
            super().__init__(a, v82, cfg, lim, env)
            self.cache = SHARED_CACHE.setdefault(a.model, {})
            self._prox, self._seen = None, None

        def init_run(self):
            super().init_run()
            inner = self.make_runtime
            from recovla.runtime.runner import disable_rtc_for, enable_rtc_for
            rt_cfg = self.cfg["runtime"]

            def make_runtime(io, setup):
                fresh = "pol" not in self.cache
                rt = inner(io, setup)
                if self._seen is None:           # この条件の最初の試行: 使い回した方策にも、この設定の RTC を入れ直す
                    if not fresh:
                        if setting_rec["mode"] == "rtc":
                            enable_rtc_for(self.cache["pol"], int(rt_cfg["rtc_guidance_horizon"]), rt_cfg["rtc_schedule"],
                                           float(rt_cfg["rtc_max_guidance_weight"]))
                        else:
                            disable_rtc_for(self.cache["pol"])
                    self._seen = rtc_config_seen(self.cache["pol"])
                    check_seen(self._seen, name)
                if rt.E != int(rt_cfg["rtc_guidance_horizon"]) or rt.mode != setting_rec["mode"]:
                    raise RuntimeError(f"{name}: PolicyRuntime の mode {rt.mode}・E {rt.E} が設定と違う")
                self._prox = D.DiagPolicy(rt.policy, shadow=shadow)
                rt.policy = self._prox
                return rt
            self.make_runtime = make_runtime

        def run_one(self, i, seed, lay, tgt):
            self._prox = None
            meta, arrays, rlog = super().run_one(i, seed, lay, tgt)
            da = self._prox.arrays()
            n_inf = len(rlog["runtime"]["inference"])
            if len(da["inf_index"]) != n_inf:
                raise RuntimeError(f"試行 {i}: 記録した推論 {len(da['inf_index'])} 本と runtime の inference {n_inf} 本が合わない")
            PENDING_DIAG[i] = da
            rv = D.runtime_values(self.cfg, name)
            meta["runtime"]["rtc"] = rv
            meta["diag"] = {"diag": diag_name, "version": 1, "arm": name, "setting": setting_rec, "runtime_values": rv,
                            "shadow": shadow,
                            "shadow_note": ("推論ごとに同じ観測・同じ雑音で RTC の引数を渡さない推論（影）を記録した。実行は誘導した側"
                                            if shadow else None),
                            "start_state": "natural（41_results.trial_list の配置と開始。種から決まる）", "induce": None,
                            "induce_version": None, "rtc_config_seen": self._seen, "n_inference": n_inf,
                            "n_shadow": int(da["has_shadow"].sum()), "diag_file": f"diag_{i:04d}.npz",
                            "script": "scripts/98_s4_d_rtc.py", "script_sha256": script_sha,
                            "module": "src/recovla/diag/rtc.py", "module_sha256": module_sha}
            return meta, arrays, rlog

    m.overlay_limits = overlay_limits
    m.make_spec = make_spec
    m.trial_paths = trial_paths
    m.check_complete = check_complete
    m.write_trial_files = write_trial_files
    m.run_json_text = run_json_text
    m.Engine = DiagEngine


def argv96(a, setting_name: str, condition: str, trials: str, extra: list, max_new: int = None) -> list:
    mode = D.setting(setting_name)["mode"]
    out = ["run", "--experiment", a.experiment, "--condition", condition, "--model", MODEL, "--trials", trials,
           "--mode", mode, "--exec-interval", str(EXEC_INTERVAL), "--no-safety"]
    if max_new:
        out += ["--max-new", str(max_new)]
    return out + list(extra)


FORBIDDEN_EXTRA = ("--model", "--mode", "--exec-interval", "--induce", "--ablate", "--time-limit-s", "--reuse-world", "--grip-gate",
                   "--trials", "--experiment", "--condition")


def run_condition(m, ops, v82, a, setting_name, shadow, condition, trials, extra, max_new=None) -> int:
    for f in extra:
        if f.split("=")[0] in FORBIDDEN_EXTRA:
            raise SystemExit(f"{f} は 98_s4_d_rtc.py が決める（D-RTC は R1v3・6 行・安全フィルタなし・60 s・自然・世界の作り直し）")
    script_sha = sha256_file(pathlib.Path(__file__))
    module_sha = sha256_file(pathlib.Path(D.__file__))
    patch96(m, setting_name, shadow, script_sha, module_sha)
    a96 = m.build_parser().parse_args(argv96(a, setting_name, condition, trials, extra, max_new))
    try:
        return m.cmd_main(a96, v82, ops)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


def cmd_run(a, extra) -> int:
    name = a.setting
    D.setting(name)
    shadow = bool(a.shadow)
    if shadow and D.setting(name)["mode"] != "rtc":
        raise SystemExit("影の推論は RTC の設定で回す（naive では誘導がない）")
    trials = a.trials or (DEFAULTS["shadow_trials"] if shadow else DEFAULTS["trials"])
    condition = a.condition or (f"shadow_{name}" if shadow else name)
    check_band(trials, shadow, a.allow_other_band)
    m = load96()
    ops = m.load_ops()
    v82 = m.load_82(False)
    print(f"[d_rtc] 設定 {name}（{D.SETTINGS[name]['ja']}）影 {shadow} → {a.experiment}\\{condition} {trials}", flush=True)
    return run_condition(m, ops, v82, a, name, shadow, condition, trials, extra)


# ---------------------------------------------------------------- rotate（種の塊ごとに交互に）
def cmd_rotate(a, extra) -> int:
    names = list(D.SETTINGS) if a.settings == "all" else [s.strip() for s in a.settings.split(",") if s.strip()]
    for n in names:
        D.setting(n)
    trials = a.trials or DEFAULTS["trials"]
    check_band(trials, False, a.allow_other_band)
    n_trials = 3 * len(trial_seeds(trials))
    block = int(a.block_trials) if a.block_trials else 3 * int(a.block_seeds)
    m = load96()
    ops = m.load_ops()
    v82 = m.load_82(False)
    if "--dry-run" in extra:
        code = 0
        for n in names:
            code = max(code, run_condition(m, ops, v82, a, n, False, n, trials, extra))
        return code
    OUT_DIAG.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.experiment}_{'-'.join(names) if len(names) < 6 else 'all'}"
    prog_p, log_p = OUT_DIAG / f"rotate_{tag}.progress.json", OUT_DIAG / f"rotate_{tag}.log.json"
    try:
        import psutil
        ct = psutil.Process().create_time()
    except Exception:                            # noqa: BLE001
        ct = None
    prog = {"experiment": a.experiment, "condition": f"rotate:{tag}", "settings": names, "trials_spec": trials, "pid": os.getpid(),
            "proc_create_time": ct, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running", "total": n_trials * len(names),
            "done": 0, "successes": None, "block_trials": block, "calls": []}

    def put(**kw):
        prog.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        m._write_atomic(prog_p, json.dumps(prog, ensure_ascii=False, indent=1, default=str))

    def done_of(n):
        """設定 n の完全な試行の数（n の包みで数える。check_complete は diag の腕も照らすので、設定ごとに包み直す）。"""
        patch96(m, n, False, sha256_file(pathlib.Path(__file__)), sha256_file(pathlib.Path(D.__file__)))
        out = v82.OUT / a.experiment / n
        cnt = 0
        for i, (seed, lay, tgt) in enumerate(v82.trial_list(trials)):
            ok, _, _ = m.check_complete("run", out, i, {"seed": seed, "target": tgt})
            cnt += int(ok)
        return cnt

    def sealed(n):
        """設定 n の条件に run.json と G_AUDIT.json があるか（96 が全部そろったときだけ書く。保つ条件 1 の監査の印）。"""
        out = v82.OUT / a.experiment / n
        return (out / "run.json").is_file() and (out / "G_AUDIT.json").is_file()

    put()
    status, code = "done", 0
    rounds = 0
    try:
        while True:
            counts = {n: done_of(n) for n in names}
            left = [n for n in names if counts[n] < n_trials]
            if not left:
                # 試行が全部そろっていても、最後の試行を書いた後・run.json と G_AUDIT を書く前に切れた設定（再起動の隙間。
                # 査読の重要 3）は、96 を 1 回呼んで書かせる（回す試行は 0 本。96 は run.json・G_AUDIT が無ければ書く）
                for n in [x for x in names if not sealed(x)]:
                    t0 = time.time()
                    c = run_condition(m, ops, v82, a, n, False, n, trials, extra)
                    prog["calls"].append({"setting": n, "code": c, "status": "seal", "wall_s": round(time.time() - t0, 1),
                                          "note": "試行はそろっていたが run.json か G_AUDIT.json が無かった"})
                    m._write_atomic(log_p, json.dumps(prog["calls"], ensure_ascii=False, indent=1))
                    if c != 0 or not sealed(n):
                        status, code = ("stopped", 1) if c == 1 else ("error", c if c in (2, 3) else 2)
                        put(stop_reason=f"{n}: 試行はそろったが run.json・G_AUDIT.json を書けていない（96 の終了コード {c}）")
                        return code
                break
            if a.rounds and rounds >= a.rounds:                      # smoke など: 決めた巡の数で止める
                status, code = "stopped", 1
                put(stop_reason=f"rounds:{a.rounds}")
                return code
            rounds += 1
            if prog.get("_last_counts") == counts:
                raise RuntimeError(f"1 巡しても完全な試行が増えない: {counts}")
            prog["_last_counts"] = counts
            for n in left:
                t0 = time.time()
                c = run_condition(m, ops, v82, a, n, False, n, trials, extra, max_new=block)
                try:
                    pr = json.loads((v82.OUT / a.experiment / n / "progress.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pr = {}
                prog["calls"].append({"setting": n, "code": c, "status": pr.get("status"), "stop_reason": pr.get("stop_reason"),
                                      "done": pr.get("done"), "wall_s": round(time.time() - t0, 1)})
                m._write_atomic(log_p, json.dumps(prog["calls"], ensure_ascii=False, indent=1))
                put(done=sum(done_of(x) for x in names), current=n)
                reason = str(pr.get("stop_reason") or "")
                if c == 0 or (c == 1 and reason.startswith("max_new")):
                    continue
                status, code = ("stopped", 1) if c == 1 else ("error", c if c in (2, 3) else 2)
                put(stop_reason=f"{n}: {reason or pr.get('error')}")
                return code
    except KeyboardInterrupt:
        status, code = "interrupted", 1
    except BaseException as e:                   # noqa: BLE001
        status, code = "error", 2
        put(error=f"{type(e).__name__}: {e}")
        raise
    finally:
        put(status=status)
    return code


# ---------------------------------------------------------------- 読むだけ
def cond_dirs(a) -> list:
    if a.dir:
        return [pathlib.Path(p) for p in a.dir]
    base = ROOT / "outputs" / "v2eval" / a.experiment
    return [base / c for c in a.condition]


def cmd_metrics(a) -> int:
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "definition": "src/recovla/diag/rtc.py の冒頭（関門 R の指標。判定はしない）",
           "module_sha256": sha256_file(pathlib.Path(D.__file__)), "conditions": {}}
    for d in cond_dirs(a):
        r = D.condition_metrics(d, horizons=(30.0, None))
        res["conditions"][d.name] = r
        for h, v in r["by_horizon"].items():
            print(f"[metrics] {d.name} 区間 {h}: {json.dumps(v['summary'], ensure_ascii=False)}", flush=True)
    if a.out:
        p = pathlib.Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        print(f"[metrics] 書いた: {p}")
    return 0


def cmd_shadow_metrics(a) -> int:
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "definition": "src/recovla/diag/rtc.py の冒頭（影の推論）", "conditions": {}}
    for d in cond_dirs(a):
        trials = []
        for p in sorted(d.glob("trial_[0-9][0-9][0-9][0-9].json")):
            meta, z, rt, diag = D.load_record(p)
            if diag is None:
                raise SystemExit(f"{p}: diag の記録がない")
            trials.append(D.shadow_trial(meta, z, rt, diag, horizon_s=30.0))
        res["conditions"][d.name] = {"summary": D.shadow_summary(trials), "n_trials": len(trials)}
        print(f"[shadow] {d.name}: {json.dumps(res['conditions'][d.name], ensure_ascii=False)}", flush=True)
    if a.out:
        p = pathlib.Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def definitions_before_run() -> dict:
    """回す前に掲示する定義（文）。中身は src\\recovla\\diag\\rtc.py の今の規則のまま（数値はモジュールの定数から入れる）。"""
    z = D.SETTINGS["ZEROS"]
    zw = {f"d={d}": [round(float(x), 3) for x in D.prefix_weights("ZEROS", d, z["horizon"])[:8]] for d in (1, 4, 5)}
    return {
        "source": "src/recovla/diag/rtc.py の冒頭・trial_metrics・summarize・shadow_trial・shadow_summary（この掲示はコードの規則を文にした"
                  "もので、規則は変えていない）。configs/s4_gates.json gates.R・metric_definitions",
        "move_ratio_30s": {
            "definition": ("移動の比 = 最初に閉じる前（行動の時刻 < 最初に gripper_closed が真になった時刻）の、方策の行動のうち保持でない行の"
                           "水平の和を u に射影したもの ÷ 必要な移動（目標の立方体の水平位置（最初に閉じた時点）− 指先の水平位置（時刻 0））の"
                           " u 成分。u は根元（原点）→ 目標の立方体（最初に閉じた時点）の水平の向き。必要な移動の |u 成分| が "
                           f"{D.NEED_MIN_M * 1e3:.0f} mm 以下の試行は除く（W\\rtc\\a3_travel.py の ratio_cmd_need と同じ）"),
            "30s_version": ("主は 30 s 版: 最初の閉じが 30 s 以内（t_first_close <= 30 s）の試行だけを使う。区間が最初の閉じまでなので、"
                            "これで最初の 30 s に限った値になる。30 s より後に最初に閉じた試行と、閉じなかった試行は除き、件数を "
                            "n_first_close_after_horizon・n_no_close に出す。副は 60 s の記録全体（最初の閉じが 60 s 以内の試行）"),
            "aggregate": ("主は中央値（move_ratio_median）。平均（move_ratio_mean）は同じ試行の集合で並べるだけ（判定に使わない）。根拠: "
                          "s4_gates.json の move_ratio の aggregate は『出どころ W\\rtc\\a3_summary.json の集計の定義に合わせる』で、"
                          "a3_summary.json の ratio_cmd_need は [中央値, 第 1 四分位, 第 3 四分位, 件数]、基準値 A 0.99・E 0.78・C 1.01 は"
                          "中央値。R.c2（>= 0.95）は 30 s 版の中央値に当てる"),
            "code": "trial_metrics(meta, z, rt, horizon_s=30.0) の move_ratio、summarize の move_ratio_median・move_ratio_mean"},
        "seam_jump_30s": ("継ぎ目の速度の跳び（主は 30 s 版）: 10 Hz の行動の速さ v = a[:3]/0.1 の、続く 2 行がどちらも保持でなく塊の番号が"
                          "変わる所（両方の行の時刻 < 30 s）の |Δv| の試行ごとの平均 → 全試行の中央値。副は 60 s の記録全体"),
        "radial_gap": "半径方向の差は最初の閉じで決まり、制限時間によらない（主・副で同じ値）。集計は閉じた試行の中央値",
        "shadow": {
            "compare_with": ("影 = 誘導した推論と同じ観測・同じ雑音（雑音の生成器の状態を推論の前に戻して引き直す）で、RTC の引数"
                             "（inference_delay・prev_chunk_left_over）を渡さない推論（＝naive と同じ、誘導しない推論）。実行は誘導した側"
                             "（current_repro）で、影は記録だけ（戻り値・次の推論の前の塊は誘導した側のまま）"),
            "metric": (f"推論ごとに、行 0 から h 行（h = {'・'.join(str(h) for h in D.ROWS)}）の後処理の後の行動の水平の和の差 (影 − 誘導)·u [mm]"
                       "（u は観測の時刻の x_des → 目標の立方体の水平の向き。正は誘導した計画が短い）。使う推論: 影があり、観測の時刻が"
                       f"最初の閉じより前かつ 30 s 以内、|立方体 − x_des| >= {D.DIR_MIN_M * 1e3:.0f} mm のもの"),
            "h_rule": (f"集計は条件の全試行の推論をまとめた h ごとの中央値。「先の計画が {D.SHORT_MM:.0f} mm 以上短い」は、どれかの h の"
                       f"中央値 >= {D.SHORT_MM:.0f} mm（R.b1 の影の側）。実行した行（offset から s = {D.EXEC_S} 行）の差の中央値も並べる（判定に使わない）"),
            "code": "DiagPolicy.infer、shadow_trial(horizon_s=30.0)、shadow_summary"},
        "ZEROS": {
            "contents": (f"mode rtc、範囲 E = {z['horizon']}、減衰の形 ZEROS、上限 β = {z['max_guidance_weight']:g}。誘導の重みは前の "
                         "min(d, E) = d 行だけ 1、ほかは 0（d = 方策に渡す inference_delay = min(直前の推論で実際にかかった行の数, s − 1)、"
                         "s = 6、d <= 5 < E なので重みのある行は d 行）。範囲 E は重みに効かない（前の塊の残りは normalize_left_over で E 行に"
                         "揃えて渡すが、重み 0 の行は誘導に効かない）。範囲と上限は文書に書かれていないので、論文の式どおりの設定（E 44・β 5）と"
                         "減衰の形だけが違うようにそろえた（実装役の決定）"),
            "weights_rows0_7": zw},
    }


def cmd_settings(a) -> int:
    rows = {}
    for n, s in D.SETTINGS.items():
        r = dict(s)
        if s["mode"] == "rtc":
            r["weights_d4_rows0_9"] = [round(float(x), 3) for x in D.prefix_weights(s["schedule"], 4, s["horizon"])[:10]]
            r["weights_d4_exec_rows4_9"] = r["weights_d4_rows0_9"][4:10]
        rows[n] = r
    out = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "settings": rows, "candidates": list(D.CANDIDATES),
           "definitions_before_run": definitions_before_run(),
           "common": "s = 6、d_init = 4、inference_delay = min(d_est, 5)、10 段、ヤコビアンなし（lerobot 0.6.1）、R1v3、安全フィルタなし、60 s",
           "paper_check": "lerobot EXP（E = 44）と arXiv 2506.07339 の式 5（H = 50、s = 6）の重みの最大差 "
                          f"{max(float(np.max(np.abs(D.prefix_weights('EXP', d, 44) - D.paper_weights(d)))) for d in range(1, 6)):.2e}（d = 1〜5）",
           "module_sha256": sha256_file(pathlib.Path(D.__file__)), "script_sha256": sha256_file(pathlib.Path(__file__))}
    text = json.dumps(out, ensure_ascii=False, indent=1)
    print(text)
    if a.out:
        p = pathlib.Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return 0


def cmd_plan(a) -> int:
    py = r".venv\Scripts\python.exe scripts\98_s4_d_rtc.py"
    print("本番（回す前に settings を掲示し、台帳で帯 190200〜190209・190220〜190223 の未使用を確かめる）:")
    print(f"  {py} rotate --settings all --block-seeds 1          # 6 設定 x 30 試行、1 種（3 試行）ごとに交互、1 プロセス")
    print("  3 本並行にするなら（同じ時期に回す）:")
    for grp in (("current_repro", "naive"), ("paper_formula_range44_cap5", "range40_cap5"), ("range10_cap5", "ZEROS")):
        print(f"  {py} rotate --settings {','.join(grp)} --block-seeds 1")
    print(f"  {py} run --setting current_repro --shadow        # 影の推論 12 試行（190220〜190223）")
    print(f"  監視: .venv\\Scripts\\python.exe scripts\\96_s4_ops.py wait --progress outputs\\s4\\d_rtc\\rotate_<タグ>.progress.json")
    print(f"  集計: {py} metrics --experiment S4DRTC --condition {' '.join(D.SETTINGS)} --out outputs\\s4\\d_rtc\\metrics.json")
    print(f"        {py} shadow-metrics --experiment S4DRTC --condition shadow_current_repro --out outputs\\s4\\d_rtc\\shadow.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="1 設定を回す（96 の続きの引数はそのまま渡る: --dry-run --max-new --accept-env-change など）")
    p.add_argument("--setting", required=True, choices=list(D.SETTINGS))
    p.add_argument("--shadow", action="store_true", help="影の推論を足す（既定の帯 190220〜190223）")
    p.add_argument("--experiment", default=DEFAULTS["experiment"])
    p.add_argument("--condition", default=None, help="既定は設定名（影は shadow_<設定名>）")
    p.add_argument("--trials", default=None, help="natural:<先頭>:<数>。既定は割り当ての帯")
    p.add_argument("--allow-other-band", action="store_true")
    p = sub.add_parser("rotate", help="複数の設定を同じ種で、種の塊ごとに交互に回す（1 プロセス、方策は読み込んだまま）")
    p.add_argument("--settings", default="all", help="all か、カンマ区切りの設定名")
    p.add_argument("--block-seeds", type=int, default=1, help="交互にする単位の種の数（1 種 = 3 試行）")
    p.add_argument("--block-trials", type=int, default=0, help="交互にする単位を試行の数で（0 なら --block-seeds。smoke 用）")
    p.add_argument("--rounds", type=int, default=0, help="この巡の数で止める（0 は最後まで。smoke 用）")
    p.add_argument("--experiment", default=DEFAULTS["experiment"])
    p.add_argument("--trials", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--allow-other-band", action="store_true")
    for name in ("metrics", "shadow-metrics"):
        p = sub.add_parser(name)
        p.add_argument("--experiment", default=DEFAULTS["experiment"])
        p.add_argument("--condition", nargs="+", default=[])
        p.add_argument("--dir", nargs="+", default=None, help="条件のフォルダを直接（--experiment/--condition の代わり）")
        p.add_argument("--out", default=None)
    p = sub.add_parser("settings")
    p.add_argument("--out", default=None)
    sub.add_parser("plan")
    return ap


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                            # noqa: BLE001
        pass
    ap = build_parser()
    a, extra = ap.parse_known_args(argv)
    if a.cmd in ("run", "rotate"):
        return (cmd_run if a.cmd == "run" else cmd_rotate)(a, extra)
    if extra:
        ap.error(f"知らない引数: {extra}")
    return {"metrics": cmd_metrics, "shadow-metrics": cmd_shadow_metrics, "settings": cmd_settings, "plan": cmd_plan}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
