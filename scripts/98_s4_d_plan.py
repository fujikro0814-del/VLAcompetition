"""段階 4 束 1（学習なしの診断）の回す計画と一括の包み（担当 D、運用役）。この道具そのものは GPU を使わない（子のプロセスが使う）。

使い方（作業場所 C:\\PAI\\recovery_vla）:
    .venv\\Scripts\\python.exe scripts\\98_s4_d_plan.py plan [--with-ex] [--out outputs\\s4\\bundle1_plan_queue.json]   # 計画の JSON と表
    .venv\\Scripts\\python.exe scripts\\98_s4_d_plan.py smoke-plan                                              # smoke の計画（2 条件）
    .venv\\Scripts\\python.exe scripts\\98_s4_d_plan.py bundle [--plan <計画>] [--lanes 3] [--dry-run] [--only 仕事,...]
    .venv\\Scripts\\python.exe scripts\\98_s4_d_plan.py status [--plan <計画>]                                   # 状態の要約（読むだけ）
    .venv\\Scripts\\python.exe scripts\\98_s4_d_plan.py wait-cmd [--plan <計画>]                                 # 監視役のコマンド
  ふだんは outputs\\s4\\runs\\run_bundle1.ps1 が bundle を起こす。切り離して起動したら、必ず監視役を背景で付ける:
    .venv\\Scripts\\python.exe scripts\\96_s4_ops.py wait --progress outputs\\s4\\runs\\bundle1_status.json
  （bundle1_status.json は 96_s4_resume.py の progress.json と同じ鍵 status・pid・proc_create_time・updated・done・total を持つので、
   wait がそのまま読む。終了コードも同じ割り当て: 0 全部完了、1 止めた、2 失敗が残った、3 異常終了、4 途絶）。

計画の 2 つの表:
  "conditions"  記録の単位（関門が読む条件。outputs\\v2eval\\<実験>\\<条件>\\ と progress.json の場所、帯・試行数・モデル・制限時間・見込み）。
  "jobs"        包みが起こす子のプロセスの単位（1 つの命令。いくつかの条件をまとめることがある）。命令は担当 A・B・C のスクリプトの
                冒頭の使い方にそろえた（計画の "adapters" に、見た形と SHA-256 を残す）。担当 A の D-RTC（rotate）と担当 C の D-復帰は、
                比べる組を 1 つのプロセスの中で種の塊ごとに交互に回すので、その命令をそのまま 1 つの仕事にする。D-E7・D-単発の開始・
                移植は 1 条件 1 仕事で、包みが子に --max-new <塊> を渡して種の塊ごとに交互に回す。

サブコマンドと、読むもの・書くもの:
  plan        読む: configs\\s4_gates.json（帯・制限時間・計算量の倍率）、docs\\目標書_段階4.md（SHA-256 を掲示板 0153 と照らすだけ）、
              outputs\\v2eval\\S4K\\K1・K2\\run.json（物差し K の実測の速さ）、診断のスクリプト（あるかどうかと SHA-256 だけ）。
              書く: outputs\\s4\\bundle1_plan_queue.json（名前の queue は 97_s4_ledger_check.py に種を使用と数えさせないため。
              PLAN_PATH の注）、標準出力に表。
  smoke-plan  書く: outputs\\s4\\bundle1_smoke_plan.json（96_s4_resume.py run の小さな 2 条件。種 44490・44491＝担当 D の小帯）。
  bundle      読む: 計画、状態（続きから回すとき）、各仕事の progress.json。
              書く: 状態 outputs\\s4\\runs\\bundle1_status.json、ログ outputs\\s4\\runs\\bundle1.log、
              仕事ごとのログ outputs\\s4\\runs\\bundle1_logs\\<仕事>.log。子（各診断のスクリプト）は outputs\\v2eval\\<実験>\\<条件>\\ に書く。
              --env-record のときだけ、始めに 96_s4_ops.py env-record を回す（outputs\\s4\\env_record.json を更新する）。
  status / wait-cmd  読むだけ。

一括の包み（bundle）の動き:
  1. 「枠」（--lanes、既定 3）の数まで子のプロセスを同時に動かす。1 つの塊（または仕事）が終わったら次を始める。学習は回っていない前提
     （学習中は評価 1 本まで＝目標書_段階4.md 第 11 節 2。そのときは --lanes 1）。
  2. 順番: 同じ群（D-E7 の 5 腕、D-復帰の 4 仕事など、比べる組）の残りの見込みが長い群から先に（長い条件から先に）。群の中は、済んだ
     塊の少ない仕事から（＝種の塊ごとに交互。目標書_段階4.md 第 4 節「比べる組は同じ種・同じ時期に、種の塊ごとに交互に回す」）。
     塊は子に --max-new <塊の試行数> を渡して作る（96_s4_resume.py と同じ引数。塊を回し終えた子は終了コード 1・stop_reason
     "max_new:N" で終わり、最後の塊は 0 で終わる）。--no-interleave なら塊に分けず、長い仕事から先に 1 つずつ流す。
  3. 1 つの仕事が失敗しても（終了コード 2・3、スクリプトが無いなど）、ほかは続ける。失敗した仕事に依存する仕事（X2 の測定）は、
     その回は回さない。
  4. 続きから: 同じコマンドをもう一度打てば、完了した仕事を飛ばし、ほかは子が自分で続きから回す（子は完全な試行を飛ばす）。
     前の回の途中で包みが消えた（再起動など）とき、状態が running の仕事は、子が生きていれば引き取って終わりを待ち、
     生きていなければ続きから回し直す。失敗・止めた仕事も次の回で回し直す。
  5. 止める合図: --stop-file（既定 outputs\\s4\\runs\\STOP_BUNDLE1）を見たら、新しい塊を始めず、今の塊を終えてから止まる
     （status=stopped、終了コード 1）。outputs\\s4\\STOP（子も今の試行の後で止まる）も新しい塊を始めない合図として見る。
  6. 空きメモリ: 物理 --min-free-gb（既定 12）未満かコミット --min-commit-free-gb（既定 6）未満なら、新しい塊を始めない
     （運用の決まり「空きメモリ 12 GB 未満なら動かさない」。並行の本数で割らない＝docs\\stage4\\ops.md 1-1）。動いている子が
     1 本もないまま --mem-timeout-min（既定 120 分）を超えたら止まる（status=memory_timeout、終了コード 1）。
     子を始めた直後は子のメモリがまだ増えていないので、次の子は --stagger-s（既定 90 s）空けてから始める。
  7. 子の記録の形・続きから回す仕組み・環境の食い違いでの停止は、子（96_s4_resume.py か、それを importlib で読み込む診断のスクリプト）の
     もの。包みは子の終了コードと progress.json（pid が子と一致するものだけ）で、塊の終わり・完了・失敗を見分ける。

包みが子（診断のスクリプト）に求めること（計画の "interface"。食い違いは批判役が見る）:
  同じ命令をもう一度打てば続きから回る。終了コードは 96_s4_resume.py と同じ（0 全部そろった、1 途中で止まった、2 エラー、
  3 引数・前提の食い違い）。progress.json を書くなら pid・status・stop_reason を 96_s4_resume.py と同じ鍵で。塊に分ける仕事は
  96_s4_resume.py の --max-new を受け付ける。--dry-run。回す前に掲示する定義（outputs\\s4\\d_start\\definitions.json など。
  計画の "pre_steps"）が無い仕事は、包みが始めない（prereq_missing の失敗）。

結果を見る前に決めてある項目（この計画は成績を読まない）: 条件の一覧と名前・帯・試行数・モデル・制限時間（目標書_段階4.md 4-2・3-1・8 節、
  s4_gates.json）、仕事のまとめ方と塊の大きさ、順番の規則（群の残りの見込みの長い順、群の中は交互）、失敗・止める合図・メモリの規則、
  見込みの計算（s4_gates.json の budget_notes と同じ式。K の実測で換算した列を並べる）。見た後に決めるのは EX の腕を足すか
  （candidate_selection の行 1・行 3 の矛盾の枝で機械的に決まる。足すときは plan --with-ex で計画を作り直し、同じ状態で続きから回す）。
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
S4 = ROOT / "outputs" / "s4"
RUNS = S4 / "runs"
GATES = ROOT / "configs" / "s4_gates.json"
CHARTER = ROOT / "docs" / "目標書_段階4.md"
V2EVAL = ROOT / "outputs" / "v2eval"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
EXPERIMENT = "S4D1"                                  # 既定の実験名（使うのは _cond の既定だけ。各診断はスクリプトの既定の実験名を使う）
# 計画の置き場。名前に queue を入れ、"queue" の鍵を持たない形にする: scripts\97_s4_ledger_check.py（書き換えない）は名前に queue を
# 含む json を scan_queue（"queue" の鍵の --trials だけを「実行の記録なし」の弱い証拠として別に数える）に回し、使用には数えない。
# 旧名 bundle1_plan.json だと tier derived の「使用」に数えられ、本番の帯がすべて使用ありになっていた（10/08 の検査役の指摘）。
# 旧版は outputs\s4\superseded\ へ移した（同じ理由で名前に queue を入れた）。run_bundle1.ps1 の -Plan の既定もこれにそろえる。
PLAN_PATH = S4 / "bundle1_plan_queue.json"
SMOKE_PLAN_PATH = S4 / "bundle1_smoke_plan.json"
# 掲示板 0153 に掲示した SHA-256（コミット f4a15bd の版）。計画に照合の結果を残す（違っても止めない。報告に書く）
BOARD_0153_SHA = {"docs/目標書_段階4.md": "afa069552fbcea7573706e59be3678b8d941da0dd1baf0eece8532e3a961860b",
                  "configs/s4_gates.json": "cf2e8c96a205bf10126bdf647bdbf2e8b9d5a315ebbe881c03bafac82b12940d"}
FINAL_STATUS = {"done": 0, "stopped": 1, "interrupted": 1, "memory_timeout": 1, "error": 2}    # 96_s4_ops.FINAL_STATUS と同じ
DEFAULT_FACTORS = {"A_nat": 1.215, "A_P1": 1.432, "B_P1": 1.795, "A_P2": 1.803, "A_P3": 1.954}   # s4_gates budget_notes.factor_60_over_30


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_ops():
    return _load(ROOT / "scripts" / "96_s4_ops.py", "s4_ops_d")


def _now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_atomic(path: pathlib.Path, text: str, tries: int = 8) -> bool:
    """96_s4_resume._write_atomic と同じ（読んでいる相手がいれば少し待つ）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.2 * (k + 1))
    return False


def _rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _abs(s) -> pathlib.Path:
    p = pathlib.Path(s)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- 見込みの計算（s4_gates.json の budget_notes と同じ式）
# 速さは outputs\s4\rules\budget60_e7_30_calc.py の R（budget_final.py。3 本並行の 1 プロセスあたり、30 s の記録の速さ）。
# 単発は 60 s の倍率（factor_60_over_30）を掛け、E7 は段階 3 と同じ持ち時間なので倍率を掛けない（改訂 2）。
RATES = {"nat_per_h": 93.7, "p1_per_h": 76.6, "p23_per_h": 64.5, "e7_s": 262.0, "e7_par_slow": 1.25, "rtc": 1.44, "pack": 0.95,
         "xpl_critique_ph": 1.3,
         "source": "outputs\\s4\\rules\\budget60_e7_30_calc.py の R（3 本並行の 1 プロセスあたり、段階 3 の 30 s の記録・旧ドライバ）"}
X2_GUESS_PH = {"gen": 0.10, "measure": 0.05}     # 推測（final.md の「GPU 数分」。束 1 の予算 36.8 には入っていない）


def load_factors(gates: dict) -> dict:
    f = dict(gates["budget_notes"]["factor_60_over_30"])
    return {k: float(f[k]) for k in DEFAULT_FACTORS}


def budget_ph(cost: str, n: int, F: dict) -> float:
    """1 条件の見込みのプロセス時間（予算の式）。n は試行数（E7 は本数）。"""
    R = RATES
    if cost == "nat":
        return n / R["nat_per_h"] * F["A_nat"]
    if cost == "rtc":
        return n / R["nat_per_h"] * F["A_nat"] * R["rtc"]
    if cost == "e7":
        return n * R["e7_s"] / 3600 * R["e7_par_slow"]
    if cost == "p1_R":
        return n / R["p1_per_h"] * F["A_P1"]
    if cost == "p1_N":
        return n / R["p1_per_h"] * F["B_P1"]
    if cost == "p23":                                 # 落下・置き損ね（A の P2・P3 の倍率の平均。N にも同じ倍率＝critique と同じ）
        return n / R["p23_per_h"] * (F["A_P2"] + F["A_P3"]) / 2
    if cost == "xpl":                                 # 移植の腕（critique の 1.3 を自然の 60 s の倍率で。120 試行で割り振る）
        return R["xpl_critique_ph"] * F["A_nat"] * n / 120
    if cost in ("x2_gen", "x2_measure"):
        return X2_GUESS_PH[cost[3:]]
    raise ValueError(cost)


def k_rate() -> dict:
    """物差し K の実測の速さ（60 s、新ドライバ、2 本並行）。run.json の n / wall_s。"""
    out = {"runs": {}}
    for c in ("K1", "K2"):
        p = V2EVAL / "S4K" / c / "run.json"
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            out["runs"][c] = {"n": j["n"], "wall_s": j["wall_s"], "per_h": round(j["n"] / (j["wall_s"] / 3600), 2)}
        except Exception as e:                       # noqa: BLE001
            out["runs"][c] = {"error": f"{type(e).__name__}: {e}"}
    rates = [v["per_h"] for v in out["runs"].values() if "per_h" in v]
    out["per_h"] = round(sum(rates) / len(rates), 2) if rates else 88.0
    out["source"] = "outputs\\v2eval\\S4K\\K1・K2\\run.json（n / wall_s。wall_s は試行の実時間の和、モデルの読み込みを含む）"
    out["note"] = "自然の 60 s・R1v3・naive・2 本並行。3 本並行では遅くなりうる（未測定）"
    return out


# ---------------------------------------------------------------- 計画: 条件（記録の単位）
def _band(gates: dict, bid: str) -> dict:
    for a in gates["bands"]["allocations"]:
        if a["id"] == bid:
            return a
    raise KeyError(bid)


def _limits(kind: str):
    if kind == "run":
        return {"time_limit_s": 60, "scored_at_s": {"primary": 30, "secondary": 60, "descriptive": 45},
                "induced_denominator": "誘発の試行は induce.t_established <= L の試行（L = 30・60）"}
    if kind == "task":
        return {"step_timeout_s": 30, "retry": 1, "task_time_limit_s": 200, "scored": "3 個とも（真値）がそのまま主（段階 3 と同じ定義）"}
    return None


def _cond(cid, *, diag, gate, model, trials, seeds, n, band, cost, group, F, kind="run", experiment=EXPERIMENT, record_condition=None,
          enabled=True, notes="", equivalent_96=None):
    rc = record_condition or cid
    out = pathlib.Path("outputs") / "v2eval" / experiment / rc
    rec = {"run": "trial_NNNN.json・.npz・runtime_NNNN.json（96_s4_resume.py run と同じ形）＋診断の条件の欄",
           "task": "run_NNNN.json・.npz・run_NNNN_runtime.json（96_s4_resume.py task と同じ形）＋診断の条件の欄"}.get(
        kind, "診断のスクリプトが決める（progress.json は同じ鍵で書く）")
    return {"id": cid, "diag": diag, "gate": gate, "group": group, "kind": kind, "enabled": enabled,
            "experiment": experiment, "record_condition": rc, "model": model, "trials_spec": trials, "seeds": list(seeds),
            "trials": n, "band": band, "time_limits": _limits(kind), "cost_kind": cost, "est": {"budget_ph": round(budget_ph(cost, n, F), 3)},
            "out_dir": str(out), "progress": str(out / "progress.json"), "records": rec, "equivalent_96": equivalent_96, "notes": notes,
            "job": None}


def _eq96(experiment, cond, model, trials, mode, induce=None):
    """96_s4_resume.py run で同じ試行になる命令（診断の欄は付かない。dry-run で帯の試行数を確かめるのに使える）。"""
    a = ["scripts\\96_s4_resume.py", "run", "--experiment", experiment, "--condition", cond, "--model", model, "--trials", trials,
         "--mode", mode, "--exec-interval", "6", "--no-safety", "--time-limit-s", "60"]
    return a + (["--induce", induce] if induce else [])


def build_conditions(gates: dict, F: dict, with_ex: bool) -> list:
    C = []
    # D-RTC（関門 R）: 6 設定 x 30 試行、同じ種。設定の名前と順は s4_gates.json gates.R.runs.settings。記録は担当 A の
    # 98_s4_d_rtc.py の既定（実験 S4DRTC、条件名＝設定名）
    b = _band(gates, "D_RTC")
    settings = gates["gates"]["R"]["runs"]["settings"]
    assert settings == ["current_repro", "paper_formula_range44_cap5", "range40_cap5", "range10_cap5", "ZEROS", "naive"], settings
    tr = f"natural:{b['range'][0]}:10"
    for s in settings:
        mode = "naive" if s == "naive" else "rtc"
        C.append(_cond(f"RTC_{s}", diag="D-RTC", gate="R", model="R1v3", trials=tr, seeds=b["range"], n=30, band="D_RTC", cost="rtc",
                       group="D-RTC", F=F, experiment="S4DRTC", record_condition=s,
                       equivalent_96=_eq96("S4DRTC", s, "R1v3", tr, mode) if s in ("naive", "current_repro") else None,
                       notes="ZEROS・paper_formula の中身は実装役が回す前に掲示（s4_gates R.known_issue。98_s4_d_rtc.py settings）"))
    b = _band(gates, "D_RTC_shadow")
    C.append(_cond("RTC_shadow", diag="D-RTC の影の推論", gate="R（R.b1）", model="R1v3", trials=f"natural:{b['range'][0]}:4",
                   seeds=b["range"], n=12, band="D_RTC_shadow", cost="rtc", group="D-RTC-shadow", F=F, experiment="S4DRTC",
                   record_condition="shadow_current_repro",
                   notes="誘導しない塊を同じ観測・同じ雑音で並べて記録。実行は current_repro（98_s4_d_rtc.py の既定）"))
    # X2（関門 R の R.b1）: 学習に使っていないエキスパートの記録を作り、R1v3 を開ループで当てる。記録は担当 A の 98_s4_x2.py の
    # 既定（outputs\s4\x2\gen\X2_gen・outputs\s4\x2\eval_X2_gen_R1v3。どちらも 96 と同じ鍵の progress.json を書く）
    b = _band(gates, "X2_gen")
    c = _cond("X2_gen", diag="X2 の生成", gate="R（R.b1）", model=None, trials=f"{b['range'][0]}:20", seeds=b["range"], n=20,
              band="X2_gen", cost="x2_gen", group="X2", F=F, kind="x2", notes="学習用の帯（学習には入れない）。見込みは推測（数分）")
    c.update(out_dir="outputs\\s4\\x2\\gen\\X2_gen", progress="outputs\\s4\\x2\\gen\\X2_gen\\progress.json",
             experiment="outputs\\s4\\x2", record_condition="gen\\X2_gen")
    C.append(c)
    c = _cond("X2_measure", diag="X2 の測定", gate="R（R.b1）", model="R1v3", trials=f"{b['range'][0]}:20", seeds=b["range"], n=20,
              band="X2_gen", cost="x2_measure", group="X2", F=F, kind="x2", notes="10・20・30・40 行先の予測の不足。GPU 数分（推測）")
    c.update(out_dir="outputs\\s4\\x2\\eval_X2_gen_R1v3", progress="outputs\\s4\\x2\\eval_X2_gen_R1v3\\progress.json",
             experiment="outputs\\s4\\x2", record_condition="eval_X2_gen_R1v3")
    C.append(c)
    # D-E7（関門 T）: E0 を同じ 40 種で 2 回、EH・ES・EO。EX は条件つき（candidate_selection の行 1・行 3 の矛盾の枝）
    b = _band(gates, "D_E7")
    notes = {"E0": "今の実行器。E7_E0_r1・r2 は同じ 40 種（回し直しのぶれ d_E0・rho_hat）",
             "EH": "学習の home の関節角へ関節空間で制限層を通して戻す", "ES": "待機位置の始めの姿勢（z 0.314）へ戻す",
             "EO": "緑→赤→青の順。計画役 Haiku 5.5 を 1 回呼びキャッシュ（API が要る）",
             "EX": "条件つき。手順の切り替えで実行系を作り直す（plan --with-ex で有効にする）"}
    for cid, rc, arm, en in (("E7_E0_r1", "E0_run1", "E0", True), ("E7_E0_r2", "E0_run2", "E0", True), ("E7_EH", "EH", "EH", True),
                             ("E7_ES", "ES", "ES", True), ("E7_EO", "EO", "EO", True), ("E7_EX", "EX", "EX", with_ex)):
        c = _cond(cid, diag="D-E7", gate="T", model="R1v3", trials=f"{b['range'][0]}:40", seeds=b["range"], n=40, band="D_E7",
                  cost="e7", group="D-E7", F=F, kind="task", enabled=en, notes=notes[arm], experiment="S4DE7", record_condition=rc)
        c["arm"] = arm
        C.append(c)
    # D-単発の開始（関門 S）: 開始 3 x 先客 2、33 試行/条件、6 条件で同じ種
    b = _band(gates, "D_single_start")
    fac = gates["gates"]["S"]["runs"]["factorial"]
    for pose in fac["start_pose"]:
        for prior in fac["prior_cube"]:
            c = _cond(f"ST_{pose}_{prior}", diag="D-単発の開始", gate="S", model="R1v3", trials=f"natural:{b['range'][0]}:11",
                      seeds=b["range"], n=33, band="D_single_start", cost="nat", group="D-start", F=F, experiment="S4DSTART",
                      record_condition=f"{pose}_{prior}",
                      notes="壁際の定義・格子への移し方は回す前に掲示（98_s4_d_start.py defs → outputs\\s4\\d_start\\definitions.json）")
            c["start"], c["prior"] = pose, prior
            C.append(c)
    # 移植の腕（S を確かめる。8-3）: 20 状態 x 3 通り x 2 回。種 190420+i が V3S3 E7_R1v3 の run_{i:04d}
    b = _band(gates, "D_transplant")
    for arm in ("XPL_as", "XPL_home", "XPL_grid"):
        for r in (1, 2):
            short = arm.split("_")[1]
            c = _cond(f"{arm}_r{r}", diag="移植の腕", gate="S（XPL）", model="R1v3", trials=f"xpl:{b['range'][0]}:20",
                      seeds=b["range"], n=20, band="D_transplant", cost="xpl", group="XPL", F=F, experiment="S4XPL",
                      record_condition=f"{short}_r{r}",
                      notes="目標は緑。2 番目の始めの添字・空き位置の決め方は回す前に掲示（definitions.json）。状態は V3S3\\E7_R1v3（読むだけ）")
            c["arm"], c["xpl_arm"], c["rep"] = arm, short, r
            C.append(c)
    # D-復帰（関門 C）: R1v3・N1v3 x 4 通り、50 種、60 s で回し 30 s（主）・60 s（副）で採点。記録は担当 C の
    # 98_s4_d_recovery.py の既定（実験 S4DREC、条件名 <モデル>_<通り>）
    b = _band(gates, "D_recovery")
    tr = f"induced:{b['range'][0]}:50"
    induce_of = {"fall_as_is": "P2", "fall_with_hold": None, "misplace": "P3", "grasp_failure": "P1"}
    for model in gates["gates"]["C"]["runs"]["models"]:
        for v in gates["gates"]["C"]["runs"]["variants"]:
            cost = ("p1_R" if model == "R1v3" else "p1_N") if v == "grasp_failure" else "p23"
            c = _cond(f"RC_{model}_{v}", diag="D-復帰", gate="C" if v == "fall_with_hold" else "C（報告）", model=model, trials=tr,
                      seeds=b["range"], n=50, band="D_recovery", cost=cost, group="D-recovery", F=F, experiment="S4DREC",
                      record_condition=f"{model}_{v}",
                      equivalent_96=_eq96("S4DREC", f"{model}_{v}", model, tr, "naive", induce_of[v]) if induce_of[v] else None,
                      notes="落下で手を止める版（診断専用、誘発の版は 98_s4_d_recovery.py が記録）" if v == "fall_with_hold"
                      else f"誘発は段階 3 と同じ {induce_of[v]}")
            c["variant"] = v
            C.append(c)
    for c in C:                                       # 帯の確かめ（種が割り当ての範囲に入っていること）
        lo, hi = _band(gates, c["band"])["range"]
        assert lo <= c["seeds"][0] <= c["seeds"][1] <= hi, c["id"]
    return C


# ---------------------------------------------------------------- 計画: 仕事（包みが起こす命令の単位）
# observed: 担当 A・B・C のスクリプトの冒頭の使い方（10/08 05:46〜05:50 の版）を読んで、その命令をそのまま使う。
#   スクリプトが変わったら、ここを直して plan を作り直す（計画に各スクリプトの SHA-256 を残すので、変わったかは比べられる）。
# 塊: D-E7・D-単発の開始・移植は、スクリプトの中で交互にしないので、包みが --max-new <塊> を渡して種の塊ごとに交互に回す
#   （どれも残りの引数を 96_s4_resume.py にそのまま渡す。--max-new は 96 の引数）。D-RTC（rotate）と D-復帰は中で交互にする。
ADAPTERS = {
    "D-RTC": {"status": "observed", "script": "scripts\\98_s4_d_rtc.py", "owner": "担当 A",
              "seen": "rotate [--settings all] [--block-seeds 1] [--tag]（6 設定を 1 プロセスで種の塊ごとに交互）、"
                      "監視 outputs\\s4\\d_rtc\\rotate_<タグ>.progress.json、既定の実験 S4DRTC・条件名＝設定名。settings で中身を掲示"},
    "D-RTC-shadow": {"status": "observed", "script": "scripts\\98_s4_d_rtc.py", "owner": "担当 A",
                     "seen": "run --setting current_repro --shadow（既定の条件 shadow_current_repro、帯 natural:190220:4）"},
    "X2": {"status": "observed", "script": "scripts\\98_s4_x2.py", "owner": "担当 A",
           "seen": "gen [--seeds 44404:20] [--name X2_gen]（outputs\\s4\\x2\\gen\\<名前>、種ごとの part_<種> で続きから）→ "
                   "eval [--name X2_gen] [--model R1v3]（outputs\\s4\\x2\\eval_<名前>_<モデル>）。どちらも --dry-run・"
                   "--accept-env-change・--max-new と、96 と同じ鍵の progress.json（出力のフォルダ）を持つ"},
    "D-E7": {"status": "observed", "script": "scripts\\98_s4_d_e7.py", "owner": "担当 B",
             "seen": "run --arm <E0|EH|ES|EO|EX> --experiment S4DE7 --condition <E0_run1|E0_run2|EH|ES|EO> --model R1v3 --trials 190300:40"
                     "（残りは 96_s4_resume.py task の引数。--exec-interval 6・--no-safety はスクリプトが入れる）。defs で定義を掲示"},
    "D-start": {"status": "observed", "script": "scripts\\98_s4_d_start.py", "owner": "担当 B",
                "seen": "start --start <開始> --prior <先客> --experiment S4DSTART --condition <名前> --model R1v3 --trials natural:190400:11"
                        "（残りは 96 run の引数）。outputs\\s4\\d_start\\definitions.json が無ければ止まる（defs で作る）"},
    "XPL": {"status": "observed", "script": "scripts\\98_s4_d_start.py", "owner": "担当 B",
            "seen": "xpl --arm <as|home|grid> --rep <1|2> --experiment S4XPL --condition <腕>_r<回> --model R1v3"
                    "（種は --seed-base 190420 + 状態の番号、--trials は自動）。definitions.json が要る"},
    "D-recovery": {"status": "observed", "script": "scripts\\98_s4_d_recovery.py", "owner": "担当 C",
                   "seen": "run --variants <通り> --models R1v3,N1v3 --block 10（R と N を種の塊ごとに交互）、"
                           "監視 outputs\\s4\\d_recovery\\progress_<実験>_<モデル>_<通り>.json、既定の実験 S4DREC"},
}
BLOCK = {"D-E7": 5, "D-start": 6, "XPL": 5}            # 包みが切る塊（試行数）。E7 は 5 本、単発は 2 種 x 3 色、移植は 5 状態
# 回す前に掲示する定義（結果を見る前に固定するもの。無ければ包みはその仕事を始めない＝prereq_missing の失敗）
PREREQ = {"D-start": ["outputs\\s4\\d_start\\definitions.json"], "XPL": ["outputs\\s4\\d_start\\definitions.json"]}
PRE_STEPS = [
    {"cmd": ".venv\\Scripts\\python.exe scripts\\98_s4_d_rtc.py settings --out outputs\\s4\\d_rtc\\settings.json",
     "writes": "outputs\\s4\\d_rtc\\settings.json（6 設定の中身と、回す前に掲示する定義＝移動の比の 30 s 版・影の推論・ZEROS）"},
    {"cmd": ".venv\\Scripts\\python.exe scripts\\98_s4_d_e7.py defs", "writes": "outputs\\s4\\d_e7\\definitions.json（腕の定義）"},
    {"cmd": ".venv\\Scripts\\python.exe scripts\\98_s4_d_start.py defs", "writes": "outputs\\s4\\d_start\\definitions.json（壁際・格子・移植の添字）"},
    {"cmd": "（掲示）上の 3 つの SHA-256 を掲示板に出す", "writes": "-"},
]


def _job(jid, covers, argv, progress, *, group, block=None, after=(), adapter, needs_api=False, notes="", requires=()):
    return {"id": jid, "covers": [c["id"] for c in covers], "group": group, "adapter": adapter, "argv": argv,
            "script": argv[0], "cmd": ".venv\\Scripts\\python.exe " + subprocess.list2cmdline(argv),
            "progress": progress, "block_trials": block, "after": list(after), "needs_api": needs_api, "requires": list(requires),
            "enabled": all(c["enabled"] for c in covers), "trials": sum(c["trials"] for c in covers),
            "est": {"budget_ph": round(sum(c["est"]["budget_ph"] for c in covers), 3)}, "notes": notes}


def build_jobs(C: list) -> list:
    by = {c["id"]: c for c in C}
    J = []
    # D-RTC: 6 設定を 1 プロセスで種の塊（1 種 = 3 試行）ごとに交互（担当 A の rotate）
    rtc = [c for c in C if c["group"] == "D-RTC"]
    J.append(_job("J_RTC_rotate", rtc, ["scripts\\98_s4_d_rtc.py", "rotate", "--settings", "all", "--tag", "bundle1"],
                  "outputs\\s4\\d_rtc\\rotate_bundle1.progress.json", group="D-RTC", adapter="observed",
                  notes="6 設定を 1 つのプロセスで交互に（方策は読み込んだまま）。塊は 98_s4_d_rtc.py の --block-seeds（既定 1 種）"))
    J.append(_job("J_RTC_shadow", [by["RTC_shadow"]], ["scripts\\98_s4_d_rtc.py", "run", "--setting", "current_repro", "--shadow"],
                  by["RTC_shadow"]["progress"], group="D-RTC-shadow", adapter="observed"))
    # X2: 生成 → 開ループの推論（担当 A）
    g, m = by["X2_gen"], by["X2_measure"]
    J.append(_job("X2_gen", [g], ["scripts\\98_s4_x2.py", "gen", "--seeds", g["trials_spec"], "--name", "X2_gen"], g["progress"],
                  group="X2", adapter="observed"))
    J.append(_job("X2_measure", [m], ["scripts\\98_s4_x2.py", "eval", "--name", "X2_gen", "--model", "R1v3"], m["progress"],
                  group="X2", adapter="observed", after=["X2_gen"]))
    # D-E7: 1 腕 1 仕事、包みが --max-new 5 で塊ごとに交互（担当 B の run。制限時間・計画役は 96 task の既定＝30 s・200 s・s4）
    for c in [c for c in C if c["group"] == "D-E7"]:
        J.append(_job(c["id"], [c], ["scripts\\98_s4_d_e7.py", "run", "--arm", c["arm"], "--experiment", c["experiment"],
                                     "--condition", c["record_condition"], "--model", c["model"], "--trials", c["trials_spec"]],
                      c["progress"], group="D-E7", block=BLOCK["D-E7"], adapter="observed", needs_api=True, notes=c["notes"]))
    # D-単発の開始・移植（担当 B の start・xpl。制限時間は 96 run の既定 60 s、naive・6 行・フィルタなしはスクリプトが入れる）
    for c in [c for c in C if c["group"] == "D-start"]:
        J.append(_job(c["id"], [c], ["scripts\\98_s4_d_start.py", "start", "--start", c["start"], "--prior", c["prior"],
                                     "--experiment", c["experiment"], "--condition", c["record_condition"], "--model", c["model"],
                                     "--trials", c["trials_spec"]],
                      c["progress"], group="D-start", block=BLOCK["D-start"], adapter="observed", requires=PREREQ["D-start"]))
    for c in [c for c in C if c["group"] == "XPL"]:
        J.append(_job(c["id"], [c], ["scripts\\98_s4_d_start.py", "xpl", "--arm", c["xpl_arm"], "--rep", str(c["rep"]),
                                     "--experiment", c["experiment"], "--condition", c["record_condition"], "--model", c["model"]],
                      c["progress"], group="XPL", block=BLOCK["XPL"], adapter="observed", requires=PREREQ["XPL"]))
    # D-復帰: 通りごとに 1 仕事（R と N を種の塊 10 ごとに交互。担当 C の run）。4 仕事は別の枠で同じ時期に回る
    rc = [c for c in C if c["group"] == "D-recovery"]
    for v in [c["variant"] for c in rc if c["model"] == "R1v3"]:
        cov = [c for c in rc if c["variant"] == v]
        J.append(_job(f"J_RC_{v}", cov, ["scripts\\98_s4_d_recovery.py", "run", "--experiment", "S4DREC", "--trials", cov[0]["trials_spec"],
                                         "--variants", v, "--models", "R1v3,N1v3", "--block", "10"],
                      f"outputs\\s4\\d_recovery\\progress_S4DREC_R1v3-N1v3_{v}.json", group="D-recovery", adapter="observed"))
    for j in J:
        for cid in j["covers"]:
            by[cid]["job"] = j["id"]
    return J


def build_plan(with_ex: bool = False) -> dict:
    gates = json.loads(GATES.read_text(encoding="utf-8"))
    F = load_factors(gates)
    tl = gates["time_limits"]
    assert tl["trial_time_limit_s"] == 60 and tl["e7_step_attempt_timeout_s"] == 30 and tl["e7_overall_limit_s"] == 200, tl
    C = build_conditions(gates, F, with_ex)
    J = build_jobs(C)
    ad = {}
    for k, v in ADAPTERS.items():
        p = _abs(v["script"])
        ad[k] = dict(v, exists=p.is_file(), sha256=sha256_file(p) if p.is_file() else None)
    return finish_plan({"schema": "recovery_vla.s4_bundle1_plan/2", "name": "bundle1",
                        "experiment": "S4DRTC・S4DE7・S4DSTART・S4XPL・S4DREC・outputs\\s4\\x2（担当のスクリプトの既定）",
                        "written": _now_s(), "by": "担当 D（運用役）、scripts\\98_s4_d_plan.py plan", "with_ex": with_ex,
                        "pre_steps": PRE_STEPS, "conditions": C, "jobs": J, "adapters": ad, "factors_60_over_30": F, "rates": RATES,
                        "s4_gates_bundle1_total": gates["budget_notes"]["bundle1_total_process_hours"]})


def finish_plan(p: dict) -> dict:
    """見込み（予算と K の換算）・合計・3 本の枠の見込みの時間・監視の場所・既定の置き場を足す。"""
    kr = k_rate()
    nat60 = RATES["nat_per_h"] / p["factors_60_over_30"]["A_nat"]
    scale = nat60 / kr["per_h"]                       # 予算の自然 60 s の速さ ÷ K の実測の速さ
    for c in p["conditions"]:
        k_applies = c["kind"] == "run"
        c["est"]["k_scaled_ph"] = round(c["est"]["budget_ph"] * (scale if k_applies else 1.0), 3)
        c["est"]["basis"] = ("予算の式に、K の実測で換算（単発の試行）" if k_applies else
                             "E7 は速さを未測定（束 0 の E7 の速さ 190150〜 は未実行）なので予算のまま" if c["kind"] == "task" else "推測")
    byc = {c["id"]: c for c in p["conditions"]}
    for j in p["jobs"]:
        j["est"]["k_scaled_ph"] = round(sum(byc[x]["est"]["k_scaled_ph"] for x in j["covers"]), 3)
    en = [c for c in p["conditions"] if c["enabled"]]
    by = {}
    for c in en:
        g = by.setdefault(c["diag"], {"conditions": 0, "trials": 0, "budget_ph": 0.0, "k_scaled_ph": 0.0})
        g["conditions"] += 1
        g["trials"] += c["trials"]
        g["budget_ph"] = round(g["budget_ph"] + c["est"]["budget_ph"], 3)
        g["k_scaled_ph"] = round(g["k_scaled_ph"] + c["est"]["k_scaled_ph"], 3)
    tb = sum(c["est"]["budget_ph"] for c in en)
    tk = sum(c["est"]["k_scaled_ph"] for c in en)
    lanes = 3
    p["k_rate"] = kr
    p["k_scale"] = {"value": round(scale, 3), "rule": "単発の試行の予算 × (予算の自然 60 s の速さ 93.7/1.215 ÷ K の実測の速さ)",
                    "budget_nat60_per_h": round(nat60, 2), "k_per_h": kr["per_h"]}
    sim = simulate(p, lanes=lanes, interleave=True)
    p["totals"] = {"by_diag": by, "conditions": len(en), "trials": sum(c["trials"] for c in en), "jobs": sum(j["enabled"] for j in p["jobs"]),
                   "budget_ph": round(tb, 2), "k_scaled_ph": round(tk, 2),
                   "budget_ph_without_x2": round(sum(c["est"]["budget_ph"] for c in en if c["kind"] != "x2"), 2),
                   "wall_h_3lanes_pack95": {"budget": round(tb / lanes / RATES["pack"], 1), "k_scaled": round(tk / lanes / RATES["pack"], 1)},
                   "wall_h_3lanes_simulated": sim["makespan_h"],
                   "simulated_basis": "仕事の k_scaled_ph を塊に割り振り、bundle と同じ順番の規則で 3 本に詰めた（読み込み・始める間隔は入れない）",
                   "s4_gates_bundle1_total": p.get("s4_gates_bundle1_total"),
                   "ex_extra_budget_ph": round(budget_ph("e7", 40, p["factors_60_over_30"]), 2)}
    p["simulated_first_blocks"] = sim["timeline"][:12]
    p.setdefault("bundle_defaults", {"status": "outputs\\s4\\runs\\bundle1_status.json", "stop_file": "outputs\\s4\\runs\\STOP_BUNDLE1",
                                     "log": "outputs\\s4\\runs\\bundle1.log", "log_dir": "outputs\\s4\\runs\\bundle1_logs"})
    p["monitor"] = {
        "bundle_status": p["bundle_defaults"]["status"],
        "wait_bundle": (".venv\\Scripts\\python.exe scripts\\96_s4_ops.py wait --progress " + p["bundle_defaults"]["status"]
                        + " --appear-min 30"),
        "condition_progress_files": {c["id"]: c["progress"] for c in en},
        "job_progress_files": {j["id"]: j["progress"] for j in p["jobs"] if j["enabled"]},
        "note": ("包みの全体は bundle の状態のファイルを 96_s4_ops.py wait で見る（progress.json と同じ鍵）。条件ごとの progress.json も "
                 "wait で読めるが、塊に分けて回すので、塊の終わりごとに status=stopped（stop_reason max_new:N）になり、wait は 1 で終わる。"
                 "条件の最後まで見るには bundle の状態か、まとめた仕事の progress（job_progress_files）を見る。"
                 "まだ始まっていない条件の progress.json は無い（wait の --appear-min を長くする）")}
    p["interface"] = INTERFACE
    p["sources"] = {"charter": {"path": "docs/目標書_段階4.md", "sha256": sha256_file(CHARTER)},
                    "gates": {"path": "configs/s4_gates.json", "sha256": sha256_file(GATES)},
                    "board_0153": BOARD_0153_SHA,
                    "budget_calc": "outputs\\s4\\rules\\budget60_e7_30_calc.py（R と倍率）、configs\\s4_gates.json budget_notes"}
    p["sources"]["matches_board_0153"] = (p["sources"]["charter"]["sha256"] == BOARD_0153_SHA["docs/目標書_段階4.md"]
                                          and p["sources"]["gates"]["sha256"] == BOARD_0153_SHA["configs/s4_gates.json"])
    p["decided_before_results"] = DECIDED_BEFORE
    p["deviations_from_task_text"] = DEVIATIONS
    return p


INTERFACE = {
    "first_draft": "最初は決まりの文書から <スクリプト> <run|task> --experiment S4D1 --condition <条件> --model --trials <種の指定> [診断の引数]"
                   " [96_s4_resume.py と同じ引数] と置いたが、担当 A・B・C のスクリプトが先にできたので、計画の仕事は各スクリプトの"
                   "冒頭の使い方（adapters の seen）にそろえた",
    "contract": "包みが子に求めるのは次だけ: 同じ命令をもう一度打てば続きから回る、終了コード 0/1/2/3、progress.json（あれば）の pid・status・"
                "stop_reason、塊に分ける仕事は 96_s4_resume.py の --max-new を受け付ける（N 本回したら終了コード 1・stop_reason=max_new:N。"
                "最後の試行まで回れば 0）、--dry-run",
    "same_as_96": ["--dry-run", "--max-new", "--stop-file", "--progress-file", "--min-free-gb", "--min-commit-free-gb", "--mem-timeout-min",
                   "--accept-env-change"],
    "writes": "outputs\\v2eval\\<実験>\\<条件>\\: 96_s4_resume.py と同じ記録・progress.json（status・pid・proc_create_time・updated・done・total・"
              "stop_reason）・run.json・G_AUDIT.json・resume_spec.json・resume_log.json。診断の条件（腕・設定・開始状態・誘発の版など）は"
              "試行の json の欄に足す（87_v2_e.py・50_e_eval.py・recovla.eval.report・time_scoring が読めること）",
    "exit_codes": {"0": "全部そろった", "1": "途中で止まった（合図・--max-new・メモリ待ちの時間切れ・Ctrl+C）", "2": "エラー",
                   "3": "引数・前提の食い違い（制限時間・環境・ドライバ）"},
    "bundle_reads": "子の終了コードと、pid が子と一致する progress.json（仕事の progress）の status・stop_reason だけ",
}

DECIDED_BEFORE = [
    "条件の一覧・名前・帯・試行数・モデル・制限時間（目標書_段階4.md 4-2・3-1・8 節、s4_gates.json）",
    "仕事のまとめ方: D-RTC 6 設定は 1 仕事（担当 A の rotate、1 種ごとに交互）、影は 1 仕事、D-復帰は通りごとに 1 仕事（R と N を 10 種ごとに交互）、"
    "D-E7・D-単発の開始・移植・X2 は 1 条件 1 仕事",
    "塊の大きさ（assumed の仕事: D-E7 5 本、D-単発の開始 6 試行＝2 種、移植 5 状態。X2 は分けない）",
    "順番の規則（群の残りの見込みの長い順、群の中は済んだ塊の少ない順＝種の塊ごとに交互）",
    "失敗・止める合図・空きメモリ 12 GB・コミット 6 GB・メモリ待ち 120 分・始める間隔 90 s の規則",
    "見込みの式（s4_gates.json budget_notes と同じ。単発は K の実測で換算した列を並べる）",
]

DEVIATIONS = [
    "担当の指示は「1 つの条件が終わったら次を始める」だが、決まりの文書 第 4 節（比べる組は同じ種・同じ時期に、種の塊ごとに交互に回す。"
    "final.md の共通の作法 6 も同じ）に従い、既定では比べる組を種の塊ごとに交互に流す（assumed の仕事は子に --max-new を渡し、"
    "observed の A・C はスクリプトの中で交互）。--no-interleave で指示どおり 1 条件ずつ流せる。止める合図は『今の条件』ではなく"
    "『今の塊』を終えて止まる（塊は最長で E7 の 5 本＝約 27 分。observed の仕事は塊に分けないので、その仕事が終わるまで）",
    "担当 A・B・C のスクリプトは指示の例（--experiment S4D1 --condition <名前> --model）と違う形を選んでいた（実験 S4DRTC・S4DE7・"
    "S4DSTART・S4XPL・S4DREC と outputs\\s4\\x2、サブコマンド rotate・run・start・xpl・gen・eval、A・C は比べる組を 1 プロセスで交互）。"
    "包みはその命令をそのまま仕事にした（adapters の observed）。実験名が診断ごとに分かれるので、記録の場所は conditions の out_dir を見る",
    "束 0 の E7 の速さ（190150〜190155）は回した記録がない（outputs\\v2eval に S4E7 が無い）。E7 の見込みは予算（段階 3 の速さ）のまま",
    "X2（生成・測定）は束 1 の予算 36.8 に入っていない（final.md「GPU 数分」）。見込みは推測で足した",
]


# ---------------------------------------------------------------- 順番の規則（bundle と simulate が同じ関数を使う）
def blocks_of(j: dict, interleave: bool) -> int:
    b = j.get("block_trials")
    if not interleave or not b:
        return 1
    return -(-int(j["trials"]) // int(b))


def _est(j: dict) -> float:
    return float(j["est"].get("k_scaled_ph", j["est"]["budget_ph"]))


def remaining_ph(j: dict, e: dict, interleave: bool) -> float:
    if e.get("state") == "done":
        return 0.0
    nb = blocks_of(j, interleave)
    return _est(j) * max(0, nb - int(e.get("blocks_done", 0))) / nb


def pick_next(jobs: list, st: dict, running: set, interleave: bool):
    """次に始める仕事（無ければ None）。jobs は回す仕事（計画の順）、st は id -> 状態の辞書、running は動いている仕事の id。
    規則: 依存（after）が全部 done の pending の仕事のうち、群の残りの見込みが長い群から。群の中は済んだ塊の少ない順、同じなら計画の順。
    --no-interleave（interleave=False）なら、仕事の見込みの長い順。"""
    group_left = {}
    for j in jobs:
        group_left[j["group"]] = group_left.get(j["group"], 0.0) + remaining_ph(j, st[j["id"]], interleave)
    ready = []
    for k, j in enumerate(jobs):
        e = st[j["id"]]
        if e.get("state") != "pending" or j["id"] in running:
            continue
        if any(st.get(d, {}).get("state") != "done" for d in j.get("after", [])):
            continue
        key = ((-round(group_left[j["group"]], 6), int(e.get("blocks_done", 0)), k) if interleave else (-_est(j), k))
        ready.append((key, j))
    return min(ready, key=lambda x: x[0])[1] if ready else None


def simulate(plan: dict, lanes: int = 3, interleave: bool = True) -> dict:
    """見込みの時間（k_scaled_ph）で、bundle と同じ規則の詰め方を机上で回す。読み込みの時間・始める間隔は入れない。"""
    jobs = [j for j in plan["jobs"] if j["enabled"]]
    st = {j["id"]: {"state": "pending", "blocks_done": 0} for j in jobs}
    t, free_at, running, timeline = 0.0, [0.0] * lanes, {}, []
    while True:
        for lane in range(lanes):
            if free_at[lane] <= t + 1e-12 and lane not in running.values():
                j = pick_next(jobs, st, set(running), interleave)
                if j is None:
                    continue
                nb = blocks_of(j, interleave)
                dur = _est(j) / nb
                running[j["id"]] = lane
                free_at[lane] = t + dur
                timeline.append({"lane": lane, "start_h": round(t, 2), "end_h": round(t + dur, 2), "id": j["id"],
                                 "block": st[j["id"]]["blocks_done"] + 1, "of": nb})
        if not running:
            break
        t = min(free_at[ln] for ln in running.values())
        for jid in [k for k, ln in running.items() if free_at[ln] <= t + 1e-12]:
            del running[jid]
            e = st[jid]
            e["blocks_done"] += 1
            j = next(x for x in jobs if x["id"] == jid)
            if e["blocks_done"] >= blocks_of(j, interleave):
                e["state"] = "done"
    return {"makespan_h": round(t, 2), "timeline": timeline}


# ---------------------------------------------------------------- smoke の計画
def build_smoke_plan() -> dict:
    """96_s4_resume.py run の小さな 2 条件（担当 D の小帯 44490〜44499 の 44490・44491）。担当 A・B・C のスクリプトが無くても
    包みの順番・塊・止める合図・再開を確かめられる。GPU は同時に 1 プロセス（--lanes 1）で回す。"""
    exp = "S4SMOKE_D"
    C, J = [], []
    for cid, seed in (("SMK_a", 44490), ("SMK_b", 44491)):
        c = _cond(cid, diag="smoke", gate="-", model="R1v3", trials=f"natural:{seed}:1", seeds=[seed, seed], n=3,
                  band="smoke_D_44490_44499", cost="nat", group="smoke", F=DEFAULT_FACTORS, experiment=exp,
                  notes="smoke（担当 D の小帯）。報告に使わない")
        C.append(c)
        J.append(_job(cid, [c], ["scripts\\96_s4_resume.py", "run", "--experiment", exp, "--condition", cid, "--model", "R1v3",
                                 "--trials", c["trials_spec"], "--mode", "naive", "--exec-interval", "6", "--no-safety", "--time-limit-s", "60"],
                      c["progress"], group="smoke", block=2, adapter="96_s4_resume"))
        c["job"] = cid
    p = {"schema": "recovery_vla.s4_bundle1_plan/2", "name": "bundle1_smoke", "experiment": exp, "written": _now_s(),
         "by": "担当 D（運用役）、scripts\\98_s4_d_plan.py smoke-plan", "with_ex": False, "conditions": C, "jobs": J, "adapters": {},
         "factors_60_over_30": dict(DEFAULT_FACTORS), "rates": RATES,
         "bundle_defaults": {"status": "outputs\\s4\\runs\\bundle1_smoke_status.json", "stop_file": "outputs\\s4\\runs\\STOP_BUNDLE1_SMOKE",
                             "log": "outputs\\s4\\runs\\bundle1_smoke.log", "log_dir": "outputs\\s4\\runs\\bundle1_smoke_logs"}}
    return finish_plan(p)


# ---------------------------------------------------------------- 表
def table_text(p: dict) -> str:
    cols = ["条件", "診断", "関門", "モデル", "記録（実験\\条件）", "種", "試行", "制限時間", "仕事", "予算 ph", "K換算 ph"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for c in p["conditions"]:
        lim = c["time_limits"] or {}
        tl = (f"{lim['time_limit_s']} s（30 主・60 副）" if "time_limit_s" in lim else
              f"手順 {lim['step_timeout_s']} s・全体 {lim['task_time_limit_s']} s" if lim else "-")
        out.append("| " + " | ".join([c["id"] + ("" if c["enabled"] else "（条件つき・無効）"), c["diag"], c["gate"], c["model"] or "-",
                                       f"{c['experiment']}\\{c['record_condition']}", f"{c['seeds'][0]}〜{c['seeds'][1]}", str(c["trials"]),
                                       tl, c["job"] or "-", f"{c['est']['budget_ph']:.2f}", f"{c['est']['k_scaled_ph']:.2f}"]) + " |")
    out += ["", "| 仕事 | 群 | 形 | 塊 | 予算 ph | K換算 ph | 命令 |", "|---|---|---|---|---|---|---|"]
    for j in p["jobs"]:
        out.append(f"| {j['id']}{'' if j['enabled'] else '（無効）'} | {j['group']} | {j['adapter']} | {j['block_trials'] or '-'} | "
                   f"{j['est']['budget_ph']:.2f} | {j['est']['k_scaled_ph']:.2f} | {j['cmd']} |")
    t = p["totals"]
    out.append("")
    out.append(f"合計（有効な {t['conditions']} 条件・{t['trials']} 試行・{t['jobs']} 仕事）: 予算 {t['budget_ph']} プロセス時間（X2 を除くと "
               f"{t['budget_ph_without_x2']}。s4_gates の束 1 の合計 {(t.get('s4_gates_bundle1_total') or {}).get('s4_r2')}）、"
               f"K の実測で換算 {t['k_scaled_ph']}。3 本の枠: 詰め込み 95% で 予算 {t['wall_h_3lanes_pack95']['budget']} 時間・K 換算 "
               f"{t['wall_h_3lanes_pack95']['k_scaled']} 時間、同じ規則で机上で詰めると {t['wall_h_3lanes_simulated']} 時間（K 換算）。"
               f"EX を足すと予算 +{t.get('ex_extra_budget_ph')}")
    out.append(f"K の実測: {p['k_rate']['per_h']} 試行/時/プロセス（2 本並行、{p['k_rate']['source']}）、換算の倍率 {p['k_scale']['value']}")
    for d, g in t["by_diag"].items():
        out.append(f"  {d}: {g['conditions']} 条件、{g['trials']} 試行、予算 {g['budget_ph']:.2f}・K 換算 {g['k_scaled_ph']:.2f}")
    return "\n".join(out)


def cmd_plan(a) -> int:
    p = build_plan(with_ex=a.with_ex)
    out = _abs(a.out)
    write_atomic(out, json.dumps(p, ensure_ascii=False, indent=1))
    print(table_text(p))
    print(f"\n[plan] {_rel(out)}（SHA-256 {sha256_file(out)[:12]}…）。決まりの文書・s4_gates が掲示板 0153 と一致: "
          f"{p['sources']['matches_board_0153']}")
    missing = sorted({j["script"] for j in p["jobs"] if j["enabled"] and not _abs(j["script"]).is_file()})
    if missing:
        print(f"[plan] まだ無いスクリプト: {missing}（bundle は無い仕事を script_missing の失敗にして、ほかを続ける）")
    return 0


def cmd_smoke_plan(a) -> int:
    p = build_smoke_plan()
    out = _abs(a.out)
    write_atomic(out, json.dumps(p, ensure_ascii=False, indent=1))
    print(table_text(p))
    print(f"\n[smoke-plan] {_rel(out)}")
    return 0


# ---------------------------------------------------------------- 一括の包み
def read_progress(path: pathlib.Path):
    for _ in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError):
            time.sleep(0.3)
    return None


def classify(rc, prog, child_pid, fresh: bool = False) -> tuple:
    """子が終わった後の扱い。(結果, 理由)。結果は done / block_done / stopped / memory_timeout / failed。
    progress.json は、pid が子と一致するか、子を始めた後に書かれた（fresh）ときだけ使う（引数の食い違いで progress.json を書く前に
    終わった子に、前の塊の stopped・max_new を読み違えないため）。pid だけで見ないのは、.venv\\Scripts\\python.exe が本物の
    python を子として起こす起動役で、Popen の pid と子が progress.json に書く pid が違うため（Windows の venv。10/08 に確かめた）。
    rc が None（引き取った子で終了コードが分からない）なら progress.json だけで決める。"""
    pr = prog if (prog and (fresh or (child_pid is not None and prog.get("pid") == child_pid))) else None
    st = (pr or {}).get("status")
    reason = str((pr or {}).get("stop_reason") or "")
    if rc == 0:
        return "done", ("" if st in (None, "done") else f"終了コード 0 だが progress の status={st}")
    if rc is None and st == "done":
        return "done", "引き取った子（progress.json で完了）"
    if rc in (1, None) and st in ("stopped", "interrupted", "memory_timeout"):
        if reason.startswith("max_new"):
            return "block_done", reason
        if st == "memory_timeout" or reason == "memory_timeout":
            return "memory_timeout", "子のメモリ待ちの時間切れ"
        return "stopped", reason or st
    if rc == 1 and pr is None:
        return "stopped", "終了コード 1（pid の合う progress.json なし）"
    if rc is None:
        return "failed", f"引き取った子が状態 {st} のまま消えた（異常終了）"
    return "failed", f"終了コード {rc}" + (f"、progress の error: {pr.get('error')}" if pr and pr.get("error") else "")


class Bundle:
    """計画の仕事を枠の数だけ並行に、塊ごとに流す。状態は bundle1_status.json（96_s4_ops.py wait が読む形）。"""

    def __init__(self, a, plan: dict, plan_path: pathlib.Path, ops):
        self.a, self.plan, self.ops = a, plan, ops
        d = plan.get("bundle_defaults") or {}
        self.status_path = _abs(a.status or d.get("status") or RUNS / "bundle1_status.json")
        self.stop_file = _abs(a.stop_file or d.get("stop_file") or RUNS / "STOP_BUNDLE1")
        self.log_path = _abs(a.log or d.get("log") or RUNS / "bundle1.log")
        self.log_dir = _abs(a.log_dir or d.get("log_dir") or RUNS / "bundle1_logs")
        self.global_stop = S4 / "STOP"
        self.plan_path = plan_path
        only = {x for x in (a.only or "").split(",") if x}
        exclude = {x for x in (a.exclude or "").split(",") if x}
        self.jobs_plan = [j for j in plan["jobs"] if j["enabled"] and (not only or j["id"] in only) and j["id"] not in exclude]
        self.by_id = {j["id"]: j for j in self.jobs_plan}
        self.interleave = not a.no_interleave
        self.jobs = {}                              # 動いている仕事: id -> {"popen" or None, "pid", "create_time", "started_t"}
        self.st = None

    # -------------------------------------------------------- 記録
    def log(self, msg: str) -> None:
        print(f"[bundle {time.strftime('%H:%M:%S')}] {msg}", flush=True)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{_now_s()} {msg}\n")
        except OSError:
            pass

    def save(self, **kw) -> None:
        s = self.st
        s.update(kw)
        es = [s["jobs"][j["id"]] for j in self.jobs_plan]
        s["done"] = sum(e["state"] == "done" for e in es)
        s["total"] = len(es)
        s["failed"] = sorted(j["id"] for j in self.jobs_plan if s["jobs"][j["id"]]["state"] == "failed")
        s["running"] = sorted(self.jobs)
        s["updated"] = _now_s()
        try:
            m = self.ops.memory_gb()
            s["free_phys_gb"], s["free_commit_gb"] = m["phys_free_gb"], m["commit_free_gb"]
        except Exception:                            # noqa: BLE001
            pass
        write_atomic(self.status_path, json.dumps(s, ensure_ascii=False, indent=1, default=str))

    # -------------------------------------------------------- 始めの確かめと引き継ぎ
    def load_state(self) -> int:
        old = None
        if self.status_path.is_file():
            try:
                old = json.loads(self.status_path.read_text(encoding="utf-8"))
            except Exception:                        # noqa: BLE001
                old = None
        if old and old.get("status") in self.ops.LIVE_STATUS and old.get("pid") and old.get("pid") != os.getpid() \
                and self.ops._alive(int(old["pid"]), old.get("proc_create_time")):
            self.log(f"{_rel(self.status_path)}: pid {old['pid']} の包みがまだ動いている（status={old.get('status')}）。二重に回さない")
            return 3
        try:
            import psutil
            ct = psutil.Process().create_time()
        except Exception:                            # noqa: BLE001
            ct = None
        jobs = (old or {}).get("jobs") or {}
        self.st = {"schema": "recovery_vla.s4_bundle_status/1", "experiment": self.plan["experiment"], "condition": self.plan["name"],
                   "cmd": "bundle", "plan": _rel(self.plan_path), "plan_sha256": sha256_file(self.plan_path),
                   "pid": os.getpid(), "proc_create_time": ct, "host": os.environ.get("COMPUTERNAME"), "started": _now_s(),
                   "status": "starting", "lanes": self.a.lanes, "interleave": self.interleave, "stop_file": _rel(self.stop_file),
                   "min_free_gb": self.a.min_free_gb, "successes": None, "stop_reason": None, "error": None,
                   "jobs": jobs, "sessions": (old or {}).get("sessions", [])}
        if old and old.get("plan_sha256") and old.get("plan_sha256") != self.st["plan_sha256"]:
            self.log(f"計画が前の回と違う（{old['plan_sha256'][:12]}… → {self.st['plan_sha256'][:12]}…）。仕事は id で引き継ぐ")
        for j in self.jobs_plan:
            e = jobs.setdefault(j["id"], {"state": "pending", "blocks_done": 0, "attempts": 0, "history": []})
            e["blocks"] = blocks_of(j, self.interleave)
            e["covers"] = j["covers"]
            e["log"] = _rel(self.log_dir / f"{j['id']}.log")
            e["progress"] = j.get("progress")
            if e["state"] == "running":
                pid, ct2 = e.get("pid"), e.get("create_time")
                if pid and self.ops._alive(int(pid), ct2):
                    st0 = time.mktime(time.strptime(e["started"], "%Y-%m-%d %H:%M:%S")) if e.get("started") else time.time()
                    self.jobs[j["id"]] = {"popen": None, "pid": int(pid), "create_time": ct2, "started_t": st0}
                    self.log(f"{j['id']}: 前の回の子 pid {pid} が生きているので引き取って終わりを待つ")
                else:
                    e["state"] = "pending"
                    e["note"] = f"前の回の途中で切れた（{_now_s()} に確認、子は残っていない）。子が続きから回す"
                    self.log(f"{j['id']}: 前の回の途中で切れていた。続きから回す")
            elif e["state"] in ("failed", "stopped", "blocked", "memory_timeout"):
                self.log(f"{j['id']}: 前の回は {e['state']}。もう一度回す（子が完全な試行を飛ばす）")
                e["state"] = "pending"
            elif e["state"] == "done":
                pr = read_progress(_abs(j["progress"])) if j.get("progress") else None
                if pr is not None and pr.get("status") != "done":
                    e["state"] = "pending"
                    e["note"] = f"状態は done だが progress.json の status={pr.get('status')}。もう一度回す"
                    self.log(f"{j['id']}: {e['note']}")
        return 0

    # -------------------------------------------------------- 子を始める・終わりを見る
    def argv_of(self, j: dict, block: bool = True) -> list:
        py = self.plan.get("python") or (str(PY) if PY.is_file() else sys.executable)
        argv = [py, str(_abs(j["argv"][0]))] + list(j["argv"][1:])
        if block and self.interleave and j.get("block_trials"):
            argv += ["--max-new", str(int(j["block_trials"]))]
        return argv

    def start(self, j: dict) -> None:
        e = self.st["jobs"][j["id"]]
        script = _abs(j["argv"][0])
        e["attempts"] = int(e.get("attempts", 0)) + 1
        missing = ([f"script_missing: {_rel(script)}"] if not script.is_file() else []) + \
            [f"prereq_missing: {r}" for r in j.get("requires", []) if not _abs(r).is_file()]
        if missing:
            e["state"] = "failed"
            e["last_error"] = "; ".join(missing)
            e["history"].append({"t": _now_s(), "result": "failed", "why": e["last_error"]})
            self.log(f"{j['id']}: 始められない（{e['last_error']}）。失敗として、ほかを続ける")
            return
        argv = self.argv_of(j)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        lf = open(self.log_dir / f"{j['id']}.log", "ab")
        lf.write(f"\n===== {_now_s()} 始める {j['id']} 塊 {e.get('blocks_done', 0) + 1}/{e['blocks']}: "
                 f"{subprocess.list2cmdline(argv)}\n".encode("utf-8"))
        lf.flush()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p = subprocess.Popen(argv, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT, env=env)
        lf.close()                                   # 子が持つ。親の手元は閉じる
        try:
            import psutil
            ct = psutil.Process(p.pid).create_time()
        except Exception:                            # noqa: BLE001
            ct = None
        self.jobs[j["id"]] = {"popen": p, "pid": p.pid, "create_time": ct, "started_t": time.time()}
        e.update({"state": "running", "pid": p.pid, "create_time": ct, "started": _now_s(), "ended": None, "exit_code": None})
        self.st["sessions"][-1]["started_jobs"].append({"id": j["id"], "block": e.get("blocks_done", 0) + 1, "pid": p.pid, "t": _now_s()})
        self.log(f"{j['id']}: 始めた pid {p.pid}（塊 {e.get('blocks_done', 0) + 1}/{e['blocks']}）")

    def poll(self) -> list:
        """終わった子を片付ける。返り値: [(id, 結果)]。"""
        fin = []
        for jid, r in list(self.jobs.items()):
            if r["popen"] is not None:
                rc = r["popen"].poll()
                if rc is None:
                    continue
            else:
                if self.ops._alive(r["pid"], r.get("create_time")):
                    continue
                rc = None
            j = self.by_id.get(jid)
            pp = _abs(j["progress"]) if j and j.get("progress") else None
            pr = read_progress(pp) if pp else None
            try:
                fresh = pp is not None and pp.stat().st_mtime >= r["started_t"] - 0.5
            except OSError:
                fresh = False
            res, why = classify(rc, pr, r["pid"], fresh)
            e = self.st["jobs"][jid]
            e.update({"ended": _now_s(), "exit_code": rc, "progress_status": (pr or {}).get("status"),
                      "progress_done": (pr or {}).get("done"), "progress_total": (pr or {}).get("total")})
            e["history"].append({"t": _now_s(), "pid": r["pid"], "rc": rc, "result": res, "why": why,
                                 "wall_min": round((time.time() - r["started_t"]) / 60, 2)})
            if res in ("done", "block_done"):
                e["blocks_done"] = int(e.get("blocks_done", 0)) + 1
            if res == "done":
                e["state"] = "done"
                e["blocks_done"] = max(e["blocks_done"], e["blocks"])
            elif res == "block_done":
                e["state"] = "pending"
            elif res in ("stopped", "memory_timeout"):
                e["state"] = res
            else:
                e["state"] = "failed"
                e["last_error"] = why
            del self.jobs[jid]
            self.log(f"{jid}: 終わった（終了コード {rc}、{res}{'：' + why if why else ''}）")
            fin.append((jid, res))
        return fin

    def block_dependents(self) -> None:
        """依存先が失敗した・この回に無い仕事は、この回は回さない（blocked。次の回でもう一度試す）。"""
        changed = True
        while changed:
            changed = False
            for j in self.jobs_plan:
                e = self.st["jobs"][j["id"]]
                if e["state"] != "pending":
                    continue
                bad = [d for d in j.get("after", []) if self.st["jobs"].get(d, {}).get("state") in ("failed", "blocked")
                       or (d not in self.by_id and self.st["jobs"].get(d, {}).get("state") != "done")]
                if bad:
                    e["state"] = "blocked"
                    e["note"] = f"依存先 {bad} が終わっていない・失敗した"
                    self.log(f"{j['id']}: 依存先 {bad} が無いので、この回は回さない")
                    changed = True

    # -------------------------------------------------------- 本体
    def run(self) -> int:
        code = self.load_state()
        if code:
            return code
        self.st["sessions"].append({"start": _now_s(), "pid": os.getpid(), "lanes": self.a.lanes, "interleave": self.interleave,
                                    "selected": [j["id"] for j in self.jobs_plan], "started_jobs": [], "end": None, "status": None})
        self.save(status="running")
        self.log(f"始める: 計画 {_rel(self.plan_path)}、仕事 {len(self.jobs_plan)}、枠 {self.a.lanes}、塊に分ける {self.interleave}、"
                 f"止める合図 {_rel(self.stop_file)}")
        stopping, stop_reason, mem_since, last_start, n_started, final = None, None, None, 0.0, 0, None
        try:
            while True:
                for jid, res in self.poll():
                    if res == "stopped" and not stopping:
                        stopping, stop_reason = "child_stopped", f"{jid} が止める合図で止まった（{self.st['jobs'][jid]['history'][-1]['why']}）"
                if not stopping:
                    for f in (self.stop_file, self.global_stop):
                        if f.exists():
                            stopping, stop_reason = "stop_file", f"stop_file:{_rel(f)}"
                            self.log(f"止める合図 {_rel(f)} を見た。今の塊を終えてから止まる（動いている {sorted(self.jobs)}）")
                            break
                if not stopping and self.a.max_jobs and n_started >= self.a.max_jobs:
                    stopping, stop_reason = "max_jobs", f"max_jobs:{self.a.max_jobs}"
                self.block_dependents()
                status = "running"
                while not stopping and len(self.jobs) < self.a.lanes:
                    j = pick_next(self.jobs_plan, self.st["jobs"], set(self.jobs), self.interleave)
                    if j is None:
                        break
                    if self.jobs and time.time() - last_start < self.a.stagger_s:
                        break
                    m = self.ops.memory_gb()
                    if m["phys_free_gb"] < self.a.min_free_gb or m["commit_free_gb"] < self.a.min_commit_free_gb:
                        mem_since = mem_since or time.time()
                        status = "memory_wait" if not self.jobs else "running"
                        self.st["wait_note"] = (f"空き 物理 {m['phys_free_gb']} GB（{self.a.min_free_gb} 未満で始めない）・"
                                                f"コミット {m['commit_free_gb']} GB（{self.a.min_commit_free_gb} 未満）")
                        if not self.jobs and self.a.mem_timeout_min and (time.time() - mem_since) / 60 > self.a.mem_timeout_min:
                            stopping, stop_reason = "memory_timeout", "memory_timeout"
                        break
                    mem_since = None
                    self.st["wait_note"] = None
                    self.start(j)
                    last_start = time.time()
                    n_started += 1
                    if self.a.max_jobs and n_started >= self.a.max_jobs:
                        break
                self.save(status=status, stop_reason=stop_reason)
                if not self.jobs and (stopping or pick_next(self.jobs_plan, self.st["jobs"], set(), self.interleave) is None):
                    break
                time.sleep(self.a.poll_s)
        except KeyboardInterrupt:
            final, stop_reason = "interrupted", "KeyboardInterrupt"
            self.log("Ctrl+C。動いている子の状態は running のまま残す（次の回で、生きていれば引き取り、消えていれば続きから回す）")
        es = {j["id"]: self.st["jobs"][j["id"]]["state"] for j in self.jobs_plan}
        if final is None:
            final = ("done" if all(s == "done" for s in es.values()) else "memory_timeout" if stopping == "memory_timeout"
                     else "stopped" if stopping else "error")
        left = {k: v for k, v in es.items() if v != "done"}
        self.st["sessions"][-1].update({"end": _now_s(), "status": final, "stop_reason": stop_reason})
        self.save(status=final, stop_reason=stop_reason, error=left if final == "error" else None)
        self.log(f"終わり: {final}（完了 {len(es) - len(left)}/{len(es)}、残り {left}）")
        return FINAL_STATUS[final]

    # -------------------------------------------------------- dry-run
    def dry_run(self) -> int:
        old = {}
        if self.status_path.is_file():
            try:
                old = json.loads(self.status_path.read_text(encoding="utf-8")).get("jobs") or {}
            except Exception:                        # noqa: BLE001
                old = {}
        print(f"[dry-run] 計画 {_rel(self.plan_path)}、状態 {_rel(self.status_path)}、枠 {self.a.lanes}、塊に分ける {self.interleave}")
        left = []
        for j in self.jobs_plan:
            e = old.get(j["id"]) or {"state": "pending", "blocks_done": 0}
            state = "done" if e.get("state") == "done" else "pending"
            if state != "done":
                left.append(dict(j, enabled=True))
            print(f"  {j['id']:28s} {state:8s} 塊 {e.get('blocks_done', 0)}/{blocks_of(j, self.interleave)} "
                  f"{'あり' if _abs(j['argv'][0]).is_file() else 'スクリプトなし'}  {subprocess.list2cmdline(self.argv_of(j)[1:])}")
        sim = simulate({"jobs": left}, lanes=self.a.lanes, interleave=self.interleave)
        print(f"[dry-run] 残りを同じ規則で机上で詰めると {sim['makespan_h']} 時間（K 換算の見込み）。始める順（先頭 15）:")
        for x in sim["timeline"][:15]:
            print(f"  枠 {x['lane']} {x['start_h']:6.2f}〜{x['end_h']:6.2f} h  {x['id']} 塊 {x['block']}/{x['of']}")
        if self.a.dry_run_children:
            for j in self.jobs_plan:
                if not _abs(j["argv"][0]).is_file():
                    print(f"[dry-run] {j['id']}: スクリプトが無い")
                    continue
                if j.get("dry_run") is False:
                    print(f"[dry-run] {j['id']}: 子に --dry-run が無いので飛ばす")
                    continue
                miss = [r for r in j.get("requires", []) if not _abs(r).is_file()]
                if miss:
                    print(f"[dry-run] {j['id']}: 回す前に掲示する定義が無い {miss}（pre_steps）")
                r = subprocess.run(self.argv_of(j, block=False) + ["--dry-run"], cwd=str(ROOT), capture_output=True,
                                   env=dict(os.environ, PYTHONIOENCODING="utf-8"))
                lines = r.stdout.decode("utf-8", "replace").strip().splitlines()
                tail = [ln for ln in lines if "dry-run" in ln][:3] or lines[-2:]
                err = r.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] if r.returncode else []
                print(f"[dry-run] {j['id']}: 子の --dry-run 終了コード {r.returncode} {tail} {err}")
        return 0


def load_plan(a) -> tuple:
    p = _abs(a.plan)
    if not p.is_file():
        raise SystemExit(f"計画 {p} が無い。先に plan（または smoke-plan）を回す")
    return json.loads(p.read_text(encoding="utf-8")), p


def cmd_bundle(a, ops=None) -> int:
    plan, path = load_plan(a)
    ops = ops or load_ops()
    b = Bundle(a, plan, path, ops)
    if a.dry_run:
        return b.dry_run()
    if a.env_record:
        r = subprocess.run([str(PY) if PY.is_file() else sys.executable, str(ROOT / "scripts" / "96_s4_ops.py"), "env-record"],
                           cwd=str(ROOT), capture_output=True)
        b.log(f"env-record 終了コード {r.returncode}")
    return b.run()


def cmd_status(a) -> int:
    plan, _ = load_plan(a)
    d = plan.get("bundle_defaults") or {}
    sp = _abs(a.status or d.get("status") or RUNS / "bundle1_status.json")
    if not sp.is_file():
        print(f"{_rel(sp)} がまだ無い（bundle を回していない）")
        return 0
    s = json.loads(sp.read_text(encoding="utf-8"))
    print(f"{_rel(sp)}: status={s.get('status')} 完了 {s.get('done')}/{s.get('total')} 動いている {s.get('running')} "
          f"失敗 {s.get('failed')} 更新 {s.get('updated')} 空き {s.get('free_phys_gb')} GB")
    for jid, e in (s.get("jobs") or {}).items():
        pr = read_progress(_abs(e["progress"])) if e.get("progress") else None
        print(f"  {jid:28s} {e.get('state'):10s} 塊 {e.get('blocks_done')}/{e.get('blocks')} 試み {e.get('attempts')} "
              f"progress {(pr or {}).get('status')} {(pr or {}).get('done')}/{(pr or {}).get('total')} {e.get('last_error') or ''}")
    return 0


def cmd_wait_cmd(a) -> int:
    plan, _ = load_plan(a)
    print("# 包みの全体（これを背景で付ける）")
    print(plan["monitor"]["wait_bundle"])
    print("# 仕事ごと（塊に分ける仕事＝D-E7・単発の開始・移植は、塊の終わりで status=stopped になり 1 で終わる。"
          "まだ始まっていない仕事は --appear-min の間だけ待つ。X2 の測定は生成の後に始まる）")
    for jid, pth in plan["monitor"]["job_progress_files"].items():
        print(f".venv\\Scripts\\python.exe scripts\\96_s4_ops.py wait --progress {pth} --appear-min 1440   # {jid}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="束 1 の計画の JSON と表")
    p.add_argument("--out", default=str(PLAN_PATH))
    p.add_argument("--with-ex", action="store_true", help="条件つきの腕 EX を有効にする（candidate_selection の行 1・行 3 の矛盾の枝）")
    p = sub.add_parser("smoke-plan", help="smoke の計画（96_s4_resume.py run の 2 条件、種 44490・44491）")
    p.add_argument("--out", default=str(SMOKE_PLAN_PATH))
    p = sub.add_parser("bundle", help="計画の仕事を枠の数だけ並行に、塊ごとに流す（続きから回せる）")
    p.add_argument("--plan", default=str(PLAN_PATH))
    p.add_argument("--status", default=None, help="状態のファイル（省略時は計画の bundle_defaults）")
    p.add_argument("--stop-file", default=None, help="止める合図（省略時は計画の bundle_defaults。既定 outputs\\s4\\runs\\STOP_BUNDLE1）")
    p.add_argument("--log", default=None)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--lanes", type=int, default=3, help="同時に動かす子の数（学習中は 1）")
    p.add_argument("--no-interleave", action="store_true", help="塊に分けず、長い仕事から 1 つずつ流す")
    p.add_argument("--only", default="", help="回す仕事の id（カンマ区切り）")
    p.add_argument("--exclude", default="", help="回さない仕事の id（カンマ区切り）")
    p.add_argument("--min-free-gb", type=float, default=12.0)
    p.add_argument("--min-commit-free-gb", type=float, default=6.0)
    p.add_argument("--mem-timeout-min", type=float, default=120.0)
    p.add_argument("--stagger-s", type=float, default=90.0, help="子を始める間隔の最小（動いている子があるとき）")
    p.add_argument("--poll-s", type=float, default=10.0)
    p.add_argument("--max-jobs", type=int, default=0, help="この数の塊を始めたら新しい塊を始めない（試験用。0 で無制限）")
    p.add_argument("--env-record", action="store_true", help="始めに 96_s4_ops.py env-record を回す")
    p.add_argument("--dry-run", action="store_true", help="順番と命令を見るだけ（何も書かない）")
    p.add_argument("--dry-run-children", action="store_true", help="--dry-run のとき、各仕事の子にも --dry-run を渡して回す")
    for name in ("status", "wait-cmd"):
        p = sub.add_parser(name)
        p.add_argument("--plan", default=str(PLAN_PATH))
        if name == "status":
            p.add_argument("--status", default=None)
    return ap


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                # noqa: BLE001
        pass
    a = build_parser().parse_args(argv)
    try:
        return {"plan": cmd_plan, "smoke-plan": cmd_smoke_plan, "bundle": cmd_bundle, "status": cmd_status,
                "wait-cmd": cmd_wait_cmd}[a.cmd](a)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr, flush=True)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
