"""最終評価の結果の表（手順書 Step J の 2、0098）: outputs/results/results.md を記録から作る。数字は手で書かない。

    .venv\\Scripts\\python.exe scripts\\53_results.py          # 集計（50_e_eval.py report・51_planner.py e7-summary）から results.md まで

読むもの: outputs/eval/EFINAL（8 組）、outputs/results/e_report_{final,pre}.json、outputs/k1/e6_R2_{final,pre}*.json、
outputs/planner/（judge_final・judge_pre・llm_final_sentences_v1・llm_dev_sentences_v1・e7_summary_E7_final・
return_compare・runs/E7_pre）。主な検定と指標の定義は 50_e_eval.py の冒頭（5c40bbd・548a335・0090・0096）。
"""
import collections
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

from recovla.common import config, goals

CFG = config.load()
ROOT = config.ROOT
OUT = config.path(CFG["paths"]["outputs"])
RES = OUT / "results"
PL = OUT / "planner"
FREEZE_TAG = "stepJ-freeze"


def _j(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def _pct(x, nd=0):
    return "—" if x is None else f"{100 * x:.{nd}f}%"


def _ci(w):
    return "—" if not w or w[0] is None else f"{100 * w[0]:.0f}〜{100 * w[1]:.0f}%"


def _p(p):
    if p is None:
        return "—"
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def _pair(d):
    """paired_binary の 1 行: x 対 y（x だけ・y だけ、対の数、p、差の区間）。"""
    lo, hi = d["diff_95ci_newcombe"]
    ci = "—" if lo is None else f"{100 * lo:+.0f}〜{100 * hi:+.0f}"
    return (f"{_pct(d['x_rate'])} 対 {_pct(d['y_rate'])}（{d['x_only']}・{d['y_only']}、{d['pairs']} 対）", _p(d["mcnemar_exact_p"]), ci)


def _wilson(k, n):
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("e_eval", ROOT / "scripts" / "50_e_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m._wilson(k, n)


def nat_failures(cond, blocked):
    """自然の失敗の形（検出器の最初に確定したもの。なければ時間切れ）と、フィルタが止め続けたもの（E5 の blocked）。"""
    from recovla.eval import failure_detect as FD
    rows = []
    bset = {(b["seed"], b["target"]) for b in blocked}
    for p in sorted((OUT / "eval" / "EFINAL" / cond).glob("trial_*.json")):
        m = _j(p)
        if m["success"]:
            continue
        ev = FD.detect_trial(np.load(p.with_suffix(".npz")))
        k = (m["seed"], m["steps"][0]["target"])
        rows.append({"seed": k[0], "target": k[1], "form": ev[0].kind if ev else "timeout", "filter_blocked": k in bset})
    return rows


def blocked_detail(on, off, blocked):
    """フィルタが止め続けた試行: 働いたこまで最も近かった障害物（目標を除く、min_dist の列）と、なしの同じ種の結果。"""
    from recovla.sim.contact import COLUMNS
    from recovla.common.seeds import COLORS
    out = []
    idx = {}
    for p in sorted((OUT / "eval" / "EFINAL" / on).glob("trial_*.json")):
        m = _j(p)
        idx[(m["seed"], m["steps"][0]["target"])] = p
    for b in blocked:
        p = idx[(b["seed"], b["target"])]
        d = np.load(p.with_suffix(".npz"))
        sa = d["safety_active"].astype(bool)
        md = d["min_dist"].astype(float).copy()
        md[:, COLORS.index(b["target"])] = np.inf
        near = collections.Counter(COLUMNS[j] for j in md[sa].argmin(1))
        off_m = _j(OUT / "eval" / "EFINAL" / off / p.name)
        off_d = np.load((OUT / "eval" / "EFINAL" / off / p.name).with_suffix(".npz"))
        touched = [COLUMNS[j] for j in range(len(COLUMNS)) if off_d["contact_robot"][:, j].astype(bool).any()]
        out.append({**b, "trial": p.name, "nearest_obstacle_when_active": dict(near),
                    "active_frames": int(sa.sum()), "frames": int(len(sa)),
                    "without_filter_success": bool(off_m["success"]), "without_filter_touched": touched})
    return out


def build(goals: dict) -> str:
    fin, pre = _j(RES / "e_report_final.json"), _j(RES / "e_report_pre.json")
    P, Q = fin["primary"], pre["primary"]
    S = fin["sets"]
    L = []
    w = L.append
    head = subprocess.run([str(ROOT / ".tools" / "git" / "cmd" / "git.exe"), "rev-parse", "--short", FREEZE_TAG + "^{commit}"],
                          cwd=ROOT, capture_output=True, text=True).stdout.strip()
    w("# 最終評価の結果")
    w("")
    del head
    w(f"このファイルは `scripts/53_results.py` が記録から作った（{time.strftime('%Y-%m-%d %H:%M')}）。数字は手で書いていない。"
      "評価はコードを凍結した後に行い、チェックポイントとデータの版は `docs/freeze/` の SHA-256 の一覧で固定している。"
      f"テスト用のシード範囲（110000〜）で回した。最終モデルは、検証用のシード範囲（199000〜）での予備評価で事前に決めた "
      f"**{fin['final_model']}**（以後変えない決まり）。予備評価の値も並べて示す。")
    w("")
    w(f"目標書: この表を作ったときの版は v{goals['version']}（タグ {goals['tag']}、`{goals['path']}` の SHA-256 `{goals['sha256']}`）。"
      f"この表の評価（{FREEZE_TAG}）は目標書 v1 の実行系、すなわち安全フィルタが立方体の真値を使い、推論中に物理演算を止める"
      "理想化した実行系によるもの（0105・0106）。")
    w("")
    # ------------------------------------------------------------------ 主な検定
    w("## 1. 主要評価項目（予備評価の前に固めたもの）")
    w("")
    w("| E | 比べるもの | 主な指標 | 最終評価 | p（McNemar） | 差の 95% 区間（ポイント） | 予備 |")
    w("|---|---|---|---|---|---|---|")
    e1, e1p = P["E1"], Q["E1"]
    w(f"| E1 | {fin['final_model']} の自然 | 成功率（Wilson） | {e1['successes']}/{e1['n']}（{_ci(e1['success_wilson'])}） | — | — | "
      f"{e1p['successes']}/{e1p['n']} |")
    e2, e2p = P["E2"], Q["E2"]
    w(f"| E2 | {fin['final_model']} の P1 | 立ち直りの率（成立が分母） | {e2['recovered']}/{e2['established']}（{_ci(e2['recovery_wilson'])}） | — | — | "
      f"{e2p['recovered']}/{e2p['established']} |")
    for key, e, lab in (("main_A_vs_B", "E3 ①", "本線 R1 対 N1"), ("sync_C_vs_D", "E3 ②", "sync R1 対 N1")):
        d, dp = P["E3"][key], Q["E3"][key]
        a, p, ci = _pair(d)
        w(f"| {e} | {lab} | P1 の立ち直り | {a} | {p}、Holm 後 **{_p(d['holm_p'])}** | {ci} | Holm 後 {_p(dp['holm_p'])} |")
    for e, lab, metric in (("E4", "naive 対 rtc（R1）", "P1 の立ち直り"), ("E5", f"{fin['final_model']} のフィルタあり対なし", "自然の接触の有無"),
                           ("E8", "R2 対 R1+", "P1 の立ち直り")):
        a, p, ci = _pair(P[e])
        w(f"| {e} | {lab} | {metric} | {a} | **{p}** | {ci} | p {_p(Q[e]['mcnemar_exact_p'])} |")
    e6, e6p = _j(OUT / "k1" / "e6_R2_final.json"), _j(OUT / "k1" / "e6_R2_pre.json")
    k6 = round(e6["accuracy_majority"] * e6["pairs"])
    w(f"| E6 | R2、指示と手がかりを一緒に差し替え | 正答率（多数決） | {k6}/{e6['pairs']}（{_ci(_wilson(k6, e6['pairs']))}） | — | — | "
      f"{round(e6p['accuracy_majority'] * e6p['pairs'])}/{e6p['pairs']} |")
    w("")
    w("**予備と結論が逆になった項目**（主要評価項目のとおり、テスト用の値で結論を書く）")
    w("")
    w(f"- **E8**: 2 段目の復帰デモ（R2）の効果は、R1+ に対して確かめられなかった（P1 {_pair(P['E8'])[0]}、p {_p(P['E8']['mcnemar_exact_p'])}）。"
      f"予備の差（p {_p(Q['E8']['mcnemar_exact_p'])}）は、回すたびのばらつき（掲示板 0087: P1 で 30 対中 9 が入れ替わる）の範囲だったと読む")
    blocked = fin["secondary"]["E5"]["blocked_by_filter"]
    w(f"- **E5**: 安全フィルタで、失敗注入なしの接触は減った（{_pair(P['E5'])[0]}、p {_p(P['E5']['mcnemar_exact_p'])}。予備は "
      f"p {_p(Q['E5']['mcnemar_exact_p'])} で差なし）。代償として、フィルタが接近を止め続けて失敗した試行が **{len(blocked)} 本**あった（下の 2）")
    w("")
    # ------------------------------------------------------------------ 自然の成功と E5 の代償
    w("## 2. 失敗注入なし（自然）の成功と、安全フィルタの代償")
    w("")
    w("| 組 | 方策・実行 | 自然の成功 | Wilson 95% |")
    w("|---|---|---|---|")
    lab = {"A": "R1・本線", "B": "N1・本線", "C": "R1・sync", "D": "N1・sync", "E": "R1・rtc d=4", "F": "R2・本線",
           "G": "R1+・本線", "H": "R2・本線・フィルタ切"}
    for s in S:
        n = S[s]["nat"]
        w(f"| {s} | {lab.get(s, s)} | {n['successes']}/{n['n']} | {_ci(n['success_wilson'])} |")
    w("")
    import importlib.util
    spec = importlib.util.spec_from_file_location("e_eval", ROOT / "scripts" / "50_e_eval.py")
    ee = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ee)
    F, G, A = (ee._rows("final", f"{s}_nat") for s in ("F", "G", "A"))
    w("**副次評価項目: 同じ種の対での自然の成功（McNemar、補正なし）**")
    w("")
    w("| 比べるもの | 成功率（x だけ・y だけ） | p | 差の 95% 区間 |")
    w("|---|---|---|---|")
    for (x, y, name) in ((F, G, "R2（F）対 R1+（G）"), (F, A, "R2（F）対 R1（A）")):
        d = ee.paired_binary(x, y, ee.SUCC)
        a, p, ci = _pair(d)
        w(f"| {name} | {a} | {p} | {ci} |")
    w("")
    w("2 段目の学習（R2）で、失敗注入なしの成功が下がった可能性がある。最終モデルは事前の決まりどおり R2 のまま。")
    w("")
    fails = nat_failures("F_nat", blocked)
    forms = collections.Counter(("フィルタで停止" if r["filter_blocked"] else r["form"]) for r in fails)
    w(f"**R2 の自然の失敗 {len(fails)} 本の内訳**（失敗の検出器の最初に確定した形。フィルタが止め続けたものは別に数える）: "
      + "、".join(f"{k} {v}" for k, v in forms.most_common()))
    w("")
    w("| 種 | 目標 | 形 |")
    w("|---|---|---|")
    for r in fails:
        w(f"| {r['seed']} | {r['target']} | {'フィルタで停止（' + r['form'] + '）' if r['filter_blocked'] else r['form']} |")
    w("")
    w("（grasp_miss＝掴み損ね、drop_table＝机への落下、drop_other＝その他の落下、timeout＝どれにも当たらず時間切れ）")
    w("")
    w("**フィルタが接近を止め続けて失敗した試行**（E5。フィルタありで失敗・なしで成功し、フィルタが 5% 以上のこまで働いたもの）")
    w("")
    w("| 種 | 目標 | 働いたこま | 働いたときに最も近かった障害物（目標を除く） | フィルタなしの同じ種 |")
    w("|---|---|---|---|---|")
    bd = blocked_detail("F_nat", "H_nat", blocked)
    for b in bd:
        near = "、".join(f"{k} {v} こま" for k, v in b["nearest_obstacle_when_active"].items())
        w(f"| {b['seed']} | {b['target']} | {b['active_frames']}/{b['frames']}（{_pct(b['safety_active_share'])}） | {near} | "
          f"{'成功' if b['without_filter_success'] else '失敗'}（触れたもの: {'、'.join(b['without_filter_touched']) or 'なし'}） |")
    w("")
    kinds = {("立方体" if k.startswith("cube_") else "箱の壁") for b in bd for k in b["nearest_obstacle_when_active"]}
    w(f"止め続けた相手（働いたときに最も近かった障害物）: {'・'.join(sorted(kinds)) or '—'}（目標外の立方体）。フィルタなしでは、"
      "その立方体に触れながら目標へ届いて成功した。接触を減らす代わりに、隣の立方体に近い目標へ届けなくなることがある、という取引。"
      "映像は同じ設定で回し直した映像（結果は本番と同じ。ビット一致ではない）")
    w("")
    # ------------------------------------------------------------------ 副の指標
    w("## 3. 副次評価項目（補正なしの記述）")
    w("")
    w("| 比べるもの | 自然の成功 | P2 の立ち直り | P3 の立ち直り | P1〜P3 の合計 |")
    w("|---|---|---|---|---|")
    names = {"E3_main": "E3 本線（A 対 B）", "E3_sync": "E3 sync（C 対 D）", "E4_naive_vs_rtc": "E4 naive 対 rtc（A 対 E）",
             "E4_sync_vs_naive": "E4 sync 対 naive（C 対 A）", "E4_sync_vs_rtc": "E4 sync 対 rtc（C 対 E）", "E8": "E8（F 対 G）",
             "E5": "E5（F 対 H）"}
    sec = fin["secondary"]
    for k, nm in names.items():
        if k not in sec:
            continue
        cells = []
        for m in ("nat_success", "P2_recovery", "P3_recovery", "P123_recovery"):
            a, p, _ci2 = _pair(sec[k][m])
            cells.append(f"{a}、p {p}")
        w(f"| {nm} | " + " | ".join(cells) + " |")
    w("")
    w("| 比べるもの | 継ぎ目の跳びの中央値 [m/s]（Wilcoxon p） | 躍度の中央値 [m/s³]（p） |")
    w("|---|---|---|")
    for k in ("E4_naive_vs_rtc", "E4_sync_vs_naive", "E8", "E5"):
        s_, j_ = sec[k]["seam_jump_mean_all"], sec[k]["jerk_rms_all"]
        w(f"| {names[k]} | {s_['x_median']:.3f} 対 {s_['y_median']:.3f}（{_p(s_['wilcoxon_p'])}） | "
          f"{j_['x_median']:.2f} 対 {j_['y_median']:.2f}（{_p(j_['wilcoxon_p'])}） |")
    w("")
    ff = sec.get("E4_failure_forms", {})
    w("E4 の自然な失敗の形: " + "、".join(f"{s} " + "・".join(f"{k} {v}" for k, v in d.items()) for s, d in ff.items()))
    w("")
    w("![E4](e4_tradeoff_final.png)")
    w("")
    w("P1 の反応時間は、対の数が少ない（手が 2 cm 近づかない試行は値がない）。R2 の P1 は、評価で制御が戻る「閉じた直後の低い姿勢」を"
      "ほとんど覆っていない（掲示板 0078・0079）。")
    w("")
    # ------------------------------------------------------------------ E6
    e6c = _j(OUT / "k1" / "e6_R2_final_cue_only.json")
    w("## 4. E6（指示の見分け、R2）")
    w("")
    w(f"- 指示と手がかりを一緒に差し替え: {k6}/{e6['pairs']}（5 回ずつの試料で {_pct(e6['accuracy_samples'])}、中間落ち "
      f"{_pct(e6['mid_drop_rate'], 1)}）")
    w(f"- 手がかりだけ差し替え: 手がかりに従う {_pct(e6c['accuracy_majority'])}、指示文に従う {_pct(e6c['follows_instruction_text_majority'])}")
    w("")
    # ------------------------------------------------------------------ E7
    llm, llm_pre = _j(PL / "llm_final_sentences_v1.json"), _j(PL / "llm_dev_sentences_v1.json")
    jd, jp = _j(PL / "judge_final.json"), _j(PL / "judge_pre.json")
    e7 = _j(PL / "e7_summary_E7_final.json")
    pre_run = _j(PL / "runs" / "E7_pre" / "run.json")
    s7 = e7["summary"]
    w("## 5. E7（上位層: LLM の分解・完了判定・複数手順の通し）")
    w("")
    w("| 項目 | 基準 | 最終評価 | 判定 | 予備 |")
    w("|---|---|---|---|---|")
    w(f"| LLM の分解（最終用の 30 文、一度だけ） | 29/30 以上 | {llm['correct']}/{llm['n']}（{_ci(_wilson(llm['correct'], llm['n']))}） | "
      f"{'届いた' if llm['correct'] >= 29 else '届かなかった'} | {llm_pre['correct']}/{llm_pre['n']}（手直し用の文） |")
    w(f"| 完了判定（照合データ、正例 100・負例 100） | 98% 以上 | {jd['agree']}/{jd['selected']}＝{_pct(jd['agreement'], 1)}（{_ci(_wilson(jd['agree'], jd['selected']))}） | "
      f"{'届いた' if jd['agreement'] >= 0.98 else '**届かなかった**'} | {jp['agree']}/{jp['selected']} |")
    pr = e7["primary"]
    w(f"| 複数手順の通し（「全部片付けて」、3 個とも） | 20 回中 10 以上（G3） | {pr['all_three']}/{pr['n']}（{_ci(pr['wilson95'])}） | "
      f"{'届いた' if pr['meets_G3'] else '届かなかった'} | {pre_run['all_three']}/{pre_run['n']}（戻す動きを入れる前の実行器） |")
    w("")
    w(f"LLM 単体（自前の検査の前）でも {sum(r['llm_steps'] == r['expected'] for r in llm['rows'])}/{llm['n']}。期待する出力のとおりに数えた。外れた文（LLM の出力と返答の文をそのまま）:")
    w("")
    for r in llm["rows"]:
        if not r["correct"]:
            w(f"- id {r['id']}「{r['text']}」: 期待 {json.dumps(r['expected'], ensure_ascii=False)}、出力 "
              f"{json.dumps(r['steps'], ensure_ascii=False)}、返答「{r['reply']}」")
    w("")
    sel = set(jd["selected_ids"])
    dis = [c for c in jd["cases"] if c["id"] in sel and c["truth"] != c["judge"]]
    fp = sum(1 for c in dis if c["judge"])
    w(f"**完了判定の食い違い {len(dis)} 件**（見逃し {len(dis) - fp}・誤って「完了」{fp}）。種類ごとの一致: "
      + "、".join(f"{k} {v[0]}/{v[1]}" for k, v in jd["by_kind"].items()))
    w("")
    for c in dis:
        cd = c["conditions"]
        w(f"- {c['seed']} {c['color']} {c['kind']}: 真値 {'箱の中' if c['truth'] else '箱の外'}・判定 {'完了' if c['judge'] else 'まだ'}"
          f"（待機位置から {cd['retreat_dist_m'] * 100:.1f} cm、条件がそろって {cd['held_s']:.2f} s、指 {'開' if cd['gripper_open'] else '閉'}）")
    w("")
    w("- 食い違いはすべて見逃し（安全側の誤り）で、実際の影響は判定が遅れること。走行の正例の見逃しは、取った時刻（真値の成功から "
      "3.5〜6 s）に手がまだ待機位置へ戻り切っていなかったもの。閾値は学習用の帯だけで決めた値のまま変えていない")
    w(f"- **限界**: 照合データの負例の種類（持って箱の上・箱の縁・箱の脇）に「別の立方体の上に乗った」がない。複数手順の通しでは、"
      f"この形の誤った「完了」が失敗 {pr['n'] - pr['all_three']} 本のうち {s7['judge_false_complete']} 本を占めた。照合データの一致率は、この誤りを測っていない")
    w("")
    w("**複数手順の通し（115000〜115019）の内訳**")
    w("")
    w("| 種 | 結果 | 種 | 結果 |")
    w("|---|---|---|---|")
    bs = e7["by_seed"]
    half = (len(bs) + 1) // 2
    for i in range(half):
        a = bs[i]
        b = bs[i + half] if i + half < len(bs) else {"seed": "", "code": ""}
        w(f"| {a['seed']} | {a['code']} | {b['seed']} | {b['code']} |")
    w("")
    w("（ok＝3 個とも、G2＝2 番目の緑で止まった、x:b＝判定は完了だが青が真値で箱に入っていない、+r2＝2 番目の手順がやり直しで完了）")
    w("")
    for k, v in s7["stopped_by_position"].items():
        w(f"- {k}: 止まった {v['stopped']}/{v['started']}（真値でも置けなかった {v['stopped_truth_not_placed']}）")
    rr = s7["returns"]
    ob = s7["opened_on_return"]
    w(f"- やり直しで完了した手順 {s7['retry_completed_steps']}。待機位置へ戻す動き {rr['n']} 回（やり直しの前 {rr['retry']}・置いた後 {rr['placed']}）、"
      f"着かなかった {rr['not_arrived']}、触れた {rr['touched_cube_or_box']}、置いた後に戻して判定が出た {rr['placed_then_judged']}/{rr['placed']}")
    w(f"- 戻す動きの始めに指を開いた {ob['n']} 回のうち、立方体を挟んでいた {ob['held']}（持ち上がっていた {ob['held_lifted']}、"
      f"別の色 {ob['held_other_color']}）。落ちた先: " + ("、".join(f"{k} {v}" for k, v in ob["landed"].items()) or "なし"))
    w(f"- 別の立方体の上に乗った誤った「完了」: {s7['judge_false_complete']} 件（通しの数で {s7['judge_false_complete']}/{pr['n']}、"
      f"3 番目まで進んだ通し {s7['runs_reaching_step3']}）")
    w(f"- 真値の成功から判定までの中央値 {e7['truth_to_judge_s_median']:.1f} s、1 回の通しは模擬の時間で中央値 {e7['run_sim_s_median']:.0f} s")
    w("")
    rc = _j(PL / "return_compare.json")
    pa = rc["primary"]
    w("**待機位置へ戻す動き（予備評価の結果を見た後に加えた変更、掲示板 0094〜0096）**")
    w("")
    w(f"- 検証（197000〜197019、3 通りの順番）で 3 個とも {pa['all_three_before']}/{pa['n']} → {pa['all_three_after']}/{pa['n']}。"
      "この値は変更のきっかけと同じ種で測ったもので、楽観側に偏りうる")
    ups = sum(1 for p in rc["pairs"] for r in p["by_seed"] if not r["before"].startswith("ok") and r["after"].startswith("ok"))
    downs = sum(1 for p in rc["pairs"] for r in p["by_seed"] if r["before"].startswith("ok") and not r["after"].startswith("ok"))
    w(f"- 同じ種の対で、失敗 → 成功 {ups}、成功 → 失敗 {downs}（回すたびのばらつきは P1 で 30 対中 9 が入れ替わる程度＝0087。検定は付けない）")
    fc = PL / "return_compare_final.json"
    if fc.is_file():
        rf = _j(fc)
        pf = rf["pairs"][0]
        sb, sa = pf["before_summary"], pf["after_summary"]
        st2 = lambda s: next(v for k, v in s["stopped_by_position"].items() if k.startswith("2:"))   # noqa: E731
        up2 = sum(1 for r in pf["by_seed"] if not r["before"].startswith("ok") and r["after"].startswith("ok"))
        dn2 = sum(1 for r in pf["by_seed"] if r["before"].startswith("ok") and not r["after"].startswith("ok"))
        w(f"- **テスト用のシード（{pf['seeds']}）での前後の比較（結果を見た後に加えた比較。どの決定にも使わない）**: 3 個とも "
          f"{sb['all_three']}/{sb['n']}（戻す動きなし）→ {sa['all_three']}/{sa['n']}（あり）、再試行で完了したサブタスク "
          f"{sb['retry_completed_steps']} → {sa['retry_completed_steps']}、2 番目で止まった {st2(sb)['stopped']}/{st2(sb)['started']} → "
          f"{st2(sa)['stopped']}/{st2(sa)['started']}、立方体の上に乗った誤った「完了」 {sb['judge_false_complete']} → {sa['judge_false_complete']}。"
          f"同じシードの対で失敗 → 成功 {up2}、成功 → 失敗 {dn2}。テスト用のシードでは、3 個とも入る数は増えなかった")
    else:
        w(f"- テスト用の種では、戻す動きを入れた実行器で {pr['all_three']}/{pr['n']}。入れる前の実行器ではこの種を回していない")
    w("")
    return "\n".join(L) + "\n"


def main() -> int:
    try:
        fp = goals.fingerprint()          # 変更履歴にない版の目標書なら作らない（0106）
    except goals.GoalsMismatch as e:
        print(f"目標書の照合に通らないので止める: {e}", file=sys.stderr)
        return 1
    py = sys.executable
    for args in (["scripts/50_e_eval.py", "report", "--stage", "final"], ["scripts/51_planner.py", "e7-summary", "--tag", "E7_final"]):
        subprocess.run([py, *args], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    text = build(fp)
    (RES / "results.md").write_text(text, encoding="utf-8")
    # 監督・決裁が読めるように、Git で管理する写しも置く（outputs は Git 管理外）
    docs = ROOT / "docs" / "results"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "results.md").write_text(text, encoding="utf-8")
    (docs / "e4_tradeoff_final.png").write_bytes((RES / "e4_tradeoff_final.png").read_bytes())
    print(RES / "results.md", docs / "results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
