"""段階 3 の最終評価の詳しい表（0146 の 3）: docs/results/results_s3.md を記録から作る。数字は手で書かない。

    .venv\\Scripts\\python.exe scripts\\54_results_s3.py

読むもの: outputs/results/v2_e_report_s3.json（87_v2_e.py report --stage s3）、outputs/v2eval/V3S3（各組の run.json・E6・E7）、
docs/results/s3_round{1,2}_verify.json（1・2 周目の検証）、outputs/results/v2_safety_main_s3_V3SF.json（安全フィルタの判定）。
値の丸めと E7 の集計は説明資料（60_paper.py）と同じ関数を使う。開発中の記号（モデル名など）は表に出さない（0147 の 2）。
"""
import importlib.util
import json
import pathlib
import sys
import time

from recovla.common import config, goals

ROOT = config.ROOT


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pm = _mod("paper", ROOT / "scripts" / "60_paper.py")
ee = _mod("e_eval", ROOT / "scripts" / "50_e_eval.py")
RES, S3 = pm.RES, pm.S3
FREEZE_TAG = "v3-s3-freeze"
MODEL = {"R1v3": "復帰デモあり", "N1v3": "復帰デモなし"}
MODE = {"naive": "非同期", "sync": "同期", "rtc": "非同期＋RTC"}
JA = {"red": "赤", "green": "緑", "blue": "青"}
_j = pm._j


def _p(p):
    return "—" if p is None else pm.pv(p)


def _pair(d):
    """同じシードの対の 1 行: x の率 対 y の率（x だけ・y だけ、対の数）、p、差の 95% 信頼区間。"""
    if not d.get("pairs"):
        return "—", "—", "—"
    return (f"{pm.pct(d['x_rate'])} 対 {pm.pct(d['y_rate'])}（{d['x_only']}・{d['y_only']}、{d['pairs']} 対）",
            _p(d["mcnemar_exact_p"]), pm.dci(d))


def set_label(r):
    return (f"{MODEL[r['model']]}・{MODE[r['mode']]}（{r['exec_interval']} ステップごと）・"
            f"安全フィルタ{'あり' if r['safety'] else 'なし'}")


def build(fp: dict) -> str:
    rep = _j(RES / "v2_e_report_s3.json")
    P, sec, sets = rep["primary"], rep["secondary"], rep["sets"]
    runs = {s: _j(S3 / f"{s}_nat" / "run.json") for s in sets}
    lab = {s: set_label(r) for s, r in runs.items()}
    v = pm.values()
    L = []
    w = L.append
    w("# 最終評価の結果（段階 3）")
    w("")
    w(f"このファイルは `scripts/54_results_s3.py` が記録から作った（{time.strftime('%Y-%m-%d %H:%M')}）。数字は手で書いていない。"
      f"評価はコード・チェックポイント・データを凍結した後（`{FREEZE_TAG}`、SHA-256 の一覧は `docs/freeze/s3_hashes.json`）に、"
      f"テスト用のシード範囲（{runs['A']['trials'].split(':')[1]}〜）で一度だけ回した。最終的な構成（モデル・実行方式・安全フィルタの有無）は、"
      "検証用のシード範囲で事前に定めた規則に従って決めた。")
    w("")
    w(f"目標書: v{fp['version']}（タグ {fp['tag']}、`{fp['path']}` の SHA-256 `{fp['sha256']}`）。"
      "実行系はセンサの出力と設置情報のみを用いる。")
    w("")
    w("**注記**")
    w("")
    w("- 段階 3 の 1 周目（エキスパートの終盤の接近を遅くしたデータで学習し直す）の採否は、検証の結果を見た後に、事前に定めた 4 条件のうち"
      "前提が崩れた 1 条件（下り始めのずれ）を除いて決めた（下の 6）")
    w("- 段階 1・2 の最終評価の実行方式（非同期・10 ステップごと・安全フィルタあり）と、段階 3 の実行方式（非同期・6 ステップごと・"
      "安全フィルタなし）は異なる")
    w("- 安全フィルタありの組（下の H）は、テスト用の評価の結果を見た後に追加した比較である")
    w("")
    # ------------------------------------------------------------------ 1. 主要評価項目
    w("## 1. 主要評価項目")
    w("")
    w("| 項目 | 比べるもの | 指標 | 結果 | p | 差の 95% 信頼区間（ポイント） |")
    w("|---|---|---|---|---|---|")
    e1, e2 = P["E1"], P["E2"]
    # 項目の列は言葉で書く（評価項目の記号は提出物で使わない。check_submission.py が文章のファイルの記号を止める）
    w(f"| 意図的な失敗なしの成功率 | 最終構成（A） | 意図的な失敗なしの成功率（Wilson） | {e1['successes']}/{e1['n']}（{pm.ci(e1['success_wilson'])}） | — | — |")
    w(f"| 把持失敗からの復帰率 | 最終構成（A） | 把持失敗からの復帰率（実際に失敗した試行が分母） | {e2['recovered']}/{e2['established']}"
      f"（{pm.ci(e2['recovery_wilson'])}） | — | — |")
    for key, e, nm in (("main_A_vs_B", "復帰デモの効果（非同期実行）", "復帰デモあり 対 なし、非同期（A 対 B）"),
                       ("sync_C_vs_D", "復帰デモの効果（同期実行）", "復帰デモあり 対 なし、同期（C 対 D）")):
        d = P["E3"][key]
        a, p, c = _pair(d)
        w(f"| {e} | {nm} | 把持失敗からの復帰（両モデルで実際に失敗した対） | {a} | {p}、Holm 補正後 **{_p(d['holm_p'])}** | {c} |")
    a, p, c = _pair(P["E4"])
    w(f"| RTC の効果 | 非同期 対 非同期＋RTC（A 対 E） | 把持失敗からの復帰 | {a} | {p} | {c} |")
    a, p, c = _pair(P["E5"])
    w(f"| 安全フィルタの効果 | 安全フィルタなし 対 あり（A 対 H） | 意図的な失敗なしで障害物に接触した試行 | {a} | {p} | {c} |")
    w(f"| 指示文とキューへの追従 | 指示文とキューを一緒に差し替え | 差し替えた色へ向かった対（多数決） | {v['e6_k']}/{v['e6_n']}（{v['e6_ci']}） | — | — |")
    w(f"| 3 個の連続タスク | 「全部片付けて」、3 個の連続タスク | 3 個すべてを収納 | {v['e7_k']}/{v['e7_n']}（{v['e7_ci']}） | — | — |")
    w("")
    w("p は同じシードの対による McNemar の正確検定。復帰デモの効果の 2 つ（非同期実行・同期実行）は Holm 法で補正した。それ以外の比較は補正していない。"
      "率の括弧内は（x だけ成功・y だけ成功、対の数）。")
    w("")
    # ------------------------------------------------------------------ 2. 組ごとの結果
    w("## 2. 組ごとの結果")
    w("")
    w("| 組 | 構成 | 意図的な失敗なしの成功 | 接触した試行 | 把持失敗: 実際に失敗／復帰 | 落下: 実際に失敗／復帰 | 置き損ね: 実際に失敗／復帰 |")
    w("|---|---|---|---|---|---|---|")
    for s, d in sets.items():
        n = d["nat"]
        cells = []
        for part in ("P1", "P2", "P3"):
            q = d.get(part) or {}
            cells.append(f"{q['established']}/{q['n']}／{q['recovered']}" if q.get("n") else "—")
        w(f"| {s} | {lab[s]} | {n['successes']}/{n['n']}（{pm.ci(n['success_wilson'])}） | {n['contact_trials']}/{n['n']} | "
          + " | ".join(cells) + " |")
    w("")
    w(f"意図的な失敗なしは {runs['A']['trials'].split(':')[2]} 配置 × 3 色、意図的な失敗ありは種類ごとに {sets['A']['P1']['n']} 試行。"
      "「実際に失敗」は意図的な失敗によって実際に失敗が生じた試行、「復帰」はそのうち制限時間内に目標を箱へ収納した試行。"
      "組 H は意図的な失敗なしのみ回した。")
    w("")
    # ------------------------------------------------------------------ 3. 副次評価項目
    w("## 3. 副次評価項目（補正なしの記述）")
    w("")
    names = {"E3_main": "復帰デモあり 対 なし、非同期（A 対 B）", "E3_sync": "復帰デモあり 対 なし、同期（C 対 D）",
             "E4_naive_vs_rtc": "非同期 対 非同期＋RTC（A 対 E）", "E4_sync_vs_naive": "同期 対 非同期（C 対 A）",
             "E5": "安全フィルタなし 対 あり（A 対 H）"}
    w("| 比べるもの | 意図的な失敗なしの成功 | 接触 | 落下からの復帰 | 置き損ねからの復帰 | 3 種類の合計の復帰 |")
    w("|---|---|---|---|---|---|")
    for k, nm in names.items():
        cells = []
        for m in ("nat_success", "nat_contact", "P2_recovery", "P3_recovery", "P123_recovery"):
            a, p, _c = _pair(sec[k][m])
            cells.append("—" if a == "—" else f"{a}、p {p}")
        w(f"| {nm} | " + " | ".join(cells) + " |")
    w("")
    w("| 比べるもの | 躍度の二乗平均平方根の中央値 [m/s³]（Wilcoxon p、対の数） |")
    w("|---|---|")
    for k, nm in names.items():
        j = sec[k]["jerk_rms_all"]
        if j.get("pairs"):
            w(f"| {nm} | {j['x_median']:.2f} 対 {j['y_median']:.2f}（{_p(j['wilcoxon_p'])}、{j['pairs']}） |")
    w("")
    # ------------------------------------------------------------------ 4. E6・E7
    w("## 4. 目標位置のキューと 3 個の連続タスク")
    w("")
    w(f"- 指示文とキューを一緒に差し替え: 差し替えた色へ {v['e6_k']}/{v['e6_n']}")
    w(f"- キューだけを差し替え: キューの色へ {v['e6c_cue']}/{v['e6c_n']}、指示文の色へ {v['e6c_text']}/{v['e6c_n']}。"
      "方策は行き先を指示文ではなくキューで決めている")
    w("")
    s7 = pm.e7_summary()
    w(f"3 個の連続タスク（指示「全部片付けて」、{v['e7_n']} 配置）: 3 個すべてを収納 {v['e7_k']}/{v['e7_n']}（{v['e7_ci']}）。"
      f"再試行で完了したサブタスク {v['e7_retry']}。完了判定の誤りは、未完了を完了とした {v['e7_fp']} 件・完了を未完了とした {v['e7_fn']} 件。"
      "停止した位置: " + "、".join(f"{k + 1} 番目のサブタスク {n}" for k, n in sorted(s7["stopped_at"].items())))
    w("")
    w("| シード | 結果 | 箱に入った色 |")
    w("|---|---|---|")
    for p in sorted((S3 / "E7_R1v3").glob("run_00??.json")):
        r = _j(p)
        if r["all_three_in_box"]:
            res = "3 個とも収納"
        elif r.get("stopped"):
            res = f"{int(r['stopped']['step']) + 1} 番目のサブタスクで停止"
        else:
            res = "停止せずに終了（未収納あり）"
        inbox = "・".join(JA[c] for c in r["plan"]["steps"] if r["final_in_box"].get(c)) or "なし"
        w(f"| {r['seed']} | {res} | {inbox} |")
    w("")
    # ------------------------------------------------------------------ 5. 監査と知覚
    e9, e10 = rep["new"]["E9"], rep["new"]["E10"]
    t = e9["total"]
    w("## 5. 実行系の監査と知覚の精度")
    w("")
    w(f"全 {t['trials']:,} 試行で、真の状態の参照 {t['g1']} 件・指令の上限超過 {t['g3']} 件。")
    w("")
    w("| 組 | 画像から推定した立方体の位置の誤差: 中央値・95 パーセンタイル [mm] | 箱の中かどうかの一致率 |")
    w("|---|---|---|")
    for s, d in e10.items():
        w(f"| {s} | {d['median_m'] * 1000:.0f}・{d['p95_m'] * 1000:.0f}（{d['n']:,} 件） | {pm.pct(d['in_box_agreement'])} |")
    w("")
    # ------------------------------------------------------------------ 6. 段階 3 の判断の記録（検証用のシード範囲）
    r1, r2 = _j(ROOT / "docs" / "results" / "s3_round1_verify.json"), _j(ROOT / "docs" / "results" / "s3_round2_verify.json")
    sf = _j(RES / "v2_safety_main_s3_V3SF.json")
    w("## 6. 段階 3 の判断の記録（検証用のシード範囲。テスト用の値ではない）")
    w("")
    w("| 判断 | 比べたもの | 意図的な失敗なしの成功（99 試行、あり・なし） | 「あり」の把持失敗: 実際に失敗／復帰 | 「あり」だけ復帰・「なし」だけ復帰 | 採否 |")
    w("|---|---|---|---|---|---|")
    ns, pr = r1["natural_success"], r1["p1_recovery_R"]
    for tag, k in (("学習し直す前", "v2"), ("学習し直した後", "v3")):
        rk, nk = f"R1{k}", f"N1{k}"
        e = r1["e3_like"][k]
        w(f"| 1 周目 | {tag}（あり・なし） | {ns[rk]}・{ns[nk]} | {pr[rk]['established']}／{pr[rk]['recovered']} | {e['r_only']}・{e['n_only']} | "
          f"{'採る（下の注）' if k == 'v3' else '—'} |")
    ns2, e2r = r2["natural_success"], r2["e3_like"]
    for tag, rk, nk, k in (("1 周目のモデル", "R1v3", "N1v3", "round1"), ("引き継ぎの区間を加えたモデル", "R2v3", "N2v3", "round2")):
        e = e2r[k]
        w(f"| 2 周目 | {tag}（あり・なし） | {ns2[rk]}・{ns2[nk]} | {e['r_established']}／{e['r_recovered']} | {e['r_only']}・{e['n_only']} | "
          f"{'採らない' if k == 'round2' and not r2['adopt'] else '—'} |")
    w("")
    w("- 1 周目の規則（検証の前に定めたもの）: (1) 下り始めのずれ（最初にグリッパを閉じる前で、指先が立方体より 130 mm 以上高い"
      "最後の時刻における、指先と立方体の水平方向の差のうちロボットの根元から立方体へ向かう成分）の中央値が、学習し直す前より 15 mm 以上 0 に近づく、(2) 意図的な失敗なしの成功が下がらない、(3) 把持失敗からの復帰率が "
      "10 ポイント以上下がらない、(4) 実行系の監査の違反が 0。下り始めのずれの条件は、実行方式を変えたことで前提"
      f"（変える前のモデルのずれが約 21 mm）が崩れ、満たせなかった（変える前 {abs(r1['descent_offset_mm_median']['R1v2']):.1f} mm → "
      f"後 {abs(r1['descent_offset_mm_median']['R1v3']):.1f} mm）。この条件を除いて採ることを、結果を見た後に判断した。"
      "改善は「あり」「なし」の両方に生じており、「あり」と「なし」の差の改善は把持失敗で「あり」だけ復帰した対の増加に限られる")
    w("- 2 周目（1 周目のモデルで方策を実行し、止まった・位置がずれたままグリッパを閉じようとした・ランダムな時刻のいずれかで"
      "エキスパートへ引き継いで立て直す区間を記録し、「あり」「なし」の両方の学習データに 120 本ずつ加えて学習し直す）の規則: "
      "(1) 意図的な失敗なしの成功が 1 周目のモデルより 5 試行以上多い、(2) 監査の違反が 0、(3) 把持失敗からの復帰率が 10 ポイント以上下がらず、"
      "「あり」だけ復帰した対と「なし」だけ復帰した対の差が 1 周目より 2 以上小さくならない、(4) 最初に閉じた時点で位置が 15 mm を超えて"
      "外れていた試行が増えない。(1)(3) を満たさず採らなかった。最終モデルは 1 周目のもの")
    sc, ct = sf["success"], sf["contact"]
    w(f"- 安全フィルタ（ありの成功率がなしより 3 ポイント以上低ければ最終構成から外す、と事前に定めた）: あり {pm.pct(sc['x_rate'])}・なし {pm.pct(sc['y_rate'])}（{sc['pairs']} 対、"
      f"{sf['drop_points']:.1f} ポイントの低下）のため最終構成から外した。接触は あり {round(ct['x_rate'] * ct['pairs'])}/{ct['pairs']}・"
      f"なし {round(ct['y_rate'] * ct['pairs'])}/{ct['pairs']}")
    w("")
    return "\n".join(L) + "\n"


def main() -> int:
    try:
        fp = goals.fingerprint()
    except goals.GoalsMismatch as e:
        print(f"目標書の照合に通らないので止める: {e}", file=sys.stderr)
        return 1
    text = build(fp)
    (RES / "results_s3.md").write_text(text, encoding="utf-8")
    docs = ROOT / "docs" / "results" / "results_s3.md"
    docs.write_text(text, encoding="utf-8")
    print(RES / "results_s3.md", docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
