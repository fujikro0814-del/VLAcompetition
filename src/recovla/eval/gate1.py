"""段階 4 の関門 1 の判定（束 1 の二重集計の片方）。configs/s4_gates.json の数値と規則を、測った値に機械的に当てはめる。

    from recovla.eval import gate1 as G1
    gates = G1.load_gates("configs/s4_gates.json")
    inp = G1.build_input(rtc_metrics=..., e7_summary=..., start_summary=..., recovery_score=..., k=...)   # 手元の診断の出力から
    res = G1.evaluate(gates, inp)            # 関門ごとの合否・使った値と閾値・枝、候補の選び方、最後の結論
    text = G1.summary_md(res)                # 1 枚の要約（利用者向けの日本語）

入力の形は docs/stage4/gate1_input_schema.md（schema "recovery_vla.s4_gate1_input/1"）。

決まり（この道具が自分で決めないこと）:
  - 数値は s4_gates.json の文から読む。文の形はこのファイルの SPECS に全文で書いておき、全文が一致したときだけ読む。
    文が少しでも違えば、その規則は「読めない」として当てはめない（その条件は判定できない）。数値も規則も推し量らない。
  - 機械が読める欄（id・metric・rule・when）があっても、基準の数値がない規則（C.b2 の「差が大きい」など）は当てはめず、
    NOT_MACHINE_READABLE に並べて結果に載せる（ファイルの行つき）。作者が決める。
  - 値がない・形が違う・分母が 0 の条件は「判定できない」。合格にしない。
  - 判定できない条件は三値の論理で扱う: all_of は 1 つでも不合格なら不合格、そうでなく 1 つでも判定できなければ判定できない。
    s4_gates.json には判定できない条件があるときの枝の規則がないので、結論がそれで変わる所は「判定できない」として止める
    （結論が変わらない所、たとえば開始状態の候補が残って滑りの合否によらず束 4 が決まる所は、そのまま出す）。
  - 比べるのは分数（fractions.Fraction）で行う。割合は k/n のまま、小数は書かれた 10 進の値のまま比べるので、
    ちょうど閾値の値は「>=」「<=」で合格になる（浮動小数の丸めで落ちない）。
  - 手元の診断の関数（recovla.diag）の判定の部分は使わない。指標の値（件数・中央値）だけを入力にもらう。
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from fractions import Fraction

GATES_SCHEMA = "recovery_vla.s4_gates/1"
INPUT_SCHEMA = "recovery_vla.s4_gate1_input/1"
RESULT_SCHEMA = "recovery_vla.s4_gate1_result/1"
# 掲示板 0162 の表の SHA-256（改行を LF にした中身。Windows の作業コピーは CRLF なので両方を出して比べる）。
# 改訂 3（ブロック 13）で bands.allocations の bundle6_u4 の行だけを変えた版。関門の数値・規則の文は 0153 の版と同じ。
# 掲示板 0153 の値（改訂 3 の前）: cf2e8c96a205bf10126bdf647bdbf2e8b9d5a315ebbe881c03bafac82b12940d
POSTED_SHA256 = "dcf0dd4c1b2a4f906936bf478f012d3aa6e9a51a81aeb12c6cdcbcf35f33362d"
POSTED_BOARD = "0162"
PASS, FAIL, UNDET = "pass", "fail", "undetermined"
JA = {PASS: "合格", FAIL: "不合格", UNDET: "判定できない"}
HIT = {PASS: "当たる", FAIL: "当たらない", UNDET: "判定できない"}       # 分岐の規則（when）の読み方
RATE_TOL = 5e-5                 # 入力の rate（4 桁に丸めた値）と k/n の食い違いの許し


# ------------------------------------------------------------------ 規則の文（全文が一致したときだけ読む）
# (規則の名前, s4_gates.json の中の場所, 条件の metric の欄の値（None は見ない）, 文の形)
# 場所の {"id": ...} は「リストの中で id がその値の要素」。文の形の {名前} は数値、{名前:word} は英数字の名前、
# {名前:list} は「、」区切りの名前の並び、{名前:dots} は「・」区切りの整数の並び。ほかの文字はそのまま一致させる。
def _c(gate, cid, key="continue"):
    return ("gates", gate, key, "all_of", {"id": cid}, "rule")


def _b(gate, bid):
    return ("gates", gate, "branching_rule", {"id": bid}, "when")


SPECS = [
    # 関門 R
    ("R.c1", _c("R", "R.c1"), "radial_gap_mm", "設定の中央値 >= naive の中央値 - {a} mm"),
    ("R.c2", _c("R", "R.c2"), "move_ratio", ">= {a}"),
    ("R.c3", _c("R", "R.c3"), "natural_success_30", "設定の成功数（{L} s の採点）>= naive の成功数（同） - {a}"),
    ("R.c4", _c("R", "R.c4"), "seam_jump_mps", "<= {a}"),
    ("R.candidates", ("gates", "R", "candidates"), None,
     "naive と current_repro を除く {n} 設定（{names:list}）。naive は基準。current_repro は崩れた基準で、採っても変化がない"),
    ("R.action", ("gates", "R", "continue", "action"), None, "R.c1〜c4 を全部満たす設定のうち、次の順位で上位 {top} つを束 2 の B2 へ送る"),
    ("R.rank1", ("gates", "R", "continue", "ranking", 0), None, "成功数が多い"),
    ("R.rank2", ("gates", "R", "continue", "ranking", 1), None, "継ぎ目の速度の跳びが小さい"),
    ("R.rank3", ("gates", "R", "continue", "ranking", 2), None, "半径方向の差の絶対値が小さい"),
    ("R.rank4", ("gates", "R", "continue", "ranking", 3), None, "なお同じなら設定の宣言順（{order:list}）"),
    ("R.stop", ("gates", "R", "stop_branch", "when"), None, "R.c1〜c4 を全部満たす設定が {zero} 個"),
    ("R.b1", _b("R", "R.b1"), None,
     "影の推論の『先の計画が naive より {a} mm 以上短い』と、X2 の『{rows:dots} 行先のどれかで予測の不足が {b} mm 以上』が両方出る"),
    ("R.b2", _b("R", "R.b2"), None, "上位 {a} つが {b} つしかない（合格が {c} 設定）"),
    # 関門 T
    ("T.E0_ref_lift", ("gates", "T", "reference_rule", "E0_ref_lift"), None,
     "max(E0_run1 の first_close_lift, E0_run2 の first_close_lift)。大きいほうを基準にして、通りにくい側に倒す"),
    ("T.d_E0", ("gates", "T", "reference_rule", "d_E0"), None, "|E0_run1 の first_close_lift - E0_run2 の first_close_lift|"),
    ("T.required_margin", ("gates", "T", "reference_rule", "required_margin"), None, "max({a}, d_E0 + {b})"),
    ("T.c1", _c("T", "T.c1"), "first_close_lift", "その腕の値 >= {a}"),
    ("T.c2", _c("T", "T.c2"), "first_close_lift", "その腕の値 >= E0_ref_lift + required_margin"),
    ("T.c3", _c("T", "T.c3"), "all_three_true", "その腕の 3 個とも（真値）の本数 >= E0_run1 と E0_run2 の大きいほうの本数"),
    ("T.action", ("gates", "T", "continue", "action"), None,
     "EH と ES の両方が合格したときは first_close_lift が大きいほうの戻し先を束 2 の v3 (a) へ送る（同じなら {tie:word}）。片方だけ合格ならその戻し先"),
    ("T.stop", ("gates", "T", "stop_branch", "when"), None, "EH も ES も T.c1〜c3 を満たさない"),
    ("T.b1", _b("T", "T.b1"), None, "EO の『緑を 1 番目にしたときの 1 回目の閉じの持ち上がり』が {a} 以下"),
    ("T.b2", _b("T", "T.b2"), None, "d_E0 > {a}"),
    # 関門 S
    ("S.high", ("gates", "S", "effect_present_rule", "all_of", 0, "rule"), None, "対比の高いほうの plus_y_shift >= {a}"),
    ("S.diff", ("gates", "S", "effect_present_rule", "all_of", 1, "rule"), None, "対比の差 >= {a}"),
    ("S.p", ("gates", "S", "effect_present_rule", "all_of", 2, "rule"), None,
     "同じ種の {n} 対で、片側のフィッシャーの正確検定 p < {a}（ふるい。主張の検定ではない）"),
    ("S.pose.value", ("gates", "S", "contrasts", "pose_effect", "value"), None, "plus_y_shift({hi:word}) - plus_y_shift({lo:word})"),
    ("S.pose.fix", ("gates", "S", "contrasts", "pose_effect", "fix"), None, "先客を格子の上にそろえる"),
    ("S.prior.value", ("gates", "S", "contrasts", "prior_effect", "value"), None, "plus_y_shift({hi:word}) - plus_y_shift({lo:word})"),
    ("S.prior.fix", ("gates", "S", "contrasts", "prior_effect", "fix"), None, "姿勢を {start:word} にそろえる"),
    # 移植の腕
    ("XPL.rep", ("transplant_arm", "reproduces_rule", "rule"), None,
     "{arm:word} の plus_y_shift >= {a}（E7 は 0.81、単発の待機位置始まりは 0.07 なので、その中間より上）"),
    ("XPL.pose", ("transplant_arm", "pose_cause_rule", "rule"), None,
     "XPL.rep を満たし、{other:word} の plus_y_shift が {base:word} より {a} 以上低い。かつ 2 回の回しのそれぞれ（各 {n} 試行）でも {b} 以上低い（再現の守り）"),
    ("XPL.prior", ("transplant_arm", "prior_cause_rule", "rule"), None,
     "XPL.rep を満たし、{other:word} の plus_y_shift が {base:word} より {a} 以上低い。かつ 2 回の回しのそれぞれ（各 {n} 試行）でも {b} 以上低い"),
    # 関門 C
    ("C.c1", _c("C", "C.c1"), None, "{variant:word} の R の {L} s の復帰 >= {a}"),
    ("C.c2", _c("C", "C.c2"), None, "{variant:word} の R-N（{L} s 以内に両方で成立した対、{L2} s の採点）>= {a}"),
    ("C.c3", _c("C", "C.c3"), None, "{L} s 以内に両方で成立した対の数 >= {a}"),
    ("C.stop", ("gates", "C", "stop_branch", "when"), None, "C.c1〜c3 のどれかを満たさない（成立した対が {a} 未満のときを含む）"),
    ("C.b1", _b("C", "C.b1"), None, "C が合格し、S の分岐でも開始状態が候補"),
    # 関門 K
    ("K.c1", _c("K", "K.c1"), None, "{n} 試行が 2 回とも最後まで終わり、種・配置・色の対応が一致している"),
    ("K.stop", ("gates", "K", "stop_branch", "when"), None, "K.c1 を満たさない（途中で止まった・対応が崩れた）"),
    ("K.b1", _b("K", "K.b1"), None, "d0_{L1} <= {a} かつ d0_{L2} <= {b}"),
    ("K.b2", _b("K", "K.b2"), None, "d0_{L1} > {a} または d0_{L2} > {b}（通りにくい側）"),
    # 束 4 の候補の選び方（上から順に、最初に当たった行で止まる）
    ("cand.1", ("candidate_selection", "rows", 0, "when"), None, "XPL.rep を満たさない（移植しても +y のずれが再現しない）"),
    ("cand.1.action", ("candidate_selection", "rows", 0, "action"), None,
     "原因は実行系の内部状態（途中での手順の切り替えなど）と見る。開始状態のデータは作らない。手順の切り替えで実行系を作り直す E7 の腕 EX を "
     "D-E7 の 40 種（190300〜190339）に足して回す（1 手順 30 s で +約 3.6 プロセス時間）。束 4 の開始状態の要因は扱わない"),
    ("cand.2", ("candidate_selection", "rows", 1, "when"), None, "S.prior と XPL.prior の両方を満たす（先客で再現）"),
    ("cand.3", ("candidate_selection", "rows", 2, "when"), None, "S.pose と XPL.pose の両方を満たす（姿勢で再現）。行 2 は満たさない"),
    ("cand.3.action", ("candidate_selection", "rows", 2, "action"), None,
     "T が合格（EH か ES が T.c1〜c3 を満たす）なら『実行器で足りる』。戻し先は束 2 の v3 (a) へ。データにするのは研究の道。"
     "T が不合格なら、姿勢では説明がつかない矛盾として束 4 に進まず、EX を足す（規則役の補い。needs_decision 参照）"),
    ("cand.4", ("candidate_selection", "rows", 3, "when"), None, "上のどれにも当たらない（どちらも再現しない）"),
    ("slip.1", ("candidate_selection", "slip_rule", 0, "when"), None, "関門 C が合格"),
    ("slip.2", ("candidate_selection", "slip_rule", 1, "when"), None, "開始状態の候補（prior_cube）と滑りの候補が両方残る"),
    ("slip.3", ("candidate_selection", "slip_rule", 2, "when"), None, "開始状態の候補がなく、滑りだけ残る"),
]

# 機械が読める形で書かれていないので当てはめない規則（結果に載せ、作者が決める）。needle はファイルの行を探す文字列
NOT_MACHINE_READABLE = [
    {"where": "gates.C.branching_rule[C.b2].when", "needle": '"id": "C.b2"',
     "why": "「fall_as_is と fall_with_hold の差が大きい」の基準（数値）がない。両方の値を並べるだけにした（報告だけの枝）"},
    {"where": "gates.S.stop_branch.when", "needle": '"when": "S.pose も S.prior も',
     "why": "「移植の腕も原因を示さない」が XPL.rep・XPL.pose・XPL.prior のどれ（の組み合わせ）を指すか欄にない。"
            "S のやめる枝は当てはめず、開始状態の結論は candidate_selection の行で出す"},
    {"where": "metrics.move_ratio.aggregate", "needle": '"move_ratio":',
     "why": "中央値か平均かが欄になく『二重集計役が実装前に掲示』とだけある。R.c2 には入力の move_ratio（掲示 0154 の決定 1 と"
            " d_rtc_settings.json の definitions_before_run に合わせ、最初の 30 s に限った中央値を渡す決まり）をそのまま当てる"},
    {"where": "metrics._time_note", "needle": '"_time_note"',
     "why": "30 s より後に最初の閉じが起きた試行を、+y のずれ・持ち上がりの分母に入れるか外すかの規則がない（件数を報告とだけある）。"
            "入力の値をそのまま使い、件数（first_close_after_30s）があれば並べる"},
    {"where": "candidate_selection.rows[order=3].action の needs_decision", "needle": '"order": 3',
     "why": "参照先の needs_decision という欄が s4_gates.json のどこにもない。文のとおり T が不合格なら EX を足すとして当てはめた"},
    {"where": "candidate_selection.rows[order=1・3] の EX", "needle": '"order": 1',
     "why": "EX を足して回した後に、その結果をどう判定に使うかの規則がない。この道具は『EX を足す』までを出す"},
    {"where": "gates.K.e7_extension.measure", "needle": '"e7_extension"',
     "why": "E7 の回し直しのぶれ（食い違い率・d_E0・rho_hat）に条件も枝もない。入力にあれば並べるだけにした"},
    {"where": "e7_sample_size_rule", "needle": '"e7_sample_size_rule"',
     "why": "検出力の計算（rho_hat と p0 から McNemar の食い違いの確率を作る方法）が文章だけ。関門 1 の判定に使わず、D2 で作者が承認する項目なので実装しない"},
    {"where": "bundle4_gates（D・G2・G3）", "needle": '"bundle4_gates"',
     "why": "条件が文章だけ（all_of の要素に id・metric・数値の欄がない）。関門 1 の範囲の外なので実装しない"},
    {"where": "gates.R.report_only_metrics・gates.T.secondary_metrics・gates.C.primary_metrics の R_minus_N_recovery_30s",
     "needle": '"report_only_metrics"',
     "why": "metrics に定義がない名前がある（descent_horizontal_correction_all_trials、far_miss_reopen_time、success_by_judge、"
            "mean_stored_count、success_at_k、intervention_counts_by_kind、R_minus_N_recovery_30s）。R_minus_N は C.c2 の文と "
            "gates.C.denominator から読んだ。ほかは判定に使わないので読まない"},
]

# s4_gates.json の外の定義を合わせて読んだ所（作者が確かめる）
OUTSIDE_DEFINITIONS = [
    {"id": "R.b1", "needle": '"id": "R.b1"',
     "note": "影の推論の側の『どれかの h（10・20・30・40 行先）の中央値 >= 15 mm』は s4_gates.json の文になく、"
             "docs/stage4/bundle1_defs/d_rtc_settings.json の definitions_before_run.shadow.h_rule（掲示 0154 の決定 2）にある。"
             "X2 の側の行の並び（文にある）と同じ並びで当てはめた"},
    {"id": "R.ranking", "needle": '"ranking"',
     "note": "『成功数が多い』の採点の時間が文にない。time_limits.scoring.primary_s（30）と R.c3 の 30 s の採点に合わせた。"
             "『継ぎ目の速度の跳び』も R.c4 に当てた値（最初の 30 s）を使う（metrics._time_note の『関門は 30 s に限った値』）"},
    {"id": "R.c3・T.c3", "needle": '"id": "R.c3"',
     "note": "数（成功数・3 個ともの本数）を比べる条件で、比べる 2 つの試行数が違うときは判定できないとした"
             "（-4 は 30 対 30、T.c3 は同じ 40 種の前提の数なので）。規則に書かれていない守り"},
]


# ------------------------------------------------------------------ 読み込み
def load_gates(path) -> dict:
    """s4_gates.json を読む。結果に載せるために、場所と SHA-256（そのままのバイト列と、改行を LF にしたもの）を添える。"""
    p = pathlib.Path(path)
    raw = p.read_bytes()
    g = json.loads(raw.decode("utf-8"))
    lf = raw.replace(b"\r\n", b"\n")
    g["__file__"] = {"path": str(p), "sha256_bytes": hashlib.sha256(raw).hexdigest(), "sha256_lf": hashlib.sha256(lf).hexdigest(),
                     "lines": lf.decode("utf-8").split("\n")}
    return g


def _get(g, path):
    cur = g
    for k in path:
        if isinstance(k, dict):
            (kk, vv), = k.items()
            hit = [x for x in cur if isinstance(x, dict) and x.get(kk) == vv]
            if len(hit) != 1:
                raise KeyError(f"{kk}={vv} の要素が {len(hit)} 個")
            cur = hit[0]
        else:
            cur = cur[k]
    return cur


def _path_str(path) -> str:
    out = ""
    for k in path:
        if isinstance(k, dict):
            out += "[" + ",".join(f"{a}={b}" for a, b in k.items()) + "]"
        elif isinstance(k, int):
            out += f"[{k}]"
        else:
            out += ("." if out else "") + k
    return out


_NUM = r"-?\d+(?:\.\d+)?"
_KIND = {"": _NUM, "word": r"[A-Za-z0-9_]+", "list": r"[A-Za-z0-9_、]+", "dots": r"[0-9・]+"}


def _compile(tmpl: str):
    parts = re.split(r"\{(\w+)(?::(\w+))?\}", tmpl)
    rx, kinds = "", {}
    for i in range(0, len(parts), 3):
        rx += re.escape(parts[i])
        if i + 1 < len(parts):
            name, kind = parts[i + 1], parts[i + 2] or ""
            rx += f"(?P<{name}>{_KIND[kind]})"
            kinds[name] = kind
    return re.compile(rx), kinds


def _line_of(g, needle) -> int | None:
    for i, ln in enumerate(g.get("__file__", {}).get("lines", []), 1):
        if needle in ln:
            return i
    return None


def _where(g, needle) -> str:
    ln = _line_of(g, needle)
    return f"configs/s4_gates.json:{ln}" if ln else "configs/s4_gates.json"


def read_rules(g) -> tuple[dict, list]:
    """SPECS の文を読む。返り値は ({規則の名前: {"text", 読んだ値…}}, [読めなかった規則])。"""
    rules, problems = {}, []
    for rid, path, metric, tmpl in SPECS:
        try:
            text = _get(g, path)
            parent = _get(g, path[:-1])
        except (KeyError, IndexError, TypeError, ValueError) as e:
            problems.append({"id": rid, "path": _path_str(path), "why": f"欄がない（{e}）"})
            continue
        if metric is not None and (not isinstance(parent, dict) or parent.get("metric") != metric):
            problems.append({"id": rid, "path": _path_str(path), "why": f"metric の欄が {metric!r} でない"})
            continue
        rx, kinds = _compile(tmpl)
        m = rx.fullmatch(text) if isinstance(text, str) else None
        if m is None:
            problems.append({"id": rid, "path": _path_str(path), "why": "文が想定の形と一致しない（推し量らずに当てはめない）",
                             "text": text})
            continue
        val = {"text": text, "path": _path_str(path)}
        for k, kind in kinds.items():
            s = m.group(k)
            val[k] = (Fraction(s) if kind == "" else s if kind == "word" else s.split("、") if kind == "list"
                      else [int(x) for x in s.split("・")])
        rules[rid] = val
    _consistency(g, rules, problems)
    for p in problems:                          # ファイルの行: 文の頭、なければ id の欄
        t = p.get("text")
        needle = t[:16] if isinstance(t, str) and t and _line_of(g, t[:16]) else f'"id": "{p["id"]}"'
        p["where"] = _where(g, needle)
    return rules, problems


def _drop(rules, problems, rid, why):
    if rid in rules:
        problems.append({"id": rid, "path": rules[rid]["path"], "why": why, "text": rules[rid]["text"]})
        del rules[rid]


def _consistency(g, rules, problems):
    """読んだ値どうし・機械の欄との食い違い。食い違えばその規則を読めなかったことにする（どちらが正しいかは決めない）。"""
    R = rules
    try:
        settings = list(g["gates"]["R"]["runs"]["settings"])
    except (KeyError, TypeError):
        settings = []
    if "R.candidates" in R:
        c = R["R.candidates"]
        if len(c["names"]) != c["n"] or not set(c["names"]) <= set(settings) or {"naive", "current_repro"} & set(c["names"]):
            _drop(R, problems, "R.candidates", "候補の数・名前が runs.settings と合わない")
    if "R.rank4" in R and "R.candidates" in R:
        order = []
        for ab in R["R.rank4"]["order"]:
            hit = [n for n in R["R.candidates"]["names"] if n == ab or n.startswith(ab + "_")]
            order.append(hit[0] if len(hit) == 1 else None)
        if None in order or sorted(order) != sorted(R["R.candidates"]["names"]):
            _drop(R, problems, "R.rank4", "宣言順の略称が候補の名前に 1 対 1 で当たらない")
        else:
            R["R.rank4"]["order_names"] = order
    if "R.stop" in R and R["R.stop"]["zero"] != 0:
        _drop(R, problems, "R.stop", "やめる枝の個数が 0 でない（想定外）")
    if "R.b2" in R and "R.action" in R and not (R["R.b2"]["a"] == R["R.action"]["top"] and R["R.b2"]["b"] == R["R.b2"]["c"] == 1):
        _drop(R, problems, "R.b2", "上位の数が R.action と合わない")
    if "R.c3" in R:
        try:
            prim = Fraction(str(g["time_limits"]["scoring"]["primary_s"]))
        except (KeyError, TypeError, ValueError):
            prim = None
        if prim != R["R.c3"]["L"]:
            _drop(R, problems, "R.c3", "採点の時間が time_limits.scoring.primary_s と合わない")
    if "T.action" in R:
        try:
            arms = list(g["gates"]["T"]["continue"]["for_arm_in"])
        except (KeyError, TypeError):
            arms = []
        if arms != ["EH", "ES"] or R["T.action"]["tie"] not in arms:
            _drop(R, problems, "T.action", "for_arm_in が [EH, ES] でない、または同じときの腕が for_arm_in にない")
    try:
        fac = g["gates"]["S"]["runs"]["factorial"]
        poses, priors = list(fac["start_pose"]), list(fac["prior_cube"])
    except (KeyError, TypeError):
        poses, priors = [], []
    if "S.pose.value" in R:
        v = R["S.pose.value"]
        if not ({v["hi"], v["lo"]} <= set(poses) and "on_grid" in priors and "S.pose.fix" in R):
            _drop(R, problems, "S.pose.value", "姿勢の対比の名前が factorial と合わない、または fix を読めない")
    if "S.prior.value" in R:
        v = R["S.prior.value"]
        if not ({v["hi"], v["lo"]} <= set(priors) and "S.prior.fix" in R and R["S.prior.fix"]["start"] in poses):
            _drop(R, problems, "S.prior.value", "先客の対比の名前が factorial と合わない、または fix を読めない")
    try:
        reps = g["transplant_arm"]["repeats"]
        arm_ids = [a["id"] for a in g["transplant_arm"]["arms"]]
    except (KeyError, TypeError):
        reps, arm_ids = None, []
    for rid in ("XPL.pose", "XPL.prior"):
        if rid in R and "XPL.rep" in R:
            if R[rid]["base"] != R["XPL.rep"]["arm"] or R[rid]["other"] not in arm_ids or reps != 2:
                _drop(R, problems, rid, "腕の名前が transplant_arm.arms と合わない、または repeats が文の 2 回と合わない")
    if "XPL.rep" in R and R["XPL.rep"]["arm"] not in arm_ids:
        _drop(R, problems, "XPL.rep", "腕の名前が transplant_arm.arms にない")
    gv = (g.get("gates", {}).get("C") or {}).get("gate_variant")
    for rid in ("C.c1", "C.c2"):
        if rid in R and R[rid]["variant"] != gv:
            _drop(R, problems, rid, "条件の版が gate_variant と合わない")
    if "C.c2" in R and R["C.c2"]["L"] != R["C.c2"]["L2"]:
        _drop(R, problems, "C.c2", "成立の時間と採点の時間が違う")
    if "C.c3" in R and "C.stop" in R and R["C.c3"]["a"] != R["C.stop"]["a"]:
        _drop(R, problems, "C.stop", "やめる枝の対の数が C.c3 と合わない")
    if "K.b1" in R and "K.b2" in R:
        a, b = R["K.b1"], R["K.b2"]
        if (a["L1"], a["L2"], a["a"], a["b"]) != (b["L1"], b["L2"], b["a"], b["b"]):
            _drop(R, problems, "K.b2", "K.b1 と K.b2 の境界が合わない（警告の境界を決められない）")
    if "K.c1" in R:
        try:
            runs = g["gates"]["K"]["runs"]
            ok = Fraction(runs["trials"]) == R["K.c1"]["n"] and runs["repeat"] == 2
        except (KeyError, TypeError):
            ok = False
        if not ok:
            _drop(R, problems, "K.c1", "試行数・回数が gates.K.runs と合わない")
    try:
        rows = g["candidate_selection"]["rows"]
        okr = [r.get("order") for r in rows] == [1, 2, 3, 4] and \
            [r.get("verdict") for r in rows] == ["internal_state", "prior_cube", "pose", "none"]
    except (KeyError, TypeError):
        okr = False
    if not okr:
        for rid in ("cand.1", "cand.2", "cand.3", "cand.4"):
            _drop(R, problems, rid, "rows の順・verdict が想定と違う")


# ------------------------------------------------------------------ 三値の論理と値の取り出し
class Undetermined(Exception):
    """値がない・形が違う・分母が 0・規則を読めない。"""


def _all(vals):
    vals = list(vals)
    if any(v is False for v in vals):
        return False
    if any(v is None for v in vals):
        return None
    return True


def _any(vals):
    vals = list(vals)
    if any(v is True for v in vals):
        return True
    if any(v is None for v in vals):
        return None
    return False


def _not(v):
    return None if v is None else not v


def _res(v) -> str:
    return UNDET if v is None else PASS if v else FAIL


def _need(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            raise Undetermined(f"値がない: {'.'.join(str(x) for x in keys)}")
        cur = cur[k]
    return cur


def _real(x, where) -> Fraction:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise Undetermined(f"形が違う（数でない）: {where}")
    if isinstance(x, float) and not math.isfinite(x):
        raise Undetermined(f"値が有限でない: {where}")
    return Fraction(x) if isinstance(x, int) else Fraction(repr(x))


def _count(x, where) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise Undetermined(f"形が違う（0 以上の整数でない）: {where}")
    return x


def _rate(d, where) -> tuple[Fraction, str, int]:
    """{"k", "n"} の割合（rate があれば k/n と合うかを確かめる）。返り値 (k/n, "k/n", n)。"""
    if not isinstance(d, dict) or "k" not in d or "n" not in d:
        raise Undetermined(f"値がない: {where}（k と n が要る）")
    k, n = _count(d["k"], where + ".k"), _count(d["n"], where + ".n")
    if k > n:
        raise Undetermined(f"形が違う（k > n）: {where}")
    if n == 0:
        raise Undetermined(f"分母が 0: {where}")
    r = d.get("rate")
    if r is not None and (isinstance(r, bool) or not isinstance(r, (int, float)) or abs(float(r) - k / n) > RATE_TOL):
        raise Undetermined(f"形が違う（rate が k/n と合わない）: {where}")
    return Fraction(k, n), f"{k}/{n}", n


def _f(x):
    """結果に書く数（分数は小数 6 桁）。"""
    if isinstance(x, Fraction):
        return int(x) if x.denominator == 1 else round(float(x), 6)
    return x


def _cond(cid, rules, rule_ids, fn, **show):
    """条件 1 つを当てはめる。fn(rules) -> (真偽, 値, 閾値, 補足の辞書)。読めない規則・値の問題は判定できない。"""
    rec = {"id": cid, "rule": " / ".join(rules[r]["text"] for r in rule_ids if r in rules), **show}
    missing = [r for r in rule_ids if r not in rules]
    if missing:
        rec.update(result=UNDET, reason=f"規則を読めない: {', '.join(missing)}")
        return None, rec
    try:
        truth, value, thr, extra = fn(rules)
    except Undetermined as e:
        rec.update(result=UNDET, reason=str(e))
        return None, rec
    rec.update(result=_res(truth), value=_f(value), threshold=_f(thr), **(extra or {}))
    return truth, rec


def _rule_missing(rules, *ids):
    return [r for r in ids if r not in rules]


# ------------------------------------------------------------------ 関門 R
def eval_R(rules, inp) -> dict:
    R = inp.get("R") if isinstance(inp.get("R"), dict) else {}
    S = R.get("settings") if isinstance(R.get("settings"), dict) else {}
    out = {"result": UNDET, "settings": {}, "passed": [], "top": None, "branch": UNDET, "branches": {}}
    if "R.candidates" not in rules:
        out["reason"] = "候補の設定を読めない（R.candidates）"
        return out
    names = rules["R.candidates"]["names"]
    naive = S.get("naive")
    truths = {}
    for name in names:
        s = S.get(name)
        conds = []

        def c1(r, s=s):
            v, b = _real(_need(s, "radial_gap_mm"), f"R.{name}.radial_gap_mm"), _real(_need(naive, "radial_gap_mm"), "R.naive.radial_gap_mm")
            thr = b - r["R.c1"]["a"]
            return v >= thr, v, thr, {"op": ">=", "naive": _f(b)}

        def c2(r, s=s):
            v = _real(_need(s, "move_ratio"), f"R.{name}.move_ratio")
            return v >= r["R.c2"]["a"], v, r["R.c2"]["a"], {"op": ">="}

        def c3(r, s=s):
            v, b = _count(_need(s, "natural_success_30"), f"R.{name}.natural_success_30"), \
                _count(_need(naive, "natural_success_30"), "R.naive.natural_success_30")
            n1, n0 = _count(_need(s, "n_trials"), f"R.{name}.n_trials"), _count(_need(naive, "n_trials"), "R.naive.n_trials")
            if n1 != n0:
                raise Undetermined(f"試行数が naive と違う（{n1} と {n0}）")
            thr = b - r["R.c3"]["a"]
            return v >= thr, v, thr, {"op": ">=", "naive": b, "n_trials": n1}

        def c4(r, s=s):
            v = _real(_need(s, "seam_jump_mps"), f"R.{name}.seam_jump_mps")
            return v <= r["R.c4"]["a"], v, r["R.c4"]["a"], {"op": "<="}

        ts = []
        for cid, fn in (("R.c1", c1), ("R.c2", c2), ("R.c3", c3), ("R.c4", c4)):
            t, rec = _cond(cid, rules, [cid], fn, metric=(rules.get(cid) or {}).get("metric") or cid)
            ts.append(t)
            conds.append(rec)
        truths[name] = _all(ts)
        info = {"result": _res(truths[name]), "conditions": conds}
        if isinstance(s, dict) and "first_close_after_30s" in s:
            info["first_close_after_30s"] = s["first_close_after_30s"]
        out["settings"][name] = info
    out["passed"] = [n for n in names if truths[n] is True]
    undet = [n for n in names if truths[n] is None]
    n_pass = len(out["passed"])
    out["result"] = PASS if n_pass else (UNDET if undet else FAIL)
    # 順位（上位 top）と枝
    miss = _rule_missing(rules, "R.action", "R.rank1", "R.rank2", "R.rank3", "R.rank4", "R.stop", "R.b2")
    if n_pass:
        out["branch"] = "continue"              # やめる枝（合格が 0 個）には当たらないことは決まる
    if miss:
        out["reason"] = f"規則を読めない: {', '.join(miss)}"
        if not n_pass:
            out["branch"] = UNDET
    elif undet:
        out["reason"] = f"判定できない設定がある（{', '.join(undet)}）。上位に入るかが決まらないので止める"
    else:
        order = rules["R.rank4"]["order_names"]

        def key(n):
            s = S[n]
            return (-_count(s["natural_success_30"], n), _real(s["seam_jump_mps"], n), abs(_real(s["radial_gap_mm"], n)), order.index(n))

        ranked = sorted(out["passed"], key=key)
        top = int(rules["R.action"]["top"])
        out["ranking"] = [{"setting": n, "natural_success_30": S[n]["natural_success_30"], "seam_jump_mps": S[n]["seam_jump_mps"],
                           "abs_radial_gap_mm": abs(S[n]["radial_gap_mm"]), "declared_order": order.index(n) + 1} for n in ranked]
        out["top"] = ranked[:top]
        if n_pass == 0:
            out["branch"] = "stop"
        else:
            out["branch"] = "continue"
            if n_pass == 1:
                out["branches"]["R.b2"] = {"applies": True, "text": rules["R.b2"]["text"]}
    out["branches"]["R.b1"] = _eval_R_b1(rules, R)
    return out


def _eval_R_b1(rules, R) -> dict:
    if "R.b1" not in rules:
        return {"result": UNDET, "reason": "規則を読めない: R.b1"}
    rb = rules["R.b1"]
    rows = [str(h) for h in rb["rows"]]

    def side(key, thr):
        d = R.get(key)
        if not isinstance(d, dict):
            raise Undetermined(f"値がない: R.{key}")
        vals = {}
        for h in rows:
            vals[h] = _real(_need(d, h), f"R.{key}.{h}")
        return any(v >= thr for v in vals.values()), {h: _f(v) for h, v in vals.items()}

    out = {"text": rb["text"], "threshold_shadow_mm": _f(rb["a"]), "threshold_x2_mm": _f(rb["b"]), "rows": rb["rows"]}
    sides = []
    for key, thr in (("shadow_plan_shorter_mm", rb["a"]), ("x2_shortfall_mm", rb["b"])):
        try:
            t, v = side(key, thr)
            out[key] = {"values": v, "result": _res(t)}
            sides.append(t)
        except Undetermined as e:
            out[key] = {"result": UNDET, "reason": str(e)}
            sides.append(None)
    t = _all(sides)
    out["result"] = _res(t)
    out["applies"] = t
    return out


# ------------------------------------------------------------------ 関門 T
def eval_T(rules, inp) -> dict:
    T = inp.get("T") if isinstance(inp.get("T"), dict) else {}
    A = T.get("arms") if isinstance(T.get("arms"), dict) else {}
    out = {"result": UNDET, "arms": {}, "reference": {}, "return_to": None, "branches": {}}

    def lift(arm):
        return _rate(_need(A, arm, "first_close_lift"), f"T.arms.{arm}.first_close_lift")

    ref = {}
    try:
        if _rule_missing(rules, "T.E0_ref_lift", "T.d_E0", "T.required_margin"):
            raise Undetermined("規則を読めない: T.reference_rule")
        (l1, s1, _), (l2, s2, _) = lift("E0_run1"), lift("E0_run2")
        ref["E0_ref_lift"] = max(l1, l2)
        ref["d_E0"] = abs(l1 - l2)
        m = rules["T.required_margin"]
        ref["required_margin"] = max(m["a"], ref["d_E0"] + m["b"])
        out["reference"] = {"E0_run1": s1, "E0_run2": s2, **{k: _f(v) for k, v in ref.items()}}
    except Undetermined as e:
        out["reference"] = {"result": UNDET, "reason": str(e)}
    truths, lifts = {}, {}
    for arm in ("EH", "ES"):
        def c1(r, arm=arm):
            v, s, _ = lift(arm)
            return v >= r["T.c1"]["a"], v, r["T.c1"]["a"], {"op": ">=", "k_n": s}

        def c2(r, arm=arm):
            v, s, _ = lift(arm)
            if "E0_ref_lift" not in ref:
                raise Undetermined(f"E0 の基準を出せない（{out['reference'].get('reason')}）")
            thr = ref["E0_ref_lift"] + ref["required_margin"]
            return v >= thr, v, thr, {"op": ">=", "k_n": s}

        def c3(r, arm=arm):
            k = _count(_need(A, arm, "all_three_true", "k"), f"T.arms.{arm}.all_three_true.k")
            n = _count(_need(A, arm, "all_three_true", "n"), f"T.arms.{arm}.all_three_true.n")
            ks, ns = [], []
            for e in ("E0_run1", "E0_run2"):
                ks.append(_count(_need(A, e, "all_three_true", "k"), f"T.arms.{e}.all_three_true.k"))
                ns.append(_count(_need(A, e, "all_three_true", "n"), f"T.arms.{e}.all_three_true.n"))
            if len({n, *ns}) != 1:
                raise Undetermined(f"本数の分母が腕で違う（{arm} {n}、E0 {ns[0]}・{ns[1]}）")
            if k > n:
                raise Undetermined(f"形が違う（k > n）: T.arms.{arm}.all_three_true")
            thr = max(ks)
            return k >= thr, k, thr, {"op": ">=", "n": n}

        ts, conds = [], []
        for cid, fn in (("T.c1", c1), ("T.c2", c2), ("T.c3", c3)):
            t, rec = _cond(cid, rules, [cid], fn, metric=(rules.get(cid) or {}).get("metric") or cid)
            ts.append(t)
            conds.append(rec)
        truths[arm] = _all(ts)
        try:
            lifts[arm] = lift(arm)[0]
        except Undetermined:
            lifts[arm] = None
        out["arms"][arm] = {"result": _res(truths[arm]), "conditions": conds}
    passed = [a for a in ("EH", "ES") if truths[a] is True]
    if passed:
        out["result"] = PASS
        if "T.action" not in rules:
            out["return_to_reason"] = "規則を読めない: T.action"
        elif any(truths[a] is None for a in ("EH", "ES")):
            out["return_to_reason"] = "もう一方の腕が判定できないので、両方合格のときの選び方を当てはめられない"
        elif len(passed) == 2:
            out["return_to"] = rules["T.action"]["tie"] if lifts["EH"] == lifts["ES"] else max(passed, key=lambda a: lifts[a])
        else:
            out["return_to"] = passed[0]
    elif all(truths[a] is False for a in ("EH", "ES")):
        out["result"] = FAIL if "T.stop" in rules else UNDET
    out["branch"] = {PASS: "continue", FAIL: "stop", UNDET: UNDET}[out["result"]]
    # T.b1（EO で緑を 1 番目にしたとき）
    t, rec = _cond("T.b1", rules, ["T.b1"], lambda r: _t_b1(r, T))
    out["branches"]["T.b1"] = {**rec, "applies": t}
    # T.b2（E0 の 2 回の差）
    if "T.b2" not in rules:
        out["branches"]["T.b2"] = {"result": UNDET, "reason": "規則を読めない: T.b2", "applies": None}
    elif "d_E0" not in ref:
        out["branches"]["T.b2"] = {"result": UNDET, "reason": out["reference"].get("reason"), "applies": None}
    else:
        t = ref["d_E0"] > rules["T.b2"]["a"]
        out["branches"]["T.b2"] = {"rule": rules["T.b2"]["text"], "result": _res(t), "value": _f(ref["d_E0"]),
                                   "threshold": _f(rules["T.b2"]["a"]), "op": ">", "applies": t}
    if isinstance(T.get("K_e7"), dict):
        out["K_e7_report_only"] = T["K_e7"]
    return out


def _t_b1(r, T):
    eo = T.get("EO_green_first")
    v, s, _ = _rate(_need(eo, "first_close_lift"), "T.EO_green_first.first_close_lift")
    colors = _need(eo, "colors")
    if colors != ["green"]:
        raise Undetermined(f"1 番目の手順の色が緑だけでない: {colors}")
    return v <= r["T.b1"]["a"], v, r["T.b1"]["a"], {"op": "<=", "k_n": s}


# ------------------------------------------------------------------ 関門 S と移植の腕
def _fisher_greater(k_hi, n_hi, k_lo, n_lo) -> float:
    from scipy.stats import fisher_exact
    return float(fisher_exact([[k_hi, n_hi - k_hi], [k_lo, n_lo - k_lo]], alternative="greater")[1])


def eval_S(rules, inp) -> dict:
    S = inp.get("S") if isinstance(inp.get("S"), dict) else {}
    C = S.get("conditions") if isinstance(S.get("conditions"), dict) else {}
    out = {"contrasts": {}}
    specs = []
    if "S.pose.value" in rules:
        v = rules["S.pose.value"]
        specs.append(("S.pose", f"{v['hi']}|on_grid", f"{v['lo']}|on_grid"))
    else:
        specs.append(("S.pose", None, None))
    if "S.prior.value" in rules:
        v, st = rules["S.prior.value"], rules["S.prior.fix"]["start"]
        specs.append(("S.prior", f"{st}|{v['hi']}", f"{st}|{v['lo']}"))
    else:
        specs.append(("S.prior", None, None))
    for cid, hi, lo in specs:
        rec = {"a": hi, "b": lo}
        if hi is None:
            out["contrasts"][cid] = {**rec, "result": UNDET, "effect": None, "reason": f"規則を読めない: {cid}.value",
                                     "conditions": []}
            continue

        def rates(hi=hi, lo=lo):
            return (_rate(_need(C, hi, "plus_y_shift"), f"S.conditions.{hi}.plus_y_shift"),
                    _rate(_need(C, lo, "plus_y_shift"), f"S.conditions.{lo}.plus_y_shift"))

        def high(r):
            (a, sa, _), (b, sb, _) = rates()
            return max(a, b) >= r["S.high"]["a"], max(a, b), r["S.high"]["a"], {"op": ">=", "a": sa, "b": sb}

        def diff(r):
            (a, sa, _), (b, sb, _) = rates()
            return a - b >= r["S.diff"]["a"], a - b, r["S.diff"]["a"], {"op": ">=", "a": sa, "b": sb}

        def pval(r):
            (a, sa, na), (b, sb, nb) = rates()
            ka, kb = int(a * na), int(b * nb)
            p = _fisher_greater(ka, na, kb, nb)
            extra = {"op": "<", "a": sa, "b": sb, "test": "片側（a > b）のフィッシャーの正確検定"}
            if na != r["S.p"]["n"] or nb != r["S.p"]["n"]:
                extra["note"] = f"分母が文の {int(r['S.p']['n'])} でない（{na}・{nb}。分母は最初の閉じが起きた試行）"
            return Fraction(repr(p)) < r["S.p"]["a"], p, r["S.p"]["a"], extra

        ts, conds = [], []
        for sub, fn, ids in (("high", high, ["S.high"]), ("diff", diff, ["S.diff"]), ("p", pval, ["S.p"])):
            t, c = _cond(f"{cid}.{sub}", rules, ids, fn)
            ts.append(t)
            conds.append(c)
        eff = _all(ts)
        info = {**rec, "effect": eff, "result": _res(eff), "conditions": conds}
        late = {k: C[k]["first_close_after_30s"] for k in (hi, lo) if isinstance(C.get(k), dict) and "first_close_after_30s" in C[k]}
        if late:
            info["first_close_after_30s"] = late
        out["contrasts"][cid] = info
    return out


def eval_XPL(rules, inp, g) -> dict:
    X = inp.get("XPL") if isinstance(inp.get("XPL"), dict) else {}
    A = X.get("arms") if isinstance(X.get("arms"), dict) else {}
    reps = (g.get("transplant_arm") or {}).get("repeats")
    out = {}

    def pooled(arm):
        if not isinstance(reps, int) or reps < 1:
            raise Undetermined("transplant_arm.repeats を読めない")
        ks, ns, each = 0, 0, []
        for i in range(1, reps + 1):
            v, s, n = _rate(_need(A, arm, f"rep{i}"), f"XPL.arms.{arm}.rep{i}")
            ks += int(v * n)
            ns += n
            each.append((v, s))
        return Fraction(ks, ns), f"{ks}/{ns}", each

    def rep(r):
        v, s, _ = pooled(r["XPL.rep"]["arm"])
        return v >= r["XPL.rep"]["a"], v, r["XPL.rep"]["a"], {"op": ">=", "k_n": s, "arm": r["XPL.rep"]["arm"]}

    t_rep, out["XPL.rep"] = _cond("XPL.rep", rules, ["XPL.rep"], rep)
    out["XPL.rep"]["satisfied"] = t_rep
    for cid in ("XPL.pose", "XPL.prior"):
        def drop_all(r, cid=cid):
            base, other = r[cid]["base"], r[cid]["other"]
            vb, sb, _ = pooled(base)
            vo, so, _ = pooled(other)
            return vb - vo >= r[cid]["a"], vb - vo, r[cid]["a"], {"op": ">=", "base": sb, "other": so, "what": f"{base} - {other}（両方の回）"}

        def drop_each(r, i, cid=cid):
            base, other = r[cid]["base"], r[cid]["other"]
            vb, sb = pooled(base)[2][i]
            vo, so = pooled(other)[2][i]
            return vb - vo >= r[cid]["b"], vb - vo, r[cid]["b"], {"op": ">=", "base": sb, "other": so, "what": f"{base} - {other}（{i + 1} 回目）"}

        parts = []
        t1, c1 = _cond(f"{cid}.both", rules, [cid], drop_all)
        parts.append((t1, c1))
        n_rep = reps if isinstance(reps, int) and reps >= 1 else 2
        for i in range(n_rep):
            parts.append(_cond(f"{cid}.rep{i + 1}", rules, [cid], lambda r, i=i: drop_each(r, i)))
        t = _all([t_rep] + [p[0] for p in parts])
        out[cid] = {"result": _res(t), "satisfied": t, "requires": "XPL.rep", "conditions": [p[1] for p in parts]}
    return out


# ------------------------------------------------------------------ 関門 C・K
def eval_C(rules, inp, g) -> dict:
    Cin = inp.get("C") if isinstance(inp.get("C"), dict) else {}
    out = {"variant": (g.get("gates", {}).get("C") or {}).get("gate_variant"), "conditions": []}

    def at(variant, L):
        return _need(Cin, variant, "by_L", f"{L:g}" if not isinstance(L, Fraction) else _lkey(L))

    def c1(r):
        d = at(r["C.c1"]["variant"], r["C.c1"]["L"])
        v, s, _ = _rate(_need(d, "recovery_R"), f"C.{r['C.c1']['variant']}.recovery_R")
        return v >= r["C.c1"]["a"], v, r["C.c1"]["a"], {"op": ">=", "k_n": s}

    def pairs(r, rid):
        d = at(r[rid]["variant"] if "variant" in r[rid] else out["variant"], r[rid]["L"])
        p = _need(d, "paired")
        n = _count(_need(p, "pairs"), "C.paired.pairs")
        ro, no = _count(_need(p, "r_only"), "C.paired.r_only"), _count(_need(p, "n_only"), "C.paired.n_only")
        if ro + no > n:
            raise Undetermined("形が違う（r_only + n_only > pairs）")
        return n, ro, no

    def c2(r):
        n, ro, no = pairs(r, "C.c2")
        if n == 0:
            raise Undetermined("分母が 0（両方で起きた組がない）")
        v = Fraction(ro - no, n)
        return v >= r["C.c2"]["a"], v, r["C.c2"]["a"], {"op": ">=", "r_only": ro, "n_only": no, "pairs": n,
                                                         "k_n": f"({ro} − {no})/{n}"}

    def c3(r):
        n, _, _ = pairs(r, "C.c3")
        return n >= r["C.c3"]["a"], n, r["C.c3"]["a"], {"op": ">="}

    ts = []
    for cid, fn in (("C.c1", c1), ("C.c2", c2), ("C.c3", c3)):
        t, rec = _cond(cid, rules, [cid], fn)
        ts.append(t)
        out["conditions"].append(rec)
    t = _all(ts)
    if t is False and "C.stop" not in rules:
        t = None
    out["result"] = _res(t)
    out["passed"] = t
    out["branch"] = {True: "continue", False: "stop", None: UNDET}[t]
    # 副次（60 s）と C.b2 の材料は並べるだけ（判定に使わない）
    sec = {}
    for variant, d in Cin.items():
        if isinstance(d, dict) and isinstance(d.get("by_L"), dict):
            sec[variant] = {L: _brief_c(x) for L, x in d["by_L"].items()}
    out["by_variant_report_only"] = sec
    return out


def _lkey(L: Fraction) -> str:
    return str(int(L)) if L.denominator == 1 else f"{float(L):g}"


def _brief_c(x) -> dict:
    o = {}
    try:
        o["recovery_R"] = _rate(x.get("recovery_R"), "recovery_R")[1]
    except (Undetermined, AttributeError):
        pass
    try:
        o["recovery_N"] = _rate(x.get("recovery_N"), "recovery_N")[1]
    except (Undetermined, AttributeError):
        pass
    p = x.get("paired") if isinstance(x, dict) else None
    if isinstance(p, dict) and p.get("pairs"):
        o["R_minus_N"] = round((p.get("r_only", 0) - p.get("n_only", 0)) / p["pairs"], 4)
        o["pairs"] = p["pairs"]
    return o


def eval_K(rules, inp) -> dict:
    K = inp.get("K") if isinstance(inp.get("K"), dict) else {}
    out = {}

    def c1(r):
        n = int(r["K.c1"]["n"])
        a = _count(_need(K, "run1_completed"), "K.run1_completed")
        b = _count(_need(K, "run2_completed"), "K.run2_completed")
        m = _need(K, "pairs_matched")
        if not isinstance(m, bool):
            raise Undetermined("形が違う（pairs_matched が真偽でない）")
        return a == n and b == n and m, f"{a}・{b}・対応 {'一致' if m else '崩れた'}", n, {"op": "=="}

    t1, out["K.c1"] = _cond("K.c1", rules, ["K.c1"], c1)
    out["K.c1"]["satisfied"] = t1
    out["d0_listed"] = t1
    if t1 is False:
        out["branch"] = "stop"
        out["warning"] = None
        return out
    if t1 is None:
        out["branch"] = UNDET
        out["warning"] = None
        return out
    out["branch"] = "continue"
    d0 = {}
    miss = _rule_missing(rules, "K.b1", "K.b2")
    if miss:
        out["warning"] = None
        out["reason"] = f"規則を読めない: {', '.join(miss)}"
        return out
    b = rules["K.b2"]
    over = []
    for L, thr in ((b["L1"], b["a"]), (b["L2"], b["b"])):
        key = _lkey(L)
        try:
            v, s, _ = _rate(_need(K, "d0", key), f"K.d0.{key}")
            d0[key] = {"k_n": s, "value": _f(v), "threshold": _f(thr), "over": v > thr}
            over.append(v > thr)
        except Undetermined as e:
            d0[key] = {"result": UNDET, "reason": str(e)}
            over.append(None)
    out["d0"] = d0
    w = _any(over)
    out["warning"] = w
    out["branch_id"] = {True: "K.b2", False: "K.b1", None: UNDET}[w]
    return out


# ------------------------------------------------------------------ 束 4 の候補の選び方と最後の結論
def eval_candidates(rules, S, X, T) -> dict:
    miss = _rule_missing(rules, "cand.1", "cand.1.action", "cand.2", "cand.3", "cand.3.action", "cand.4")
    out = {"trace": []}
    if miss:
        out.update(row=None, verdict=None, result=UNDET, reason=f"規則を読めない: {', '.join(miss)}", start_candidate=None, ex_arm=None)
        return out
    rep = X["XPL.rep"].get("satisfied")
    rows = [(1, "internal_state", _not(rep)),
            (2, "prior_cube", _all([S["contrasts"]["S.prior"]["effect"], X["XPL.prior"]["satisfied"]])),
            (3, "pose", _all([S["contrasts"]["S.pose"]["effect"], X["XPL.pose"]["satisfied"]])),
            (4, "none", True)]
    for order, verdict, t in rows:
        out["trace"].append({"row": order, "when": rules[f"cand.{order}"]["text"], "result": _res(t)})
        if t is None:
            out.update(row=None, verdict=None, result=UNDET, start_candidate=None, ex_arm=None,
                       reason=f"行 {order} の条件が判定できない（判定できないときの枝の規則がないので止める）")
            return out
        if t:
            out.update(row=order, verdict=verdict, result="determined")
            break
    v = out["verdict"]
    out["start_candidate"] = v == "prior_cube"
    if v == "internal_state":
        out["ex_arm"] = True
    elif v == "pose":
        tr = T["result"]
        out["t_result"] = tr
        out["ex_arm"] = {PASS: False, FAIL: True, UNDET: None}[tr]
        out["executor_suffices"] = {PASS: True, FAIL: False, UNDET: None}[tr]
        if tr == UNDET:
            out["reason"] = "行 3 に当たったが関門 T が判定できないので、EX を足すかが決まらない（束 4 に開始状態が入らないことは決まる）"
    else:
        out["ex_arm"] = False
    return out


def conclude(R, T, S, X, C, K, cand) -> dict:
    und = []
    # 束 2: RTC の設定
    if R["branch"] == "continue" and R["top"] is None:
        rtc = {"result": UNDET, "settings": None, "b2": True, "reason": R.get("reason")}
        und.append("束 2 の RTC の設定（関門 R の上位）")
    elif R["branch"] == "continue":
        rtc = {"result": "determined", "settings": R["top"], "b2": True}
    elif R["branch"] == "stop":
        rtc = {"result": "determined", "settings": [], "b2": False, "action": "RTC の設定の調整をやめ、束 5 の F0 だけを行う"}
    else:
        rtc = {"result": UNDET, "settings": None, "b2": None, "reason": R.get("reason")}
        und.append("束 2 の RTC の設定（関門 R）")
    # 束 2: v3 (a) の戻し先
    if T["result"] == PASS and T["return_to"]:
        v3a = {"result": "determined", "include": True, "return_to": T["return_to"]}
    elif T["result"] == PASS:
        v3a = {"result": UNDET, "include": True, "return_to": None, "reason": T.get("return_to_reason")}
        und.append("束 2 の v3 (a) の戻し先（関門 T）")
    elif T["result"] == FAIL:
        v3a = {"result": "determined", "include": False, "return_to": None}
    else:
        v3a = {"result": UNDET, "include": None, "return_to": None}
        und.append("束 2 の v3 (a) を入れるか（関門 T）")
    # 束 4
    start, slip = cand.get("start_candidate"), C.get("passed")
    research = []
    if start is True:
        b4 = {"result": "determined", "candidate": "start_state_prior_cube"}
        if slip is True:
            research.append("滑りのデータ（開始状態を優先するため。slip_rule の 2 行目・C.b1）")
        elif slip is None:
            b4["note"] = "関門 C は判定できないが、開始状態の候補が残るので束 4 は開始状態に決まる（滑りは束 4 に入らない）"
    elif start is False:
        if slip is True:
            b4 = {"result": "determined", "candidate": "slip"}
        elif slip is False:
            b4 = {"result": "determined", "candidate": "none"}
        else:
            b4 = {"result": UNDET, "candidate": None, "reason": "開始状態の候補はないが、関門 C が判定できない"}
            und.append("束 4 の候補（関門 C）")
    else:
        b4 = {"result": UNDET, "candidate": None, "reason": cand.get("reason")}
        und.append("束 4 の候補（候補の選び方）")
    if b4.get("candidate") == "none":
        research.append("束 4（候補がない。テスト 1 は H2 と、関門 R・T で通ったものだけ）")
    v = cand.get("verdict")
    if v in ("internal_state", "none"):
        research.append("開始状態のデータ")
    if v == "pose" and cand.get("executor_suffices"):
        research.append("姿勢のデータ（実行器で足りるため）")
    if C.get("passed") is False:
        research.append("落下と置き損ねの改善（関門 C のやめる枝）")
    ex = cand.get("ex_arm")
    if ex is None:
        und.append("EX の腕を足すか（候補の選び方の行 1・3）")
    if R["branches"].get("R.b1", {}).get("applies") is True:
        research.append("学習時の RTC（R.b1、研究の道の R-3）")
    if T["branches"].get("T.b1", {}).get("applies") is True:
        research.append("弱い層のデータ（T.b1。単独では学習の理由にしない）")
    warnings = []
    if K.get("warning") is True:
        warnings.append("K.b2: 回し直しの食い違い率が前例を超える。成功率の小さな差は仕組みの指標で述べる")
    if K.get("d0_listed") is False:
        warnings.append("K.c1 を満たさない: その回の d0 は載せず、前例の範囲を載せる。予備で 1 回だけ回し直す")
    if K.get("branch") == UNDET or (K.get("d0_listed") and K.get("warning") is None):
        und.append("関門 K の警告")
    if T["branches"].get("T.b2", {}).get("applies") is True:
        warnings.append("T.b2: 今の実行器の 2 回の差が大きい。D2 の前に 3 個の連続タスクの本数を計算し直す")
    return {"status": "undetermined" if und else "determined", "undetermined": und,
            "bundle2": {"rtc": rtc, "v3a": v3a}, "bundle4": b4, "ex_arm": ex, "research_path": research, "warnings": warnings}


def evaluate(g, inp) -> dict:
    """関門 1 を当てはめる。g は load_gates の結果（または同じ形の辞書）、inp は入力の辞書。"""
    if not isinstance(inp, dict):
        raise ValueError("入力が JSON の辞書でない")
    meta = g.get("__file__", {})
    rules, problems = read_rules(g)
    warn = []
    if g.get("schema") != GATES_SCHEMA:
        warn.append(f"s4_gates.json の schema が {GATES_SCHEMA} でない: {g.get('schema')!r}")
    if inp.get("schema") != INPUT_SCHEMA:
        warn.append(f"入力の schema が {INPUT_SCHEMA} でない: {inp.get('schema')!r}")
    if meta.get("sha256_lf") and meta["sha256_lf"] != POSTED_SHA256:
        warn.append(f"s4_gates.json の SHA-256 が掲示の値と違う（掲示 {POSTED_BOARD}）")
    R, T, S = eval_R(rules, inp), eval_T(rules, inp), eval_S(rules, inp)
    X, C, K = eval_XPL(rules, inp, g), eval_C(rules, inp, g), eval_K(rules, inp)
    cand = eval_candidates(rules, S, X, T)
    res = {"schema": RESULT_SCHEMA,
           "gates_file": {"path": meta.get("path"), "sha256_bytes": meta.get("sha256_bytes"), "sha256_lf": meta.get("sha256_lf"),
                          "posted_sha256": POSTED_SHA256, "matches_posted": meta.get("sha256_lf") == POSTED_SHA256,
                          "schema": g.get("schema"), "version": g.get("version"), "status": g.get("status")},
           "input_schema": inp.get("schema"), "warnings": warn,
           "gates": {"R": R, "T": T, "S": S, "XPL": X, "C": C, "K": K},
           "candidate_selection": cand,
           "conclusion": conclude(R, T, S, X, C, K, cand),
           "rules_read": {k: {kk: (_f(vv) if isinstance(vv, Fraction) else vv) for kk, vv in v.items()} for k, v in rules.items()},
           "rules_unreadable": problems,
           "not_machine_readable": [{"where": _where(g, x["needle"]), "item": x["where"], "why": x["why"]} for x in NOT_MACHINE_READABLE],
           "read_with_outside_definitions": [{"where": _where(g, x["needle"]), "id": x["id"], "note": x["note"]} for x in OUTSIDE_DEFINITIONS]}
    return res


# ------------------------------------------------------------------ 手元の診断の出力から入力を組む（値を写すだけ。判定はしない）
def _kn(d) -> dict:
    return {"k": d["k"], "n": d["n"]}


def input_from_rtc(metrics: dict, shadow: dict = None, x2: dict = None, horizon: str = "30") -> dict:
    """98_s4_d_rtc.py metrics の出力（conditions[設定名].by_horizon["30"].summary）から関門 R の入力。
    shadow は shadow-metrics の出力（条件は 1 つ）、x2 は 98_s4_x2.py の summary.json。"""
    out = {"settings": {}}
    for name, c in metrics["conditions"].items():
        s = c["by_horizon"][horizon]["summary"]
        out["settings"][name] = {"n_trials": s["n_trials"], "radial_gap_mm": s["radial_gap_mm_median"][0],
                                 "move_ratio": s["move_ratio_median"][0], "seam_jump_mps": s["seam_jump_mps_median"][0],
                                 "natural_success_30": s["successes_at"]["30"],
                                 "first_close_after_30s": s["n_first_close_after_horizon"], "n_no_close": s["n_no_close"]}
    if shadow is not None:
        conds = list(shadow["conditions"].values())
        if len(conds) != 1:
            raise ValueError(f"影の推論の条件が 1 つでない（{len(conds)}）")
        out["shadow_plan_shorter_mm"] = {h: v[0] for h, v in conds[0]["summary"]["shadow_minus_guided_mm_median"].items()}
    if x2 is not None:
        out["x2_shortfall_mm"] = {h: v[0] for h, v in x2["summary"]["expert_minus_pred_mm_median"].items()}
    return out


def input_from_e7(summary: dict) -> dict:
    """98_s4_d_e7.py summary の出力（gate_T_inputs）から関門 T の入力。E0 の基準・d_E0 は使わず、こちらで計算する。"""
    g = summary["gate_T_inputs"]
    out = {"arms": {k: {"first_close_lift": _kn(v["first_close_lift"]), "all_three_true": _kn(v["all_three_true"])}
                    for k, v in g["arms"].items() if k in ("E0_run1", "E0_run2", "EH", "ES")}}
    if "EO_green_first" in g:
        out["EO_green_first"] = {"first_close_lift": _kn(g["EO_green_first"]["first_close_lift"]), "colors": g["EO_green_first"]["colors"]}
    if "K_e7" in g:
        out["K_e7"] = g["K_e7"]
    return out


def input_from_start(summary: dict) -> tuple[dict, dict]:
    """98_s4_d_start.py summary の出力（gate_S_inputs・xpl_inputs）から関門 S と移植の腕の入力。対比・p 値は使わない。"""
    S = {"conditions": {}}
    for k, v in (summary.get("gate_S_inputs") or {}).get("conditions", {}).items():
        S["conditions"][k] = {"plus_y_shift": _kn(v["plus_y_shift"]), "first_close_after_30s": v.get("first_close_after_30s")}
    X = {"arms": {}}
    for arm, v in (summary.get("xpl_inputs") or {}).get("arms", {}).items():
        X["arms"][arm] = {k: _kn(x) for k, x in v.items() if re.fullmatch(r"rep\d+", k)}
    return S, X


def input_from_recovery(score: dict) -> dict:
    """98_s4_d_recovery.py score の出力（variants[版].material.by_L）から関門 C の入力。"""
    out = {}
    for v, ent in score["variants"].items():
        out[v] = {"by_L": {}}
        for L, x in ent["material"]["by_L"].items():
            rr, rn, p = x["recovery_R"], x["recovery_N"], x["paired"]
            out[v]["by_L"][L] = {"recovery_R": {"k": rr["recovered"], "n": rr["established"]},
                                 "recovery_N": {"k": rn["recovered"], "n": rn["established"]},
                                 "paired": {"pairs": p["pairs"], "r_only": p["r_only"], "n_only": p["n_only"]}}
    return out


def build_input(rtc_metrics=None, rtc_shadow=None, x2=None, e7_summary=None, start_summary=None, recovery_score=None, k=None,
                source: dict = None) -> dict:
    """手元の診断の出力（読んだ辞書）から入力を組む。k は関門 K の入力そのもの（形は入力の定義の K）。"""
    inp = {"schema": INPUT_SCHEMA, "source": source or {}}
    if rtc_metrics is not None:
        inp["R"] = input_from_rtc(rtc_metrics, rtc_shadow, x2)
    if e7_summary is not None:
        inp["T"] = input_from_e7(e7_summary)
    if start_summary is not None:
        inp["S"], inp["XPL"] = input_from_start(start_summary)
    if recovery_score is not None:
        inp["C"] = input_from_recovery(recovery_score)
    if k is not None:
        inp["K"] = k
    return inp


# ------------------------------------------------------------------ 1 枚の要約（利用者向けの日本語。60_paper.py の FORBIDDEN の語を使わない）
SETTING_JA = {"paper_formula_range44_cap5": "論文の式の重み（範囲 44・上限 5）", "range40_cap5": "範囲 40・上限 5",
              "range10_cap5": "範囲 10・上限 5", "ZEROS": "重みを前の行だけにする（範囲 44・上限 5）",
              "naive": "RTC なし（基準）", "current_repro": "今の RTC の再現"}
ARM_JA = {"EH": "学習の home の姿勢へ戻す", "ES": "学習の待機位置の始めの姿勢へ戻す"}
COND_JA = {"R.c1": "最初に閉じた位置の半径方向の差（中央値、mm）", "R.c2": "移動の比（最初の 30 秒、中央値）",
           "R.c3": "通常の試行の成功数（30 秒の採点）", "R.c4": "動作の切り替わりでの速度の跳び（最初の 30 秒、中央値、m/s）",
           "T.c1": "2 番目の手順の 1 回目の閉じで持ち上がった割合", "T.c2": "同じ割合（今の実行器の基準 + 要求の幅）",
           "T.c3": "3 個とも箱に入った本数", "C.c1": "落下で手を止める版の復帰（30 秒）",
           "C.c2": "復帰デモあり − なし の差（両方で失敗が起きた組、30 秒）", "C.c3": "両方で失敗が起きた組の数",
           "high": "高いほうの +y のずれ", "diff": "差", "p": "片側の正確検定の p"}
CAND_JA = {"internal_state": "移植してもずれが再現しない（原因は実行系の内部の状態）", "prior_cube": "先客で再現",
           "pose": "姿勢で再現", "none": "どちらも再現しない"}
B4_JA = {"start_state_prior_cube": "開始状態（先客のデータ）", "slip": "滑りのデータ", "none": "なし（束 4 は研究の道へ）"}


def _v(rec) -> str:
    if rec.get("result") == UNDET:
        return "—"
    v = rec.get("value")
    kn = rec.get("k_n")
    if kn:
        return f"{kn} = {v:.3f}" if isinstance(v, float) else f"{kn}"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _row(name, rec) -> str:
    thr = rec.get("threshold")
    t = "—" if thr is None else f"{rec.get('op', '')} {thr:.4g}" if isinstance(thr, float) else f"{rec.get('op', '')} {thr}"
    why = f"（{rec['reason']}）" if rec.get("reason") else ""
    return f"| {name} | {_v(rec)} | {t} | {JA[rec['result']]}{why} |".replace("|on_grid", "／on_grid").replace("|wall_side", "／wall_side")


def summary_md(res: dict) -> str:
    c = res["conclusion"]
    G = res["gates"]
    gf = res["gates_file"]
    o = ["# 関門 1 の判定（1 枚）", ""]
    o.append(f"- 決まりのファイル: `{gf['path']}`（SHA-256、改行 LF: `{gf['sha256_lf']}`。"
             f"{'公開した値と一致' if gf['matches_posted'] else '公開した値と違う。確かめるまで使わない'}）")
    o.append(f"- 全体: {'結論が出た' if c['status'] == 'determined' else '判定できない所があるので、その結論は止めた'}")
    o += ["", "## 結論", ""]
    rtc = c["bundle2"]["rtc"]
    if rtc["result"] == UNDET:
        o.append(f"- 束 2 の RTC の設定: 判定できない（{rtc.get('reason') or '規則がない'}）")
    elif rtc["b2"]:
        o.append("- 束 2 の RTC の設定: " + "、".join(SETTING_JA.get(n, n) for n in rtc["settings"]) + " を B2 へ送る")
    else:
        o.append("- 束 2 の RTC の設定: 条件を満たす設定がない。調整をやめ、本文の書き直しだけにする（B2 は回さない）")
    v3 = c["bundle2"]["v3a"]
    if v3["result"] == UNDET:
        o.append(f"- 束 2 の戻し先（v3 (a)）: 判定できない（{v3.get('reason') or '関門 T が判定できない'}）")
    elif v3["include"]:
        o.append(f"- 束 2 の戻し先（v3 (a)）: {ARM_JA.get(v3['return_to'], v3['return_to'])}")
    else:
        o.append("- 束 2 の戻し先（v3 (a)）: 入れない（どちらの戻し先も条件を満たさない）")
    b4 = c["bundle4"]
    if b4["result"] == UNDET:
        o.append(f"- 束 4 の候補: 判定できない（{b4.get('reason') or '規則がない'}）")
    else:
        o.append(f"- 束 4 の候補: {B4_JA[b4['candidate']]}" + (f"。{b4['note']}" if b4.get("note") else ""))
    ex = c["ex_arm"]
    o.append("- 実行系を作り直す腕（EX）: " + ("判定できない" if ex is None else "足す" if ex else "足さない"))
    if c["research_path"]:
        o.append("- 研究の道へ回すもの: " + "、".join(c["research_path"]))
    for w in c["warnings"]:
        o.append(f"- 警告: {w}")
    if c["undetermined"]:
        o.append("- 判定できない所: " + "、".join(c["undetermined"]) + "（判定できないときの枝の規則がないので、そこで止めた）")
    o += ["", "## 関門 R（RTC の設定）", "", "| 条件 | 使った値 | 閾値 | 結果 |", "|---|---|---|---|"]
    for name, s in G["R"]["settings"].items():
        for rec in s["conditions"]:
            o.append(_row(f"{SETTING_JA.get(name, name)}: {rec['id']} {COND_JA.get(rec['id'], '')}", rec))
    o.append("")
    o.append(f"- 合格した設定: {'、'.join(SETTING_JA.get(n, n) for n in G['R']['passed']) or 'なし'}。"
             f"枝: {_branch_ja(G['R']['branch'])}")
    b1 = G["R"]["branches"].get("R.b1", {})
    o.append(f"- 分岐 R.b1（先の計画が短い。合否とは独立）: {HIT[b1.get('result', UNDET)]}")
    o += ["", "## 関門 T（2 番目の手順の始めに戻す）", ""]
    ref = G["T"]["reference"]
    if "E0_ref_lift" in ref:
        o.append(f"- 今の実行器の 2 回: {ref['E0_run1']}・{ref['E0_run2']}。基準 {ref['E0_ref_lift']:.3f}、2 回の差 {ref['d_E0']:.3f}、"
                 f"要求の幅 {ref['required_margin']:.3f}")
    else:
        o.append(f"- 今の実行器の基準: 判定できない（{ref.get('reason')}）")
    o += ["", "| 条件 | 使った値 | 閾値 | 結果 |", "|---|---|---|---|"]
    for arm, a in G["T"]["arms"].items():
        for rec in a["conditions"]:
            o.append(_row(f"{ARM_JA[arm]}: {rec['id']} {COND_JA.get(rec['id'], '')}", rec))
    o.append("")
    o.append(f"- 関門 T: {JA[G['T']['result']]}。分岐 T.b1（緑を 1 番目にしても持ち上がりが低い）: "
             f"{HIT[G['T']['branches']['T.b1']['result']]}、分岐 T.b2（今の実行器の 2 回の差が大きい）: {HIT[G['T']['branches']['T.b2']['result']]}")
    o += ["", "## 関門 S と移植の腕（開始状態）", "", "| 条件 | 使った値 | 閾値 | 結果 |", "|---|---|---|---|"]
    for cid, ct in G["S"]["contrasts"].items():
        lab = "姿勢の効き" if cid == "S.pose" else "先客の効き"
        for rec in ct["conditions"]:
            sub = rec["id"].split(".")[-1]
            o.append(_row(f"{lab}: {COND_JA.get(sub, sub)}", rec))
    for cid in ("XPL.rep",):
        o.append(_row("移植してそのまま: +y のずれ", G["XPL"][cid]))
    part = {"both": "2 回合わせて", "rep1": "1 回目", "rep2": "2 回目"}
    for cid, lab in (("XPL.pose", "腕だけ home へ移した"), ("XPL.prior", "赤だけ格子へ移した")):
        for rec in G["XPL"][cid]["conditions"]:
            sub = rec["id"].split(".")[-1]
            o.append(_row(f"{lab}ときに +y のずれが下がった幅（{part.get(sub, sub)}）", rec))
    o.append("")
    cs = res["candidate_selection"]
    o.append(f"- 候補の選び方: " + (f"行 {cs['row']}（{CAND_JA[cs['verdict']]}）" if cs.get("row") else f"判定できない（{cs.get('reason')}）"))
    o += ["", "## 関門 C（落下・手を止める版）", "", "| 条件 | 使った値 | 閾値 | 結果 |", "|---|---|---|---|"]
    for rec in G["C"]["conditions"]:
        o.append(_row(f"{rec['id']} {COND_JA.get(rec['id'], '')}", rec))
    o.append("")
    o.append(f"- 関門 C: {JA[G['C']['result']]}（60 秒の採点は並べるだけで判定に使わない）")
    o += ["", "## 関門 K（回し直しの食い違い率）", ""]
    K = G["K"]
    o.append(f"- K.c1（2 回とも最後まで終わり、対応が一致）: {JA[K['K.c1']['result']]}")
    for L, d in (K.get("d0") or {}).items():
        if d.get("result") == UNDET:
            o.append(f"- {L} 秒の採点: 判定できない（{d['reason']}）")
        else:
            o.append(f"- {L} 秒の採点: {d['k_n']} = {d['value']:.3f}（警告の境界 > {d['threshold']}）")
    o.append(f"- 警告: {'判定できない' if K.get('warning') is None and K.get('d0_listed') else 'あり' if K.get('warning') else 'なし'}")
    o += ["", "## 規則として当てはめなかった所", ""]
    o.append(f"- 文が想定の形と違って読めなかった規則: {len(res['rules_unreadable'])} 件"
             + ("（" + "、".join(p["id"] for p in res["rules_unreadable"]) + "）" if res["rules_unreadable"] else ""))
    o.append(f"- 機械が読める形で書かれていないので当てはめなかった規則: {len(res['not_machine_readable'])} 件（結果の JSON の not_machine_readable）")
    o.append(f"- 決まりのファイルの外の定義を合わせて読んだ所: {len(res['read_with_outside_definitions'])} 件（read_with_outside_definitions）")
    return "\n".join(o) + "\n"


def _branch_ja(b) -> str:
    return {"continue": "続ける", "stop": "やめる", UNDET: "判定できない"}.get(b, str(b))
