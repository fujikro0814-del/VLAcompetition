"""関門 1 の判定（recovla.eval.gate1・scripts/98_s4_gate1.py）の検査。合成の入力だけで回す（numpy・scipy だけ）。

基準の入力 base() は、どの関門も合格・候補の選び方は行 3（姿勢で再現、T 合格）・関門 C 合格・K の警告なしになるように作った。
各検査はその写しを 1 か所ずつ変える。
"""
import copy
import importlib.util
import json
import pathlib

import pytest

from recovla.eval import gate1 as G

ROOT = pathlib.Path(__file__).resolve().parents[1]
GATES = G.load_gates(ROOT / "configs" / "s4_gates.json")


def _script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def kn(k, n):
    return {"k": k, "n": n}


def setting(rad, mr, seam, succ, n=30):
    return {"n_trials": n, "radial_gap_mm": rad, "move_ratio": mr, "seam_jump_mps": seam, "natural_success_30": succ}


def base() -> dict:
    return {
        "schema": G.INPUT_SCHEMA,
        "R": {"settings": {"naive": setting(-5.0, 1.0, 0.0137, 25),
                           "current_repro": setting(-35.9, 0.78, 0.0135, 20),
                           "paper_formula_range44_cap5": setting(-6.0, 0.98, 0.014, 24),
                           "range40_cap5": setting(-10.0, 0.96, 0.015, 24),
                           "range10_cap5": setting(-20.0, 1.0, 0.012, 26),     # R.c1 で落ちる（-20 < -15）
                           "ZEROS": setting(-4.0, 0.90, 0.012, 26)},           # R.c2 で落ちる
              "shadow_plan_shorter_mm": {"10": 5.0, "20": 8.0, "30": 10.0, "40": 12.0},
              "x2_shortfall_mm": {"10": 3.0, "20": 6.0, "30": 9.0, "40": 14.9}},
        "T": {"arms": {"E0_run1": {"first_close_lift": kn(10, 40), "all_three_true": kn(12, 40)},
                       "E0_run2": {"first_close_lift": kn(8, 40), "all_three_true": kn(10, 40)},
                       "EH": {"first_close_lift": kn(30, 40), "all_three_true": kn(14, 40)},
                       "ES": {"first_close_lift": kn(26, 40), "all_three_true": kn(13, 40)}},
              "EO_green_first": {"first_close_lift": kn(30, 40), "colors": ["green"]}},
        "S": {"conditions": {"standby_end|on_grid": {"plus_y_shift": kn(20, 33)},
                             "standby_start|on_grid": {"plus_y_shift": kn(3, 33)},
                             "standby_end|wall_side": {"plus_y_shift": kn(22, 33)}}},
        "XPL": {"arms": {"XPL_as": {"rep1": kn(16, 20), "rep2": kn(15, 20)},
                         "XPL_home": {"rep1": kn(4, 20), "rep2": kn(5, 20)},
                         "XPL_grid": {"rep1": kn(15, 20), "rep2": kn(14, 20)}}},
        "C": {"fall_with_hold": {"by_L": {"30": {"recovery_R": kn(10, 40), "recovery_N": kn(3, 40),
                                                 "paired": {"pairs": 38, "r_only": 8, "n_only": 2}},
                                          "60": {"recovery_R": kn(14, 44), "paired": {"pairs": 42, "r_only": 9, "n_only": 3}}}}},
        "K": {"run1_completed": 99, "run2_completed": 99, "pairs_matched": True, "d0": {"30": kn(12, 99), "60": kn(7, 99)}},
    }


def run(inp, gates=GATES):
    return G.evaluate(gates, inp)


def cond(rec_list, cid):
    return next(c for c in rec_list if c["id"] == cid)


# ------------------------------------------------------------------ 決まりのファイルを読む

def test_real_gates_every_rule_text_is_read():
    rules, problems = G.read_rules(GATES)
    assert problems == []
    assert len(rules) == len(G.SPECS)
    assert rules["R.c2"]["a"] == G.Fraction("0.95") and rules["K.b2"]["a"] == G.Fraction("0.364")
    assert rules["R.rank4"]["order_names"] == ["paper_formula_range44_cap5", "range40_cap5", "range10_cap5", "ZEROS"]
    assert GATES["__file__"]["sha256_lf"] == G.POSTED_SHA256          # 改行が CRLF の作業コピーでも掲示の値と一致
    # 掲示板 0165（改訂 4。test2_misplace の行を足しただけの版）の値。0162 の値は dcf0dd4c…、0153 の値は cf2e8c96…（gate1.py のコメント）
    assert (G.POSTED_BOARD, G.POSTED_SHA256) == ("0165", "7f2f651cafa9cf97b5548324d3fb8ea0bec5e891cca9c8859c7dd9c0347646e9")


def test_not_machine_readable_list_has_file_lines():
    res = run(base())
    assert all(x["where"].startswith("configs/s4_gates.json:") for x in res["not_machine_readable"])
    assert any("C.b2" in x["item"] for x in res["not_machine_readable"])


def test_changed_rule_text_is_not_applied():
    g = copy.deepcopy(GATES)
    G._get(g, ("gates", "R", "continue", "all_of", {"id": "R.c2"}))["rule"] = "> 0.95"
    res = run(base(), g)
    assert [p["id"] for p in res["rules_unreadable"]] == ["R.c2"]
    c2 = cond(res["gates"]["R"]["settings"]["paper_formula_range44_cap5"]["conditions"], "R.c2")
    assert c2["result"] == G.UNDET and "規則を読めない" in c2["reason"]
    assert res["conclusion"]["bundle2"]["rtc"]["result"] == G.UNDET


def test_threshold_is_read_from_json():
    g = copy.deepcopy(GATES)
    G._get(g, ("gates", "R", "continue", "all_of", {"id": "R.c2"}))["rule"] = ">= 0.97"
    res = run(base(), g)
    assert res["gates"]["R"]["passed"] == ["paper_formula_range44_cap5"]     # 0.96 の range40 が落ちる
    assert res["gates"]["R"]["branches"]["R.b2"]["applies"] is True


# ------------------------------------------------------------------ 基準の入力

def test_base_conclusion():
    res = run(base())
    c = res["conclusion"]
    assert c["status"] == "determined", c["undetermined"]
    assert c["bundle2"]["rtc"]["settings"] == ["paper_formula_range44_cap5", "range40_cap5"]   # 成功数が同じ → 跳びの小さいほう
    assert c["bundle2"]["v3a"] == {"result": "determined", "include": True, "return_to": "EH"}
    assert res["candidate_selection"]["row"] == 3 and res["candidate_selection"]["executor_suffices"] is True
    assert c["bundle4"]["candidate"] == "slip" and c["ex_arm"] is False
    assert res["gates"]["K"]["warning"] is False and res["gates"]["K"]["branch_id"] == "K.b1"
    assert res["gates"]["R"]["branches"]["R.b1"]["applies"] is False
    assert res["gates"]["T"]["branches"]["T.b1"]["applies"] is False


# ------------------------------------------------------------------ 各関門の合格と不合格

def test_R_fail_all_settings():
    inp = base()
    for n in ("paper_formula_range44_cap5", "range40_cap5"):
        inp["R"]["settings"][n]["seam_jump_mps"] = 0.02
    res = run(inp)
    assert res["gates"]["R"]["result"] == G.FAIL and res["gates"]["R"]["branch"] == "stop"
    assert res["conclusion"]["bundle2"]["rtc"] == {"result": "determined", "settings": [], "b2": False,
                                                    "action": "RTC の設定の調整をやめ、束 5 の F0 だけを行う"}


def test_R_ranking_by_success_first():
    inp = base()
    inp["R"]["settings"]["range40_cap5"]["natural_success_30"] = 25
    assert run(inp)["gates"]["R"]["top"] == ["range40_cap5", "paper_formula_range44_cap5"]


def test_R_c3_needs_same_trial_count():
    inp = base()
    inp["R"]["settings"]["range40_cap5"]["n_trials"] = 29
    res = run(inp)
    c3 = cond(res["gates"]["R"]["settings"]["range40_cap5"]["conditions"], "R.c3")
    assert c3["result"] == G.UNDET and "試行数" in c3["reason"]
    assert res["conclusion"]["bundle2"]["rtc"]["result"] == G.UNDET


def test_R_b1_both_sides():
    inp = base()
    inp["R"]["shadow_plan_shorter_mm"]["30"] = 15.0
    assert run(inp)["gates"]["R"]["branches"]["R.b1"]["applies"] is False       # X2 の側がまだ 14.9
    inp["R"]["x2_shortfall_mm"]["40"] = 15.0
    res = run(inp)
    assert res["gates"]["R"]["branches"]["R.b1"]["applies"] is True
    assert any("R.b1" in x for x in res["conclusion"]["research_path"])


def test_T_fail_both_arms():
    inp = base()
    inp["T"]["arms"]["EH"]["first_close_lift"] = kn(23, 40)    # 0.575 < 0.60
    inp["T"]["arms"]["ES"]["all_three_true"] = kn(11, 40)      # E0 の大きいほう 12 に届かない
    res = run(inp)
    assert res["gates"]["T"]["result"] == G.FAIL and res["gates"]["T"]["branch"] == "stop"
    assert res["conclusion"]["bundle2"]["v3a"]["include"] is False


def test_T_one_arm_and_tie():
    inp = base()
    inp["T"]["arms"]["EH"]["all_three_true"] = kn(11, 40)
    assert run(inp)["gates"]["T"]["return_to"] == "ES"
    inp = base()
    inp["T"]["arms"]["ES"]["first_close_lift"] = kn(30, 40)    # 同じ持ち上がり → EH
    assert run(inp)["gates"]["T"]["return_to"] == "EH"


def test_T_b1_and_b2():
    inp = base()
    inp["T"]["EO_green_first"]["first_close_lift"] = kn(24, 40)   # ちょうど 0.60 は「以下」に入る
    inp["T"]["arms"]["E0_run1"]["first_close_lift"] = kn(17, 40)  # 0.425 と 0.2 → 差 0.225 > 0.20
    res = run(inp)
    assert res["gates"]["T"]["branches"]["T.b1"]["applies"] is True
    assert res["gates"]["T"]["branches"]["T.b2"]["applies"] is True
    assert res["gates"]["T"]["reference"]["required_margin"] == pytest.approx(0.325)
    assert any("T.b2" in w for w in res["conclusion"]["warnings"])


def test_C_fail_when_pairs_below_20():
    inp = base()
    inp["C"]["fall_with_hold"]["by_L"]["30"]["paired"] = {"pairs": 19, "r_only": 8, "n_only": 2}
    res = run(inp)
    assert res["gates"]["C"]["result"] == G.FAIL
    assert cond(res["gates"]["C"]["conditions"], "C.c3")["result"] == G.FAIL
    assert res["conclusion"]["bundle4"]["candidate"] == "none"
    assert any("落下と置き損ね" in x for x in res["conclusion"]["research_path"])


def test_C_uses_30s_not_60s():
    inp = base()
    inp["C"]["fall_with_hold"]["by_L"]["30"]["recovery_R"] = kn(7, 40)     # 30 s は 0.175（60 s は 0.32 のまま）
    res = run(inp)
    assert cond(res["gates"]["C"]["conditions"], "C.c1")["result"] == G.FAIL


# ------------------------------------------------------------------ ちょうど閾値

def test_boundary_R():
    inp = base()
    inp["R"]["settings"]["paper_formula_range44_cap5"] = setting(-15.0, 0.95, 0.019, 21)   # どれもちょうど
    res = run(inp)
    s = res["gates"]["R"]["settings"]["paper_formula_range44_cap5"]
    assert s["result"] == G.PASS, s
    inp["R"]["settings"]["paper_formula_range44_cap5"] = setting(-15.0, 0.9499, 0.019, 21)
    assert run(inp)["gates"]["R"]["settings"]["paper_formula_range44_cap5"]["result"] == G.FAIL
    inp["R"]["settings"]["paper_formula_range44_cap5"] = setting(-15.0, 0.95, 0.0191, 21)
    assert run(inp)["gates"]["R"]["settings"]["paper_formula_range44_cap5"]["result"] == G.FAIL
    inp["R"]["settings"]["paper_formula_range44_cap5"] = setting(-15.0, 0.95, 0.019, 20)
    assert run(inp)["gates"]["R"]["settings"]["paper_formula_range44_cap5"]["result"] == G.FAIL


def test_boundary_T():
    inp = base()
    inp["T"]["arms"]["E0_run1"]["first_close_lift"] = kn(14, 40)    # 0.35 と 0.30 → 基準 0.35 + 要求の幅 0.25 = 0.60
    inp["T"]["arms"]["E0_run2"]["first_close_lift"] = kn(12, 40)
    inp["T"]["arms"]["EH"]["first_close_lift"] = kn(24, 40)         # ちょうど 0.60（T.c1 と T.c2 の両方の境界）
    inp["T"]["arms"]["EH"]["all_three_true"] = kn(12, 40)           # ちょうど E0 の大きいほう
    res = run(inp)
    assert res["gates"]["T"]["arms"]["EH"]["result"] == G.PASS
    inp["T"]["arms"]["E0_run1"]["first_close_lift"] = kn(16, 40)    # 0.40 と 0.30 → 差 0.10 + 0.10 = 0.20 < 0.25 なので幅 0.25
    inp["T"]["arms"]["EH"]["first_close_lift"] = kn(26, 40)         # 0.65 = 0.40 + 0.25
    assert run(inp)["gates"]["T"]["arms"]["EH"]["result"] == G.PASS
    inp["T"]["arms"]["EH"]["first_close_lift"] = kn(25, 40)
    assert cond(run(inp)["gates"]["T"]["arms"]["EH"]["conditions"], "T.c2")["result"] == G.FAIL


def test_boundary_S_XPL_C():
    inp = base()
    inp["S"]["conditions"]["standby_end|on_grid"]["plus_y_shift"] = kn(8, 20)     # 0.40
    inp["S"]["conditions"]["standby_start|on_grid"]["plus_y_shift"] = kn(3, 20)   # 0.15 → 差 0.25
    res = run(inp)
    ct = res["gates"]["S"]["contrasts"]["S.pose"]["conditions"]
    assert cond(ct, "S.pose.high")["result"] == G.PASS and cond(ct, "S.pose.diff")["result"] == G.PASS
    assert cond(ct, "S.pose.p")["result"] == G.FAIL                               # 20 と 20 では p が 0.05 を超える
    inp = base()
    inp["XPL"]["arms"]["XPL_as"] = {"rep1": kn(8, 20), "rep2": kn(8, 20)}          # ちょうど 0.40
    assert run(inp)["gates"]["XPL"]["XPL.rep"]["result"] == G.PASS
    inp["XPL"]["arms"]["XPL_as"] = {"rep1": kn(8, 20), "rep2": kn(7, 20)}
    assert run(inp)["gates"]["XPL"]["XPL.rep"]["result"] == G.FAIL
    inp = base()
    inp["XPL"]["arms"]["XPL_home"] = {"rep1": kn(13, 20), "rep2": kn(8, 20)}      # 2 回合わせて 0.775 - 0.525 = 0.25、各回 0.15・0.35
    x = run(inp)["gates"]["XPL"]["XPL.pose"]
    assert x["result"] == G.PASS, x
    inp = base()
    inp["C"]["fall_with_hold"]["by_L"]["30"] = {"recovery_R": kn(8, 40), "paired": {"pairs": 20, "r_only": 3, "n_only": 1}}
    res = run(inp)
    assert [c["result"] for c in res["gates"]["C"]["conditions"]] == [G.PASS] * 3


def test_boundary_K():
    inp = base()
    inp["K"]["d0"]["30"] = kn(364, 1000)            # ちょうど 0.364 は警告にしない（> のとき警告）
    assert run(inp)["gates"]["K"]["warning"] is False
    inp["K"]["d0"]["30"] = kn(365, 1000)
    assert run(inp)["gates"]["K"]["warning"] is True


# ------------------------------------------------------------------ 値がない・形が違う・分母が 0

def test_missing_value_setting_is_undetermined():
    inp = base()
    del inp["R"]["settings"]["range40_cap5"]["move_ratio"]
    res = run(inp)
    s = res["gates"]["R"]["settings"]["range40_cap5"]
    assert s["result"] == G.UNDET and "値がない" in cond(s["conditions"], "R.c2")["reason"]
    assert res["gates"]["R"]["branch"] == "continue"                 # 合格が 1 つあるので、やめる枝ではない
    assert res["conclusion"]["bundle2"]["rtc"]["result"] == G.UNDET   # 上位 2 つは決まらない
    assert res["conclusion"]["status"] == "undetermined"


def test_missing_value_not_needed_when_already_failed():
    inp = base()
    inp["R"]["settings"]["ZEROS"]["radial_gap_mm"] = None             # ZEROS は R.c2 で落ちているので結論は変わらない
    res = run(inp)
    assert res["gates"]["R"]["settings"]["ZEROS"]["result"] == G.FAIL
    assert res["conclusion"]["bundle2"]["rtc"]["result"] == "determined"


def test_zero_denominator_and_bad_shape():
    inp = base()
    inp["T"]["arms"]["ES"]["first_close_lift"] = kn(0, 0)
    inp["S"]["conditions"]["standby_end|wall_side"]["plus_y_shift"] = kn(30, 33)   # S.prior は効きあり → 行 2 は XPL.prior しだい
    inp["XPL"]["arms"]["XPL_grid"]["rep1"] = {"k": "15", "n": 20}
    inp["C"]["fall_with_hold"]["by_L"]["30"]["paired"] = {"pairs": 0, "r_only": 0, "n_only": 0}
    res = run(inp)
    assert "分母が 0" in cond(res["gates"]["T"]["arms"]["ES"]["conditions"], "T.c1")["reason"]
    assert res["gates"]["T"]["result"] == G.PASS and res["gates"]["T"]["return_to"] is None   # EH は合格、選び方が決まらない
    assert res["conclusion"]["bundle2"]["v3a"]["result"] == G.UNDET
    assert res["gates"]["XPL"]["XPL.prior"]["result"] == G.UNDET
    assert res["candidate_selection"]["result"] == G.UNDET                 # 行 2 が判定できないので止める
    assert res["gates"]["C"]["result"] == G.FAIL                            # 対 0 は C.c3 で落ちる（C.c2 は判定できない）
    assert cond(res["gates"]["C"]["conditions"], "C.c2")["result"] == G.UNDET
    inp["S"]["conditions"]["standby_end|wall_side"]["plus_y_shift"] = kn(22, 33)   # S.prior が効きなし → 行 2 は不合格に決まる
    assert run(inp)["candidate_selection"]["row"] == 3


def test_rate_must_match_k_over_n():
    inp = base()
    inp["S"]["conditions"]["standby_start|on_grid"]["plus_y_shift"] = {"k": 3, "n": 33, "rate": 0.5}
    res = run(inp)
    assert res["gates"]["S"]["contrasts"]["S.pose"]["result"] == G.UNDET


def test_empty_input_stops_everywhere():
    res = run({"schema": G.INPUT_SCHEMA})
    c = res["conclusion"]
    assert c["status"] == "undetermined"
    assert c["bundle2"]["rtc"]["result"] == G.UNDET and c["bundle4"]["result"] == G.UNDET and c["ex_arm"] is None
    assert res["gates"]["K"]["K.c1"]["result"] == G.UNDET


def test_start_candidate_wins_even_if_C_undetermined():
    inp = base()
    inp["S"]["conditions"]["standby_end|wall_side"]["plus_y_shift"] = kn(30, 33)
    inp["XPL"]["arms"]["XPL_grid"] = {"rep1": kn(5, 20), "rep2": kn(6, 20)}
    del inp["C"]
    res = run(inp)
    assert res["gates"]["C"]["result"] == G.UNDET
    assert res["conclusion"]["bundle4"]["candidate"] == "start_state_prior_cube"


def test_C_undetermined_without_start_candidate_stops():
    inp = base()
    del inp["C"]
    res = run(inp)
    assert res["conclusion"]["bundle4"]["result"] == G.UNDET


# ------------------------------------------------------------------ 候補の選び方の 3 通り

def test_candidate_prior_cube():
    """先客で再現: S.prior と XPL.prior を満たす。T の結果は問わない（T を不合格にしても同じ）。"""
    inp = base()
    inp["S"]["conditions"]["standby_end|wall_side"]["plus_y_shift"] = kn(30, 33)   # 0.91 対 0.61
    inp["XPL"]["arms"]["XPL_grid"] = {"rep1": kn(5, 20), "rep2": kn(6, 20)}
    inp["T"]["arms"]["EH"]["first_close_lift"] = kn(10, 40)
    inp["T"]["arms"]["ES"]["first_close_lift"] = kn(10, 40)
    res = run(inp)
    cs = res["candidate_selection"]
    assert cs["row"] == 2 and cs["verdict"] == "prior_cube" and cs["ex_arm"] is False
    assert res["gates"]["T"]["result"] == G.FAIL
    c = res["conclusion"]
    assert c["bundle4"]["candidate"] == "start_state_prior_cube"
    assert any("滑り" in x for x in c["research_path"])       # 関門 C も合格 → 開始状態を優先し、滑りは研究の道へ


def test_candidate_pose_with_T_pass():
    """姿勢で再現して T も合格: 実行器で足りる。束 4 に開始状態は入らない。"""
    res = run(base())
    cs = res["candidate_selection"]
    assert (cs["row"], cs["verdict"], cs["executor_suffices"], cs["ex_arm"]) == (3, "pose", True, False)
    assert res["conclusion"]["bundle4"]["candidate"] == "slip"
    assert any("姿勢のデータ" in x for x in res["conclusion"]["research_path"])


def test_candidate_pose_with_T_fail_adds_EX():
    inp = base()
    inp["T"]["arms"]["EH"]["first_close_lift"] = kn(10, 40)
    inp["T"]["arms"]["ES"]["first_close_lift"] = kn(10, 40)
    res = run(inp)
    assert res["candidate_selection"]["row"] == 3 and res["candidate_selection"]["ex_arm"] is True
    assert res["conclusion"]["ex_arm"] is True


def test_candidate_neither_reproduces():
    """どちらも再現しない: 移植では再現する（XPL.rep）が、姿勢も先客も原因を示さない → 行 4。"""
    inp = base()
    inp["XPL"]["arms"]["XPL_home"] = {"rep1": kn(15, 20), "rep2": kn(14, 20)}
    res = run(inp)
    cs = res["candidate_selection"]
    assert (cs["row"], cs["verdict"], cs["ex_arm"]) == (4, "none", False)
    assert any("開始状態のデータ" in x for x in res["conclusion"]["research_path"])


def test_candidate_not_reproduced_by_transplant():
    inp = base()
    inp["XPL"]["arms"]["XPL_as"] = {"rep1": kn(3, 20), "rep2": kn(2, 20)}
    res = run(inp)
    cs = res["candidate_selection"]
    assert (cs["row"], cs["verdict"], cs["ex_arm"]) == (1, "internal_state", True)
    assert res["gates"]["XPL"]["XPL.pose"]["result"] == G.FAIL       # XPL.rep を満たさないので


# ------------------------------------------------------------------ 関門 K の警告

def test_K_warning_over_0364():
    inp = base()
    inp["K"]["d0"]["30"] = kn(37, 99)                  # 0.374 > 0.364
    res = run(inp)
    assert res["gates"]["K"]["warning"] is True and res["gates"]["K"]["branch_id"] == "K.b2"
    assert any(w.startswith("K.b2") for w in res["conclusion"]["warnings"])
    inp = base()
    inp["K"]["d0"]["30"] = kn(36, 99)                  # 0.3636 <= 0.364
    inp["K"]["d0"]["60"] = kn(37, 99)                  # 60 s の側だけが超えても警告
    assert run(inp)["gates"]["K"]["warning"] is True


def test_K_warning_with_one_side_missing():
    inp = base()
    del inp["K"]["d0"]["60"]
    assert run(inp)["gates"]["K"]["warning"] is None            # 30 s は前例の内、60 s が分からない
    inp["K"]["d0"]["30"] = kn(37, 99)
    assert run(inp)["gates"]["K"]["warning"] is True            # 片方が超えれば警告は決まる


def test_K_c1_fail_no_d0():
    inp = base()
    inp["K"]["run2_completed"] = 97
    res = run(inp)
    assert res["gates"]["K"]["branch"] == "stop" and res["gates"]["K"]["d0_listed"] is False
    assert any(w.startswith("K.c1") for w in res["conclusion"]["warnings"])


# ------------------------------------------------------------------ 全部不合格

def test_all_fail_conclusion():
    inp = base()
    for n in ("paper_formula_range44_cap5", "range40_cap5", "range10_cap5", "ZEROS"):
        inp["R"]["settings"][n]["move_ratio"] = 0.80
    inp["T"]["arms"]["EH"]["first_close_lift"] = kn(10, 40)
    inp["T"]["arms"]["ES"]["first_close_lift"] = kn(10, 40)
    inp["S"]["conditions"]["standby_end|on_grid"]["plus_y_shift"] = kn(3, 33)
    inp["XPL"]["arms"]["XPL_as"] = {"rep1": kn(2, 20), "rep2": kn(1, 20)}
    inp["C"]["fall_with_hold"]["by_L"]["30"] = {"recovery_R": kn(2, 40), "paired": {"pairs": 36, "r_only": 2, "n_only": 2}}
    res = run(inp)
    c = res["conclusion"]
    assert c["status"] == "determined"
    assert c["bundle2"]["rtc"]["settings"] == [] and c["bundle2"]["rtc"]["b2"] is False
    assert c["bundle2"]["v3a"]["include"] is False
    assert c["bundle4"]["candidate"] == "none"
    assert c["ex_arm"] is True                                   # 移植で再現しない → 行 1 で EX を足す
    for word in ("束 4", "開始状態のデータ", "落下と置き損ね"):
        assert any(word in x for x in c["research_path"]), word


# ------------------------------------------------------------------ 要約・入力の組み立て・CLI

def test_summary_md_has_no_forbidden_words():
    tr = _script("s4_time_report", "98_s4_time_report.py")
    pats = tr.forbidden_patterns()
    inps = [base(), {"schema": G.INPUT_SCHEMA}]
    x = base()
    x["XPL"]["arms"]["XPL_grid"]["rep1"] = {"k": "15", "n": 20}
    x["K"]["d0"]["30"] = kn(37, 99)
    inps.append(x)
    for inp in inps:
        md = G.summary_md(run(inp))
        assert tr.forbidden_hits(md, pats) == [], tr.forbidden_hits(md, pats)
        assert "R1v3" not in md and "N1v3" not in md


def test_build_input_from_diag_outputs():
    from recovla.diag import rtc as D
    rows = [{"radial_gap_mm": -5.0 + i, "move_ratio": 1.0, "seam_jump_mps": 0.013, "success_at": {"30": i < 2, "45": True, "60": True},
             "horizon_s": 30.0, "t_first_close": 5.0, "first_close_after_horizon": False, "descent_start_mm": None,
             "descent_red_mm": None, "far_whiff_s": None} for i in range(3)]
    summ = D.summarize(rows)
    metrics = {"conditions": {n: {"by_horizon": {"30": {"summary": summ}}} for n in ("naive", "range40_cap5")}}
    shadow = {"conditions": {"shadow_current_repro": {"summary": {"shadow_minus_guided_mm_median": {"10": [3.0, 9], "20": [16.0, 9],
                                                                                                    "30": [None, 0], "40": [1.0, 9]}}}}}
    e7 = {"gate_T_inputs": {"arms": {"EH": {"first_close_lift": {"k": 3, "n": 4, "rate": 0.75}, "all_three_true": {"k": 1, "n": 4, "rate": 0.25},
                                            "timed_out": 0}},
                            "EO_green_first": {"first_close_lift": {"k": 1, "n": 2, "rate": 0.5}, "colors": ["green"]}}}
    start = {"gate_S_inputs": {"conditions": {"standby_end|on_grid": {"plus_y_shift": {"k": 2, "n": 3, "rate": 0.6667},
                                                                       "first_close_after_30s": 1}}},
             "xpl_inputs": {"arms": {"XPL_as": {"both": {"k": 3, "n": 4}, "rep1": {"k": 2, "n": 2, "lift_k": 0}, "rep2": {"k": 1, "n": 2}}}}}
    score = {"variants": {"fall_with_hold": {"material": {"by_L": {"30": {
        "recovery_R": {"recovered": 2, "established": 5}, "recovery_N": {"recovered": 1, "established": 5},
        "paired": {"pairs": 4, "r_only": 1, "n_only": 0, "both": 1}}}}}}}
    inp = G.build_input(rtc_metrics=metrics, rtc_shadow=shadow, e7_summary=e7, start_summary=start, recovery_score=score,
                        k={"run1_completed": 99})
    assert inp["R"]["settings"]["naive"] == {"n_trials": 3, "radial_gap_mm": -4.0, "move_ratio": 1.0, "seam_jump_mps": 0.013,
                                             "natural_success_30": 2, "first_close_after_30s": 0, "n_no_close": 0}
    assert inp["R"]["shadow_plan_shorter_mm"] == {"10": 3.0, "20": 16.0, "30": None, "40": 1.0}
    assert inp["T"]["arms"]["EH"] == {"first_close_lift": kn(3, 4), "all_three_true": kn(1, 4)}
    assert inp["XPL"]["arms"]["XPL_as"] == {"rep1": kn(2, 2), "rep2": kn(1, 2)}
    assert inp["C"]["fall_with_hold"]["by_L"]["30"]["paired"] == {"pairs": 4, "r_only": 1, "n_only": 0}
    res = run(inp)                                               # 欠けた所は判定できないになるだけ（例外にしない）
    assert res["conclusion"]["status"] == "undetermined"
    assert res["gates"]["R"]["branches"]["R.b1"]["shadow_plan_shorter_mm"]["result"] == G.UNDET   # 30 行先が None


def test_cli_writes_result_and_summary(tmp_path):
    cli = _script("s4_gate1", "98_s4_gate1.py")
    p = tmp_path / "in.json"
    p.write_text(json.dumps(base(), ensure_ascii=False), encoding="utf-8")
    out, md = tmp_path / "res.json", tmp_path / "sum.md"
    rc = cli.main(["--gates", str(ROOT / "configs" / "s4_gates.json"), "--input", str(p), "--out", str(out), "--md", str(md)])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["schema"] == G.RESULT_SCHEMA and res["gates_file"]["matches_posted"] is True
    assert res["conclusion"]["bundle4"]["candidate"] == "slip" and len(res["input_sha256"]) == 64
    assert md.read_text(encoding="utf-8").startswith("# 関門 1 の判定")
    inp = base()
    del inp["K"]
    p.write_text(json.dumps(inp), encoding="utf-8")
    assert cli.main(["--input", str(p), "--out", str(out)]) == 3
    p.write_text("{", encoding="utf-8")
    assert cli.main(["--input", str(p)]) == 2
