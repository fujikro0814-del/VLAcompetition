"""関門 1 の二重集計の B（scripts/98_s4_gate1_b.py）の検査。合成の記録と入力だけで回す。

- B が A の部品（recovla.diag・recovla.eval・98_s4_d_*）を読み込まないこと
- 記録の読み方の境界（0→1、持ち上がり、誘発の成立 < L、成功 <= L）
- 規則の当てはめが A（recovla.eval.gate1）と同じ結論になること（test_gate1.py の基準の入力とその変形で）
- E7 の本数の提案（McNemar の検出力）の筋
"""
import copy
import importlib.util
import pathlib
import re

import numpy as np

from recovla.eval import gate1 as G

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = _load("s4_gate1_b", ROOT / "scripts" / "98_s4_gate1_b.py")
TG = _load("test_gate1_for_b", ROOT / "tests" / "test_gate1.py")
GATES = G.load_gates(ROOT / "configs" / "s4_gates.json")


def test_b_does_not_import_a_parts():
    src = (ROOT / "scripts" / "98_s4_gate1_b.py").read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    imports = re.findall(r"^\s*(?:from\s+(\S+)\s+import|import\s+(\S+))", code, flags=re.M)
    mods = {a or b for a, b in imports}
    assert not any(m.startswith(("recovla.diag", "recovla.eval")) for m in mods)
    assert "spec_from_file_location" not in code and "import_module" not in code   # スクリプトを別の道で読み込まない


def test_rules_constants_match_gates_text():
    rules, problems = G.read_rules(GATES)
    assert problems == []
    assert B.RULES["R"]["c2"] == rules["R.c2"]["a"] and B.RULES["R"]["c4"] == rules["R.c4"]["a"]
    assert B.RULES["R"]["c1_mm"] == rules["R.c1"]["a"] and B.RULES["R"]["c3"] == rules["R.c3"]["a"]
    assert B.RULES["T"]["c1"] == rules["T.c1"]["a"] and B.RULES["T"]["b2"] == rules["T.b2"]["a"]
    assert (B.RULES["T"]["margin_min"], B.RULES["T"]["margin_add"]) == (rules["T.required_margin"]["a"], rules["T.required_margin"]["b"])
    assert B.RULES["C"]["c3"] == rules["C.c3"]["a"] and B.RULES["K"]["warn"] == rules["K.b2"]["a"]
    assert B.RULES["XPL"]["rep"] == rules["XPL.rep"]["a"] and B.RULES["XPL"]["each"] == rules["XPL.pose"]["b"]
    assert B.RULES["R"]["candidates"] == rules["R.rank4"]["order_names"]
    assert B.POSTED_LF_SHA == G.POSTED_SHA256


def _z(gc, cube_z, tip_y=0.0, cube_y=0.0):
    n = len(gc)
    cube = np.zeros((n, 3, 3))
    cube[:, 0, 2] = cube_z
    cube[:, 0, 1] = cube_y
    tip = np.zeros((n, 3))
    tip[:, 1] = tip_y
    return {"gripper_closed": np.array(gc, bool), "cube_pos": cube, "fingertip": tip, "sim_time": np.arange(n) * 0.05}


def test_rises_lift_plus_y():
    rest = B.CUBE_REST_Z
    z = _z([0, 0, 1, 1, 0, 1, 1], [rest, rest, rest, rest + 0.021, rest, rest + 0.05, rest], tip_y=0.011)
    assert list(B.rises(z["gripper_closed"])) == [2, 5]
    assert B.lifted(z, 2, 0) is True                      # 開くまでに 2.1 cm
    z2 = _z([0, 1, 1, 0, 0], [rest, rest + 0.02, rest + 0.02, rest + 0.5, rest])
    assert B.lifted(z2, 1, 0) is False                    # ちょうど 2 cm は超えない。開いた後は数えない
    assert B.plus_y(z, 2, 0) is True and B.plus_y(_z([1], [rest], tip_y=0.01), 0, 0) is False
    assert list(B.rises([1, 1, 0, 1])) == [0, 3]          # こま 0 で閉じていればこま 0


def test_induced_boundaries():
    m = {"success": True, "t_success": 30.0, "induce": {"established": True, "t_established": 30.0}}
    assert B.succ_at(m, 30.0) is True                    # 成功は <= L
    assert B.est_before(m, 30.0) is False                # 成立は < L（掲示板 0155 の 1）
    assert B.est_before(m, 60.0) is True
    assert B.est_before({"induce": {"established": False, "t_established": 1.0}}, 30.0) is False


def _b_from_a(inp):
    """A の入力の形を B の規則の関数の形に（XPL の回だけ包みが違う）。"""
    x = {"arms": {arm: {rep: {"plus_y_shift": v} for rep, v in reps.items()} for arm, reps in inp["XPL"]["arms"].items()}}
    return inp["R"], inp["T"], inp["S"], x, inp["C"], inp["K"]


def _both(inp):
    a = G.evaluate(GATES, inp)
    R, T, S, X, C, K = _b_from_a(inp)
    rR, rT, rS, rX, rC, rK = B.rule_R(R), B.rule_T(T), B.rule_S(S), B.rule_XPL(X), B.rule_C(C), B.rule_K(K)
    cand = B.rule_candidates(rS, rX, rT)
    return a, B.conclude(rR, rT, rC, rK, cand), cand


def _same(a, b, cand):
    c = a["conclusion"]
    assert c["bundle2"]["rtc"]["settings"] == b["bundle2"]["rtc_settings"]
    assert c["bundle2"]["v3a"]["include"] == b["bundle2"]["v3a_include"]
    assert c["bundle2"]["v3a"]["return_to"] == b["bundle2"]["v3a_return_to"]
    assert c["bundle4"]["candidate"] == b["bundle4"]
    assert c["ex_arm"] == b["ex_arm"]
    assert a["candidate_selection"]["row"] == cand["row"]
    assert B.research_keys(c["research_path"]) == B.research_keys(b["research_path"])


def test_rules_agree_with_a_on_base_and_variants():
    variants = []
    inp = TG.base()
    variants.append(inp)
    v = copy.deepcopy(inp)                                  # 行 1: 移植で再現しない → EX、滑り
    v["XPL"]["arms"]["XPL_as"] = {"rep1": TG.kn(2, 20), "rep2": TG.kn(3, 20)}
    variants.append(v)
    v = copy.deepcopy(v)                                    # さらに C 不合格（対が 19）→ 候補なし
    v["C"]["fall_with_hold"]["by_L"]["30"]["paired"] = {"pairs": 19, "r_only": 8, "n_only": 2}
    variants.append(v)
    v = copy.deepcopy(inp)                                  # 行 3 で T 不合格 → EX
    v["T"]["arms"]["EH"]["first_close_lift"] = TG.kn(5, 40)
    v["T"]["arms"]["ES"]["first_close_lift"] = TG.kn(5, 40)
    variants.append(v)
    v = copy.deepcopy(inp)                                  # 行 2: 先客で再現
    v["XPL"]["arms"]["XPL_grid"] = {"rep1": TG.kn(4, 20), "rep2": TG.kn(5, 20)}
    v["S"]["conditions"]["standby_end|wall_side"] = {"plus_y_shift": TG.kn(30, 33)}
    v["S"]["conditions"]["standby_end|on_grid"] = {"plus_y_shift": TG.kn(10, 33)}
    variants.append(v)
    v = copy.deepcopy(inp)                                  # R: 合格 0 → やめる
    for n in ("paper_formula_range44_cap5", "range40_cap5"):
        v["R"]["settings"][n]["move_ratio"] = 0.90
    variants.append(v)
    for x in variants:
        a, b, cand = _both(x)
        _same(a, b, cand)


def test_r_ranking_ties_and_exact_threshold():
    inp = TG.base()
    s = inp["R"]["settings"]
    s["range10_cap5"]["radial_gap_mm"] = -15.0              # ちょうど naive − 10 は合格
    s["ZEROS"]["move_ratio"] = 0.95                         # ちょうど 0.95 は合格
    r = B.rule_R(inp["R"])
    assert r["settings"]["range10_cap5"]["pass"] and r["settings"]["ZEROS"]["pass"]
    assert r["top"] == ["ZEROS", "range10_cap5"]            # 成功 26・跳び 0.012 が同じ → |差| の小さい ZEROS が先
    a = G.evaluate(GATES, inp)
    assert a["gates"]["R"]["top"] == r["top"]


def test_fisher_matches_scipy_fisher_exact():
    from scipy.stats import fisher_exact
    for ka, na, kb, nb in ((3, 24, 2, 33), (20, 33, 3, 33), (0, 10, 0, 10)):
        assert abs(B.fisher_greater(ka, na, kb, nb) - fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="greater")[1]) < 1e-12


def test_mcnemar_power_monotone_and_proposal():
    p = [B.mcnemar_power(n, 0.25, 0.40, 0.35) for n in (100, 150, 200)]
    assert p[0] < p[1] < p[2] and 0.5 < p[0] < 0.95
    assert B.mcnemar_power(150, 0.25, 0.40, 0.35, alpha=0.01) < p[1]
    T = {"arms": {"E0_run1": {"all_three_true": TG.kn(8, 40)}, "E0_run2": {"all_three_true": TG.kn(12, 40)}},
         "K_e7": {"rho_hat_phi": 0.3546}}
    e = B.e7_proposal(T)
    assert e["p0"] == 0.25 and e["proposal_n"] == 150
