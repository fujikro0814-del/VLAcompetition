"""束 4 の評価の入口（scripts/98_s4_b4_eval.py）と二重集計（src/recovla/eval/b4.py・scripts/98_s4_b4_b.py・98_s4_b4_check.py）の検査。
CPU だけ。記録は合成、保存点は偽のファイル（評価・GPU・API は使わない）。"""
import importlib.util
import json
import os
import pathlib
import random
import sys
import time
import types
import zlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recovla.eval import b4 as A  # noqa: E402


def _load(name, mod):
    spec = importlib.util.spec_from_file_location(mod, ROOT / "scripts" / name)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


EV = _load("98_s4_b4_eval.py", "t_b4_eval")
B = _load("98_s4_b4_b.py", "t_b4_b")
CK = _load("98_s4_b4_check.py", "t_b4_check")

ENV = {"driver": "610.88", "torch": "2.5.1", "torch_cuda": "12.4", "os_build": "26100.1", "git_head": "abc123"}
FILES = {"scripts/98_s4_b4_eval.py": "f" * 64, "configs/s4_gates.json": A.POSTED_GATES_SHA256}
SHA = {m: (str(i) * 64)[:64] for i, m in enumerate(EV.MODELS)}


# ---------------------------------------------------------------- 合成の記録
def write_cond(root, phase, experiment, cond, part, model, outcome, *, drop=(), env=None, sha=None, audit=True, limit=60.0,
               files=None):
    """outcome(seed, target) -> (established, t_established, success, t_success)。"""
    lo, hi = EV.PHASES[phase]["parts"][part]["band"]
    d = root / experiment / cond
    d.mkdir(parents=True, exist_ok=True)
    i = 0
    for s in range(lo, hi + 1):
        for tgt in (("red", "green", "blue") if part == "nat" else ("blue",)):
            if (s, tgt) in drop or s in drop:
                i += 1
                continue
            est, te, ok, ts = outcome(s, tgt)
            meta = {"trial": i, "seed": s, "target": tgt, "experiment": experiment, "condition": cond, "model": {"name": model},
                    "success": ok, "t_success": ts, "time_limit_s": limit,
                    "induce": {"kind": EV.PART_INDUCE[part], "established": est, "t_established": te},
                    "runtime": {"mode": "naive", "exec_interval": 6, "safety_filter": False},
                    "env": dict(env or ENV),
                    "b4": {"phase": phase, "part": part, "model": model, "ckpt": {"sha256": sha or SHA[model]},
                           "files_sha256": FILES if files is None else files}}
            (d / f"trial_{i:04d}.json").write_text(json.dumps(meta), encoding="utf-8")
            i += 1
    (d / "run.json").write_text(json.dumps({"time_limits": {"time_limit_s": limit}, "env_segments": [{"env": {}}]}), encoding="utf-8")
    if audit:
        (d / "G_AUDIT.json").write_text(json.dumps({"met": True}), encoding="utf-8")
    return d


def rnd_outcome(seed, p_est=0.8, p_ok=0.4):
    def f(s, tgt):
        r = random.Random(f"{seed}-{s}-{tgt}")
        est = r.random() < p_est
        te = round(r.uniform(1.0, 40.0), 2) if est else None
        ok = r.random() < p_ok
        ts = round(r.uniform(5.0, 59.0), 2) if ok else None
        return est, te, ok, ts
    return f


def build_phase(tmp, phase, rates=None, per=None, skip=(), **kw):
    """per: {条件名: write_cond の引数（その条件だけ）}。skip: 書かない条件名。"""
    root = tmp / "v2eval"
    lay = EV.layout_of(phase, root=str(root))
    for name, v in lay["conds"].items():
        if name in skip:
            continue
        part, model = name.split(".", 1)
        p_ok = (rates or {}).get(name, 0.4)
        seed = 7 if part == "nat" else zlib.crc32(name.encode())       # 自然は R4・R1v3 で同じ結果（対の差 0）
        write_cond(root, phase, lay["experiment"], v["cond"], part, model, rnd_outcome(seed, p_ok=p_ok),
                   **dict(kw, **(per or {}).get(name, {})))
    return lay


# ---------------------------------------------------------------- 帯
def test_bands_accept_exact_allocation_and_reject_others():
    assert EV.check_phase_bands("G2") == [] and EV.check_phase_bands("G3") == [] and EV.check_phase_bands("T2") == []
    for ph in EV.PHASES:
        for part in EV.PHASES[ph]["parts"]:
            assert EV.check_band(ph, part, EV.default_trials(ph, part), EV.PHASES[ph]["experiment"], False) == ""
    assert EV.default_trials("G2", "nat") == "natural:192100:66"
    assert EV.default_trials("T2", "P3") == "induced:168000:100"
    assert EV.default_trials("G3", "P2") == "induced:192500:100"
    assert "完全に一致" in EV.check_band("T2", "P2", "induced:167000:99", "S4T2", False)
    assert "完全に一致" in EV.check_band("G2", "P3", "induced:192700:100", "S4B4G2", False)       # 割り当てなしの帯
    assert "種類" in EV.check_band("T2", "P1", "natural:166000:100", "S4T2", False)
    assert "実験名" in EV.check_band("T2", "P2", "induced:167000:100", "S4B4G2", False)            # 段階の実験名だけ
    assert "S4SMOKE" in EV.check_band("T2", "P2", "induced:167000:100", "S4SMOKE_X", False)
    # smoke は 44680〜44683 だけ、S4SMOKE の実験名、--allow-smoke
    assert EV.check_band("T2", "P2", "induced:44680:4", "S4SMOKE_B4", True) == ""
    assert EV.check_band("G2", "nat", "natural:44683:1", "S4SMOKE_B4", True) == ""
    assert "44680" in EV.check_band("T2", "P2", "induced:44679:2", "S4SMOKE_B4", True)
    assert "44680" in EV.check_band("T2", "P2", "induced:44682:3", "S4SMOKE_B4", True)
    assert "S4SMOKE" in EV.check_band("T2", "P2", "induced:44680:1", "S4B4X", True)
    assert "allow-smoke" in EV.check_band("T2", "P2", "induced:44680:1", "S4SMOKE_B4", False)


def test_bands_follow_s4_gates(tmp_path):
    g = json.loads(EV.GATES.read_text(encoding="utf-8"))
    for a in g["bands"]["allocations"]:
        if a["id"] == "test2_misplace":
            a["range"] = [168000, 168049]
    assert any("test2_misplace" in x for x in EV.check_phase_bands("T2", g))
    assert "合わない" in EV.check_band("T2", "P3", "induced:168000:100", "S4T2", False, g)
    # 関門 2・3 の割り当ては bundle4_gate2 の中で重ならない
    lo = {(ph, part): v["band"] for ph in ("G2", "G3") for part, v in EV.PHASES[ph]["parts"].items()}
    spans = sorted(lo.values())
    assert all(b[0] > a[1] for a, b in zip(spans, spans[1:]))
    assert all(192100 <= a and b <= 192999 for a, b in spans)


def test_conditions_and_estimate_match_board_0165():
    assert [c for c, _, _ in EV.conditions("G2")] == ["nat_R4", "nat_R1v3", "P1_R4", "P1_N4", "P1_R1v3", "P2_R4", "P2_R1v3",
                                                      "P3_R4", "P3_N4", "P3_R1v3"]
    assert [c for c, _, _ in EV.conditions("G3")] == ["P1_R4s1001", "P1_N4s1001", "P2_R4s1001", "P2_R1v3s1001"]
    e = {ph: EV.estimate(ph) for ph in EV.PHASES}
    assert (e["G2"]["trials"], e["G2"]["process_h"], e["G2"]["parallel3_h"]) == (1196, 24.6, 8.6)
    assert (e["T2"]["trials"], e["T2"]["process_h"], e["T2"]["parallel3_h"]) == (1296, 27.3, 9.6)
    assert (e["G3"]["trials"], e["G3"]["process_h"]) == (400, 9.6)
    argv = EV.build_96_argv(types.SimpleNamespace(experiment="S4T2"), "T2", "P3", "R4", "induced:168000:100", 10)
    assert argv[:9] == ["run", "--experiment", "S4T2", "--condition", "P3_R4", "--model", "R4", "--trials", "induced:168000:100"]
    for x in ("--no-safety", "naive", "--induce", "P3", "--max-new", "10"):
        assert x in argv
    assert argv[argv.index("--time-limit-s") + 1] == "60" and argv[argv.index("--exec-interval") + 1] == "6"
    assert "--induce" not in EV.build_96_argv(types.SimpleNamespace(experiment="S4T2"), "T2", "nat", "R4", "natural:165000:66")
    assert EV.block_of("nat", 10) == 30 and EV.block_of("P1", 10) == 10


# ---------------------------------------------------------------- 保存点
def fake_run(root, kind, base, seed, stamp, ok=True, content=b"weights", smoke=False):
    train = root / "outputs" / "s4" / ("train_b4" if kind == "b4" else "train")
    wrap = root / "outputs" / "s4" / ("b4_wrap" if kind == "b4" else "seed_wrap")
    name = f"train_{base}s{seed}{'_smoke' if smoke else ''}_{stamp}_{stamp}"
    ck = train / name / "checkpoints" / "020000" / "pretrained_model"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "model.safetensors").write_bytes(content)
    (ck / "config.json").write_text("{}", encoding="utf-8")
    wrap.mkdir(parents=True, exist_ok=True)
    pc = ({"train_config_ok": ok, "exit_code": 0, "checkpoints_ok": True} if kind == "b4" else
          {"train_config_only_seed_and_names": ok, "same_dataset_fingerprint": True, "exit_code": 0, "checkpoints_ok": True})
    (wrap / f"postcheck_{name}.json").write_text(json.dumps(pc), encoding="utf-8")
    return name


def test_ckpt_mapping_and_sha(tmp_path):
    with pytest.raises(SystemExit, match="後の点検"):
        EV.resolve_ckpt("R4", root=tmp_path)
    fake_run(tmp_path, "b4", "R4", 1000, "20261013-100000", ok=False)            # 後の点検が通らない → 使わない
    fake_run(tmp_path, "b4", "R4", 1000, "20261013-090000", smoke=True)          # smoke → 名前で外れる
    fake_run(tmp_path, "b4", "R4", 1001, "20261014-040000")                      # 種が違う
    with pytest.raises(SystemExit, match="後の点検"):
        EV.resolve_ckpt("R4", root=tmp_path)
    good = fake_run(tmp_path, "b4", "R4", 1000, "20261013-110000")
    r = EV.resolve_ckpt("R4", root=tmp_path)
    assert r["run"] == good and r["path"] == f"outputs/s4/train_b4/{good}/checkpoints/020000/pretrained_model"
    assert r["postcheck"] == f"outputs/s4/b4_wrap/postcheck_{good}.json"
    assert EV.resolve_ckpt("R4s1001", root=tmp_path)["run"].startswith("train_R4s1001_")
    # 通った実行が 2 つ → 止める。--run で選べる
    second = fake_run(tmp_path, "b4", "R4", 1000, "20261013-120000", content=b"other")
    with pytest.raises(SystemExit, match="2 つ以上"):
        EV.resolve_ckpt("R4", root=tmp_path)
    assert EV.resolve_ckpt("R4", choose={"R4": second}, root=tmp_path)["run"] == second
    with pytest.raises(SystemExit, match="--run"):
        EV.resolve_ckpt("R4", choose={"R4": "train_R4s1000_x"}, root=tmp_path)
    # SHA-256: 中身で決まり、パスによらない
    i1 = EV.ckpt_info("R4", choose={"R4": good}, root=tmp_path)
    i2 = EV.ckpt_info("R4", choose={"R4": second}, root=tmp_path)
    assert i1["sha256"] != i2["sha256"] and set(i1["files_sha256"]) == {"model.safetensors", "config.json"}
    third = fake_run(tmp_path, "b4", "N4", 1000, "20261013-130000", content=b"weights")
    assert EV.ckpt_info("N4", root=tmp_path)["sha256"] == i1["sha256"]
    assert EV.ckpt_info("R4", choose={"R4": good}, expect={"R4": i1["sha256"]}, root=tmp_path)["expected_sha256"] == i1["sha256"]
    with pytest.raises(SystemExit, match="掲示した値"):
        EV.ckpt_info("R4", choose={"R4": good}, expect={"R4": "0" * 64}, root=tmp_path)
    assert third
    # 束 3 の R1v3s1001（seed_wrap の後の点検）
    with pytest.raises(SystemExit):
        EV.resolve_ckpt("R1v3s1001", root=tmp_path)
    b3 = fake_run(tmp_path, "b3", "R1v3", 1001, "20261009-020000")
    assert EV.resolve_ckpt("R1v3s1001", root=tmp_path)["path"] == f"outputs/s4/train/{b3}/checkpoints/020000/pretrained_model"


def test_ckpt_r1v3_and_inject():
    v82 = types.SimpleNamespace(CKPT={"R1v3": f"outputs/train/{EV.R1V3_RUN}/checkpoints/020000/pretrained_model"})
    assert EV.resolve_ckpt("R1v3", v82)["run"] == EV.R1V3_RUN
    with pytest.raises(SystemExit, match="違う"):
        EV.resolve_ckpt("R1v3", types.SimpleNamespace(CKPT={"R1v3": "outputs/train/train_R1v3_2099/checkpoints/020000/pretrained_model"}))
    with pytest.raises(SystemExit):
        EV.resolve_ckpt("R1v3", types.SimpleNamespace(CKPT={}))
    EV.inject_ckpt(v82, {"name": "R4", "path": "outputs/s4/train_b4/x/checkpoints/020000/pretrained_model"})
    assert v82.CKPT["R4"].endswith("x/checkpoints/020000/pretrained_model")
    EV.inject_ckpt(v82, {"name": "R4", "path": "outputs/s4/train_b4/x/checkpoints/020000/pretrained_model"})   # 同じなら通す
    with pytest.raises(SystemExit, match="別の保存点"):
        EV.inject_ckpt(v82, {"name": "R4", "path": "outputs/s4/train_b4/y/checkpoints/020000/pretrained_model"})
    assert EV.parse_pairs(["R4=abc"], "--expect") == {"R4": "abc"}
    with pytest.raises(SystemExit):
        EV.parse_pairs(["X9=abc"], "--expect")


def test_patch96_adds_b4_and_spec_keys():
    class Eng:
        def __init__(self, *a):
            pass

        def run_one(self, i, seed, lay, tgt):
            return {"seed": seed}, {}, {}

    r96 = types.SimpleNamespace(Engine=Eng, make_spec=lambda a: {"model": a.model},
                                run_json_text=lambda a, kind, rows, *x, **k: json.dumps({"n": len(rows)}))
    info = {"phase": "T2", "part": "P2", "model": "R4", "ckpt": {"path": "p", "sha256": "s" * 64}}
    EV.patch96(r96, info)
    EV.patch96(r96, dict(info, part="P3"))                     # 包み直しても重ならない
    meta, _, _ = r96.Engine().run_one(0, 167000, None, None)
    assert meta["b4"]["part"] == "P3"
    spec = r96.make_spec(types.SimpleNamespace(model="R4"))
    assert spec["b4_ckpt_sha256"] == "s" * 64 and spec["b4_part"] == "P3" and spec["model"] == "R4"
    assert json.loads(r96.run_json_text(None, "run", [1, 2]))["b4"]["model"] == "R4"


def test_env_mismatch(tmp_path):
    cur = dict(ENV)
    assert EV.env_mismatch(cur, {}) == {}
    assert "driver" in EV.env_mismatch(dict(cur, driver="560.94"), {})
    d1 = write_cond(tmp_path, "T2", "S4T2", "P2_R4", "P2", "R4", rnd_outcome(1))
    d2 = write_cond(tmp_path, "T2", "S4T2", "P2_N4", "P2", "N4", rnd_outcome(2), env=dict(ENV, torch="2.6.0"))
    rec = EV.record_envs([d1, d2, tmp_path / "S4T2" / "none"])
    mm = EV.env_mismatch(cur, rec)
    assert list(mm["records"]) == ["P2_N4"] and rec["none"] == []


# ---------------------------------------------------------------- 二重集計
@pytest.mark.parametrize("phase", ["G2", "G3", "T2"])
def test_a_and_b_agree(tmp_path, phase):
    lay = build_phase(tmp_path, phase)
    for params in ({}, {"guard_mode": "interval", "h2_in_family": False}, {"g3_with_r1v3s1001": False}):
        a, b, diffs = CK.run_check(lay, params)
        assert diffs == [], diffs[:5]
        assert a["status"] == "complete" and a["result"] is not None
    if phase == "G2":
        assert a["result"]["decision"]["next"] in ("test2", "research_guard", "no_test2")
    if phase == "G3":
        assert a["result"]["direction"] is None and a["result"]["direction_used"] is False


def test_gate2_decision_rules(tmp_path):
    # R4 が P2・P1 で高く、自然は同じ → 守りを満たし狙いに届く
    rates = {"P2.R4": 0.9, "P2.R1v3": 0.1, "P1.R4": 0.9, "P1.N4": 0.1, "P1.R1v3": 0.5, "nat.R4": 0.5, "nat.R1v3": 0.5}
    lay = build_phase(tmp_path / "a", "G2", rates)
    a, b, diffs = CK.run_check(lay, {})
    assert not diffs
    r = a["result"]
    assert r["guards"]["p1_r4_minus_n4"]["pass"] and r["aim"]["pass"] and r["aim"]["diff"] >= 0.15
    # 自然は R4・R1v3 で同じ結果なので対の差 0
    assert r["guards"]["natural"]["pairs"] == 198 and r["guards"]["natural"]["diff"] == 0.0 and r["guards"]["natural"]["pass"]
    # R4 − N4 が +20 に届かない → 守りを割る → 研究の道
    lay2 = build_phase(tmp_path / "b", "G2", dict(rates, **{"P1.N4": 0.9}))
    r2 = CK.run_check(lay2, {})[0]["result"]
    assert not r2["guards"]["p1_r4_minus_n4"]["pass"] and r2["decision"]["next"] == "research_guard"
    # 守りは満たすが狙いに届かない
    lay3 = build_phase(tmp_path / "c", "G2", dict(rates, **{"P2.R1v3": 0.9}))
    r3 = CK.run_check(lay3, {})[0]["result"]
    assert r3["guards_pass"] and not r3["aim"]["pass"] and r3["decision"]["next"] == "no_test2"


def test_boundary_exactly_30s(tmp_path):
    """成立がちょうど 30 s の試行は 30 s の分母に入らない（< 30）。成功がちょうど 30 s は成功（<= 30）。"""
    def f_R(s, tgt):
        if s == 167000:
            return True, 30.0, True, 31.0          # 30 s の分母に入らない（60 s には入る）
        if s == 167001:
            return True, 29.99, True, 30.0         # 30 s で成功
        return True, 10.0, False, None

    def f_N(s, tgt):
        if s in (167000, 167001):
            return True, 5.0, False, None
        return True, 10.0, False, None

    root = tmp_path / "v2eval"
    lay = EV.layout_of("T2", root=str(root))
    for name, v in lay["conds"].items():
        part, model = name.split(".", 1)
        f = f_R if (part == "P2" and model == "R4") else (f_N if part == "P2" else rnd_outcome(3))
        write_cond(root, "T2", "S4T2", v["cond"], part, model, f)
    a, b, diffs = CK.run_check(lay, {})
    assert not diffs
    prim = a["result"]["primary"]
    assert prim["H1"]["pairs"] == 99 and prim["H1"]["b"] == 1 and prim["H1"]["c"] == 0
    r30 = a["result"]["secondary"]["rates"]["P2.R4"]["30"]
    r60 = a["result"]["secondary"]["rates"]["P2.R4"]["60"]
    assert (r30["k"], r30["n"]) == (1, 99) and (r60["k"], r60["n"]) == (2, 100)
    six = a["result"]["secondary"]["sixty"]["H1_form"]
    assert six["pairs"] == 100 and six["b"] == 2
    for impl in (A, B):
        r = {"induce": {"established": True, "t_established": 30.0}, "success": True, "t_success": 30.0}
        est = impl.est_before if impl is A else impl.established_before
        suc = impl.succ_by if impl is A else impl.success_within
        assert est(r, 30.0) is False and est(r, 60.0) is True and suc(r, 30.0) is True


def test_holm_and_exact_test():
    assert A.binom_two_sided(0, 0) == 1.0 == B.exact_two_sided(0, 0)
    for b, c in ((10, 2), (3, 3), (0, 7), (15, 4), (40, 22)):
        assert A.binom_two_sided(b, c) == pytest.approx(B.exact_two_sided(b, c), rel=1e-9)
    ps = {"H1": 0.03, "H2": 0.01}
    ha, hb = A.holm(ps), B.holm_b(ps)
    assert ha == hb
    assert ha["order"] == ["H2", "H1"] and ha["by"]["H2"]["p_holm"] == pytest.approx(0.02) and ha["by"]["H1"]["p_holm"] == pytest.approx(0.03)
    assert ha["by"]["H2"]["rejected_stepdown"] and ha["by"]["H1"]["rejected_stepdown"]
    h = A.holm({"H1": 0.04, "H2": 0.03})
    assert h["by"]["H2"]["p_holm"] == pytest.approx(0.06) and h["by"]["H1"]["p_holm"] == pytest.approx(0.06)
    assert not h["by"]["H2"]["rejected_stepdown"] and not h["by"]["H1"]["rejected_stepdown"]
    assert A.holm({"H1": 0.6})["by"]["H1"]["p_holm"] == 0.6


def test_p3_is_not_in_family(tmp_path):
    lay = build_phase(tmp_path / "a", "T2")
    a1 = CK.run_check(lay, {})[0]["result"]
    assert a1["holm"]["order"] == sorted(a1["holm"]["order"], key=lambda k: (a1["primary"][k]["p"], k))
    assert set(a1["holm"]["order"]) == {"H1", "H2"} and a1["holm"]["m"] == 2
    # 置き損ねの結果を極端に変えても、H1・H2 の p・Holm は変わらない
    lay2 = build_phase(tmp_path / "b", "T2", {"P3.R4": 1.0, "P3.R1v3": 0.0, "P3.N4": 0.0})
    a2, b2, diffs = CK.run_check(lay2, {})
    assert not diffs
    for k in ("H1", "H2"):
        assert a2["result"]["primary"][k] == a1["primary"][k]
    assert a2["result"]["holm"] == a1["holm"]
    p3 = a2["result"]["secondary"]["p3"]["30"]["R4_vs_R1v3"]
    assert p3["b"] > 10 and p3["c"] == 0 and p3["p"] < 0.001
    assert "p_holm" not in p3 and "族の外" in a2["result"]["secondary"]["p3"]["label"]
    # H2 を族から外すと H1 だけの Holm（m = 1）
    a3 = CK.run_check(lay2, {"h2_in_family": False})[0]["result"]
    assert a3["holm"]["m"] == 1 and a3["primary"]["H2"]["p_holm"] is None and a3["primary"]["H2"]["in_family"] is False
    assert a3["primary"]["H1"]["p_holm"] == a3["primary"]["H1"]["p"]


def test_incomplete_is_not_judged(tmp_path):
    # 試行が 1 本欠けている
    root = tmp_path / "v2eval"
    lay = EV.layout_of("G3", root=str(root))
    for name, v in lay["conds"].items():
        part, model = name.split(".", 1)
        write_cond(root, "G3", "S4B4G3", v["cond"], part, model, rnd_outcome(5), drop=(192450,) if name == "P1.R4s1001" else ())
    a, b, diffs = CK.run_check(lay, {})
    assert not diffs and a["status"] == b["status"] == "incomplete" and a["result"] is None
    assert not a["checks"]["P1.R4s1001"]["ok"] and a["checks"]["P1.N4s1001"]["ok"]
    # R1v3s1001 を使わないなら、無くても足りる
    lay2 = dict(lay, conds={k: v for k, v in lay["conds"].items() if k != "P2.R1v3s1001"})
    a2 = A.analyze(lay2, {"g3_with_r1v3s1001": True})
    assert a2["missing_conditions"] == ["P2.R1v3s1001"] and a2["status"] == "incomplete"


@pytest.mark.parametrize("bad", ["audit", "sha", "env", "limit", "posted"])
def test_incomplete_variants(tmp_path, bad):
    root = tmp_path / "v2eval"
    lay = EV.layout_of("G3", root=str(root))
    for name, v in lay["conds"].items():
        part, model = name.split(".", 1)
        kw = {}
        if name == "P2.R4s1001":
            kw = {"audit": dict(audit=False), "sha": dict(sha="9" * 64), "env": dict(env=dict(ENV, driver="560.94")),
                  "limit": dict(limit=30.0), "posted": {}}[bad]
        write_cond(root, "G3", "S4B4G3", v["cond"], part, model, rnd_outcome(6), **kw)
    params = {"ckpt_sha256": {"N4s1001": "e" * 64}} if bad == "posted" else {}
    a, b, diffs = CK.run_check(lay, params)
    assert not diffs and a["status"] == "incomplete" and b["status"] == "incomplete" and a["result"] is None
    if bad == "sha":                        # 同じモデル（R4s1001）の P1 と P2 で保存点が違う
        assert a["checks"]["P2.R4s1001"]["ok"] and a["cross"]["models"]["R4s1001"] is None and not a["cross"]["ok"]
    if bad == "env":                        # 条件の中は 1 種類、条件をまたぐと違う
        assert a["checks"]["P2.R4s1001"]["ok"] and not a["cross"]["ok"]
    if bad in ("audit", "limit"):
        assert not a["checks"]["P2.R4s1001"]["ok"]
    if bad == "posted":
        assert not a["checks"]["P1.N4s1001"]["ok"]


def test_check_command_writes_only_when_agreeing(tmp_path, monkeypatch):
    lay = build_phase(tmp_path, "T2")
    lp, pp = tmp_path / "layout.json", tmp_path / "params.json"
    lp.write_text(json.dumps(lay), encoding="utf-8")
    pp.write_text(json.dumps({"guard_mode": "point", "ckpt_sha256": {"_note": "x"}, "_x": 1}), encoding="utf-8")
    led = tmp_path / "ledger.json"
    time.sleep(0.01)
    led.write_text(json.dumps({"problems": {"unreadable_records": []}}), encoding="utf-8")
    os.utime(led, (time.time() + 5, time.time() + 5))
    out = tmp_path / "res.json"
    rc = CK.main(["check", "--layout", str(lp), "--params", str(pp), "--out", str(out), "--ledger-json", str(led), "--no-git"])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["double_count"]["agree"] and res["entry_audit"]["versions"]["ok"] and res["status"] == "complete"
    assert "H1" in out.with_suffix(".md").read_text(encoding="utf-8")
    # A と B が食い違えば書かない（B の数え方を 1 つずらす）
    real = B.paired_induced

    def off(fx, fy, lim):
        r = real(fx, fy, lim)
        r["b"] += 1
        return r
    orig_load = CK._load

    def load_patched(path, name):
        m = orig_load(path, name)
        if name == "s4_b4_b":
            m.paired_induced = off
        return m
    monkeypatch.setattr(CK, "_load", load_patched)
    out2 = tmp_path / "res2.json"
    rc = CK.main(["check", "--layout", str(lp), "--params", str(pp), "--out", str(out2), "--ledger-json", str(led), "--no-git"])
    assert rc == 1 and not out2.exists()
    # 台帳の照合の結果が記録より古ければ未完
    os.utime(led, (1, 1))
    monkeypatch.setattr(CK, "_load", orig_load)
    rc = CK.main(["check", "--layout", str(lp), "--params", str(pp), "--out", str(out2), "--ledger-json", str(led), "--no-git"])
    assert rc == 3 and json.loads(out2.read_text(encoding="utf-8"))["result"] is None


def test_heads_check_without_git():
    assert CK.check_heads(["a"])["ok"]
    assert not CK.check_heads(["a", "b"], use_git=False)["ok"]
    assert not CK.check_heads(["None"])["ok"]


def test_params_validation():
    with pytest.raises(ValueError):
        A.params_with_defaults({"guard_mode": "x"})
    with pytest.raises(ValueError):
        A.params_with_defaults({"unknown": 1})
    with pytest.raises(ValueError):
        B.fill({"h2_in_family": 1})
    with pytest.raises(ValueError):
        A.analyze({"phase": "X", "root": ".", "experiment": "E", "conds": {}}, {})


# ---------------------------------------------------------------- 点検の指摘・作者の判断（10/09）の直し
def test_files_sha_must_be_one_kind_across_conditions(tmp_path):
    """条件の中では 1 種類でも、条件をまたいで入口のスクリプトの版が違えば未完（A・B とも）。"""
    other = dict(FILES, **{"scripts/98_s4_b4_eval.py": "e" * 64})
    lay = build_phase(tmp_path, "G3", per={"P2.R4s1001": {"files": other}})
    a, b, diffs = CK.run_check(lay, {})
    assert not diffs and a["status"] == b["status"] == "incomplete" and a["result"] is None
    assert a["checks"]["P2.R4s1001"]["ok"] and a["checks"]["P1.R4s1001"]["ok"]
    assert not a["cross"]["ok"] and a["cross"]["files_sha256"] is None and b["cross"]["files_sha256"] is None
    # そろっていれば cross に 1 つの組が残る
    lay2 = build_phase(tmp_path / "ok", "G3")
    a2, b2, d2 = CK.run_check(lay2, {})
    assert not d2 and a2["cross"]["files_sha256"] == FILES == b2["cross"]["files_sha256"]
    # 試行に files_sha256 が無い → 条件の点検で未完
    lay3 = build_phase(tmp_path / "none", "G3", per={"P1.N4s1001": {"files": {}}})
    a3, b3, d3 = CK.run_check(lay3, {})
    assert not d3 and not a3["checks"]["P1.N4s1001"]["ok"] and not b3["checks"]["P1.N4s1001"]["ok"]


def test_gates_sha_must_match_posted(tmp_path):
    assert A.POSTED_GATES_SHA256 == B.GATES_POSTED == CK.POSTED_GATES_SHA256
    assert A.POSTED_GATES_SHA256.startswith("7f2f651c")
    # 記録の s4_gates.json の値が掲示と違う → A・B とも未完
    bad = dict(FILES, **{"configs/s4_gates.json": "d" * 64})
    lay = build_phase(tmp_path, "G3", files=bad)
    a, b, diffs = CK.run_check(lay, {})
    assert not diffs and a["status"] == b["status"] == "incomplete"
    assert all(not c["ok"] for c in a["checks"].values())
    # 入口: 今のファイルは掲示の値と一致する。違うファイル・記録が無ければ満たさない
    good = {"cross": {"files_sha256": dict(FILES)}}
    assert CK.check_gates_sha(good)["ok"]
    assert not CK.check_gates_sha({"cross": {"files_sha256": None}})["ok"]
    g2 = tmp_path / "g.json"
    g2.write_bytes(CK.GATES.read_bytes() + b" ")
    r = CK.check_gates_sha(good, g2)
    assert not r["ok"] and "今の" in r["note"]
    # CRLF にしても LF にそろえて照らす（gate1.py と同じ）
    g3 = tmp_path / "g3.json"
    g3.write_bytes(CK.GATES.read_bytes().replace(b"\n", b"\r\n"))
    assert CK.check_gates_sha(good, g3)["ok"]


def test_heads_check_adds_b4_child_without_editing_audit(monkeypatch):
    seen = {}

    class FakeAudit:
        FAMILY_SCRIPT = {"E7": "scripts/98_s4_d_e7.py"}

        def order_heads(self, root, heads):
            return list(heads)

        def compare_heads(self, root, order, fam):
            seen["fam"], seen["script"] = fam, self.FAMILY_SCRIPT.get(fam)
            return {"changed": [], "added_only": []}

    monkeypatch.setattr(CK, "_load", lambda path, name: FakeAudit())
    assert CK.check_heads(["a", "b"])["ok"]
    assert seen == {"fam": "B4", "script": "scripts/98_s4_b4_eval.py"}
    # 98_s4_d_audit.py そのものは変えていない（既存の家族の動きは同じ）
    au = _load("98_s4_d_audit.py", "t_b4_audit_real")
    assert "B4" not in au.FAMILY_SCRIPT
    assert set(au.FAMILY_SCRIPT) == {"RTC", "E7", "ST", "XPL", "RC", "X2"}


def test_paired_guards_count_same_seed_pairs(tmp_path):
    """守りの P1 は同じ種で両方とも 30 s より前に成立した対で数える（作者の判断 10/09）。"""
    def f_R(s, tgt):                                       # R4: 種 166000〜166049 だけ成立、全部成功
        return (s < 166050), (5.0 if s < 166050 else None), True, 10.0

    def f_N(s, tgt):                                       # N4: 全部成立、偶数の種だけ成功
        return True, 5.0, s % 2 == 0, (10.0 if s % 2 == 0 else None)

    root = tmp_path / "v2eval"
    lay = EV.layout_of("T2", root=str(root))
    for name, v in lay["conds"].items():
        part, model = name.split(".", 1)
        f = (f_R if model == "R4" else f_N) if part == "P1" else rnd_outcome(9)
        write_cond(root, "T2", "S4T2", v["cond"], part, model, f)
    a, b, diffs = CK.run_check(lay, {})
    assert not diffs and a["status"] == "complete"
    for g in (a["result"]["secondary"]["guards"], b["result"]["secondary"]["guards"]):
        rn = g["p1_r4_minus_n4"]
        assert (rn["pairs"], rn["b"], rn["c"], rn["both"], rn["neither"]) == (50, 25, 0, 25, 0)
        assert rn["diff"] == pytest.approx(0.5)            # 各腕の割合の差なら 1.0 − 0.5 = 0.5 だが、分母は対の 50
        assert rn["x_k"] == 50 and rn["y_k"] == 25 and "x" not in rn
    from recovla.eval import stats as S
    d, lo, hi = S.paired_diff_ci(25, 25, 0, 0)
    assert a["result"]["secondary"]["guards"]["p1_r4_minus_n4"]["newcombe95"] == pytest.approx([lo, hi], rel=1e-12)
    assert b["result"]["secondary"]["guards"]["p1_r4_minus_n4"]["newcombe95"] == pytest.approx([lo, hi], rel=1e-9)
    md = A.summary_md(dict(a, entry_audit={"problems": []}))
    assert "対 50、b 25、c 0" in md
    # 関門 2・関門 3 も同じ形（対の数を持つ）
    lay2 = build_phase(tmp_path / "g2", "G2")
    r2 = CK.run_check(lay2, {})[0]["result"]
    assert "pairs" in r2["guards"]["p1_r4_vs_r1v3"] and "pairs" in r2["guards"]["p1_r4_minus_n4"]
    lay3 = build_phase(tmp_path / "g3", "G3")
    r3 = CK.run_check(lay3, {})[0]["result"]
    assert "pairs" in r3["p1_r4_minus_n4"] and "x" in r3["direction"]      # 向き（落下）は各腕の分母のまま


def test_natural_pairs_disagree_newcombe_matches(tmp_path):
    """自然の 2 腕の結果が食い違うとき、対ありの Newcombe の区間が A・B・recovla.eval.stats で一致する。"""
    from recovla.eval import stats as S

    def f_x(s, tgt):
        ok = (s + len(tgt)) % 3 != 0
        return False, None, ok, (20.0 if ok else None)

    def f_y(s, tgt):
        ok = (s * 7 + len(tgt)) % 4 == 0
        return False, None, ok, (25.0 if ok else None)

    root = tmp_path / "v2eval"
    lay = EV.layout_of("G2", root=str(root))
    for name, v in lay["conds"].items():
        part, model = name.split(".", 1)
        f = (f_x if model == "R4" else f_y) if part == "nat" else rnd_outcome(11)
        write_cond(root, "G2", "S4B4G2", v["cond"], part, model, f)
    a, b, diffs = CK.run_check(lay, {})
    assert not diffs and a["status"] == "complete"
    na, nb = a["result"]["guards"]["natural"], b["result"]["guards"]["natural"]
    assert na["pairs"] == 198 and na["x_only"] > 0 and na["y_only"] > 0
    n11 = na["x_k"] - na["x_only"]
    n00 = 198 - n11 - na["x_only"] - na["y_only"]
    d, lo, hi = S.paired_diff_ci(n11, na["x_only"], na["y_only"], n00)
    assert na["newcombe95"] == pytest.approx([lo, hi], rel=1e-12)
    assert nb["newcombe95"] == pytest.approx(na["newcombe95"], rel=1e-9)
    assert na["diff"] == pytest.approx(d) and lo < d < hi


def test_g2_p3_missing_or_bad_is_not_incomplete(tmp_path):
    """関門 2 の置き損ね（P3）は記述だけ。欠けても・点検を満たさなくても未完にしない（作者の判断 10/09）。"""
    lay = build_phase(tmp_path / "a", "G2", skip=("P3.N4",))
    a, b, diffs = CK.run_check(lay, {})
    assert not diffs and a["status"] == b["status"] == "complete"
    assert a["missing_conditions"] == [] and a["optional"] == b["optional"]
    assert a["optional"]["missing"] == ["P3.N4"] and a["optional"]["used"] == ["P3.R4", "P3.R1v3"]
    p3 = a["result"]["p3"]
    assert p3["used"] == ["R4", "R1v3"] and p3["not_used"] == ["N4"] and set(p3["rates"]) == {"R4", "R1v3"}
    md = A.summary_md(dict(a, entry_audit={"problems": []}))
    assert "P3.N4" in md and "置き損ね（P3。記述だけ" in md and "30 s" in md
    # P3 の 1 条件が点検を満たさない（G_AUDIT が無い）、もう 1 つは保存点が判定の条件と違う → 使わない。未完にしない
    lay2 = build_phase(tmp_path / "b", "G2", per={"P3.R1v3": {"audit": False}, "P3.R4": {"sha": "9" * 64}})
    a2, b2, d2 = CK.run_check(lay2, {})
    assert not d2 and a2["status"] == "complete" and a2["optional"]["excluded"] == ["P3.R4", "P3.R1v3"]
    assert a2["result"]["p3"]["used"] == ["N4"]
    # 入口の HEAD の照合は使わなかった条件を入れない
    a2["checks"]["P3.R4"]["git_heads"] = ["zzz"]
    ea = CK.cross_audit(a2, lay2, ledger_json=str(tmp_path / "no_ledger.json"), use_git=False)["entry_audit"]
    assert ea["versions"]["ok"]
    # 判定に使う条件（P1）が欠ければ未完のまま
    lay3 = build_phase(tmp_path / "c", "G2", skip=("P1.N4",))
    a3 = CK.run_check(lay3, {})[0]
    assert a3["status"] == "incomplete" and not a3["checks"]["P1.N4"]["ok"] and a3["result"] is None


def test_driver_other_than_61088_warns_but_does_not_stop(tmp_path):
    lay = build_phase(tmp_path, "G3", env=dict(ENV, driver="560.94"))
    a, b, diffs = CK.run_check(lay, {})
    assert not diffs and a["status"] == "complete"
    assert a["cross"]["driver"] == "560.94" and a["cross"]["driver_warning"] is True and b["cross"]["driver_warning"] is True
    assert "警告" in A.summary_md(dict(a, entry_audit={"problems": []}))
    lay2 = build_phase(tmp_path / "ok", "G3")
    a2 = CK.run_check(lay2, {})[0]
    assert a2["cross"]["driver"] == "610.88" and a2["cross"]["driver_warning"] is False
    assert "警告" not in A.summary_md(dict(a2, entry_audit={"problems": []}))
    # 入口: ドライバだけ違う → 止めない。記録と今の環境が違う → 止める（96 の --accept-env-change で進む）
    out = tmp_path / "v2"
    write_cond(out, "T2", "S4T2", "P2_R4", "P2", "R4", rnd_outcome(1), env=dict(ENV, driver="560.94"))
    r96 = types.SimpleNamespace(read_env=lambda ops: dict(ENV, driver="560.94"), env_brief=lambda c: "env")
    v82 = types.SimpleNamespace(OUT=out)
    ns = lambda extra: types.SimpleNamespace(experiment="S4T2", extra=extra)        # noqa: E731
    assert EV.phase_env_gate(r96, None, v82, ns([]), "T2", ["P2_R4"]) == 0
    r96b = types.SimpleNamespace(read_env=lambda ops: dict(ENV), env_brief=lambda c: "env")
    assert EV.phase_env_gate(r96b, None, v82, ns([]), "T2", ["P2_R4"]) == 3
    assert EV.phase_env_gate(r96b, None, v82, ns(["--accept-env-change"]), "T2", ["P2_R4"]) == 0


def test_forbidden_extra_rejects_abbreviations():
    for f in ("--induc", "--induce=P1", "--time-limit", "--time-limit-s=30", "--mod", "--trial", "--no-saf", "--exec", "--diag"):
        assert EV.forbidden_flag(f, EV.FORBIDDEN_EXTRA), f
    # 96 の run の通してよい引数は拒まない
    for f in ("--dry-run", "--min-free-gb", "--min-free-gb=8", "--accept-env-change", "--accept-spec-change", "--max-new",
              "--mem-timeout-min", "--min-commit-free-gb", "--ignore-quiet", "--quiet-window", "--allow-82-change",
              "--stop-file", "--progress-file", "30", "-x"):
        assert not EV.forbidden_flag(f, EV.FORBIDDEN_EXTRA), f
    rot = ("--max-new", "--progress-file", "--stop-file") + EV.FORBIDDEN_EXTRA
    assert EV.forbidden_flag("--max", rot) and EV.forbidden_flag("--progress", rot)
    with pytest.raises(SystemExit, match="決める"):
        EV.run_condition(None, None, None, types.SimpleNamespace(), "T2", "P2_R4", ["--induc", "P3"])
    assert EV.has_flag(["--accept-env"], "--accept-env-change", len("--accept-e"))
    assert not EV.has_flag(["--accept-"], "--accept-env-change", len("--accept-e"))


def test_summary_md_t2_order_and_sixty(tmp_path):
    lay = build_phase(tmp_path, "T2")
    a = CK.run_check(lay, {})[0]
    md = A.summary_md(dict(a, entry_audit={"problems": []}))
    keys = ["| H1 |", "| H2 |", "## 守り", "## 置き損ね", "## 60 s の採点", "## 曲線の材料"]
    pos = [md.index(k) for k in keys]
    assert pos == sorted(pos), pos
    assert "H1 の形（R4 対 R1v3）" in md and "60 s R4 対 N4" in md and "| P2.R4 |" in md
    lay2 = build_phase(tmp_path / "g2", "G2")
    md2 = A.summary_md(dict(CK.run_check(lay2, {})[0], entry_audit={"problems": []}))
    assert md2.index("## 守り") < md2.index("## 狙い") < md2.index("## 置き損ね") < md2.index("## 結論")
    assert "- R1v3: 30 s" in md2 and "60 s" in md2


def test_rotate_tag_splits_progress_files():
    ns = lambda **k: types.SimpleNamespace(**dict({"experiment": "S4T2", "parts": None, "models": None}, **k))   # noqa: E731
    assert EV.rotate_tag(ns()) == "S4T2"
    assert EV.rotate_tag(ns(parts="P1,nat")) == "S4T2_P1-nat" != EV.rotate_tag(ns(parts="P2"))
    assert EV.rotate_tag(ns(parts="P3", models="R4,N4")) == "S4T2_P3_N4-R4"
