"""段階 4 束 4（滑りのデータ）の道具の検査: scripts/97_s4_b4_data.py・97_s4_b4_manifest.py・97_s4_b4_gateD.py・99_s4_train_b4.py。
CPU だけ。シミュレーション・描画・学習・GPU は使わない（偽のデータ・偽の生成器・偽の設定）。

使い方: PYTHONPATH=src python -m pytest -q tests/test_s4_bundle4.py -p no:cacheprovider
"""
import hashlib
import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


D = _load("s4_b4_data_t", "scripts/97_s4_b4_data.py")
M = _load("s4_b4_manifest_t", "scripts/97_s4_b4_manifest.py")
G = _load("s4_b4_gateD_t", "scripts/97_s4_b4_gateD.py")
T = _load("s4_b4_train_t", "scripts/99_s4_train_b4.py")
F30 = D.load_f30()


def fake_layout(seed, lk):
    cols = ("red", "green", "blue") if lk == "empty" else (("green", "blue") if lk == "prefilled_1" else ("blue",))
    return types.SimpleNamespace(table_colors=cols, start="home" if seed % 2 else "retreat")


# ---------------------------------------------------------------- 指定と帯
def test_plan_counts_seeds_and_same_rules_as_stage3():
    rec = D.build_plan(D.NEED, D.SEED_BASE, F30, fake_layout)
    assert {lk: c["need"] for lk, c in rec["B"].items()} == F30.split_counts(30) == {"empty": 22, "prefilled_1": 4, "prefilled_2": 4}
    assert {lk: c["need"] for lk, c in rec["C"].items()} == F30.split_counts(20) == {"empty": 15, "prefilled_1": 3, "prefilled_2": 2}
    b = [c["seed"] for cl in rec["B"].values() for c in cl["candidates"]]
    c = [c["seed"] for cl in rec["C"].values() for c in cl["candidates"]]
    assert b == list(range(44600, 44660)) and c == list(range(44700, 44740))          # 候補は必要数の 2 倍（30_f.py の CANDIDATE_FACTOR）
    cand = rec["B"]["empty"]["candidates"]
    assert [x["color"] for x in cand[:4]] == ["red", "green", "blue", "red"]            # 机上の色を順に回す（30_f.py と同じ）
    chunks = D.chunks_of(rec)
    assert len(chunks) == 6 and sum(len(s) for _, s in chunks) == 200                  # 復帰の候補 100 ＋ 同じ（種、色）の通常 100
    name, specs = chunks[0]
    assert specs[0][0] == "B" and specs[len(specs) // 2][0] == "n" and specs[0][1:] == specs[len(specs) // 2][1:]
    assert "A" not in rec                                                              # 掴み損ねは足さない


LEDGER = """
| 44400 | 44799 | 予定 | 段階4 | あり | 段階 4 の smoke・データ |
| 44404 | 44423 | 使用済み | 段階4 | あり | X2 |
| 44450 | 44455 | 使用済み | 段階4 | あり | smoke |
| 44610 | 44610 | 使用済み | 段階4 | あり | 誰かが使った（試験用の行） |
"""


def test_band_rejection():
    assert D.ledger_used(LEDGER) == [(44404, 44423), (44450, 44455), (44610, 44610)]
    assert D.band_problems(range(44700, 44740), LEDGER) == []
    assert D.band_problems([44610], LEDGER)                                            # 台帳で使用済み
    assert D.band_problems([44410], LEDGER)                                            # X2
    assert D.band_problems([44505], LEDGER)                                            # 掲示板 0164 の smoke の予約
    assert D.band_problems([44399], LEDGER) and D.band_problems([44800], LEDGER)       # 学習用の帯の外
    # 実際の台帳でも、決めた帯（候補・予備・smoke）は重ならない
    seeds = list(range(44600, 44684)) + list(range(44700, 44780))
    assert D.band_problems(seeds) == []


# ---------------------------------------------------------------- 採る（30_f.py と同じ規則）と本数の確かめ
def _results(rec, fail=()):
    by = {}
    for kind, cells in rec.items():
        for cell in cells.values():
            for c in cell["candidates"]:
                for k in (kind, "n"):
                    ok = (k, c["seed"], c["color"]) not in fail
                    by[(k, c["seed"], c["color"])] = {"kind": k, "layout_seed": c["seed"], "color": c["color"], "success": ok,
                                                       "attempts": [{"name": f"{k}_{c['seed']}_{c['color']}_r0"}], "_run": "outputs/gen/x"}
    return by


def test_choose_takes_both_success_in_order_and_counts():
    rec = D.build_plan(D.NEED, D.SEED_BASE, F30, fake_layout)
    first = rec["B"]["empty"]["candidates"][0]
    res = D.choose(rec, _results(rec, fail={("n", first["seed"], first["color"])}), F30.DROPPED_STOP)
    assert res["count_ok"] and res["by_kind"]["B"]["got"] == 30 and res["by_kind"]["C"]["got"] == 20
    assert first["seed"] not in [c["seed"] for c in res["chosen"]]                      # 相手の通常が失敗したら両方から外す
    assert res["by_kind"]["B"]["dropped"] == 1 and not res["stop_drop_rule"]
    for c in res["chosen"]:
        assert c["recovery"]["key"].split("_")[1:3] == c["twin"]["key"].split("_")[1:3]
    # 捨てた割合が 10% を超えたら止める・足りない枠を出す
    bad = {("B", c["seed"], c["color"]) for c in rec["B"]["prefilled_1"]["candidates"][:6]}
    res = D.choose(rec, _results(rec, fail=bad), F30.DROPPED_STOP)
    assert not res["count_ok"] and res["short_cells"][0]["layout_kind"] == "prefilled_1" and res["stop_drop_rule"]


# ---------------------------------------------------------------- verify と捨てた割合の例外（--accept-drop-rule）
def _verify_setup(tmp_path, monkeypatch, fails, need=5, n_cand=8):
    """B の枠 1 つ（必要 need・候補 n_cand）。fails = {(種類, 種, 色): 失敗の名前}。成功はエピソードのファイルも作る。"""
    monkeypatch.setattr(D, "B4", tmp_path)
    monkeypatch.setattr(D.config, "path", lambda p: pathlib.Path(p))
    monkeypatch.setattr(D, "stage3_identity", lambda: {"ok": True})
    cands = [{"seed": 44600 + i, "color": "red", "layout_kind": "empty", "start": "home"} for i in range(n_cand)]
    rec = {"B": {"empty": {"need": need, "candidates": cands}}}
    run = tmp_path / "gen" / "S4B4_B_empty"
    run.mkdir(parents=True)
    rows = []
    for c in cands:
        for k in ("B", "n"):
            f = fails.get((k, c["seed"], c["color"]))
            if f:
                land = f == D.LANDING_FAILURE
                att = [{"name": f"{k}_{c['seed']}_red_r{i}", "failure": f,
                        "inject": {"status": "landing_invalid" if land else "confirmed", "reason": "clearance" if land else None}}
                       for i in range(4)]
            else:
                att = [{"name": f"{k}_{c['seed']}_red_r0", "failure": None, "inject": {"status": "confirmed", "reason": None}}]
                d = run / att[0]["name"]
                d.mkdir()
                (d / "meta.json").write_text("{}", encoding="utf-8")
                (d / "data.npz").write_bytes(b"x")
            rows.append({"kind": k, "layout_seed": c["seed"], "color": c["color"], "layout_kind": "empty", "success": not f,
                         "attempts": att})
    (run / "generation.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (tmp_path / "plan.json").write_text(json.dumps({"need": {"B": need}, "recovery": rec}), encoding="utf-8")
    (tmp_path / "gen_state.json").write_text(json.dumps({"chunks": {"B_empty": {"run": str(run), "complete": True}}}),
                                             encoding="utf-8")
    return lambda: json.loads((tmp_path / "data.json").read_text(encoding="utf-8"))


LAND = D.LANDING_FAILURE


def test_verify_without_flag_is_unchanged(tmp_path, monkeypatch):
    read = _verify_setup(tmp_path, monkeypatch, {("B", 44600, "red"): LAND})       # 捨てた 1/6 = 16.7% > 10%
    assert D.main(["verify"]) == 1
    d = read()
    assert d["stop_drop_rule"] and d["count_ok"] and not d["ok"]
    assert not {"drop_rule_exception", "ok_reason", "dropped_candidates"} & set(d)   # 引数なしは今までと同じ中身
    read = _verify_setup(tmp_path / "b", monkeypatch, {})
    assert D.main(["verify"]) == 0 and read()["ok"]


def test_verify_accept_drop_rule_applies_when_all_landing(tmp_path, monkeypatch):
    read = _verify_setup(tmp_path, monkeypatch, {("B", 44600, "red"): LAND, ("B", 44602, "red"): LAND})
    assert D.main(["verify", "--accept-drop-rule", "0169"]) == 0
    d = read()
    assert d["ok"] and d["stop_drop_rule"] and "例外（掲示 0169）" in d["ok_reason"]          # 隠さない
    ex = d["drop_rule_exception"]
    assert ex["applied"] and ex["board"] == "0169" and not ex["problems"] and ex["written"]
    assert ex["by_kind"]["B"]["dropped"] == 2 and ex["by_kind"]["B"]["examined"] == 7 and ex["by_kind"]["B"]["over_limit"]
    assert [(c["seed"], c["color"], c["layout_kind"]) for c in ex["dropped_candidates"]] == [(44600, "red", "empty"), (44602, "red", "empty")]
    c = ex["dropped_candidates"][0]
    assert c["failure"] == LAND and c["inject_reason"] == "clearance" and c["attempts"] == 4 and c["twin_success"]
    # 上限以下なら例外は使わない（記録もしない）
    read = _verify_setup(tmp_path / "b", monkeypatch, {})
    assert D.main(["verify", "--accept-drop-rule", "0169"]) == 0
    d = read()
    assert d["ok"] and "drop_rule_exception" not in d and d["ok_reason"].startswith("決まりどおり")


@pytest.mark.parametrize("fails,need,why", [
    ({("B", 44600, "red"): LAND, ("B", 44601, "red"): "script_place_failed"}, 5, "landing でない"),    # 台本の失敗が混じる
    ({("n", 44600, "red"): "script_place_failed"}, 5, "相手の通常デモが失敗"),                         # 相手が失敗
    ({("B", 44600 + i, "red"): LAND for i in range(4)}, 5, "本数がそろわない"),                           # 本数不足
])
def test_verify_accept_drop_rule_refuses(tmp_path, monkeypatch, fails, need, why):
    read = _verify_setup(tmp_path, monkeypatch, fails, need=need)
    assert D.main(["verify", "--accept-drop-rule", "0169"]) == 1
    d = read()
    assert not d["ok"] and d["stop_drop_rule"] and not d["drop_rule_exception"]["applied"]
    assert any(why in p for p in d["drop_rule_exception"]["problems"]) and d["ok_reason"].startswith("例外は効かない")


def test_verify_out_and_board_number(tmp_path, monkeypatch):
    _verify_setup(tmp_path, monkeypatch, {("B", 44600, "red"): LAND})
    out = tmp_path / "tmp" / "check.json"
    assert D.main(["verify", "--accept-drop-rule", "0169", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] and not (tmp_path / "data.json").exists()   # 書き先だけに書く
    for bad in ("169", "abcd", "01690"):
        with pytest.raises(SystemExit) as e:
            D.main(["verify", "--accept-drop-rule", bad])
        assert e.value.code == 2


def test_manifest_build_reads_ok_with_exception(tmp_path, monkeypatch):
    r1, n1 = _base_manifests()
    monkeypatch.setattr(M, "_sources", lambda: ({}, {"R1": {"sha256": M.EXPECTED_SHA["R1"], "manifest": r1, "path": tmp_path / "r1"},
                                                    "N1": {"sha256": M.EXPECTED_SHA["N1"], "manifest": n1, "path": tmp_path / "n1"}}))
    monkeypatch.setattr(M, "B4", tmp_path)
    data = {"ok": True, "stop_drop_rule": True, "ok_reason": "例外（掲示 0169）", "drop_rule_exception": {"applied": True},
            "chosen": _chosen()}
    (tmp_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
    assert M.main(["build", "--dry-run"]) == 0
    (tmp_path / "data.json").write_text(json.dumps(dict(data, ok=False)), encoding="utf-8")
    assert M.main(["build", "--dry-run"]) == 1


def test_run_chunks_resumes(tmp_path):
    calls = []

    def fake_generate(specs, run, workers, render, rig_kind):
        assert render and rig_kind == "v3"
        calls.append(run.name)
        run.mkdir(parents=True)
        (run / "generation.jsonl").write_text("\n".join(json.dumps({"i": i}) for i in range(len(specs))) + "\n", encoding="utf-8")
        (run / "timing.json").write_text("{}", encoding="utf-8")
        return [{"success": True} for _ in specs]

    spec_cls = lambda s, c, lk, k: (s, c, lk, k)                                       # noqa: E731
    chunks = [("B_empty", [("B", 1, "red", "empty"), ("n", 1, "red", "empty")]), ("C_empty", [("C", 2, "red", "empty")])]
    st = tmp_path / "state.json"
    plan = D.run_chunks(chunks, st, tmp_path / "gen", fake_generate, spec_cls, 1, dry_run=True)
    assert plan["run"] == ["B_empty", "C_empty"] and not calls and not st.exists()     # dry-run は何もしない
    D.run_chunks(chunks, st, tmp_path / "gen", fake_generate, spec_cls, 1, dry_run=False)
    assert len(calls) == 2
    s = json.loads(st.read_text(encoding="utf-8"))
    run_c = pathlib.Path(s["chunks"]["C_empty"]["run"])
    (run_c / "timing.json").unlink()                                                   # 書きかけにする
    plan = D.run_chunks(chunks, st, tmp_path / "gen", fake_generate, spec_cls, 1, dry_run=False)
    assert plan["skip"] == ["B_empty"] and plan["run"] == ["C_empty"] and len(calls) == 3
    assert any(p.name.startswith(run_c.name + "_incomplete_") for p in run_c.parent.iterdir())   # 退避した


# ---------------------------------------------------------------- マニフェスト（本数・R と N の対称・SHA）
def _base_manifests():
    normal = [{"run": "outputs/gen/F", "key": f"n_{20000 + i}_red_r0"} for i in range(240)]
    rec = ([{"run": "outputs/gen/F", "key": f"A_{30000 + i}_red_r0"} for i in range(40)]
           + [{"run": "outputs/gen/F", "key": f"B_{31000 + i}_red_r0"} for i in range(30)]
           + [{"run": "outputs/gen/F", "key": f"C_{32000 + i}_red_r0"} for i in range(20)])
    twins = [{"run": "outputs/gen/F", "key": "n_" + e["key"].split("_", 1)[1]} for e in rec]
    return {"entries": normal + rec}, {"entries": normal + twins}


def _chosen():
    out = []
    for kind, n, base in (("B", 30, 44600), ("C", 20, 44700)):
        for i in range(n):
            out.append({"kind": kind, "layout_kind": "empty", "seed": base + i, "color": "green",
                        "recovery": {"run": "outputs/gen/S4B4", "key": f"{kind}_{base + i}_green_r0"},
                        "twin": {"run": "outputs/gen/S4B4", "key": f"n_{base + i}_green_r1"}})
    return out


def test_manifest_counts_and_symmetry():
    r1, n1 = _base_manifests()
    ch = _chosen()
    r4, n4 = M.compose(r1, n1, ch)
    chk = M.check(r1, n1, r4, n4, ch, F30.layout_targets)
    assert chk["ok"], chk
    assert chk["items"]["new_counts"] == {"R4": 380, "N4": 380, "want": 380, "ok": True}
    assert chk["items"]["composition"]["R4"] == {"n": 240, "A": 40, "B": 60, "C": 40}
    # N に違う配置の通常を入れたら、R と N の対称が崩れて止まる
    bad = [dict(c) for c in ch]
    bad[0] = dict(bad[0], twin={"run": "x", "key": "n_44999_green_r0"})
    r4b, n4b = M.compose(r1, n1, bad)
    chk = M.check(r1, n1, r4b, n4b, bad, F30.layout_targets)
    assert not chk["ok"] and not chk["items"]["same_layouts_and_targets"]["ok"]
    # 本数が足りない（1 本少ない）
    r4c, n4c = M.compose(r1, n1, ch[:-1])
    assert M.check(r1, n1, r4c, n4c, ch, F30.layout_targets)["items"]["new_counts"]["ok"] is False


def test_manifest_sha_and_base_sha_guard(tmp_path, monkeypatch):
    from recovla.data import convert as C
    p = tmp_path / "R4_x.json"
    C.write_manifest(p, "R4_x", [{"run": "r", "key": "n_1_red_r0"}], "rule")
    assert M.sha256_bytes(p.read_bytes()) == hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):                                               # 同じ名前は上書きしない
        C.write_manifest(p, "R4_x", [], "rule")
    r1, n1 = _base_manifests()
    monkeypatch.setattr(M, "_sources", lambda: ({}, {"R1": {"sha256": "0" * 64, "manifest": r1, "path": tmp_path / "r1"},
                                                    "N1": {"sha256": M.EXPECTED_SHA["N1"], "manifest": n1, "path": tmp_path / "n1"}}))
    monkeypatch.setattr(M, "B4", tmp_path)
    (tmp_path / "data.json").write_text(json.dumps({"ok": True, "chosen": _chosen()}), encoding="utf-8")
    assert M.main(["build", "--dry-run"]) == 1                                         # 元の R1v3 のマニフェストが掲示の値と違う


# ---------------------------------------------------------------- 関門 D
def test_d2_coverage_and_baseline_reproduction():
    eval_xy = [(0.1 * (i % 19), 0.1 * (i // 19)) for i in range(38)]                  # 10 cm 間隔（互いに 3 cm より離れた 38 点）
    r1 = [(0.0, 0.9)] * 28 + [(eval_xy[0][0] + 0.01, eval_xy[0][1]), (eval_xy[1][0], eval_xy[1][1] - 0.02)]  # 2 本だけ近い
    r4 = r1 + [(x + 0.005, y) for x, y in eval_xy[:25]]
    it = G.item_d2(eval_xy, r4, r1)
    assert it["baseline_reproduced"] and it["R1v3_baseline"]["beyond"] == 36
    assert it["status"] == "pass" and it["R4"]["within"] == 25
    it = G.item_d2(eval_xy, r1 + [(x + 0.005, y) for x, y in eval_xy[:10]], r1)
    assert it["status"] == "fail"                                                      # 12/38 < 0.50
    it = G.item_d2(eval_xy, r4, r1 + [(x, y) for x, y in eval_xy[2:4]])               # 物差しが元の値を再現しない
    assert it["status"] == "要確認"
    assert G.coverage([(0, 0)], [(0.03, 0)])["within"] == 1                           # ちょうど 3 cm は以内


def _episode(root, key, cube_xy, target="green", png=True):
    d = root / key
    (d / "overhead").mkdir(parents=True)
    cube = np.zeros((3, 3, 3))
    cube[:, 1, :2] = cube_xy
    np.savez(d / "data.npz", cube_pos=cube, x=np.arange(3))
    (d / "meta.json").write_text(json.dumps({"target": target}), encoding="utf-8")
    if png:
        (d / "overhead" / "000000.png").write_bytes(b"\x89PNG")
    return d


def test_d3_images_d4_bitwise_d5_and_conclusion(tmp_path, monkeypatch):
    monkeypatch.setattr(G.config, "path", lambda p: pathlib.Path(p))
    ch = []
    for i, kind in enumerate(["B"] * 6 + ["C"] * 6):
        key = f"{kind}_{44600 + i}_green_r0"
        _episode(tmp_path / "gen", key, (0.4, 0.0))
        ch.append({"kind": kind, "layout_kind": "empty", "recovery": {"run": str(tmp_path / "gen"), "key": key}})
    d3 = G.item_d3(ch, tmp_path / "img", images_ok=False)
    assert d3["status"] == "要確認" and len(d3["images"]) == 10
    assert G.item_d3(ch, tmp_path / "img2", images_ok=True)["status"] == "pass"
    assert G.item_d3(ch[:5], tmp_path / "img3", images_ok=True)["status"] == "fail"   # 10 枚に足りない
    picks = G.regen_picks(ch)
    assert [c["kind"] for c in picks] == ["B", "B", "B", "C", "C"]
    a = tmp_path / "gen" / ch[0]["recovery"]["key"] / "data.npz"
    b = tmp_path / "copy.npz"
    b.write_bytes(a.read_bytes())
    assert G.compare_npz(a, b)["identical"]
    z = dict(np.load(a))
    z["x"] = z["x"].astype(np.int32)                                                   # 値は同じでも型が違えば一致しない
    np.savez(tmp_path / "c.npz", **z)
    assert G.compare_npz(a, tmp_path / "c.npz")["differ"] == ["x"]
    assert G.item_d4([(a, b)] * 5)["status"] == "pass" and G.item_d4([(a, tmp_path / "c.npz")] * 5)["status"] == "fail"
    assert G.item_d4([])["status"] == "要確認"
    ok_items = {k: {"ok": True} for k in ("normal_part_identical", "same_layouts_and_targets", "new_counts")}
    assert G.item_d5({"items": ok_items})["status"] == "pass"
    assert G.item_d5({"items": dict(ok_items, new_counts={"ok": False})})["status"] == "fail"
    items = {"d1": {"status": "対象外"}, "d2": {"status": "pass"}, "d3": {"status": "pass"}, "d4": {"status": "pass"}, "d5": {"status": "pass"}}
    assert G.conclude(items) == {"verdict": "学習に進む", "pass": True}
    assert G.conclude(dict(items, d3={"status": "要確認"}))["pass"] is False
    assert G.conclude(dict(items, d2={"status": "fail"}))["verdict"].startswith("研究の道へ")


def test_eval_and_train_drop_states(tmp_path, monkeypatch):
    monkeypatch.setattr(G.config, "path", lambda p: pathlib.Path(p))
    ev = tmp_path / "A_P2"
    ev.mkdir()
    for i, (est, t_est) in enumerate(((True, 1.0), (False, None), (True, 2.0))):
        cube = np.zeros((5, 3, 3))
        cube[:, 0, 0] = np.arange(5) * 0.1                                             # 赤の x がこまごとに 0.1 ずつ
        np.savez(ev / f"trial_{i:04d}.npz", sim_time=np.arange(5) * 0.5, cube_pos=cube)
        (ev / f"trial_{i:04d}.json").write_text(json.dumps({"target": "red", "induce": {"established": est, "t_established": t_est}}),
                                                encoding="utf-8")
    xy = G.eval_drop_states(ev)
    assert [round(x, 6) for x, _ in xy] == [0.2, 0.4]                                   # 成立の時刻のこま（1.0 s → 2、2.0 s → 4）
    _episode(tmp_path / "g", "B_1_green_r0", (0.5, 0.1))
    _episode(tmp_path / "g", "C_2_green_r0", (0.7, 0.1))
    tr = G.train_drop_states([{"run": str(tmp_path / "g"), "key": "B_1_green_r0"}, {"run": str(tmp_path / "g"), "key": "C_2_green_r0"}])
    assert tr == [(0.5, 0.1)]                                                          # 落下（B）だけ


# ---------------------------------------------------------------- 学習の設定の違いの検査
class FakeSW:
    BASE_SEED = 1000
    EXISTING_RUN = {"R1v3": "x", "N1v3": "y"}

    def __init__(self, extra=None):
        self.extra = extra or {}
        real = _load("s4_train_seed_t", "scripts/99_s4_train_seed.py")
        self.deep_diff = real.deep_diff

    def load_cfg(self, seed):
        return {"train": {"seed": seed}}

    def build_cfg(self, base, seed, smoke, workers, cfg_all, h40):
        cfg = {"dataset": f"outputs/datasets/{base}_x", "steps": 1000 if smoke else 20000, "batch_size": 32, "save_freq": 5000,
               "seed": cfg_all["train"]["seed"], "num_workers": workers, "note": f"Stage 3 {base}", "fast_query": True}
        if seed != 1000:
            cfg.update(self.extra)
        return cfg, {"dataset": cfg["dataset"]}, {"adopt": False}


DATA_B4 = {"datasets": {"R4": {"dataset": "outputs/datasets/R4_x", "convert_exit": 0, "verify_exit": 0, "verify_pass": True},
                        "N4": {"dataset": "outputs/datasets/N4_x", "convert_exit": 0, "verify_exit": 0, "verify_pass": True}}}


def test_train_config_only_data_note_seed_differ(monkeypatch):
    monkeypatch.setattr(T.config, "path", lambda p: pathlib.Path(p))
    cfg, ref, diffs, ds, _ = T.build_b4_cfg("R4", 1000, False, 3, FakeSW(), None, DATA_B4)
    assert {d[0] for d in diffs} == {"/dataset", "/note"} and cfg["steps"] == ref["steps"] == 20000
    cfg, ref, diffs, ds, _ = T.build_b4_cfg("N4", 1001, False, 3, FakeSW(), None, DATA_B4)
    assert {d[0] for d in diffs} == {"/dataset", "/note", "/seed"} and cfg["seed"] == 1001 and "N4_x" in cfg["dataset"]
    with pytest.raises(SystemExit):                                                    # 種 1001 の重ねがステップも変えていたら止める
        T.build_b4_cfg("R4", 1001, False, 3, FakeSW(extra={"steps": 30000}), None, DATA_B4)
    with pytest.raises(SystemExit):                                                    # 変換・検査が通っていないデータ
        T.build_b4_cfg("R4", 1000, False, 3, FakeSW(), None, {"datasets": {"R4": dict(DATA_B4["datasets"]["R4"], verify_pass=False)}})
    assert T.NAMES == {"R4": "R1v3", "N4": "N1v3"} and T.SEEDS == (1000, 1001)


def test_train_post_check_allows_only_seed_paths_and_dataset(tmp_path, monkeypatch):
    def mk(d, tc, loss=0.03):
        (d / "checkpoints" / "020000" / "pretrained_model").mkdir(parents=True)
        for c in ("005000", "010000", "015000"):
            (d / "checkpoints" / c).mkdir()
        (d / "checkpoints" / "020000" / "pretrained_model" / "train_config.json").write_text(json.dumps(tc), encoding="utf-8")
        (d / "train_run.json").write_text(json.dumps({"exit_code": 0, "log_summary": {"loss_last": loss}}), encoding="utf-8")
    old_tc = {"seed": 1000, "output_dir": "a", "job_name": "a", "dataset": {"root": "R1", "repo_id": "local/R1"}, "steps": 20000}
    new_tc = {"seed": 1001, "output_dir": "b", "job_name": "b", "dataset": {"root": "R4", "repo_id": "local/R4"}, "steps": 20000}
    mk(tmp_path / "old", old_tc)
    mk(tmp_path / "new", new_tc)
    sw = FakeSW()
    sw.EXISTING_RUN = {"R1v3": str(tmp_path / "old"), "N1v3": str(tmp_path / "old")}
    monkeypatch.setattr(T.config, "path", lambda p: pathlib.Path(p))
    res = T.post_check(tmp_path / "new", "R4", sw)
    assert res["train_config_ok"] and res["checkpoints_ok"] and res["exit_code"] == 0
    mk(tmp_path / "new2", dict(new_tc, steps=30000))
    assert not T.post_check(tmp_path / "new2", "R4", sw)["train_config_ok"]
    mk(tmp_path / "new3", dict(new_tc, seed=1002))                                    # 種 1000・1001 のほかは通さない
    assert not T.post_check(tmp_path / "new3", "R4", sw)["train_config_ok"]
    mk(tmp_path / "new4", new_tc)                                                     # N4 の学習なのに R4 のデータ
    assert not T.post_check(tmp_path / "new4", "N4", sw)["train_config_ok"]


def test_train_cli_refuses_other_seeds():
    for argv in (["R4", "--seed", "1002"], ["N4", "--seed", "999"], ["R1v3", "--seed", "1000"]):
        with pytest.raises(SystemExit) as e:
            T.main(argv)
        assert e.value.code == 2                                                       # argparse が拒む（学習も dry-run もしない）


# ---------------------------------------------------------------- 直し（統合のときの点検）で足した検査
def test_ledger_reserved_and_planned_rows_block_but_not_the_band_itself():
    led = LEDGER + "| 44620 | 44625 | 予約 | 段階4 | あり | 予約の行 |\n| 44750 | 44751 | 予定 | 段階4 | あり | 予定の行 |\n"
    used = D.ledger_used(led)
    assert (44400, 44799) not in used and (44620, 44625) in used and (44750, 44751) in used
    assert D.band_problems([44621], led) and D.band_problems([44750], led) and not D.band_problems([44700], led)
    # 掲示板 0164 の smoke（44500〜44503・44510〜44512・44520〜44522）と、決めた帯（候補・予備・smoke）の間に重なりがない
    smoke_0164 = set(range(44500, 44504)) | set(range(44510, 44513)) | set(range(44520, 44523))
    mine = set(range(44600, 44684)) | set(range(44700, 44780))
    assert not smoke_0164 & mine and not D.band_problems(sorted(mine))


def test_stage3_identity_rules():
    from recovla.common import code_version
    cfg_now = {"inject": {"B": 1}, "expert": {"v": 3}, "scene": {}, "sim": {}}
    v3 = {"rig": "v3", "config_used": dict(cfg_now),
          "code_version": {"git_commit": "abc", "code_sha256": code_version.code_version()["code_sha256"]}}
    seen = []

    def clean(c):
        seen.append(c)
        return ""
    r = D.stage3_identity(v3, git_diff=clean, cfg_now=cfg_now)
    assert r["ok"] and seen == ["abc"]
    assert not D.stage3_identity(v3, git_diff=lambda c: "src/recovla/expert/inject.py\n", cfg_now=cfg_now)["ok"]  # 誘発が変わった
    assert not D.stage3_identity(v3, git_diff=lambda c: None, cfg_now=cfg_now)["ok"]                            # git で確かめられない
    assert not D.stage3_identity(v3, git_diff=clean, cfg_now=dict(cfg_now, inject={"B": 2}))["ok"]               # 誘発の設定が違う
    assert not D.stage3_identity(dict(v3, rig="v2"), git_diff=clean, cfg_now=cfg_now)["ok"]
    bad = dict(v3, code_version={"git_commit": "abc", "code_sha256": "0" * 64})
    assert not D.stage3_identity(bad, git_diff=clean, cfg_now=cfg_now)["ok"]


@pytest.mark.skipif(not (ROOT / "outputs" / "f" / "data_v3.json").is_file(), reason="段階 3 の data_v3.json がない（作者の PC だけ）")
def test_stage3_identity_on_this_pc():
    r = D.stage3_identity()
    assert r["ok"], r


def test_worker_history_and_regenerate_replays_same_worker_order(tmp_path, monkeypatch):
    monkeypatch.setattr(G.config, "path", lambda p: pathlib.Path(p))
    run = tmp_path / "S4B4_B_empty"
    run.mkdir()
    rows = []
    for i, (kind, seed, pid, ok) in enumerate((("B", 1, 11, True), ("B", 2, 22, True), ("B", 3, 11, False),
                                               ("B", 4, 11, True), ("n", 1, 22, True))):
        att = [{"name": f"{kind}_{seed}_red_r{r}"} for r in range(2 if seed == 4 else 1)]
        rows.append({"kind": kind, "layout_seed": seed, "color": "red", "layout_kind": "empty", "success": ok,
                     "attempts": att, "worker_pid": pid})
    (run / "generation.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    prev, row = G.worker_history(run, "B_4_red_r1")
    assert [r["layout_seed"] for r in prev] == [1, 3] and row["layout_seed"] == 4      # 同じ働き手（11）の前の 2 本（失敗も含む）
    assert G.worker_history(run, "B_2_red_r0")[0] == []                                # 働き手の最初の仕事
    with pytest.raises(ValueError):
        G.worker_history(run, "B_3_red_r0")                                            # 保存していない（失敗）

    calls, rigs = [], []

    class FakeG:
        EpisodeSpec = staticmethod(lambda s, c, lk, k: (k, s, c, lk))

        @staticmethod
        def run_spec(rig, spec, run_dir, render):
            calls.append((id(rig), spec, pathlib.Path(run_dir).name, render))
            name = f"{spec[0]}_{spec[1]}_{spec[2]}_r{1 if spec[1] == 4 else 0}"
            d = pathlib.Path(run_dir) / name
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / "data.npz", x=np.arange(3))
            return {"success": True, "attempts": [{"name": name}]}

    class FakeRig:
        def __init__(self):
            rigs.append(self)
            self.closed = False

        def close(self):
            self.closed = True

    picks = [{"kind": "B", "recovery": {"run": str(run), "key": "B_4_red_r1"}},
             {"kind": "B", "recovery": {"run": str(run), "key": "B_2_red_r0"}}]
    out = G.regenerate(picks, tmp_path / "regen", make_rig=FakeRig, G=FakeG)
    assert len(rigs) == 2 and all(r.closed for r in rigs)                              # 本ごとに新しい世界
    assert [(c[1][1], c[2]) for c in calls] == [(1, "_history"), (3, "_history"), (4, "B_4_red_r1"), (2, "B_2_red_r0")]
    assert calls[0][0] == calls[1][0] == calls[2][0] != calls[3][0] and all(c[3] for c in calls)   # 同じ世界で順に、描画あり
    assert [o["history"] for o in out] == [2, 0] and all(o["same_saved_attempt"] for o in out)
    # 元と同じ配列なら pass、保存した試みの回が違えば一致しない
    for o in out:
        (pathlib.Path(o["original"]).parent).mkdir(parents=True, exist_ok=True)
        np.savez(o["original"], x=np.arange(3))
    five = (out * 3)[:5]
    assert G.item_d4(five)["status"] == "pass"
    assert G.item_d4([dict(five[0], same_saved_attempt=False)] + five[1:])["status"] == "fail"


def test_manifest_reverify_reads_files(tmp_path, monkeypatch):
    from recovla.data import convert as C
    r1, n1 = _base_manifests()
    ch = _chosen()
    r4, n4 = M.compose(r1, n1, ch)
    monkeypatch.setattr(M, "ROOT", tmp_path)
    man = {"manifests": {}}
    for name, ent in (("R4", r4), ("N4", n4)):
        C.write_manifest(tmp_path / f"{name}.json", name, ent, "rule")
        man["manifests"][name] = {"manifest": f"{name}.json", "sha256": M.sha256_bytes((tmp_path / f"{name}.json").read_bytes())}
    src = {"R1": {"manifest": r1, "sha256": M.EXPECTED_SHA["R1"]}, "N1": {"manifest": n1, "sha256": M.EXPECTED_SHA["N1"]}}
    res = M.reverify(man, src, {"chosen": ch}, F30.layout_targets)
    assert res["ok"], res
    assert G.item_d5(res)["status"] == "pass"
    # ファイルを後から書き換えた（N4 の相手を 1 本だけ別の配置に）→ SHA も対称も落ちる
    m = json.loads((tmp_path / "N4.json").read_text(encoding="utf-8"))
    m["entries"][-1] = {"run": "x", "key": "n_44999_green_r0"}
    (tmp_path / "N4.json").write_text(json.dumps(m), encoding="utf-8")
    res = M.reverify(man, src, {"chosen": ch}, F30.layout_targets)
    assert not res["ok"] and not res["items"]["files_same_sha_as_built"]["ok"] and not res["items"]["same_layouts_and_targets"]["ok"]
    assert G.item_d5(res)["status"] == "fail" and G.item_d5({})["status"] == "fail"
    # 元の R1v3 が掲示の値と違えば落ちる
    src_bad = dict(src, R1=dict(src["R1"], sha256="0" * 64))
    assert not M.reverify(man, src_bad, {"chosen": ch}, F30.layout_targets)["items"]["base_sha_as_posted"]["ok"]
    # 掴み損ね（A）を足した chosen は通さない
    assert not M.check(r1, n1, *M.compose(r1, n1, [dict(ch[0], kind="A")]), [dict(ch[0], kind="A")], F30.layout_targets)["ok"]
