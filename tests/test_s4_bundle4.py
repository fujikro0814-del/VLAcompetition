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
    assert G.item_d5({"check": {"items": ok_items}})["status"] == "pass"
    assert G.item_d5({"check": {"items": dict(ok_items, new_counts={"ok": False})}})["status"] == "fail"
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
