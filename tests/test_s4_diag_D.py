"""scripts/98_s4_d_plan.py（束 1 の回す計画と一括の包み、担当 D）の確かめ（CPU だけ。GPU・シミュレーション・nvidia-smi を使わない）。

使い方: .venv\\Scripts\\python.exe -m pytest -q tests\\test_s4_diag_D.py -p no:cacheprovider
読むもの: scripts/98_s4_d_plan.py（importlib）、configs/s4_gates.json、outputs/s4/rules/budget60_e7_30.json（あれば）、
  outputs/v2eval/S4K/K1・K2/run.json（あれば。無ければ K の速さは既定の 88）、outputs/s4/runs/run_bundle1.ps1（あれば）。
書くもの: pytest の一時フォルダだけ（偽の子のスクリプト・計画・状態・ログ）。
包みの試験は、本物の子の代わりに一時フォルダの偽の子（python で 1 本ずつ「試行」を数え、96_s4_resume.py と同じ鍵の progress.json を書く）を
  subprocess で起こす。96_s4_ops は偽物（メモリは十分、pid は生きていない）に差し替える。
"""
import importlib.util
import json
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("s4_d_plan_t", ROOT / "scripts" / "98_s4_d_plan.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s4_d_plan_t"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def plan(mod):
    return mod.build_plan()


@pytest.fixture(scope="module")
def gates():
    return json.loads((ROOT / "configs" / "s4_gates.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 計画
def test_sources_match_latest_board(mod, plan):
    """最新の掲示（掲示板 0165。改訂 4）の SHA-256 が作業コピーの目標書・s4_gates.json と一致する。0153・0162 の値は記録として残る。"""
    assert mod.BOARD_LATEST[0] == "0165"
    assert mod.BOARD_LATEST[1] is mod.BOARD_0165_SHA
    assert plan["sources"]["matches_board_latest"] is True
    assert plan["sources"]["matches_board_0153"] is False           # 改訂 3 で両方とも変わった
    assert mod.BOARD_0153_SHA["configs/s4_gates.json"].startswith("cf2e8c96")
    assert mod.BOARD_0162_SHA["docs/目標書_段階4.md"].startswith("694338887a7a")      # 改訂 3 の値（記録として残す）
    assert mod.BOARD_0162_SHA["configs/s4_gates.json"].startswith("dcf0dd4c1b2a")
    assert mod.BOARD_0165_SHA["configs/s4_gates.json"].startswith("7f2f651c")


def test_conditions_match_charter(plan):
    en = [c for c in plan["conditions"] if c["enabled"]]
    ids = [c["id"] for c in plan["conditions"]]
    assert len(ids) == len(set(ids))
    count = {}
    for c in en:
        count[c["diag"]] = count.get(c["diag"], 0) + 1
    # 目標書_段階4.md 4-2・8 節: D-RTC 6 設定＋影、X2 の生成と測定、D-E7 5 腕（EX は条件つき）、単発の開始 6、移植 3 x 2、D-復帰 8
    assert count == {"D-RTC": 6, "D-RTC の影の推論": 1, "X2 の生成": 1, "X2 の測定": 1, "D-E7": 5, "D-単発の開始": 6, "移植の腕": 6, "D-復帰": 8}
    trials = {}
    for c in en:
        trials[c["diag"]] = trials.get(c["diag"], 0) + c["trials"]
    assert trials["D-RTC"] == 180 and trials["D-RTC の影の推論"] == 12 and trials["D-E7"] == 200
    assert trials["D-単発の開始"] == 198 and trials["移植の腕"] == 120 and trials["D-復帰"] == 400
    ex = [c for c in plan["conditions"] if c["id"] == "E7_EX"]
    assert len(ex) == 1 and not ex[0]["enabled"]


def test_with_ex_enables_ex(mod):
    p = mod.build_plan(with_ex=True)
    assert any(c["id"] == "E7_EX" and c["enabled"] for c in p["conditions"])
    assert any(j["id"] == "E7_EX" and j["enabled"] for j in p["jobs"])


def test_seeds_inside_allocations(plan, gates):
    alloc = {a["id"]: a["range"] for a in gates["bands"]["allocations"]}
    for c in plan["conditions"]:
        lo, hi = alloc[c["band"]]
        assert lo <= c["seeds"][0] <= c["seeds"][1] <= hi, c["id"]
        assert c["seeds"] == [lo, hi], c["id"]               # 帯を全部使う（割り当てと同じ範囲）
    # 種の指定が帯の先頭から始まり、数が試行数に合う
    for c in plan["conditions"]:
        spec = c["trials_spec"].split(":")
        assert int(spec[-2]) == c["seeds"][0], c["id"]
        n = int(spec[-1])
        assert n * (3 if spec[0] == "natural" else 1) == c["trials"], c["id"]


def test_time_limits(plan):
    for c in plan["conditions"]:
        if c["kind"] == "run":
            assert c["time_limits"]["time_limit_s"] == 60 and c["time_limits"]["scored_at_s"]["primary"] == 30
        elif c["kind"] == "task":
            assert (c["time_limits"]["step_timeout_s"], c["time_limits"]["task_time_limit_s"], c["time_limits"]["retry"]) == (30, 200, 1)
    # 仕事の命令は制限時間を 96 の既定（run 60 s、task 30 s・200 s）以外に変えない
    for j in plan["jobs"]:
        for flag in ("--time-limit-s", "--step-timeout-s", "--task-time-limit-s"):
            if flag in j["argv"]:
                v = j["argv"][j["argv"].index(flag) + 1]
                assert v in ("60", "30", "200"), (j["id"], flag, v)


def test_budget_matches_gates(plan, gates):
    t = plan["totals"]
    assert abs(t["budget_ph_without_x2"] - gates["budget_notes"]["bundle1_total_process_hours"]["s4_r2"]) < 0.1
    b = t["by_diag"]
    p = ROOT / "outputs" / "s4" / "rules" / "budget60_e7_30.json"
    if p.is_file():
        b1 = json.loads(p.read_text(encoding="utf-8"))["budget"]["60s"]["b1"]
        assert abs(b["D-RTC"]["budget_ph"] + b["D-RTC の影の推論"]["budget_ph"] - b1["rtc"]) < 0.06
        assert abs(b["D-E7"]["budget_ph"] - b1["d_e7_5arms"]) < 0.06
        assert abs(b["D-単発の開始"]["budget_ph"] - b1["single_start"]) < 0.06
        assert abs(b["D-復帰"]["budget_ph"] - b1["d_recovery"]) < 0.06
        assert abs(b["移植の腕"]["budget_ph"] - b1["transplant"]) < 0.06
    # K の換算は単発だけに掛かり、E7 は予算のまま
    for c in plan["conditions"]:
        if c["kind"] == "task":
            assert c["est"]["k_scaled_ph"] == c["est"]["budget_ph"]


def test_jobs_cover_each_condition_once(plan):
    en = {c["id"] for c in plan["conditions"] if c["enabled"]}
    seen = []
    for j in plan["jobs"]:
        if j["enabled"]:
            seen += j["covers"]
    assert sorted(seen) == sorted(en)
    by = {j["id"]: j for j in plan["jobs"]}
    assert by["X2_measure"]["after"] == ["X2_gen"]
    for j in plan["jobs"]:
        assert j["argv"][0].startswith("scripts\\98_s4_"), j["id"]
        if j["group"] == "D-E7":
            assert j["block_trials"] == 5 and j["needs_api"]
        if j["group"] in ("D-RTC", "D-recovery", "X2", "D-RTC-shadow"):
            assert j["block_trials"] is None                # 中で交互にする・分けない仕事に --max-new を渡さない
        if j["group"] in ("D-start", "XPL"):
            assert j["requires"] == ["outputs\\s4\\d_start\\definitions.json"]
    rc = [j for j in plan["jobs"] if j["group"] == "D-recovery"]
    assert len(rc) == 4 and all(len(j["covers"]) == 2 for j in rc)


def test_monitor_paths(plan):
    m = plan["monitor"]
    assert m["bundle_status"] == "outputs\\s4\\runs\\bundle1_status.json"
    assert set(m["condition_progress_files"]) == {c["id"] for c in plan["conditions"] if c["enabled"]}
    for cid, pth in m["condition_progress_files"].items():
        assert pth is None or pth.endswith("progress.json")


def test_smoke_plan_uses_own_band(mod):
    p = mod.build_smoke_plan()
    assert [j["id"] for j in p["jobs"]] == ["SMK_a", "SMK_b"]
    for c in p["conditions"]:
        assert 44490 <= c["seeds"][0] <= c["seeds"][1] <= 44499
        assert c["experiment"].startswith("S4SMOKE_")
    for j in p["jobs"]:
        assert j["argv"][0] == "scripts\\96_s4_resume.py" and j["block_trials"] == 2
    assert p["bundle_defaults"]["status"] != "outputs\\s4\\runs\\bundle1_status.json"


# ---------------------------------------------------------------- 順番の規則
def _j(jid, group, est, trials=10, block=5, after=()):
    return {"id": jid, "group": group, "est": {"budget_ph": est, "k_scaled_ph": est}, "trials": trials, "block_trials": block,
            "after": list(after), "enabled": True, "covers": [jid]}


def test_pick_next_longest_group_then_round_robin(mod):
    jobs = [_j("a1", "A", 1.0), _j("a2", "A", 1.0), _j("b1", "B", 3.0), _j("c1", "C", 0.5, after=["a1"])]
    st = {j["id"]: {"state": "pending", "blocks_done": 0} for j in jobs}
    assert mod.pick_next(jobs, st, set(), True)["id"] == "b1"          # 群の残りが最も長い
    assert mod.pick_next(jobs, st, {"b1"}, True)["id"] == "a1"
    st["a1"]["blocks_done"] = 1
    assert mod.pick_next(jobs, st, {"b1"}, True)["id"] == "a2"          # 群の中は済んだ塊の少ない順（交互）
    st["a1"]["state"] = "done"
    st["a2"]["state"] = "done"
    st["b1"]["state"] = "done"
    assert mod.pick_next(jobs, st, set(), True)["id"] == "c1"           # 依存先が done になってから
    assert mod.pick_next(jobs, {**st, "a1": {"state": "failed", "blocks_done": 1}}, set(), True) is None


def test_no_interleave_longest_job_first(mod):
    jobs = [_j("a1", "A", 1.0), _j("a2", "A", 2.0), _j("b1", "B", 1.5)]
    st = {j["id"]: {"state": "pending", "blocks_done": 0} for j in jobs}
    assert mod.pick_next(jobs, st, set(), False)["id"] == "a2"
    assert mod.blocks_of(jobs[0], False) == 1 and mod.blocks_of(jobs[0], True) == 2


def test_simulate_bounds(mod, plan):
    sim = mod.simulate(plan, lanes=3, interleave=True)
    tot = plan["totals"]["k_scaled_ph"]
    assert tot / 3 - 1e-6 <= sim["makespan_h"] <= tot
    one = mod.simulate(plan, lanes=1, interleave=True)
    assert abs(one["makespan_h"] - tot) < 0.05


def test_classify(mod):
    c = mod.classify
    assert c(0, {"pid": 5, "status": "done"}, 5)[0] == "done"
    assert c(1, {"pid": 5, "status": "stopped", "stop_reason": "max_new:5"}, 5)[0] == "block_done"
    assert c(1, {"pid": 5, "status": "stopped", "stop_reason": "stop_file:x"}, 5)[0] == "stopped"
    assert c(1, {"pid": 5, "status": "memory_timeout", "stop_reason": "memory_timeout"}, 5)[0] == "memory_timeout"
    assert c(2, {"pid": 5, "status": "error", "error": "boom"}, 5)[0] == "failed"
    # 前の塊の progress.json（pid が違う）を、引数の食い違いで止まった子（終了コード 3）の結果に読み違えない
    assert c(3, {"pid": 4, "status": "stopped", "stop_reason": "max_new:5"}, 5)[0] == "failed"
    # venv の起動役: Popen の pid と progress の pid が違っても、子を始めた後に書かれた progress は使う
    assert c(1, {"pid": 4, "status": "stopped", "stop_reason": "max_new:5"}, 5, True)[0] == "block_done"
    # 引き取った子（終了コードが分からない）は progress だけで決める
    assert c(None, {"pid": 5, "status": "done"}, 5)[0] == "done"
    assert c(None, {"pid": 5, "status": "running"}, 5)[0] == "failed"


# ---------------------------------------------------------------- 包み（偽の子で）
FAKE_CHILD = r'''
import argparse, json, os, pathlib, sys
ap = argparse.ArgumentParser()
ap.add_argument("--name"); ap.add_argument("--out"); ap.add_argument("--total", type=int); ap.add_argument("--order")
ap.add_argument("--failflag", default=""); ap.add_argument("--max-new", type=int, default=0); ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
with open(a.order, "a", encoding="utf-8") as f:
    f.write(a.name + "\n")
if a.failflag and pathlib.Path(a.failflag).exists():
    (out / "progress.json").write_text(json.dumps({"pid": os.getpid(), "status": "error", "error": "fake failure"}), encoding="utf-8")
    sys.exit(2)
sf = out / "done.txt"
done = int(sf.read_text()) if sf.exists() else 0
ran, status, reason, code = 0, "done", None, 0
while done < a.total:
    if a.max_new and ran >= a.max_new:
        status, reason, code = "stopped", f"max_new:{a.max_new}", 1
        break
    done += 1
    ran += 1
sf.write_text(str(done))
(out / "progress.json").write_text(json.dumps({"pid": os.getpid(), "status": status, "stop_reason": reason, "done": done,
                                               "total": a.total}), encoding="utf-8")
sys.exit(code)
'''


def _fake_ops():
    return types.SimpleNamespace(LIVE_STATUS={"starting", "loading", "running", "quiet_wait", "memory_wait"},
                                 _alive=lambda pid, ct=None: False,
                                 memory_gb=lambda: {"phys_free_gb": 50.0, "commit_free_gb": 50.0})


def _mk_plan(tmp: pathlib.Path, jobs_spec):
    child = tmp / "fake_child.py"
    child.write_text(FAKE_CHILD, encoding="utf-8")
    order = tmp / "order.txt"
    jobs = []
    for jid, group, total, block, after in jobs_spec:
        d = tmp / "out" / jid
        jobs.append({"id": jid, "group": group, "covers": [jid], "enabled": True, "trials": total, "block_trials": block,
                     "after": list(after), "requires": [], "est": {"budget_ph": 1.0, "k_scaled_ph": 1.0},
                     "argv": [str(child), "--name", jid, "--out", str(d), "--total", str(total), "--order", str(order),
                              "--failflag", str(tmp / f"FAIL_{jid}")],
                     "progress": str(d / "progress.json")})
    p = {"name": "t_bundle", "experiment": "T", "python": sys.executable, "jobs": jobs,
         "bundle_defaults": {"status": str(tmp / "status.json"), "stop_file": str(tmp / "STOP"), "log": str(tmp / "bundle.log"),
                             "log_dir": str(tmp / "logs")}}
    path = tmp / "plan.json"
    path.write_text(json.dumps(p), encoding="utf-8")
    return path, order


def _run(mod, path, *extra):
    a = mod.build_parser().parse_args(["bundle", "--plan", str(path), "--lanes", "1", "--poll-s", "0.05", "--stagger-s", "0", *extra])
    return mod.cmd_bundle(a, ops=_fake_ops())


def _order(order):
    return order.read_text(encoding="utf-8").split() if order.exists() else []


def test_bundle_interleaves_and_resumes(mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g", 3, 2, ()), ("b", "g", 3, 2, ())])
    assert _run(mod, path) == 0
    assert _order(order) == ["a", "b", "a", "b"]                         # 種の塊ごとに交互
    st = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert st["status"] == "done" and st["done"] == st["total"] == 2
    assert {"status", "pid", "proc_create_time", "updated", "done", "total"} <= set(st)   # 96_s4_ops.py wait が読む鍵
    assert all(e["state"] == "done" and e["blocks_done"] == 2 for e in st["jobs"].values())
    # 同じコマンドをもう一度: 完了した仕事は飛ばす
    assert _run(mod, path) == 0
    assert _order(order) == ["a", "b", "a", "b"]


def test_bundle_stop_file_and_resume(mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g", 3, 2, ()), ("b", "g", 3, 2, ())])
    (tmp_path / "STOP").write_text("", encoding="utf-8")
    assert _run(mod, path) == 1                                         # 合図があれば何も始めない
    assert _order(order) == []
    (tmp_path / "STOP").unlink()
    assert _run(mod, path, "--max-jobs", "1") == 1                      # 1 塊で止まる（今の塊は終える）
    assert _order(order) == ["a"]
    st = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert st["status"] == "stopped" and st["jobs"]["a"]["blocks_done"] == 1 and st["jobs"]["a"]["state"] == "pending"
    assert _run(mod, path) == 0                                         # 続きから
    assert _order(order) == ["a", "b", "a", "b"]
    assert json.loads((tmp_path / "out" / "a" / "progress.json").read_text())["done"] == 3


def test_bundle_failure_does_not_stop_others(mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g1", 2, None, ()), ("dep", "g1", 2, None, ("a",)), ("b", "g2", 2, None, ())])
    (tmp_path / "FAIL_a").write_text("", encoding="utf-8")
    assert _run(mod, path) == 2
    st = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert st["jobs"]["a"]["state"] == "failed" and st["jobs"]["b"]["state"] == "done"
    assert st["jobs"]["dep"]["state"] == "blocked" and st["status"] == "error"
    assert "dep" not in _order(order)
    (tmp_path / "FAIL_a").unlink()
    assert _run(mod, path) == 0                                         # 直してから同じコマンド: 失敗した仕事と依存先だけ回す
    assert _order(order).count("b") == 1 and _order(order)[-2:] == ["a", "dep"]


def test_bundle_missing_script_or_prereq(mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g", 2, None, ()), ("b", "g", 2, None, ())])
    p = json.loads(path.read_text(encoding="utf-8"))
    p["jobs"][0]["argv"][0] = str(tmp_path / "no_such_script.py")
    p["jobs"][1]["requires"] = [str(tmp_path / "definitions.json")]
    path.write_text(json.dumps(p), encoding="utf-8")
    assert _run(mod, path) == 2
    st = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert st["jobs"]["a"]["last_error"].startswith("script_missing")
    assert st["jobs"]["b"]["last_error"].startswith("prereq_missing")
    assert _order(order) == []


def test_bundle_recovers_running_entry_after_crash(mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g", 2, None, ())])
    # 前の回が再起動で消えた: 状態は running のまま、包みも子も生きていない
    (tmp_path / "status.json").write_text(json.dumps({"status": "running", "pid": 999999, "jobs": {
        "a": {"state": "running", "pid": 999998, "blocks_done": 0, "attempts": 1, "history": []}}}), encoding="utf-8")
    assert _run(mod, path) == 0
    st = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert st["jobs"]["a"]["state"] == "done" and st["jobs"]["a"]["attempts"] == 2


def test_bundle_refuses_second_instance(mod, tmp_path):
    path, _ = _mk_plan(tmp_path, [("a", "g", 2, None, ())])
    (tmp_path / "status.json").write_text(json.dumps({"status": "running", "pid": 12345, "jobs": {}}), encoding="utf-8")
    ops = _fake_ops()
    ops._alive = lambda pid, ct=None: True
    a = mod.build_parser().parse_args(["bundle", "--plan", str(path), "--lanes", "1", "--poll-s", "0.05"])
    assert mod.cmd_bundle(a, ops=ops) == 3


def test_bundle_memory_gate(mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g", 2, None, ())])
    ops = _fake_ops()
    ops.memory_gb = lambda: {"phys_free_gb": 5.0, "commit_free_gb": 50.0}
    a = mod.build_parser().parse_args(["bundle", "--plan", str(path), "--lanes", "1", "--poll-s", "0.02", "--mem-timeout-min", "0.001"])
    assert mod.cmd_bundle(a, ops=ops) == 1                              # 12 GB 未満なら始めず、待ちの時間切れで止まる
    assert _order(order) == []
    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["status"] == "memory_timeout"


def test_status_is_readable_by_wait(mod, tmp_path):
    path, _ = _mk_plan(tmp_path, [("a", "g", 1, None, ())])
    assert _run(mod, path) == 0
    spec = importlib.util.spec_from_file_location("s4_ops_t", ROOT / "scripts" / "96_s4_ops.py")
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    t = {"kind": "progress", "path": tmp_path / "status.json", "t0": 0}
    ops.evaluate_target(t, types.SimpleNamespace(appear_min=10, stall_min=30))
    assert t["final"] == {"outcome": "done", "code": 0}


def test_ps1_is_ascii():
    p = ROOT / "outputs" / "s4" / "runs" / "run_bundle1.ps1"
    if not p.is_file():
        pytest.skip("run_bundle1.ps1 が無い（outputs は Git の外）")
    assert all(b < 128 for b in p.read_bytes())
