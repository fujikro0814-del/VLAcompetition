"""運用の守りの直し（依頼 5。査読 docs/stage4/review_cloud.md の重要 3・5・6、軽微 5・6・7・11・12・16・17）の確かめ。
CPU だけ。GPU・シミュレーション・nvidia-smi・ネットワークを使わない（子は偽のプロセス、時刻は偽の値、LLM は偽のクライアント）。

使い方: .venv\\Scripts\\python.exe -m pytest -q tests\\test_s4_ops_fix.py -p no:cacheprovider
読むもの: scripts/96_s4_ops.py・97_s4_ledger_check.py・98_s4_d_plan.py・98_s4_d_rtc.py・98_s4_d_recovery.py（importlib）、
  src/recovla/planner/decompose_s4.py、configs/s4_gates.json（D-復帰の帯の確かめ）。書くもの: pytest の一時フォルダだけ。
"""
import argparse
import datetime as dt
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def plan_mod():
    return _load(ROOT / "scripts" / "98_s4_d_plan.py", "s4_d_plan_fix_t")


@pytest.fixture(scope="module")
def ops():
    return _load(ROOT / "scripts" / "96_s4_ops.py", "s4_ops_fix_t")


@pytest.fixture(scope="module")
def ledger():
    return _load(ROOT / "scripts" / "97_s4_ledger_check.py", "s4_ledger_fix_t")


# ================================================================ 98_s4_d_plan.py（一括の包み）
FAKE_CHILD = r'''
import argparse, json, os, pathlib, sys
ap = argparse.ArgumentParser()
ap.add_argument("--name"); ap.add_argument("--out"); ap.add_argument("--total", type=int); ap.add_argument("--order")
ap.add_argument("--flags"); ap.add_argument("--max-new", type=int, default=0)
a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
fl = pathlib.Path(a.flags)
with open(a.order, "a", encoding="utf-8") as f:
    f.write(a.name + "\n")

def prog(**kw):
    (out / "progress.json").write_text(json.dumps(dict({"pid": os.getpid()}, **kw)), encoding="utf-8")

def take(flag):
    """フラグのファイルの数を 1 減らす（"always" は減らさない）。残っていれば True。"""
    p = fl / f"{flag}_{a.name}"
    if not p.exists():
        return False
    t = p.read_text().strip()
    if t == "always":
        return True
    n = int(t or "1")
    if n <= 1:
        p.unlink()
    else:
        p.write_text(str(n - 1))
    return True

if take("MEMTO"):                       # 96 の子のメモリ待ちの時間切れ
    prog(status="memory_timeout", stop_reason="memory_timeout"); sys.exit(1)
if take("MEMTOROT"):                    # rotate の外側の包みの形（中の 96 の時間切れを stopped で返す）
    prog(status="stopped", stop_reason="range10_cap5: memory_timeout"); sys.exit(1)
if take("SELFSTOP"):                    # 子が自分の <条件>\STOP で止まった
    prog(status="stopped", stop_reason="stop_file:" + str(out / "STOP")); sys.exit(1)
if take("CRASH"):                       # progress を書かずに、捕まえない例外で終了コード 1（壊れた resume_log.json）
    print("json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)  resume_log.json", file=sys.stderr)
    sys.exit(1)
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
if status == "done" and not take("NOSEAL"):
    (out / "run.json").write_text("{}", encoding="utf-8")
    (out / "G_AUDIT.json").write_text('{"met": true}', encoding="utf-8")
prog(status=status, stop_reason=reason, done=done, total=a.total)
sys.exit(code)
'''


def _fake_ops(**kw):
    o = types.SimpleNamespace(LIVE_STATUS={"starting", "loading", "running", "quiet_wait", "memory_wait"},
                              _alive=lambda pid, ct=None: False,
                              memory_gb=lambda: {"phys_free_gb": 50.0, "commit_free_gb": 50.0})
    o.__dict__.update(kw)
    return o


def _mk_plan(tmp: pathlib.Path, jobs_spec, with_conditions=False):
    """jobs_spec: [(id, 群, 試行数, 塊)]。with_conditions なら、各仕事が 1 つの run の条件（out_dir＝子の出力）をまとめる。"""
    child = tmp / "fake_child.py"
    child.write_text(FAKE_CHILD, encoding="utf-8")
    (tmp / "flags").mkdir(exist_ok=True)
    order = tmp / "order.txt"
    jobs, conds = [], []
    for jid, group, total, block in jobs_spec:
        d = tmp / "out" / jid
        jobs.append({"id": jid, "group": group, "covers": [jid], "enabled": True, "trials": total, "block_trials": block,
                     "after": [], "requires": [], "est": {"budget_ph": 1.0, "k_scaled_ph": 1.0},
                     "argv": [str(child), "--name", jid, "--out", str(d), "--total", str(total), "--order", str(order),
                              "--flags", str(tmp / "flags")],
                     "progress": str(d / "progress.json")})
        conds.append({"id": jid, "kind": "run", "enabled": True, "out_dir": str(d), "progress": str(d / "progress.json")})
    p = {"name": "t_bundle", "experiment": "T", "python": sys.executable, "jobs": jobs,
         "bundle_defaults": {"status": str(tmp / "status.json"), "stop_file": str(tmp / "STOP"), "log": str(tmp / "bundle.log"),
                             "log_dir": str(tmp / "logs")}}
    if with_conditions:
        p["conditions"] = conds
    path = tmp / "plan.json"
    path.write_text(json.dumps(p), encoding="utf-8")
    return path, order


def _flag(tmp, flag, jid, n="1"):
    (tmp / "flags" / f"{flag}_{jid}").write_text(str(n), encoding="utf-8")


def _run(mod, path, *extra, ops=None):
    a = mod.build_parser().parse_args(["bundle", "--plan", str(path), "--lanes", "1", "--poll-s", "0.05", "--stagger-s", "0", *extra])
    return mod.cmd_bundle(a, ops=ops or _fake_ops())


def _order(order):
    return order.read_text(encoding="utf-8").split() if order.exists() else []


def _status(tmp):
    return json.loads((tmp / "status.json").read_text(encoding="utf-8"))


def test_classify_memory_timeout_of_outer_wrappers_and_crash(plan_mod):
    c = plan_mod.classify
    # rotate（98_s4_d_rtc）と D-復帰（98_s4_d_recovery）は、中の 96 の時間切れを status=stopped で返す
    assert c(1, {"pid": 5, "status": "stopped", "stop_reason": "range10_cap5: memory_timeout"}, 5)[0] == "memory_timeout"
    assert c(1, {"pid": 5, "status": "stopped",
                 "stop_reason": "R1v3_misplace: 96 の終了コード 1（memory_timeout）"}, 5)[0] == "memory_timeout"
    assert c(1, {"pid": 5, "status": "stopped", "stop_reason": "stop_file:x\\STOP"}, 5)[0] == "stopped"
    # 終了コード 1 で、この回の progress が無い: 止まったのではなく落ちた
    assert c(1, None, 5)[0] == "failed"
    assert c(1, {"pid": 4, "status": "stopped", "stop_reason": "max_new:5"}, 5)[0] == "failed"


def test_child_memory_timeout_requeues_and_does_not_stop_bundle(plan_mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g1", 2, None), ("b", "g2", 2, None)])
    _flag(tmp_path, "MEMTO", "a", 1)                                     # a は 1 回だけメモリ待ちの時間切れ
    assert _run(plan_mod, path) == 0                                     # 全体は止まらず、a は後で回し直して完了
    st = _status(tmp_path)
    assert st["status"] == "done" and all(e["state"] == "done" for e in st["jobs"].values())
    assert [h["result"] for h in st["jobs"]["a"]["history"]] == ["memory_timeout", "done"]
    assert _order(order).count("a") == 2 and _order(order).count("b") == 1
    log = (tmp_path / "bundle.log").read_text(encoding="utf-8")
    assert "メモリ待ちの時間切れ" in log


def test_rotate_style_memory_timeout_is_not_a_global_stop(plan_mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("rot", "g1", 2, None), ("b", "g2", 2, None)])
    _flag(tmp_path, "MEMTOROT", "rot", 1)
    assert _run(plan_mod, path) == 0
    st = _status(tmp_path)
    assert st["jobs"]["rot"]["history"][0]["result"] == "memory_timeout" and st["jobs"]["rot"]["state"] == "done"
    assert st["jobs"]["b"]["state"] == "done"


def test_repeated_memory_timeout_stops_only_that_job(plan_mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g1", 2, None), ("b", "g2", 2, None)])
    _flag(tmp_path, "MEMTO", "a", "always")
    assert _run(plan_mod, path, "--mem-timeout-repeat", "2") == 1         # メモリで止めた仕事が残った: 終了コード 1
    st = _status(tmp_path)
    assert st["status"] == "memory_timeout"
    assert st["jobs"]["a"]["state"] == "mem_stopped" and st["jobs"]["a"]["mem_timeouts"] == 2
    assert "続けて 2 回" in st["jobs"]["a"]["last_error"]
    assert st["jobs"]["b"]["state"] == "done"
    assert _order(order).count("a") == 2
    # 次の回は回し直す（数え直し）
    (tmp_path / "flags" / "MEMTO_a").unlink()
    assert _run(plan_mod, path) == 0
    assert _status(tmp_path)["jobs"]["a"]["state"] == "done"


def test_child_self_stop_stops_only_that_job(plan_mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g1", 2, None), ("b", "g2", 2, None)])
    _flag(tmp_path, "SELFSTOP", "a", 1)
    assert _run(plan_mod, path) == 1                                     # 止まった仕事が残る: 1（全体の合図ではない）
    st = _status(tmp_path)
    assert st["jobs"]["a"]["state"] == "stopped" and st["jobs"]["b"]["state"] == "done"
    assert "b" in _order(order)                                          # ほかの仕事は続けた
    assert _run(plan_mod, path) == 0                                     # 次の回で回し直す


def test_stop_file_still_stops_everything(plan_mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g1", 2, None), ("b", "g2", 2, None)])
    (tmp_path / "STOP").write_text("", encoding="utf-8")
    assert _run(plan_mod, path) == 1 and _order(order) == []
    assert _status(tmp_path)["stop_reason"].startswith("stop_file:")


def test_crash_without_progress_is_failure_with_hint(plan_mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g1", 2, None)])
    _flag(tmp_path, "CRASH", "a", 1)
    assert _run(plan_mod, path) == 2
    e = _status(tmp_path)["jobs"]["a"]
    assert e["state"] == "failed" and "JSONDecodeError" in e["last_error"] and "resume_log.json" in e["last_error"]


def test_done_without_run_json_is_called_once_more(plan_mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g1", 2, None)], with_conditions=True)
    _flag(tmp_path, "NOSEAL", "a", 1)                                    # 1 回目は run.json・G_AUDIT を書かずに 0 で終わる
    assert _run(plan_mod, path) == 0
    e = _status(tmp_path)["jobs"]["a"]
    assert [h["result"] for h in e["history"]] == ["seal_retry", "done"] and e["state"] == "done"
    assert (tmp_path / "out" / "a" / "run.json").is_file() and (tmp_path / "out" / "a" / "G_AUDIT.json").is_file()


def test_done_without_run_json_twice_is_failure(plan_mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g1", 2, None)], with_conditions=True)
    _flag(tmp_path, "NOSEAL", "a", "always")
    assert _run(plan_mod, path) == 2
    e = _status(tmp_path)["jobs"]["a"]
    assert e["state"] == "failed" and "run.json・G_AUDIT.json" in e["last_error"]


def test_previous_done_without_run_json_is_rerun(plan_mod, tmp_path):
    """再起動の隙間: 前の回の状態は done だが、条件に run.json・G_AUDIT が無い（重要 3）。"""
    path, order = _mk_plan(tmp_path, [("a", "g1", 2, None)], with_conditions=True)
    assert _run(plan_mod, path) == 0 and _order(order) == ["a"]
    (tmp_path / "out" / "a" / "G_AUDIT.json").unlink()
    assert _run(plan_mod, path) == 0 and _order(order) == ["a", "a"]       # もう 1 回呼んで書かせた
    assert (tmp_path / "out" / "a" / "G_AUDIT.json").is_file()
    assert _run(plan_mod, path) == 0 and _order(order) == ["a", "a"]       # そろっていれば呼ばない


def test_lock_refuses_second_bundle(plan_mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g1", 2, None)])
    held = plan_mod._lock(tmp_path / "status.json.lock")                  # ほかの包みがロックを持っている
    try:
        assert held is not None
        assert _run(plan_mod, path) == 3 and _order(order) == []
    finally:
        plan_mod._unlock(held)
    assert _run(plan_mod, path) == 0                                     # 外れれば回る（ファイルが残っていても邪魔をしない）
    assert (tmp_path / "status.json.lock").exists()


def test_lock_is_released_when_holder_process_dies(plan_mod, tmp_path):
    lp = tmp_path / "s.lock"
    code = (f"import importlib.util,sys,time\nspec=importlib.util.spec_from_file_location('p',{str(ROOT / 'scripts' / '98_s4_d_plan.py')!r})\n"
            f"m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\nf=m._lock(__import__('pathlib').Path({str(lp)!r}))\n"
            "print('held' if f else 'no', flush=True)\ntime.sleep(60)\n")
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "held"
        assert plan_mod._lock(lp) is None                                # 生きている間は取れない
    finally:
        p.kill()
        p.wait()
    f = None
    for _ in range(20):                                                  # 強制終了の後は OS が外す（再起動のまね）
        f = plan_mod._lock(lp)
        if f:
            break
        time.sleep(0.1)
    assert f is not None
    plan_mod._unlock(f)


def test_broken_status_file_is_moved_aside(plan_mod, tmp_path):
    path, order = _mk_plan(tmp_path, [("a", "g1", 2, None)])
    (tmp_path / "status.json").write_bytes(b"\x00" * 64)                 # 電源断の 0 埋め
    assert _run(plan_mod, path) == 0
    assert list(tmp_path.glob("status.json.broken_*")) and _status(tmp_path)["status"] == "done"


def test_lanes_drop_while_memory_is_short(plan_mod, tmp_path, ops):
    path, _ = _mk_plan(tmp_path, [("a", "g1", 2, None), ("b", "g2", 2, None)])
    plan, p = plan_mod.load_plan(types.SimpleNamespace(plan=str(path)))
    a = plan_mod.build_parser().parse_args(["bundle", "--plan", str(path), "--lanes", "3"])
    b = plan_mod.Bundle(a, plan, p, ops)
    assert b.lanes_now() == 3
    # 動いている子の progress が memory_wait（その子を始めた後に書かれたもの）なら印を見つける
    d = tmp_path / "out" / "a"
    d.mkdir(parents=True)
    (d / "progress.json").write_text(json.dumps({"status": "memory_wait", "wait_note": "空き 物理 9 GB"}), encoding="utf-8")
    b.jobs["a"] = {"popen": None, "pid": 1, "create_time": None, "started_t": time.time() - 5}
    mem, hung = b.scan_children()
    assert "memory_wait" in mem and hung == []
    b.mem_short_t = time.time()
    assert b.lanes_now() == 2                                            # 3 → 2
    b.mem_short_t = time.time() - 31 * 60
    assert b.lanes_now() == 3                                            # 30 分見なければ戻す
    a1 = plan_mod.build_parser().parse_args(["bundle", "--plan", str(path), "--lanes", "1"])
    b1 = plan_mod.Bundle(a1, plan, p, ops)
    b1.mem_short_t = time.time()
    assert b1.lanes_now() == 1                                           # 1 より減らさない
    # 前の子の progress（子を始める前に書かれたもの）は見ない
    b.jobs["a"]["started_t"] = time.time() + 60
    assert b.scan_children() == ("", [])


def test_bundle_reports_hung_child(plan_mod, tmp_path, ops):
    path, _ = _mk_plan(tmp_path, [("a", "g1", 2, None)])
    plan, p = plan_mod.load_plan(types.SimpleNamespace(plan=str(path)))
    a = plan_mod.build_parser().parse_args(["bundle", "--plan", str(path)])
    b = plan_mod.Bundle(a, plan, p, ops)
    d = tmp_path / "out" / "a"
    d.mkdir(parents=True)
    old = (dt.datetime.now() - dt.timedelta(minutes=40)).strftime("%Y-%m-%d %H:%M:%S")
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (d / "progress.json").write_text(json.dumps({"status": "running", "current": 7, "current_started": old, "updated": now,
                                                 "time_limits": {"time_limit_s": 60.0}}), encoding="utf-8")
    b.jobs["a"] = {"popen": None, "pid": 1, "create_time": None, "started_t": time.time() - 5}
    mem, hung = b.scan_children()
    assert mem == "" and len(hung) == 1 and hung[0]["job"] == "a" and hung[0]["current"] == 7
    assert hung[0]["allowed_min"] == 19.0                               # 60 s × 4 + 15 分


def test_write_atomic_reports_failure(plan_mod, tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}", encoding="utf-8")
    if os.name != "nt":
        pytest.skip("置き換え先を開いている間の失敗は Windows の動き")
    with open(p, "rb"):
        assert plan_mod.write_atomic(p, "{\"a\": 1}", tries=2) is False   # 読み手が掴んでいる間は置き換えられない（Windows）
    assert plan_mod.write_atomic(p, "{\"a\": 1}") is True


# ================================================================ 96_s4_ops.py（監視役）
NOW = dt.datetime(2026, 10, 8, 12, 0, 0)


def _ts(minutes_ago):
    return (NOW - dt.timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S")


def _wait_args(**kw):
    a = dict(appear_min=10, stall_min=30, hung_factor=4.0, hung_margin_min=15.0)
    a.update(kw)
    return types.SimpleNamespace(**a)


def test_hung_check_thresholds(ops):
    run = {"status": "running", "current_started": _ts(18), "time_limits": {"time_limit_s": 60.0}}
    assert ops.hung_check(run, now=NOW) is None                          # 許す 60 × 4 s + 15 分 = 19 分
    run["current_started"] = _ts(20)
    h = ops.hung_check(run, now=NOW)
    assert h and h["elapsed_min"] == 20.0 and h["allowed_min"] == 19.0
    task = {"status": "running", "current_started": _ts(25),
            "time_limits": {"step_timeout_s": 30.0, "retry": 1, "task_time_limit_s": 200.0}}
    assert ops.hung_check(task, now=NOW) is None                         # E7: 200 × 4 s + 15 分 = 28.3 分
    task["current_started"] = _ts(29)
    assert ops.hung_check(task, now=NOW)
    assert ops.hung_check(dict(task, status="memory_wait"), now=NOW) is None      # 待ちの間は当てない
    assert ops.hung_check(task, now=NOW, factor=0) is None
    assert ops.hung_check({"status": "running", "current_started": _ts(29)}, now=NOW)   # 制限時間が無ければ 200 s とみなす


def test_wait_detects_hung_trial_despite_heartbeat(ops, tmp_path, monkeypatch):
    """心拍のスレッドが updated を書き続けても、current_started からの経過で固まりを見つける（重要 6）。"""
    monkeypatch.setattr(ops, "_now", lambda: NOW)
    monkeypatch.setattr(ops, "_alive", lambda pid, ct=None: True)
    p = tmp_path / "progress.json"
    p.write_text(json.dumps({"status": "running", "pid": 123, "updated": _ts(0), "current": 4, "current_started": _ts(45),
                             "time_limits": {"time_limit_s": 60.0}, "condition": "K1"}), encoding="utf-8")
    t = {"kind": "progress", "path": p, "t0": time.time()}
    ops.evaluate_target(t, _wait_args())
    assert t["final"] == {"outcome": "hung", "code": 4} and t["hung"][0]["current"] == 4
    # 試行が進んでいれば（current_started が新しい）何もしない
    p.write_text(json.dumps({"status": "running", "pid": 123, "updated": _ts(0), "current_started": _ts(2),
                             "time_limits": {"time_limit_s": 60.0}}), encoding="utf-8")
    t = {"kind": "progress", "path": p, "t0": time.time()}
    ops.evaluate_target(t, _wait_args())
    assert "final" not in t


def test_wait_unknown_status_and_unreadable_get_stall_rule(ops, tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "_now", lambda: NOW)
    monkeypatch.setattr(ops, "_alive", lambda pid, ct=None: True)
    monkeypatch.setattr(ops.time, "sleep", lambda s: None)
    p = tmp_path / "progress.json"
    p.write_text(json.dumps({"status": "weird", "pid": 1, "updated": _ts(31)}), encoding="utf-8")
    t = {"kind": "progress", "path": p, "t0": time.time()}
    ops.evaluate_target(t, _wait_args())
    assert t["final"] == {"outcome": "stalled", "code": 4}                # 未知の状態にも途絶の判定
    # updated の無い未知の状態は、ファイルの更新時刻で測る
    p.write_text(json.dumps({"status": "weird", "pid": 1}), encoding="utf-8")
    os.utime(p, (time.time() - 40 * 60, time.time() - 40 * 60))
    t = {"kind": "progress", "path": p, "t0": time.time()}
    ops.evaluate_target(t, _wait_args())
    assert t["final"]["outcome"] == "stalled"
    # 読めない（0 埋め）: 1 回目は待ち、--stall-min 分続いたらコード 4
    p.write_bytes(b"\x00" * 100)
    t = {"kind": "progress", "path": p, "t0": time.time()}
    ops.evaluate_target(t, _wait_args())
    assert "final" not in t and t["unreadable_n"] == 1
    t["unreadable_since"] = time.time() - 31 * 60
    ops.evaluate_target(t, _wait_args())
    assert t["final"] == {"outcome": "unreadable", "code": 4}
    # BOM 付きは読める
    p.write_text(json.dumps({"status": "done"}), encoding="utf-8-sig")
    t = {"kind": "progress", "path": p, "t0": time.time()}
    ops.evaluate_target(t, _wait_args())
    assert t["final"] == {"outcome": "done", "code": 0}


def test_wait_reads_hung_children_from_bundle_status(ops, tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "_now", lambda: NOW)
    monkeypatch.setattr(ops, "_alive", lambda pid, ct=None: True)
    p = tmp_path / "bundle1_status.json"
    p.write_text(json.dumps({"status": "running", "pid": 1, "updated": _ts(0),
                             "hung_children": [{"job": "E7_EH", "current": 3, "elapsed_min": 40}]}), encoding="utf-8")
    t = {"kind": "progress", "path": p, "t0": time.time()}
    ops.evaluate_target(t, _wait_args())
    assert t["final"] == {"outcome": "hung_child", "code": 4} and t["hung"][0]["job"] == "E7_EH"


def test_wait_default_timeout_and_summary(ops, tmp_path, capsys):
    p = tmp_path / "progress.json"
    p.write_text(json.dumps({"status": "done", "condition": "K1"}), encoding="utf-8")
    assert ops.main(["wait", "--progress", str(p), "--interval", "0.01"]) == 0
    out = capsys.readouterr().out
    assert "時間切れ 1440.0 分" in out                                    # 時間切れの既定（24 時間）
    p.write_text(json.dumps({"status": "running", "pid": os.getpid(), "updated": "2999-01-01 00:00:00"}), encoding="utf-8")
    assert ops.main(["wait", "--progress", str(p), "--interval", "0.01", "--timeout-min", "0.001"]) == 5


def test_wait_pid_uses_create_time(ops):
    me = os.getpid()
    ct = ops._create_time(me)
    assert ct is not None and abs(ct - time.time()) < 3 * 24 * 3600
    assert ops._alive(me, ct) and not ops._alive(me, ct - 100)           # PID の使い回しを見分ける
    t = {"kind": "pid", "pid": me, "create_time": ct - 100}
    ops.evaluate_target(t, _wait_args())
    assert t["final"]["outcome"] == "pid_gone"


def test_dump_raises_when_target_is_held(ops, tmp_path):
    if os.name != "nt":
        pytest.skip("置き換え先を開いている間の失敗は Windows の動き")
    p = tmp_path / "x.json"
    p.write_text("{}", encoding="utf-8")
    with open(p, "rb"):
        with pytest.raises(OSError):
            ops._dump({"a": 1}, p, tries=2)
    ops._dump({"a": 1}, p)
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}


def test_backup_verify_does_not_skip_missing_hashes(ops, tmp_path):
    dest = tmp_path / "dest"
    (dest / "d").mkdir(parents=True)
    (dest / "d" / "f.txt").write_text("abc", encoding="utf-8")
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"groups": {"g": {"files": [{"path": "d/f.txt", "bytes": 3, "sha256": None}]}}}), encoding="utf-8")
    a = types.SimpleNamespace(manifest=str(man), dest=str(dest), quick=False)
    assert ops.cmd_backup_verify(a) == 1                                 # --no-hash の一覧でハッシュを照合したことにしない
    a.quick = True
    assert ops.cmd_backup_verify(a) == 0


# ================================================================ 97_s4_ledger_check.py（台帳の照合）
LEDGER = """# 種の台帳（試験）

## 照合用の対応表

| 開始 | 終了 | 状態 | 層 | 記録 | 備考 |
|---|---|---|---|---|---|
| 190000 | 199999 | 予定 | 帯 | なし:予定 | 段階 4 |
| 190200 | 190209 | 使用済み | 実績 | あり | 試験 |
| 44404 | 44404 | 使用済み | 実績 | なし:試験（記録は X2 の試験だけ） | smoke |
"""


def _ledger_root(tmp: pathlib.Path) -> pathlib.Path:
    (tmp / "docs").mkdir(parents=True)
    (tmp / "docs" / "ledger.md").write_text(LEDGER, encoding="utf-8")
    c = tmp / "outputs" / "v2eval" / "S4T" / "C1"
    c.mkdir(parents=True)
    (c / "trial_0000.json").write_text(json.dumps({"seed": 190200, "trial": 0}), encoding="utf-8")
    return tmp


def _ledger_main(mod, root, *extra):
    out = root / "ledger_check.json"
    code = mod.main(["--root", str(root), "--ledger", str(root / "docs" / "ledger.md"), "--out", str(out), *extra])
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_ledger_baseline_ok(ledger, tmp_path):
    root = _ledger_root(tmp_path)
    code, rep = _ledger_main(ledger, root)
    assert code == 0 and rep["summary"]["parse_errors"] == 0


def test_ledger_bom_string_seed_and_upper_ext(ledger, tmp_path):
    root = _ledger_root(tmp_path)
    c = root / "outputs" / "v2eval" / "S4T" / "C1"
    (c / "trial_0001.json").write_text(json.dumps({"seed": 190201}), encoding="utf-8-sig")       # BOM 付き
    (c / "trial_0002.json").write_text(json.dumps({"seed": "190202"}), encoding="utf-8")         # 文字列の種
    (c / "TRIAL_0003.JSON").write_text(json.dumps({"seed": 190203}), encoding="utf-8")           # 大文字の拡張子
    code, rep = _ledger_main(ledger, root)
    used = {s for r in rep["used_ranges"] for s in range(r["range"][0], r["range"][1] + 1)}
    assert {190200, 190201, 190202, 190203} <= used
    assert code == 0 and rep["summary"]["parse_errors"] == 0


def test_ledger_unreadable_record_is_a_problem(ledger, tmp_path):
    root = _ledger_root(tmp_path)
    d = root / "outputs" / "misc" / "x"
    d.mkdir(parents=True)
    (d / "trial_0005.json").write_bytes(b"\x00" * 128)                 # 書きかけ・0 埋め、種を拾う控えが無い
    code, rep = _ledger_main(ledger, root)
    assert code == 1 and rep["summary"]["parse_errors"] == 1 and rep["summary"]["unreadable_records"] == 1
    assert rep["problems"]["unreadable_records"][0]["path"].replace("\\", "/").endswith("misc/x/trial_0005.json")
    # 帯の確認だけでも「問題なし」にしない
    code, rep = _ledger_main(ledger, root, "--bands", "190300-190309", "--bands-only")
    assert code == 1 and rep["bands"][0]["unused"] is True and rep["ok"] is False
    (d / "trial_0005.json").unlink()
    code, rep = _ledger_main(ledger, root, "--bands", "190300-190309", "--bands-only")
    assert code == 0


def test_ledger_incomplete_seed_recovered_from_resume_spec(ledger, tmp_path):
    root = _ledger_root(tmp_path)
    c = root / "outputs" / "v2eval" / "S4T" / "C2"
    inc = c / "_incomplete_20261008-020000"
    inc.mkdir(parents=True)
    (inc / "trial_0002.json").write_bytes(b"\x00" * 256)                # 電源断で 0 埋め、退避された
    (c / "resume_spec.json").write_text(json.dumps({"trials": "natural:190204:2"}), encoding="utf-8-sig")
    code, rep = _ledger_main(ledger, root)
    assert code == 0 and rep["summary"]["parse_errors"] == 0 and rep["summary"]["parse_errors_seed_recovered"] == 1
    w = rep["warnings"]["unreadable_records_seed_recovered"][0]
    assert "resume_spec.json" in w["seeds_from"] and w["seed_ranges"] == [[190204, 190205]]


def test_ledger_incomplete_x2_part_seed_from_folder_name(ledger, tmp_path):
    root = _ledger_root(tmp_path)
    part = root / "outputs" / "s4" / "x2" / "gen" / "X2_gen" / "_incomplete_20261008-020000" / "part_44404"
    part.mkdir(parents=True)
    (part / "generation.jsonl").write_text('{"seed": 444', encoding="utf-8")       # 書きかけの行
    (part / "timing.json").write_bytes(b"")                                        # 空
    code, rep = _ledger_main(ledger, root)
    assert code == 0 and rep["summary"]["parse_errors"] == 0
    rec = rep["warnings"]["unreadable_records_seed_recovered"]
    assert len(rec) == 2 and all("part_44404" in r["seeds_from"] for r in rec)
    used = {s for r in rep["used_ranges"] for s in range(r["range"][0], r["range"][1] + 1)}
    assert 44404 in used


def test_ledger_jsonl_bad_line_outside_incomplete_is_a_problem(ledger, tmp_path):
    root = _ledger_root(tmp_path)
    d = root / "outputs" / "gen" / "G1"
    d.mkdir(parents=True)
    (d / "generation.jsonl").write_text('{"seed": 190206}\n{"seed": 1902', encoding="utf-8")
    code, rep = _ledger_main(ledger, root)
    assert code == 1 and "1 行" in rep["problems"]["unreadable_records"][0]["error"]


def test_ledger_short_table_row_is_reported(ledger, tmp_path):
    root = _ledger_root(tmp_path)
    led = root / "docs" / "ledger.md"
    led.write_text(LEDGER + "| 50000 | 50010 | 使用済み |\n", encoding="utf-8")
    code, rep = _ledger_main(ledger, root)
    assert code == 1 and any("列が 6 未満" in str(x.get("detail")) for x in rep["problems"]["ledger_inconsistency"])


# ================================================================ decompose_s4.py（計画役のキャッシュ）
RAW = json.dumps({"steps": ["blue", "red"], "reply": "青と赤を入れます"}, ensure_ascii=False)


class _FakeClient:
    def __init__(self, raw=RAW, before=None):
        self.raw, self.before, self.calls = raw, before, []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if self.before:
            self.before()
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=self.raw)], model=kw["model"],
                                     stop_reason="end_turn", usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))


@pytest.fixture
def D4(tmp_path, monkeypatch):
    from recovla.planner import decompose_s4 as m
    monkeypatch.setattr(m, "CACHE", tmp_path / "llm_cache")
    return m


def _key_path(D4, text="青と赤"):
    msg = D4.user_message(text, ["red", "green", "blue"], [])
    k = D4.cache_key(D4.DEFAULT_MODEL, msg)
    return k, D4.CACHE / f"{k}.json"


def test_cache_write_leaves_no_tmp_and_reads_bom(D4):
    out = D4.decompose("青と赤", ["red", "green", "blue"], [], cli=_FakeClient())
    k, p = _key_path(D4)
    assert out["valid"] and p.is_file() and not list(D4.CACHE.glob("*.tmp"))
    rec = json.loads(p.read_text(encoding="utf-8"))
    p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8-sig")       # BOM 付きでも読める
    again = D4.decompose("青と赤", ["red", "green", "blue"], [], cli=_FakeClient(raw="壊れた答え"))
    assert again["from_cache"] is True and again["steps"] == ["blue", "red"]


@pytest.mark.parametrize("broken", [b"", b"\x00" * 300, b'{"key": "x", "raw":', b'{"key": "other", "raw": "{}"}'])
def test_broken_cache_is_quarantined_and_refetched(D4, broken):
    k, p = _key_path(D4)
    D4.CACHE.mkdir(parents=True)
    p.write_bytes(broken)
    cli = _FakeClient()
    out = D4.decompose("青と赤", ["red", "green", "blue"], [], cli=cli)    # 例外にならない
    assert len(cli.calls) == 1 and out["valid"] and out["from_cache"] is False
    moved = list(D4.CACHE.glob(f"{k}.json.broken_*"))
    assert len(moved) == 1 and moved[0].read_bytes() == broken           # 消さずに退避
    assert json.loads(p.read_text(encoding="utf-8"))["key"] == k


def test_concurrent_first_writer_wins(D4):
    """2 つの枠が同時に初めて同じ鍵を呼んだ: 後の者は自分の答えを捨てて先の写しを使う（同じ入力の答えはキャッシュで保証）。"""
    k, p = _key_path(D4)
    first = json.dumps({"steps": ["red", "blue"], "reply": "赤と青を入れます"}, ensure_ascii=False)

    def other_process_writes():
        D4.CACHE.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"key": k, "raw": first, "from_cache": False, "model": "claude-haiku-5-5"},
                                ensure_ascii=False), encoding="utf-8")

    out = D4.decompose("青と赤", ["red", "green", "blue"], [], cli=_FakeClient(before=other_process_writes))
    assert out["steps"] == ["red", "blue"] and out["from_cache"] is True
    assert json.loads(p.read_text(encoding="utf-8"))["raw"] == first
    assert not list(D4.CACHE.glob("*.tmp"))


def test_use_cache_false_overwrites(D4):
    D4.decompose("青と赤", ["red", "green", "blue"], [], cli=_FakeClient())
    other = json.dumps({"steps": ["red", "blue"], "reply": "x"}, ensure_ascii=False)
    out = D4.decompose("青と赤", ["red", "green", "blue"], [], cli=_FakeClient(raw=other), use_cache=False)
    k, p = _key_path(D4)
    assert out["steps"] == ["red", "blue"] and json.loads(p.read_text(encoding="utf-8"))["raw"] == other


# ================================================================ 98_s4_d_rtc.py rotate・98_s4_d_recovery.py run の「済み」
class _FakeV82:
    def __init__(self, out, n):
        self.OUT, self.n = out, n

    def trial_list(self, trials):
        return [(44430 + i // 3, None, ("red", "green", "blue")[i % 3]) for i in range(self.n)]


def _seal(d: pathlib.Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text("{}", encoding="utf-8")
    (d / "G_AUDIT.json").write_text('{"met": true}', encoding="utf-8")


@pytest.fixture
def rtc(tmp_path, monkeypatch):
    m = _load(ROOT / "scripts" / "98_s4_d_rtc.py", "s4_d_rtc_fix_t")
    v82 = _FakeV82(tmp_path / "v2eval", 3)
    fake96 = types.SimpleNamespace(load_ops=lambda: None, load_82=lambda allow: v82,
                                   check_complete=lambda kind, out, i, expect: (True, "ok", {}),
                                   _write_atomic=lambda p, text: (p.write_text(text, encoding="utf-8"), True)[1])
    monkeypatch.setattr(m, "load96", lambda: fake96)
    monkeypatch.setattr(m, "patch96", lambda *a, **k: None)
    monkeypatch.setattr(m, "check_band", lambda *a, **k: None)
    monkeypatch.setattr(m, "OUT_DIAG", tmp_path / "d_rtc")
    m._calls = []
    return m


def _rot_args():
    return argparse.Namespace(settings="naive,ZEROS", trials="natural:44430:1", block_trials=None, block_seeds=1, tag="t",
                              experiment="S4SMOKE_A", rounds=0, allow_other_band=False)


def test_rotate_seals_conditions_whose_trials_are_complete(rtc, tmp_path, monkeypatch):
    """全試行がそろっていても run.json・G_AUDIT が無い設定には 96 を 1 回呼ぶ（重要 3）。"""
    def fake_run_condition(m, ops, v82, a, n, shadow, cond, trials, extra, max_new=None):
        rtc._calls.append((n, max_new))
        _seal(v82.OUT / a.experiment / n)
        return 0
    monkeypatch.setattr(rtc, "run_condition", fake_run_condition)
    _seal(tmp_path / "v2eval" / "S4SMOKE_A" / "naive")                   # naive は印あり、ZEROS は試行だけ
    assert rtc.cmd_rotate(_rot_args(), []) == 0
    assert rtc._calls == [("ZEROS", None)]                               # 試行は回さない（--max-new なし・todo 0）
    prog = json.loads((tmp_path / "d_rtc" / "rotate_t.progress.json").read_text(encoding="utf-8"))
    assert prog["status"] == "done" and prog["calls"][0]["status"] == "seal"
    rtc._calls.clear()
    assert rtc.cmd_rotate(_rot_args(), []) == 0 and rtc._calls == []     # そろっていれば呼ばない


def test_rotate_is_not_done_when_seal_fails(rtc, tmp_path, monkeypatch):
    monkeypatch.setattr(rtc, "run_condition", lambda *a, **k: 0)        # 0 を返したのに書けていない
    assert rtc.cmd_rotate(_rot_args(), []) == 2
    prog = json.loads((tmp_path / "d_rtc" / "rotate_t.progress.json").read_text(encoding="utf-8"))
    assert prog["status"] == "error" and "run.json・G_AUDIT.json" in prog["stop_reason"]


@pytest.fixture
def rec(tmp_path, monkeypatch):
    m = _load(ROOT / "scripts" / "98_s4_d_recovery.py", "s4_d_rec_fix_t")
    v82 = _FakeV82(tmp_path / "v2eval", 1)

    class _Prog:
        def __init__(self, path, ops, base):
            self.path, self.d = path, dict(base)

        def update(self, **kw):
            self.d.update(kw)
            self.path.write_text(json.dumps(self.d, ensure_ascii=False, default=str), encoding="utf-8")

        def start_heartbeat(self):
            pass

        def stop_heartbeat(self):
            pass

    fake96 = types.SimpleNamespace(load_ops=lambda: None, load_82=lambda allow: v82, Progress=_Prog, _now_s=lambda: "now",
                                   check_complete=lambda kind, out, i, expect: (True, "ok", {}),
                                   trial_paths=lambda kind, out, i: {"json": out / f"trial_{i:04d}.json"})
    monkeypatch.setattr(m, "load_96", lambda: fake96)
    monkeypatch.setattr(m, "patch_96", lambda r96, DR: r96)
    monkeypatch.setattr(m, "build_96_args", lambda r96, a, model, v, n, dry: types.SimpleNamespace(cond=f"{model}_{v}", n=n))
    monkeypatch.setattr(m, "OUT_S4", tmp_path / "d_recovery")
    m._calls = []
    return m


def _rec_args():
    return argparse.Namespace(trials="induced:44470:1", experiment="S4SMOKE_C", models="R1v3,N1v3", variants="fall_with_hold",
                              block=10, max_new=0, dry_run=False, progress_file=None, min_free_gb=12.0, min_commit_free_gb=6.0,
                              mem_timeout_min=120.0, quiet_window="", accept_env_change=False)


def test_recovery_seals_conditions_whose_trials_are_complete(rec, tmp_path, monkeypatch):
    def fake_call(r96, v82, ops, ns):
        rec._calls.append((ns.cond, ns.n))
        _seal(v82.OUT / "S4SMOKE_C" / ns.cond)
        return 0
    monkeypatch.setattr(rec, "_call_96", fake_call)
    _seal(tmp_path / "v2eval" / "S4SMOKE_C" / "R1v3_fall_with_hold")
    assert rec.cmd_run(_rec_args()) == 0
    assert rec._calls == [("N1v3_fall_with_hold", 0)]                    # 試行は 0 本、印の無い条件だけ
    rec._calls.clear()
    assert rec.cmd_run(_rec_args()) == 0 and rec._calls == []


def test_recovery_is_not_done_without_seal(rec, tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "_call_96", lambda r96, v82, ops, ns: 0)
    assert rec.cmd_run(_rec_args()) == 2
    prog = json.loads((tmp_path / "d_recovery" / "progress_S4SMOKE_C_R1v3-N1v3_fall_with_hold.json").read_text(encoding="utf-8"))
    assert prog["status"] == "error" and "run.json・G_AUDIT.json" in prog["error"]
