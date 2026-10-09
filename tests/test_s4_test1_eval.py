"""テスト 1 の起動の入口（scripts/98_s4_test1_eval.py）と、解析の版の照合 (b)・案 B の G-P1 の同じ種の対（src/recovla/eval/test1.py・
scripts/98_s4_test1_b.py・scripts/98_s4_test1.py）の検査。CPU だけ。記録は合成、保存点は偽のファイル（評価・GPU・API は使わない）。

使い方: PYTHONPATH=src python -m pytest -q tests/test_s4_test1_eval.py -p no:cacheprovider
"""
import importlib.util
import json
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recovla.eval import test1 as T1  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


EV = _load(ROOT / "scripts" / "98_s4_test1_eval.py", "t1e_entry")
B = _load(ROOT / "scripts" / "98_s4_test1_b.py", "t1e_b")
CHK = _load(ROOT / "scripts" / "98_s4_test1.py", "t1e_check")
H = _load(ROOT / "tests" / "test_s4_test1.py", "t1e_helpers")          # 合成の記録の作り方（make_e7・make_p1・make_natural）

ENTRY_SHA, EXEC_SHA, RTC_MOD_SHA = "e" * 64, "x" * 64, "r" * 64
CK_SHA = {m: (str(i) * 64)[:64] for i, m in enumerate(EV.REGISTERED)}


# ---------------------------------------------------------------- 帯
def test_bands_follow_s4_gates_and_accept_exact_specs_only():
    assert EV.check_tables() == []
    for br in (1, 2, 3, 4):
        for c in EV.conditions(br, "ZEROS" if br in (1, 3) else None):
            assert EV.check_band(c, c["trials"], "S4T1", False) == "", c
    c = EV.find(EV.conditions(4), "P1_R1v3")
    assert "完全に一致" in EV.check_band(c, "induced:162000:99", "S4T1", False)
    assert "完全に一致" in EV.check_band(c, "induced:162001:100", "S4T1", False)
    assert "種類" in EV.check_band(c, "natural:162000:100", "S4T1", False)
    assert "S4T1" in EV.check_band(c, "induced:162000:100", "S4T1X", False)
    e7 = EV.find(EV.conditions(2), "E7_R1v3_v3")
    assert e7["trials"] == "160000:150" and "完全に一致" in EV.check_band(e7, "160000:200", "S4T1", False)   # e7_band_extended は偽
    nat = EV.find(EV.conditions(4), "nat_N1v3s1002")
    assert nat["trials"] == "natural:161000:33" and "完全に一致" in EV.check_band(nat, "natural:161000:66", "S4T1", False)
    g = json.loads(EV.GATES.read_text(encoding="utf-8"))
    for a in g["bands"]["allocations"]:
        if a["id"] == "test1_P1":
            a["range"] = [162000, 162049]
    assert EV.check_tables(g) and "合わない" in EV.check_band(c, "induced:162000:100", "S4T1", False, g)


def test_smoke_and_health_bands():
    c = EV.find(EV.conditions(4), "P1_N1v3")
    led = "| 44500 | 44502 | 使用済み | x |\n| 44600 | 44609 | 予定 | y |\n"
    assert EV.check_band(c, "induced:44700:2", "S4SMOKE_T1", True, ledger_text=led) == ""
    assert EV.check_band(c, "induced:44605:2", "S4SMOKE_T1", True, ledger_text=led) == ""          # 予定は使用済みではない
    assert "使用済み" in EV.check_band(c, "induced:44501:2", "S4SMOKE_T1", True, ledger_text=led)
    assert "X2" in EV.check_band(c, "induced:44410:2", "S4SMOKE_T1", True, ledger_text=led)
    assert "smoke・データの帯" in EV.check_band(c, "induced:44799:2", "S4SMOKE_T1", True, ledger_text=led)
    assert "S4SMOKE" in EV.check_band(c, "induced:44700:2", "S4T1", True, ledger_text=led)
    assert "allow-smoke" in EV.check_band(c, "induced:44700:2", "S4SMOKE_T1", False, ledger_text=led)
    hc = EV.conditions("health")
    assert [x["model"] for x in hc] == ["R1v3s1001", "N1v3s1001", "R1v3s1002", "N1v3s1002"]
    for spec in EV.HEALTH_SPECS:
        assert EV.check_band(hc[0], spec, "S4T1HC", False) == ""
    assert "S4T1HC" in EV.check_band(hc[0], "selection:190600:33", "S4T1", False)
    assert "完全に一致" in EV.check_band(hc[0], "selection:190600:32", "S4T1HC", False)


# ---------------------------------------------------------------- 枝
@pytest.mark.parametrize("branch, n_conds, trials", [(1, 18, 2240), (2, 15, 1842), (3, 15, 1790), (4, 12, 1392)])
def test_branch_conditions_match_registration(branch, n_conds, trials):
    rs = "ZEROS" if branch in (1, 3) else None
    conds = EV.conditions(branch, rs)
    assert len(conds) == n_conds and sum(EV.n_trials(c["trials"]) for c in conds) == trials == EV.BRANCHES[branch]["trials"]
    names = {c["cond"] for c in conds}
    e7 = {"E7_R1v3_cur", "E7_R1v3_v3", "E7_N1v3_v3"}
    rtc = {"nat_R1v3_rtc", "P1_R1v3_rtc", "P1_N1v3_rtc"}
    assert (e7 <= names) is (branch in (1, 2)) and not (e7 & names and branch in (3, 4))
    assert (rtc <= names) is (branch in (1, 3)) and not (rtc & names and branch in (2, 4))
    for c in conds:
        if c["rtc"]:
            assert c["rtc"] == "ZEROS" and c["kind"] == "run"
        if c["group"] == "e7":
            assert c["kind"] == "task" and c["arm"] in ("R1v3_cur", "R1v3_v3", "N1v3_v3")
    # 案 B の枝は種 1000 の naive の P1 も回す（副次の記述）
    assert {"P1_R1v3", "P1_N1v3"} <= names
    # layout・params が A・B の受け入れと合う（足りない条件がない）
    lay, params = EV.layout_params(branch, rs, {"entry_sha256": ENTRY_SHA})
    p = T1.params_with_defaults(params)
    assert B.fill(params)["plan"] == p["plan"] == ("B" if branch in (1, 3) else "A")
    have = {c[0] for c in T1.conditions(dict(lay, root="/nonexistent"), p)}
    assert set(T1.required_conditions(p)) <= have and len(have) == n_conds
    assert p["rtc_p1_n"] == (100 if branch in (1, 3) else 50) and p["e7_n"] == 150 and p["e7_band_extended"] is False
    with pytest.raises(SystemExit):
        EV.conditions(branch, None if rs else "ZEROS")                        # 案 B に設定が無い・案 A に設定がある


def test_example_layout_of_check_matches_entry_names():
    lay, _ = EV.layout_params(1, "ZEROS")
    ex = CHK.EXAMPLE_LAYOUT
    flat = lambda x: x if isinstance(x, str) else x["cond"]                  # noqa: E731
    assert {k: flat(v) for k, v in ex["e7"].items()} == {k: v["cond"] for k, v in lay["e7"].items()}
    assert {k: flat(v) for k, v in ex["rtc"].items()} == {k: v["cond"] for k, v in lay["rtc"].items()}
    for grp in ("p1", "natural"):
        for layer in ("1000", "1001", "1002"):
            assert {s: flat(ex[grp][layer][s]) for s in "RN"} == {s: lay[grp][layer][s]["cond"] for s in "RN"}


def test_rtc_setting_must_be_a_gate_r_candidate():
    assert EV.rtc_info("ZEROS")["mode"] == "rtc"
    with pytest.raises(SystemExit):
        EV.rtc_info("naive")
    with pytest.raises(SystemExit):
        EV.rtc_info("current_repro")


def test_96_argv_for_task_and_single():
    conds = EV.conditions(1, "range10_cap5")
    t = EV.build_96_argv(EV.find(conds, "E7_N1v3_v3"), "S4T1", "160000:150", 5)
    assert t[:9] == ["task", "--experiment", "S4T1", "--condition", "E7_N1v3_v3", "--model", "N1v3", "--trials", "160000:150"]
    for k, v in (("--planner", "s4"), ("--step-timeout-s", "30"), ("--task-time-limit-s", "200"), ("--exec-interval", "6"),
                 ("--max-new", "5")):
        assert t[t.index(k) + 1] == v
    assert "--no-safety" in t
    r = EV.build_96_argv(EV.find(conds, "P1_N1v3_rtc"), "S4T1", "induced:162000:100")
    assert r[r.index("--mode") + 1] == "rtc" and r[r.index("--induce") + 1] == "P1" and r[r.index("--time-limit-s") + 1] == "60"
    n = EV.build_96_argv(EV.find(conds, "nat_R1v3s1001"), "S4T1", "natural:161000:33")
    assert n[n.index("--mode") + 1] == "naive" and "--induce" not in n


# ---------------------------------------------------------------- 引数
def test_forbidden_flags_prefix_match():
    names = EV.FORBIDDEN_EXTRA
    for f in ("--induc", "--induce=P2", "--time-limit=30", "--model=N1v3", "--plan", "--step-t", "--task-time", "--no-s", "--text"):
        assert EV.forbidden_flag(f, names), f
    for f in ("--min-free-gb", "--min-free-gb=12", "--accept-env-change", "--accept-spec-change", "--quiet-window", "x", "--"):
        assert not EV.forbidden_flag(f, names), f
    assert EV.forbidden_flag("--max", EV.ROTATE_FORBIDDEN) and EV.forbidden_flag("--stop", EV.ROTATE_FORBIDDEN)
    assert EV.has_flag(["--accept-env"], "--accept-env-change", len("--accept-e"))
    assert not EV.has_flag(["--accept"], "--accept-env-change", len("--accept-e"))


def test_env_driver_warns_only_and_records_mismatch(tmp_path):
    cur = {"driver": "620.00", "torch": "2.11.0", "torch_cuda": "12.8", "os_build": "x"}
    mm = EV.env_mismatch(cur, {})
    assert list(mm) == ["driver"]                                                  # ドライバだけなら警告（止めない）
    d = tmp_path / "S4T1" / "P1_R1v3"
    d.mkdir(parents=True)
    (d / "trial_0000.json").write_text(json.dumps({"env": dict(cur, torch="2.12.0")}), encoding="utf-8")
    assert "P1_R1v3" in EV.env_mismatch(cur, EV.record_envs([d]))["records"]


# ---------------------------------------------------------------- 保存点
def _fake_ckpt(root, name, content=b"w", postcheck=True, last=None):
    base, run, wrap = EV.REGISTERED[name]
    d = root / base / run / "checkpoints" / "020000" / "pretrained_model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(content)
    if last is not None:
        ld = root / base / run / "checkpoints" / "last" / "pretrained_model"
        ld.mkdir(parents=True, exist_ok=True)
        (ld / "model.safetensors").write_bytes(last)
    if wrap:
        (root / wrap).mkdir(parents=True, exist_ok=True)
        (root / wrap / f"postcheck_{run}.json").write_text(json.dumps(
            {"train_config_only_seed_and_names": postcheck, "same_dataset_fingerprint": True, "exit_code": 0, "checkpoints_ok": True}),
            encoding="utf-8")


def test_ckpt_mapping_sha_and_expect(tmp_path):
    for m in EV.REGISTERED:
        _fake_ckpt(tmp_path, m, content=m.encode())
    v82 = types.SimpleNamespace(CKPT={m: EV.registered_path(m) for m in ("R1v3", "N1v3")})
    infos = {m: EV.ckpt_info(m, v82, root=tmp_path) for m in EV.REGISTERED}
    assert infos["R1v3s1001"]["path"] == ("outputs/s4/train/train_R1v3s1001_20261009-002457_20261009-002457/checkpoints/020000/"
                                         "pretrained_model")
    assert infos["R1v3s1001"]["postcheck"] == "outputs/s4/seed_wrap/postcheck_train_R1v3s1001_20261009-002457_20261009-002457.json"
    assert len({i["sha256"] for i in infos.values()}) == 6 and all(i["registered"] for i in infos.values())
    assert EV.ckpt_info("N1v3s1002", v82, expect={"N1v3s1002": infos["N1v3s1002"]["sha256"]}, root=tmp_path)
    with pytest.raises(SystemExit, match="掲示した値"):
        EV.ckpt_info("N1v3s1002", v82, expect={"N1v3s1002": "0" * 64}, root=tmp_path)
    with pytest.raises(SystemExit, match="smoke"):
        EV.resolve_ckpt("R1v3s1001", v82, {"R1v3s1001": "train_R1v3s1001_smoke_20261008-200000_20261008-200000"}, tmp_path)
    with pytest.raises(SystemExit, match="82 の CKPT"):
        EV.resolve_ckpt("R1v3", types.SimpleNamespace(CKPT={"R1v3": "outputs/train/train_R1v3_2099/checkpoints/020000/pretrained_model"}),
                        root=tmp_path)
    # 82 の写しの CKPT に足す（元の値と違えば止める）
    EV.inject_ckpt(v82, infos["R1v3s1001"])
    assert v82.CKPT["R1v3s1001"] == infos["R1v3s1001"]["path"]
    with pytest.raises(SystemExit, match="別の保存点"):
        EV.inject_ckpt(v82, dict(infos["R1v3s1001"], path="outputs/s4/train/other/checkpoints/020000/pretrained_model"))


def test_ckpt_postcheck_and_last(tmp_path):
    _fake_ckpt(tmp_path, "N1v3s1001", postcheck=False)
    with pytest.raises(SystemExit, match="後の点検"):
        EV.resolve_ckpt("N1v3s1001", root=tmp_path)
    _fake_ckpt(tmp_path, "R1v3s1002", content=b"a", last=b"a")
    assert EV.resolve_ckpt("R1v3s1002", root=tmp_path)["warnings"] == []
    _fake_ckpt(tmp_path, "N1v3s1002", content=b"a", last=b"b")
    assert "last" in EV.resolve_ckpt("N1v3s1002", root=tmp_path)["warnings"][0]
    with pytest.raises(SystemExit, match="掲示した値"):
        EV.check_expect_files({"executor_v3": "0" * 64}, EV.files_sha256())
    fs = EV.files_sha256()
    assert fs["configs/s4_gates.json"] == EV.POSTED_GATES_SHA256                   # LF にそろえた値が 0165 の掲示の値
    EV.check_expect_files({"entry": fs["scripts/98_s4_test1_eval.py"]}, fs)


def test_patches_restore_between_conditions():
    class Eng:
        def __init__(self, *a):
            pass

        def run_one(self, i, seed, lay, tgt):
            return {"seed": seed, "induce": {"kind": "P1"}, "diag": {"induce": None}}, {}, {}

        def run_one_task(self, i, seed):
            return {"seed": seed}, {}, {}

    r96 = types.SimpleNamespace(Engine=Eng, SPEC_KEYS=("a",), make_spec=lambda a: {"m": 1},
                                run_json_text=lambda a, kind, rows, *x, **k: json.dumps({"n": len(rows)}))
    EV.restore96(r96)
    info = {"branch": 1, "role": "rtc.p1", "rtc": {"setting": "ZEROS"}, "induce": "P1", "ckpt": {"path": "p", "sha256": "s"},
            "files_sha256": {"scripts/98_s4_test1_eval.py": ENTRY_SHA}}
    EV.wrap_t1(r96, info)
    meta, _, _ = r96.Engine().run_one(0, 162000, None, None)
    assert meta["t1"]["role"] == "rtc.p1" and meta["diag"]["induce"] == "P1"     # d_rtc の自然の値を誘発に直す
    assert r96.make_spec(None)["t1_rtc"] == "ZEROS" and r96.make_spec(None)["t1_entry_sha256"] == ENTRY_SHA
    EV.restore96(r96)
    assert r96.Engine is Eng and r96.make_spec(None) == {"m": 1}
    EV.wrap_t1(r96, dict(info, role="e7.cur", rtc=None, induce=None))
    assert r96.Engine().run_one_task(0, 160000)[0]["t1"]["role"] == "e7.cur"


# ---------------------------------------------------------------- rotate（ワーカーで分け合う）
def _conds():
    out = []
    for c in EV.conditions(2):
        if c["cond"] in ("E7_R1v3_cur", "E7_R1v3_v3", "P1_R1v3", "P1_N1v3"):
            out.append(dict(c))
    return out


def test_pick_next_alternates_by_block_and_skips_live():
    conds = _conds()
    done = {c["cond"]: 0 for c in conds}
    assert EV.pick_next(conds, done, set())["cond"] == "E7_R1v3_cur"
    assert EV.pick_next(conds, done, {"E7_R1v3_cur"})["cond"] == "E7_R1v3_v3"
    done["E7_R1v3_cur"] = 5                                                       # 1 塊（5 種）先に進んだ
    assert EV.pick_next(conds, done, {"E7_R1v3_v3"})["cond"] == "P1_R1v3"           # 先に進んだ腕はもう 1 塊は取らない
    done.update(E7_R1v3_v3=5, P1_R1v3=10, P1_N1v3=10)
    assert EV.pick_next(conds, done, set())["cond"] == "E7_R1v3_cur"               # 塊の番号が同じ（どれも 2 塊目）なら E7 が先
    done.update(E7_R1v3_cur=150, E7_R1v3_v3=150, P1_R1v3=100, P1_N1v3=100)
    assert EV.pick_next(conds, done, set()) is None


class _Prog:
    def __init__(self):
        import threading
        self.d, self.lock, self.writes = {"calls": []}, threading.RLock(), []

    def start(self):
        pass

    def stop(self):
        pass

    def put(self, **kw):
        self.d.update(kw)

    def write(self, p, text):
        self.writes.append(p)


class _Claims:
    def __init__(self):
        self.held = set()

    def take(self, name, worker):
        if name in self.held:
            return False
        self.held.add(name)
        return True

    def drop(self, name):
        self.held.discard(name)

    def live(self, name):
        return False


def test_shared_loop_runs_to_end_and_alternates(tmp_path):
    conds = _conds()
    done = {c["cond"]: 0 for c in conds}
    order = []

    def call(name):
        c = EV.find(conds, name)
        done[name] = min(EV.n_trials_of(c), done[name] + EV.block_trials(c))
        order.append(name)
        full = done[name] >= EV.n_trials_of(c)
        return (0 if full else 1), {"status": "done" if full else "stopped", "stop_reason": None if full else "max_new:5"}

    prog = _Prog()
    rc = EV.shared_loop(conds, lambda n: done[n], call, lambda n: False, _Claims(), 1, prog, tmp_path / "log.json", lambda: "")
    assert rc == 0 and all(done[c["cond"]] == EV.n_trials_of(c) for c in conds)
    assert order[:4] == ["E7_R1v3_cur", "E7_R1v3_v3", "P1_R1v3", "P1_N1v3"]
    assert prog.d["status"] == "done" and len(prog.d["calls"]) == len(order)


def test_shared_loop_stops_like_b1(tmp_path):
    conds = _conds()
    # 呼び出しで増えない → エラー
    with pytest.raises(RuntimeError, match="増えない"):
        EV.shared_loop(conds, lambda n: 0, lambda n: (1, {"stop_reason": "max_new:5"}), lambda n: False, _Claims(), 1, _Prog(),
                       tmp_path / "l.json", lambda: "")
    # 子がエラー → 止まる（終了コード 2）
    prog = _Prog()
    assert EV.shared_loop(conds, lambda n: 0, lambda n: (2, {"error": "x"}), lambda n: False, _Claims(), 1, prog,
                          tmp_path / "l.json", lambda: "") == 2 and prog.d["status"] == "error"
    # 子が max_new 以外の理由で止まった（止める合図）→ 1
    assert EV.shared_loop(conds, lambda n: 0, lambda n: (1, {"stop_reason": "stop_file:x"}), lambda n: False, _Claims(), 1, _Prog(),
                          tmp_path / "l.json", lambda: "") == 1
    # 止める合図 → 何も回さずに 1
    called = []
    assert EV.shared_loop(conds, lambda n: 0, lambda n: called.append(n), lambda n: False, _Claims(), 1, _Prog(),
                          tmp_path / "l.json", lambda: "stop_file:STOP") == 1 and not called
    # 全部ほかで回っている → 待つ（時間切れでエラー）
    slept = []
    with pytest.raises(RuntimeError, match="待っても"):
        EV.shared_loop(conds, lambda n: 0, lambda n: (0, {}), lambda n: True, _Claims(), 2, _Prog(), tmp_path / "l.json",
                       lambda: "", wait_s=10, max_idle_s=30, sleep=slept.append)
    assert slept == [10, 10, 10]


def test_claims_are_exclusive_and_stale_ones_are_dropped(tmp_path):
    alive = {"v": True}
    c1 = EV.Claims(tmp_path / "claims", lambda pid, ct: alive["v"])
    assert c1.take("P1_R1v3", 1)
    p = tmp_path / "claims" / "P1_R1v3.claim"
    data = json.loads(p.read_text(encoding="utf-8"))
    p.write_text(json.dumps(dict(data, pid=data["pid"] + 1)), encoding="utf-8")   # ほかのワーカーの占有に見せる
    assert c1.live("P1_R1v3") and not c1.take("P1_R1v3", 2)
    alive["v"] = False                                                            # そのワーカーが死んだ
    assert not c1.live("P1_R1v3") and not p.exists() and c1.take("P1_R1v3", 2)
    c1.drop("P1_R1v3")
    assert not p.exists()


def test_child_argv_passes_branch_and_rtc():
    a = types.SimpleNamespace(cmd="rotate", branch=1, rtc_setting="ZEROS", experiment="S4T1", run=["R1v3s1001=x"], expect=["entry=e"],
                              allow_smoke=False, trials=None, health_spec=EV.HEALTH_SPECS[0])
    argv = EV.child_argv(a, "P1_N1v3_rtc", ["--min-free-gb", "12"], 10)
    s = " ".join(argv)
    assert " run --branch 1 --rtc-setting ZEROS --one P1_N1v3_rtc --experiment S4T1 --max-new 10" in s
    assert "--run R1v3s1001=x" in s and "--expect entry=e" in s and argv[-2:] == ["--min-free-gb", "12"]
    h = EV.child_argv(types.SimpleNamespace(**dict(vars(a), cmd="health")), "health_R1v3s1001", [], 33)
    assert " health --health-spec selection:190600:33 --one health_R1v3s1001" in " ".join(h)
    ns, extra = EV.build_parser().parse_known_args(argv[2:])
    assert ns.cmd == "run" and ns.cond == "P1_N1v3_rtc" and ns.branch == 1 and extra == ["--min-free-gb", "12"]


# ---------------------------------------------------------------- 解析の版の照合 (b)
def _edit_all(d: pathlib.Path, fn):
    for p in sorted(list(d.glob("trial_*.json")) + list(d.glob("run_[0-9][0-9][0-9][0-9].json"))):
        m = json.loads(p.read_text(encoding="utf-8"))
        fn(m)
        p.write_text(json.dumps(m), encoding="utf-8")


def build_branch(tmp, branch=1, rtc_out=None, naive_p1_out=None, mark=True):
    """入口の layout の条件名で合成の記録を作り、入口が書く印（t1・b1・v3・diag）を足す。"""
    rs = "ZEROS" if branch in (1, 3) else None
    versions = {"entry_sha256": ENTRY_SHA, "executor_v3_sha256": EXEC_SHA if branch in (1, 2) else None,
                "rtc_setting": rs, "rtc_module_sha256": RTC_MOD_SHA if rs else None, "ckpt_sha256": dict(CK_SHA)}
    lay, params = EV.layout_params(branch, rs, versions)
    lay["root"] = str(tmp / "v2eval")
    base = tmp / "v2eval" / "S4T1"
    R, N = H.p1_pair_outcomes(13, 0, 10, 20)
    for c in EV.conditions(branch, rs):
        d = base / c["cond"]
        if c["group"] == "e7":
            H.make_e7(d, c["model"], 150, (lambda i: i < 40) if c["arm"] != "R1v3_cur" else (lambda i: i < 30), c["arm"] != "R1v3_cur")
        elif c["group"] == "p1":
            if c["cond"] == "P1_R1v3_rtc" and rtc_out is not None:
                out = rtc_out
            elif c["cond"] == "P1_R1v3" and naive_p1_out is not None:
                out = naive_p1_out
            else:
                out = R if c["model"].startswith("R") else N
            H.make_p1(d, c["model"], out)
        else:
            H.make_natural(d, c["model"], 161000 + (66 if c["trials"].endswith(":66") else 33) - 1, lambda i: i % 3 != 0)
        if not mark:
            continue

        def add(m, c=c):
            m["t1"] = {"role": c["key"], "files_sha256": {"configs/s4_gates.json": EV.POSTED_GATES_SHA256,
                                                          "scripts/98_s4_test1_eval.py": ENTRY_SHA},
                       "ckpt": {"path": EV.registered_path(c["model"]), "sha256": CK_SHA[c["model"]]},
                       "rtc": {"setting": c["rtc"], "module_sha256": RTC_MOD_SHA} if c["rtc"] else None}
            if c["group"] == "e7":
                m["b1"] = {"executor_version": "current" if c["arm"] == "R1v3_cur" else "v3.1",
                           "files_sha256": {"src/recovla/runtime/executor_v3.py": EXEC_SHA}}
            if c["rtc"]:
                m["diag"] = {"arm": c["rtc"]}
        _edit_all(d, add)
    return lay, params


def both(lay, params):
    a = T1.analyze(lay, params)
    b = B.run_b(lay, params)
    return a, b, CHK.compare(a, b)


@pytest.mark.parametrize("branch", [1, 2, 3, 4])
def test_versions_complete_and_a_equals_b(tmp_path, branch):
    lay, params = build_branch(tmp_path, branch)
    a, b, diffs = both(lay, params)
    assert diffs == [], diffs[:5]
    assert a["status"] == "complete", [p for c in a["checks"].values() for p in c["problems"]][:5]
    vc = a["version_check"]
    assert vc == {"done": True, "ok": True, "problems": []}
    assert a["checks"]["p1.1001.R"]["versions"]["ckpt_sha256"] == CK_SHA["R1v3s1001"]
    if branch in (1, 2):
        assert a["checks"]["e7.v3"]["versions"]["executor_v3_sha256"] == EXEC_SHA
    if branch in (1, 3):
        assert a["checks"]["rtc.p1_n"]["versions"]["rtc_setting"] == "ZEROS"
    assert "版の照合 (b): 満たす" in T1.summary_md(a)


def _set(path_fn):
    return path_fn


VERSION_BREAKERS = {
    "gates": ("P1_N1v3", lambda m: m["t1"]["files_sha256"].update({"configs/s4_gates.json": "dcf0dd4c" + "0" * 56}), "p1.1000.N"),
    "entry_mixed": ("nat_R1v3", lambda m: m["t1"]["files_sha256"].update({"scripts/98_s4_test1_eval.py": "f" * 64})
                    if m["trial"] == 7 else None, "natural.1000.R"),
    "v3_version": ("E7_R1v3_v3", lambda m: m["b1"].update(executor_version="v3.0") if m["run"] == 3 else None, "e7.v3"),
    "v3_settings": ("E7_N1v3_v3", lambda m: m["v3"].update(settings={"a_goal": "EH"}) if m["run"] == 0 else None, "e7.n1v3_v3"),
    "exec_sha": ("E7_R1v3_v3", lambda m: m["b1"]["files_sha256"].update({"src/recovla/runtime/executor_v3.py": "9" * 64}), "e7.v3"),
    "cur_version": ("E7_R1v3_cur", lambda m: m["b1"].update(executor_version="v3.1"), "e7.cur"),
    "rtc_name": ("P1_N1v3_rtc", lambda m: (m["t1"]["rtc"].update(setting="range10_cap5"), m["diag"].update(arm="range10_cap5")),
                 "rtc.p1_n"),
    "rtc_module": ("nat_R1v3_rtc", lambda m: m["t1"]["rtc"].update(module_sha256="1" * 64), "rtc.natural"),
    "naive_rtc_mark": ("P1_R1v3", lambda m: m.update(diag={"arm": "ZEROS"}) if m["trial"] == 5 else None, "p1.1000.R"),
    "ckpt_path": ("P1_R1v3s1002", lambda m: m["t1"]["ckpt"].update(
        path="outputs/s4/train/train_R1v3s1002_20261010-000000_20261010-000000/checkpoints/020000/pretrained_model"), "p1.1002.R"),
    "ckpt_sha": ("nat_N1v3s1001", lambda m: m["t1"]["ckpt"].update(sha256="7" * 64), "natural.1001.N"),
    "no_t1": ("P1_N1v3s1002", lambda m: m.pop("t1") if m["trial"] == 0 else None, "p1.1002.N"),
    "role": ("P1_N1v3s1001", lambda m: m["t1"].update(role="p1.1001.R"), "p1.1001.N"),
}


@pytest.mark.parametrize("breaker", sorted(VERSION_BREAKERS))
def test_version_check_failures_are_incomplete(tmp_path, breaker):
    lay, params = build_branch(tmp_path, 1)
    cond, fn, key = VERSION_BREAKERS[breaker]
    _edit_all(tmp_path / "v2eval" / "S4T1" / cond, fn)
    a, b, diffs = both(lay, params)
    assert diffs == [], diffs[:5]
    assert a["status"] == b["status"] == "incomplete" and a["primary"] is None and a["face_switch"] is None
    assert not a["checks"][key]["ok"] and not b["checks"][key]["ok"]
    assert a["version_check"]["done"] and a["version_check"]["ok"] is False
    assert "版の照合 (b): 満たさない" in T1.summary_md(a)


def test_version_check_needs_posted_values(tmp_path):
    lay, params = build_branch(tmp_path, 4)
    a, b, diffs = both(lay, dict(params, versions=None))           # 記録に t1 があるのに掲示の値が無い → 未完
    assert diffs == [] and a["status"] == "incomplete" and a["version_check"]["done"]
    a, b, diffs = both(lay, dict(params, versions=dict(params["versions"], entry_sha256="0" * 64)))
    assert diffs == [] and a["status"] == "incomplete"
    # 入口を通っていない合成の記録で params.versions も無いなら照らさない（今までの道具と同じ。1 枚に書く）
    lay2, params2 = build_branch(tmp_path / "plain", 4, mark=False)
    a, b, diffs = both(lay2, dict(params2, versions=None))
    assert diffs == [] and a["status"] == "complete" and a["version_check"] == {"done": False, "ok": None, "problems": []}
    assert "照らしていない" in T1.summary_md(a)
    # params.versions を書けば、t1 の無い記録は未完
    a, b, diffs = both(lay2, params2)
    assert diffs == [] and a["status"] == "incomplete"


def test_cross_condition_versions(tmp_path):
    """条件ごとには 1 種類でも、条件をまたいで入口の版・同じモデルの保存点が違えば未完。"""
    lay, params = build_branch(tmp_path, 4)
    _edit_all(tmp_path / "v2eval" / "S4T1" / "nat_N1v3", lambda m: m["t1"]["files_sha256"].update(
        {"scripts/98_s4_test1_eval.py": "f" * 64}))
    a, b, diffs = both(lay, dict(params, versions=dict(params["versions"])))
    assert diffs == [] and a["status"] == "incomplete" and a["version_check"]["problems"]


# ---------------------------------------------------------------- 案 B の G-P1（同じ種の対）
def _gp1_outcomes(c_only):
    """rtc（X）と naive（Y）の P1。対: 両方成立 20（両方成功）・c_only（naive だけ成功）・(30 − c_only)（どちらも失敗）、
    X だけ成立して成功 50（naive は不成立）。対の差は −c_only/50、各腕の分母の差は X 0.70 − Y (20+c_only)/50。"""
    X, Y = [], []
    for _ in range(20):
        X.append((5.0, 20.0)), Y.append((5.0, 20.0))
    for _ in range(c_only):
        X.append((5.0, None)), Y.append((5.0, 20.0))
    for _ in range(30 - c_only):
        X.append((5.0, None)), Y.append((5.0, None))
    for _ in range(50):
        X.append((5.0, 20.0)), Y.append((None, None))
    return X, Y


@pytest.mark.parametrize("c_only, mode, g_p1", [(4, "point", True), (5, "point", True), (6, "point", False), (2, "interval", False)])
def test_plan_b_g_p1_counts_same_seed_pairs(tmp_path, c_only, mode, g_p1):
    X, Y = _gp1_outcomes(c_only)
    lay, params = build_branch(tmp_path, 3, rtc_out=X, naive_p1_out=Y)
    a, b, diffs = both(lay, dict(params, guard_mode=mode))
    assert diffs == [], diffs[:5]
    c2 = a["face_switch"]["c2"]
    assert c2["g_p1_count"] == "paired" and c2["g_p1_pairs"] == 50 and (c2["g_p1_b"], c2["g_p1_c"]) == (0, c_only)
    assert c2["g_p1_diff"] == pytest.approx(-c_only / 50) and c2["g_p1"] is g_p1
    u = c2["g_p1_unpaired"]
    assert (u["imp_k"], u["imp_n"], u["base_k"], u["base_n"]) == (70, 100, 20 + c_only, 50)
    assert u["diff"] == pytest.approx(0.70 - (20 + c_only) / 50) and u["diff"] > 0          # 各腕の分母なら満たす側（記述だけ）
    lo, hi = c2["g_p1_ci95"]
    assert lo < c2["g_p1_diff"] <= hi


def test_plan_b_g_p1_without_pairs_fails_and_plan_a_is_structural(tmp_path):
    X = [(5.0, 20.0)] * 50 + [(None, None)] * 50
    Y = [(None, None)] * 50 + [(5.0, 20.0)] * 50                                  # 同じ種で両方成立した対が 0
    lay, params = build_branch(tmp_path, 3, rtc_out=X, naive_p1_out=Y)
    a, b, diffs = both(lay, params)
    assert diffs == [] and a["face_switch"]["c2"]["g_p1_pairs"] == 0 and a["face_switch"]["c2"]["g_p1"] is False
    lay, params = build_branch(tmp_path / "a", 4, naive_p1_out=[(None, None)] * 100)       # 案 A: 成立が 0 でも構造上 0
    a, b, diffs = both(lay, params)
    c2 = a["face_switch"]["c2"]
    assert diffs == [] and c2["g_p1_count"] == "structural" and c2["g_p1"] is True and c2["g_p1_pairs"] == 0


# ---------------------------------------------------------------- 入口（98_s4_test1.py）の条件をまたぐ点検
def test_entry_gates_check_and_child_scripts(tmp_path, monkeypatch):
    assert CHK.check_gates_now()["ok"]
    other = tmp_path / "g.json"
    other.write_bytes(EV.GATES.read_bytes().replace(b'"seeds": 66', b'"seeds": 67', 1))
    assert not CHK.check_gates_now(other)["ok"]
    crlf = tmp_path / "g_crlf.json"
    crlf.write_bytes(EV.GATES.read_bytes().replace(b"\n", b"\r\n"))
    assert CHK.check_gates_now(crlf)["ok"]                                          # LF にそろえて計算
    fake = types.SimpleNamespace(CHILD_SCRIPTS=("scripts/96_s4_resume.py",),
                                 order_heads=lambda root, heads: sorted(heads),
                                 compare_heads=lambda root, order, fam: {"changed": [], "added_only": []})
    monkeypatch.setattr(CHK, "_load_script", lambda name, mod: fake)
    assert CHK.check_heads({"a": 0, "b": 1})["ok"]
    assert set(CHK.ENTRY_CHILD_SCRIPTS) <= set(fake.CHILD_SCRIPTS) and "scripts/96_s4_resume.py" in fake.CHILD_SCRIPTS
    lay, params = build_branch(tmp_path / "r", 4)
    monkeypatch.setattr(CHK, "GATES", other)
    monkeypatch.setattr(CHK, "check_gates_now", lambda path=other: {"ok": False, "note": "違う"})
    a, b, diffs = both(lay, params)
    assert diffs == [] and a["status"] == "complete"
    led = tmp_path / "ledger.json"
    led.write_text(json.dumps({"problems": {"unreadable_records": []}}), encoding="utf-8")
    out = CHK.cross_audit(a, lay, str(led))
    assert out["status"] == "incomplete" and any(p.startswith("gates_sha256") for p in out["entry_audit"]["problems"])
