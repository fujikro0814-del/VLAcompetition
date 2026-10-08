"""段階 4 束 6 (iii) の VLM の試験台（src/recovla/vlm/、scripts/57_vlm_bench.py）。CPU だけ。ネットワーク・GPU・描画・outputs は使わない。
API は偽のクライアント（同期の messages.create と、Message Batches の create・retrieve・results）。

確かめること: 正解の規則（掴み損ね・落下・置き損ね・停滞・成功・誘発との突き合わせ・取りこぼしの場面）、描く時刻の寄せ方、
IK（描画なしの運動学だけ）、問いの組み立て（温度を送らない・思考と effort の形・構造化出力）、キャッシュ（同じ鍵は二度呼ばない）、
Batch の custom_id の突き合わせ（順不同・失敗・知らない custom_id）、費用の計算と予算、採点（2 つの実装の一致）。

使い方: .venv\\Scripts\\python.exe -m pytest -q tests\\test_vlm_bench.py -p no:cacheprovider
"""
import hashlib
import json
import random
import types

import numpy as np
import pytest

from recovla.sim import frames
from recovla.vlm import ask as A
from recovla.vlm import cost as C
from recovla.vlm import labels as L
from recovla.vlm import score as S

DT = 0.05
BOX = np.array([0.45, 0.25, 0.0])
REST = L.CUBE_REST_Z


# ---------------------------------------------------------------- 合成の記録
def _interp(keys, t):
    """keys: [(t, tip3, closed, cube3)]。t での値（線形補間、closed は直前のキー）。"""
    ts = [k[0] for k in keys]
    j = max(0, int(np.searchsorted(ts, t, side="right")) - 1)
    if j >= len(keys) - 1:
        k = keys[-1]
        return np.array(k[1], float), k[2], np.array(k[3], float)
    a, b = keys[j], keys[j + 1]
    u = (t - a[0]) / max(b[0] - a[0], 1e-12)
    tip = (1 - u) * np.array(a[1], float) + u * np.array(b[1], float)
    cube = (1 - u) * np.array(a[3], float) + u * np.array(b[3], float)
    return tip, a[2], cube


def make_z(keys, t_end, ci=0, fingers_moving=True):
    t = np.round(np.arange(0.0, t_end + 1e-9, DT), 6)
    n = len(t)
    tip = np.zeros((n, 3))
    gc = np.zeros(n, bool)
    cube = np.zeros((n, 3, 3))
    others = [np.array([0.35, -0.15, REST]), np.array([0.55, -0.15, REST])]
    for f, x in enumerate(t):
        tp, cl, cp = _interp(keys, x)
        tip[f], gc[f] = tp, cl
        cube[f, ci] = cp
        o = [k for k in range(3) if k != ci]
        cube[f, o[0]], cube[f, o[1]] = others
    lin = np.zeros_like(cube)
    lin[1:] = np.diff(cube, axis=0) / DT
    inbox = np.array([[frames.in_box(cube[f, k], BOX) for k in range(3)] for f in range(n)])
    fing = np.where(gc[:, None], 0.02, 0.04) * np.ones((n, 2))
    quat = np.tile(np.array([1.0, 0, 0, 0]), (n, 3, 1))
    return {"sim_time": t, "fingertip": tip, "gripper_closed": gc, "cube_pos": cube, "cube_quat": quat,
            "cube_linvel": lin, "cube_in_box": inbox, "fingers": fing, "ee_pos": tip + np.array([0, 0, 0.1]),
            "ee_quat": np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (n, 1)), "target": np.full(n, ci, np.int8)}


C0 = (0.40, -0.10, REST)


def scenario(kind: str):
    """掴む→持ち上げる→運ぶ→手を離れる（kind で変える）。返り値 (keys, t_end)。"""
    up = (0.40, -0.10, REST + 0.12)
    k = [(0.0, (0.40, -0.10, 0.25), False, C0), (2.0, C0, False, C0), (2.05, C0, True, C0), (3.0, up, True, up)]
    if kind == "success":                 # 箱の真上へ運び、下ろして止まり、開く → 箱の底へ
        over = (0.45, 0.25, 0.10)
        inside = (0.45, 0.25, frames.BOX_FLOOR_Z + 0.02)
        k += [(5.0, over, True, over), (6.0, over, True, over), (6.05, over, False, over), (6.25, over, False, inside),
              (12.0, (0.33, 0.15, 0.30), False, inside)]
    elif kind == "drop":                  # 運ぶ途中（水平に 0.15 m/s）で開く → 机へ
        a = (0.40, 0.05, REST + 0.12)
        b = (0.40, 0.20, REST + 0.12)
        land = (0.40, 0.065, REST)
        k += [(4.0, a, True, a), (4.05, (0.40, 0.0575, REST + 0.12), False, (0.40, 0.0575, REST + 0.12)),
              (4.25, (0.40, 0.0875, REST + 0.12), False, land), (5.0, b, False, land), (12.0, b, False, land)]
    elif kind == "misplace":              # 別の所へ運び、下りながら（0.1 m/s）開く → 机へ
        a = (0.55, -0.05, REST + 0.12)
        b = (0.55, -0.05, REST + 0.07)
        land = (0.55, -0.05, REST)
        k += [(4.5, a, True, a), (5.0, b, True, b), (5.05, (0.55, -0.05, REST + 0.065), False, (0.55, -0.05, REST + 0.065)),
              (5.25, (0.55, -0.05, REST + 0.06), False, land), (12.0, (0.55, -0.05, 0.2), False, land)]
    elif kind == "rim":                   # 箱の壁の上で止まって開く → 壁の上に残る
        over = (0.45, 0.33, 0.12)
        rim = (0.45, 0.33, frames.BOX_WALL_TOP_Z + 0.02)
        k += [(5.0, over, True, over), (6.0, over, True, over), (6.05, over, False, over), (6.25, over, False, rim),
              (12.0, (0.33, 0.15, 0.30), False, rim)]
    elif kind == "slip":                  # 閉じたまま運ぶ途中で滑り落ちる
        a = (0.40, 0.05, REST + 0.12)
        land = (0.40, 0.06, REST)
        k += [(4.0, a, True, a), (4.2, (0.40, 0.08, REST + 0.12), True, land), (12.0, (0.40, 0.20, REST + 0.12), True, land)]
    else:
        raise ValueError(kind)
    return k, 12.0


def miss_keys():
    off = (0.43, -0.10, REST)             # 3 cm ずれて閉じる → 持ち上がらない
    return [(0.0, (0.43, -0.10, 0.25), False, C0), (2.0, off, False, C0), (2.05, off, True, C0),
            (3.5, (0.43, -0.10, 0.15), True, C0), (8.0, (0.43, -0.10, 0.15), True, C0)], 8.0


def stall_keys():
    p = (0.40, -0.05, 0.10)
    return [(0.0, (0.40, -0.05, 0.25), False, C0), (2.0, p, False, C0), (9.0, p, False, C0)], 9.0


# ---------------------------------------------------------------- 正解の規則
@pytest.mark.parametrize("kind,want", [("success", "success"), ("drop", "drop"), ("misplace", "misplace"),
                                       ("rim", "misplace"), ("slip", "drop")])
def test_release_classes(kind, want):
    keys, te = scenario(kind)
    z = make_z(keys, te)
    rel = L.release_events(z, 0)
    assert len(rel) == 1
    assert rel[0]["cls"] == want, rel[0]
    assert abs(rel[0]["t"] - {"success": 6.05, "drop": 4.05, "misplace": 5.05, "rim": 6.05, "slip": 4.05}[kind]) < 0.051
    assert L.grasp_miss_events(z, 0) == []


def test_release_class_table():
    assert L.release_class(True, True, True, 0.0, 0.0) == "success"
    assert L.release_class(False, False, True, 0.0, 0.2) == "misplace"       # 縁の上
    assert L.release_class(False, True, False, -0.2, 0.0) == "drop"          # 閉じたまま滑った
    assert L.release_class(False, True, True, 0.05, 0.0) == "drop"           # 上がりながら
    assert L.release_class(False, True, True, -0.1, 0.1) == "misplace"       # 下りながら
    assert L.release_class(False, True, True, 0.0, 0.01) == "misplace"       # 止まって
    assert L.release_class(False, True, True, 0.0, 0.15) == "drop"           # 運ぶ途中


def test_grasp_miss_and_anchor():
    keys, te = miss_keys()
    z = make_z(keys, te)
    ev = L.trial_events(z, 0)
    gm = [e for e in ev if e["kind"] == "grasp_miss"]
    assert len(gm) == 1 and abs(gm[0]["t"] - 2.05) < 0.051
    assert abs(gm[0]["anchor"] - gm[0]["t"] - L.GRASP_ANCHOR_DELAY_S) < 1e-9
    assert L.release_events(z, 0) == []


def test_stall():
    keys, te = stall_keys()
    z = make_z(keys, te)
    st = L.stall_events(z, 0)
    assert len(st) == 1 and abs(st[0]["t"] - 2.0) < 0.051
    ev = L.trial_events(z, 0)
    assert [e["cls"] for e in ev] == ["stall"] and abs(ev[0]["anchor"] - st[0]["t"] - L.STALL_ANCHOR_S) < 1e-9


def test_success_cuts_later_events():
    keys, te = scenario("success")
    keys = keys + [(13.0, (0.33, 0.15, 0.30), False, keys[-1][3]), (20.0, (0.33, 0.15, 0.30), False, keys[-1][3])]
    z = make_z(keys, 20.0)
    ev = L.trial_events(z, 0)
    assert ev[-1]["cls"] == "success"                         # 成功の後の静止は停滞にしない
    assert not any(e["cls"] == "stall" for e in ev)


def _inf(t_end, start=0.6, step=0.6):
    return [{"i": i, "t_obs": round(start + i * step, 6), "state": [0.0] * 8 + [0.1 * k for k in range(7)] + [0.0, 0.0]}
            for i in range(int((t_end - start) / step) + 1)]


def test_resolve_time_snap_and_ik():
    t = np.round(np.arange(0, 10.0001, DT), 6)
    inf = _inf(5.0)
    r = L.resolve_time(2.0, inf, t)                           # 最も近い推論は 1.8（0.2 s）
    assert r["joints_source"] == "inference" and abs(r["t_render"] - 1.8) < 1e-9
    assert r["joints"] == [0.1 * k for k in range(7)]
    assert abs(t[r["frame"]] - r["t_render"]) < 1e-9
    r = L.resolve_time(8.0, inf, t)                           # 推論が無い（停止の後）→ IK、こまは求めた時刻
    assert r["joints_source"] == "ik" and abs(r["t_render"] - 8.0) < 1e-9 and r["ik_seed"] is not None


def _single(z, meta, idx=0):
    return {"meta": meta, "z": z, "inference": _inf(float(z["sim_time"][-1])), "index": idx}


def test_failure_items_induce_match_and_exclusions():
    keys, te = miss_keys()
    z = make_z(keys, te)
    meta = {"seed": 1, "target": "red", "success": False,
            "induce": {"kind": "P1", "established": True, "t_fire": 1.7, "t_established": 4.0, "info": {"t_closed": 2.05}}}
    its = L.failure_items(_single(z, meta), "X/A_P1", "A_P1")
    inc = L.included(its)
    assert len(inc) == 1 and inc[0]["label"] == "grasp_miss" and inc[0]["label_source"] == "induce"
    assert len(inc[0]["frames"]) == len(L.FRAME_OFFSETS)
    # 誘発が P2（落下）なのに規則の出来事が掴み損ねしかない → 合う出来事が無い（no_rule_match）
    meta2 = dict(meta, induce={"kind": "P2", "established": True, "t_fire": 2.0, "t_established": 2.6, "info": {}})
    its2 = L.failure_items(_single(z, meta2), "X/A_P2", "A_P2")
    assert any("no_rule_match" in x["exclude"] for x in its2)


def test_failure_items_conflict_is_excluded():
    keys, te = scenario("drop")
    z = make_z(keys, te)
    meta = {"seed": 1, "target": "red", "success": False,
            "induce": {"kind": "P3", "established": True, "t_fire": 3.9, "t_established": 4.5, "info": {}}}
    its = L.failure_items(_single(z, meta), "X/A_P3", "A_P3")
    x = next(x for x in its if x["label_source"] == "induce")
    assert x["label"] == "misplace" and x["rule_class"] == "drop" and "conflict" in x["exclude"]


def test_s4_variant_mapping():
    assert L.induce_class({"induce": {"kind": "P2", "variant": "fall_with_hold"}}) == "drop"
    assert L.induce_class({"induce": {"kind": "P1"}}, "R1v3_grasp_failure") == "grasp_miss"
    assert L.induce_class({"induce": {"kind": "P3"}}) == "misplace"


def _task_rec():
    """赤が t=5 から箱の中（持たれていない）。試み 0 は 10 s で時間切れ（retry）、試み 1 は 20 s で時間切れ → 取りこぼし。"""
    inside = (0.45, 0.25, frames.BOX_FLOOR_Z + 0.02)
    keys = [(0.0, (0.33, 0.15, 0.30), False, C0), (4.9, (0.33, 0.15, 0.30), False, C0),
            (5.0, (0.33, 0.15, 0.30), False, inside), (20.0, (0.33, 0.15, 0.30), False, inside)]
    z = make_z(keys, 20.0)
    meta = {"seed": 7, "t_end": 20.0,
            "steps": [{"step": 0, "color": "red", "t_start": 1.0, "judged_complete": False, "t_judge": None, "t_end": 20.0,
                       "attempts": [{"attempt": 0, "t_begin": 1.0, "t_judge": None}, {"attempt": 1, "t_begin": 12.0, "t_judge": None}]}],
            "returns": [{"step": 0, "attempt": 0, "kind": "retry", "t_begin": 10.0, "t_end": 12.0}]}
    rows = [{"t": x, "step": 0, "attempt": 0, "phase": "run", "ok": False, "held_s": 0.0, "box_pixels": 60, "wrist_in": 700,
             "depth_ok": True, "retreat_dist_m": 0.01} for x in np.arange(1.0, 20.0, 0.05)]
    return {"meta": meta, "z": z, "inference": _inf(19.8), "judge_rows": rows, "index": 3}


def test_completion_items_fn_scene_and_decisions():
    items, steps = L.completion_items(_task_rec(), "X/E7")
    assert steps[0]["fn_scene"] and [a["attempt"] for a in steps[0]["fn_attempts"]] == [0, 1]
    by = {x["anchor"]: x for x in items}
    assert by["start+1"]["label"] is False and by["start+1"]["executor"]["decision"] is None
    assert by["ref"]["label"] is True and by["ref"]["executor"]["decision"] is False and by["ref"]["scene"]["fn_point"]
    assert by["a0_end"]["label"] is True and by["a0_end"]["executor"]["decision"] is False and by["a0_end"]["scene"]["fn_point"]
    assert "window_out_of_record" in by["ref+1"]["exclude"]
    assert by["ref"]["executor"]["row"]["box_pixels"] == 60


def test_truth_excludes_held_cube():
    inside = (0.45, 0.25, frames.BOX_FLOOR_Z + 0.02)
    keys = [(0.0, inside, True, inside), (2.0, inside, True, inside)]
    z = make_z(keys, 2.0)
    tr = L.truth_in_box(z, 5, 0)
    assert tr["cube_in_box_raw"] and tr["held"] and not tr["in_box"]


# ---------------------------------------------------------------- IK（描画なし）
def test_ik_recovers_hand_pose():
    from recovla.vlm import render as R
    r = R.SceneRenderer(views=())
    q0 = np.array(R.HOME_Q) + np.array([0.2, 0.1, -0.1, 0.15, 0.05, -0.1, 0.2])
    cubes = np.array([[0.4, -0.1, REST], [0.5, -0.1, REST], [0.6, -0.1, REST]])
    quats = np.tile([1.0, 0, 0, 0], (3, 1))
    r.set_state(q0, [0.04, 0.04], cubes, quats)
    p, q = r.hand_pose()
    r.set_state(R.HOME_Q, [0.04, 0.04], cubes, quats)
    out = r.ik(p, q, seed=list(np.array(R.HOME_Q) + 0.05))
    assert out["ik_err_pos_m"] < 1e-3 and out["ik_err_rot_rad"] < 1e-2
    r.close()


# ---------------------------------------------------------------- 問い・キャッシュ・Batch（偽のクライアント）
def _msg(text, model, tin=500, tout=30):
    return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)], model=model, stop_reason="end_turn",
                                 usage=types.SimpleNamespace(input_tokens=tin, output_tokens=tout, cache_creation_input_tokens=None,
                                                             cache_read_input_tokens=None), _request_id="req_x")


class FakeClient:
    def __init__(self, answer="yes"):
        self.answer = answer
        self.calls, self.batch_reqs, self.retrieves = [], [], 0
        self.messages = types.SimpleNamespace(create=self._create,
                                              batches=types.SimpleNamespace(create=self._bcreate, retrieve=self._bretrieve,
                                                                            results=self._bresults))

    def _create(self, **kw):
        assert "temperature" not in kw and "extra_body" not in kw
        self.calls.append(kw)
        return _msg(json.dumps({"answer": self.answer, "reason": "r"}), kw["model"])

    def _bcreate(self, requests):
        self.batch_reqs = list(requests)
        return types.SimpleNamespace(id="msgbatch_1", processing_status="in_progress")

    def _bretrieve(self, bid):
        self.retrieves += 1
        return types.SimpleNamespace(processing_status="ended" if self.retrieves >= 2 else "in_progress", request_counts=None)

    def _bresults(self, bid):
        out = []
        for k, r in enumerate(self.batch_reqs):
            if k == 0:
                out.append(types.SimpleNamespace(custom_id=r["custom_id"], result=types.SimpleNamespace(type="errored", error={"type": "x"})))
            else:
                out.append(types.SimpleNamespace(custom_id=r["custom_id"], result=types.SimpleNamespace(
                    type="succeeded", message=_msg(json.dumps({"answer": "no", "reason": "r"}), r["params"]["model"], 400, 20))))
        out.append(types.SimpleNamespace(custom_id="kunknown", result=types.SimpleNamespace(type="succeeded", message=None)))
        random.Random(0).shuffle(out)
        return iter(out)


def _setup_images(tmp_path, n_items=3, task="completion", view="overhead256"):
    d = tmp_path / "images" / task / view
    d.mkdir(parents=True)
    man = {"items": {}}
    items = []
    for i in range(n_items):
        b = f"png-bytes-{i}".encode()
        (d / f"it{i}_f0.png").write_bytes(b)
        man["items"][f"it{i}"] = [{"file": f"it{i}_f0.png", "sha256": hashlib.sha256(b).hexdigest(),
                                   "state": {"t_render": 1.0}}]
        items.append({"task": task, "item_id": f"it{i}", "color": "green", "label": i % 2 == 0, "exclude": [],
                      "executor": {"decision": None}, "scene": {"fn_point": False}})
    return items, man


@pytest.mark.parametrize("model", list(A.MODEL_SETTINGS))
def test_build_params_forms(model):
    p = A.build_params("completion", model, [{"type": "text", "text": "q"}])
    assert "temperature" not in p
    assert p["output_config"]["format"]["type"] == "json_schema"
    if model == "claude-haiku-5-5":
        assert p["thinking"] == {"type": "disabled"} and "effort" not in p["output_config"]
    elif model == "claude-sonnet-5-5":
        assert p["thinking"] == {"type": "between_tools"} and p["output_config"]["effort"] == "low"
    else:
        assert "thinking" not in p and p["output_config"]["effort"] == "low"


def test_cache_key_depends_on_inputs(tmp_path):
    items, man = _setup_images(tmp_path, 2)
    f0, f1 = man["items"]["it0"], man["items"]["it1"]
    k0 = A.cache_key("completion", "claude-haiku-5-5", items[0], f0)
    assert k0 == A.cache_key("completion", "claude-haiku-5-5", items[0], f0)
    assert k0 != A.cache_key("completion", "claude-opus-5-5", items[0], f0)          # モデル
    assert k0 != A.cache_key("completion", "claude-haiku-5-5", items[0], f1)          # 画像
    assert k0 != A.cache_key("completion", "claude-haiku-5-5", dict(items[0], color="red"), f0)   # 問いの文
    assert len(A.custom_id(k0)) <= 64


def test_sync_cache_no_second_call(tmp_path):
    items, man = _setup_images(tmp_path, 3)
    qs = A.questions("completion", "overhead256", "claude-haiku-5-5", items, man)
    cli = FakeClient("yes")
    recs = A.ask_sync(qs, cli, cache_dir=tmp_path / "cache", out=tmp_path, log=None)
    assert len(cli.calls) == 3 and all(r["parsed"]["answer"] == "yes" for r in recs)
    assert all(r["usage"]["input_tokens"] == 500 and r["wall_s"] is not None for r in recs)
    content = cli.calls[0]["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[-1]["text"].startswith("質問: 緑")
    recs2 = A.ask_sync(qs, cli, cache_dir=tmp_path / "cache", out=tmp_path, log=None)
    assert len(cli.calls) == 3 and all(r["from_cache"] for r in recs2)


def test_image_sha_mismatch_raises(tmp_path):
    items, man = _setup_images(tmp_path, 1)
    (tmp_path / "images" / "completion" / "overhead256" / "it0_f0.png").write_bytes(b"changed")
    qs = A.questions("completion", "overhead256", "claude-haiku-5-5", items, man)
    with pytest.raises(RuntimeError):
        A.ask_sync(qs, FakeClient(), cache_dir=tmp_path / "cache", out=tmp_path, log=None)


def test_batch_submit_and_collect_by_custom_id(tmp_path):
    items, man = _setup_images(tmp_path, 4)
    qs = A.questions("completion", "overhead256", "claude-sonnet-5-5", items, man)
    cli = FakeClient()
    A.ask_sync(qs[:1], cli, cache_dir=tmp_path / "cache", out=tmp_path, log=None)       # 1 問はキャッシュ済み
    recs = A.submit_batches(qs, cli, batches_dir=tmp_path / "b", cache_dir=tmp_path / "cache", out=tmp_path, log=None)
    assert len(recs) == 1 and len(cli.batch_reqs) == 3                                    # キャッシュ済みは出さない
    assert all("temperature" not in r["params"] for r in cli.batch_reqs)
    res = A.collect_batch("msgbatch_1", cli, batches_dir=tmp_path / "b", cache_dir=tmp_path / "cache", poll_s=0,
                          log=None, sleep=lambda s: None)
    assert res["counts"]["succeeded"] == 2 and res["counts"]["errored"] == 1 and res["counts"]["unknown_custom_id"] == 1
    m = json.loads((tmp_path / "b" / "msgbatch_1.json").read_text(encoding="utf-8"))["map"]
    for cid, v in m.items():
        rec = A.cache_get(v["key"], tmp_path / "cache")
        if rec is not None:
            assert rec["item_id"] == v["item_id"] and rec["mode"] == "batch" and rec["custom_id"] == cid
            assert abs(rec["usd"] - C.cost_usd("claude-sonnet-5-5", {"input_tokens": 400, "output_tokens": 20}, batch=True)) < 1e-12
    errored = [cid for cid in m if A.cache_get(m[cid]["key"], tmp_path / "cache") is None]
    assert len(errored) == 1


def test_batch_plan_splits(tmp_path, monkeypatch):
    items, man = _setup_images(tmp_path, 5)
    qs = A.questions("completion", "overhead256", "claude-haiku-5-5", items, man)
    monkeypatch.setattr(A, "BATCH_MAX_REQUESTS", 2)
    assert [len(c) for c in A.batch_plan(qs, tmp_path)] == [2, 2, 1]


# ---------------------------------------------------------------- 費用
def test_cost_prices_and_batch():
    assert C.cost_usd("claude-haiku-5-5", {"input_tokens": 1e6, "output_tokens": 1e6}) == pytest.approx(0.60)
    assert C.cost_usd("claude-sonnet-5-5", {"input_tokens": 1e6, "output_tokens": 1e6}) == pytest.approx(12.0)
    assert C.cost_usd("claude-opus-5-5", {"input_tokens": 1e6, "output_tokens": 1e6}, batch=True) == pytest.approx(12.0)
    with pytest.raises(KeyError):
        C.cost_usd("claude-unknown", {})


def test_estimate_budget_and_calib():
    plan = [{"task": "failure", "view": "presentation", "view_out": (768, 432), "n_images": 4, "model": "claude-opus-5-5",
             "n_questions": 1000, "batch": True}]
    e = C.estimate(plan, budget=30.0)
    tok = e["rows"][0]["tokens_per_question"]
    assert tok["source"] == "formula"
    assert tok["input_tokens"] == int(np.ceil((4 * C.image_tokens_formula(768, 432) + C.TEXT_TOKENS["failure"]) * C.MARGIN_IN))
    assert e["within_budget"] == (e["total_usd"] <= 30.0)
    calib = {("claude-opus-5-5", "failure", "768x432"): {"input_tokens": 2000, "output_tokens": 200}}
    e2 = C.estimate(plan, calib, budget=1.0, spent=0.5)
    assert e2["rows"][0]["tokens_per_question"]["source"] == "smoke" and not e2["within_budget"]


# ---------------------------------------------------------------- 採点
def _rows_completion():
    def r(label, ans, dec=None, fnp=False, mode="batch", wall=None):
        return {"item_id": f"i{random.random()}", "task": "completion", "view": "v", "model": "m", "label": label,
                "answered": ans is not None, "answer": ans, "executor_decision": dec, "fn_point": fnp, "condition": None,
                "label_source": None, "usage": {"input_tokens": 100, "output_tokens": 10}, "usd": 0.001, "mode": mode,
                "wall_s": wall, "stop_reason": "end_turn", "model_returned": "m"}
    return [r(True, "yes", True), r(True, "no", False, True), r(True, "unsure"), r(False, "no", False), r(False, "yes"),
            r(False, "invalid"), r(True, "yes", mode="sync", wall=1.0), r(True, None), r(False, "no", mode="sync", wall=3.0)]


def test_score_completion():
    s = S.score_completion(_rows_completion())
    c = s["counts"]
    assert (c["tp"], c["fn"], c["fp"], c["tn"], c["unsure"], c["invalid"], c["n"]) == (2, 2, 1, 3, 1, 1, 8)
    assert s["dual_check"] and s["accuracy"]["k"] == 5 and s["miss_rate"]["n"] == 4
    assert s["decision_points"]["executor"] == {"miss": 1, "false_complete": 0}
    assert s["decision_points"]["vlm"]["miss"] == 1
    assert s["fn_points"]["n"] == 1 and s["fn_points"]["vlm_yes"] == 0
    assert s["latency_s_median"] == 2.0 and s["n_unanswered"] == 1


def test_score_failure_confusion():
    rows = []
    pairs = [("grasp_miss", "grasp_miss"), ("grasp_miss", "drop"), ("drop", "drop"), ("misplace", "drop"),
             ("success", "success"), ("stall", "invalid"), ("success", "grasp_miss")]
    for lab, a in pairs:
        rows.append({"item_id": "x", "task": "failure", "view": "v", "model": "m", "label": lab, "answered": True, "answer": a,
                     "executor_decision": None, "fn_point": False, "condition": "A_P1", "label_source": "rule",
                     "usage": {}, "usd": 0.0, "mode": "batch", "wall_s": None, "stop_reason": "end_turn", "model_returned": "m"})
    s = S.score_failure(rows)
    assert s["dual_check"] and s["accuracy"]["k"] == 3 and s["accuracy"]["n"] == 7
    assert s["per_class"]["grasp_miss"]["recall"]["rate"] == 0.5 and s["per_class"]["drop"]["precision"]["k"] == 1
    assert s["confusion"]["stall"]["invalid"] == 1 and s["invalid"]["k"] == 1
    assert s["failure_vs_success"]["k"] == 5         # 失敗を失敗のどれかと答えた 4 ＋ 成功を成功 1


def test_parse_answer_rejects_outside_enum():
    assert A.parse_answer("failure", json.dumps({"answer": "fell", "reason": ""})) is None
    assert A.parse_answer("completion", "not json") is None
    assert A.parse_answer("failure", json.dumps({"answer": "drop", "reason": "x"}))["answer"] == "drop"
