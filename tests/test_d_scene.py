"""Step D の完了条件（手順書 §5）。重い測定は scripts/11_d_checks.py が行い、outputs/d/*.json に結果を書く。
ここはその結果を読んで判定する（無ければ、何を実行すればよいかを示して落ちる）。"""
import json

import pytest

from recovla.common import config

pytestmark = pytest.mark.needs_outputs
CFG = config.load()
OUT = config.path(CFG["paths"]["outputs"]) / "d"
MIN_PER_KIND = 100
MIN_FIRST_TRY = 0.95


def _need(name: str) -> dict:
    p = OUT / f"{name}.json"
    if not p.is_file():
        pytest.fail(f"{p} がない。先に scripts/11_d_checks.py {name} を実行する")
    return json.loads(p.read_text(encoding="utf-8"))


def test_scene_premises():
    """場面の前提: 色が設定どおり、発表用カメラを足しても記録用カメラの描画が変わらない、俯瞰カメラに全部写る。"""
    r = _need("scene")
    assert r["colors_match_config"] and r["presentation_camera_matches_config"]
    assert r["presentation_removed_ncam"]["with"] == r["presentation_removed_ncam"]["without"] + 1
    assert r["recording_render_max_diff_with_vs_without_presentation"] == 0 and r["states_compared"] >= 9
    assert r["visibility"]["worst_visible_fraction"] == 1.0


def test_condition_1_script_success_per_layout_kind():
    r = _need("physics")
    assert set(r["by_kind"]) == {"empty", "prefilled_1", "prefilled_2"}
    for kind, row in r["by_kind"].items():
        assert row["episodes"] >= MIN_PER_KIND, kind
        assert row["first_try_success"] / row["episodes"] >= MIN_FIRST_TRY, (kind, row)
        assert "retries_histogram" in row and "failures_by_kind" in row          # 作り直しの内訳
        assert row["saved"] + row["dropped"] == row["episodes"]


def test_condition_2_pairs_first_frame_pixel_identical():
    r = _need("pairs")
    assert r["pairs"]["empty"] >= 10 and r["pairs"]["prefilled_1"] >= 5
    assert set(r["runs"]) == {"1", "8"}                                          # 並列 1 と 8 の両方（R6）
    assert r["first_frame_pixel_identical_all"]
    assert r["first_frame_state_bit_identical_in_pairs"]
    assert r["arrays_identical_workers_1_vs_8"]
    ri = r["render_invariant_vs_physics_run"]
    assert ri["identical"] == ri["episodes"] > 0                                 # 描画の有無で物理が変わらない


def test_condition_3_handover():
    r = _need("handover")
    assert r["trials"] >= 50
    assert r["first_try_rate"] >= MIN_FIRST_TRY                                   # 1 と同等（同じ基準）
    assert len(r["phase_at_handover"]) >= 5                                        # 引き継ぎの時点が段階に散らばる


def test_condition_4_replay_five_random_episodes():
    r = _need("replay")
    assert len(r["episodes"]) == 5 and r["tol_m"] == 0.005
    assert r["window_pass_all"] and r["step_bit_exact_all"] and r["negative_controls_detected_all"]


def test_condition_5_conversion():
    r = _need("convert")
    assert r["convert_exit"] == 0 and r["verify_exit"] == 0 and r["verify_pass"] and not r["checks_fail"]
    assert r["task_color_checks"] == r["episodes"] > 0                             # 全エピソードで task と目標の色が一致
    assert r["flags_carried"]
    assert r["two_views_value_level"] >= r["episodes"] + 2                         # 各本の画像照合＋視点ごとの識別
    assert r["lerobot_actions_replay"]["all_within_tol"]


def test_condition_6_no_contact_in_300_episodes():
    r = _need("physics")
    assert r["episodes_total"] >= 300
    assert r["contact_episodes_total"] == 0
    assert all(v["contact_episodes"] == 0 for v in r["episodes_with_obstacle_contact"].values())
    assert r["min_dist_per_frame_m"] and r["min_dist_by_phase_m"]                  # 分布（全体と段階ごと）を示す
    assert r["min_dist_per_episode_m"]["p0"] > 0.0


def test_condition_7_generation_time_and_throughput():
    r = _need("throughput")
    ws = [row["workers"] for row in r["rows"]]
    assert ws == [1, 4, 8, 12]
    assert all(row["episodes_per_hour"] > 0 and row["per_episode_wall_s"] for row in r["rows"])


def test_condition_8_representative_videos():
    r = _need("videos")
    kinds = [p["kind"] for p in r["pairs"]]
    assert kinds.count("empty") == 3 and kinds.count("prefilled_1") == 1
    assert len(r["videos"]) == 3 * 3 + 2
    for v in r["videos"]:
        assert (config.ROOT / v).is_file()
