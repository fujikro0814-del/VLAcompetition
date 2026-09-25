# 0013 への対応（PR #1 の取り込み、検査の目印、場面の検査、物理の速さ）

- 番号: 0015 / 差出: main / 宛先: human、review / 日時: 2026-09-25 18:28（日本時間）
- 返信先: 0013
- 状態: 未読
- 関係する枝・PR・コミット: PR #1（cloud/setup-check、取り込みのコミット 8c74e0d）、main（このファイルを含むコミット）

## 要点（3行以内）

0013 の 1・3・4・5 を行った。PR #1 を main に取り込んだ。検査に目印 render・windows・needs_outputs・torch を付けた。
場面のバイト一致は windows の目印にし、OS によらない検査（数と主な配列の一致）を足した。物理の速さは STATUS に残した。
本線の検査は 103 件すべて合格。支線の既定の選び方では 77 件が対象（26 件を除外）。

## 本文

1. **PR #1 の取り込み**: `git merge --no-ff origin/cloud/setup-check`（8c74e0d）。変更は 0012・0013 と `scripts/cloud/bench_physics.py` の 3 ファイルだけであることを確かめた
3. **検査の目印**（`pyproject.toml` に登録。`docs/interfaces/README.md` の「支線の検査」に既定の実行を書き足した）

   | 目印 | 件数 | 検査 |
   |---|---|---|
   | render | 8 | `test_closed_loop_eval.py` の 3 件（`EvalRig` が描画器を作る）、`test_replay.py` の 5 件 |
   | windows | 12 | `test_push_check.py` の 10 件、`test_c_port.py` の `test_environment_matches_the_lock`・`test_scene_compiles_to_the_same_model_as_the_source` |
   | needs_outputs | 6 | `test_c_port.py` の `test_controller_and_recording_values_equal_the_legacy_raw_meta`・`test_training_placements_equal_the_legacy_evaluation`・`test_condition_1`〜`4` |
   | torch | 6 | `test_convert_to_lerobot.py` の 4 件、`test_vla_observation.py` の 1 件、`test_closed_loop_eval.py` の 1 件 |

   - 数は 0012 の分類と同じ（windows は 0012 の 11 件＋4 の場面の検査 1 件）。`pytest --co -m <目印>` で数えた
   - 既定の実行 `-m "not render and not windows and not needs_outputs"` は、この PC で 77 件合格・26 件除外。torch の 6 件は対象に残る（0013 の 2 で torch が入れば通る。0012 の参考の venv では 6 件とも合格していた）
4. **場面の検査**
   - バイト一致の検査に `windows` を付けた
   - OS によらない検査 `test_scene_compiles_to_the_reference_on_any_os` を足した。数（nq・nv・nu・nbody・njnt・ngeom・nmesh・ncam・nmocap・nkey）は完全一致、主な配列 35 種（体・関節・形状・駆動器・カメラ・メッシュの頂点数・keyframe）は整数は完全一致、実数は相対 1e-6・絶対 1e-9 の中
   - 参照値 `tests/fixtures/scene_g0_reference.json` は `scripts/04_scene_reference.py` がこの PC で書いた。書いたときの `mj_saveModel` の SHA-256 は `9a328745…`（流用元と同じ）で、検査はこの値も照合する
   - 要約の関数は `src/recovla/common/model_summary.py`
5. **物理の速さ**: STATUS の「計算機の目安」に、0012 の値（制御器あり 実時間の 5.9 倍、物理だけ 48.3 倍、4 CPU、1 process）を残した

検査: `.venv\Scripts\python.exe -m pytest -p no:cacheprovider` → 103 件合格（Step C の 102 件＋4 の 1 件）。

## 求めること

判断を求めるものではない（報告）。支線に渡す指示書で、既定の実行の命令（上の 3）を使ってほしい。
