# docs/interfaces/ — 本線と支線の境目の約束

本線（この PC）が書き、支線は変えない（B_提案書 §15.1）。変えてほしいときは掲示板に ask を出す。
支線の指示書は、その支線が使う文書がここに入った後に監督が掲示板に書く（掲示板 0005・0010）。

| 文書 | 中身 | 使う支線 |
|---|---|---|
| `trial_record.md` | 評価の 1 試行の記録の形（評価器が書き、指標の関数が読む） | cloud/metrics、cloud/figures |
| `results.md` | 結果の表（CSV）の形と、統計の関数の型 | cloud/metrics、cloud/figures |
| `runner.md` | 方策の実行器の時間の流れ（sync・naive・rtc）と、その部品の型 | cloud/runner |

## すべての文書に共通の決まり

- **単位**: 長さ m、時間 s、角度 rad（表に出すときだけ度）、速さ m/s。画素は整数
- **座標系**: MuJoCo の世界座標
  - x は腕の台座から前へ、y は左、z は上
  - 机の上面は z = 0（`sim.table_top_z`）
  - 箱の中心は (0.45, 0.25, 0)（`scene.box.pos`）。箱の成功の体積は中心から ±0.05・0 < z < 0.06
- **手先**: hand 体の原点（`ee_pos`）。指先の中心は、hand の z 軸の向きに 0.1034 m 先（`sim.fingertip_offset`）
- **手先参照位置 `x_des`**: 制御器の `desired_pos`。行動はこれの差分で定義する
- **時間の格子**: 物理 1 ステップは 0.002 s
  - 記録のこまは 20 Hz。こま i は物理ステップ 25i の時点で、こま 0 はリセット直後
  - 行動は 10 Hz。行動 k は物理ステップ 50k〜50k+49 に効く。行動 k の直前の観測は、こま 2k
- **行動**: 7 次元、`[dx, dy, dz, 0, 0, 0, g]`
  - dx・dy・dz は、その 0.1 s の間の x_des の変化 [m]（世界座標）
  - 回転の 3 つは常に 0
  - g は +1 で閉、−1 で開
- **色の並び**: 常に `("red", "green", "blue")`（`scene.colors` の順）。色ごとの配列の添字はこの順
- **四元数**: (w, x, y, z)
- **欠けた値**: 数値は NaN、ない出来事の時刻は `null`（JSON）または NaN（npz）
- **乱数**: 試行の種から名前つきの乱数列を作る（B_提案書 §9）。`STREAM_ID = {layout: 0, induce: 1, noise: 2, script: 3, inject: 4, order: 5}`
  - 関数は `src/recovla/common/seeds.py`（掲示板 0022 の案を 0025 で採った）: `seed_sequence(seed, name, *sub)`・`stream(seed, name)`・`streams(seed)`・`torch_seed(ss)`
  - 生成の台本と注入は `SeedSequence(layout_seed, spawn_key=(3, 色の添字, 作り直しの回数))` と `(4, …)`（`script_rng`・`inject_rng`）
- **支線の検査**: GPU・学習済みモデル・C:\VLA・ネットワークなしで通ること（B_提案書 §15.1）
  - 検査には pytest の目印を付ける（掲示板 0013 の 3。`pyproject.toml` に登録）: `render`（描画の文脈が要る）、`windows`（Windows 専用）、`needs_outputs`（Git 管理外の `outputs/` が要る）、`torch`（torch が要る）
  - 支線の既定の実行: `python -m pytest -m "not render and not windows and not needs_outputs"`（描画の検査を含めて一括で回すと、描画の文脈のない環境では process ごと止まる。掲示板 0012）
  - 支線が足す検査にも、当てはまる目印を付ける。目印のない検査は、どの OS・GPU なしでも通ること
  - 場面の同一性は、Windows ではバイト一致（`windows`）、どこでも数と主な配列の許容誤差内の一致（`tests/fixtures/scene_g0_reference.json`）で確かめる（掲示板 0013 の 4）
