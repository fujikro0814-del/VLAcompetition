# inject — 失敗の注入と、途中からの保存（Step F）

本線が書く。支線（`cloud/inject-sweep`）は変えない。変えてほしいときは掲示板に ask を出す。
中身の正は `src/recovla/expert/inject.py` と `src/recovla/expert/generate.py` の `run_attempt`（このファイルは要約）。
数値はすべて `configs/default.yaml` の `inject`。

## 1. 使い方

```python
from recovla.expert import generate as G
from recovla.sim.rig import SimRig

rig = SimRig(render=False, cfg=cfg)                 # cfg を渡すと注入の値を差し替えられる（下の §4）
a = G.run_attempt(rig, G.EpisodeSpec(layout_seed, color, kind="A"), retry, None, False)
```

- `kind` は `"A"`（掴み損ね）・`"B"`（搬送中の落下）・`"C"`（置き外し）。`"n"`（通常）・`"h"`（引き継ぎ）は Step D のまま
- `run_dir=None, render=False` なら何も書かない（描画なし・物理だけ）。`run_dir` を渡すと成功だけを `<run_dir>/<kind>_<種>_<色>_r<回>/` に保存する
- 注入のパラメータは `seeds.inject_rng(layout_seed, color, retry)` から試みの開始時に全部引く（`inject.sample_params`）。台本のパラメータは `seeds.script_rng` から（Step D と同じ）
- 1 回の試みは、描画なしでこの PC の 1 プロセスで約 1.5〜3 秒

## 2. 注入の中身

| 種類 | 発動 | 注入 | 保存を始める時点（確定） |
|---|---|---|---|
| A | 台本が descend・close に入った | 把持点を世界の x（指は世界の y の向きに閉じる）へ ±`A.lateral_offset_m` ずらす、または閉じる高さを `A.raise_close_m` 上げる（`A.raise_ratio` で raise）→ 閉じる → 手先を `A.lift_m` 持ち上げる | 持ち上げ終わって手が止まり、指が閉じ切り（指の速さが `expert.close_settle_s` 続けて止まる）、手先の上がり ≥ `A.confirm.hand_rise_min_m`、立方体の上がり < `A.confirm.cube_rise_max_m` |
| B | 掴んで持ち上げ、carry に入った | 最初の carry の時点の箱の中心までの水平距離を r0 とし、距離が `B.min_dist_from_box_m + u·(r0 − B.min_dist_from_box_m)` を切ったら開く（u は一様。評価の P2 と同じ決め方） | 開いていて、指の開き > 立方体の幅＋`landing.finger_gap_margin_m`、立方体が指の間になく、机の上で静止（`landing.rest_speed` 未満が `landing.rest_hold_s`） |
| C | 台本が箱の上で開く指令を出した | 開かずに搬送の高さへ上がり、ずらし先（`scene.region` の一様。他の机上の立方体・箱の外寸から外接円どうしで `C.min_clearance_m`）の真上へ運び、解放の高さへ下ろして開く | B と同じ |

確定の後、最初の 10 Hz の境目（物理ステップが 50 の倍数）から記録を始め、台本を作り直して（同じパラメータ、時計は 0 から）立て直す。

## 3. 返り値（`run_attempt` の辞書のうち、Step F で足したもの）

| キー | 中身 |
|---|---|
| `success` | 立て直して箱に入れ、静止した |
| `failure` | `None`、または下の区分 |
| `inject.status` | `confirmed`・`not_effective`・`landing_invalid`・`natural_failure`（`armed`・`active` のまま時間切れもある） |
| `inject.reason` | `cube_followed`（A で立方体が付いてきた）、`timeout`、`no_rest`、`landing_fail` をカンマでつないだもの など |
| `inject.info` | 確定の時点の量。A: `hand_rise_m`・`cube_rise_m`・`finger_rest_s`。B: `r0_m`・`threshold_m`。B・C: `finger_gap_m`・`rest_s`・`drop_point`。着地の検査: `landing_ok`・`landing_fail`（`tilt`・`over_box`・`clearance`・`workspace`・`view`）・`tilt_deg`・`clearance_m`（相手ごと。`cube_<色>`・`box_wall_<xp|xn|yp|yn>`）・`yaw_rel_deg`（指に対する角、90° の対称性を除いて −45〜45） |
| `inject.params` | 引いたパラメータ（`a_mode`・`a_offset_m`・`a_lift_m`・`b_u`・`c_drop_xy`） |
| `t_record_start`・`record_start_step`・`recovery_duration_s` | 記録を始めた時刻・物理ステップ・そこからの所要時間 |

**結果の 4 区分**（手順書 Step F の 5。集計はこの対応で数える）:

| 区分 | 条件 |
|---|---|
| 注入が効かなかった | `failure == "inject_not_effective"` |
| 着地が不自然 | `failure == "inject_landing_invalid"` |
| 立て直しに失敗 | 確定したのに `success` でない（`failure` が `grasp`・`slip`・`place`・`timeout`） |
| 成功 | `success` |
| （区分の外） | `inject_natural_failure`・`natural_before_inject`（注入の前に自然に失敗した）、確定の前の `timeout`。件数だけ別に数える |

完了条件 1: 種類ごとに、成功 ÷（成功＋立て直しに失敗）≥ 90%。着地が不自然 ÷（注入が効いた数＝成功＋立て直しに失敗＋着地が不自然）≤ `landing.max_invalid_ratio`。成功を `|yaw_rel_deg| ≤ landing_orientation_split_deg` とそれ以外に分けても示す。

## 4. 値の差し替え（振るとき）

`config.load()` の辞書を深く複写し、`cfg["inject"][...]` を書き換えて `SimRig(render=False, cfg=cfg)` に渡す。`run_attempt` は `rig.cfg` を使う。`configs/default.yaml` そのものは本線だけが変える。

## 5. 支線の検査

`tests/test_f_units.py` の注入から保存・再生までの検査は、この PC で選んだ種の結果に依るので `windows` の目印を付けてある。支線は既定の実行（`-m "not render and not windows and not needs_outputs"`）で回す。
