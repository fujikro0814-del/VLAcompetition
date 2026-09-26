# K1 の閉ループの映像（2026-09-26）

決裁の指示（チャット、2026-09-26）で、確認用の映像を GitHub に入れる。元は `outputs/eval/`（Git 管理外）。
俯瞰と手首を横に並べた、方策の入力と同じ画像の mp4（20 fps）。閉ループ 30 回のうち先頭の 6 回（種 191000〜191005）。
条件はどちらも 50 手・同期、評価器の TF32 は行列積なし（掲示板 0042）。

| ファイル | モデル | 種 | 目標 | 結果 | 届いた段階 | 持ち上げた立方体 | 接触（こま） |
|---|---|---|---|---|---|---|---|
| `K1_first_closed_trial_0000_raw.mp4` | 最初の K1（行動エキスパートのみ、90 本） | 191000 | red | 失敗 | 接近 | — | 1 |
| `K1_first_closed_trial_0001_raw.mp4` | 同 | 191001 | green | 失敗 | 接近 | — | 0 |
| `K1_first_closed_trial_0002_raw.mp4` | 同 | 191002 | blue | 失敗 | 接近 | — | 2 |
| `K1_first_closed_trial_0003_raw.mp4` | 同 | 191003 | red | 失敗 | 接近 | blue（違う色） | 2 |
| `K1_first_closed_trial_0004_raw.mp4` | 同 | 191004 | green | 失敗 | 接近 | — | 5 |
| `K1_first_closed_trial_0005_raw.mp4` | 同 | 191005 | blue | 失敗 | 接近 | — | 0 |
| `K1_lora_closed_trial_0000_raw.mp4` | K1 の LoRA の枝（0040。300 本） | 191000 | red | 失敗 | 接近 | — | 0 |
| `K1_lora_closed_trial_0001_raw.mp4` | 同 | 191001 | green | 失敗 | 接近 | — | 0 |
| `K1_lora_closed_trial_0002_raw.mp4` | 同 | 191002 | blue | 失敗 | 接近 | — | 1 |
| `K1_lora_closed_trial_0003_raw.mp4` | 同 | 191003 | red | 失敗 | 接近 | — | 1 |
| `K1_lora_closed_trial_0004_raw.mp4` | 同 | 191004 | green | 失敗 | 接近 | — | 1 |
| `K1_lora_closed_trial_0005_raw.mp4` | 同 | 191005 | blue | 失敗 | 接近 | — | 3 |

| `K1_cue_closed_trial_0000_raw.mp4` | 手がかりつきの K1（0048。状態 18 次元、行動エキスパートのみ、300 本） | 191000 | red | **成功**（10.6 s） | 退避 | red | 0 |
| `K1_cue_closed_trial_0001_raw.mp4` | 同 | 191001 | green | **成功**（9.9 s） | 退避 | green | 0 |
| `K1_cue_closed_trial_0002_raw.mp4` | 同 | 191002 | blue | **成功**（10.2 s） | 退避 | blue | 0 |
| `K1_cue_closed_trial_0003_raw.mp4` | 同 | 191003 | red | **成功**（11.1 s） | 退避 | red | 0 |
| `K1_cue_closed_trial_0004_raw.mp4` | 同 | 191004 | green | **成功**（11.9 s） | 退避 | green | 0 |
| `K1_cue_closed_trial_0005_raw.mp4` | 同 | 191005 | blue | **成功**（11.1 s） | 退避 | blue | 0 |

出所: `outputs/k1/closed.json`・`closed_lora.json`・`closed_cue.json`。結果の全体は `docs/E_報告.md` と掲示板 0037・0046・0051。
