# 台本の担当（Step D）

- 番号: 0014 / 差出: human / 宛先: main / 日時: 2026-09-25 18:28（日本時間。チャットでの回答を本線が書き写した。回答の時刻は 18:10〜18:25 の間）
- 返信先: なし（本線がチャットで尋ねた。提案書 §15.4 では台本は支線 cloud/expert の予定だったが、枠は cloud/metrics・cloud/runner で埋まっている）
- 状態: 対応中
- 関係する枝・PR・コミット: なし

## 要点（3行以内）

台本（src/recovla/expert/）は本線が書く。cloud/expert は作らない。
条件: phase_of を純関数にし、生成の枠は描画を切って物理だけでも回せるようにする（Step F の注入の調整を支線 cloud/inject-sweep に出すため）。
cloud/metrics・cloud/runner の指示書は監督が作成済みで、人がクラウドに渡す。runner は download.pytorch.org の許可の後。

## 本文

本線の問い（チャット）: 台本は誰が書くか。(1) 本線が書く、(2) 支線の枠が空くまで待つ、(3) cloud/expert を先にする。

決裁の回答（2 回。同じ内容）:

1. (1) 本線が書く
2. 条件: `phase_of` を純関数にし、生成の枠は描画を切って物理だけでも回せる形にする。Step F の注入の調整を、枠が空いた時点で支線 `cloud/inject-sweep` に出すため
3. `cloud/metrics`・`cloud/runner` の指示書は監督が作成済みで、人がクラウドに渡す
4. `cloud/runner` は、人がクラウドの環境に download.pytorch.org を許可してから始める（0013 の 2 と同じ）

## 求めること

本線は Step D で台本を 2 の条件で書く。

（本線の理解。違っていれば知らせてほしい）`docs/interfaces/expert.md` は書かない（cloud/expert がないため）。
`cloud/inject-sweep` の境目の文書は、Step F で注入を作るときに本線が書き、main に入れてから監督が指示書を書く（§15.4 と同じ順）。
