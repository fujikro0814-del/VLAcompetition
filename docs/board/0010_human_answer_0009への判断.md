# 0009 への判断（Step C 承認）

- 番号: 0010 / 差出: human / 宛先: main / 日時: 2026-09-25 17:57（日本時間。チャットでの指示を本線が書き写した）
- 返信先: 0009
- 状態: 対応中
- 関係する枝・PR・コミット: なし

## 要点（3行以内）

Step C を承認する。d は (c)（Step G の最初に RTC ありで測って決める）。
Step D の前に docs/interfaces/ の 3 本を main に入れ、入ったら cloud/metrics・cloud/runner を始める。
pytest に faulthandler_timeout = 300。掲示板の既存ファイルは以後直接直さない。

## 本文

Step C を承認する（0009 への回答）。Step D に進む前に次を行うこと。

1. d の決め方：(c) を採る。Step G の最初に RTC ありで推論を 50 回以上（最初の 1 回を除く）測り、d = ceil(95 パーセンタイル / 0.1 秒) とする。s=10 は変えない。d が 5 を超えたら止めて報告する。推論時間の内訳（前処理・VLM の前半・10 段の生成）は測って記録するだけにし、高速化はしない。
2. Step D の前に docs/interfaces/ の trial_record.md・results.md・runner.md を main に入れる（提案書 §15.4。データの形・関数の型・単位・座標系）。これが入ったら支線 cloud/metrics・cloud/runner を始める。
3. pytest の設定に faulthandler_timeout = 300 を入れる（止まったときにスタックを残すため）。
4. 掲示板の既存ファイルは以後直接直さない。訂正は新しい番号で出す（STATUS.md を除く）。

## 求めること

本線は 1〜4 を行い、終わったら報告する。
