"""提出用リポジトリ（recovery-vla-panda）の中身を書き出す（0099 の 3-3・0101 の 3・4）。このスクリプト自体は書き出さない。

    .venv\\Scripts\\python.exe scripts\\70_export_submission.py export --dest C:\\PAI\\recovery-vla-panda
    .venv\\Scripts\\python.exe scripts\\70_export_submission.py scan --dest C:\\PAI\\recovery-vla-panda   # 残った語の一覧
    .venv\\Scripts\\python.exe scripts\\70_export_submission.py youtube [--github-url https://github.com/<ユーザー名>/recovery-vla-panda]
        # 動画の確かめ（長さ 1〜3 分・場面の倍速が 2 倍まで）と、YouTube のタイトル・概要欄の下書き（outputs/submission/youtube_*）
    .venv\\Scripts\\python.exe scripts\\70_export_submission.py package --legacy   # 10/8 版で不要（Zip の提出はなくなった）。既定では使わない

- 提出（課題の 10/8 版）: omnicampus に説明資料の PDF と GitHub のリポジトリの URL（public）の 2 つ。動画は自分の YouTube に
  上げて URL を出す（Zip はなくなった）。手順は docs/stage4/submission_1008.md
- push は scripts\\push_submission.ps1 から（検査 scripts\\check_submission.py に通ったときだけ）
- README.md は原稿 paper/README_submission.md に説明資料と同じ値（paper/build/values.json）を差し込んで作る。LICENSE（MIT）も
  ここで作る。著作権者は omnicampus のアカウント名（ACCOUNT。60_paper.py・62_video.py と同じ値。本名を入れる欄は作らない）で、
  ACCOUNT が仮の値のままなら export と youtube は止まる。THIRD_PARTY_NOTICES.md だけは書き出し先で手で書く

- 入れるもの: INCLUDE の一覧（コード・設定・検査・場面と説明資料の原稿・環境の作り方・凍結のハッシュの一覧）
- 入れないもの: 掲示板・作業記録（docs の大半）・卒研や C:\\VLA に触れるもの（取り込みと照合のスクリプト、予備実験の
  場面と検査）・push の道具・outputs・models・鍵
- 文の置き換え: REPLACE を順にかける（掲示板の番号・内部の文書への参照を消す、流用元を中立の書き方に、開発中の語を
  分野の普通の語に）。置き換えの件数をファイルごとに数えて export_report.json に書く
"""
import argparse
import json
import pathlib
import re
import shutil
import subprocess

from recovla.common import config

ROOT = config.ROOT
PAPER_PY, VIDEO_PY = ROOT / "scripts" / "60_paper.py", ROOT / "scripts" / "62_video.py"
PLACEHOLDER = "アカウント名"                   # 60_paper.py・62_video.py の ACCOUNT の仮の値（check_submission.py も読む）


def _const(path: pathlib.Path, name: str):
    """スクリプトを読み込まずに、モジュールの一番上の `name = "..."` の値を取る（60_paper・62_video の ACCOUNT・FORBIDDEN）。"""
    import ast
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"{path.name} に {name} がない")


# omnicampus のアカウント名（説明資料・動画のファイル名、YouTube のタイトル、LICENSE の著作権者に入る）。60_paper.py の値を使う
ACCOUNT = _const(PAPER_PY, "ACCOUNT")


def account_problems(acc_p: str, acc_v: str, allow_placeholder: bool = False) -> list:
    """60_paper.py と 62_video.py の ACCOUNT が同じで、仮の値でないこと。問題の一覧を返す。"""
    problems = []
    if acc_p != acc_v:
        problems.append(f"60_paper.py と 62_video.py の ACCOUNT が違う: {acc_p!r} / {acc_v!r}")
    if not (acc_p or "").strip():
        problems.append("ACCOUNT が空")
    if PLACEHOLDER in (acc_p, acc_v) and not allow_placeholder:
        problems.append(f"ACCOUNT が仮の値 {PLACEHOLDER!r} のまま（60_paper.py・62_video.py を omnicampus のアカウント名に直す）")
    return problems


def account_or_exit(allow_placeholder: bool = False, paper: pathlib.Path = PAPER_PY, video: pathlib.Path = VIDEO_PY) -> str:
    """ACCOUNT を返す。2 つのスクリプトで違う・仮の値のまま（--allow-placeholder なし）なら止める。"""
    acc_p, acc_v = _const(paper, "ACCOUNT"), _const(video, "ACCOUNT")
    problems = account_problems(acc_p, acc_v, allow_placeholder)
    if problems:
        raise SystemExit("止める: " + " / ".join(problems))
    return acc_p

INCLUDE = [
    "pyproject.toml", ".gitignore",
    "src/recovla", "configs/default.yaml", "configs/demo",
    # 段階 3（0146 の 3）: センサだけの実行系（v2）の設定と、段階 3 のエキスパートの設定
    "configs/sensor_v1.yaml", "configs/runtime_v2.yaml", "configs/runtime_v2_color.yaml", "configs/runtime_v2_derived.yaml",
    "configs/runtime_v2_judge.yaml", "configs/latency_v1.json", "configs/expert_v3.yaml",
    # robot_only.xml は実行系のロボットのモデル（runtime/robot_model.py・harness/setup.py が読む）。10/08 に抜けを見つけて足した
    "assets/mjcf/scene_3cube.xml", "assets/mjcf/panda", "assets/mjcf/robot_only.xml",
    "env/setup_env.ps1", "env/requirements-lock.txt", "env/session_env.example.ps1",
    "tests", "docs/interfaces",
    "paper/template.html", "paper/fig1.svg", "paper/fig_recovery.svg",
    "docs/freeze/s3_hashes.json", "docs/freeze/pip_freeze.txt",
    # docs/results: 最終評価の詳しい表（results_s3.md）と、センサだけの実行系（v2）・段階 2・3 の判断に使った記録の写し
    # （書き出すスクリプト 82・83・86・91・92〜95 などの出力）。段階 1 より前の表と図は EXCLUDE で外す
    "docs/results",
    "scripts/10_gen_normal.py", "scripts/11_d_checks.py", "scripts/20_k1.py", "scripts/23_color_fit.py",
    "scripts/24_cue_check.py", "scripts/30_f.py", "scripts/40_h.py", "scripts/41_results.py", "scripts/42_g_checks.py",
    "scripts/46_r2.py", "scripts/47_steps.py", "scripts/48_rtc_redo.py", "scripts/49_contact.py", "scripts/50_e_eval.py",
    "scripts/51_planner.py", "scripts/52_freeze.py", "scripts/53_results.py", "scripts/54_results_s3.py", "scripts/55_extra_s3.py",
    "scripts/60_paper.py", "scripts/61_demo.py", "scripts/62_video.py", "scripts/63_demo_v2.py", "scripts/replay_check.py",
    "scripts/80_g3_measure.py", "scripts/81_hand_check.py", "scripts/82_v2_eval.py", "scripts/83_perception_check.py",
    "scripts/85_v2_judge.py", "scripts/86_stage2_a.py", "scripts/87_v2_e.py", "scripts/88_cause.py", "scripts/89_gate.py",
    "scripts/90_rerun.py", "scripts/91_fast_query_check.py", "scripts/92_s3_round0.py", "scripts/93_s3_l0.py",
    "scripts/94_s3_round1.py", "scripts/95_s3_round2.py", "scripts/check_g1_boundary.py",
]
EXCLUDE = [r"__pycache__", r"\.pyc$", r"tests/test_c_port\.py$", r"tests/test_push_check\.py$",
           # 提出の道具（このスクリプト・check_submission.py・push_submission.ps1）のテスト。道具は入れないのでテストも入れない
           r"tests/test_submission_1008\.py$",
           r"tests/fixtures/scene_g0_reference\.json$", r"tests/planner/fixtures/final_sentences\.json$",
           # 以前の版（目標 v1、9/29 提出）の表と図（0149: 提出用リポジトリの最初の 2 コミットに残る）
           r"^docs/results/results\.md$", r"^docs/results/e4_tradeoff_final\.png$",
           # 段階 4（作業中、10/08）: 今は入れない。段階 4 を載せると決めたら、ここを外して INCLUDE に足す
           r"^src/recovla/diag/", r"^src/recovla/eval/time_scoring\.py$", r"^src/recovla/planner/decompose_s4\.py$",
           r"^tests/test_s4_", r"^tests/test_time_scoring\.py$", r"(?:^|[/_])s4(?:[_./]|$)",
           # 10/08 に抜けを見つけて足した: 段階 4 の関門の判定（configs/s4_gates.json を読む。書き出し先のテストが止まる）と、
           # 作業中の VLM の試験台（束 6 (iii)、本番は束 1 の後）。載せると決めたら外す
           r"^src/recovla/eval/gate1\.py$", r"^tests/test_gate1\.py$", r"^src/recovla/vlm/", r"^tests/test_vlm_bench\.py$"]
# あれば入れる（束 5・6 で作っている途中のもの）。ないときは書き出しの最後に名前を出す
OPTIONAL = ["scripts/56_intervention_s3.py"]
# configs/g0.yaml は以前の実験の記録への参照（学習の実行・チェックポイント・評価・シード）を含むので、そのままは入れない。
# コードが読む値（立方体 1 個の場面・再生の許容・記録の道具の指示と配置の範囲）だけを書く
G0_YAML = """# 立方体 1 個の場面（以前に自作した遠隔操作と記録の道具で使っていた場面）の設定。
# recovla.common.config.load("g0") で default.yaml の上に重ねる。記録の道具（record/recorder.py・replay.py）と
# 再生の確認（scripts/replay_check.py）が読む。3 色の場面の評価には使わない。

scene:
  box: {inner_half: %(inner)s, success_inner_half: %(succ)s}   # 立方体 1 個の場面（scene_g0.xml、内寸 12 cm）

g0:
  scene: assets/mjcf/scene_g0.xml
  replay_tol_m: %(rtol)s                                          # 再生の確認: 手先 5 mm
  replay_image_tol: %(itol)s                                      # 再生の確認: 画像 ±2（GPU の描画の揺らぎ）
  instruction: "%(instr)s"
  placement:                                                   # 記録の道具の立方体の配置の範囲
%(placement)s
"""
EXTRA_FILES = ["assets/mjcf/scene_g0.xml"]
VERBATIM = {"docs/freeze/s3_hashes.json"}               # 文の置き換えをかけない（凍結の一覧のパスがずれる）
KEEP = ("THIRD_PARTY_NOTICES.md",)                       # 書き出し先で手で書くファイル（書き出しで消さない）
# README.md は 10/08 から書き出しで作る: 原稿 paper/README_submission.md の {{キー}} に、説明資料と同じ値の一覧
# （paper/build/values.json、scripts/60_paper.py build が作る）を差し込む。数字を手で書かないため（説明資料の 1 ページ目と同じ中身）
README_TMPL = ROOT / "paper" / "README_submission.md"
VALUES = ROOT / "paper" / "build" / "values.json"
# LICENSE（10/08 の作者の決定: MIT）。著作権者は omnicampus のアカウント名（ACCOUNT と同じ値）。課題の 10/8 版で、PDF・コード・
# 動画に氏名・所属・メールアドレスを載せない（アカウント名は可）ため、本名を入れる欄は作らない。仮の値のままなら export が止まり、
# 書き出し先の LICENSE に仮の値が残っていれば check_submission.py が止める
LICENSE_HOLDER = ACCOUNT
LICENSE_YEAR = "2026"
MIT_TEXT = """MIT License

Copyright (c) {year} {holder}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def render_readme() -> str:
    """README の原稿に値を差し込む。キーがない・値の一覧がないときは止める（手で書いた数字に戻さない）。"""
    if not VALUES.is_file():
        raise SystemExit(f"値の一覧がない: {VALUES}（先に scripts\\60_paper.py build）")
    v = json.loads(VALUES.read_text(encoding="utf-8"))
    missing = sorted({k for k in re.findall(r"\{\{(\w+)\}\}", README_TMPL.read_text(encoding="utf-8")) if k not in v})
    if missing:
        raise SystemExit(f"README の原稿のキーが値の一覧にない: {', '.join(missing)}（60_paper.py build をやり直す）")
    return re.sub(r"\{\{(\w+)\}\}", lambda m: v[m.group(1)], README_TMPL.read_text(encoding="utf-8"))
TEXT_EXT = {".py", ".yaml", ".yml", ".xml", ".ps1", ".md", ".txt", ".toml", ".html", ".svg", ".json", ".cfg", ".ini"}

# (パターン, 置き換え)。上から順に。掲示板の番号・内部の文書への参照は消す
REF = r"(?:掲示板|決裁|監督|本線|board)"
NUM = r"0\d{3}(?:\s*〜\s*0\d{3})?(?:\s*の\s*\d+)?(?:\s*の\s*\(\d+\))?(?:\s*の後の回答)?"
REPLACE = [
    (r"流用元（C:\\VLA）", "以前に自作した遠隔操作と記録の道具"),
    # 60_paper.py の FORBIDDEN の中の「C:\\VLA」の型は、値を変えずに V を \x56 と書いて、文字としては残さない
    (r'"C:\\\\\\\\VLA"', lambda m: r'"C:\\\\\x56LA"'),
    # 10/08 に直した: 以前の型（C:\\\\?VLA\\\\?…）は VLA の後ろに \\ を要したので、「C:\\VLA・」のような書き方を残していた
    (r"C:\\{1,2}VLA(?:\\{1,2}[^\s）)」・、]*)?", "以前に自作した遠隔操作と記録の道具"),
    (r"流用元", "以前に自作した遠隔操作と記録の道具"),
    (rf"[（(]\s*{REF}?\s*{NUM}(?:\s*[・、,]\s*{REF}?\s*{NUM})*\s*[）)]", ""),
    (rf"{REF}\s*{NUM}(?:\s*[・、,]\s*{NUM})*", ""),
    (rf"＝\s*{NUM}(?:\s*[・、]\s*{NUM})*", ""),
    (rf"(?<=[\s、（(「〜\"。・＋]){NUM}(?:\s*[・、]\s*{NUM})*(?=[\s）)、。・:」\"（]|$)", ""),
    (r"(?i)board\s*0\d{3}(?:\s*[/・,]\s*0\d{3})*", ""),
    (r"（(?:[0-9a-f]{7}・?)+）", ""),
    (r"[0-9a-f]{7}・[0-9a-f]{7}・?", ""),
    (r"B_提案書\s*(?:§\s*[\d.]+)?(?:\s*の\s*\d+)?", ""),
    (r"(?:手順書|計画書)\s*(?:v2\s*)?(?:Step\s*[A-J]\s*)?(?:§\s*[\d.]+)?(?:\s*の\s*\d+)?", ""),
    (r"変更一覧_?v3\s*(?:の\s*\d+)?", ""),
    # 開発中の語 → 分野の普通の語
    (r"塊", "アクションチャンク"),
    (r"こま", "フレーム"),
    (r"保存点", "チェックポイント"),
    # 10/07 の語の指示（60_paper.py の FORBIDDEN）: キュー→目標位置の入力、設置情報→既知の情報、対→組、特権情報・解放は使わない、
    # 物理演算の停止には触れない。10/08 から提出用リポジトリの全ファイルにも同じ語の検査をかける（check_submission.py）
    (r"目標の(位置の)?手がかり", "目標位置の入力"),
    (r"手がかり", "目標位置の入力"),
    (r"目標位置のキュー", "目標位置の入力"),
    (r"キュー", "目標位置の入力"),
    (r"設置情報の既知の", "既知の"),
    (r"設置情報", "既知の情報"),
    (r"特権情報を持つ", "真の状態を参照できる"),
    (r"特権情報", "真の状態"),
    (r"解放の高さ", "グリッパを開く高さ"),
    (r"解放のとき", "破棄のとき"),
    (r"使い、推論中に物理演算を止める", "使う"),
    (r"(\d)(\s*)対(?=\s*[）)、，,中]|\s*$)", r"\1\2組"),   # 「27 対」（組の数）
    (r"(\d)(\s*)対(?!応)", r"\1\2と"),                    # 「11 対 8」「R1 対 N1」（比べる 2 つ）
    (r"(?<![反相絶一])対([はがを])", r"組\1"),
    (r"誘発する", "意図的に失敗させる"),
    (r"誘発し", "意図的に失敗させ"),
    (r"誘発", "意図的な失敗"),
    (r"成立しなかった", "実際には失敗しなかった"),
    (r"成立した", "実際に失敗した"),
    (r"成立率", "実際に失敗した割合"),
    (r"成立", "実際の失敗"),
    (r"失敗を注入する", "意図的に失敗させる"),
    (r"失敗を注入し", "意図的に失敗させ"),
    (r"(失敗)?注入する", "意図的に失敗させる"),
    (r"(失敗)?注入し(?=[たて、 ])", "意図的に失敗させ"),
    (r"失敗注入", "意図的な失敗"),
    (r"注入", "意図的な失敗"),
    (r"立ち直り", "復帰"),
    (r"立ち直", "復帰"),
    (r"台本", "スクリプト化したエキスパート"),
    (r"継ぎ目の跳び", "チャンク境界での速度の不連続"),
    (r"継ぎ目", "チャンク境界"),
    (r"流れの一致", "Flow Matching"),
    (r"主な検定", "主要評価項目の検定"),
    (r"副の指標", "副次評価項目"),
    (r"自然な失敗", "意図的な失敗なしで起きた失敗"),
    (r"不自然", "不適切"),
    (r"自然な", "意図的な失敗なしの"),
    (r"自然に", "ひとりでに"),
    (r"自然", "意図的な失敗なし"),
    (r"(?<![見])通し(?![てたまか])", "3 個の連続タスク"),
    (r"(?<=Wilson )(95% )?区間", "95% 信頼区間"),
    (r"(?<=95% )区間", "信頼区間"),
    (r"(?<!信頼)区間", "部分"),
    (r"遅れて", "遅延して"),
    (r"遅れる", "遅延する"),
    (r"遅れた", "遅延した"),
    (r"遅れ", "遅延"),
    (r"種類", "@@SHURUI@@"),
    (r"(?<![各一機人品])種(?![別目])", "シード"),
    (r"@@SHURUI@@", "種類"),
    (r"(選択用|最終評価用|学習用|評価用|検証用|テスト用)の帯", r"\1のシード範囲"),
    (r"帯", "シード範囲"),
    (r"(\d|万|千)\s*手(?![先首順法元])", r"\1 ステップ"),
    (r"(\w)\s*手目", r"\1 ステップ目"),
    (r"(\})\s*手(?![先首順法元])", r"\1 ステップ"),
    (r"[（(]\s*[甲乙丙]\s*[）)]", ""),
    (r"[甲乙丙]", ""),
    (r"本線", "本構成"),
    (r"支線", "別の環境"),
    (r"監督", "レビュー"),
    (r"決裁", "判断"),
    (r"掲示板", "設計メモ"),
    (r"予備実験", "以前の実験"),
    (r"卒研|卒業研究", "以前の研究"),
    (r"学生", "利用者"),
    (r"模型", "モデル"),
    (r"関所", "監査の関門"),
    (r"諮る", "確認する"),
    (r"諮った", "確認した"),
    (r"諮り", "確認し"),
    (r"諮", "確認"),
]


def included_files():
    out = []
    for inc in INCLUDE + [o for o in OPTIONAL if (ROOT / o).exists()]:
        p = ROOT / inc
        if not p.exists():
            raise SystemExit(f"INCLUDE のファイルがない: {inc}")
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
        for f in files:
            rel = f.relative_to(ROOT).as_posix()
            if any(re.search(e, rel) for e in EXCLUDE):
                continue
            out.append(rel)
    return out


def transform(text: str):
    counts = {}
    for pat, rep in REPLACE:
        text, n = re.subn(pat, rep, text)
        if n:
            counts[pat] = counts.get(pat, 0) + n
    # 番号を消した後に残る区切りを片付ける
    text = re.sub(r"（\s*の\s*\d+\s*[＋・、]?\s*", "（", text)
    text = re.sub(r"（\s*[・、＋]\s*", "（", text)
    text = re.sub(r"\s*[・、＋]\s*）", "）", text)
    text = re.sub(r"（\s*）", "", text)
    return text, counts


def license_text(holder: str) -> str:
    return MIT_TEXT.format(year=LICENSE_YEAR, holder=holder)


def cmd_export(a) -> None:
    dest = pathlib.Path(a.dest)
    account_or_exit(a.allow_placeholder)               # LICENSE の著作権者（LICENSE_HOLDER = ACCOUNT）が仮の値なら、消す前に止める
    readme = render_readme()                           # 消す前に作れることを確かめる
    if dest.exists():                                  # 書き出し先を作り直す（.git と手書きのファイル KEEP は残す）
        if dest.name != "recovery-vla-panda":
            raise SystemExit(f"書き出し先の名前が違う: {dest}")
        for p in dest.iterdir():
            if p.name in (".git",) + KEEP:
                continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    report = {"files": {}, "total": 0}
    for rel in included_files():
        src, dst = ROOT / rel, dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (src.suffix.lower() in TEXT_EXT or src.name == ".gitignore") and rel not in VERBATIM:
            t, c = transform(src.read_text(encoding="utf-8"))
            dst.write_text(t, encoding="utf-8", newline="")
            if c:
                report["files"][rel] = c
                report["total"] += sum(c.values())
        elif rel in VERBATIM:                          # 一覧の値はそのまま、説明の文だけ置き換える
            d = json.loads(src.read_text(encoding="utf-8"))
            d["note"] = transform(d["note"])[0]
            dst.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        else:
            shutil.copyfile(src, dst)
    import yaml
    g = yaml.safe_load((ROOT / "configs" / "g0.yaml").read_text(encoding="utf-8"))
    pl = dict(g["g0"]["placement"])
    (dest / "configs" / "g0.yaml").write_text(G0_YAML % {
        "inner": g["scene"]["box"]["inner_half"], "succ": g["scene"]["box"]["success_inner_half"],
        "rtol": g["g0"]["replay_tol_m"], "itol": g["g0"]["replay_image_tol"], "instr": g["g0"]["instruction"],
        "placement": "\n".join(f"    {k}: {json.dumps(v)}" for k, v in pl.items())}, encoding="utf-8")
    for rel in EXTRA_FILES:
        t, c = transform((ROOT / rel).read_text(encoding="utf-8"))
        (dest / rel).write_text(t, encoding="utf-8", newline="")
    (dest / "README.md").write_text(readme, encoding="utf-8", newline="")          # 原稿は語の決まりに沿って書くので置き換えをかけない
    (dest / "LICENSE").write_text(license_text(LICENSE_HOLDER), encoding="utf-8", newline="")
    (ROOT / "outputs" / "submission").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "submission" / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                                         encoding="utf-8")
    print("files", len(included_files()), "replacements", report["total"], "in", len(report["files"]), "files")
    missing = [o for o in OPTIONAL if not (ROOT / o).exists()]
    if missing:
        print("まだないので入れなかった（OPTIONAL）:", ", ".join(missing))


SCAN = [r"塊", r"こま", r"種(?!類)", r"帯", r"台本", r"誘発", r"成立", r"立ち直", r"手がかり", r"継ぎ目", r"遅れ",
        r"流れの一致", r"保存点", r"自然", r"(?<![見])通し(?![てたまか])", r"主な検定", r"副の指標", r"(?<!信頼)区間", r"[甲乙丙]", r"本線", r"支線",
        r"監督", r"決裁", r"掲示板", r"流用元", r"卒研", r"卒業研究", r"C:\\+VLA", r"学生", r"予備実験", r"(\d|万)\s*手(?![先首順法])",
        r"模型", r"手順書", r"提案書", r"計画書", r"(?<![\d.#_a-fA-Fx])0\d{3}(?![\dA-Fa-f])", r"student", r"fujikro", r"secec", r"注入"]


def cmd_scan(a) -> None:
    dest = pathlib.Path(a.dest)
    hits = {}
    for f in sorted(dest.rglob("*")):
        if not f.is_file() or ".git" in f.parts or (f.suffix.lower() not in TEXT_EXT and f.name != ".gitignore"):
            continue
        t = f.read_text(encoding="utf-8", errors="replace")
        t = re.sub(r"[0-9a-f]{40,64}", lambda m: "h" * len(m.group(0)), t)      # ハッシュの値は調べない
        for pat in SCAN:
            for m in re.finditer(pat, t):
                line = t.count("\n", 0, m.start()) + 1
                hits.setdefault(f.relative_to(dest).as_posix(), []).append((pat, line, t[max(0, m.start() - 20):m.end() + 20].replace("\n", " ")))
    n = sum(len(v) for v in hits.values())
    (ROOT / "outputs" / "submission" / "scan.json").write_text(json.dumps(hits, ensure_ascii=False, indent=1), encoding="utf-8")
    print("hits", n, "files", len(hits))
    for k, v in list(hits.items())[: a.show]:
        print(k, len(v), v[:3])


ZIP_LIMIT = 200 * 1000 * 1000                  # 課題の 9/30 版: omnicampus の Zip は 200MB 以下（MB を小さい方の 10^6 で取る）


def cmd_package(a) -> None:
    """omnicampus に出す Zip（動画 MP4 1 つ＋説明資料 PDF 1 つ）を作り、課題の 9/30 版の決まりを確かめる。

    10/8 版で不要（Zip の提出はなくなり、PDF と GitHub の URL を出し、動画は YouTube に上げる）。消さずに残すが、既定では
    使わない（--legacy を付けたときだけ動く）。動画の確かめは youtube を使う。
    """
    if not a.legacy:
        print("package は課題の 10/8 版で不要（Zip の提出はなくなった）。動画の確かめと概要欄の下書きは youtube を使う。"
              "古い Zip をどうしても作るときだけ --legacy を付ける。")
        raise SystemExit(2)
    import zipfile
    build = ROOT / "paper" / "build"
    acc_p, acc_v = _const(ROOT / "scripts" / "60_paper.py", "ACCOUNT"), _const(ROOT / "scripts" / "62_video.py", "ACCOUNT")
    problems = []
    if acc_p != acc_v:
        problems.append(f"60_paper.py と 62_video.py の ACCOUNT が違う: {acc_p!r} / {acc_v!r}")
    if PLACEHOLDER in (acc_p, acc_v) and not a.allow_placeholder:
        problems.append(f"ACCOUNT が {PLACEHOLDER!r} のまま（omnicampus のアカウント名に直してから build し直す）")
    pdf, mp4 = build / f"説明資料_PAI最終課題_{acc_p}.pdf", build / f"動画_PAI最終課題_{acc_v}.mp4"
    for f in (pdf, mp4):
        if not f.is_file():
            problems.append(f"ない: {f}（60_paper.py build・62_video.py build をやり直す）")
    if mp4.is_file():
        import av
        with av.open(str(mp4)) as c:
            sec = float(c.duration) / av.time_base
        if not 60 <= sec <= 180:   # 10/08 の規則の変更: 1〜3 分
            problems.append(f"動画の長さ {sec:.1f} s（1〜3 分の外）")
    for f, base in ((pdf, "paper.pdf"), (mp4, "video.mp4")):   # 提出名のファイルが、照合した組み上げ（paper.pdf・video.mp4）と同じ中身か
        b = build / base
        if f.is_file() and (not b.is_file() or f.read_bytes() != b.read_bytes()):
            problems.append(f"{f.name} が {base} と違う（build と check をやり直す）")
    chk = build / "check.json"
    if chk.is_file() and json.loads(chk.read_text(encoding="utf-8")).get("problems"):
        problems.append("説明資料の照合（60_paper.py check）が不合格のまま")
    if problems:
        print("Zip を作らない:")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    out = pathlib.Path(a.out) if a.out else ROOT / "outputs" / "submission"
    out.mkdir(parents=True, exist_ok=True)
    zp = out / f"PAI最終課題_{acc_p}.zip"
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_STORED) as z:   # MP4・PDF は圧縮済み。名前は UTF-8 の印つきで入る
        z.write(pdf, pdf.name)
        z.write(mp4, mp4.name)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        bad = z.testzip()
    size = zp.stat().st_size
    ok = size <= ZIP_LIMIT and bad is None and sorted(names) == sorted([pdf.name, mp4.name])
    print(json.dumps({"zip": str(zp), "bytes": size, "limit": ZIP_LIMIT, "names": names, "video_s": round(sec, 1),
                      "ok": ok}, ensure_ascii=False, indent=1))
    raise SystemExit(0 if ok else 1)


# ------------------------------------------------------------------ YouTube（課題の 10/8 版）
# 動画は自分の YouTube に上げ、URL を出す。タイトルは「PAI最終課題_<omnicampus のアカウント名>」。長さは 1〜3 分、倍速は 2 倍まで
# で、倍速は動画の中か概要欄に明記する。概要欄の数字は説明資料と同じ値の一覧（paper/build/values.json）から差し込み、手で書かない
VIDEO_YAML = ROOT / "configs" / "demo" / "video_s3.yaml"
MAX_SPEED = 2.0                                # 課題の 10/8 版: 倍速は 2 倍まで（62_video.py の MAX_SPEED と同じ）
VIDEO_MIN_S, VIDEO_MAX_S = 60, 180             # 課題の 10/8 版: 動画は 1〜3 分
CLIP_KEYS = ("clip", "clip_with", "clip_without", "t0", "t1")       # どれかがある場面は映像の場面（倍速を書く）
# 概要欄に出す場面の名前（場面の id → 名前）。ない id はそのまま出す
SCENE_NAMES = {"contrast": "冒頭の左右比較（復帰デモあり／なし）", "contrast_p1": "把持を意図的に失敗させた評価の 1 回",
               "task": "「全部片付けて」の実演（3 個を順に片付ける）"}
YT_TITLE = "PAI最終課題_{account}"
YT_TITLE_MAX, YT_DESC_MAX = 100, 5000          # YouTube の上限（文字数）。概要欄に < と > は使えない
GITHUB_BLANK = "（ここに提出する GitHub のリポジトリの URL を入れる。public にしてから）"
# 概要欄の原稿。{{キー}} は values.json の値と、ここで計算する値（idea・speed_lines・github_url・v_len）
YT_TMPL = """{{idea}}

■ 倍速（動画の中でも、映像の場面ごとに右上へ「実時間」「N 倍速」と表示）
{{speed_lines}}

■ コード（GitHub）
{{github_url}}

■ 説明資料の要点
・人手のデモ集めなし: 学習データ {{eps_auto}} 本（成功デモ {{n_normal}}・復帰デモ {{n_recovery}}）をすべてスクリプトで自動で作り、VLA（SmolVLA）を追加学習した
・把持を意図的に失敗させた同じ配置・同じ失敗の {{e3a_pairs}} 組で、掴み直して成功したのは復帰デモあり {{e3a_xk}} 回、なし {{e3a_yk}} 回（Holm 補正後 p {{e3a_holmeq}}）
・「全部片付けて」の 1 文から LLM が順番を決め、VLA が 3 個を運ぶ実演では、{{e7_n}} 配置中 {{e7_k}} 配置で 3 個とも片付いた。シミュレーションのみで、落下・置き損ねからはほとんど回復しない

動画の長さ {{v_len}}。詳しい条件と数字は説明資料に記載
"""
# 上げるときの設定（課題の 10/8 版）。下書きと一緒に出す
YT_SETTINGS = [("タイトル", "上の 1 行（PAI最終課題_<omnicampus のアカウント名>）"),
               ("公開設定", "限定公開以上（最低条件。全体公開なら加点）。非公開は不可"),
               ("視聴者（子ども向け）", "いいえ、子ども向けではありません"),
               ("年齢制限", "いいえ、18 歳以上の視聴者のみに制限しません"),
               ("プレミア公開", "しない（インスタントプレミア公開も使わない。すぐ公開する）"),
               ("概要欄", "上の下書きを貼り、GitHub の URL の欄を埋める")]


def mmss(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def speed_label(sp: float) -> str:
    """62_video.py の画面の札と同じ書き方（1 倍から 0.05 以内なら実時間）。"""
    return "実時間（1 倍速）" if abs(sp - 1) <= 0.05 else f"{sp:.1f} 倍速"


def scene_speeds(scenes: list):
    """場面の表（video_s3.yaml の scenes）から、映像の場面の時刻と倍速 (t1 − t0) / dur_s を出す。(行, 合計の秒, 問題)。"""
    rows, problems, t = [], [], 0.0
    for sc in scenes:
        dur = float(sc["dur_s"])
        start, t = t, t + dur
        if not any(k in sc for k in CLIP_KEYS):
            continue                                   # 図と文字だけの場面（倍速なし）
        if "t1" not in sc:
            problems.append(f"場面 {sc['id']} に t1 がなく倍速を計算できない（映像の長さで決まる。t1 を書く）")
            continue
        sp = (float(sc["t1"]) - float(sc.get("t0", 0.0))) / dur
        rows.append({"id": sc["id"], "name": SCENE_NAMES.get(sc["id"], sc["id"]), "start": mmss(start), "end": mmss(t),
                     "speed": round(sp, 4), "label": speed_label(sp)})
        if sp > MAX_SPEED + 1e-9:
            problems.append(f"場面 {sc['id']} の倍速 {sp:.2f} が {MAX_SPEED:g} 倍を超える（dur_s を延ばす）")
    return rows, t, problems


def video_seconds(path: pathlib.Path):
    """動画の長さ（秒）と読んだ道具。ffprobe があればそれで、なければ OpenCV（フレーム数 ÷ fps）。読めなければ (None, 理由)。"""
    exe = shutil.which("ffprobe")
    if exe:
        r = subprocess.run([exe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                            str(path)], capture_output=True, text=True)
        try:
            return float(r.stdout.strip().splitlines()[0]), "ffprobe"
        except (ValueError, IndexError):
            pass
    try:
        import cv2
    except ImportError:
        return None, "ffprobe も OpenCV もない"
    cap = cv2.VideoCapture(str(path))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if n > 0 and fps > 0:
        return n / fps, "OpenCV"
    return None, "ffprobe でも OpenCV でも読めない"


def idea_sentence() -> str:
    """1 文のアイデア。README の原稿の最初の太字の段落（説明資料の冒頭と同じ考え方の文）をそのまま使う。"""
    m = re.search(r"^\*\*(.+?)\*\*\s*$", README_TMPL.read_text(encoding="utf-8"), flags=re.M)
    if not m:
        raise SystemExit(f"1 文のアイデア（最初の太字の段落）が {README_TMPL.name} にない")
    return m.group(1)


def youtube_text(account: str, rows: list, values: dict, idea: str, github_url: str, seconds: float):
    """YouTube のタイトルと概要欄。原稿のキーが値の一覧にないときは (タイトル, None, 足りないキー)。"""
    lines = [f"・{r['start']}〜{r['end']} {r['name']}: {r['label']}" for r in rows]
    lines.append("・ほかの場面は図と文字だけ（倍速なし）")
    v = dict(values, idea=idea, speed_lines="\n".join(lines), github_url=github_url or GITHUB_BLANK, v_len=mmss(seconds))
    missing = sorted({k for k in re.findall(r"\{\{(\w+)\}\}", YT_TMPL) if k not in v})
    title = YT_TITLE.format(account=account)
    if missing:
        return title, None, missing
    # 概要欄に < と > は使えないので、差し込む値の < > は全角にする（p の値「< 0.001」など）
    fill = lambda m: str(v[m.group(1)]).replace("<", "＜").replace(">", "＞")
    return title, re.sub(r"\{\{(\w+)\}\}", fill, YT_TMPL), []


def forbidden_hits(text: str) -> list:
    """60_paper.py の FORBIDDEN（使わない語）に当たった所。60_paper.py は読み込まない（設定と結果を読むため重い）。"""
    return [(p, m.group(0)) for p in _const(PAPER_PY, "FORBIDDEN") for m in re.finditer(p, text)]


def cmd_youtube(a) -> None:
    """動画 MP4 を確かめ（長さ 1〜3 分・場面の倍速が 2 倍まで）、YouTube のタイトルと概要欄の下書きを出す。

    読むもの: 60_paper.py・62_video.py の ACCOUNT、configs/demo/video_s3.yaml、paper/build/values.json、
              paper/build/動画_PAI最終課題_<アカウント名>.mp4、paper/README_submission.md（1 文のアイデア）
    書くもの: outputs/submission/youtube_title.txt・youtube_description.txt・youtube.json（--out で変える）。
              問題があれば何も書かずに exit 1
    """
    import yaml
    acc_p, acc_v = _const(PAPER_PY, "ACCOUNT"), _const(VIDEO_PY, "ACCOUNT")
    problems = account_problems(acc_p, acc_v, a.allow_placeholder)
    rows, total, p = scene_speeds(yaml.safe_load(pathlib.Path(a.scenes).read_text(encoding="utf-8"))["scenes"])
    problems += p
    mp4 = pathlib.Path(a.video) if a.video else ROOT / "paper" / "build" / f"動画_PAI最終課題_{acc_v}.mp4"
    sec, how = None, ""
    if not mp4.is_file():
        problems.append(f"動画がない: {mp4}（62_video.py build）")
    else:
        sec, how = video_seconds(mp4)
        if sec is None:
            problems.append(f"動画の長さを読めない: {mp4}（{how}）")
        elif not VIDEO_MIN_S <= sec <= VIDEO_MAX_S:
            problems.append(f"動画の長さ {sec:.1f} s（1〜3 分の外）")
        elif abs(sec - total) > 0.5:
            problems.append(f"動画の長さ {sec:.1f} s が場面の表の合計 {total:g} s と違う（概要欄の時刻と倍速がずれる。62_video.py build）")
    values_p = pathlib.Path(a.values)
    values = json.loads(values_p.read_text(encoding="utf-8")) if values_p.is_file() else None
    if values is None:
        problems.append(f"値の一覧がない: {values_p}（先に scripts\\60_paper.py build）")
    if a.github_url and not re.fullmatch(r"https://github\.com/[A-Za-z0-9-]+/[A-Za-z0-9._-]+/?", a.github_url):
        problems.append(f"GitHub の URL の形が違う: {a.github_url}（https://github.com/<ユーザー名>/<リポジトリ>）")
    title = desc = None
    if values is not None:
        title, desc, missing = youtube_text(acc_p, rows, values, idea_sentence(), a.github_url, sec if sec else total)
        if missing:
            problems.append(f"概要欄の原稿のキーが値の一覧にない: {', '.join(missing)}（60_paper.py build をやり直す）")
    if desc is not None:
        for pat, hit in forbidden_hits(title + "\n" + desc):
            problems.append(f"使わない語 {hit!r}（60_paper.py の FORBIDDEN {pat!r}）")
        if re.search(r"[<>]", title + desc):
            problems.append("タイトルか概要欄に < か > がある（YouTube では使えない）")
        if len(title) > YT_TITLE_MAX or len(desc) > YT_DESC_MAX:
            problems.append(f"長すぎる: タイトル {len(title)} 字（上限 {YT_TITLE_MAX}）、概要欄 {len(desc)} 字（上限 {YT_DESC_MAX}）")
    if problems:
        print("YouTube の下書きを書かない:")
        for q in problems:
            print("  " + q)
        raise SystemExit(1)
    out = pathlib.Path(a.out) if a.out else ROOT / "outputs" / "submission"
    out.mkdir(parents=True, exist_ok=True)
    (out / "youtube_title.txt").write_text(title + "\n", encoding="utf-8")
    (out / "youtube_description.txt").write_text(desc, encoding="utf-8")
    (out / "youtube.json").write_text(json.dumps({
        "title": title, "description": desc, "video": str(mp4), "video_s": round(sec, 2), "read_with": how,
        "scenes_total_s": total, "speeds": rows, "max_speed": MAX_SPEED, "github_url": a.github_url,
        "settings": dict(YT_SETTINGS)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("=== タイトル ===")
    print(title)
    print("=== 概要欄 ===")
    print(desc)
    print("=== 上げるときの設定（課題の 10/8 版） ===")
    for k, val in YT_SETTINGS:
        print(f"  {k}: {val}")
    if not a.github_url:
        print("注意: GitHub の URL の欄が空（--github-url で入れるか、貼った後に手で埋める）")
    print("書いた:", out / "youtube_title.txt", out / "youtube_description.txt", out / "youtube.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("export")
    s.add_argument("--dest", required=True)
    s.add_argument("--allow-placeholder", action="store_true",
                   help="試しだけ: ACCOUNT が仮の名前のままでも書き出す（LICENSE に仮の値が入り、check_submission.py が止める）")
    s = sub.add_parser("scan")
    s.add_argument("--dest", required=True)
    s.add_argument("--show", type=int, default=30)
    s = sub.add_parser("youtube", help="動画の確かめと YouTube のタイトル・概要欄の下書き（課題の 10/8 版）")
    s.add_argument("--video", default="", help="動画 MP4（既定 paper/build/動画_PAI最終課題_<アカウント名>.mp4）")
    s.add_argument("--scenes", default=str(VIDEO_YAML), help="場面の表（既定 configs/demo/video_s3.yaml）")
    s.add_argument("--values", default=str(VALUES), help="説明資料と同じ値の一覧（既定 paper/build/values.json）")
    s.add_argument("--github-url", default="", help="提出する GitHub のリポジトリの URL（空なら欄だけ作る）")
    s.add_argument("--out", default="", help="下書きの置き場所（既定 outputs/submission）")
    s.add_argument("--allow-placeholder", action="store_true", help="試しだけ: ACCOUNT が仮の名前のままでも下書きを出す")
    s = sub.add_parser("package", help="10/8 版で不要（Zip の提出はなくなった）。既定では使わない。--legacy のときだけ動く")
    s.add_argument("--legacy", action="store_true", help="10/8 版で不要。古い Zip をどうしても作るときだけ")
    s.add_argument("--out", default="", help="Zip の置き場所（既定 outputs/submission）")
    s.add_argument("--allow-placeholder", action="store_true", help="試しだけ: ACCOUNT が仮の名前のままでも作る")
    a = ap.parse_args(argv)
    {"export": cmd_export, "scan": cmd_scan, "youtube": cmd_youtube, "package": cmd_package}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
