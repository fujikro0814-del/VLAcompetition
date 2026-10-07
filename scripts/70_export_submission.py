"""提出用リポジトリ（recovery-vla-panda）の中身を書き出す（0099 の 3-3・0101 の 3・4）。このスクリプト自体は書き出さない。

    .venv\\Scripts\\python.exe scripts\\70_export_submission.py export --dest C:\\PAI\\recovery-vla-panda
    .venv\\Scripts\\python.exe scripts\\70_export_submission.py scan --dest C:\\PAI\\recovery-vla-panda   # 残った語の一覧

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

from recovla.common import config

ROOT = config.ROOT

INCLUDE = [
    "pyproject.toml", ".gitignore",
    "src/recovla", "configs/default.yaml", "configs/demo",
    # 段階 3（0146 の 3）: センサだけの実行系（v2）の設定と、段階 3 のエキスパートの設定
    "configs/sensor_v1.yaml", "configs/runtime_v2.yaml", "configs/runtime_v2_color.yaml", "configs/runtime_v2_derived.yaml",
    "configs/runtime_v2_judge.yaml", "configs/latency_v1.json", "configs/expert_v3.yaml",
    "assets/mjcf/scene_3cube.xml", "assets/mjcf/panda",
    "env/setup_env.ps1", "env/requirements-lock.txt", "env/session_env.example.ps1",
    "tests", "docs/interfaces",
    "paper/template.html", "paper/fig1.svg", "paper/fig_recovery.svg",
    "docs/freeze/s3_hashes.json", "docs/freeze/pip_freeze.txt",
    "docs/results/results_s3.md", "docs/results/s3_round1_verify.json", "docs/results/s3_round2_verify.json",
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
           r"tests/fixtures/scene_g0_reference\.json$", r"tests/planner/fixtures/final_sentences\.json$"]
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
KEEP = ("README.md", "THIRD_PARTY_NOTICES.md")          # 書き出し先で手で書くファイル（書き出しで消さない）
TEXT_EXT = {".py", ".yaml", ".yml", ".xml", ".ps1", ".md", ".txt", ".toml", ".html", ".svg", ".json", ".cfg", ".ini"}

# (パターン, 置き換え)。上から順に。掲示板の番号・内部の文書への参照は消す
REF = r"(?:掲示板|決裁|監督|本線|board)"
NUM = r"0\d{3}(?:\s*〜\s*0\d{3})?(?:\s*の\s*\d+)?(?:\s*の\s*\(\d+\))?(?:\s*の後の回答)?"
REPLACE = [
    (r"流用元（C:\\VLA）", "以前に自作した遠隔操作と記録の道具"),
    (r"C:\\\\?VLA\\\\?[^\s）)」]*", "以前に自作した遠隔操作と記録の道具"),
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
    (r"手がかり", "目標位置のキュー"),
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
    for inc in INCLUDE:
        p = ROOT / inc
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


def cmd_export(a) -> None:
    dest = pathlib.Path(a.dest)
    if dest.exists():                                  # 書き出し先を作り直す（.git と README などの手書きのファイルは残す）
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
    (ROOT / "outputs" / "submission").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "submission" / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                                         encoding="utf-8")
    print("files", len(included_files()), "replacements", report["total"], "in", len(report["files"]), "files")


SCAN = [r"塊", r"こま", r"種(?!類)", r"帯", r"台本", r"誘発", r"成立", r"立ち直", r"手がかり", r"継ぎ目", r"遅れ",
        r"流れの一致", r"保存点", r"自然", r"(?<![見])通し(?![てたまか])", r"主な検定", r"副の指標", r"(?<!信頼)区間", r"[甲乙丙]", r"本線", r"支線",
        r"監督", r"決裁", r"掲示板", r"流用元", r"卒研", r"卒業研究", r"VLA\\", r"学生", r"予備実験", r"(\d|万)\s*手(?![先首順法])",
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("export")
    s.add_argument("--dest", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--dest", required=True)
    s.add_argument("--show", type=int, default=30)
    a = ap.parse_args(argv)
    {"export": cmd_export, "scan": cmd_scan}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
