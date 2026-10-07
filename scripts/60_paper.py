"""説明資料（PDF、2〜4 ページ）を結果のファイルから作り、照合する。数字は手で書かない。段階 3 の版（0146・0147）。

    .venv\\Scripts\\python.exe scripts\\60_paper.py build     # 値の計算 → 図 2 → HTML → PDF（Edge のヘッドレス印刷）
    .venv\\Scripts\\python.exe scripts\\60_paper.py check     # 照合: 手で書いた数字がない・値が結果と一致・使わない語が 0・ページ数

原稿は paper/template.html と paper/fig1.svg。数字はすべて {{キー}} の差し込み口で、values() が段階 3 の最終評価の結果
（outputs/results/v2_e_report_s3.json、outputs/v2eval/V3S3 の E6・E7、テスト用のシード範囲 140000〜）と、設定・学習の記録・
凍結の一覧（docs/freeze/s3_hashes.json）・動画の場面の表（configs/demo/video_s3.yaml）から計算する。
目標書 v1 の版（9/29 提出用）は git のタグ stepJ-freeze の版にある。
出力: paper/build/（paper.html・図・values.json）と、paper/build/説明資料_PAI最終課題_<アカウント名>.pdf
"""
import argparse
import collections
import html
import json
import pathlib
import re
import subprocess
import sys

from recovla.common import config, goals

CFG = config.load_v2()
ROOT = config.ROOT
OUT = config.path(CFG["paths"]["outputs"])
RES = OUT / "results"
S3 = OUT / "v2eval" / "V3S3"
PAPER = ROOT / "paper"
BUILD = PAPER / "build"
VIDEO = ROOT / "configs" / "demo" / "video_s3.yaml"
FREEZE = ROOT / "docs" / "freeze" / "s3_hashes.json"
ACCOUNT = "アカウント名"                     # 提出の前に手で直す（ファイル名だけに入る）
PDF_NAME = f"説明資料_PAI最終課題_{ACCOUNT}.pdf"
EDGE = pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
FONT = pathlib.Path(r"C:\Windows\Fonts\BIZ-UDGothicR.ttc")


def _j(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def _mod(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pct(x):
    return f"{round(100 * x)}%"


def ci(w):
    return f"{round(100 * w[0])}〜{round(100 * w[1])}%"


def dci(d):
    lo, hi = d["diff_95ci_newcombe"]
    return f"{round(100 * lo):+d}〜{round(100 * hi):+d}".replace("+0〜", "0〜")


def pv(p):
    if p < 0.001:
        return "< 0.001"
    return f"{p:.2g}" if p < 0.1 else f"{p:.2f}"


def pv_eq(p):
    """本文で「p {{…eq}}」と書く形: 「= 0.016」または「< 0.001」。"""
    s = pv(p)
    return s if s.startswith("<") else f"= {s}"


def mmss(s):
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def video_scenes() -> list:
    import yaml
    return yaml.safe_load(VIDEO.read_text(encoding="utf-8"))["scenes"]


def v3_dataset_dir() -> str:
    """復帰デモありの学習データ（段階 3 の 1 周目、凍結の一覧に入っているもの）。"""
    return _j(OUT / "f" / "data_v3.json")["datasets"]["R1"]["dataset"].replace("\\", "/")


def e7_summary() -> dict:
    """3 個の連続タスク（E7）の記録から: 3 個とも、再試行で完了したサブタスク、完了判定の偽陰性・偽陽性、止まった位置。"""
    runs = [_j(p) for p in sorted((S3 / "E7_R1v3").glob("run_00??.json"))]
    out = {"n": len(runs), "all_three": 0, "retry_completed": 0, "false_neg": 0, "false_pos": 0, "stopped_at": collections.Counter()}
    for r in runs:
        out["all_three"] += bool(r["all_three_in_box"])
        for s in r["steps"]:
            truth = r["truth_success_t"].get(s["color"]) is not None
            if len(s["attempts"]) > 1 and truth:
                out["retry_completed"] += 1
            if s["judged_complete"] and not truth:
                out["false_pos"] += 1
            if not s["judged_complete"] and r["final_in_box"].get(s["color"]):
                out["false_neg"] += 1
        if r.get("stopped"):
            out["stopped_at"][int(r["stopped"]["step"])] += 1
    return out


def values() -> dict:
    ee = _mod("e_eval", ROOT / "scripts" / "50_e_eval.py")
    st3 = _mod("v2e", ROOT / "scripts" / "87_v2_e.py").STAGES["s3"]
    rep = _j(RES / "v2_e_report_s3.json")
    P, sec, sets = rep["primary"], rep["secondary"], rep["sets"]
    v = {}
    # 設定・学習の記録
    fps = int(CFG["convert"]["fps"])
    rows = int(st3["sets"]["A"][3])
    v["policy_hz"] = str(fps)
    v["exec_rows"] = str(rows)
    v["exec_s"] = f"{rows / fps:g}"
    lat = _j(ROOT / "configs" / "latency_v1.json")["kinds"]["policy"]
    v["lat_p50"], v["lat_p95"] = f"{lat['p50']:.2f}", f"{lat['p95']:.2f}"
    ck = _j(FREEZE)["checkpoints"]
    pcfg = _j(ROOT / ck["R1v3"] / "pretrained_model" / "config.json")
    v["chunk"] = str(pcfg["chunk_size"])
    info = _j(ROOT / v3_dataset_dir() / "meta" / "info.json")
    v["act_dim"] = str(info["features"]["action"]["shape"][0])
    sd = info["features"]["observation.state"]["shape"][0]
    cue = sum(1 for n in info["features"]["observation.state"]["names"] if n.startswith("cue_"))
    v["state_dim"], v["cue_dim"], v["state_base"] = str(sd), str(cue), str(sd - cue)
    v["cam_hz"] = f"{CFG['sensor']['frame_hz']:g}"
    v["phys_ms"] = f"{float(CFG['sim']['timestep']) * 1000:g}"
    v["train_steps"] = f"{int(pathlib.Path(ck['R1v3']).name):,}"
    v3 = _j(OUT / "f" / "data_v3.json")["datasets"]
    cnt = collections.Counter(e["key"].split("_")[0] for e in _j(ROOT / v3["R1"]["manifest"])["entries"])
    v["n_normal"] = str(cnt["n"])
    v["n_recovery"] = str(sum(c for k, c in cnt.items() if k != "n"))
    v["n_n1"] = str(len(_j(ROOT / v3["N1"]["manifest"])["entries"]))
    ts = float(CFG["sim"]["timestep"])
    v["phys_hz"] = f"{1 / ts:g}"
    sf = CFG["safety_filter"]
    v["sf_dmin_mm"] = f"{sf['d_min_m'] * 1000:g}"
    v["sf_detect_cm"] = f"{sf['d_detect_m'] * 100:g}"
    v["sf_extra_mm"] = f"{float(CFG['runtime_v2']['safety_extra_margin_m']) * 1000:.0f}"
    v["judge_hold_s"] = f"{float(CFG['runtime_v2']['judge']['hold_s']):g}"
    v["time_limit"] = f"{float(CFG['eval']['time_limit_s']):g}"
    # テスト用の試行の数
    v["nat_n"] = str(P["E1"]["n"])
    v["nat_layouts"] = str(P["E1"]["n"] // 3)
    v["p_n"] = str(sets["A"]["P1"]["n"])
    # 主要評価項目
    v["e1_k"], v["e1_n"] = str(P["E1"]["successes"]), str(P["E1"]["n"])
    v["e1_pct"], v["e1_ci"] = pct(P["E1"]["successes"] / P["E1"]["n"]), ci(P["E1"]["success_wilson"])
    v["e2_k"], v["e2_n"] = str(P["E2"]["recovered"]), str(P["E2"]["established"])
    v["e2_pct"], v["e2_ci"] = pct(P["E2"]["recovery_rate"]), ci(P["E2"]["recovery_wilson"])

    def pair(prefix, d, holm=None):
        v[f"{prefix}_x"], v[f"{prefix}_y"] = pct(d["x_rate"]), pct(d["y_rate"])
        v[f"{prefix}_xk"], v[f"{prefix}_yk"] = str(d["x_only"] + d["both"]), str(d["y_only"] + d["both"])
        v[f"{prefix}_xo"], v[f"{prefix}_yo"] = str(d["x_only"]), str(d["y_only"])
        v[f"{prefix}_pairs"], v[f"{prefix}_ci"] = str(d["pairs"]), dci(d)
        v[f"{prefix}_p"], v[f"{prefix}_peq"] = pv(d["mcnemar_exact_p"]), pv_eq(d["mcnemar_exact_p"])
        if holm is not None:
            v[f"{prefix}_holm"], v[f"{prefix}_holmeq"] = pv(holm), pv_eq(holm)
    pair("e3a", P["E3"]["main_A_vs_B"], P["E3"]["main_A_vs_B"]["holm_p"])
    pair("e3s", P["E3"]["sync_C_vs_D"], P["E3"]["sync_C_vs_D"]["holm_p"])
    pair("nat_rn", sec["E3_main"]["nat_success"])
    pair("e4", sec["E4_naive_vs_rtc"]["P1_recovery"])
    pair("nat_nr", sec["E4_naive_vs_rtc"]["nat_success"])
    pair("con_nr", sec["E4_naive_vs_rtc"]["nat_contact"])
    jr = sec["E4_naive_vs_rtc"]["jerk_rms_all"]
    v["jerk_naive"], v["jerk_rtc"], v["jerk_pairs"] = f"{jr['x_median']:.1f}", f"{jr['y_median']:.1f}", str(jr["pairs"])
    pair("e5", sec["E5"]["nat_contact"])           # 本線（フィルタなし）対 フィルタあり。x が本線
    pair("e5s", sec["E5"]["nat_success"])
    for part in ("P2", "P3"):
        s = sets["A"][part]
        v[f"a_{part.lower()}_k"], v[f"a_{part.lower()}_n"] = str(s["recovered"]), str(s["established"])
    # E6（目標位置のキュー）
    e6, e6c = _j(S3 / "e6_R1v3_both.json"), _j(S3 / "e6_R1v3_cue_only.json")
    k6 = round(e6["accuracy_majority"] * e6["pairs"])
    v["e6_k"], v["e6_n"], v["e6_ci"] = str(k6), str(e6["pairs"]), ci(ee._wilson(k6, e6["pairs"]))
    v["e6c_cue"] = str(round(e6c["accuracy_majority"] * e6c["pairs"]))
    v["e6c_text"] = str(round(e6c["follows_instruction_text_majority"] * e6c["pairs"]))
    v["e6c_n"] = str(e6c["pairs"])
    # E7（3 個の連続タスク）
    s7 = e7_summary()
    v["e7_k"], v["e7_n"], v["e7_ci"] = str(s7["all_three"]), str(s7["n"]), ci(ee._wilson(s7["all_three"], s7["n"]))
    v["e7_retry"], v["e7_fn"], v["e7_fp"] = str(s7["retry_completed"]), str(s7["false_neg"]), str(s7["false_pos"])
    top_step, top_n = s7["stopped_at"].most_common(1)[0]
    v["e7_stop2"], v["e7_stop_pos"] = str(top_n), str(top_step + 1)
    v["e7_stopped"] = str(sum(s7["stopped_at"].values()))
    # E9（G1〜G3 の監査）・E10（知覚の精度）
    e9 = rep["new"]["E9"]["total"]
    v["e9_trials"] = f"{e9['trials']:,}"
    v["e9_viol"] = str(e9["g1"] + e9["g2_stops"] + e9["g2_early"] + e9["g3"])
    pa = rep["new"]["E10"]["A"]
    v["perc_med_mm"], v["perc_p95_mm"] = f"{pa['median_m'] * 1000:.0f}", f"{pa['p95_m'] * 1000:.0f}"
    # 動画の場面の時刻（本文の「動画 m:ss〜m:ss」）
    t = 0.0
    for sc in video_scenes():
        v[f"v_{sc['id']}"] = f"{mmss(t)}〜{mmss(t + sc['dur_s'])}"
        t += sc["dur_s"]
    v["v_total"] = mmss(t)
    return v


def fig2(v) -> bool:
    """図 2: 同じ配置での復帰デモあり（上段）となし（下段）。configs/demo/video_s3.yaml の fig2 に場面と時刻を書く。"""
    import yaml
    spec = yaml.safe_load(VIDEO.read_text(encoding="utf-8")).get("fig2")
    if not spec:
        return False
    import av
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(FONT), 26)
    rows = []
    for key, label in (("with", "復帰デモ\nあり"), ("without", "復帰デモ\nなし")):
        c = av.open(str(OUT / "demo_v2" / spec[key]["clip"]))
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        c.close()
        tiles = []
        for tt in spec["times"]:
            img = Image.fromarray(frames[min(int(round(tt * 30)), len(frames) - 1)][120:600, 250:1090]).resize((420, 240))
            dr = ImageDraw.Draw(img)
            dr.rectangle((4, 4, 104, 38), fill=(255, 255, 255))
            dr.text((10, 6), f"{tt:.1f} s", fill=(20, 20, 20), font=font)
            tiles.append(img)
        row = Image.new("RGB", (420 * len(tiles) + 200, 240), (255, 255, 255))
        ImageDraw.Draw(row).multiline_text((8, 80), label, fill=(20, 20, 20), font=ImageFont.truetype(str(FONT), 27), spacing=8)
        for i, im in enumerate(tiles):
            row.paste(im, (200 + 420 * i, 0))
        rows.append(row)
    out = Image.new("RGB", (rows[0].width, 240 * 2 + 8), (255, 255, 255))
    out.paste(rows[0], (0, 0))
    out.paste(rows[1], (0, 248))
    out.save(BUILD / "fig2.png")
    return True


def render(v) -> str:
    def sub(text):
        def rep(m):
            k = m.group(1)
            if k not in v:
                raise KeyError(k)
            return f'<span class="num" data-k="{k}">{html.escape(v[k])}</span>'
        return re.sub(r"\{\{(\w+)\}\}", rep, text)
    svg = sub((PAPER / "fig1.svg").read_text(encoding="utf-8")).replace('<span class="num" data-k=', '<tspan data-k=').replace("</span>", "</tspan>")
    body = (PAPER / "template.html").read_text(encoding="utf-8").replace("{{fig1_svg}}", "@@FIG1@@")
    return sub(body).replace("@@FIG1@@", svg)


def goals_or_exit() -> dict:
    """変更履歴にない版の目標書なら作らない・照合しない（0106）。"""
    try:
        return goals.fingerprint()
    except goals.GoalsMismatch as e:
        print(f"目標書の照合に通らないので止める: {e}", file=sys.stderr)
        raise SystemExit(1)


def cmd_build(a) -> None:
    fp = goals_or_exit()
    BUILD.mkdir(parents=True, exist_ok=True)
    for old in ("fig3.png",):                       # 9/29 版の図（段階 3 では使わない）
        (BUILD / old).unlink(missing_ok=True)
    (BUILD / "goals.json").write_text(json.dumps(fp, ensure_ascii=False, indent=1), encoding="utf-8")
    v = values()
    (BUILD / "values.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    if not fig2(v):
        (BUILD / "fig2.png").unlink(missing_ok=True)
        print("図 2 の場面が未設定（configs/demo/video_s3.yaml の fig2）", file=sys.stderr)
    (BUILD / "paper.html").write_text(render(v), encoding="utf-8")
    pdf = BUILD / "paper.pdf"
    subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    (BUILD / "paper.html").as_uri()], check=True, capture_output=True, timeout=180)
    (BUILD / PDF_NAME).write_bytes(pdf.read_bytes())
    print(BUILD / PDF_NAME, "pages", pdf_pages(pdf))


def pdf_pages(p) -> int:
    return len(re.findall(rb"/Type\s*/Page(?!s)", pathlib.Path(p).read_bytes()))


# 原稿の地の文に書いてよい数字（数字が値ではなく名前や固有の決まりの一部であるもの）
ALLOWED = [r"fig2\.png", r"表 1", r"95 パーセンタイル", r"two3", r"Physical AI 応用 1 講座", r"Haiku 4\.5", r"SmolVLM2", r"Apache-2\.0", r"図 [12]",
           r"[12] 回目", r"工夫 [1-6]", r"3 (色|個)", r"95% 信頼区間", r"1 試行", r"[1-6]\. ", r"7 軸", r"画像 2 枚", r"1 行ずつ",
           r"× 3 色", r"A4", r"RGB-D", r"#[0-9a-f]{6}", r"\d+(\.\d+)?(mm|pt|px)", r"viewBox=\"[^\"]*\"", r"\b\d+(\.\d+)?%?\"",
           r"[xy][12]?=\"[^\"]*\"", r"points=\"[^\"]*\"", r"d=\"[^\"]*\"", r"rotate\([^)]*\)", r"\b(width|height|rx|dx|refX|refY|markerWidth|markerHeight)=\"[^\"]*\"",
           r"opacity:\.\d+", r"stroke-dasharray:[\d ]+", r"stroke-width:[\d.]+", r"font-size:[\d.]+(px|pt)", r"line-height:[\d.]+",
           r"\d+(\.\d+)?(fr|mm)", r"gap: \d+mm", r"margin[^;]*;", r"padding[^;]*;", r"h[123]", r"m1[co]", r"M0,0 L10,5 L0,10 z",
           r"lang=\"ja\"", r"utf-8", r"-webkit", r"0 10 10", r"font-weight: ?\d+", r"border[^;]*;", r"\bgrid-template-columns[^;]*;"]
# 使わない語（開発中に作った語・作業記録・卒研に触れる語）。「手」は手先・手首などは使うので、数と組になった形だけ
# 語の一覧は Unicode エスケープで書く（書き出しの置き換えと語の検索が、この一覧そのものに当たらないように）
FORBIDDEN = ["\u584a", "\u3053\u307e", "\u7a2e(?!\u985e)", "\u5e2f", "\u53f0\u672c", "\u8a98\u767a", "\u6210\u7acb",
             "\u7acb\u3061\u76f4", "\u624b\u304c\u304b\u308a", "\u7d99\u304e\u76ee", "\u9045\u308c",
             "\u6d41\u308c\u306e\u4e00\u81f4", "\u4fdd\u5b58\u70b9", "\u81ea\u7136", "\u901a\u3057",
             "\u4e3b\u306a\u691c\u5b9a", "\u526f\u306e\u6307\u6a19", "(?<!\u4fe1\u983c)\u533a\u9593",
             "[\u7532\u4e59\u4e19]", "\u672c\u7dda", "\u652f\u7dda", "\u76e3\u7763", "\u6c7a\u88c1",
             "\u63b2\u793a\u677f", "\u6d41\u7528\u5143", "\u5352\u7814", "\u5352\u696d\u7814\u7a76", "C:\\\\VLA",
             "\u5b66\u751f", "\u4e88\u5099\u5b9f\u9a13", "(\\d|\u4e07)\\s*\u624b(?![\u5148\u9996\u9806\u6cd5])",
             "\u6a21\u578b",
             # 開発中の記号（決裁の原則 10/07: 本文でも使わない。0147 の 2）
             "(?<![A-Za-z0-9])[RNP][123](?![0-9])", "R1\\+", "[RN][12]v[23]", "(?<![A-Za-z])E(?:[1-9]|10)(?![0-9])", "(?<![A-Za-z])G[1-6](?![0-9])"]


def cmd_check(a) -> None:
    fp = goals_or_exit()
    v = values()
    problems = []
    built_goals = BUILD / "goals.json"
    if not built_goals.is_file() or _j(built_goals) != fp:
        problems.append("組み上げたときの目標書の版が今の版と違う（build をやり直す）")
    if not (BUILD / "fig2.png").is_file():
        problems.append("図 2 が未作成（configs/demo/video_s3.yaml の fig2）")
    # (1) 原稿に手で書いた数字がない
    for name in ("template.html", "fig1.svg"):
        text = (PAPER / name).read_text(encoding="utf-8")
        text = re.sub(r"\{\{\w+\}\}", " ", text)
        text = re.sub(r"<style>.*?</style>", " ", text, flags=re.S)
        for pat in ALLOWED:
            text = re.sub(pat, " ", text)
        for m in re.finditer(r"\d", text):
            problems.append(f"{name}: 手で書いた数字 …{text[max(0, m.start() - 15):m.end() + 15]!r}")
    # (2) 組み上げた HTML の差し込み口の値が、今の結果から計算した値と一致
    built = (BUILD / "paper.html").read_text(encoding="utf-8")
    n_slots = 0
    for m in re.finditer(r'data-k="(\w+)">([^<]*)<', built):
        n_slots += 1
        if html.unescape(m.group(2)) != v[m.group(1)]:
            problems.append(f"値の食い違い {m.group(1)}: {m.group(2)} != {v[m.group(1)]}")
    # (3) 使わない語が 0
    plain = re.sub(r"<style>.*?</style>", " ", built, flags=re.S)
    plain = html.unescape(re.sub(r"<[^>]+>", " ", plain))
    hits = [(p, mm.group(0), plain[max(0, mm.start() - 12):mm.end() + 12]) for p in FORBIDDEN for mm in re.finditer(p, plain)]
    problems += [f"使わない語 {h[1]!r}: …{h[2]}…" for h in hits]
    # (4) ページ数
    pages = pdf_pages(BUILD / "paper.pdf")
    if not 2 <= pages <= 4:
        problems.append(f"ページ数 {pages}（2〜4）")
    res = {"goals": fp, "slots": n_slots, "keys": len(v), "pages": pages, "forbidden_hits": len(hits), "problems": problems}
    (BUILD / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    raise SystemExit(1 if problems else 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    a = ap.parse_args(argv)
    {"build": cmd_build, "check": cmd_check}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
