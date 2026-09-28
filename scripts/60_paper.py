"""説明資料（PDF、2〜4 ページ）を結果のファイルから作り、照合する。数字は手で書かない。

    .venv\\Scripts\\python.exe scripts\\60_paper.py build     # 値の計算 → 図 2・図 3 → HTML → PDF（Edge のヘッドレス印刷）
    .venv\\Scripts\\python.exe scripts\\60_paper.py check     # 照合: 手で書いた数字がない・値が結果と一致・使わない語が 0・ページ数

原稿は paper/template.html と paper/fig1.svg。数字はすべて {{キー}} の差し込み口で、values() が結果の JSON
（outputs/results/e_report_final.json など、テスト用のシード範囲の値）と設定・学習の記録から計算する。
出力: paper/build/（paper.html・図・values.json）と、paper/build/説明資料_PAI最終課題_<アカウント名>.pdf
"""
import argparse
import html
import json
import math
import pathlib
import re
import subprocess
import sys

import numpy as np

from recovla.common import config

CFG = config.load()
ROOT = config.ROOT
OUT = config.path(CFG["paths"]["outputs"])
RES = OUT / "results"
PL = OUT / "planner"
PAPER = ROOT / "paper"
BUILD = PAPER / "build"
ACCOUNT = "アカウント名"                     # 提出の前に手で直す（ファイル名だけに入る）
PDF_NAME = f"説明資料_PAI最終課題_{ACCOUNT}.pdf"
EDGE = pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
FONT = pathlib.Path(r"C:\Windows\Fonts\BIZ-UDGothicR.ttc")
CKPT_R2 = OUT / "train" / "train_R2_20260927-145256_20260927-145256"


def _j(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def _ee():
    import importlib.util
    spec = importlib.util.spec_from_file_location("e_eval", ROOT / "scripts" / "50_e_eval.py")
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
    """本文で「p {{…eq}}」と書く形: 「= 0.0044」または「< 0.001」。"""
    s = pv(p)
    return s if s.startswith("<") else f"= {s}"


def values() -> dict:
    ee = _ee()
    fin = _j(RES / "e_report_final.json")
    P, sec = fin["primary"], fin["secondary"]
    v = {}
    # 設定・学習の記録
    rt, sim, pl = CFG["runtime"], CFG["sim"], CFG["planner"]
    fps = int(CFG["convert"]["fps"])
    v["policy_hz"] = str(fps)
    v["cam_fps"] = str(fps)
    v["infer_period_s"] = f"{int(rt['exec_interval']) / fps:g}"
    v["delay_d"] = str(int(rt["delay_steps"]))
    v["delay_s"] = f"{int(rt['delay_steps']) / fps:g}"
    pcfg = _j(CKPT_R2 / "checkpoints" / "010000" / "pretrained_model" / "config.json")
    info = _j(OUT / "datasets" / "R2cue_20260927-140453_statsR1" / "meta" / "info.json")
    v["chunk"] = str(pcfg["chunk_size"])
    v["chunk_s"] = f"{pcfg['chunk_size'] / fps:g}"
    v["act_dim"] = str(info["features"]["action"]["shape"][0])
    sd = info["features"]["observation.state"]["shape"][0]
    cue = sum(1 for n in info["features"]["observation.state"]["names"] if n.startswith("cue_"))
    v["state_dim"], v["cue_dim"], v["state_base"] = str(sd), str(cue), str(sd - cue)
    ts = float(sim["timestep"])
    v["phys_ms"] = f"{ts * 1000:g}"
    v["phys_hz"] = f"{1 / ts:g}"
    sf = CFG["safety_filter"]
    v["sf_dmin_mm"] = f"{sf['d_min_m'] * 1000:g}"
    v["sf_detect_cm"] = f"{sf['d_detect_m'] * 100:g}"
    v["judge_hold_s"] = f"{float(pl['completion_hold_s']):g}"
    v["step_timeout"] = f"{float(pl['step_timeout_s']):g}"
    v["llm_temp"] = f"{float(pl['temperature']):g}"
    v["time_limit"] = f"{float(CFG['eval']['time_limit_s']):g}"
    m1 = _j(OUT / "manifests" / "R1_20260926-113711.json")
    import collections
    cnt = collections.Counter(e["key"].split("_")[0] for e in m1["entries"])
    v["n_normal"] = str(cnt["n"])
    v["n_recovery"] = str(sum(c for k, c in cnt.items() if k != "n"))
    v["n_n1"] = str(len(_j(OUT / "manifests" / "N1_20260926-113711.json")["entries"]))
    v["n_r2"] = str(len(_j(OUT / "manifests" / "R2_20260927-140453.json")["entries"]) - len(m1["entries"]))
    # テスト用の試行の数
    st = ee.STAGES["final"]
    v["nat_layouts"] = st["nat"].split(":")[2]
    v["nat_n"] = str(P["E1"]["n"])
    v["p_n"] = st["P1"].split(":")[2]
    # 主要評価項目
    v["e1_k"], v["e1_n"] = str(P["E1"]["successes"]), str(P["E1"]["n"])
    v["e1_pct"], v["e1_ci"] = pct(P["E1"]["successes"] / P["E1"]["n"]), ci(P["E1"]["success_wilson"])
    v["e2_k"], v["e2_n"] = str(P["E2"]["recovered"]), str(P["E2"]["established"])
    v["e2_pct"], v["e2_ci"] = pct(P["E2"]["recovery_rate"]), ci(P["E2"]["recovery_wilson"])

    def pair(prefix, d, holm=None):
        v[f"{prefix}_x"], v[f"{prefix}_y"] = pct(d["x_rate"]), pct(d["y_rate"])
        v[f"{prefix}_pairs"], v[f"{prefix}_ci"] = str(d["pairs"]), dci(d)
        v[f"{prefix}_p"], v[f"{prefix}_peq"] = pv(d["mcnemar_exact_p"]), pv_eq(d["mcnemar_exact_p"])
        if holm is not None:
            v[f"{prefix}_holm"], v[f"{prefix}_holmeq"] = pv(holm), pv_eq(holm)
    pair("e3a", P["E3"]["main_A_vs_B"], P["E3"]["main_A_vs_B"]["holm_p"])
    pair("e3s", P["E3"]["sync_C_vs_D"], P["E3"]["sync_C_vs_D"]["holm_p"])
    pair("e4", P["E4"])
    pair("e5", P["E5"])
    v["e5_x_k"] = str(P["E5"]["both"] + P["E5"]["x_only"])
    v["e5_y_k"] = str(P["E5"]["both"] + P["E5"]["y_only"])
    v["e5_x"], v["e5_y"] = f"{v['e5_x_k']}/{P['E5']['pairs']}", f"{v['e5_y_k']}/{P['E5']['pairs']}"
    v["e5_blocked"] = str(len(sec["E5"]["blocked_by_filter"]))
    pair("e8", P["E8"])
    pair("e8p3", sec["E8"]["P3_recovery"])
    pair("e8nat", sec["E8"]["nat_success"])
    pair("nat_rn", sec["E3_main"]["nat_success"])
    pair("nat_nr", sec["E4_naive_vs_rtc"]["nat_success"])
    pair("sync_p123", sec["E4_sync_vs_naive"]["P123_recovery"])
    sj = sec["E4_naive_vs_rtc"]["seam_jump_mean_all"]
    v["seam_naive"], v["seam_rtc"], v["seam_pairs"] = f"{sj['x_median']:.3f}", f"{sj['y_median']:.3f}", str(sj["pairs"])
    p2 = [s["P2"]["recovery_rate"] for s in fin["sets"].values() if s["P2"]["recovery_rate"] is not None]
    v["p2_max"] = pct(max(p2))
    # E6
    e6, e6c = _j(OUT / "k1" / "e6_R2_final.json"), _j(OUT / "k1" / "e6_R2_final_cue_only.json")
    k6 = round(e6["accuracy_majority"] * e6["pairs"])
    v["e6_k"], v["e6_n"], v["e6_ci"] = str(k6), str(e6["pairs"]), ci(ee._wilson(k6, e6["pairs"]))
    v["e6_cue"] = pct(e6c["accuracy_majority"])
    # E7
    llm, jd, e7 = _j(PL / "llm_final_sentences_v1.json"), _j(PL / "judge_final.json"), _j(PL / "e7_summary_E7_final.json")
    v["llm_k"], v["llm_n"], v["llm_ci"] = str(llm["correct"]), str(llm["n"]), ci(ee._wilson(llm["correct"], llm["n"]))
    v["llm_crit"] = "29"
    v["judge_k"], v["judge_n"] = str(jd["agree"]), str(jd["selected"])
    v["judge_pct"], v["judge_ci"] = f"{100 * jd['agreement']:.1f}%", ci(ee._wilson(jd["agree"], jd["selected"]))
    v["judge_crit"] = "98"
    sel = set(jd["selected_ids"])
    v["judge_dis"] = str(sum(1 for c in jd["cases"] if c["id"] in sel and c["truth"] != c["judge"]))
    pr = e7["primary"]
    v["e7_k"], v["e7_n"], v["e7_ci"] = str(pr["all_three"]), str(pr["n"]), ci(pr["wilson95"])
    v["e7_fail"] = str(pr["n"] - pr["all_three"])
    v["e7_false"] = str(e7["summary"]["judge_false_complete"])
    v["e7_retry"] = str(e7["summary"]["retry_completed_steps"])
    rc = _j(PL / "return_compare_final.json")["pairs"][0]
    v["nr_k"], v["nr_n"] = str(rc["before_summary"]["all_three"]), str(rc["before_summary"]["n"])
    v["nr_retry"] = str(rc["before_summary"]["retry_completed_steps"])
    # 図 2（動画の場面の回し直し）
    d2 = _j(OUT / "demo" / "nat_110001_R1" / "meta.json")
    v["fig2_miss"] = f"{[e['t'] for e in d2['events'] if e['kind'] == 'grasp_miss'][0]:.0f}"
    v["fig2_succ"] = f"{d2['t_success']:.0f}"
    return v


# 基準の値（29 文・98%）は評価の一覧（50_e_eval.py の E7）に書いた基準。照合で一覧の文と突き合わせる
CRITERIA = {"llm_crit": "30 文中 29 文以上", "judge_crit": "基準 98% 以上"}


def fig2(v) -> None:
    import av
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(FONT), 26)
    rows = []
    for clip, label in (("nat_110001_R1", "R1\n（復帰デモ\nあり）"), ("nat_110001_N1", "N1\n（復帰デモ\nなし）")):
        m = _j(OUT / "demo" / clip / "meta.json")
        miss = [e["t"] for e in m["events"] if e["kind"] == "grasp_miss"][0]
        t_end = m["t_end"]
        times = [1.0, miss, miss + 3.0, t_end - 0.1]
        c = av.open(str(OUT / "demo" / clip / "presentation.mp4"))
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        c.close()
        tiles = []
        for t in times:
            img = Image.fromarray(frames[min(int(round(t * 30)), len(frames) - 1)][120:600, 250:1090])
            img = img.resize((420, 240))
            dr = ImageDraw.Draw(img)
            dr.rectangle((4, 4, 104, 38), fill=(255, 255, 255))
            dr.text((10, 6), f"{t:.1f} s", fill=(20, 20, 20), font=font)
            tiles.append(img)
        row = Image.new("RGB", (420 * 4 + 200, 240), (255, 255, 255))
        ImageDraw.Draw(row).multiline_text((8, 80), label.replace("（", "\n（"), fill=(20, 20, 20), font=ImageFont.truetype(str(FONT), 27), spacing=8)
        for i, im in enumerate(tiles):
            row.paste(im, (200 + 420 * i, 0))
        rows.append(row)
    out = Image.new("RGB", (rows[0].width, 240 * 2 + 8), (255, 255, 255))
    out.paste(rows[0], (0, 0))
    out.paste(rows[1], (0, 248))
    out.save(BUILD / "fig2.png")


def fig3(v) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
    ee = _ee()
    fin = _j(RES / "e_report_final.json")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for s, lab, col in (("C", "同期実行", "#2d5f8b"), ("A", "非同期実行（naive）", "#c0392b"), ("E", "非同期実行（RTC）", "#1f6b5c")):
        S = ee._set_rows("final", s)
        allr = [r for p in ee.PARTS for r in S[p].values()]
        seam = [r["seam_jump_mean"] for r in allr if ee._ok(r["seam_jump_mean"])]
        q = np.percentile(seam, [25, 50, 75])
        p1 = fin["sets"][s]["P1"]
        est = [r for p in ("P1", "P2", "P3") for r in S[p].values() if ee.EST(r)]
        rec = sum(ee.SUCC(r) for r in est)
        lo, hi = p1["recovery_wilson"]
        ax.errorbar(q[1], p1["recovery_rate"], xerr=[[q[1] - q[0]], [q[2] - q[1]]],
                    yerr=[[max(0, p1["recovery_rate"] - lo)], [max(0, hi - p1["recovery_rate"])]], fmt="o", color=col,
                    capsize=3, label=f"{lab}  P1 {p1['recovered']}/{p1['established']}")
        ax.plot(q[1], rec / len(est), "o", mfc="none", color=col, ms=8)
    ax.set_xlabel("チャンク境界での速度の不連続 [m/s]")
    ax.set_ylabel("復帰成功率（塗り: P1、白抜き: P1〜P3）")
    ax.set_ylim(-0.03, 0.75)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(BUILD / "fig3.png", dpi=220)
    plt.close(fig)


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


def cmd_build(a) -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    v = values()
    (BUILD / "values.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    fig2(v)
    fig3(v)
    (BUILD / "paper.html").write_text(render(v), encoding="utf-8")
    pdf = BUILD / "paper.pdf"
    subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    (BUILD / "paper.html").as_uri()], check=True, capture_output=True, timeout=180)
    (BUILD / PDF_NAME).write_bytes(pdf.read_bytes())
    print(BUILD / PDF_NAME, "pages", pdf_pages(pdf))


def pdf_pages(p) -> int:
    return len(re.findall(rb"/Type\s*/Page(?!s)", pathlib.Path(p).read_bytes()))


# 原稿の地の文に書いてよい数字（数字が値ではなく名前や固有の決まりの一部であるもの）
ALLOWED = [r"fig[23]\.png", r"表 1", r"two3", r"Physical AI 応用 1 講座", r"Haiku 4\.5", r"SmolVLM2", r"Apache-2\.0", r"図 [123]", r"P[123]", r"R[12]\+?",
           r"N1", r"[12] 段目", r"[12] 回目", r"工夫 4", r"3 (色|個)", r"95% 信頼区間", r"1 試行", r"[1-5]\. ", r"7 軸", r"画像 2 枚", r"1 行ずつ",
           r"× 3 色", r"A4", r"#[0-9a-f]{6}", r"\d+(\.\d+)?(mm|pt|px)", r"viewBox=\"[^\"]*\"", r"\b\d+(\.\d+)?%?\"",
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
             "\u6a21\u578b"]


def cmd_check(a) -> None:
    v = values()
    problems = []
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
    # 基準の値が評価の一覧の文と同じ
    doc = (ROOT / "scripts" / "50_e_eval.py").read_text(encoding="utf-8")
    for k, phrase in CRITERIA.items():
        if phrase not in doc:
            problems.append(f"基準 {k} が評価の一覧（50_e_eval.py）に見つからない: {phrase}")
    # (3) 使わない語が 0
    plain = re.sub(r"<style>.*?</style>", " ", built, flags=re.S)
    plain = html.unescape(re.sub(r"<[^>]+>", " ", plain))
    hits = [(p, mm.group(0), plain[max(0, mm.start() - 12):mm.end() + 12]) for p in FORBIDDEN for mm in re.finditer(p, plain)]
    problems += [f"使わない語 {h[1]!r}: …{h[2]}…" for h in hits]
    # (4) ページ数
    pages = pdf_pages(BUILD / "paper.pdf")
    if not 2 <= pages <= 4:
        problems.append(f"ページ数 {pages}（2〜4）")
    res = {"slots": n_slots, "keys": len(v), "pages": pages, "forbidden_hits": len(hits), "problems": problems}
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
