"""段階 4 の時間ごとの採点（recovla.eval.time_scoring）。60 s で回した記録から 30・45・60 s の成否と累積の成功率の図を作る。

    .venv\\Scripts\\python.exe scripts\\98_s4_time_report.py --dirs <条件A> <条件B> --labels 復帰デモあり 復帰デモなし ^
        --out outputs\\results\\s4_time_nat.json --svg outputs\\results\\s4_time_nat.svg
    .venv\\Scripts\\python.exe scripts\\98_s4_time_report.py --dirs <条件A>_P1 <条件B>_P1 --labels 復帰デモあり 復帰デモなし ^
        --induced --out outputs\\results\\s4_time_p1.json --svg outputs\\results\\s4_time_p1.svg

読むもの: --dirs の各フォルダの trial_*.json（1 つか 2 つ。2 つなら種と目標の色で対にして比べる）。
  フォルダに trial_*.json がなく run_*.json（3 個の連続タスク）があれば、手順ごとの成功時刻の分布と、3 個そろった時刻
  （all_three_in_box が真の試行の truth_success_t の最大。0155 の 1-2）の曲線を JSON に出す（連続タスクは制限時間で経過が
  変わるので、30 s での採点し直しはしない。time_scoring の冒頭）。
数える前の確かめ（time_scoring.check_records）: 記録に誘発の試行があるのに --induced が無い、--induced なのに誘発の試行が
  無い、自然と誘発が混ざっている、制限時間が 2 種類以上、環境が 2 つ以上なら、数えずに終了コード 2。
  誘発の分母は T 秒より前に成立した試行（t_established < T）。曲線は時刻ごとの分母で描き、図に分母を書く（0155 の 1-1・1-3）。
書くもの: --out の JSON（primary＝30 s の主な指標、secondary＝45・60 s と曲線、補正なし）と、--svg の図（横に時間、
  縦に累積の成功率の階段、30 s と 60 s に縦の線）。図の語は 60_paper.py の FORBIDDEN に当たらないかを確かめ、
  当たれば標準エラーに書いて終了コード 1。
--labels は図と JSON に出す名前。利用者向けの日本語にする（モデルの記号は書かない）。省くと「条件 A」「条件 B」
（--svg を付けるときは必須）。
"""
import argparse
import ast
import html
import json
import pathlib
import re
import sys

from recovla.eval import time_scoring as TS

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER_SCRIPT = ROOT / "scripts" / "60_paper.py"
# 見た目は説明資料（paper/template.html の色と字）に合わせる
INK, MUTED, RULE = "#1d2126", "#555c64", "#c9ced3"
SERIES = (("#2d5f8b", ""), ("#955619", "6 4"))           # 線の色と破線（2 本目は破線にして色だけに頼らない）
FONT = '"BIZ UDPGothic", "Yu Gothic", "Meiryo", sans-serif'


# ------------------------------------------------------------------ 図

def curve_svg(rep: dict, marks=(30.0, 60.0)) -> str:
    """累積の成功率の階段の曲線（2 条件を重ねる）。rep は time_scoring.time_report の結果。"""
    induced = rep["settings"]["induced_only"]
    W, H = 640, (420 if induced else 400)                      # 誘発の条件は凡例の下に分母の行を足す
    x0, x1, y0, y1 = 72, 612, 56, 316
    curves = rep["secondary"]["curves"]
    t_max = max(c["t"][-1] for c in curves.values())

    def X(t):
        return x0 + (x1 - x0) * t / t_max

    def Y(r):
        return y1 - (y1 - y0) * r

    kind = "意図的な失敗" if induced else "通常の試行"
    # 誘発の条件の曲線は時刻ごとの分母（その時刻より前に失敗が起きた試行。0155 の 1-3）。図にそう書き、凡例に 30・60 秒の分母を出す
    sub = "分母は各時刻より前に意図的な失敗が起きた試行（時刻で変わる。凡例に 30 秒と 60 秒の分母）" if induced else "分母はすべての試行"
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">',
         f"<title>{kind}の累積の成功率</title>",
         f"<style>text{{font-family:{FONT};fill:{INK}}} .tk{{font-size:12px;fill:{MUTED}}} .ax{{font-size:13px}} "
         f".tt{{font-size:15px;font-weight:700}} .ts{{font-size:12px;fill:{MUTED}}} .lg{{font-size:13px}} .mk{{font-size:12px;fill:{MUTED}}}</style>",
         f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text class="tt" x="{x0}" y="24">{kind}の累積の成功率</text>',
         f'<text class="ts" x="{x0}" y="42">{sub}</text>']
    for i in range(5):                                          # 横の目盛り線（0・25・50・75・100%）
        r = i / 4
        o.append(f'<line x1="{x0}" y1="{Y(r):.1f}" x2="{x1}" y2="{Y(r):.1f}" stroke="{RULE}" stroke-width="1"/>')
        o.append(f'<text class="tk" x="{x0 - 8}" y="{Y(r) + 4:.1f}" text-anchor="end">{round(100 * r)}%</text>')
    for t in range(0, int(t_max) + 1, 10):
        o.append(f'<text class="tk" x="{X(t):.1f}" y="{y1 + 18}" text-anchor="middle">{t}</text>')
    o.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{INK}" stroke-width="1"/>')
    o.append(f'<text class="ax" x="{(x0 + x1) / 2:.1f}" y="{y1 + 40}" text-anchor="middle">経過時間（秒）</text>')
    o.append(f'<text class="ax" x="18" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" '
             f'transform="rotate(-90 18 {(y0 + y1) / 2:.1f})">累積の成功率</text>')
    for m in marks:                                             # 30 s（主な指標）と 60 s
        if m <= t_max + TS.EPS:
            o.append(f'<line x1="{X(m):.1f}" y1="{y0}" x2="{X(m):.1f}" y2="{y1}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="2 3"/>')
            o.append(f'<text class="mk" x="{X(m):.1f}" y="{y0 - 6}" text-anchor="middle">{m:g} 秒</text>')
    for i, (s, c) in enumerate(curves.items()):
        pts = [(t, r) for t, r in zip(c["t"], c["rate"]) if r is not None]     # 分母が 0 の時刻（誘発の始めのほう）は描かない
        if not c["n"] or not pts:
            continue
        col, dash = SERIES[i % len(SERIES)]
        d = [f"M{X(pts[0][0]):.1f},{Y(pts[0][1]):.1f}"]
        for t, r in pts[1:]:                                    # 右連続の階段: t で上がる
            d.append(f"H{X(t):.1f}V{Y(r):.1f}")
        da = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<path d="{"".join(d)}" fill="none" stroke="{col}" stroke-width="2.2"{da}/>')
        p = rep["primary"]["conditions"][s]
        ly, lx = y1 + 62, x0 + 280 * i                          # 凡例は図の下に横に並べる
        o.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 28}" y2="{ly - 4}" stroke="{col}" stroke-width="2.2"{da}/>')
        o.append(f'<text class="lg" x="{lx + 36}" y="{ly}">{html.escape(c["label"])}（{TS.PRIMARY_LIMIT_S:g} 秒で '
                 f'{p["successes"]}/{p["n"]}）</text>')
        if induced:                                             # 時刻ごとの分母（30 秒と曲線の終わり）
            n_end = c["n_at"][-1]
            o.append(f'<text class="mk" x="{lx + 36}" y="{ly + 16}">分母 {TS.PRIMARY_LIMIT_S:g} 秒 {p["n"]}・'
                     f'{c["t"][-1]:g} 秒 {n_end}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


# ------------------------------------------------------------------ 使わない語（60_paper.py の FORBIDDEN を読むだけ）

def forbidden_patterns(path=PAPER_SCRIPT) -> list:
    """60_paper.py を実行せずに FORBIDDEN の一覧だけを取り出す（あちらは読み込むときに設定と結果を読むため）。"""
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "FORBIDDEN" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{path} に FORBIDDEN がない")


def forbidden_hits(svg: str, patterns=None) -> list:
    """図の中の語（タグと style を除いた文字）で使わない語に当たったもの。60_paper.py の check と同じ取り出し方。"""
    patterns = forbidden_patterns() if patterns is None else patterns
    plain = re.sub(r"<style>.*?</style>", " ", svg, flags=re.S)
    plain = html.unescape(re.sub(r"<[^>]+>", " ", plain))
    return [(p, m.group(0)) for p in patterns for m in re.finditer(p, plain)]


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--limits", nargs="+", type=float, default=list(TS.DEFAULT_LIMITS))
    ap.add_argument("--induced", action="store_true",
                    help="分母を T 秒より前に失敗が成立した試行に絞る（誘発の記録なのに付けない・自然の記録に付けると止まる）")
    ap.add_argument("--step", type=float, default=TS.GRID_STEP_S, help="曲線の格子の刻み [s]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--svg", default=None)
    a = ap.parse_args(argv)
    if len(a.dirs) > 2:
        ap.error("--dirs は 1 つか 2 つ")
    if a.svg and not a.labels:
        ap.error("--svg には --labels が要る（図に仮の名前が出ないように）")
    labels = a.labels or ["条件 A", "条件 B"][:len(a.dirs)]
    if len(labels) != len(a.dirs):
        ap.error("--labels の数が --dirs と違う")
    trials = [TS.load_trials(d) for d in a.dirs]
    if not any(trials):
        runs = [TS.load_runs(d) for d in a.dirs]
        if not any(runs):
            print(f"trial_*.json も run_*.json もない: {a.dirs}", file=sys.stderr)
            return 1
        res = {"tasks": {lab: TS.task_step_times(r) for lab, r in zip(labels, runs)},
               "all_three": {lab: TS.task_all_three(r) for lab, r in zip(labels, runs)}}
        if a.svg:
            print("連続タスクには図を作らない（採点し直しをしないため。3 個そろった時刻の曲線は JSON の all_three）", file=sys.stderr)
    else:
        # --induced の付け忘れ・付け違いは数えずに止める（査読の軽微 3。96 の score は誘発の試行を記録から見分ける）
        probs = [f"{d}: {p}" for d, rs in zip(a.dirs, trials) for p in TS.check_records(rs, a.induced)]
        if probs:
            for p in probs:
                print(p, file=sys.stderr)
            return 2
        res = TS.time_report(trials[0], trials[1] if len(trials) > 1 else None, labels, a.limits, a.induced, a.step)
    res["dirs"] = [str(d) for d in a.dirs]
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(text, encoding="utf-8")
    rc = 0
    if a.svg and "primary" in res:
        svg = curve_svg(res)
        pathlib.Path(a.svg).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.svg).write_text(svg, encoding="utf-8")
        hits = forbidden_hits(svg)
        for p, w in hits:
            print(f"図に使わない語 {w!r}（{p}）", file=sys.stderr)
        rc = 1 if hits else 0
    print(json.dumps(res.get("primary", res), ensure_ascii=False, indent=1) if not a.out else a.out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
