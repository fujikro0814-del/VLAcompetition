"""段階 4 の関門 1 の判定（recovla.eval.gate1。二重集計の片方）。測った値を configs/s4_gates.json に機械的に当てはめる。

    .venv\\Scripts\\python.exe scripts\\98_s4_gate1.py --gates configs\\s4_gates.json --input outputs\\s4\\gate1_input.json ^
        --out outputs\\s4\\gate1_result.json --md outputs\\s4\\gate1_summary.md
    （入力を手元の診断の出力から組むとき。--input の代わりに。組んだ入力は --make-input に書く）
    .venv\\Scripts\\python.exe scripts\\98_s4_gate1.py --gates configs\\s4_gates.json --make-input outputs\\s4\\gate1_input.json ^
        --rtc-metrics outputs\\s4\\d_rtc\\metrics.json --rtc-shadow outputs\\s4\\d_rtc\\shadow.json --x2 <98_s4_x2.py の summary.json> ^
        --e7-summary <98_s4_d_e7.py summary の出力> --start-summary <98_s4_d_start.py summary の出力> ^
        --recovery-score <98_s4_d_recovery.py score の出力> --k-input <関門 K の入力の JSON> ^
        --out outputs\\s4\\gate1_result.json --md outputs\\s4\\gate1_summary.md

読むもの: --gates（s4_gates.json。読むだけ）と、--input の測った値の JSON（形は docs/stage4/gate1_input_schema.md）。
  --input の代わりに --rtc-metrics などを渡すと、手元の診断の出力の値（件数・中央値）だけを写して入力を組む
  （判定の部分は使わない）。関門 K の入力は形が決まった出力がないので、--k-input に入力の定義の K の形で渡す。
書くもの: --out の結果の JSON（関門ごとの合否・使った値と閾値・選んだ枝、候補の選び方、最後の結論、読めなかった規則と
  機械が読める形で書かれていない規則の一覧（ファイルの行つき）、決まりのファイルの SHA-256、使った入力）と、
  --md の 1 枚の要約（利用者向けの日本語）。要約の語は 60_paper.py の FORBIDDEN に当たらないかを確かめ、
  当たれば標準エラーに書いて終了コード 1。
終了コード: 0 結論が出た / 3 判定できない所があり、その結論を止めた（ファイルは書く）/ 1 要約に使わない語 / 2 入力を読めない。
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

from recovla.eval import gate1 as G1

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def _write(p, text):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def forbidden_hits(text: str) -> list:
    """60_paper.py の FORBIDDEN に当たった語（98_s4_time_report.py の取り出し方をそのまま使う。60_paper.py は読み込まない）。"""
    spec = importlib.util.spec_from_file_location("s4_time_report", ROOT / "scripts" / "98_s4_time_report.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.forbidden_hits(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gates", default=str(ROOT / "configs" / "s4_gates.json"))
    ap.add_argument("--input", default=None, help="測った値の JSON（docs/stage4/gate1_input_schema.md）")
    ap.add_argument("--out", default=None, help="結果の JSON")
    ap.add_argument("--md", default=None, help="1 枚の要約（md）")
    src = ap.add_argument_group("入力を手元の診断の出力から組む（--input の代わり）")
    src.add_argument("--rtc-metrics", default=None)
    src.add_argument("--rtc-shadow", default=None)
    src.add_argument("--x2", default=None)
    src.add_argument("--e7-summary", default=None)
    src.add_argument("--start-summary", default=None)
    src.add_argument("--recovery-score", default=None)
    src.add_argument("--k-input", default=None)
    src.add_argument("--make-input", default=None, help="組んだ入力を書く先")
    a = ap.parse_args(argv)
    parts = {k: getattr(a, k) for k in ("rtc_metrics", "rtc_shadow", "x2", "e7_summary", "start_summary", "recovery_score", "k_input")}
    given = {k: v for k, v in parts.items() if v}
    if a.input and given:
        ap.error("--input と、診断の出力から組む引数は一緒に使えない")
    if not a.input and not given:
        ap.error("--input か、診断の出力（--rtc-metrics など）が要る")
    if given.get("rtc_shadow") or given.get("x2"):
        if not given.get("rtc_metrics"):
            ap.error("--rtc-shadow・--x2 には --rtc-metrics が要る")
    try:
        gates = G1.load_gates(a.gates)
        if a.input:
            raw = pathlib.Path(a.input).read_bytes()
            inp = json.loads(raw.decode("utf-8"))
        else:
            rd = {k: _read(v) for k, v in given.items()}
            inp = G1.build_input(rtc_metrics=rd.get("rtc_metrics"), rtc_shadow=rd.get("rtc_shadow"), x2=rd.get("x2"),
                                 e7_summary=rd.get("e7_summary"), start_summary=rd.get("start_summary"),
                                 recovery_score=rd.get("recovery_score"), k=rd.get("k_input"),
                                 source={k: str(v) for k, v in given.items()})
            raw = json.dumps(inp, ensure_ascii=False, indent=1).encode("utf-8")
            if a.make_input:
                _write(a.make_input, raw.decode("utf-8"))
        res = G1.evaluate(gates, inp)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"入力を読めない: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    res["input_sha256"] = hashlib.sha256(raw).hexdigest()
    res["input"] = inp
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        _write(a.out, text)
    rc = 0 if res["conclusion"]["status"] == "determined" else 3
    if a.md:
        md = G1.summary_md(res)
        _write(a.md, md)
        hits = forbidden_hits(md)
        for p, w in hits:
            print(f"要約に使わない語 {w!r}（{p}）", file=sys.stderr)
        if hits:
            rc = 1
    print(json.dumps(res["conclusion"], ensure_ascii=False, indent=1))
    for p in res["rules_unreadable"]:
        print(f"[gate1] 読めなかった規則 {p['id']}（{p['where']}）: {p['why']}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
