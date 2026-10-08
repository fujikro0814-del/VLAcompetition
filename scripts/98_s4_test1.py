"""段階 4 テスト 1 の解析の入口。実装 A（src/recovla/eval/test1.py）と実装 B（scripts/98_s4_test1_b.py）を同じ記録に当て、
結果が一致したときだけ判定の 1 枚（JSON と Markdown）を書く（事前登録の案 第 7 節 1・9、掲示板 0155 の 2-7）。

    .venv\\Scripts\\python.exe scripts\\98_s4_test1.py example --out outputs\\s4\\test1\\       # layout・params の雛形を書く
    .venv\\Scripts\\python.exe scripts\\98_s4_test1.py check --layout outputs\\s4\\test1\\layout.json --params outputs\\s4\\test1\\params.json ^
        [--out outputs\\s4\\test1\\result.json --md outputs\\s4\\test1\\result.md]
終了コード: 0 一致して判定した（または、両方が「未完」で一致した。未完なら判定は書かない）/ 1 A と B が一致しない（判定を書かない）/
  2 入力を読めない / 3 未完（入口の点検を満たさない条件がある。A・B の点検の結果は一致）。
一致の基準（事前登録の案 第 7 節 1）: 件数・真偽・文字列は完全に一致。p 値・割合・区間・中央値は相対 1e-9 以内（A は scipy、
  B は自前の計算なので、浮動小数の丸めの差だけを許す）。入口の点検は、条件ごとの合否（ok）と足りない条件の一覧を照らす
  （理由の文は実装ごとに違うので照らさない）。
params の P-n・作者の判断（D1〜D7）の値は、結果の JSON の params にそのまま残る。
"""
import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
REL = 1e-9

EXAMPLE_LAYOUT = {
    "_note": "条件名（outputs/v2eval/<experiment>/<条件>）。model を書けば試行の json のモデル名と照らす。rtc は案 B のときだけ",
    "root": "outputs/v2eval", "experiment": "S4T1",
    "e7": {"v3": {"cond": "E7_R1v3_v3", "model": "R1v3"}, "cur": {"cond": "E7_R1v3_cur", "model": "R1v3"},
           "n1v3_v3": {"cond": "E7_N1v3_v3", "model": "N1v3"}},
    "p1": {"1000": {"R": {"cond": "P1_R1v3", "model": "R1v3"}, "N": {"cond": "P1_N1v3", "model": "N1v3"}},
           "1001": {"R": "P1_R1v3s1001", "N": "P1_N1v3s1001"}, "1002": {"R": "P1_R1v3s1002", "N": "P1_N1v3s1002"}},
    "natural": {"1000": {"R": "nat_R1v3", "N": "nat_N1v3"}, "1001": {"R": "nat_R1v3s1001", "N": "nat_N1v3s1001"},
                "1002": {"R": "nat_R1v3s1002", "N": "nat_N1v3s1002"}},
    "rtc": {"natural": "nat_R1v3_rtc", "p1": "P1_R1v3_rtc"},
}
EXAMPLE_PARAMS = {
    "h1": True, "e7_n": 150, "plan": "A", "guard_mode": "point", "ni_margin": 0.10, "h2_layers": ["1001", "1002"],
    "rtc_p1_n": 50, "e7_band_extended": False, "c4_on_time": None,
    "p_fill": {"P-1": "B1 を採った（例）", "P-2": "ES（例）", "P-3": "B2 を採らない（例）", "P-4": 150, "P-6": "naive",
               "P-7": "1001・1002 とも 2 万手", "_note": "結果で埋まる所（P-1〜P-10）と作者の判断（D1〜D8）を、出どころとともに書く"},
}


def _load_b():
    spec = importlib.util.spec_from_file_location("s4_test1_b", ROOT / "scripts" / "98_s4_test1_b.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _flatten(x, path="", out=None):
    out = {} if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            _flatten(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _flatten(v, f"{path}[{i}]", out)
    else:
        out[path] = x
    return out


def _skip(path: str) -> bool:
    return path.startswith("checks.") and ".problems" in path


def compare(a: dict, b: dict) -> list:
    """一致しない所の列（空なら一致）。"""
    fa = {k: v for k, v in _flatten(json.loads(json.dumps(a))).items() if not _skip(k)}
    fb = {k: v for k, v in _flatten(json.loads(json.dumps(b))).items() if not _skip(k)}
    diffs = [f"A だけにある: {k}" for k in sorted(set(fa) - set(fb))] + [f"B だけにある: {k}" for k in sorted(set(fb) - set(fa))]
    for k in sorted(set(fa) & set(fb)):
        x, y = fa[k], fb[k]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            if x != y:
                diffs.append(f"{k}: A={x!r} B={y!r}")
        elif isinstance(x, int) and isinstance(y, int):
            if x != y:
                diffs.append(f"{k}: A={x} B={y}")
        else:
            if math.isnan(float(x)) or math.isnan(float(y)) or abs(x - y) > REL * max(1.0, abs(x), abs(y)):
                diffs.append(f"{k}: A={x!r} B={y!r}")
    return diffs


def run_check(layout: dict, params: dict) -> tuple:
    from recovla.eval import test1 as T1
    a = T1.analyze(layout, params)
    b = _load_b().run_b(layout, params)
    return a, b, compare(a, b)


def cmd_check(args) -> int:
    from recovla.eval import test1 as T1
    try:
        layout = json.loads(pathlib.Path(args.layout).read_text(encoding="utf-8"))
        params = json.loads(pathlib.Path(args.params).read_text(encoding="utf-8"))
        a, b, diffs = run_check(layout, params)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"入力を読めない: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if diffs:
        print(f"実装 A と B が一致しない（{len(diffs)} 所。判定を書かない）:", file=sys.stderr)
        for d in diffs[:50]:
            print("  " + d, file=sys.stderr)
        return 1
    res = dict(a, double_count={"implementations": ["src/recovla/eval/test1.py", "scripts/98_s4_test1_b.py"], "agree": True,
                                "rule": "件数・真偽は完全一致、実数は相対 1e-9 以内"},
               layout=layout, written=time.strftime("%Y-%m-%d %H:%M:%S"))
    md = T1.summary_md(res)
    out = pathlib.Path(args.out) if args.out else ROOT / "outputs" / "s4" / "test1" / "result.json"
    mdp = pathlib.Path(args.md) if args.md else out.with_suffix(".md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    mdp.write_text(md, encoding="utf-8")
    print(md)
    return 0 if res["status"] == "complete" else 3


def cmd_example(args) -> int:
    d = pathlib.Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "layout.json").write_text(json.dumps(EXAMPLE_LAYOUT, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "params.json").write_text(json.dumps(EXAMPLE_PARAMS, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"書いた: {d / 'layout.json'}、{d / 'params.json'}（値は例。回す前に事前登録の値に直す）")
    return 0


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="A と B を当てて照らし、一致したら判定の 1 枚を書く")
    p.add_argument("--layout", required=True)
    p.add_argument("--params", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--md", default=None)
    p = sub.add_parser("example", help="layout・params の雛形を書く")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"check": cmd_check, "example": cmd_example}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
