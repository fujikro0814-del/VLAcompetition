"""段階 4 束 4 の解析の入口（関門 2・関門 3・テスト 2）。実装 A（src/recovla/eval/b4.py）と実装 B（scripts/98_s4_b4_b.py）を同じ記録に
当て、結果が一致したときだけ判定の 1 枚（JSON と Markdown）を書く（事前登録 v2 の草案 第 7 節 1、掲示板 0155 の 2-7）。

    .venv\\Scripts\\python.exe scripts\\98_s4_b4_eval.py layout --phase G2 --out outputs\\s4\\b4_eval\\layout_G2.json
    .venv\\Scripts\\python.exe scripts\\98_s4_b4_check.py example --out outputs\\s4\\b4_eval\\        # params の雛形
    .venv\\Scripts\\python.exe scripts\\98_s4_b4_check.py check --layout outputs\\s4\\b4_eval\\layout_G2.json --params outputs\\s4\\b4_eval\\params.json ^
        [--out outputs\\s4\\b4_eval\\result_G2.json --md ...] [--ledger-json <97 の結果>] [--no-git]
終了コード: 0 一致して判定した / 1 A と B が一致しない（判定を書かない）/ 2 入力を読めない / 3 未完（入口の点検を満たさない。
  A・B の点検の結果は一致）。
一致の基準: 件数・真偽・文字列は完全に一致。p 値は相対 1e-9、割合・区間は相対 1e-9（0 の近くだけ絶対 1e-12）。入口の点検は、
  条件ごとの合否・HEAD・保存点の SHA-256・環境と、足りない条件・条件をまたぐ点検の合否を照らす（理由の文は照らさない）。
条件をまたぐ入口の点検（A・B が一致した後に、ここで 1 回だけ。満たさなければ未完）:
  2-5 版: 全条件の試行の HEAD が 2 つ以上なら、98_s4_d_audit.py の order_heads・compare_heads（子が読み込むファイル）で、いちばん古い
      HEAD にあるファイルが後の HEAD で変わっていないことを git で確かめる（98_s4_b4_eval.py は試行の b4.files_sha256 を A・B が照らす）。
  2-6 台帳: 97_s4_ledger_check.py の build_report（または --ledger-json。記録より新しいこと）で parse_errors が 0。
"""
import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
REL = 1e-9
ABS = 1e-12
P_KEYS = {"p", "p_holm"}
LEDGER = ROOT / "docs" / "種の台帳.md"
EXAMPLE_PARAMS = {
    "guard_mode": "point", "h2_in_family": True, "g3_with_r1v3s1001": True,
    "ckpt_sha256": {"_note": "掲示した保存点の SHA-256（P-2）。書けば記録の値と照らす（空なら照らさない）"},
    "p_fill": {"_note": "結果で埋まる所（P-1〜P-3）と作者の判断（D2・D4）を、出どころとともに書く"},
}


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
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
    return ".problems" in path or path.startswith("problems")


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
            tol = 0.0 if k.rsplit(".", 1)[-1] in P_KEYS else ABS
            if math.isnan(float(x)) or math.isnan(float(y)) or not math.isclose(x, y, rel_tol=REL, abs_tol=tol):
                diffs.append(f"{k}: A={x!r} B={y!r}")
    return diffs


def run_check(layout: dict, params: dict) -> tuple:
    from recovla.eval import b4 as B4
    a = B4.analyze(layout, params)
    b = _load(ROOT / "scripts" / "98_s4_b4_b.py", "s4_b4_b").run_b(layout, params)
    return a, b, compare(a, b)


# ---------------------------------------------------------------- 条件をまたぐ入口の点検（0155 の 2-5・2-6）
def check_heads(heads: list, use_git: bool = True) -> dict:
    """heads: 最初に現れた順の HEAD。2 つ以上なら、子が読み込むファイルが同じかを git で照らす。"""
    if "None" in heads:
        return {"ok": False, "heads": heads, "note": "HEAD の無い試行がある"}
    if len(heads) == 1:
        return {"ok": True, "heads": heads, "note": "HEAD が 1 つ"}
    if not heads:
        return {"ok": False, "heads": heads, "note": "HEAD がない"}
    if not use_git:
        return {"ok": False, "heads": heads, "note": "HEAD が 2 つ以上（--no-git では照らさない）"}
    try:
        au = _load(ROOT / "scripts" / "98_s4_d_audit.py", "s4_audit_for_b4")
        order = au.order_heads(ROOT, {h: [i] for i, h in enumerate(heads)})
        cmp = au.compare_heads(ROOT, order, "B4")
    except Exception as e:                              # noqa: BLE001
        return {"ok": False, "heads": heads, "note": f"git で照らせない: {type(e).__name__}: {e}"}
    return {"ok": not cmp["changed"], "heads": order, "changed": cmp["changed"], "added_only": cmp["added_only"],
            "note": "子が読み込むファイルは同じ" if not cmp["changed"] else "HEAD の間で子が読み込むファイルが違う"}


def _newest_mtime(dirs: list) -> float:
    t = 0.0
    for d in dirs:
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() == ".json":
                    t = max(t, p.stat().st_mtime)
    return t


def check_ledger(ledger_json, dirs: list) -> dict:
    if ledger_json:
        p = pathlib.Path(ledger_json)
        try:
            rep = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception as e:                          # noqa: BLE001
            return {"ok": False, "source": str(p), "note": f"台帳の照合の結果が読めない: {type(e).__name__}"}
        if p.stat().st_mtime < _newest_mtime(dirs):
            return {"ok": False, "source": str(p), "note": "台帳の照合の結果が、記録より古い（97 を回し直す）"}
        src = str(p)
    else:
        try:
            rep, _, _ = _load(ROOT / "scripts" / "97_s4_ledger_check.py", "s4_ledger_for_b4").build_report(ROOT, LEDGER, [])
        except Exception as e:                          # noqa: BLE001
            return {"ok": False, "source": "97_s4_ledger_check.build_report", "note": f"照合が回らない: {type(e).__name__}: {e}"}
        src = "97_s4_ledger_check.build_report（この場で）"
    unread = (rep.get("problems") or {}).get("unreadable_records")
    n = len(unread) if isinstance(unread, list) else (rep.get("summary") or {}).get("parse_errors")
    return {"ok": n == 0, "source": src, "parse_errors": n}


def cross_audit(res: dict, layout: dict, ledger_json=None, use_git: bool = True) -> dict:
    """A・B が一致した結果に、条件をまたぐ点検を足す。満たさなければ未完にして、判定を消す。"""
    heads = []
    for name in sorted(res["checks"]):
        for h in res["checks"][name].get("git_heads") or []:
            if h not in heads:
                heads.append(h)
    base = pathlib.Path(layout["root"]) / layout["experiment"]
    dirs = [base / v["cond"] for v in (layout.get("conds") or {}).values()]
    ea = {"versions": check_heads(heads, use_git), "ledger": check_ledger(ledger_json, dirs)}
    ea["problems"] = [f"{k}: {v.get('note') or v}" for k, v in ea.items() if not v["ok"]]
    out = dict(res, entry_audit=ea)
    if ea["problems"] and out["status"] == "complete":
        out.update(status="incomplete", result=None)
    return out


def cmd_check(args) -> int:
    from recovla.eval import b4 as B4
    try:
        layout = json.loads(pathlib.Path(args.layout).read_text(encoding="utf-8"))
        params = json.loads(pathlib.Path(args.params).read_text(encoding="utf-8"))
        params = {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v)
                  for k, v in params.items() if not k.startswith("_")}
        a, b, diffs = run_check(layout, params)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"入力を読めない: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if diffs:
        print(f"実装 A と B が一致しない（{len(diffs)} 所。判定を書かない）:", file=sys.stderr)
        for d in diffs[:50]:
            print("  " + d, file=sys.stderr)
        return 1
    a = cross_audit(a, layout, args.ledger_json, not args.no_git)
    res = dict(a, double_count={"implementations": ["src/recovla/eval/b4.py", "scripts/98_s4_b4_b.py"], "agree": True,
                                "rule": "件数・真偽・文字列は完全一致、実数は相対 1e-9 以内"},
               layout=layout, written=time.strftime("%Y-%m-%d %H:%M:%S"))
    md = B4.summary_md(res)
    out = pathlib.Path(args.out) if args.out else ROOT / "outputs" / "s4" / "b4_eval" / f"result_{layout['phase']}.json"
    mdp = pathlib.Path(args.md) if args.md else out.with_suffix(".md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    mdp.write_text(md, encoding="utf-8")
    print(md)
    return 0 if res["status"] == "complete" else 3


def cmd_example(args) -> int:
    d = pathlib.Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.json").write_text(json.dumps(EXAMPLE_PARAMS, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"書いた: {d / 'params.json'}（値は既定。回す前に事前登録 v2 の値に直す）")
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
    p.add_argument("--ledger-json", default=None, help="97_s4_ledger_check.py の結果（記録より新しいこと）。無ければこの場で照らす")
    p.add_argument("--no-git", action="store_true", help="HEAD が 2 つ以上のとき git で照らさない（未完にする）")
    p = sub.add_parser("example", help="params の雛形を書く")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"check": cmd_check, "example": cmd_example}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
