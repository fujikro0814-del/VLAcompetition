"""段階 4 束 4: R4・N4 のマニフェストを作り、変換する。R4 = R1v3 の 330 本＋新しい滑りの復帰デモ、N4 = N1v3 の 330 本＋同じ本数の
同じ配置の通常デモ（R と N の違いは復帰区間の有無だけ＝目標書 2 節。30_f.py gen-data の R1・N1 と同じ組み方）。

使い方（作業場所 C:\\PAI\\recovery_vla。試すときは必ず --dry-run）:
    .venv\\Scripts\\python.exe scripts\\97_s4_b4_manifest.py build --dry-run     # 組み立てと検査だけ（何も書かない）
    .venv\\Scripts\\python.exe scripts\\97_s4_b4_manifest.py build               # outputs\\manifests\\R4_<日時>.json・N4_<日時>.json を書き、SHA-256 を出す
    .venv\\Scripts\\python.exe scripts\\97_s4_b4_manifest.py convert [--dry-run] # LeRobot のデータセットに変換して検査（30_f.py と同じ引数）
                                                                              → outputs\\s4\\b4\\data_b4.json（学習の入口が読む）
検査（どれか 1 つでも落ちたら書かない・止める）:
  - 元の R1v3・N1v3 のマニフェスト（outputs\\f\\data_v3.json の datasets.R1・N1.manifest）の SHA-256 が、束 3 で掲示した値
    （docs\\stage4\\seed_replication.md 3 節）と同じで、どちらも 330 本。
  - 新しい本数: R4 も N4 も 330 ＋ 足した本数（outputs\\s4\\b4\\data.json の chosen。B 30・C 20 なら 380）。種類別の数
    （R4 は A 40・B 60・C 40・通常 240、N4 は通常 380）。
  - R と N の通常デモの集合が同じ: 通常の部分（R1v3 の 240 本）は同じ名前、残り（R4 の復帰・N4 の相手の通常）は（種、色）の
    集合が同じ（30_f.py の layout_targets と同じ数え方）。同じ名前が 2 回出ない。
  - data.json の ok（本数がそろい、捨てた割合が 10% 以下か作者承認の例外＝drop_rule_exception、ファイルがある）。
読むもの: outputs\\f\\data_v3.json、元のマニフェスト、outputs\\s4\\b4\\data.json、scripts\\30_f.py（importlib。layout_targets と変換の引数の決まり）。
書くもの: outputs\\manifests\\R4_*.json・N4_*.json、outputs\\datasets\\R4_*・N4_*（convert）、outputs\\s4\\b4\\manifest.json・data_b4.json。
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

from recovla.common import config

ROOT = config.ROOT
CFG = config.load()
OUTPUTS = config.path(CFG["paths"]["outputs"])
B4 = OUTPUTS / "s4" / "b4"
# 束 3 の掲示（docs/stage4/seed_replication.md 3 節）の値。違えば止める
EXPECTED_SHA = {"R1": "85dd9dbe951b8244ed2606f9aacb3ab5d635d3ade4dcc86a36b39519bc1d810f",
                "N1": "cebcb4a3684cffe3b8a23ec29c823cbcef4e608cb344e8b7de2bb34cac31ddaa"}
BASE_N = 330


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_f30():
    spec = importlib.util.spec_from_file_location("s3_f30_m", ROOT / "scripts" / "30_f.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def kind_of(key: str) -> str:
    return key.split("_")[0]


def compose(r1: dict, n1: dict, chosen: list) -> tuple:
    """(R4 の entries, N4 の entries)。R4 = R1v3 ＋ 復帰、N4 = N1v3 ＋ 相手の通常（chosen の順）。"""
    r4 = list(r1["entries"]) + [dict(c["recovery"]) for c in chosen]
    n4 = list(n1["entries"]) + [dict(c["twin"]) for c in chosen]
    return r4, n4


def check(r1: dict, n1: dict, r4: list, n4: list, chosen: list, layout_targets) -> dict:
    """本数・種類別の数・通常デモの集合。layout_targets は 30_f.py のもの（名前 <種類>_<種>_<色>_r<回> から（種、色））。"""
    k_add = len(chosen)
    keys_r, keys_n = [e["key"] for e in r4], [e["key"] for e in n4]
    normal_r = [e["key"] for e in r1["entries"] if kind_of(e["key"]) == "n"]
    comp_r, comp_n = collections.Counter(map(kind_of, keys_r)), collections.Counter(map(kind_of, keys_n))
    add_kinds = collections.Counter(c["kind"] for c in chosen)
    base_r = collections.Counter(kind_of(e["key"]) for e in r1["entries"])
    want_r = dict(base_r)
    for k, v in add_kinds.items():
        want_r[k] = want_r.get(k, 0) + v
    items = {
        "base_counts": {"R1v3": len(r1["entries"]), "N1v3": len(n1["entries"]), "ok": len(r1["entries"]) == BASE_N == len(n1["entries"])},
        "new_counts": {"R4": len(r4), "N4": len(n4), "want": BASE_N + k_add, "ok": len(r4) == len(n4) == BASE_N + k_add},
        "composition": {"R4": dict(comp_r), "N4": dict(comp_n), "R4_want": want_r,
                        "ok": dict(comp_r) == want_r and set(comp_n) == {"n"} and comp_n["n"] == BASE_N + k_add},
        "no_duplicates": {"ok": len(set(keys_r)) == len(keys_r) and len(set(keys_n)) == len(keys_n)},
        "normal_part_identical": {"ok": sorted(normal_r) == sorted(k for k in keys_n if k in set(normal_r))
                                  and set(normal_r) <= set(keys_n)},
        "same_layouts_and_targets": {"ok": layout_targets(r4, normal_r) == layout_targets(n4, normal_r)},
        "added_are_slip_only": {"ok": all(c["kind"] in ("B", "C") and kind_of(c["recovery"]["key"]) == c["kind"] for c in chosen)},
        "twins_are_normal_on_same_layout": {"ok": all(kind_of(c["twin"]["key"]) == "n" and
                                                      c["recovery"]["key"].split("_")[1:3] == c["twin"]["key"].split("_")[1:3]
                                                      for c in chosen)},
    }
    return {"items": items, "ok": all(v["ok"] for v in items.values())}


def _sources():
    v3 = json.loads((OUTPUTS / "f" / "data_v3.json").read_text(encoding="utf-8"))
    out = {}
    for k in ("R1", "N1"):
        p = config.path(v3["datasets"][k]["manifest"])
        raw = p.read_bytes()
        out[k] = {"path": p, "sha256": sha256_bytes(raw), "manifest": json.loads(raw.decode("utf-8"))}
    return v3, out


def cmd_build(a) -> int:
    f30 = load_f30()
    v3, src = _sources()
    data = json.loads((B4 / "data.json").read_text(encoding="utf-8"))
    prob = [f"{k} のマニフェストの SHA-256 が掲示の値と違う（{src[k]['sha256'][:12]}…）" for k in src if src[k]["sha256"] != EXPECTED_SHA[k]]
    if not data.get("ok"):
        prob.append("outputs\\s4\\b4\\data.json の ok が真でない（97_s4_b4_data.py verify）")
    r4, n4 = compose(src["R1"]["manifest"], src["N1"]["manifest"], data["chosen"])
    chk = check(src["R1"]["manifest"], src["N1"]["manifest"], r4, n4, data["chosen"], f30.layout_targets)
    if not chk["ok"]:
        prob.append("本数・組み立ての検査に落ちた: " + ", ".join(k for k, v in chk["items"].items() if not v["ok"]))
    print(json.dumps(chk, ensure_ascii=False, indent=1))
    if prob:
        print("止める: " + " / ".join(prob), file=sys.stderr)
        return 1
    if a.dry_run:
        print("[dry-run] 書かない")
        return 0
    from recovla.data import convert as C
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rec = {}
    for name, entries, rule in (
            ("R4", r4, f"Stage 4 bundle 4 R4: R1v3 manifest ({src['R1']['sha256'][:12]}) + slip recovery B/C (scripts/97_s4_b4_data.py)"),
            ("N4", n4, f"Stage 4 bundle 4 N4: N1v3 manifest ({src['N1']['sha256'][:12]}) + normal demos on the same layouts")):
        dname = f"{name}_{stamp}"
        mpath = OUTPUTS / "manifests" / f"{dname}.json"
        C.write_manifest(mpath, dname, entries, rule)
        rec[name] = {"manifest": str(mpath.relative_to(ROOT)).replace("\\", "/"), "name": dname,
                     "sha256": sha256_bytes(mpath.read_bytes()), "episodes": len(entries)}
        print(f"{name}: {rec[name]['manifest']}  sha256 {rec[name]['sha256']}  {len(entries)} 本")
    B4.mkdir(parents=True, exist_ok=True)
    (B4 / "manifest.json").write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "manifests": rec, "check": chk,
                                                  "base": {k: {"path": str(v["path"].relative_to(ROOT)), "sha256": v["sha256"]}
                                                           for k, v in src.items()}},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def reverify(man: dict, src: dict, data: dict, layout_targets) -> dict:
    """書いた R4・N4 のマニフェストのファイルを読み直して確かめる（関門 D の d5 が使う。manifest.json の記録を信じない）:
    ファイルの SHA-256 が build のときと同じ、元の R1v3・N1v3 の SHA-256 が掲示の値、中身が R1v3・N1v3 ＋ data.json の
    chosen から組み立て直したものと順まで同じ、そのうえで check の全項目。"""
    files = {k: json.loads((ROOT / m["manifest"]).read_bytes().decode("utf-8")) for k, m in man["manifests"].items()}
    sha_now = {k: sha256_bytes((ROOT / m["manifest"]).read_bytes()) for k, m in man["manifests"].items()}
    r4, n4 = compose(src["R1"]["manifest"], src["N1"]["manifest"], data["chosen"])
    chk = check(src["R1"]["manifest"], src["N1"]["manifest"], files["R4"]["entries"], files["N4"]["entries"], data["chosen"],
                layout_targets)
    items = dict(chk["items"])
    items["files_same_sha_as_built"] = {"ok": all(sha_now[k] == man["manifests"][k]["sha256"] for k in sha_now)}
    items["base_sha_as_posted"] = {"ok": all(src[k]["sha256"] == EXPECTED_SHA[k] for k in ("R1", "N1"))}
    items["entries_equal_recomposed"] = {"ok": files["R4"]["entries"] == r4 and files["N4"]["entries"] == n4}
    return {"items": items, "ok": all(v["ok"] for v in items.values()), "sha256": sha_now}


def cue_args_like_v3(v3: dict) -> list:
    """R1v3 を変換したときと同じ引数（outputs\\f\\data_v3.json の cue_convert_args。30_f.py が記録したもの）。"""
    args = v3.get("cue_convert_args")
    if args is None:
        raise SystemExit("outputs\\f\\data_v3.json に cue_convert_args がない（R1v3 と同じ変換の引数が分からない）")
    return list(args)


def cmd_convert(a) -> int:
    v3, _ = _sources()
    man = json.loads((B4 / "manifest.json").read_text(encoding="utf-8"))
    args = cue_args_like_v3(v3)
    out = {}
    for name, m in man["manifests"].items():
        mpath = ROOT / m["manifest"]
        if sha256_bytes(mpath.read_bytes()) != m["sha256"]:
            print(f"{name} のマニフェストが build のときと違う", file=sys.stderr)
            return 1
        ds = OUTPUTS / "datasets" / m["name"]
        cmd1 = [sys.executable, "-m", "recovla.data.convert", "--manifest", str(mpath), "--out", str(ds), "--name", m["name"], *args]
        cmd2 = [sys.executable, "-m", "recovla.data.convert", "--verify", str(ds)]
        if a.dry_run:
            print("[dry-run]", " ".join(cmd1))
            continue
        log = B4 / f"convert_{m['name']}.log"
        with open(log, "w", encoding="utf-8") as f:
            c1 = subprocess.run(cmd1, stdout=f, stderr=subprocess.STDOUT)
            c2 = subprocess.run(cmd2, stdout=f, stderr=subprocess.STDOUT)
        text = log.read_text(encoding="utf-8", errors="replace")
        info = json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8")) if (ds / "meta" / "info.json").is_file() else {}
        out[name] = {"manifest": m["manifest"], "manifest_sha256": m["sha256"], "dataset": str(ds.relative_to(ROOT)).replace("\\", "/"),
                     "episodes": m["episodes"], "frames": info.get("total_frames"), "convert_exit": c1.returncode,
                     "verify_exit": c2.returncode, "verify_pass": "[verify] PASS" in text, "log": str(log.relative_to(ROOT))}
    if a.dry_run:
        return 0
    (B4 / "data_b4.json").write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "rig": "v3",
                                                 "cue_convert_args": args, "datasets": out}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    ok = all(v["convert_exit"] == 0 and v["verify_exit"] == 0 and v["verify_pass"] for v in out.values())
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if ok else 1


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "convert"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return {"build": cmd_build, "convert": cmd_convert}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
