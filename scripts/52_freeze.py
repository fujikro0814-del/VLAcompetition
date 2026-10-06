"""Step J の凍結（手順書 Step J の 1）: Git の管理外（outputs・models）の版を SHA-256 の一覧で固定する。

    .venv\\Scripts\\python.exe scripts\\52_freeze.py write      # docs/freeze/stepJ_hashes.json と pip_freeze.txt を書く
    .venv\\Scripts\\python.exe scripts\\52_freeze.py verify     # 今のファイルが一覧と一致するか（違いを表示して終了コード 1）

対象: 評価に使う 4 つの保存点（R1・N1 の 3 万手、R2・R1+ の 1 万手。training_state も含む）、変換済みのデータセット
（outputs/datasets の全部）、マニフェスト（outputs/manifests）、出発点のモデル（models の全部）。生の生成の記録
（outputs/gen、約 150 万ファイル）は、マニフェストとデータセットのハッシュで間接に固定し、一覧には入れない。
一覧はこのファイルを含むコミットに入れ、そのコミットにタグを打つ（stepJ-freeze）。
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

from recovla.common import config

ROOT = config.ROOT
OUT_DIR = ROOT / "docs" / "freeze"
CKPTS = {
    "R1": "outputs/train/train_R1_20260926-153959_20260926-153959/checkpoints/030000",
    "N1": "outputs/train/train_N1_20260926-191928_20260926-191928/checkpoints/030000",
    "R2": "outputs/train/train_R2_20260927-145256_20260927-145256/checkpoints/010000",
    "R1plus": "outputs/train/train_R1plus_20260927-160208_20260927-160208/checkpoints/010000",
}
TREES = ["outputs/datasets", "outputs/manifests", "models"]


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def targets() -> list:
    out = []
    for d in list(CKPTS.values()) + TREES:
        out += sorted(p for p in (ROOT / d).rglob("*") if p.is_file())     # 無いフォルダは rglob が空を返す
    return out


def run_level_targets() -> list:
    """学習の実行のフォルダの直下のファイル（conversion.json など。評価で方策を読むときに conversion.json を読む）。
    最初の一覧（stepJ_hashes.json）から漏れていたので、追補（stepJ_hashes_addendum.json）に入れる（0098）。"""
    out = []
    for c in CKPTS.values():
        d = (ROOT / c).parents[1]
        if d.is_dir():                                   # 置いていない実行は数えない（verify では「違い」として出る）
            out += sorted(p for p in d.iterdir() if p.is_file())
    return out


def listing(paths=None) -> dict:
    return {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in (targets() if paths is None else paths)}


def cmd_addendum(a) -> None:
    files = listing(run_level_targets())
    for k in files:
        files[k]["mtime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime((ROOT / k).stat().st_mtime))
    res = {"note": "stepJ_hashes.json の追補（0098）。学習の実行のフォルダの直下のファイルが最初の一覧から漏れていた。"
                   "どれも更新時刻が凍結（stepJ-freeze、2026-09-28 13:50）より前で、凍結の後に書き換えていない",
           "n_files": len(files), "files": files, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT_DIR / "stepJ_hashes_addendum.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "files"}, ensure_ascii=False))
    for k, v in files.items():
        print(v["mtime"], k)


def cmd_write(a) -> None:
    t0 = time.time()
    files = listing()
    head = subprocess.run([str(ROOT / ".tools" / "git" / "cmd" / "git.exe"), "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    res = {"note": "Step J の凍結。一覧はこのファイルを含むコミット（タグ stepJ-freeze）の版に対応する。"
                   "parent_head はこの一覧を書いたときの HEAD（一覧のコミットの親）",
           "parent_head": head, "checkpoints": CKPTS, "trees": TREES, "n_files": len(files),
           "total_bytes": sum(v["bytes"] for v in files.values()), "files": files,
           "written": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stepJ_hashes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    # この .venv には pip が入っていないので、入っている部品の名前と版を importlib.metadata から書く（pip freeze と同じ形）
    import importlib.metadata as md
    pkgs = sorted({f"{d.metadata['Name']}=={d.version}" for d in md.distributions() if d.metadata["Name"]},
                  key=str.lower)
    head_line = f"# python {sys.version.split()[0]}（{len(pkgs)} 個、importlib.metadata）\n"
    (OUT_DIR / "pip_freeze.txt").write_text(head_line + "\n".join(pkgs) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "files"}, ensure_ascii=False))


def cmd_verify(a) -> None:
    ref = json.loads((OUT_DIR / "stepJ_hashes.json").read_text(encoding="utf-8"))["files"]
    now = listing()
    add = OUT_DIR / "stepJ_hashes_addendum.json"
    if add.is_file():
        ref = {**ref, **{k: {kk: v[kk] for kk in ("bytes", "sha256")}
                         for k, v in json.loads(add.read_text(encoding="utf-8"))["files"].items()}}
        now = {**now, **listing(run_level_targets())}
    missing = sorted(k for k in ref if k not in now)
    changed = sorted(k for k in ref if k in now and now[k] != ref[k])
    extra = sorted(k for k in now if k not in ref)      # 凍結の後に増えたもの（検査が作るキャッシュなど）。参考として出す
    print(json.dumps({"n_ref": len(ref), "n_present": len(ref) - len(missing), "n_changed": len(changed),
                      "changed": changed[:50], "n_missing": len(missing), "missing": missing[:20],
                      "n_extra": len(extra), "extra": extra[:10]}, ensure_ascii=False))
    raise SystemExit(1 if changed or missing else 0)


# ---- 段階 3 の最終評価の凍結（0142・0143）: 評価に使う保存点（R1v3・N1v3 の 2 万手と実行のフォルダの直下）、その学習データと
# マニフェスト、出発点のモデル。一覧は docs/freeze/s3_hashes.json、タグは v3-s3-freeze
S3_FILE = OUT_DIR / "s3_hashes.json"


def s3_targets() -> tuple:
    ev = importlib_load("ev82", ROOT / "scripts" / "82_v2_eval.py")
    v3 = json.loads((ROOT / "outputs" / "f" / "data_v3.json").read_text(encoding="utf-8"))
    ckpts = {m: str((ROOT / ev.CKPT[m]).parent.relative_to(ROOT).as_posix()) for m in ("R1v3", "N1v3")}
    dirs = list(ckpts.values()) + [v3["datasets"][k]["dataset"].replace("\\", "/") for k in ("R1", "N1")] + ["models"]
    paths = []
    for d in dirs:
        paths += sorted(p for p in (ROOT / d).rglob("*") if p.is_file())
    for c in ckpts.values():
        paths += sorted(p for p in (ROOT / c).parents[1].iterdir() if p.is_file())
    paths += [ROOT / v3["datasets"][k]["manifest"] for k in ("R1", "N1")]
    return ckpts, dirs, paths


def importlib_load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cmd_write_s3(a) -> None:
    t0 = time.time()
    ckpts, dirs, paths = s3_targets()
    files = listing(paths)
    head = subprocess.run([str(ROOT / ".tools" / "git" / "cmd" / "git.exe"), "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    res = {"note": "段階 3 の最終評価の凍結（0143）。一覧はこのファイルを含むコミット（タグ v3-s3-freeze）の版に対応する。"
                   "parent_head はこの一覧を書いたときの HEAD", "parent_head": head, "checkpoints": ckpts, "trees": dirs,
           "n_files": len(files), "total_bytes": sum(v["bytes"] for v in files.values()), "files": files,
           "written": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    S3_FILE.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "files"}, ensure_ascii=False))


def cmd_verify_s3(a) -> None:
    ref = json.loads(S3_FILE.read_text(encoding="utf-8"))["files"]
    now = listing(s3_targets()[2])
    missing = sorted(k for k in ref if k not in now)
    changed = sorted(k for k in ref if k in now and now[k] != ref[k])
    print(json.dumps({"n_ref": len(ref), "n_changed": len(changed), "changed": changed[:50], "n_missing": len(missing),
                      "missing": missing[:20]}, ensure_ascii=False))
    raise SystemExit(1 if changed or missing else 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("write")
    sub.add_parser("verify")
    sub.add_parser("addendum")
    sub.add_parser("write-s3")
    sub.add_parser("verify-s3")
    a = ap.parse_args(argv)
    {"write": cmd_write, "verify": cmd_verify, "addendum": cmd_addendum, "write-s3": cmd_write_s3,
     "verify-s3": cmd_verify_s3}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
