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
        out += sorted(p for p in (ROOT / d).rglob("*") if p.is_file())
    return out


def listing() -> dict:
    return {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in targets()}


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
    bad = sorted(k for k in set(ref) | set(now) if ref.get(k) != now.get(k))
    print(json.dumps({"n_ref": len(ref), "n_now": len(now), "differ": bad[:50], "n_differ": len(bad)},
                     ensure_ascii=False))
    raise SystemExit(1 if bad else 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("write")
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    {"write": cmd_write, "verify": cmd_verify}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
