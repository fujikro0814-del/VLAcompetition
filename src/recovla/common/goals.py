"""目標書（docs/目標書.md）の版の照合（0106。目標書 v2 の「変更の手続き」）.

目標書を変えてよいのは、決裁が承認して版を上げたときだけで、そのとき本線はタグ goals-v<版> を打ち、
文書の末尾の変更履歴に「→ v<版>」の行を残す。ここでは次を確かめる:
  1. 作業ツリーの docs/目標書.md のバイト列が、いちばん新しい goals-v* タグの時点の同じファイルと一致する
  2. そのタグの版が、文書の変更履歴に記録されている（「→ v<版>」の行がある）
どちらかが崩れていれば GoalsMismatch を投げる。結果の表・説明資料を作るスクリプトは、作る前に
fingerprint() を呼び、返った辞書（版・タグ・SHA-256）を記録に書く。numpy-free.
"""
import hashlib
import pathlib
import re
import subprocess

from recovla.common.code_version import ROOT, _git, git_executable

GOALS_PATH = "docs/目標書.md"
TAG_PREFIX = "goals-v"


class GoalsMismatch(RuntimeError):
    pass


def _tags(root) -> list:
    out = _git(root, "tag", "--list", f"{TAG_PREFIX}*")
    if out is None:
        raise GoalsMismatch("git が使えないので目標書の版を照合できない")
    tags = [t.strip() for t in out.splitlines() if re.fullmatch(rf"{TAG_PREFIX}\d+", t.strip())]
    return sorted(tags, key=lambda t: int(t[len(TAG_PREFIX):]))


def _blob_at(root, tag) -> bytes:
    out = subprocess.run([git_executable(), "-C", str(root), "show", f"{tag}:{GOALS_PATH}"],
                         capture_output=True, timeout=30)
    if out.returncode != 0:
        raise GoalsMismatch(f"タグ {tag} に {GOALS_PATH} がない")
    return out.stdout


def fingerprint(root=ROOT) -> dict:
    """照合に通れば {"version", "tag", "sha256", "path"} を返す。通らなければ GoalsMismatch."""
    root = pathlib.Path(root)
    path = root / GOALS_PATH
    if not path.is_file():
        raise GoalsMismatch(f"{GOALS_PATH} がない")
    data = path.read_bytes()
    tags = _tags(root)
    if not tags:
        raise GoalsMismatch(f"{TAG_PREFIX}* のタグがない")
    tag = tags[-1]
    if _blob_at(root, tag) != data:
        raise GoalsMismatch(f"{GOALS_PATH} がタグ {tag} の版と違う（変更履歴に記録された版ではない）")
    version = int(tag[len(TAG_PREFIX):])
    history = data.decode("utf-8").split("## 変更履歴", 1)
    if len(history) != 2 or not re.search(rf"→\s*v{version}\b", history[1]):
        raise GoalsMismatch(f"変更履歴に v{version} の行がない")
    return {"version": version, "tag": tag, "sha256": hashlib.sha256(data).hexdigest(), "path": GOALS_PATH}


def main(argv=None) -> int:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    try:
        fp = fingerprint(argv[0] if argv else ROOT)
    except GoalsMismatch as e:
        print(f"目標書の照合: 不合格: {e}")
        return 1
    print(f"目標書の照合: 合格（v{fp['version']}、{fp['tag']}、sha256 {fp['sha256']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
