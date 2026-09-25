"""Code version of the pipeline, recorded by the generator, the training launcher and the evaluator
(流用元 code_version.py。変えたのは git の場所と対象ファイルの一覧だけ).

git_commit is the HEAD of the <ROOT> repository (git is not on PATH; the portable MinGit in
<ROOT>\\.tools\\git is used) and git_dirty tells whether the work tree differed from it, since a commit
alone does not identify uncommitted code. The SHA-256 of the files that decide recording, conversion,
observation, training launch and evaluation is recorded as well, so the version stays identifiable
without git. numpy-free.
"""
import hashlib
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]           # <ROOT>
PORTABLE_GIT = ROOT / ".tools" / "git" / "cmd" / "git.exe"
CODE_FILES = (
    "src/recovla/sim/control.py", "src/recovla/sim/controller_ik.py", "src/recovla/sim/device.py",
    "src/recovla/sim/render.py", "src/recovla/record/recorder.py", "src/recovla/record/replay.py",
    "assets/mjcf/scene_g0.xml", "assets/mjcf/panda/panda.xml",
    "src/recovla/data/vla_image_spec.py", "src/recovla/data/vla_state.py", "src/recovla/data/vla_observation.py",
    "src/recovla/data/convert.py", "src/recovla/policy/train_launcher.py", "src/recovla/eval/closed_loop.py",
    "src/recovla/common/code_version.py", "src/recovla/common/config.py", "configs/default.yaml", "configs/g0.yaml",
)


def file_sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def git_executable():
    found = shutil.which("git")
    if found:
        return found
    return str(PORTABLE_GIT) if PORTABLE_GIT.is_file() else None


def _git(root, *args):
    git = git_executable()
    if git is None:
        return None
    try:
        out = subprocess.run([git, "-C", str(root), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def git_commit(root=ROOT):
    """HEAD of the enclosing git work tree, or None (no git, or not a repository)."""
    out = _git(root, "rev-parse", "HEAD")
    return out.strip() if out else None


def git_dirty(root=ROOT):
    """True if tracked or untracked (not ignored) files differ from HEAD; None without git."""
    out = _git(root, "status", "--porcelain")
    return None if out is None else bool(out.strip())


def code_version(root=ROOT) -> dict:
    root = pathlib.Path(root)
    files = {f: (file_sha256(root / f) if (root / f).is_file() else None) for f in CODE_FILES}
    combined = hashlib.sha256("\n".join(f"{f} {h}" for f, h in files.items()).encode()).hexdigest()
    commit = git_commit(root)
    return {"git_commit": commit, "git_dirty": git_dirty(root) if commit else None,
            "code_sha256": combined, "files_sha256": files}
