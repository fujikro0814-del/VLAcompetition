"""Code version of the VLA pipeline, recorded by the scripted generator, the training launcher and the
closed-loop evaluator (scripted route check, 2026-09-17).

git_commit is the HEAD of the pytools/panda_teleop repository (created 2026-09-17 with the portable MinGit in
02_環境/git; git is not on PATH) and git_dirty tells whether the work tree differed from it, since a commit
alone does not identify uncommitted code. The SHA-256 of the files that decide recording, conversion,
observation, training launch and evaluation is recorded as well (the same idea as meta.json
"software.sha256" written by teleop/collect.py), so the version stays identifiable without git.
numpy-free; runs in python311 and in the LeRobot venv.
"""
import hashlib
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
PORTABLE_GIT = ROOT.parents[2] / "02_環境" / "git" / "cmd" / "git.exe"
CODE_FILES = (
    "teleop/collect.py", "teleop/recorder.py", "teleop/ledger.py", "teleop/controller_ik.py", "teleop/app.py",
    "teleop/replay.py", "assets/panda/teleop_scene.xml", "assets/panda/panda.xml",
    "vla_image_spec.py", "vla_state.py", "vla_observation.py", "convert_to_lerobot.py",
    "train_launcher.py", "scripted_demo.py", "closed_loop_eval.py", "code_version.py",
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
