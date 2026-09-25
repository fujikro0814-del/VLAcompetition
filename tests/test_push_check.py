"""push の前の検査（scripts/check_before_push.ps1）の単体検査（B_提案書 §3.2 の 10、掲示板 0002・0004）。

一時的な Git のリポジトリを作り、きれいな文書は通り、鍵・管理外の置き場所・証明書・手元の照合文字列は
（索引にある場合も、コミットの履歴だけにある場合も）止まることを確かめる。照合文字列はこの検査の中で作る
（本物の値は使わない）。
"""
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GIT = ROOT / ".tools" / "git" / "cmd" / "git.exe"
CHECK = ROOT / "scripts" / "check_before_push.ps1"
FAKE_PROXY = "198.51.100.7:3128"                 # 文書用のアドレス帯（RFC 5737）
FAKE_KEY = "sk-ant-" + "api03-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"


def git(repo, *args):
    subprocess.run([str(GIT), "-C", str(repo), *args], check=True, capture_output=True)


def check(repo, patterns) -> subprocess.CompletedProcess:
    return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(CHECK),
                           "-Repo", str(repo), "-Upstream", "none", "-PatternFile", str(patterns)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    git(r, "config", "core.autocrlf", "false")
    (r / "ok.md").write_text("鍵らしい文字列（sk-ant- など）は置かない。終了時に Y:\\ へ複写。hf_home を使う。\n"
                             "BEGIN CERTIFICATE という名前だけの言及。\n", encoding="utf-8")
    git(r, "add", "ok.md")
    git(r, "commit", "-q", "-m", "clean")
    patterns = tmp_path / "patterns.txt"
    patterns.write_text("# test\n" + FAKE_PROXY + "\n", encoding="utf-8")
    return r, patterns


def test_clean_documents_pass(repo):
    r, patterns = repo
    res = check(r, patterns)
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.parametrize("name, content", [
    ("key.md", f"key = {FAKE_KEY}\n"),
    ("proxy.md", f"proxy http://{FAKE_PROXY}\n"),
    ("outputs/x.txt", "x\n"),
    ("docs/local/note.md", "x\n"),
    ("c.pem", "x\n"),
    ("cert.txt", "-" * 5 + "BEGIN CERTIFICATE" + "-" * 5 + "\n"),
    ("id.txt", "-" * 5 + "BEGIN OPENSSH PRIVATE KEY" + "-" * 5 + "\n"),
])
def test_staged_problems_stop_the_push(repo, name, content):
    r, patterns = repo
    p = r / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    git(r, "add", "-f", name)
    res = check(r, patterns)
    assert res.returncode == 1, res.stdout
    assert FAKE_KEY not in res.stdout and FAKE_PROXY not in res.stdout      # 中身は表示しない


def test_problem_only_in_history_stops_the_push(repo):
    r, patterns = repo
    (r / "key.md").write_text(f"key = {FAKE_KEY}\n", encoding="utf-8")
    git(r, "add", "key.md")
    git(r, "commit", "-q", "-m", "bad")
    git(r, "rm", "-q", "key.md")
    git(r, "commit", "-q", "-m", "removed")
    res = check(r, patterns)
    assert res.returncode == 1 and "コミット" in res.stdout


def test_missing_pattern_file_stops_the_push(repo, tmp_path):
    r, _ = repo
    res = check(r, tmp_path / "no_such_file.txt")
    assert res.returncode == 1
