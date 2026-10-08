"""提出（課題の 10/8 版）の道具: 70_export_submission.py の youtube と ACCOUNT の扱い、check_submission.py の個人情報の検査、
push_submission.ps1 の作者・コミッタの固定。CPU だけ。合成のファイルだけを使う（本物の動画・PDF・値の一覧は読まない）。

使い方: .venv\\Scripts\\python.exe -m pytest -q tests\\test_submission_1008.py -p no:cacheprovider
読むもの: scripts/70_export_submission.py・scripts/check_submission.py（importlib）、scripts/push_submission.ps1（文字列）、
          scripts/60_paper.py の FORBIDDEN（構文木）、paper/README_submission.md（1 文のアイデア）。
書くもの: pytest の一時フォルダだけ（git の全体の設定も一時フォルダのものに差し替える）。
"""
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# 合成の個人情報（.invalid は例示用に予約されたドメイン。実在の人・アドレスではない）
FAKE_MAIL = "taro.yamada@lab.invalid"
NOREPLY = "12345678+example-user@users.noreply.github.com"
VALUES = {"eps_auto": "120", "n_normal": "60", "n_recovery": "60", "e3a_pairs": "27", "e3a_xk": "11", "e3a_yk": "3",
          "e3a_holmeq": "= 0.04", "e7_n": "20", "e7_k": "6"}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def ex():
    return _load(ROOT / "scripts" / "70_export_submission.py", "export_submission_1008")


@pytest.fixture(scope="module")
def cs():
    return _load(ROOT / "scripts" / "check_submission.py", "check_submission_1008")


@pytest.fixture()
def clean_git(tmp_path, monkeypatch):
    """git の全体の設定を空の一時ファイルにする（手元の user.name・user.email を検査に混ぜない）。そのファイルを返す。"""
    g = tmp_path / "gitconfig"
    g.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(g))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    return g


def make_mp4(path, seconds, fps=2):
    """合成の動画（64×48、灰色）。OpenCV で書けなければ飛ばす。"""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    if not w.isOpened():
        pytest.skip("OpenCV で MP4 を書けない")
    frame = np.full((48, 64, 3), 128, np.uint8)
    for _ in range(int(seconds * fps)):
        w.write(frame)
    w.release()
    return path


def make_pdf(path, text):
    """文字 1 行だけの最小の PDF（Helvetica）。"""
    stream = f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET".encode("ascii")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 100] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    path.write_bytes(bytes(out))
    return path


def export_dir(tmp_path, files: dict):
    d = tmp_path / "recovery-vla-panda"
    d.mkdir()
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text, encoding="utf-8")
    return d


def video_texts(tmp_path, texts):
    p = tmp_path / "video_texts.json"
    p.write_text(json.dumps({"seconds": 120.0, "texts": texts}, ensure_ascii=False), encoding="utf-8")
    return p


# ------------------------------------------------------------------ 個人情報の検査（check_submission.py --pii-only）
@pytest.mark.parametrize("where, files, texts, pdf_text, expect", [
    ("text", {"docs/a.md": "連絡先\n" + FAKE_MAIL + "\n"}, ["ok"], None, "docs/a.md:2 メールアドレス"),
    ("win path", {"scripts/b.py": "P = r'C:\\Users\\taro\\work'\n"}, ["ok"], None, "scripts/b.py:1 個人のパス"),
    ("json path", {"c.json": '{"p": "C:\\\\Users\\\\hanako\\\\x"}\n'}, ["ok"], None, "c.json:1 個人のパス"),
    ("linux path", {"d.yaml": "a: 1\nb: /home/jiro/data\n"}, ["ok"], None, "d.yaml:2 個人のパス"),
    ("mac path", {"e.txt": "/Users/saburo/Desktop\n"}, ["ok"], None, "e.txt:1 個人のパス"),
    ("video", {"ok.md": "なし\n"}, ["題名", "作者 " + FAKE_MAIL], None, "video_texts.json の 2 番目の文字:1 メールアドレス"),
    ("pdf", {"ok.md": "なし\n"}, ["ok"], "contact " + FAKE_MAIL, "paper.pdf の 1 ページ:1 メールアドレス"),
])
def test_pii_stops(cs, tmp_path, clean_git, capsys, where, files, texts, pdf_text, expect):
    d = export_dir(tmp_path, files)
    pdf = tmp_path / "paper.pdf"
    if pdf_text is not None:
        pytest.importorskip("pypdf")
        make_pdf(pdf, pdf_text)
    rc = cs.main(["--repo", str(d), "--pii-only", "--pdf", str(pdf), "--video-texts", str(video_texts(tmp_path, texts))])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert expect in out, out
    assert "taro.yamada" not in out and "hanako" not in out          # 当たった文字は伏せて出す


def test_pii_passes_noreply_and_allowed(cs, tmp_path, clean_git, capsys):
    d = export_dir(tmp_path, {
        "README.md": f"作者 {NOREPLY}\n例 user@example.com、clone は git@github.com:example-user/repo.git\n",
        "COMMIT.txt": "Co-Authored-By: Claude <noreply@anthropic.com>\n",
        "docs/how.md": "C:\\Users\\<名前>\\x、/home/<名前>/x、C:\\Users\\%USERNAME%\\x、/home/$USER/x、C:\\Users\\Public\\x\n"
                       "https://example.com/home/page と C:\\Users\\acct01\\work（アカウント名）\n",
        "LICENSE": "Copyright (c) 2026 acct01\n",
        "data.bin": "taro@lab.invalid は テキストでないので見ない\n",
    })
    pdf = tmp_path / "paper.pdf"
    if importlib.util.find_spec("pypdf"):
        make_pdf(pdf, "author acct01 " + NOREPLY)
    rc = cs.main(["--repo", str(d), "--pii-only", "--pii-allow", "acct01", "--pdf", str(pdf),
                  "--video-texts", str(video_texts(tmp_path, ["PAI最終課題_acct01", "Claude"]))])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "個人情報の検査: 合格" in out


def test_pii_git_identity(cs, tmp_path, clean_git, capsys):
    if cs.git_exe() is None:
        pytest.skip("git がない")
    clean_git.write_text("[user]\n\tname = Taro Yamada\n\temail = " + NOREPLY + "\n", encoding="utf-8")
    d = export_dir(tmp_path, {"a.md": "作った人: taro yamada\n", "b.md": "Taro Yamadaya は別の語\n"})
    rc = cs.main(["--repo", str(d), "--pii-only", "--pdf", str(tmp_path / "none.pdf"),
                  "--video-texts", str(video_texts(tmp_path, ["ok"]))])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "a.md:1 git の設定の user.name の値" in out
    assert "b.md" not in out                                       # 語の途中には当てない
    assert NOREPLY.split("+")[0] not in out                        # noreply のアドレスは許す


@pytest.fixture()
def sub_repo(cs, tmp_path, clean_git, monkeypatch):
    """合成の提出用リポジトリ（git）。検査の git は PATH のもの、照合用の文字列は一時ファイルに差し替える。"""
    exe = shutil.which("git")
    if exe is None:
        pytest.skip("git がない")
    monkeypatch.setattr(cs, "GIT", pathlib.Path(exe))
    pats = tmp_path / "patterns.txt"
    pats.write_text("# test\n198.51.100.7:3128\n", encoding="utf-8")       # 文書用のアドレス帯（RFC 5737）
    monkeypatch.setattr(cs, "LOCAL_PATTERNS", pats)
    r = tmp_path / "recovery-vla-panda"
    r.mkdir()

    def commit(files):
        for rel, text in files.items():
            (r / rel).write_text(text, encoding="utf-8")
        for args in (["add", "-A"], ["-c", "user.name=acct01", "-c", "user.email=" + NOREPLY, "commit", "-q", "-m", "提出版"]):
            subprocess.run([exe, "-C", str(r), *args], check=True, capture_output=True)
        return cs.main(["--repo", str(r), "--upstream", "origin/main", "--pdf", str(tmp_path / "none.pdf"),
                        "--video-texts", str(video_texts(tmp_path, ["PAI最終課題_acct01"]))])
    subprocess.run([exe, "-C", str(r), "init", "-q", "-b", "main"], check=True, capture_output=True)
    return commit


def test_check_head_pii_and_license(sub_repo, capsys):
    assert sub_repo({"README.md": "説明\n", "LICENSE": "Copyright (c) 2026 acct01\n"}) == 0, capsys.readouterr().out
    rc = sub_repo({"README.md": "説明\n作者 " + FAKE_MAIL + "\n", "LICENSE": "Copyright (c) 2026 アカウント名\n"})
    out = capsys.readouterr().out
    assert rc == 1
    assert "個人情報: README.md:2 メールアドレス" in out
    assert "LICENSE の著作権者が仮の値" in out


def test_pii_unreadable_pdf_is_reported(cs, tmp_path, clean_git, capsys):
    d = export_dir(tmp_path, {"a.md": "なし\n"})
    bad = tmp_path / "paper.pdf"
    bad.write_bytes(b"not a pdf")
    (tmp_path / "paper.html").write_text("<p>" + FAKE_MAIL + "</p>", encoding="utf-8")
    rc = cs.main(["--repo", str(d), "--pii-only", "--pdf", str(bad), "--video-texts", str(video_texts(tmp_path, ["ok"]))])
    out = capsys.readouterr().out
    assert "PDF:" in out                                           # 読めない旨（pypdf がない・壊れている）
    assert rc == 1 and "paper.html（PDF の代わり）" in out           # 組み上げた HTML を代わりに調べる


# ------------------------------------------------------------------ YouTube の下書き（70_export_submission.py youtube）
def scenes_yaml(tmp_path, task_t1=60.0):
    p = tmp_path / "video.yaml"
    p.write_text(f"""scenes:
  - {{id: contrast, dur_s: 25, clip_with: a.mp4, clip_without: b.mp4, t0: 5.0, t1: 30.0}}
  - {{id: title, dur_s: 6}}
  - {{id: task, dur_s: 30, clip: c.mp4, t0: 0.0, t1: {task_t1}}}
""", encoding="utf-8")
    return p


def run_youtube(ex, tmp_path, *, seconds=61, task_t1=60.0, extra=()):
    mp4 = make_mp4(tmp_path / "v.mp4", seconds)
    vals = tmp_path / "values.json"
    vals.write_text(json.dumps(VALUES), encoding="utf-8")
    out = tmp_path / "out"
    argv = ["youtube", "--video", str(mp4), "--scenes", str(scenes_yaml(tmp_path, task_t1)), "--values", str(vals),
            "--out", str(out), "--allow-placeholder", *extra]
    return ex.main(argv), out


def test_youtube_draft_states_speed(ex, tmp_path, capsys):
    rc, out = run_youtube(ex, tmp_path, extra=("--github-url", "https://github.com/example-user/recovery-vla-panda"))
    assert rc == 0
    desc = (out / "youtube_description.txt").read_text(encoding="utf-8")
    title = (out / "youtube_title.txt").read_text(encoding="utf-8").strip()
    assert title.startswith("PAI最終課題_")
    assert "■ 倍速" in desc
    assert "・0:00〜0:25 冒頭の左右比較（復帰デモあり／なし）: 実時間（1 倍速）" in desc
    assert "・0:31〜1:01 「全部片付けて」の実演（3 個を順に片付ける）: 2.0 倍速" in desc
    assert "https://github.com/example-user/recovery-vla-panda" in desc
    assert "復帰デモあり 11 回、なし 3 回" in desc and "20 配置中 6 配置" in desc     # 値の一覧から差し込む
    assert "{{" not in desc and ex.forbidden_hits(title + desc) == []
    j = json.loads((out / "youtube.json").read_text(encoding="utf-8"))
    assert [r["speed"] for r in j["speeds"]] == [1.0, 2.0] and 60 <= j["video_s"] <= 62
    assert "子ども向け" in capsys.readouterr().out                    # 上げるときの設定も出す


def test_youtube_stops_over_2x(ex, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        run_youtube(ex, tmp_path, task_t1=62.0)                      # 62 / 30 = 2.07 倍
    assert e.value.code == 1
    assert "場面 task の倍速 2.07 が 2 倍を超える" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()                           # 何も書かない


def test_youtube_stops_out_of_range_length(ex, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        run_youtube(ex, tmp_path, seconds=50)
    assert e.value.code == 1
    assert "1〜3 分の外" in capsys.readouterr().out


def test_youtube_stops_forbidden_word(ex, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ex, "idea_sentence", lambda: "掲示板" + "の番号を書いた文")          # 60_paper.py の FORBIDDEN に当たる
    with pytest.raises(SystemExit) as e:
        run_youtube(ex, tmp_path)
    assert e.value.code == 1
    assert "使わない語" in capsys.readouterr().out


# ------------------------------------------------------------------ ACCOUNT（アカウント名）と LICENSE
def stub(tmp_path, name, account):
    p = tmp_path / name
    p.write_text(f'ACCOUNT = "{account}"\n', encoding="utf-8")
    return p


def test_account_placeholder_stops(ex, tmp_path):
    ph = ex.PLACEHOLDER
    with pytest.raises(SystemExit, match="仮の値"):
        ex.account_or_exit(False, stub(tmp_path, "p.py", ph), stub(tmp_path, "v.py", ph))
    with pytest.raises(SystemExit, match="違う"):
        ex.account_or_exit(False, stub(tmp_path, "p.py", "acct01"), stub(tmp_path, "v.py", "acct02"))
    assert ex.account_or_exit(False, stub(tmp_path, "p.py", "acct01"), stub(tmp_path, "v.py", "acct01")) == "acct01"
    assert ex.account_or_exit(True, stub(tmp_path, "p.py", ph), stub(tmp_path, "v.py", ph)) == ph   # 試しだけ
    assert "Copyright (c) 2026 acct01" in ex.license_text("acct01")


def test_export_stops_before_writing_when_placeholder(ex, tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_const", lambda path, name: ex.PLACEHOLDER)
    dest = tmp_path / "recovery-vla-panda"
    with pytest.raises(SystemExit, match="仮の値"):
        ex.main(["export", "--dest", str(dest)])
    assert not dest.exists()


def test_youtube_stops_when_placeholder(ex, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ex, "_const", lambda path, name: ex.PLACEHOLDER if name == "ACCOUNT" else [])
    vals = tmp_path / "values.json"
    vals.write_text(json.dumps(VALUES), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        ex.main(["youtube", "--video", str(make_mp4(tmp_path / "v.mp4", 61)), "--scenes", str(scenes_yaml(tmp_path)),
                 "--values", str(vals), "--out", str(tmp_path / "out")])
    assert e.value.code == 1
    assert "仮の値" in capsys.readouterr().out


def test_package_is_not_default(ex):
    with pytest.raises(SystemExit) as e:
        ex.main(["package"])
    assert e.value.code == 2


def test_this_test_is_not_exported(ex, cs):
    assert "tests/test_submission_1008.py" not in ex.included_files()
    assert any(re.search(p, "tests/test_submission_1008.py") for p in cs.FORBIDDEN_PATHS)


# ------------------------------------------------------------------ push_submission.ps1（Windows PowerShell 5.1 で読む）
def test_push_submission_ps1_encoding_and_identity_guard():
    raw = (ROOT / "scripts" / "push_submission.ps1").read_bytes()
    assert raw.isascii() or raw.startswith(b"\xef\xbb\xbf")         # ASCII か BOM 付き UTF-8
    t = raw.decode("utf-8-sig")
    assert re.search(r"\[string\]\$AuthorName = ''", t) and re.search(r"\[string\]\$AuthorEmail = ''", t)   # 既定は空
    body = t[t.index("function Assert-Identity"):]
    body = body[:body.index("\nfunction ")]
    assert "$AuthorName.Trim() -eq ''" in body and "$AuthorEmail.Trim() -eq ''" in body and body.count("throw") >= 3
    assert "users\\.noreply\\.github\\.com" in body
    rebuild = t[t.index("if ($RebuildHistory)"):]
    assert rebuild.count("Assert-Identity") >= 2                     # (b) 作る・(c) 送る の両方で止める
    assert rebuild.count("Test-HistoryIdentity") >= 2                # 作り直した後と送る前に確かめる
    assert "--format='%an %ae %cn %ce'" in t
    for v in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        assert f"$env:{v} = " in t
    assert "gh repo edit" not in t and "--visibility" not in t      # GitHub の設定は変えない
