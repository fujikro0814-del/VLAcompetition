"""提出用リポジトリ（recovery-vla-panda）の push の前の検査（束 5 の 1）。scripts\\push_submission.ps1 が呼ぶ。単独でも回せる。

    .venv\\Scripts\\python.exe scripts\\check_submission.py --repo C:\\PAI\\recovery-vla-panda [--upstream origin/main]
    .venv\\Scripts\\python.exe scripts\\check_submission.py --repo C:\\PAI\\recovery-vla-panda --worktree   # 書き出した直後（未コミット）の下見
    .venv\\Scripts\\python.exe scripts\\check_submission.py --repo C:\\PAI\\recovery-vla-panda --pii-only   # 個人情報 (9) だけ（PDF・動画の文字も）

どれか 1 つでも当たれば exit 1（push しない）。開発用のリポジトリの check_before_push.ps1 と違い、目標書・G1 の検査はしない
（提出用リポジトリには目標書がない）。
 (1) 作業ツリーがきれい（未コミットの変更・追跡外のファイルがない）。検査した中身と送る中身を一致させるため（--worktree では見ない）
 (2) 鍵・証明書: 送るコミットすべて（upstream..HEAD）と HEAD の全ファイルに、鍵の形の文字列・証明書がない。鍵の拡張子のファイルがない
 (3) 手元だけの文字列（開発用のリポジトリの .local/push_check_patterns.txt、固定の文字列）と、個人の名前の文字列: 同上
 (4) 使わない語: 60_paper.py の FORBIDDEN を HEAD の全テキストファイルにかける。ただし記号の型（R1・P1・E1・G1 など、ASCII だけの型）は
     コードの識別子・実行例の引数として使うので、文章のファイル（.md・.html・.svg）の、コードの部分（```…```・`…`）を除いた所だけにかける。
     例外は 2 つだけ（ALLOW_CONTEXT: 「通して・通した」、SYMBOL_OK_FILES: 記録の形式の設計文書 docs/interfaces/）
 (5) 入れないもの: 段階 4・作業記録・outputs・models・push の道具・取り込みのスクリプトなどのパスがない
 (6) 100 MB を超えるファイルがない（GitHub の上限。20 MB を超えたら注意だけ）、.json がすべて読める
 (7) 送るコミットの作者・コミッタのメールが GitHub の noreply、コミット文に鍵・手元の文字列・個人の名前がない
 (8) LICENSE がない（注意だけ）。著作権者（= omnicampus のアカウント名）が仮の値（70_export_submission.py の PLACEHOLDER）のまま
     なら止める（--worktree の下見では注意だけ）
 (9) 個人情報（課題の 10/8 版: PDF・コード・動画に氏名・所属・メールアドレスを載せない。omnicampus のアカウント名は可）:
     HEAD（--worktree では作業ツリー）の全テキストのファイル、送るコミットのコミット文、説明資料の PDF の本文とメタデータ（pypdf で
     読む。読めなければその旨を出し、組み上げた HTML paper/build/paper.html を代わりに調べる）、動画の文字の一覧
     （paper/build/video_texts.json）に、メールアドレスの形・個人のパス（C:\\Users\\<名前>・/home/<名前>・/Users/<名前>）・
     git の設定の user.name と user.email の値（git config --get で取れれば）がない。許す語は PII_ALLOW（GitHub の noreply・
     共同作成者の行・例示用のドメイン）と、60_paper.py の ACCOUNT・--pii-allow・.local/pii_allow.txt（1 行 1 語）
     --pii-only: (9) だけを、--repo のフォルダの全テキストのファイル（.git を除く）と PDF・動画の文字にかける（.tools の git がなくても回る）
途中のコミット（upstream..HEAD の HEAD 以外）とコミット文の使わない語は、止めずに件数だけ示す（--strict-history で途中のコミットも止める）。
鍵・手元の文字列は「ファイル名と件数」だけを出し、中身は出さない（ログに残さないため）。
"""
import argparse
import ast
import json
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]           # 開発用のリポジトリ（60_paper.py と .local の照合文字列がある）
GIT = ROOT / ".tools" / "git" / "cmd" / "git.exe"
PAPER = ROOT / "scripts" / "60_paper.py"
LOCAL_PATTERNS = ROOT / ".local" / "push_check_patterns.txt"

# 目印の文字列は、この検査自身に当たらないように分けて書く（check_before_push.ps1 と同じ型）
_D = "-" * 5
KEY_PATTERNS = {
    "Anthropic の鍵": r"sk-ant-[A-Za-z0-9_-]{20,}",
    "OpenAI の鍵": r"sk-(?:proj-)?[A-Za-z0-9]{32,}",
    "GitHub の鍵": r"gh[pousr]_[A-Za-z0-9]{20,}",
    "GitHub の鍵(PAT)": r"github_pat_[A-Za-z0-9_]{20,}",
    "Hugging Face の鍵": r"hf_[A-Za-z0-9]{30,}",
    "AWS の鍵": r"AKIA[0-9A-Z]{16}",
    "Google の鍵": r"AIza[0-9A-Za-z_-]{35}",
    "秘密鍵": _D + r"BEGIN [A-Z ]*PRIVATE KEY" + _D,
    "証明書": _D + r"BEGIN CERTIFICATE" + _D,
    "鍵の代入": r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
}
# 個人・端末の名前（Windows のユーザー名・アカウント名・学内の略称）。大文字小文字を区別しない
PERSONAL = [r"C:\\+Users\\+", "student", "fujikro", "secec"]
# 入れないもののパス（提出用リポジトリの中の相対パス）
FORBIDDEN_PATHS = [
    r"^(outputs|models|\.local|\.venv|\.tools|\.python|\.cache|sealed)/", r"^docs/(board|local)/", r"目標書", r"種の台帳",
    r"(^|/)(push|check_before_push|push_submission)\.ps1$", r"^scripts/(70_export_submission|check_submission)\.py$",
    r"^scripts/0[1-4]_", r"^tests/test_push_check\.py$", r"^tests/test_submission_1008\.py$",
    # 段階 4（作業中）
    r"^src/recovla/diag/", r"time_scoring", r"decompose_s4", r"(?:^|[/_])s4(?:[_./]|$)", r"^scripts/9[6-9]_",
    r"^src/recovla/eval/gate1\.py$", r"^tests/test_gate1\.py$",
    # 鍵・大きなデータ・チェックポイント
    r"\.(pem|crt|cer|der|p12|pfx|key)$", r"(^|/)\.env$", r"\.(safetensors|ckpt|pt|pth|bin|parquet|npz|mp4|zip)$",
]
# FORBIDDEN の語のうち、普通の日本語の形にも当たるものの例外（当たった所がこの型の中に入っていれば数えない）。
# 「通し」は「通して・通した」（動詞）を除く。70_export_submission.py の置き換えと語の一覧（SCAN）と同じ例外
ALLOW_CONTEXT = {"通し": r"見通し|通し(?=[てたまか])"}
# 記号の型（R1・P1・E1 など）を許すファイル。記録の形式の設計文書は、コードが書く値そのもの（model の値 R1・N1 など）を並べるため
SYMBOL_OK_FILES = [r"^docs/interfaces/"]
TEXT_EXT = {".py", ".yaml", ".yml", ".xml", ".ps1", ".md", ".txt", ".toml", ".html", ".svg", ".json", ".cfg", ".ini", ".csv"}
PROSE_EXT = {".md", ".html", ".svg"}
BIG, WARN_BIG = 100 * 1024 * 1024, 20 * 1024 * 1024
# (9) 個人情報（課題の 10/8 版）。当たった文字は一部を伏せて出す（ログに残さないため。場所で探す）
BUILD = ROOT / "paper" / "build"
EMAIL_RE = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
HOME_RE = r"(?i)(?:(?<![A-Za-z])[A-Z]:[\\/]+Users|(?<![\w.%:-])/(?:home|Users))[\\/]+(?P<name>[^\\/\s\"'<>|:*?`]+)"
# 個人のパスで許す名前（だれのものでもない名前・環境変数の書き方）。小文字で比べる
PII_GENERIC_USERS = {"public", "default", "default user", "all users", "user", "username", "runner", "runneradmin",
                     "%username%", "$env:username", "${env:username}", "$user", "${user}", "~"}
# 許すメールアドレス・名前（全体が一致すれば許す。大文字小文字を区別しない）。GitHub の noreply（作者・コミッタのアドレス）、
# 共同作成者の行（Co-Authored-By: Claude <noreply@anthropic.com>）、GitHub の決まったアドレス、例示用のドメイン（RFC 2606）
PII_ALLOW = [r"[\w.+-]+@users\.noreply\.github\.com", r"noreply@anthropic\.com", r"Claude", r"noreply@github\.com",
             r"git@github\.com", r"[\w.+-]+@example\.(?:com|org|net)"]
PII_ALLOW_FILE = ROOT / ".local" / "pii_allow.txt"           # 手元だけの許す語（1 行 1 語、# で始まる行は注釈）


def git(repo, *args, check=True) -> str:
    r = subprocess.run([str(GIT), "-C", str(repo), "-c", "core.quotepath=false", *args], capture_output=True)
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout.decode("utf-8", "replace")


def forbidden_patterns():
    """60_paper.py の FORBIDDEN を、読み込まずに（import の副作用なしで）取り出す。記号の型（ASCII だけ）と語の型に分ける。"""
    for node in ast.parse(PAPER.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FORBIDDEN" for t in node.targets):
            pats = ast.literal_eval(node.value)
            break
    else:
        raise SystemExit("60_paper.py に FORBIDDEN がない")
    symbols = [p for p in pats if p.isascii() and "VLA" not in p]
    words = [p for p in pats if p not in symbols]
    return words, symbols


def paper_const(name: str):
    """60_paper.py の一番上の `name = ...` の値を、読み込まずに取る（ACCOUNT）。"""
    for node in ast.parse(PAPER.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"60_paper.py に {name} がない")


def export_const(name: str):
    """70_export_submission.py の一番上の `name = "..."` の値を、読み込まずに取る（LICENSE の仮の名前）。"""
    for node in ast.parse((ROOT / "scripts" / "70_export_submission.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"70_export_submission.py に {name} がない")


def strip_code(text: str, ext: str) -> str:
    """文章のファイルから、コードの部分を空白に置き換える（行番号を保つため改行は残す）。"""
    blank = lambda m: re.sub(r"[^\n]", " ", m.group(0))
    if ext == ".md":
        text = re.sub(r"```.*?```", blank, text, flags=re.S)
        text = re.sub(r"`[^`\n]*`", blank, text)
    else:                                                     # .html・.svg: タグ・style・script・<code> の中は見ない
        text = re.sub(r"<(style|script|code|pre)\b.*?</\1>", blank, text, flags=re.S | re.I)
        text = re.sub(r"<[^>]*>", blank, text)
    return text


def tree_blobs(repo, rev):
    """rev の全ファイル（パス → blob の SHA）。"""
    out = {}
    for line in git(repo, "ls-tree", "-r", rev).splitlines():
        meta, _, path = line.partition("\t")
        mode, typ, sha = meta.split()
        if typ == "blob":
            out[path] = sha
    return out


class Blobs:
    """blob の中身を git cat-file --batch でまとめて読む（同じ blob は一度だけ）。"""

    def __init__(self, repo):
        self.repo, self.cache = repo, {}

    def load(self, shas):
        need = [s for s in dict.fromkeys(shas) if s not in self.cache]
        if not need:
            return
        out = subprocess.run([str(GIT), "-C", str(self.repo), "cat-file", "--batch"], input=("\n".join(need) + "\n").encode(),
                             capture_output=True, check=True).stdout
        pos = 0
        for s in need:
            nl = out.index(b"\n", pos)
            sha, typ, size = out[pos:nl].decode().split()
            size = int(size)
            self.cache[s] = out[nl + 1:nl + 1 + size]
            pos = nl + 1 + size + 1

    def __getitem__(self, sha):
        self.load([sha])
        return self.cache[sha]


def scan_secrets(text: str, local_pats):
    """鍵・手元の文字列・個人の名前の件数（名前 → 件数）。中身は返さない。"""
    out = {}
    for name, pat in KEY_PATTERNS.items():
        n = len(re.findall(pat, text))
        if n:
            out[name] = n
    for i, p in enumerate(local_pats, 1):
        n = text.count(p)
        if n:
            out[f"手元の照合文字列 {i} 行目"] = n
    for p in PERSONAL:
        n = len(re.findall(p, text, flags=re.I))
        if n:
            out[f"個人・端末の名前 {p!r}"] = n
    return out


def scan_forbidden(path: str, text: str, words, symbols):
    ext = pathlib.PurePosixPath(path).suffix.lower()
    hits = []
    targets = [(words, text)]
    if ext in PROSE_EXT and not any(re.search(r, path) for r in SYMBOL_OK_FILES):
        targets.append((symbols, strip_code(text, ext)))
    for pats, t in targets:
        for p in pats:
            ok = [m.span() for m in re.finditer(ALLOW_CONTEXT[p], t)] if p in ALLOW_CONTEXT else []
            for m in re.finditer(p, t):
                if any(s <= m.start() and m.end() <= e for s, e in ok):
                    continue
                line = t.count("\n", 0, m.start()) + 1
                hits.append({"file": path, "line": line, "pattern": p, "match": m.group(0),
                             "context": t[max(0, m.start() - 15):m.end() + 15].replace("\n", " ").strip()})
    return hits


def is_text(path: str) -> bool:
    p = pathlib.PurePosixPath(path)
    return p.suffix.lower() in TEXT_EXT or p.name in (".gitignore", "LICENSE", "README", "NOTICE")


# ------------------------------------------------------------------ (9) 個人情報
def git_exe():
    """git の実行ファイル。.tools のものがなければ PATH の git（--pii-only・テスト用）。どちらもなければ None。"""
    return str(GIT) if GIT.is_file() else shutil.which("git")


def pii_allowed(s: str, allow: set) -> bool:
    return any(re.fullmatch(p, s, flags=re.I) for p in PII_ALLOW) or s.strip().lower() in allow


def pii_setup(repo, extra_allow):
    """許す語（小文字の集合）と、探す git の設定の値 [(種類, 型)] と注意。"""
    notes = []
    allow = {s.strip().lower() for s in extra_allow if s.strip()}
    acc = paper_const("ACCOUNT")
    if acc and acc != export_const("PLACEHOLDER"):
        allow.add(acc.strip().lower())                       # omnicampus のアカウント名は載せてよい
    if PII_ALLOW_FILE.is_file():
        allow |= {l.strip().lower() for l in PII_ALLOW_FILE.read_text(encoding="utf-8-sig").splitlines()
                  if l.strip() and not l.startswith("#")}
    terms = []
    exe = git_exe()
    if exe is None:
        notes.append("git がないので、git の設定の user.name・user.email の値は調べていない")
        return allow, terms, notes
    for key in ("user.name", "user.email"):
        vals = set()
        for d in (repo, ROOT):                               # 提出用リポジトリと開発用のリポジトリの設定（全体の設定を含む）
            r = subprocess.run([exe, "-C", str(d), "config", "--get", key], capture_output=True)
            v = r.stdout.decode("utf-8", "replace").strip()
            if v:
                vals.add(v)
        for v in sorted(vals):
            if pii_allowed(v, allow):
                continue
            if len(v) < 3:
                notes.append(f"git の設定の {key} が短すぎる（{len(v)} 字）ので探していない")
                continue
            pat = re.escape(v)
            if v.isascii():
                pat = rf"(?<![A-Za-z0-9]){pat}(?![A-Za-z0-9])"
            terms.append((f"git の設定の {key} の値", re.compile(pat, re.I)))
    return allow, terms, notes


def _mask(kind: str, m) -> str:
    s = m.group(0)
    if kind == "個人のパス":
        k = m.start("name") - m.start()
        return s[:k] + s[k] + "***"
    if kind == "メールアドレス":
        local, _, dom = s.partition("@")
        return local[:1] + "***@" + dom
    return s[:1] + "***"


def scan_pii(where: str, text: str, terms, allow) -> list:
    """メールアドレスの形・個人のパス・git の設定の値に当たった所（許す語を除く）。"""
    hits, seen = [], set()

    def add(kind, m):
        line = text.count("\n", 0, m.start()) + 1
        if (line, m.start()) not in seen:
            seen.add((line, m.start()))
            hits.append({"where": where, "line": line, "kind": kind, "masked": _mask(kind, m)})
    for m in re.finditer(EMAIL_RE, text):
        if not pii_allowed(m.group(0), allow):
            add("メールアドレス", m)
    for m in re.finditer(HOME_RE, text):
        name = m.group("name")
        if name.lower() not in PII_GENERIC_USERS and not pii_allowed(name, allow):
            add("個人のパス", m)
    for kind, rx in terms:
        for m in rx.finditer(text):
            add(kind, m)
    return hits


def pdf_texts(pdf: pathlib.Path):
    """PDF の本文（ページごと）とメタデータ。[(場所, 文)] と注意。読めなければ (None, 理由)。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None, "pypdf がない（.venv\\Scripts\\python.exe -m pip install pypdf）"
    try:
        r = PdfReader(str(pdf))
        out = [(f"{pdf.name} の {i} ページ", p.extract_text() or "") for i, p in enumerate(r.pages, 1)]
        meta = r.metadata or {}
        out.append((f"{pdf.name} のメタデータ", "\n".join(f"{k} {v}" for k, v in meta.items())))
    except Exception as e:                                   # 壊れた PDF・暗号化など
        return None, f"読めない（{type(e).__name__}: {e}）"
    if not any(t.strip() for _, t in out[:-1]):
        return out, f"{pdf.name} の本文の文字を取り出せない（画像だけの PDF か）。目でも確かめる"
    return out, ""


def pii_side_targets(pdf_arg: str, video_texts_arg: str):
    """書き出し先の外の提出物: 説明資料の PDF と動画の文字の一覧。[(場所, 文)] と注意。"""
    targets, notes = [], []
    if pdf_arg:
        pdf = pathlib.Path(pdf_arg)
    else:
        pdf = BUILD / f"説明資料_PAI最終課題_{paper_const('ACCOUNT')}.pdf"
        if not pdf.is_file():
            pdf = BUILD / "paper.pdf"
    if not pdf.is_file():
        notes.append(f"説明資料の PDF がない: {pdf}（PDF の個人情報は調べていない。60_paper.py build の後にやり直す）")
    else:
        got, why = pdf_texts(pdf)
        if why:
            notes.append(f"PDF: {why}")
        if got is not None:
            targets += got
        else:
            html_p = pdf.parent / "paper.html"
            if html_p.is_file():
                notes.append(f"PDF の本文を読めないので、組み上げた HTML {html_p.name} を代わりに調べた")
                targets.append((f"{html_p.name}（PDF の代わり）", html_p.read_text(encoding="utf-8", errors="replace")))
    vt = pathlib.Path(video_texts_arg) if video_texts_arg else BUILD / "video_texts.json"
    if not vt.is_file():
        notes.append(f"動画の文字の一覧がない: {vt}（動画の個人情報は調べていない。62_video.py build の後にやり直す）")
    else:
        texts = json.loads(vt.read_text(encoding="utf-8")).get("texts", [])
        targets += [(f"{vt.name} の {i} 番目の文字", t) for i, t in enumerate(texts, 1)]
    return targets, notes


def pii_lines(hits) -> list:
    return [f"個人情報: {h['where']}:{h['line']} {h['kind']}（{h['masked']}）" for h in hits]


def main_pii_only(a, repo) -> int:
    """--pii-only: 書き出し先のフォルダ・PDF・動画の文字に、個人情報の検査だけをかける。"""
    if not repo.is_dir():
        raise SystemExit(f"フォルダがない: {repo}")
    allow, terms, notes = pii_setup(repo, a.pii_allow)
    hits, n = [], 0
    for f in sorted(repo.rglob("*")):
        rel = f.relative_to(repo)
        if not f.is_file() or ".git" in rel.parts or not is_text(rel.as_posix()):
            continue
        n += 1
        hits += scan_pii(rel.as_posix(), f.read_bytes().decode("utf-8", errors="replace"), terms, allow)
    side, sn = pii_side_targets(a.pdf, a.video_texts)
    notes += sn
    for where, text in side:
        hits += scan_pii(where, text, terms, allow)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"repo": str(repo), "files": n, "pii_hits": hits, "notes": notes,
                                                    "ok": not hits}, ensure_ascii=False, indent=1), encoding="utf-8")
    for x in notes:
        print("注意: " + x)
    if hits:
        print(f"個人情報の検査: 不合格（{len(hits)} 件）。push しない。")
        for x in pii_lines(hits):
            print("  " + x)
        return 1
    print(f"個人情報の検査: 合格（{repo}、{n} ファイル、PDF・動画の文字 {len(side)} 件）")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=r"C:\PAI\recovery-vla-panda")
    ap.add_argument("--upstream", default="origin/main")
    ap.add_argument("--worktree", action="store_true", help="HEAD ではなく作業ツリー（追跡外を含む、.gitignore は除く）を調べる。push には使わない")
    ap.add_argument("--strict-history", action="store_true", help="途中のコミットの使わない語でも止める")
    ap.add_argument("--json", default="", help="結果を書く JSON のパス")
    ap.add_argument("--show", type=int, default=40, help="使わない語の表示の上限")
    ap.add_argument("--pii-only", action="store_true", help="個人情報の検査 (9) だけを、--repo のフォルダ（.git を除く）・PDF・動画の文字にかける")
    ap.add_argument("--pii-allow", action="append", default=[], help="個人情報の検査で許す語（何度でも。アカウント名は自動で許す）")
    ap.add_argument("--pdf", default="", help="説明資料の PDF（既定 paper/build/説明資料_PAI最終課題_<アカウント名>.pdf、なければ paper.pdf）")
    ap.add_argument("--video-texts", default="", help="動画の文字の一覧（既定 paper/build/video_texts.json）")
    a = ap.parse_args(argv)
    repo = pathlib.Path(a.repo)
    if repo.resolve() == ROOT.resolve():
        raise SystemExit("開発用のリポジトリには使わない（scripts\\push.ps1 を使う）")
    if a.pii_only:
        return main_pii_only(a, repo)
    if not GIT.is_file():
        raise SystemExit(f"git がない: {GIT}")

    problems, notes = [], []
    pii_allow, pii_terms, pii_notes = pii_setup(repo, a.pii_allow)
    notes += pii_notes
    pii_hits = []
    words, symbols = forbidden_patterns()
    local_pats = []
    if LOCAL_PATTERNS.is_file():
        local_pats = [l for l in LOCAL_PATTERNS.read_text(encoding="utf-8-sig").splitlines() if l and not l.startswith("#")]
    else:
        problems.append(f"照合用の文字列のファイルがない: {LOCAL_PATTERNS}")

    # (1) 作業ツリーがきれい
    status = git(repo, "status", "--porcelain", "--untracked-files=all")
    if status.strip() and not a.worktree:
        problems.append(f"作業ツリーに未コミットの変更・追跡外のファイルがある（{len(status.splitlines())} 件）。コミットしてからやり直す")

    # 調べる対象のファイル（相対パス → 中身の読み方）
    blobs = Blobs(repo)
    if a.worktree:
        files = [f for f in git(repo, "ls-files", "--cached", "--others", "--exclude-standard").splitlines()
                 if f and (repo / f).is_file()]
        read = lambda f: (repo / f).read_bytes()
        target_name = "作業ツリー"
    else:
        head_tree = tree_blobs(repo, "HEAD")
        files = sorted(head_tree)
        blobs.load(head_tree.values())
        read = lambda f: blobs[head_tree[f]]
        target_name = "HEAD " + git(repo, "rev-parse", "--short", "HEAD").strip()

    # (5) 入れないもののパス
    for f in files:
        for p in FORBIDDEN_PATHS:
            if re.search(p, f):
                problems.append(f"入れないもののパス: {f}（{p}）")
                break

    # (2)(3)(4)(6) 中身
    fhits = []
    for f in files:
        data = read(f)
        if len(data) > BIG:
            problems.append(f"100 MB を超える: {f}（{len(data) / 1e6:.1f} MB）")
        elif len(data) > WARN_BIG:
            notes.append(f"20 MB を超える: {f}（{len(data) / 1e6:.1f} MB）")
        if not is_text(f):
            continue
        text = data.decode("utf-8", errors="replace")
        for k, n in scan_secrets(text, local_pats).items():
            problems.append(f"{target_name}: {f}: {k}: {n} 件")
        fhits += scan_forbidden(f, text, words, symbols)
        pii_hits += scan_pii(f, text, pii_terms, pii_allow)
        if f.lower().endswith(".json"):
            try:
                json.loads(text)
            except ValueError as e:
                problems.append(f"JSON が読めない: {f}（{e}）")
    if fhits:
        problems.append(f"{target_name}: 使わない語 {len(fhits)} 件（{len({h['file'] for h in fhits})} ファイル）")
    # LICENSE（MIT、70_export_submission.py が書き出す）: 著作権者（= omnicampus のアカウント名）が仮の値のままなら止める
    if "LICENSE" not in files:
        notes.append("LICENSE がない（70_export_submission.py export で作る）")
    else:
        lic = read("LICENSE").decode("utf-8", errors="replace")
        holder_ph = export_const("PLACEHOLDER")
        if holder_ph in lic:
            (notes if a.worktree else problems).append(
                f"LICENSE の著作権者が仮の値「{holder_ph}」のまま（60_paper.py・62_video.py の ACCOUNT を omnicampus のアカウント名に直して書き出し直す）")
    # (9) 個人情報: 書き出し先の外の提出物（説明資料の PDF・動画の文字）
    side, side_notes = pii_side_targets(a.pdf, a.video_texts)
    notes += side_notes
    for where, text in side:
        pii_hits += scan_pii(where, text, pii_terms, pii_allow)

    # 送るコミットすべて: 鍵・手元の文字列（git grep、中身は出さない）と、途中のコミットの使わない語（件数だけ）
    commits = []
    if not a.worktree:
        has_up = subprocess.run([str(GIT), "-C", str(repo), "rev-parse", "--verify", "--quiet", a.upstream],
                                capture_output=True).returncode == 0
        if not has_up:
            notes.append(f"{a.upstream} がない（fetch していない・初回）: 全コミットを調べる")
        commits = [c for c in git(repo, "rev-list", f"{a.upstream}..HEAD" if has_up else "HEAD").splitlines() if c]
        head = git(repo, "rev-parse", "HEAD").strip()
        # コミットの作者・コミット文も公開される: メールは GitHub の noreply だけ、文に鍵・個人の名前がない（使わない語は件数だけ）
        for c in commits:
            meta = git(repo, "show", "-s", "--format=%ae%n%ce", c).split()
            for e in meta:
                if not e.endswith("@users.noreply.github.com"):
                    problems.append(f"コミット {c[:7]}: 作者・コミッタのメールが GitHub の noreply でない")
                    break
            msg = git(repo, "show", "-s", "--format=%B", c)
            for k, n in scan_secrets(msg, local_pats).items():
                if not k.startswith("個人・端末の名前 'fujikro'"):          # GitHub のユーザー名（URL に出るもの）は除く
                    problems.append(f"コミット {c[:7]} のコミット文: {k}: {n} 件")
            pii_hits += scan_pii(f"コミット {c[:7]} のコミット文", msg, pii_terms, pii_allow)
            nm = len(scan_forbidden("COMMIT_MSG.md", msg, words, []))
            if nm:
                notes.append(f"コミット {c[:7]} のコミット文: 使わない語 {nm} 件")
        seen = {}                                           # (パス, blob) → (鍵などの件数, 使わない語の件数, 個人情報)。変わらないファイルは一度だけ調べる
        for c in commits:
            tree = tree_blobs(repo, c)
            for f in tree:
                for p in FORBIDDEN_PATHS:
                    if re.search(p, f):
                        problems.append(f"コミット {c[:7]}: 入れないもののパス: {f}")
                        break
            if c == head:
                continue                                    # HEAD は上で全部調べた
            blobs.load(s for f, s in tree.items() if is_text(f))
            n_forb = 0
            for f, s in tree.items():
                if not is_text(f):
                    continue
                if (f, s) not in seen:
                    text = blobs[s].decode("utf-8", errors="replace")
                    seen[(f, s)] = (scan_secrets(text, local_pats), len(scan_forbidden(f, text, words, symbols)),
                                    scan_pii(f, text, pii_terms, pii_allow))
                sec, nf, ph = seen[(f, s)]
                for k, n in sec.items():
                    problems.append(f"コミット {c[:7]}: {f}: {k}: {n} 件")
                problems += [f"コミット {c[:7]}（途中）: " + x for x in pii_lines(ph)]       # 途中のコミットも公開される
                n_forb += nf
            if n_forb:
                (problems if a.strict_history else notes).append(f"コミット {c[:7]}（途中）: 使わない語 {n_forb} 件")

    problems += pii_lines(pii_hits)
    res = {"repo": str(repo), "target": target_name, "files": len(files), "commits_to_push": len(commits),
           "forbidden_words": len(words), "forbidden_symbols_prose_only": len(symbols),
           "forbidden_hits": fhits, "pii_hits": pii_hits, "problems": problems, "notes": notes, "ok": not problems}
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for h in fhits[: a.show]:
        print(f"  使わない語 {h['file']}:{h['line']} {h['match']!r} …{h['context']}…")
    if len(fhits) > a.show:
        print(f"  …ほか {len(fhits) - a.show} 件（--json で全部）")
    for n in notes:
        print("注意: " + n)
    if problems:
        print(f"提出用リポジトリの検査: 不合格（{len(problems)} 件）。push しない。")
        for p in sorted(set(problems)):
            print("  " + p)
        return 1
    print(f"提出用リポジトリの検査: 合格（{target_name}、{len(files)} ファイル、送るコミット {len(commits)}）")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
