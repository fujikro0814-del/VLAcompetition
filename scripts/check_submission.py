"""提出用リポジトリ（recovery-vla-panda）の push の前の検査（束 5 の 1）。scripts\\push_submission.ps1 が呼ぶ。単独でも回せる。

    .venv\\Scripts\\python.exe scripts\\check_submission.py --repo C:\\PAI\\recovery-vla-panda [--upstream origin/main]
    .venv\\Scripts\\python.exe scripts\\check_submission.py --repo C:\\PAI\\recovery-vla-panda --worktree   # 書き出した直後（未コミット）の下見

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
 (8) 注意だけ（止めない）: LICENSE がない、または著作権者の名前が仮の値（70_export_submission.py の LICENSE_HOLDER_PLACEHOLDER）のまま
途中のコミット（upstream..HEAD の HEAD 以外）とコミット文の使わない語は、止めずに件数だけ示す（--strict-history で途中のコミットも止める）。
鍵・手元の文字列は「ファイル名と件数」だけを出し、中身は出さない（ログに残さないため）。
"""
import argparse
import ast
import json
import pathlib
import re
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
    r"^scripts/0[1-4]_", r"^tests/test_push_check\.py$",
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=r"C:\PAI\recovery-vla-panda")
    ap.add_argument("--upstream", default="origin/main")
    ap.add_argument("--worktree", action="store_true", help="HEAD ではなく作業ツリー（追跡外を含む、.gitignore は除く）を調べる。push には使わない")
    ap.add_argument("--strict-history", action="store_true", help="途中のコミットの使わない語でも止める")
    ap.add_argument("--json", default="", help="結果を書く JSON のパス")
    ap.add_argument("--show", type=int, default=40, help="使わない語の表示の上限")
    a = ap.parse_args(argv)
    repo = pathlib.Path(a.repo)
    if not GIT.is_file():
        raise SystemExit(f"git がない: {GIT}")
    if repo.resolve() == ROOT.resolve():
        raise SystemExit("開発用のリポジトリには使わない（scripts\\push.ps1 を使う）")

    problems, notes = [], []
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
        if f.lower().endswith(".json"):
            try:
                json.loads(text)
            except ValueError as e:
                problems.append(f"JSON が読めない: {f}（{e}）")
    if fhits:
        problems.append(f"{target_name}: 使わない語 {len(fhits)} 件（{len({h['file'] for h in fhits})} ファイル）")
    # LICENSE（MIT、70_export_submission.py が書き出す）: 著作権者の名前が仮の値のままなら注意（作者が最終の段階で入れる）
    if "LICENSE" not in files:
        notes.append("LICENSE がない（70_export_submission.py export で作る）")
    else:
        lic = read("LICENSE").decode("utf-8", errors="replace")
        holder_ph = export_const("LICENSE_HOLDER_PLACEHOLDER")
        if holder_ph in lic:
            notes.append(f"LICENSE の著作権者の名前が仮の値「{holder_ph}」のまま（70_export_submission.py の LICENSE_HOLDER を直して書き出し直す）")

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
            nm = len(scan_forbidden("COMMIT_MSG.md", msg, words, []))
            if nm:
                notes.append(f"コミット {c[:7]} のコミット文: 使わない語 {nm} 件")
        seen = {}                                           # (パス, blob) → (鍵などの件数, 使わない語の件数)。変わらないファイルは一度だけ調べる
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
                    seen[(f, s)] = (scan_secrets(text, local_pats), len(scan_forbidden(f, text, words, symbols)))
                sec, nf = seen[(f, s)]
                for k, n in sec.items():
                    problems.append(f"コミット {c[:7]}: {f}: {k}: {n} 件")
                n_forb += nf
            if n_forb:
                (problems if a.strict_history else notes).append(f"コミット {c[:7]}（途中）: 使わない語 {n_forb} 件")

    res = {"repo": str(repo), "target": target_name, "files": len(files), "commits_to_push": len(commits),
           "forbidden_words": len(words), "forbidden_symbols_prose_only": len(symbols),
           "forbidden_hits": fhits, "problems": problems, "notes": notes, "ok": not problems}
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
