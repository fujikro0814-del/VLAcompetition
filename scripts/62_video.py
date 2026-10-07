"""提出用の動画（約 2 分、1920×1080、30 fps、H.264）を組み立てる。段階 3 の版（0146・0147、決裁の原則 10/07）。

    .venv\\Scripts\\python.exe scripts\\62_video.py build     # paper/build/動画_PAI最終課題_<アカウント名>.mp4
    .venv\\Scripts\\python.exe scripts\\62_video.py check     # 画面の文字に使わない語（開発中の記号を含む）がないか・長さ・場面の再現

原則: 動画は簡易な説明にとどめ、詳しい説明は説明資料に任せる。開発中の記号（モデルや失敗注入の番号）は画面に出さない。
場面の並びと長さは configs/demo/video_s3.yaml（説明資料の「動画 m:ss〜m:ss」も同じ表から計算する）。
映像は scripts/63_demo_v2.py で同じ設定で回し直したもの（outputs/demo_v2）で、元の記録と同じ経過のもの（meta.json の
same_course）だけを使う。字幕の数字は説明資料の値（paper/build/values.json、scripts/60_paper.py build）から入れる。
目標書 v1 の版（9/29 提出用）は git のタグ stepJ-freeze の版にある。
"""
import argparse
import json
import re
import subprocess

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from recovla.common import config

CFG = config.load_v2()
ROOT = config.ROOT
OUT = config.path(CFG["paths"]["outputs"])
DEMO = OUT / "demo_v2"
EVAL = OUT / "v2eval" / "V3DEMO"
BUILD = ROOT / "paper" / "build"
VIDEO = ROOT / "configs" / "demo" / "video_s3.yaml"
ACCOUNT = "アカウント名"
OUT_NAME = f"動画_PAI最終課題_{ACCOUNT}.mp4"
W, H, FPS = 1920, 1080, 30
FONT_R = r"C:\Windows\Fonts\BIZ-UDGothicR.ttc"
FONT_B = r"C:\Windows\Fonts\BIZ-UDGothicB.ttc"
BG, INK, MUTED = (246, 247, 244), (29, 33, 38), (91, 99, 109)
C_A, C_B, C_C = (45, 95, 139), (31, 107, 92), (149, 86, 25)
WITH, WITHOUT = "復帰デモありで学習したモデル", "復帰デモなしで学習したモデル"        # 説明資料と同じ語（0147 の 2）
RERUN_NOTE = "同じ設定で回し直した映像（テスト用の評価と同じ経過のもの）。推論の間も物理演算は止まらない"
COLOR = {"red": "赤", "green": "緑", "blue": "青"}
TEXTS = []                                     # 画面に出した文字（check で語を調べる）


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size)


def text(d, xy, s, size=32, fill=INK, bold=False, anchor="la"):
    TEXTS.append(s)
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def spec() -> dict:
    s = yaml.safe_load(VIDEO.read_text(encoding="utf-8"))
    return {sc["id"]: sc for sc in s["scenes"]}, s["scenes"]


class Clip:
    """outputs/demo_v2/<場面>/trial_NN.mp4 を、単調に進む時刻で読む。same_course が偽の場面は使わない。"""

    def __init__(self, ref):
        import av
        clip, trial = ref.split("/")
        k = int(trial.replace("trial_", "").replace(".mp4", ""))
        meta_p = DEMO / clip / "meta.json"
        if meta_p.is_file():
            rows = [r for r in json.loads(meta_p.read_text(encoding="utf-8"))["rows"] if r["trial"] == k]
            if not rows or not rows[0]["same_course"]:
                raise SystemExit(f"{ref} は元の記録と同じ経過ではないので使わない")
        self.n = len(np.load(DEMO / clip / f"trial_{k:02d}_frames.npz")["t"])
        self.c = av.open(str(DEMO / clip / f"trial_{k:02d}.mp4"))
        self.it = self.c.decode(video=0)
        self.i, self.img = -1, None
        self.dur = self.n / FPS

    def at(self, t):
        want = min(max(0, int(round(t * FPS))), self.n - 1)
        while self.i < want:
            try:
                self.img = next(self.it).to_ndarray(format="rgb24")
            except StopIteration:
                break
            self.i += 1
        return self.img


def header(d, title, speed=None, note=None):
    d.rectangle((0, 0, W, 86), fill=(29, 33, 38))
    text(d, (40, 22), title, 40, (255, 255, 255), True)
    if speed:
        text(d, (W - 40, 26), f"{speed:.1f} 倍速" if abs(speed - 1) > 0.05 else "実時間", 36, (255, 214, 102), True, "ra")
    if note:
        text(d, (40, H - 44), note, 22, MUTED)


def _speed(sc, clip_dur):
    t0 = float(sc.get("t0", 0.0))
    t1 = float(sc.get("t1", clip_dur))
    return t0, (t1 - t0) / float(sc["dur_s"])


# ------------------------------------------------------------------ 場面
def card(sc, title, lines):
    def f(j):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        if title:
            text(d, (W // 2, 330), title, 64, INK, True, "ma")
        y = 470 if title else 300
        for s, size, col in lines:
            text(d, (W // 2, y), s, size, col, False, "ma")
            y += size + 26
        return im
    return int(sc["dur_s"] * FPS), f


def single(sc, title, label, caption):
    clip = Clip(sc["clip"])
    t0, sp = _speed(sc, clip.dur)

    def f(j):
        t = t0 + j * sp / FPS
        im = Image.new("RGB", (W, H), BG)
        im.paste(Image.fromarray(clip.at(t)).resize((1440, 810)), (0, 86))
        d = ImageDraw.Draw(im)
        header(d, title, sp, RERUN_NOTE)
        text(d, (1470, 120), label, 28, INK, True)
        y = 200
        for s in caption:
            text(d, (1470, y), s, 26, MUTED)
            y += 40
        return im
    return int(sc["dur_s"] * FPS), f


def pair(sc, title, top):
    cl = [Clip(sc["clip_with"]), Clip(sc["clip_without"])]
    t0, sp = _speed(sc, max(c.dur for c in cl))

    def f(j):
        t = t0 + j * sp / FPS
        im = Image.new("RGB", (W, H), BG)
        for i, c in enumerate(cl):
            im.paste(Image.fromarray(c.at(t)).resize((960, 540)), (960 * i, 150))
        d = ImageDraw.Draw(im)
        header(d, title, sp, RERUN_NOTE)
        text(d, (40, 100), top, 28, C_C, True)
        for i, lab in enumerate((WITH, WITHOUT)):
            text(d, (960 * i + 30, 710), lab, 32, C_B if i == 0 else INK, True)
        return im
    return int(sc["dur_s"] * FPS), f


def task_scene(sc, title):
    clip = Clip(sc["clip"])
    run = json.loads((EVAL / sc["clip"].split("/")[0] / "run_0000.json").read_text(encoding="utf-8"))
    plan = [COLOR[c] for c in run["plan"]["steps"]]
    t0, sp = _speed(sc, clip.dur)

    def f(j):
        t = t0 + j * sp / FPS
        im = Image.new("RGB", (W, H), BG)
        im.paste(Image.fromarray(clip.at(t)).resize((1440, 810)), (0, 86))
        d = ImageDraw.Draw(im)
        header(d, title, sp, RERUN_NOTE)
        text(d, (1470, 110), WITH, 24, INK, True)
        text(d, (1470, 160), f"指示「{run['text']}」", 30, INK, True)
        text(d, (1470, 210), "LLM による分解:", 24, MUTED)
        text(d, (1470, 246), " → ".join(plan), 34, C_A, True)
        y = 320
        for s in run["steps"]:
            tj = s["attempts"][-1]["t_judge"]
            done = s["judged_complete"] and tj is not None and tj <= t
            active = s["attempts"][0]["t_begin"] <= t and not done
            mark = "✓" if done else ("▶" if active else "・")
            text(d, (1470, y), f"{mark} {COLOR[s['color']]}を箱へ", 32, C_B if done else (INK if active else MUTED), True)
            y += 50
            if len(s["attempts"]) > 1 and s["attempts"][1]["t_begin"] <= t:
                text(d, (1510, y), "待機位置へ戻して再試行", 24, C_C)
                y += 38
        return im
    return int(sc["dur_s"] * FPS), f


def fig1_image():
    """図 1 だけの HTML を Edge で画像にする（説明資料と同じ SVG・値）。"""
    import importlib.util
    s = importlib.util.spec_from_file_location("paper", ROOT / "scripts" / "60_paper.py")
    pm = importlib.util.module_from_spec(s)
    s.loader.exec_module(pm)
    v = json.loads((BUILD / "values.json").read_text(encoding="utf-8"))
    body = pm.render(v)
    style = re.search(r"<style>.*?</style>", body, re.S).group(0)
    svg = re.search(r"<svg.*?</svg>", body, re.S).group(0)
    page = BUILD / "fig1_only.html"
    page.write_text(f"<!doctype html><meta charset='utf-8'>{style}<body style='margin:0;background:#fff'>"
                    f"<div style='width:1800px;padding:20px 60px'>{svg}</div></body>", encoding="utf-8")
    png = BUILD / "fig1_video.png"
    subprocess.run([str(pm.EDGE), "--headless", "--disable-gpu", "--hide-scrollbars", f"--screenshot={png}",
                    "--window-size=1920,1400", page.as_uri()], check=True, capture_output=True, timeout=120)
    return Image.open(png).convert("RGB")


def image_scene(sc, img, title, lines):
    im0 = img.copy()
    im0.thumbnail((1500, H - 100 - 44 * len(lines) - 60))      # 説明文が図に重ならない大きさ
    dur = float(sc["dur_s"])

    def f(j):
        im = Image.new("RGB", (W, H), BG)
        im.paste(im0, ((W - im0.width) // 2, 96))
        d = ImageDraw.Draw(im)
        header(d, title)
        k = min(len(lines), 1 + j // int(dur * FPS / max(1, len(lines))))
        y = 96 + im0.height + 20
        for s in lines[:k]:
            text(d, (W // 2, y), s, 30, INK, False, "ma")
            y += 44
        return im
    return int(dur * FPS), f


def scenes(v):
    S, order = spec()
    fig1 = fig1_image()
    make = {
        "title": lambda sc: card(sc, "失敗からの復帰を学習するロボット", [
            ("日本語の指示で 3 色の立方体を箱へ片付ける（シミュレーション）", 34, MUTED),
            ("失敗からの復帰を、人手なしで自動生成したデモで学習する", 36, INK)]),
        "problem": lambda sc: single(sc, "課題: 復帰デモなしで学習すると、把持に失敗した後に回復しない", WITHOUT,
                                     ["把持に失敗した後も、", "何も持たずに箱へ向かい、", "掴み直さない"]),
        "method": lambda sc: image_scene(sc, fig1, "仕組み", [
            "カメラ画像と関節の状態だけで動く（シミュレーションの正解の情報は使わない）",
            "わざと失敗させ、自動の教師が回復する様子を記録して、学習データに加える"]),
        "contrast": lambda sc: pair(sc, "比較: 同じ配置で、失敗を注入しない試行", "左は把持に失敗した後に掴み直して成功し、右は回復しない"),
        "task": lambda sc: task_scene(sc, "実演: 日本語の指示から 3 個を順に片付ける"),
        "numbers": lambda sc: card(sc, "主な結果（評価に使っていない配置で測定）", [
            (f"把持の失敗から片方だけが回復した組: 復帰デモあり {v['e3a_xo']} 組、復帰デモなし {v['e3a_yo']} 組", 34, INK),
            (f"失敗を注入しないときの成功率 {v['e1_pct']}（{v['e1_k']}/{v['e1_n']}）", 34, INK),
            (f"3 個を続けて片付けられたのは {v['e7_k']}/{v['e7_n']}", 34, INK),
            ("詳しい数字と条件は説明資料に記載", 26, MUTED)]),
        "limits": lambda sc: card(sc, "限界と今後", [
            ("シミュレーションのみで、実機では確かめていない", 34, INK),
            ("落下や置き損ねからは、ほとんど回復できない", 34, INK)]),
    }
    return [make[sc["id"]](sc) for sc in order]


def cmd_build(a) -> None:
    import av
    v = json.loads((BUILD / "values.json").read_text(encoding="utf-8"))
    sc = scenes(v)
    total = sum(n for n, _ in sc)
    out = BUILD / "video.mp4"
    box = av.open(str(out), mode="w")
    st = box.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": "22", "preset": "medium"}
    for n, f in sc:
        for j in range(n):
            fr = av.VideoFrame.from_image(f(j))
            for p in st.encode(fr):
                box.mux(p)
    for p in st.encode():
        box.mux(p)
    box.close()
    (BUILD / OUT_NAME).write_bytes(out.read_bytes())
    uniq = sorted(set(TEXTS))
    (BUILD / "video_texts.json").write_text(json.dumps({"seconds": total / FPS, "texts": uniq}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(BUILD / OUT_NAME, f"{total / FPS:.1f} s", f"{out.stat().st_size / 1e6:.1f} MB")


def cmd_check(a) -> None:
    import importlib.util
    s = importlib.util.spec_from_file_location("paper", ROOT / "scripts" / "60_paper.py")
    pm = importlib.util.module_from_spec(s)
    s.loader.exec_module(pm)
    info = json.loads((BUILD / "video_texts.json").read_text(encoding="utf-8"))
    joined = "\n".join(info["texts"])
    hits = [(p, m.group(0)) for p in pm.FORBIDDEN for m in re.finditer(p, joined)]
    problems = [f"使わない語 {h[1]!r}（{h[0]}）" for h in hits]
    if not 60 <= info["seconds"] <= 300:
        problems.append(f"長さ {info['seconds']:.0f} s")
    S, order = spec()
    want = sum(float(sc["dur_s"]) for sc in order)
    if abs(info["seconds"] - want) > 0.5:
        problems.append(f"長さ {info['seconds']:.1f} s が場面の表の合計 {want:.0f} s と違う（説明資料の時刻がずれる）")
    size = (BUILD / OUT_NAME).stat().st_size
    res = {"seconds": round(info["seconds"], 1), "mb": round(size / 1e6, 1), "texts": len(info["texts"]), "problems": problems}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    raise SystemExit(1 if problems else 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    a = ap.parse_args(argv)
    {"build": cmd_build, "check": cmd_check}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
