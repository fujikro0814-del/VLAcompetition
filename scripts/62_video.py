"""提出用の動画（1〜5 分、1920×1080、30 fps、H.264）を、場面の回し直し（scripts/61_demo.py、outputs/demo）と
説明資料の値（paper/build/values.json、scripts/60_paper.py build）から組み立てる。字幕の数字も values.json から入れる。

    .venv\\Scripts\\python.exe scripts\\62_video.py build     # paper/build/動画_PAI最終課題_<アカウント名>.mp4
    .venv\\Scripts\\python.exe scripts\\62_video.py check     # 画面の文字に使わない語がないか・長さ・使った場面が再現したものか

重ね表示: モデルの名前（常に）、倍率、失敗注入と復帰の出来事の名前、アクションチャンクの予測経路（推論遅延の d ステップと
その先を色分け）、手先の速さの波形（チャンク境界に縦線）、3 個の連続タスクではサブタスクの一覧（完了で印）。
"""
import argparse
import json
import pathlib
import re
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from recovla.common import config

CFG = config.load()
ROOT = config.ROOT
DEMO = config.path(CFG["paths"]["outputs"]) / "demo"
BUILD = ROOT / "paper" / "build"
ACCOUNT = "アカウント名"
OUT_NAME = f"動画_PAI最終課題_{ACCOUNT}.mp4"
W, H, FPS = 1920, 1080, 30
FONT_R = r"C:\Windows\Fonts\BIZ-UDGothicR.ttc"
FONT_B = r"C:\Windows\Fonts\BIZ-UDGothicB.ttc"
BG, INK, MUTED = (246, 247, 244), (29, 33, 38), (91, 99, 109)
C_A, C_B, C_C = (45, 95, 139), (31, 107, 92), (149, 86, 25)
MODEL_NAME = {"R1": "復帰デモありで学習したモデル（R1）", "N1": "復帰デモなしで学習したモデル（N1）", "R2": "最終モデル（R2）"}
MODE_NAME = {"naive": "非同期実行", "sync": "同期実行", "rtc": "非同期実行＋RTC"}
EVENT_NAME = {"grasp_miss": "掴み損ね", "fire": "失敗注入（掴む位置をずらす）", "success": "箱に入った（成功）"}
RERUN_NOTE = "同じ設定で回し直した映像（結果は評価の本番と同じ。ビット一致ではない）"
TEXTS = []                                     # 画面に出した文字（check で語を調べる）


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size)


def text(d, xy, s, size=32, fill=INK, bold=False, anchor="la"):
    TEXTS.append(s)
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


class Clip:
    """outputs/demo/<場面> の発表用カメラの映像を、単調に進む時刻で読む。"""

    def __init__(self, cid):
        import av
        self.cid = cid
        self.meta = json.loads((DEMO / cid / "meta.json").read_text(encoding="utf-8"))
        if not self.meta["reproduced"]:
            raise SystemExit(f"{cid} は回し直しで再現していないので使わない")
        self.fr = np.load(DEMO / cid / "frames.npz")
        self.tr = np.load(DEMO / cid / "trace.npz") if (DEMO / cid / "trace.npz").exists() else None
        self.c = av.open(str(DEMO / cid / "presentation.mp4"))
        self.it = self.c.decode(video=0)
        self.i, self.img = -1, None
        self.n = len(self.fr["t"])
        self.cam = self.meta["camera"]

    def at(self, t):
        want = min(max(0, int(round(t * FPS))), self.n - 1)
        while self.i < want:
            try:
                self.img = next(self.it).to_ndarray(format="rgb24")
            except StopIteration:
                break
            self.i += 1
        return self.img, min(self.i, self.n - 1)

    def project(self, p):
        """世界の点 (N,3) → 発表用カメラの画素 (N,2)。"""
        R = np.array(self.cam["xmat"])
        pc = (np.asarray(p) - np.array(self.cam["pos"])) @ R        # カメラの x・y・z 軸の成分（z は後ろ向き）
        f = (self.cam["height"] / 2) / np.tan(np.radians(self.cam["fovy"]) / 2)
        z = -pc[:, 2]
        return np.stack([self.cam["width"] / 2 + f * pc[:, 0] / z, self.cam["height"] / 2 - f * pc[:, 1] / z], 1)

    def chunk_path(self, t):
        """その時刻に実行中のアクションチャンクの予測経路（指先の高さに下げたもの）と、推論遅延の行数。"""
        if self.tr is None:
            return None
        k = int(t / 0.1)
        ex = self.tr["exec"]
        if k >= len(ex) or ex[k][0] < 0:
            return None
        i = int(ex[k][0])
        pts = self.tr["chunk_xdes_pred"][i] - np.array([0, 0, float(CFG["sim"]["fingertip_offset"])])
        return self.project(pts)

    def switches(self):
        ex = self.tr["exec"] if self.tr is not None else np.zeros((0, 2))
        return [k * 0.1 for k in range(1, len(ex)) if ex[k][0] != ex[k - 1][0] and ex[k][0] >= 0]

    def speed(self):
        t, p = self.fr["t"], self.fr["fingertip"]
        v = np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)]
        return t, v


def draw_video(canvas, clip, t, box, path=True):
    """clip の時刻 t のフレームを box=(x, y, w, h) に置き、予測経路を重ねる。"""
    img, _ = clip.at(t)
    x, y, w, h = box
    im = Image.fromarray(img).resize((w, h))
    d = ImageDraw.Draw(im)
    pp = clip.chunk_path(t) if path else None
    if pp is not None:
        s = w / clip.cam["width"]
        dd = int(CFG["runtime"]["delay_steps"])
        pts = [(float(a * s), float(b * s)) for a, b in pp]
        d.line(pts[:dd + 1], fill=(230, 126, 34), width=5)
        d.line(pts[dd:], fill=(0, 170, 200), width=4)
    canvas.paste(im, (x, y))


def draw_speed(d, clip, t, box, window=8.0):
    """手先の速さ（直近 window 秒）と、チャンク境界の縦線。"""
    x, y, w, h = box
    d.rectangle(box_xyxy(box), fill=(255, 255, 255), outline=(207, 212, 207))
    tt, v = clip.speed()
    t0 = max(0.0, t - window)
    sel = (tt >= t0) & (tt <= t)
    vmax = 0.25
    for s in clip.switches():
        if t0 <= s <= t:
            xs = x + (s - t0) / window * w
            d.line([(xs, y + 4), (xs, y + h - 4)], fill=(149, 86, 25), width=1)
    if sel.sum() > 1:
        pts = [(x + (a - t0) / window * w, y + h - 6 - min(b, vmax) / vmax * (h - 34)) for a, b in zip(tt[sel], v[sel])]
        d.line(pts, fill=C_A, width=3)
    text(d, (x + 10, y + 6), "手先の速さ（縦線はチャンク境界）", 20, MUTED)


def box_xyxy(b):
    return (b[0], b[1], b[0] + b[2], b[1] + b[3])


def header(d, title, speed=None, note=None):
    d.rectangle((0, 0, W, 86), fill=(29, 33, 38))
    text(d, (40, 22), title, 40, (255, 255, 255), True)
    if speed:
        text(d, (W - 40, 26), f"{speed:g} 倍速" if speed != 1 else "実時間", 36, (255, 214, 102), True, "ra")
    if note:
        text(d, (40, H - 44), note, 22, MUTED)


def events_of(clip):
    m = clip.meta
    ev = []
    ind = m.get("induce") or {}
    if ind.get("t_fire") is not None:
        ev.append((ind["t_fire"], EVENT_NAME["fire"]))
    for e in m.get("events", []):
        ev.append((e["t"], EVENT_NAME.get(e["kind"], e["kind"])))
    if m.get("t_success") is not None:
        ev.append((m["t_success"], EVENT_NAME["success"]))
    return sorted(ev)


def draw_events(d, clip, t, xy, size=28):
    x, y = xy
    for te, name in events_of(clip):
        if te <= t:
            col = C_B if "成功" in name else (C_C if "注入" in name else (192, 57, 43))
            text(d, (x, y), f"{te:5.1f} s  {name}", size, col, True)
            y += size + 12


def legend(d, x, y):
    d.line([(x, y + 14), (x + 40, y + 14)], fill=(230, 126, 34), width=5)
    text(d, (x + 50, y), "推論遅延の間に過ぎる部分", 20, MUTED)
    d.line([(x, y + 50), (x + 40, y + 50)], fill=(0, 170, 200), width=4)
    text(d, (x + 50, y + 36), "アクションチャンクの予測経路", 20, MUTED)


# ------------------------------------------------------------------ 場面
def card(lines, dur, title=None):
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
    return int(dur * FPS), f


def single(cid, title, t0, t1, speed, note=None, extra=None):
    clip = Clip(cid)
    n = int((t1 - t0) / speed * FPS)

    def f(j):
        t = t0 + j * speed / FPS
        im = Image.new("RGB", (W, H), BG)
        draw_video(im, clip, t, (0, 86, 1440, 810))
        d = ImageDraw.Draw(im)
        header(d, title, speed, note or RERUN_NOTE)
        m = clip.meta["clip"]
        text(d, (1470, 110), MODEL_NAME[m["model"]], 26, INK, True)
        text(d, (1470, 150), MODE_NAME.get(m.get("mode", "naive"), ""), 24, MUTED)
        text(d, (1470, 190), f"時刻 {t:5.1f} s", 26, INK)
        draw_events(d, clip, t, (1470, 250), 24)
        draw_speed(d, clip, t, (40, 910, 1380, 120))
        legend(d, 1470, 930)
        if extra:
            extra(d, t)
        return im
    return n, f


def pair(cids, labels, title, t0, t1, speed, note=None, top=None):
    clips = [Clip(c) for c in cids]
    n = int((t1 - t0) / speed * FPS)

    def f(j):
        t = t0 + j * speed / FPS
        im = Image.new("RGB", (W, H), BG)
        for i, c in enumerate(clips):
            x = 0 if i == 0 else 960
            draw_video(im, c, t, (x, 130, 960, 540))
        d = ImageDraw.Draw(im)
        header(d, title, speed, note or RERUN_NOTE)
        if top:
            text(d, (40, 96), top, 26, C_C, True)
        for i, (c, lab) in enumerate(zip(clips, labels)):
            x = 0 if i == 0 else 960
            m = c.meta["clip"]
            text(d, (x + 30, 690), lab or MODEL_NAME[m["model"]], 28, INK, True)
            text(d, (x + 30, 730), f"{MODE_NAME.get(m.get('mode', 'naive'), '')}　時刻 {t:5.1f} s", 26, C_A, True)
            draw_events(d, c, t, (x + 30, 770), 22)
            draw_speed(d, c, t, (x + 30, 920, 900, 110))
        return im
    return n, f


def task_scene(cid, title, speed, v):
    clip = Clip(cid)
    tk = clip.meta["task"]
    names = {"red": "赤", "green": "緑", "blue": "青"}
    plan = [names[c] for c in tk["plan"]["steps"]]
    t1 = clip.meta["t_end"] + 1.0
    n = int(t1 / speed * FPS)

    def f(j):
        t = j * speed / FPS
        im = Image.new("RGB", (W, H), BG)
        draw_video(im, clip, t, (0, 86, 1440, 810), path=False)
        d = ImageDraw.Draw(im)
        header(d, title, speed, RERUN_NOTE + "。推論の間は物理を止めている（推論遅延はステップ数で模擬）")
        text(d, (1470, 110), MODEL_NAME["R2"], 26, INK, True)
        text(d, (1470, 160), f"指示「{tk['text']}」", 30, INK, True)
        text(d, (1470, 210), "LLM（Claude Haiku 4.5）の分解:", 22, MUTED)
        text(d, (1470, 244), " → ".join(plan), 32, C_A, True)
        y = 310
        for s in tk["steps"]:
            tj = s["attempts"][-1]["t_judge"]
            done = tj is not None and tj <= t
            active = s["attempts"][0]["t_begin"] <= t and not done
            mark = "✓" if done else ("▶" if active else "・")
            col = C_B if done else (INK if active else MUTED)
            text(d, (1470, y), f"{mark} {names[s['color']]}を箱へ", 32, col, True)
            y += 50
            if len(s["attempts"]) > 1 and s["attempts"][1]["t_begin"] <= t:
                text(d, (1510, y), f"再試行（{s['attempts'][1]['t_begin']:.0f} s〜）", 24, C_C)
                y += 38
        for r in tk["returns"]:
            if r["t_begin"] <= t <= r["t_end"] + 1.5:
                text(d, (1470, 620), "手を待機位置へ戻して再試行", 26, C_C, True)
        text(d, (1470, 680), f"時刻 {t:5.1f} s", 26, INK)
        draw_speed(d, clip, t, (40, 910, 1380, 120))
        return im
    return n, f


def fig1_image():
    """図 1 だけの HTML を Edge で画像にする（説明資料と同じ SVG・値）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("paper", ROOT / "scripts" / "60_paper.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
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


def image_scene(img, title, lines, dur):
    im0 = img.copy()
    im0.thumbnail((1500, 880))

    def f(j):
        im = Image.new("RGB", (W, H), BG)
        im.paste(im0, ((W - im0.width) // 2, 100))
        d = ImageDraw.Draw(im)
        header(d, title)
        k = min(len(lines), 1 + j // int(dur * FPS / max(1, len(lines))))
        y = H - 40 - 44 * len(lines)
        for s in lines[:k]:
            text(d, (W // 2, y), s, 30, INK, False, "ma")
            y += 44
        return im
    return int(dur * FPS), f


def scenes(v):
    fig1 = fig1_image()
    return [
        card([("Franka Panda（MuJoCo のシミュレーション）が、日本語の指示で 3 色の立方体を片付ける", 34, MUTED),
              ("掴み損ねなどの失敗から復帰する動きを、人手の収録なしで作った復帰デモで学習した", 34, INK)], 10,
             "失敗から復帰する VLA"),
        single("nat_110001_N1", "問題: 成功例だけで学習すると、掴み損ねた後に立て直せない", 3.0, 26.0, 1.5),
        image_scene(fig1, "仕組み: 三層の制御ループ",
                    ["上位層: LLM が指示をサブタスクに分け、カメラで完了を判定する",
                     f"VLA 層: SmolVLA が {v['policy_hz']} Hz でアクションチャンク（{v['chunk_s']} s 分）を生成し、推論遅延 {v['delay_s']} s を見込んで非同期に実行",
                     "低位制御: 安全フィルタで障害物への接近を削り、逆運動学で関節を動かす",
                     f"学習: 成功例 {v['n_normal']} 本に、失敗を注入してから立て直す復帰デモ {v['n_recovery']} 本を自動生成して足した"], 32),
        task_scene("task_115007", "実演: 日本語の指示 → LLM の分解 → 3 個を片付ける", 3.0, v),
        pair(["nat_110001_R1", "nat_110001_N1"], [None, None], "対比: 同じシード・失敗注入なしで起きた掴み損ね", 3.0, 17.5, 1.0,
             top="左は掴み直して箱へ入れ、右は立て直せない"),
        pair(["p1_111004_R1", "p1_111004_N1"], [None, None], "対比: 失敗を人為的に起こした場面（失敗注入 P1）", 2.0, 17.0, 1.5,
             top="閉じる直前に手先を横へずらして掴み損ねさせる（失敗注入）"),
        pair(["p1_111004_R1", "p1_111004_R1_rtc"], [None, None], "RTC のトレードオフ: 滑らかになるが復帰しなくなる（失敗注入 P1）", 2.0, 17.0, 1.5,
             top=f"チャンク境界での速度の不連続 {v['seam_naive']} → {v['seam_rtc']} m/s、P1 の復帰 {v['e4_x']} → {v['e4_y']}"),
        card([(f"復帰デモの効果（P1 からの復帰）: 非同期実行 {v['e3a_x']} 対 {v['e3a_y']}（{v['e3a_pairs']} 対）、同期実行 {v['e3s_x']} 対 {v['e3s_y']}（{v['e3s_pairs']} 対）", 32, INK),
              (f"失敗注入なしの成功: 最終モデル {v['e1_k']}/{v['e1_n']}（{v['e1_pct']}）", 32, INK),
              (f"安全フィルタ: 失敗注入なしで障害物に触れた試行 {v['e5_y']} → {v['e5_x']}（代償: 止め続けて失敗 {v['e5_blocked']} 本）", 32, INK),
              (f"目標位置のキュー: 指示した色に向かった {v['e6_k']}/{v['e6_n']}　LLM の分解: {v['llm_k']}/{v['llm_n']} 文", 32, INK),
              (f"3 個の連続タスク: {v['e7_k']}/{v['e7_n']}（再試行で完了したサブタスク {v['e7_retry']}）", 32, INK),
              ("どれもテスト用のシード範囲の値。R1＝復帰デモありで学習、N1＝なしで学習", 26, MUTED)], 22, "主な数字"),
        card([("シミュレーションだけ（実機では未検証。復帰デモの作り方は実機でも使える見込み）", 32, INK),
              (f"落下（P2）からの復帰は弱い（どの条件でも {v['p2_max']} 以下）。推論の高速化が次の課題", 32, INK),
              (f"2 段目の復帰デモ（R2）は、R1+ に対して P1 の復帰の差を確かめられなかった（{v['e8_x']} 対 {v['e8_y']}）", 32, INK),
              (f"3 個の連続タスクの失敗 {v['e7_fail']} 本のうち {v['e7_false']} 本は、立方体の上に乗せたのを「完了」と判定したもの", 32, INK)], 18,
             "限界と今後"),
    ]


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
    spec = importlib.util.spec_from_file_location("paper", ROOT / "scripts" / "60_paper.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    info = json.loads((BUILD / "video_texts.json").read_text(encoding="utf-8"))
    joined = "\n".join(info["texts"])
    hits = [(p, m.group(0)) for p in pm.FORBIDDEN for m in re.finditer(p, joined)]
    problems = [f"使わない語 {h[1]!r}（{h[0]}）" for h in hits]
    if not 60 <= info["seconds"] <= 300:
        problems.append(f"長さ {info['seconds']:.0f} s")
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
