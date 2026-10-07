"""説明資料（PDF。課題の 9/30 版はページ数を指定しない）を結果のファイルから作り、照合する。数字は手で書かない。段階 3 の版（0146・0147）。

    .venv\\Scripts\\python.exe scripts\\60_paper.py build     # 値の計算 → 図 2 → HTML → PDF（Edge のヘッドレス印刷）
    .venv\\Scripts\\python.exe scripts\\60_paper.py check     # 照合: 手で書いた数字がない・値が結果と一致・使わない語が 0・ページ数

原稿は paper/template.html と paper/fig1.svg。数字はすべて {{キー}} の差し込み口で、values() が段階 3 の最終評価の結果
（outputs/results/v2_e_report_s3.json、outputs/v2eval/V3S3 の E6・E7、テスト用のシード範囲 140000〜）と、設定・学習の記録・
凍結の一覧（docs/freeze/s3_hashes.json）・動画の場面の表（configs/demo/video_s3.yaml）から計算する。
目標書 v1 の版（9/29 提出用）は git のタグ stepJ-freeze の版にある。
出力: paper/build/（paper.html・図・values.json）と、paper/build/説明資料_PAI最終課題_<アカウント名>.pdf
"""
import argparse
import collections
import html
import json
import pathlib
import re
import subprocess
import sys

from recovla.common import config, goals

CFG = config.load_v2()
ROOT = config.ROOT
OUT = config.path(CFG["paths"]["outputs"])
RES = OUT / "results"
S3 = OUT / "v2eval" / "V3S3"
PAPER = ROOT / "paper"
BUILD = PAPER / "build"
VIDEO = ROOT / "configs" / "demo" / "video_s3.yaml"
FREEZE = ROOT / "docs" / "freeze" / "s3_hashes.json"
ACCOUNT = "アカウント名"                     # 提出の前に手で直す（ファイル名だけに入る）
PDF_NAME = f"説明資料_PAI最終課題_{ACCOUNT}.pdf"
ALPHA = "0.05"                               # 主要評価項目の基準（目標書 v3: Holm 補正の後で p < 0.05）
GPU = "NVIDIA GeForce RTX 4080 SUPER"        # 推論時間の分布（configs/latency_v1.json）を測った機器（nvidia-smi、10/07 に確認）
EXTRA = "extra_s3.json"                      # 査読への対応で足した参考の集計（scripts/55_extra_s3.py）
EDGE = pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
FONT = pathlib.Path(r"C:\Windows\Fonts\yumin.ttf")         # 図 2 の文字（本文と同じ游明朝、10/07 の指示）


def _j(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def _mod(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pct(x):
    return f"{round(100 * x)}%"


def ci(w):
    return f"{round(100 * w[0])}〜{round(100 * w[1])}%"


def dci(d):
    lo, hi = d["diff_95ci_newcombe"]
    return f"{round(100 * lo):+d}〜{round(100 * hi):+d}".replace("+0〜", "0〜")


def pv(p):
    if p < 0.001:
        return "< 0.001"
    return f"{p:.2g}" if p < 0.1 else f"{p:.2f}"


def pv_eq(p):
    """本文で「p {{…eq}}」と書く形: 「= 0.016」または「< 0.001」。"""
    s = pv(p)
    return s if s.startswith("<") else f"= {s}"


def mmss(s):
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def video_scenes() -> list:
    import yaml
    return yaml.safe_load(VIDEO.read_text(encoding="utf-8"))["scenes"]


def v3_dataset_dir() -> str:
    """復帰デモありの学習データ（段階 3 の 1 周目、凍結の一覧に入っているもの）。"""
    return _j(OUT / "f" / "data_v3.json")["datasets"]["R1"]["dataset"].replace("\\", "/")


def e7_summary() -> dict:
    """3 個の連続タスク（E7）の記録から: 3 個とも、再試行で完了したサブタスク、完了判定の偽陰性・偽陽性、止まった位置。"""
    runs = [_j(p) for p in sorted((S3 / "E7_R1v3").glob("run_00??.json"))]
    out = {"n": len(runs), "all_three": 0, "retry_completed": 0, "false_neg": 0, "false_pos": 0, "stopped_at": collections.Counter(),
           "judged": 0, "not_judged": 0, "stopped_all_three": 0, "audit_viol": 0}
    for r in runs:
        out["all_three"] += bool(r["all_three_in_box"])
        out["audit_viol"] += int(r["audit"]["g1"]["violations"]) + int(r["audit"]["g3"]["total_violations"])
        for s in r["steps"]:
            truth = r["truth_success_t"].get(s["color"]) is not None
            if len(s["attempts"]) > 1 and truth:
                out["retry_completed"] += 1
            out["judged" if s["judged_complete"] else "not_judged"] += 1
            if s["judged_complete"] and not truth:
                out["false_pos"] += 1
            if not s["judged_complete"] and r["final_in_box"].get(s["color"]):
                out["false_neg"] += 1
        if r.get("stopped"):
            out["stopped_at"][int(r["stopped"]["step"])] += 1
            out["stopped_all_three"] += bool(r["all_three_in_box"])      # 3 個とも入っていたが、最後の完了判定が出ずに止まった
    return out


def values() -> dict:
    ee = _mod("e_eval", ROOT / "scripts" / "50_e_eval.py")
    st3 = _mod("v2e", ROOT / "scripts" / "87_v2_e.py").STAGES["s3"]
    rep = _j(RES / "v2_e_report_s3.json")
    P, sec, sets = rep["primary"], rep["secondary"], rep["sets"]
    v = {}
    # 設定・学習の記録
    fps = int(CFG["convert"]["fps"])
    rows = int(st3["sets"]["A"][3])
    v["policy_hz"] = str(fps)
    v["exec_rows"] = str(rows)
    v["exec_s"] = f"{rows / fps:g}"
    lat = _j(ROOT / "configs" / "latency_v1.json")["kinds"]["policy"]
    v["lat_p50"], v["lat_p95"] = f"{lat['p50']:.2f}", f"{lat['p95']:.2f}"
    ck = _j(FREEZE)["checkpoints"]
    pcfg = _j(ROOT / ck["R1v3"] / "pretrained_model" / "config.json")
    v["chunk"] = str(pcfg["chunk_size"])
    info = _j(ROOT / v3_dataset_dir() / "meta" / "info.json")
    v["act_dim"] = str(info["features"]["action"]["shape"][0])
    an = info["features"]["action"]["names"]                     # dx, dy, dz, drx, dry, drz, gripper
    v["act_pos"] = str(sum(n in ("dx", "dy", "dz") for n in an))
    v["act_rot"] = str(sum(n.startswith("dr") for n in an))
    v["act_grip"] = str(sum(n == "gripper" for n in an))
    sd = info["features"]["observation.state"]["shape"][0]
    cue = sum(1 for n in info["features"]["observation.state"]["names"] if n.startswith("cue_"))
    v["state_dim"], v["cue_dim"], v["state_base"] = str(sd), str(cue), str(sd - cue)
    sn = info["features"]["observation.state"]["names"]                # 状態の内訳（関節・手先の位置・姿勢のずれ・指）
    v["st_joint"] = str(sum(n.startswith("joint") for n in sn))
    v["st_pos"] = str(sum(n in ("eef_x", "eef_y", "eef_z") for n in sn))
    v["st_rot"] = str(sum(n.startswith("eef_rot") for n in sn))
    v["st_finger"] = str(sum(n.startswith("finger") for n in sn))
    v["cam_hz"] = f"{CFG['sensor']['frame_hz']:g}"
    v["phys_ms"] = f"{float(CFG['sim']['timestep']) * 1000:g}"
    v["train_steps"] = f"{int(pathlib.Path(ck['R1v3']).name):,}"
    v3 = _j(OUT / "f" / "data_v3.json")["datasets"]
    cnt = collections.Counter(e["key"].split("_")[0] for e in _j(ROOT / v3["R1"]["manifest"])["entries"])
    v["n_normal"] = str(cnt["n"])
    v["n_recovery"] = str(sum(c for k, c in cnt.items() if k != "n"))
    v["n_n1"] = str(len(_j(ROOT / v3["N1"]["manifest"])["entries"]))
    v["n_rec_grasp"], v["n_rec_drop"], v["n_rec_place"] = str(cnt["A"]), str(cnt["B"]), str(cnt["C"])   # 復帰デモの種類（A 把持失敗・B 落下・C 置き損ね）
    ia = CFG["inject"]["A"]
    v["inj_lat_lo_cm"], v["inj_lat_hi_cm"] = (f"{x * 100:g}" for x in ia["lateral_offset_m"])
    v["inj_raise_lo_cm"], v["inj_raise_hi_cm"] = (f"{x * 100:g}" for x in ia["raise_close_m"])
    sc = CFG["eval"]["success"]
    v["succ_speed_cms"], v["succ_hold_s"] = f"{float(sc['rest_speed']) * 100:g}", f"{float(sc['rest_hold_s']):g}"
    tcfg = _j(ROOT / ck["R1v3"] / "pretrained_model" / "train_config.json")          # 初期値（Hugging Face のキャッシュの名前から）
    m_init = re.search(r"models--([^\\/]+?)--([^\\/]+)", tcfg["policy"]["pretrained_path"])
    v["init_model"] = f"{m_init.group(1)}/{m_init.group(2)}"
    tcfg_n = _j(ROOT / ck["N1v3"] / "pretrained_model" / "train_config.json")
    for key in ("batch_size", "steps", "seed"):                                        # 表 2 は両モデル共通の値だけを書く
        assert tcfg[key] == tcfg_n[key], key
    assert tcfg["optimizer"]["lr"] == tcfg_n["optimizer"]["lr"]
    v["batch"], v["lr"] = str(tcfg["batch_size"]), f"{tcfg['optimizer']['lr']:g}"
    v["opt"] = tcfg["optimizer"]["type"].replace("adamw", "AdamW")
    v["warmup"] = f"{tcfg['scheduler']['num_warmup_steps']:,}"
    v["gpu"] = GPU
    info_n = _j(ROOT / v3["N1"]["dataset"] / "meta" / "info.json")
    v["frames_r"], v["frames_n"] = f"{info['total_frames']:,}", f"{info_n['total_frames']:,}"
    v["eps_r"], v["eps_n"] = str(info["total_episodes"]), str(info_n["total_episodes"])
    succ_r = {e["key"] for e in _j(ROOT / v3["R1"]["manifest"])["entries"] if e["key"].startswith("n_")}
    succ_n = {e["key"] for e in _j(ROOT / v3["N1"]["manifest"])["entries"]}
    assert succ_r <= succ_n                                                            # ありの成功デモは、なしの成功デモの一部
    ts = float(CFG["sim"]["timestep"])
    v["phys_hz"] = f"{1 / ts:g}"
    sf = CFG["safety_filter"]
    extra = float(CFG["runtime_v2"]["safety_extra_margin_m"])     # 実行系の安全フィルタは最小距離と対象の距離の両方に足す
    v["sf_dmin_mm"] = f"{sf['d_min_m'] * 1000:g}"
    v["sf_extra_mm"] = f"{extra * 1000:.1f}"
    v["sf_dmin_tot_mm"] = f"{(sf['d_min_m'] + extra) * 1000:.1f}"
    v["sf_detect_tot_mm"] = f"{(sf['d_detect_m'] + extra) * 1000:.1f}"
    sfd = _j(RES / "v2_safety_main_s3_V3SF.json")                  # 検証用の確定判定（規則と、検証用での成功率の低下）
    v["sf_rule_pt"] = re.search(r"(\d+(?:\.\d+)?) ポイント", sfd["rule"]).group(1)
    v["sf_val_drop"] = f"{round(sfd['drop_points'])}"
    v["alpha"] = ALPHA
    v["offset_gain_mm"] = re.search(r"OFFSET_GAIN_MM = ([\d.]+)", (ROOT / "scripts" / "94_s3_round1.py").read_text(encoding="utf-8")).group(1).rstrip("0").rstrip(".")
    lo, hi = CFG["eval"]["P1"]["lateral_offset_m"]
    v["p1_lo_cm"], v["p1_hi_cm"] = f"{lo * 100:g}", f"{hi * 100:g}"
    v["p1_win_s"], v["p1_lift_cm"] = f"{float(CFG['eval']['P1']['confirm_window_s']):g}", f"{float(CFG['eval']['P1']['confirm_lift_m']) * 100:g}"
    v["sf_dmin_src_mm"] = re.search(r"1 パーセンタイル ([\d.]+) mm", (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")).group(1)
    v["sf_detect_mm"] = f"{sf['d_detect_m'] * 1000:g}"
    v["judge_hold_s"] = f"{float(CFG['runtime_v2']['judge']['hold_s']):g}"
    v["time_limit"] = f"{float(CFG['eval']['time_limit_s']):g}"
    # テスト用の試行の数
    v["nat_n"] = str(P["E1"]["n"])
    v["nat_layouts"] = str(P["E1"]["n"] // 3)
    v["p_n"] = str(sets["A"]["P1"]["n"])
    # 主要評価項目
    v["e1_k"], v["e1_n"] = str(P["E1"]["successes"]), str(P["E1"]["n"])
    v["e1_pct"], v["e1_ci"] = pct(P["E1"]["successes"] / P["E1"]["n"]), ci(P["E1"]["success_wilson"])
    v["e2_k"], v["e2_n"] = str(P["E2"]["recovered"]), str(P["E2"]["established"])
    v["e2_pct"], v["e2_ci"] = pct(P["E2"]["recovery_rate"]), ci(P["E2"]["recovery_wilson"])

    def pair(prefix, d, holm=None):
        v[f"{prefix}_x"], v[f"{prefix}_y"] = pct(d["x_rate"]), pct(d["y_rate"])
        v[f"{prefix}_xk"], v[f"{prefix}_yk"] = str(d["x_only"] + d["both"]), str(d["y_only"] + d["both"])
        v[f"{prefix}_xo"], v[f"{prefix}_yo"] = str(d["x_only"]), str(d["y_only"])
        v[f"{prefix}_pairs"], v[f"{prefix}_ci"] = str(d["pairs"]), dci(d)
        v[f"{prefix}_p"], v[f"{prefix}_peq"] = pv(d["mcnemar_exact_p"]), pv_eq(d["mcnemar_exact_p"])
        if holm is not None:
            v[f"{prefix}_holm"], v[f"{prefix}_holmeq"] = pv(holm), pv_eq(holm)
    pair("e3a", P["E3"]["main_A_vs_B"], P["E3"]["main_A_vs_B"]["holm_p"])
    pair("e3s", P["E3"]["sync_C_vs_D"], P["E3"]["sync_C_vs_D"]["holm_p"])
    pair("nat_rn", sec["E3_main"]["nat_success"])
    pair("e4", sec["E4_naive_vs_rtc"]["P1_recovery"])
    pair("nat_nr", sec["E4_naive_vs_rtc"]["nat_success"])
    pair("con_nr", sec["E4_naive_vs_rtc"]["nat_contact"])
    jr = sec["E4_naive_vs_rtc"]["jerk_rms_all"]
    v["jerk_naive"], v["jerk_rtc"], v["jerk_pairs"] = f"{jr['x_median']:.1f}", f"{jr['y_median']:.1f}", str(jr["pairs"])
    pair("e5", sec["E5"]["nat_contact"])           # 本線（フィルタなし）対 フィルタあり。x が本線
    pair("e5s", sec["E5"]["nat_success"])
    for part in ("P2", "P3"):
        s = sets["A"][part]
        v[f"a_{part.lower()}_k"], v[f"a_{part.lower()}_n"] = str(s["recovered"]), str(s["established"])
        v[f"a_{part.lower()}_ci"] = ci(s["recovery_wilson"])
    pair("e3p2", sec["E3_main"]["P2_recovery"])                # 落下・置き損ねの復帰（復帰デモあり 対 なし、両モデルとも実際に失敗した組）
    pair("e3p3", sec["E3_main"]["P3_recovery"])
    v["b_p1_est"] = str(sets["B"]["P1"]["established"])
    v["e2_n_trials"] = str(sets["A"]["P1"]["n"])
    # 査読への対応で足した参考の集計（結果を見た後に加えたもの）
    ex = _j(RES / EXTRA)
    pa = ex["p1_all"]
    v["p1all_n"], v["p1all_xk"], v["p1all_yk"] = str(pa["pairs"]), str(pa["x_k"]), str(pa["y_k"])
    v["p1all_peq"] = pv_eq(pa["mcnemar_exact_p"])
    ng = ex["natural_grasp_failure"]
    v["nat_gf_xk"], v["nat_gf_xn"] = str(ng["A"]["recovered"]), str(ng["A"]["grasp_failed"])
    v["nat_gf_yk"], v["nat_gf_yn"] = str(ng["B"]["recovered"]), str(ng["B"]["grasp_failed"])
    v["nat_gf_trials"] = str(ng["A"]["trials"])
    # E6（目標位置のキュー）
    e6, e6c = _j(S3 / "e6_R1v3_both.json"), _j(S3 / "e6_R1v3_cue_only.json")
    k6 = round(e6["accuracy_majority"] * e6["pairs"])
    v["e6_k"], v["e6_n"], v["e6_ci"] = str(k6), str(e6["pairs"]), ci(ee._wilson(k6, e6["pairs"]))
    v["e6c_cue"] = str(round(e6c["accuracy_majority"] * e6c["pairs"]))
    v["e6c_text"] = str(round(e6c["follows_instruction_text_majority"] * e6c["pairs"]))
    v["e6c_n"] = str(e6c["pairs"])
    v["e6_layouts"] = e6["trials"].split(":")[1]
    v["e6_samples"] = str(sum(e6["rows"][0]["votes"].values()))         # 1 通りあたりの生成の回数
    v["e6c_text_ci"] = ci(ee._wilson(round(e6c["follows_instruction_text_majority"] * e6c["pairs"]), e6c["pairs"]))
    # E7（3 個の連続タスク）
    s7 = e7_summary()
    v["e7_k"], v["e7_n"], v["e7_ci"] = str(s7["all_three"]), str(s7["n"]), ci(ee._wilson(s7["all_three"], s7["n"]))
    v["e7_retry"], v["e7_fn"], v["e7_fp"] = str(s7["retry_completed"]), str(s7["false_neg"]), str(s7["false_pos"])
    top_step, top_n = s7["stopped_at"].most_common(1)[0]
    v["e7_stop2"], v["e7_stop_pos"] = str(top_n), str(top_step + 1)
    v["e7_stopped"] = str(sum(s7["stopped_at"].values()))
    v["e7_judged"], v["e7_notjudged"] = str(s7["judged"]), str(s7["not_judged"])
    v["e7_stop_all3"] = str(s7["stopped_all_three"])
    v["e7_sys"] = str(s7["n"] - sum(s7["stopped_at"].values()))        # 停止せずに全サブタスクの完了判定が出た試行
    v["e7_sys_ci"] = ci(ee._wilson(s7["n"] - sum(s7["stopped_at"].values()), s7["n"]))
    v["e7_viol"] = str(s7["audit_viol"])
    # E9（G1〜G3 の監査）・E10（知覚の精度）
    e9 = rep["new"]["E9"]["total"]
    v["e9_trials"] = f"{e9['trials']:,}"
    v["e9_viol"] = str(e9["g1"] + e9["g3"])           # 本文に書くのは真の状態の参照と指令の上限超過（10/07）
    byc = rep["new"]["E9"]["by_condition"]            # 構成 A〜E は各 nat＋P1〜P3、H（安全フィルタあり）は nat のみ
    cfgs = sorted({k.split("_")[0] for k in byc if not k.startswith("H")})
    per = {c: sum(r["trials"] for k, r in byc.items() if k.split("_")[0] == c) for c in cfgs}
    assert len(set(per.values())) == 1, per
    v["e9_cfg_n"], v["e9_per_cfg"] = str(len(cfgs)), str(next(iter(per.values())))
    v["e9_h"] = str(sum(r["trials"] for k, r in byc.items() if k.startswith("H")))
    pa = rep["new"]["E10"]["A"]
    v["perc_med_mm"], v["perc_p95_mm"] = f"{pa['median_m'] * 1000:.0f}", f"{pa['p95_m'] * 1000:.0f}"
    # 動画の場面の時刻（本文の「動画 m:ss〜m:ss」）
    t = 0.0
    span = {}
    for sc in video_scenes():
        span[sc["id"]] = (t, t + sc["dur_s"])
        v[f"v_{sc['id']}"] = f"{mmss(t)}〜{mmss(t + sc['dur_s'])}"
        t += sc["dur_s"]
    if "task_intro" in span:                    # 実演の条件の説明から実演の終わりまでを、本文では一続きの場面とする
        v["v_task"] = f"{mmss(span['task_intro'][0])}〜{mmss(span['task'][1])}"
    v["v_total"] = mmss(t)
    # 動画の実演の条件（上位層の実行器の設定）
    v["step_timeout"] = f"{float(CFG['planner']['step_timeout_s']):g}"
    v["retry"] = str(int(CFG["planner"]["retry"]))
    return v


def fig2(v) -> bool:
    """図 2: 同じ配置での復帰デモあり（上段）となし（下段）。configs/demo/video_s3.yaml の fig2 に場面と時刻を書く。"""
    import yaml
    spec = yaml.safe_load(VIDEO.read_text(encoding="utf-8")).get("fig2")
    if not spec:
        return False
    import av
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(FONT), 30)                     # 図の中の文字は 10/07 の指示で 1 pt 大きく（26→30 px）
    rows = []
    for key, label in (("with", "復帰デモ\nあり"), ("without", "復帰デモ\nなし")):
        c = av.open(str(OUT / "demo_v2" / spec[key]["clip"]))
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        c.close()
        tiles = []
        for tt in spec["times"]:
            img = Image.fromarray(frames[min(int(round(tt * 30)), len(frames) - 1)][120:600, 250:1090]).resize((420, 240))
            dr = ImageDraw.Draw(img)
            dr.rectangle((4, 4, 116, 42), fill=(255, 255, 255))
            dr.text((10, 6), f"{tt:.1f} s", fill=(20, 20, 20), font=font)
            tiles.append(img)
        row = Image.new("RGB", (420 * len(tiles) + 200, 240), (255, 255, 255))
        ImageDraw.Draw(row).multiline_text((8, 80), label, fill=(20, 20, 20), font=ImageFont.truetype(str(FONT), 31), spacing=8)
        for i, im in enumerate(tiles):
            row.paste(im, (200 + 420 * i, 0))
        rows.append(row)
    out = Image.new("RGB", (rows[0].width, 240 * 2 + 8), (255, 255, 255))
    out.paste(rows[0], (0, 0))
    out.paste(rows[1], (0, 248))
    out.save(BUILD / "fig2.png")
    return True


PIE_KINDS = [("n", "成功デモ", "#2a78d6"), ("A", "復帰デモ（把持失敗）", "#eb6834"),
             ("B", "復帰デモ（落下）", "#1baf7a"), ("C", "復帰デモ（置き損ね）", "#eda100")]   # 色は dataviz の既定の順（隣り合う組で検証済み）


def data_mix() -> dict:
    """復帰デモありの学習データの種類ごとの本数とフレーム数（変換の記録 meta/conversion.json の sources と episodes の長さ）。"""
    import pyarrow.parquet as pq
    root = ROOT / v3_dataset_dir()
    ep = pq.read_table(sorted((root / "meta" / "episodes").rglob("*.parquet"))[0], columns=["episode_index", "length"]).to_pydict()
    length = dict(zip(ep["episode_index"], ep["length"]))
    n, f = collections.Counter(), collections.Counter()
    for s in _j(root / "meta" / "conversion.json")["sources"]:
        n[s["kind"]] += 1
        f[s["kind"]] += length[s["episode_index"]]
    return {"episodes": dict(n), "frames": dict(f)}


def pie_svg() -> str:
    """図 3: 学習データの内訳の円グラフ（左が本数、右がフレーム数）。扇形の間に白い隙間、名前と割合は凡例と扇形の外に文字で書く。"""
    import math
    mix = data_mix()
    out = ['<svg viewBox="0 0 1080 360" role="img" aria-label="復帰デモありの学習データの内訳。本数とフレーム数の円グラフ。">']
    for cx, key, title in ((200, "episodes", "本数"), (560, "frames", "フレーム数")):
        tot = sum(mix[key].values())
        out.append(f'<text class="bt" x="{cx}" y="30" text-anchor="middle">{title}（計 {tot:,}）</text>')
        a0 = -math.pi / 2
        r, cy = 125, 190
        for k, _, col in PIE_KINDS:
            val = mix[key].get(k, 0)
            a1 = a0 + 2 * math.pi * val / tot
            x0, y0, x1, y1 = cx + r * math.cos(a0), cy + r * math.sin(a0), cx + r * math.cos(a1), cy + r * math.sin(a1)
            large = 1 if a1 - a0 > math.pi else 0
            out.append(f'<path d="M{cx},{cy} L{x0:.1f},{y0:.1f} A{r},{r} 0 {large} 1 {x1:.1f},{y1:.1f} z" '
                       f'style="fill:{col};stroke:#fff;stroke-width:2"/>')
            am = (a0 + a1) / 2
            lx, ly = cx + (r + 26) * math.cos(am), cy + (r + 26) * math.sin(am)
            anchor = "start" if math.cos(am) > 0.2 else ("end" if math.cos(am) < -0.2 else "middle")
            out.append(f'<text class="t" x="{lx:.1f}" y="{ly + 5:.1f}" text-anchor="{anchor}">{round(100 * val / tot)}%</text>')
            a0 = a1
    y = 120
    for k, name, col in PIE_KINDS:
        e, fr = mix["episodes"].get(k, 0), mix["frames"].get(k, 0)
        out.append(f'<rect x="790" y="{y - 14}" width="18" height="18" rx="3" style="fill:{col}"/>')
        out.append(f'<text class="t" x="818" y="{y}">{name}</text>')
        out.append(f'<text class="tm" x="818" y="{y + 22}">{e} 本・{fr:,} フレーム</text>')
        y += 58
    out.append("</svg>")
    return "\n".join(out)


def render(v) -> str:
    def sub(text):
        def rep(m):
            k = m.group(1)
            if k not in v:
                raise KeyError(k)
            return f'<span class="num" data-k="{k}">{html.escape(v[k])}</span>'
        return re.sub(r"\{\{(\w+)\}\}", rep, text)
    def svg(name):
        return sub((PAPER / name).read_text(encoding="utf-8")).replace('<span class="num" data-k=', '<tspan data-k=').replace("</span>", "</tspan>")
    figs = {"fig1_svg": lambda: svg("fig1.svg"), "fig_rec_svg": lambda: svg("fig_recovery.svg"),   # 図 1（構成）・図 2（作り方）
            "fig_pie_svg": pie_svg}                                                                   # 図 3（学習データの内訳）
    body = (PAPER / "template.html").read_text(encoding="utf-8")
    for key in figs:
        body = body.replace("{{" + key + "}}", f"@@{key}@@")
    body = sub(body)
    for key, make in figs.items():
        body = body.replace(f"@@{key}@@", make())
    return body


def goals_or_exit() -> dict:
    """変更履歴にない版の目標書なら作らない・照合しない（0106）。"""
    try:
        return goals.fingerprint()
    except goals.GoalsMismatch as e:
        print(f"目標書の照合に通らないので止める: {e}", file=sys.stderr)
        raise SystemExit(1)


def cmd_build(a) -> None:
    fp = goals_or_exit()
    BUILD.mkdir(parents=True, exist_ok=True)
    for old in ("fig3.png",):                       # 9/29 版の図（段階 3 では使わない）
        (BUILD / old).unlink(missing_ok=True)
    (BUILD / "goals.json").write_text(json.dumps(fp, ensure_ascii=False, indent=1), encoding="utf-8")
    v = values()
    (BUILD / "values.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    if not fig2(v):
        (BUILD / "fig2.png").unlink(missing_ok=True)
        print("図 2 の場面が未設定（configs/demo/video_s3.yaml の fig2）", file=sys.stderr)
    (BUILD / "paper.html").write_text(render(v), encoding="utf-8")
    pdf = BUILD / "paper.pdf"
    subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    (BUILD / "paper.html").as_uri()], check=True, capture_output=True, timeout=180)
    (BUILD / PDF_NAME).write_bytes(pdf.read_bytes())
    print(BUILD / PDF_NAME, "pages", pdf_pages(pdf))


def pdf_pages(p) -> int:
    return len(re.findall(rb"/Type\s*/Page(?!s)", pathlib.Path(p).read_bytes()))


# 原稿の地の文に書いてよい数字（数字が値ではなく名前や固有の決まりの一部であるもの）
ALLOWED = [r"fig2\.png", r"表 [12]",r"(95|1) パーセンタイル",r"two3", r"Physical AI 応用 1 講座", r"Haiku 4\.5", r"SmolVLM2", r"Apache-2\.0", r"図 [1-4]",
           r"[12] 回目", r"工夫 [1-6]", r"\b[1-6]\.[1-6](?= )",r"3 (色|個)", r"95% 信頼区間", r"1 試行", r"[1-6]\. ", r"7 軸", r"画像 2 枚", r"1 行ずつ",
           r"× 3 色", r"A4", r"RGB-D", r"#[0-9a-f]{6}", r"\d+(\.\d+)?(mm|pt|px)", r"viewBox=\"[^\"]*\"", r"\b\d+(\.\d+)?%?\"",
           r"[xy][12]?=\"[^\"]*\"", r"points=\"[^\"]*\"", r"d=\"[^\"]*\"", r"rotate\([^)]*\)", r"\b(width|height|rx|dx|refX|refY|markerWidth|markerHeight)=\"[^\"]*\"",
           r"opacity:\.\d+", r"stroke-dasharray:[\d ]+", r"stroke-width:[\d.]+", r"font-size:[\d.]+(px|pt)", r"line-height:[\d.]+",
           r"\d+(\.\d+)?(fr|mm)", r"gap: \d+mm", r"margin[^;]*;", r"padding[^;]*;", r"h[123]", r"m1[co]", r"M0,0 L10,5 L0,10 z",
           r"lang=\"ja\"", r"utf-8", r"-webkit", r"0 10 10", r"font-weight: ?\d+", r"border[^;]*;", r"\bgrid-template-columns[^;]*;"]
# 使わない語（開発中に作った語・作業記録・卒研に触れる語）。「手」は手先・手首などは使うので、数と組になった形だけ
# 語の一覧は Unicode エスケープで書く（書き出しの置き換えと語の検索が、この一覧そのものに当たらないように）
FORBIDDEN = ["\u584a", "\u3053\u307e", "\u7a2e(?!\u985e)", "\u5e2f", "\u53f0\u672c", "\u8a98\u767a", "\u6210\u7acb",
             "\u7acb\u3061\u76f4", "\u624b\u304c\u304b\u308a", "\u7d99\u304e\u76ee", "\u9045\u308c",
             "\u6d41\u308c\u306e\u4e00\u81f4", "\u4fdd\u5b58\u70b9", "\u81ea\u7136", "\u901a\u3057",
             "\u4e3b\u306a\u691c\u5b9a", "\u526f\u306e\u6307\u6a19", "(?<!\u4fe1\u983c)\u533a\u9593",
             "[\u7532\u4e59\u4e19]", "\u672c\u7dda", "\u652f\u7dda", "\u76e3\u7763", "\u6c7a\u88c1",
             "\u63b2\u793a\u677f", "\u6d41\u7528\u5143", "\u5352\u7814", "\u5352\u696d\u7814\u7a76", "C:\\\\VLA",
             "\u5b66\u751f", "\u4e88\u5099\u5b9f\u9a13", "(\\d|\u4e07)\\s*\u624b(?![\u5148\u9996\u9806\u6cd5])",
             "\u6a21\u578b",
             # 開発中の記号（決裁の原則 10/07: 本文でも使わない。0147 の 2）
             "(?<![A-Za-z0-9])[RNP][123](?![0-9])", "R1\\+", "[RN][12]v[23]", "(?<![A-Za-z])E(?:[1-9]|10)(?![0-9])", "(?<![A-Za-z])G[1-6](?![0-9])",
             # 10/07 の指示: 失敗を起こす操作は「意図的な失敗」と書く。物理演算の扱いは開発の過程のことなので書かない
             "\u6ce8\u5165", "\u7269\u7406\u6f14\u7b97.{0,6}(\u505c\u6b62|\u6b62\u307e|\u6b62\u3081)",
             "(\u505c\u6b62|\u6b62\u307e|\u6b62\u3081).{0,6}\u7269\u7406\u6f14\u7b97",
             # 10/07 \u306e\u6307\u793a\uff08\u8a9e\u306e\u7f6e\u304d\u63db\u3048\uff09: \u5bfe\u2192\u7d44\u3001\u8a2d\u7f6e\u60c5\u5831\u2192\u65e2\u77e5\u306e\u60c5\u5831\u3001\u30ad\u30e5\u30fc\u2192\u76ee\u6a19\u4f4d\u7f6e\u306e\u5165\u529b\u3001\u7279\u6a29\u60c5\u5831\u3001\u504f\u4f4d\u3001\u89e3\u653e\u3001\u30d3\u30c3\u30c8\u5358\u4f4d\u3001\u4ee5\u524d\u306e\u7248
             "\\d\\s*\u5bfe(?!\u5fdc)", "\u5bfe[\u306f\u304c\u3092]", "\u8a2d\u7f6e\u60c5\u5831", "\u30ad\u30e5\u30fc", "\u7279\u6a29\u60c5\u5831",
             "\u504f\u4f4d", "\u89e3\u653e", "\u30d3\u30c3\u30c8\u5358\u4f4d", "\u4ee5\u524d\u306e\u7248"]


def cmd_check(a) -> None:
    fp = goals_or_exit()
    v = values()
    problems = []
    built_goals = BUILD / "goals.json"
    if not built_goals.is_file() or _j(built_goals) != fp:
        problems.append("組み上げたときの目標書の版が今の版と違う（build をやり直す）")
    if not (BUILD / "fig2.png").is_file():
        problems.append("図 2 が未作成（configs/demo/video_s3.yaml の fig2）")
    # (1) 原稿に手で書いた数字がない
    for name in ("template.html", "fig1.svg", "fig_recovery.svg"):
        text = (PAPER / name).read_text(encoding="utf-8")
        text = re.sub(r"\{\{\w+\}\}", " ", text)
        text = re.sub(r"<style>.*?</style>", " ", text, flags=re.S)
        for pat in ALLOWED:
            text = re.sub(pat, " ", text)
        for m in re.finditer(r"\d", text):
            problems.append(f"{name}: 手で書いた数字 …{text[max(0, m.start() - 15):m.end() + 15]!r}")
    # (2) 組み上げた HTML の差し込み口の値が、今の結果から計算した値と一致
    built = (BUILD / "paper.html").read_text(encoding="utf-8")
    n_slots = 0
    for m in re.finditer(r'data-k="(\w+)">([^<]*)<', built):
        n_slots += 1
        if html.unescape(m.group(2)) != v[m.group(1)]:
            problems.append(f"値の食い違い {m.group(1)}: {m.group(2)} != {v[m.group(1)]}")
    # (3) 使わない語が 0
    plain = re.sub(r"<style>.*?</style>", " ", built, flags=re.S)
    plain = html.unescape(re.sub(r"<[^>]+>", " ", plain))
    hits = [(p, mm.group(0), plain[max(0, mm.start() - 12):mm.end() + 12]) for p in FORBIDDEN for mm in re.finditer(p, plain)]
    problems += [f"使わない語 {h[1]!r}: …{h[2]}…" for h in hits]
    # (4) ページ数
    pages = pdf_pages(BUILD / "paper.pdf")
    if pages < 2:                                  # 課題の 9/30 版はページ数を指定しない（9/17 版の 2〜4 は外した、10/07）
        problems.append(f"ページ数 {pages}")
    res = {"goals": fp, "slots": n_slots, "keys": len(v), "pages": pages, "forbidden_hits": len(hits), "problems": problems}
    (BUILD / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
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
