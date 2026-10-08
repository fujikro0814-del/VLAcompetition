"""段階 4 束 4: 関門 D（学習の前の検査。configs/s4_gates.json の bundle4_gates.D の 5 項目）を機械で確かめ、JSON に出す。
1 つでも満たさなければ学習に進まず、研究の道へ（D.stop_branch）。

使い方（作業場所 C:\\PAI\\recovery_vla）:
    .venv\\Scripts\\python.exe scripts\\97_s4_b4_gateD.py check [--eval-dir outputs\\v2eval\\V3S3\\A_P2] [--images-ok] [--regen]
        → outputs\\s4\\b4\\gateD.json（項目ごとの pass・fail・対象外・要確認と、全体の結論）と outputs\\s4\\b4\\gateD_images\\（目視用の 10 枚）
    --regen は d4 の回し直し（5 本をシミュレーションで描画ありで作り直す。運用役だけ）。付けなければ d4 は「要確認」。
    --images-ok は、d3 の 10 枚を作者が目で見て問題がないと確かめた後にだけ付ける（付けなければ d3 は「要確認」）。

5 項目（s4_gates.json の文と、この道具の当てはめ。結果を見る前に決めた。docs\\stage4\\bundle4_protocol.md 第 3 節）:
  d1 E7 の 2 番目の始めの状態ベクトルの 80% 以上が、新しいデータの中で最近傍距離 0.28〜0.33 以内
     → 対象外。開始状態の要因（束 4 の候補が prior_cube のとき）の項目で、滑りのデータは開始状態を変えない（gate1_sheet.md の結論）。
  d2 落下の直後の状態の 50% 以上が 3 cm 以内（今は 38 中 36 が 3 cm 超）。滑りの要因を扱うときだけ
     → 当てはめ: 評価（段階 3 の落下の誘発 V3S3\\A_P2。誘発が成立した試行）の「成立の時刻の目標の立方体の水平位置」と、
       学習のデータ（R4 の落下の復帰デモ＝種類 B の、記録の最初のこま＝誘発が確定した直後）の目標の立方体の水平位置の
       最近傍距離。3 cm 以内の割合 >= 0.50 で pass。
     → 物差しの照合: 同じ計算を R1v3 の落下の復帰デモ（B 30 本）に当てて「38 中 36 が 3 cm 超」を再現できなければ、物差しが
       元の計算（final.md、非公開）と違うので「要確認」にして止める（--baseline-want 36/38 で変えられる）。
  d3 格子の外の先客を描いた画像 10 枚を目で確かめる
     → 滑りの版: 新しい復帰デモ（B・C）の記録の最初のこま（俯瞰）10 枚を書き出す。作者が見て --images-ok を付けるまで「要確認」。
  d4 層 (i)（方策を通さない部分）の回し直し 5 本がビット一致
     → 新しい復帰デモ 5 本（B 3・C 2、data.json の chosen の先頭から）を、元の生成と同じ並びで作り直す: 本ごとに新しい世界で、
       同じ働き手が前に回した指定（generation.jsonl の同じ worker_pid の前の行）を作り直しを含めて順に回してから、その本の指定を
       作り直しを含めて回す（世界の使い回しで始めの指の開きが変わるため。docs\\stage4\\ops.md 5-1）。同じ回の試みが保存され、
       data.npz の全部の配列（画像は含まない）が型・形・バイト列まで一致すること。
  d5 R と N で通常デモの集合が同じ → R4・N4 のマニフェストのファイルを読み直し、97_s4_b4_manifest.py の reverify（build の
     検査の全項目＋SHA-256＋R1v3・N1v3 と data.json から組み立て直したものとの一致）。manifest.json の記録は信じない。
読むもの: outputs\\s4\\b4\\data.json・manifest.json、R4・N4・R1v3 のマニフェスト、エピソードの meta.json・data.npz・overhead\\000000.png、
  --eval-dir の trial_*.json・.npz。書くもの: outputs\\s4\\b4\\gateD.json・gateD_images\\・gateD_regen\\（--regen）。
"""
import argparse
import json
import pathlib
import shutil
import sys
import time

import numpy as np

from recovla.common import config
from recovla.common.seeds import COLORS

ROOT = config.ROOT
CFG = config.load()
OUTPUTS = config.path(CFG["paths"]["outputs"])
B4 = OUTPUTS / "s4" / "b4"
NEAR_M = 0.03
COVER_MIN = 0.50
N_IMAGES = 10
N_REGEN = {"B": 3, "C": 2}
DEFAULT_EVAL = "outputs/v2eval/V3S3/A_P2"
BASELINE_WANT = (36, 38)                      # s4_gates の文「今は 38 中 36 が 3 cm 超」


# ---------------------------------------------------------------- d2 落下の直後の状態の覆い
def eval_drop_states(eval_dir: pathlib.Path) -> list:
    """評価の落下の試行のうち誘発が成立したものの、成立の時刻のこまの目標の立方体の水平位置 [(x, y)]。"""
    out = []
    for p in sorted(eval_dir.glob("trial_[0-9][0-9][0-9][0-9].json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        ind = m.get("induce") or {}
        if not ind.get("established") or ind.get("t_established") is None:
            continue
        z = np.load(p.with_suffix(".npz"))
        t = np.asarray(z["sim_time"], float)
        f = int(min(np.searchsorted(t, float(ind["t_established"]) - 1e-9), len(t) - 1))
        ci = COLORS.index(m["target"])
        out.append(tuple(float(v) for v in np.asarray(z["cube_pos"])[f, ci, :2]))
    return out


def train_drop_states(entries: list) -> list:
    """マニフェストの落下の復帰デモ（種類 B）の、記録の最初のこまの目標の立方体の水平位置。"""
    out = []
    for e in entries:
        if not e["key"].startswith("B_"):
            continue
        d = config.path(e["run"]) / e["key"]
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        z = np.load(d / "data.npz")
        ci = COLORS.index(meta["target"])
        out.append(tuple(float(v) for v in np.asarray(z["cube_pos"])[0, ci, :2]))
    return out


def coverage(eval_xy: list, train_xy: list, near: float = NEAR_M) -> dict:
    if not eval_xy or not train_xy:
        return {"n_eval": len(eval_xy), "n_train": len(train_xy), "within": 0, "frac": None, "nn_m": []}
    E, T = np.asarray(eval_xy, float), np.asarray(train_xy, float)
    nn = np.sqrt(((E[:, None, :] - T[None, :, :]) ** 2).sum(-1)).min(axis=1)
    k = int((nn <= near + 1e-12).sum())
    return {"n_eval": len(E), "n_train": len(T), "within": k, "beyond": len(E) - k, "frac": k / len(E),
            "nn_m": [round(float(x), 5) for x in nn]}


def item_d2(eval_xy: list, r4_xy: list, r1_xy: list, baseline_want=BASELINE_WANT) -> dict:
    new, base = coverage(eval_xy, r4_xy), coverage(eval_xy, r1_xy)
    reproduced = (base["beyond"], base["n_eval"]) == tuple(baseline_want) if base["frac"] is not None else False
    if new["frac"] is None:
        status = "要確認"
        why = "評価か学習の状態が 0 件"
    elif not reproduced:
        status = "要確認"
        why = f"R1v3 で「{baseline_want[1]} 中 {baseline_want[0]} が 3 cm 超」を再現できない（今の物差しで {base.get('beyond')}/{base['n_eval']}）。物差しを作者と確かめる"
    else:
        status = "pass" if new["frac"] >= COVER_MIN else "fail"
        why = f"R4 で 3 cm 以内 {new['within']}/{new['n_eval']} = {new['frac']:.3f}（基準 >= {COVER_MIN}）"
    return {"status": status, "why": why, "R4": new, "R1v3_baseline": base, "baseline_reproduced": reproduced,
            "definition": "評価の成立の時刻の目標の水平位置と、学習の落下の復帰デモの最初のこまの目標の水平位置の最近傍距離 <= 3 cm"}


# ---------------------------------------------------------------- d3 目視用の画像
def item_d3(chosen: list, out_dir: pathlib.Path, images_ok: bool, n: int = N_IMAGES, camera: str = "overhead") -> dict:
    picks = [c for c in chosen if c["kind"] == "B"][: n // 2] + [c for c in chosen if c["kind"] == "C"][: n - n // 2]
    picks += [c for c in chosen if c not in picks][: n - len(picks)]
    out_dir.mkdir(parents=True, exist_ok=True)
    written, missing = [], []
    for i, c in enumerate(picks[:n]):
        src = config.path(c["recovery"]["run"]) / c["recovery"]["key"] / camera / "000000.png"
        if src.is_file():
            dst = out_dir / f"{i:02d}_{c['recovery']['key']}.png"
            shutil.copyfile(src, dst)
            written.append(str(dst.relative_to(ROOT)) if dst.is_relative_to(ROOT) else str(dst))
        else:
            missing.append(str(src))
    if len(written) < n:
        status, why = "fail", f"画像が {len(written)} 枚しか書けない（欠け {len(missing)}）"
    elif images_ok:
        status, why = "pass", "作者が目で確かめた（--images-ok）"
    else:
        status, why = "要確認", f"{n} 枚を書き出した。作者が見て、問題がなければ --images-ok を付けて回し直す"
    return {"status": status, "why": why, "images": written, "missing": missing,
            "what": "新しい復帰デモ（落下・置き損ね）の記録の最初のこま（俯瞰）。落ちた立方体が机の上の妥当な位置にあるか"}


# ---------------------------------------------------------------- d4 回し直しのビット一致
def compare_npz(a: pathlib.Path, b: pathlib.Path) -> dict:
    za, zb = np.load(a), np.load(b)
    ka, kb = set(za.files), set(zb.files)
    diff = sorted(k for k in ka & kb if not (za[k].shape == zb[k].shape and za[k].dtype == zb[k].dtype
                                             and za[k].tobytes() == zb[k].tobytes()))
    return {"identical": ka == kb and not diff, "only_a": sorted(ka - kb), "only_b": sorted(kb - ka), "differ": diff}


def regen_picks(chosen: list) -> list:
    out = []
    for kind, k in N_REGEN.items():
        out += [c for c in chosen if c["kind"] == kind][:k]
    return out


def worker_history(run_dir: pathlib.Path, key: str) -> tuple:
    """(同じ働き手が key の指定より前に処理した指定の行, key の指定の行)。

    生成（recovla.expert.generate.generate）は働き手ごとに 1 つの世界を作り、その働き手の指定を順に使い回して回す。
    世界の reset は前の試みが残したハンドの力の上限のまま落ち着かせるので（docs/stage4/ops.md 5-1 の (A)）、保存した
    エピソードの始めの状態は、同じ働き手が前に回した試みに依る。generation.jsonl は投入の順（Pool.imap）に並び、
    各働き手は投入の順に仕事を取るので、同じ worker_pid の前の行が、その働き手が前に回した指定の全部になる。"""
    rows = [json.loads(ln) for ln in (run_dir / "generation.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    idx = next((i for i, r in enumerate(rows) if r.get("success") and r["attempts"][-1]["name"] == key), None)
    if idx is None:
        raise ValueError(f"{key} が {run_dir}\\generation.jsonl に成功として載っていない")
    pid = rows[idx].get("worker_pid")
    if pid is None:
        raise ValueError(f"{run_dir}\\generation.jsonl に worker_pid がない（働き手の順を組み立て直せない）")
    return [r for r in rows[:idx] if r.get("worker_pid") == pid], rows[idx]


def spec_of(row: dict, G):
    if row["kind"] == "h":
        raise ValueError("引き継ぎ（kind h）の行は作り直せない（束 4 の生成には出ない）")
    return G.EpisodeSpec(int(row["layout_seed"]), row["color"], row["layout_kind"], row["kind"])


def regenerate(picks: list, out_root: pathlib.Path, make_rig=None, G=None) -> list:
    """5 本を、元の生成と同じ並びで作り直す: 本ごとに新しい世界（生成の働き手の初めと同じ）を作り、同じ働き手が前に回した
    指定を作り直しを含めて同じ順に回してから（outputs\\s4\\b4\\gateD_regen\\<日時>\\<名前>\\_history へ）、その本の指定を
    作り直しを含めて回す（run_spec。描画あり＝元と同じ）。make_rig・G は差し替えられる（テスト）。"""
    if G is None:
        from recovla.expert import generate as G
    if make_rig is None:
        from recovla.harness.gen_v2 import SensedDrivenRig

        def make_rig():
            return SensedDrivenRig(G.rig_config("v3"))
    out = []
    for c in picks:
        key = c["recovery"]["key"]
        run_dir = config.path(c["recovery"]["run"])
        prev, row = worker_history(run_dir, key)
        here = out_root / key
        (here / "_history").mkdir(parents=True, exist_ok=True)
        rig = make_rig()
        t0 = time.perf_counter()
        try:
            for r in prev:
                G.run_spec(rig, spec_of(r, G), str(here / "_history"), True)
            res = G.run_spec(rig, spec_of(row, G), str(here), True)
        finally:
            rig.close()
        out.append({"original": run_dir / key / "data.npz", "regen": here / key / "data.npz", "history": len(prev),
                    "same_saved_attempt": bool(res["success"]) and res["attempts"][-1]["name"] == key,
                    "wall_s": round(time.perf_counter() - t0, 1)})
    return out


def item_d4(pairs: list) -> dict:
    """pairs: regenerate の出力（または (元, 作り直し) の組）。5 本とも、同じ回の試みが保存され、data.npz の全部の配列が
    型・形・バイト列まで一致すれば pass。"""
    if not pairs:
        return {"status": "要確認", "why": "回し直していない（--regen を付けて回す）", "rows": []}
    rows = []
    for p in pairs:
        p = p if isinstance(p, dict) else {"original": p[0], "regen": p[1], "same_saved_attempt": True}
        a, b = pathlib.Path(p["original"]), pathlib.Path(p["regen"])
        r = compare_npz(a, b) if b.is_file() else {"identical": False, "missing": str(b)}
        r["identical"] = bool(r["identical"] and p.get("same_saved_attempt"))
        rows.append({**{k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in p.items()}, **r})
    ok = all(r["identical"] for r in rows) and len(rows) == sum(N_REGEN.values())
    return {"status": "pass" if ok else "fail", "why": f"{sum(r['identical'] for r in rows)}/{len(rows)} 本がビット一致", "rows": rows}


# ---------------------------------------------------------------- d5・全体
def item_d5(reverified: dict) -> dict:
    """reverified: 97_s4_b4_manifest.py の reverify の出力（書いたマニフェストのファイルを読み直した検査）。全項目が通れば pass。"""
    it = (reverified or {}).get("items") or {}
    ok = bool(it) and all(v.get("ok") for v in it.values())
    return {"status": "pass" if ok else "fail", "why": "R4・N4 のマニフェストのファイルを読み直した検査（97_s4_b4_manifest.py reverify。"
            "通常デモの集合・同じ配置と目標・本数・SHA-256・組み立て直しとの一致）", "items": it}


def load_manifest_tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location("s4_b4_manifest_for_gateD", ROOT / "scripts" / "97_s4_b4_manifest.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def conclude(items: dict) -> dict:
    st = [v["status"] for k, v in items.items()]
    if any(s == "fail" for s in st):
        verdict = "研究の道へ（学習に進まない）"
    elif any(s == "要確認" for s in st):
        verdict = "要確認（確かめが済むまで学習に進まない）"
    else:
        verdict = "学習に進む"
    return {"verdict": verdict, "pass": all(s in ("pass", "対象外") for s in st)}


def cmd_check(a) -> int:
    data = json.loads((B4 / "data.json").read_text(encoding="utf-8"))
    man = json.loads((B4 / "manifest.json").read_text(encoding="utf-8"))
    mt = load_manifest_tool()
    f30 = mt.load_f30()
    _, src = mt._sources()
    reverified = mt.reverify(man, src, data, f30.layout_targets)
    r4 = json.loads((ROOT / man["manifests"]["R4"]["manifest"]).read_text(encoding="utf-8"))
    r1 = src["R1"]["manifest"]
    eval_xy = eval_drop_states(config.path(a.eval_dir))
    items = {
        "d1_start_state_coverage": {"status": "対象外", "why": "開始状態の要因の項目。滑りのデータは開始状態を変えない（gate1_sheet.md）"},
        "d2_drop_state_within_3cm": item_d2(eval_xy, train_drop_states(r4["entries"]), train_drop_states(r1["entries"]),
                                            tuple(int(x) for x in a.baseline_want.split("/"))),
        "d3_images_10": item_d3(data["chosen"], B4 / "gateD_images", a.images_ok),
        "d4_layer_i_regen_5": item_d4(regenerate(regen_picks(data["chosen"]), B4 / "gateD_regen" / time.strftime("%Y%m%d-%H%M%S"))
                                      if a.regen else []),
        "d5_same_normal_set": item_d5(reverified),
    }
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "gate": "bundle4_gates.D", "eval_dir": a.eval_dir, "items": items,
           **conclude(items)}
    B4.mkdir(parents=True, exist_ok=True)
    (B4 / "gateD.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k, v in items.items():
        print(f"{k}: {v['status']}（{v['why']}）")
    print(f"結論: {res['verdict']}")
    return 0 if res["pass"] else 1


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                    # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--eval-dir", default=DEFAULT_EVAL)
    p.add_argument("--images-ok", action="store_true")
    p.add_argument("--regen", action="store_true")
    p.add_argument("--baseline-want", default=f"{BASELINE_WANT[0]}/{BASELINE_WANT[1]}", help="R1v3 で 3 cm を超える本数/成立した本数")
    a = ap.parse_args(argv)
    return {"check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
