"""段階 2 の発動の条件 (a) の判定（目標書 v2 の段階 2、0107 の 9、0108 の 2 の 7）。sensor-v1 のタグの後に 1 回だけ回す。

    .venv\\Scripts\\python.exe scripts\\86_stage2_a.py --seeds 199500:10

検証用のシードで、エキスパート（通常、空の箱・既定の開始姿勢）を、目標書 v2 の指令の口（harness/driven.py）で制限層あり・なしで回す。
(a)「G3 に合わせるためにエキスパートの動きが変わる」の判定（回す前に固めた、0107 の 9）:
  制限層が一度でも指令を切り詰めた（出力と入力の差 > 1e-9 rad/s）エピソードが 1 本でもあれば満たす。
  あわせて、グリッパの指令の速さ（旧版は位置の段差で指が最大 0.26 m/s、v2 は開き幅 0.08 m/s）と、制限層なしのときの上限違反を記録する。
出力: outputs/stage2/a_decision.json（写しを docs/results/stage2_a_decision.json）
"""
import argparse
import json
import time

from recovla.common import config

CFG = config.load_v2()
OUT = config.path(CFG["paths"]["outputs"]) / "stage2"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="199500:10")
    a = ap.parse_args(argv)
    from recovla.expert import generate as G
    from recovla.harness.driven import DrivenRig
    from recovla.sim import scene
    base, n = map(int, a.seeds.split(":"))
    rows = []
    for lim in (True, False):
        rig = DrivenRig(render=False, limiter_enabled=lim, cfg=CFG)
        for s in range(base, base + n):
            lay = scene.sample_layout(s, "empty", start="home")
            sp = G.EpisodeSpec(s, lay.table_colors[0], "empty", "n", start="home")
            c0 = rig.motion.limiter.n_clipped
            r = G.run_attempt(rig, sp, 0, None, False)
            au = rig.audit_summary()
            rows.append({"seed": s, "color": sp.color, "limiter": lim, "success": bool(r["success"]),
                         "clipped_ticks": int(rig.motion.limiter.n_clipped - c0),
                         "command_violations": au["violations"], "max_ratio": au["max_ratio"]})
            print(f"[a] limiter {lim} seed {s} ok {r['success']} clipped {rows[-1]['clipped_ticks']} "
                  f"violations {au['total_violations']}", flush=True)
        rig.close()
    on = [r for r in rows if r["limiter"]]
    res = {"seeds": a.seeds, "rule": "制限層ありで、制限層が指令を切り詰めたエピソードが 1 本でもあれば (a) を満たす（0107 の 9）",
           "episodes_clipped": sum(r["clipped_ticks"] > 0 for r in on), "n": len(on),
           "a_satisfied": any(r["clipped_ticks"] > 0 for r in on),
           "success_limiter_on": sum(r["success"] for r in on),
           "success_limiter_off": sum(r["success"] for r in rows if not r["limiter"]),
           "gripper_note": "v2 のハンドは開き幅 0.08 m/s（指 1 本 0.04 m/s）。旧版は位置の段差で指が最大 0.26 m/s（0107 の 8-1）",
           "rows": rows, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    OUT.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    (OUT / "a_decision.json").write_text(txt, encoding="utf-8")
    (config.ROOT / "docs" / "results" / "stage2_a_decision.json").write_text(txt + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
