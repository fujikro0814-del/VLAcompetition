"""通常デモの生成（手順書 Step D の 2・3・5、B_提案書 §6・§13）。

    .venv\\Scripts\\python.exe scripts\\10_gen_normal.py --name R1_normal --seeds empty:20000-20199 prefilled_1:... \\
        [--workers 8] [--no-render] [--kind-by-ratio 20000-20399]

- --seeds KIND:A-B は、配置の種 A〜B（両端を含む）を配置の種類 KIND で作る。机上の全色を目標にする（組）
- --kind-by-ratio A-B は、配置の種類を配置の乱数列から scene.layout_kinds の割合で決める
- 出力は outputs/gen/<name>_<日時>/（あれば拒否）。generation.jsonl・run.json・timing.json と、成功したエピソード
- --no-render は描画をしない（物理・判定・記録の配列は同じ。画像がないので学習には使えない。台本の調整・集計用）
"""
import argparse
import sys
import time

from recovla.common import config
from recovla.expert import generate as G
from recovla.sim import scene


def parse_range(s: str) -> range:
    a, b = s.split("-")
    return range(int(a), int(b) + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", nargs="*", default=[], help="KIND:A-B（KIND は empty・prefilled_1・prefilled_2）")
    ap.add_argument("--kind-by-ratio", nargs="*", default=[], help="A-B（配置の種類を割合で決める）")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    specs = []
    for item in a.seeds:
        kind, rng_ = item.split(":")
        if kind not in scene.KINDS:
            ap.error(f"unknown kind {kind}")
        specs += G.plan_specs({kind: parse_range(rng_)})
    for rng_ in a.kind_by_ratio:
        for seed in parse_range(rng_):
            lay = scene.sample_layout(seed)
            specs += [G.EpisodeSpec(seed, c, None) for c in lay.table_colors]
    if not specs:
        ap.error("no specs (give --seeds or --kind-by-ratio)")
    out = config.path(config.load()["paths"]["outputs"]) / "gen" / f"{a.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    results = G.generate(specs, out, workers=a.workers, render=not a.no_render)
    ok = sum(r["success"] for r in results)
    first = sum(r["first_try_success"] for r in results)
    print(f"[gen] {out}: {len(results)} specs, saved {ok}, first-try {first}, dropped {len(results) - ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
