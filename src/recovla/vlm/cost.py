"""VLM の試験台の費用（料金表は作者の指示 10/08、per MTok の入力/出力。Batch は半額）。

    cost_usd("claude-opus-5-5", {"input_tokens": 1200, "output_tokens": 300}, batch=True)
    estimate(plan, calib)       # 本番の前の見込み（問いの数 x 1 問の入力・出力のトークン x 単価）

1 問のトークンの見込み（estimate_question_tokens）:
  smoke の実測（calib: {(モデル, 課題, 視点): {"input_tokens", "output_tokens"}}）があればそれに余裕の倍率を掛けた値、
  無ければ「画像のトークン = ceil(幅 x 高さ / 750)（仮定。実測で置き換える）」+ 文のトークン（仮定 TEXT_TOKENS）と、
  出力の仮定 OUTPUT_TOKENS（モデルごと。思考を切れない Opus は多めに置く）。
"""
import math

PRICES = {  # USD per MTok（入力, 出力）
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
BATCH_FACTOR = 0.5
CACHE_WRITE_FACTOR = 1.25          # usage に cache_creation_input_tokens があれば（この試験台は cache_control を使わない）
CACHE_READ_FACTOR = 0.1
DEFAULT_BUDGET_USD = 30.0
# 文のトークン（system＋問いの文）: 10/08 の smoke の実測。完了の判定は入力 693（256 px）・954（512 px）から画像の式の分
# （88・350）を引いた 605、失敗の分類は count_tokens 2611（768x432 を 4 枚）から 4 x 443 を引いた 839
TEXT_TOKENS = {"completion": 605, "failure": 839}
# 出力のトークン（1 問）: 完了の判定の smoke の実測は Haiku 52・60、Sonnet 48、Opus 49（思考の塊 0）。失敗の分類は理由が長く、
# Opus は思考を切れないので多めに置く（仮定。本番の同期の 20 問の実測で置き換わる）
OUTPUT_TOKENS = {"claude-haiku-5-5": 100, "claude-sonnet-5-5": 150, "claude-opus-5-5": 400}
MARGIN_IN = 1.1
MARGIN_OUT = 1.5


def image_tokens_formula(w: int, h: int) -> int:
    return int(math.ceil(w * h / 750.0))


def cost_usd(model: str, usage: dict, batch: bool = False) -> float:
    if model not in PRICES:
        raise KeyError(f"料金表に無いモデル: {model}")
    pi, po = PRICES[model]
    u = usage or {}
    tin = float(u.get("input_tokens") or 0) + CACHE_WRITE_FACTOR * float(u.get("cache_creation_input_tokens") or 0) \
        + CACHE_READ_FACTOR * float(u.get("cache_read_input_tokens") or 0)
    c = (tin * pi + float(u.get("output_tokens") or 0) * po) / 1e6
    return c * (BATCH_FACTOR if batch else 1.0)


def estimate_question_tokens(model: str, task: str, view_out: tuple, n_images: int, calib: dict = None) -> dict:
    """1 問の入力・出力のトークンの見込みと出どころ。view_out は (幅, 高さ)。"""
    c = (calib or {}).get((model, task, view_out_key(view_out)))
    if c:
        return {"input_tokens": int(math.ceil(c["input_tokens"] * MARGIN_IN)),
                "output_tokens": int(math.ceil(c["output_tokens"] * MARGIN_OUT)), "source": "smoke"}
    img = image_tokens_formula(*view_out) * n_images
    return {"input_tokens": int(math.ceil((img + TEXT_TOKENS[task]) * MARGIN_IN)),
            "output_tokens": int(math.ceil(OUTPUT_TOKENS[model] * MARGIN_OUT)), "source": "formula"}


def view_out_key(view_out) -> str:
    return f"{int(view_out[0])}x{int(view_out[1])}"


def estimate(plan: list, calib: dict = None, batch: bool = True, budget: float = DEFAULT_BUDGET_USD, spent: float = 0.0) -> dict:
    """plan: [{"task", "view", "view_out": (w, h), "n_images", "model", "n_questions", "batch": bool?}]。"""
    rows, total = [], 0.0
    for p in plan:
        tok = estimate_question_tokens(p["model"], p["task"], p["view_out"], p["n_images"], calib)
        b = p.get("batch", batch)
        per_q = cost_usd(p["model"], tok, batch=b)
        c = per_q * p["n_questions"]
        total += c
        rows.append(dict(p, view_out=list(p["view_out"]), tokens_per_question=tok, usd_per_question=round(per_q, 6),
                         usd=round(c, 4), batch=b))
    return {"rows": rows, "total_usd": round(total, 4), "spent_usd": round(spent, 4), "budget_usd": budget,
            "within_budget": spent + total <= budget + 1e-12, "prices_per_mtok": PRICES, "batch_factor": BATCH_FACTOR}
