"""段階 4 の計画役（Haiku 5.5）。凍結した decompose.py と同じ指示・スキーマ・検査で、呼び出しだけを変えたもの（作者の決定 10/08）。

    from recovla.planner import decompose_s4 as D4
    out = D4.decompose("全部片付けて", table_colors=["red", "blue"], box_colors=["green"])
    # 戻り値は decompose.decompose と同じ形 {"steps", "reply", "llm_reply", "llm_steps", "valid", "checks", "cache",
    #   "from_cache", "usage", "stop_reason"} に、印 "planner_variant": "s4_haiku55" と "model" を足したもの

decompose.py（凍結。configs の planner.model: claude-haiku-4-5、温度 0）との違い:
- モデルは引数 model（既定 claude-haiku-5-5）。configs/default.yaml は書き換えない
- 温度を送らない（Haiku 5.5 は temperature を送ると 400「`temperature` is deprecated for this model.」）。
  同じ入力の答えはキャッシュで保証する（キャッシュに無い入力は、呼ぶたびに答えが変わりうる）
- 思考は {"type": "disabled"} を明示する（Haiku 5.5 は adaptive と disabled を受ける。enabled は受けない）
- キャッシュの鍵に model・"temperature": None・thinking を入れる（Haiku 4.5 のキャッシュと混ざらない）
- 応答の記録（outputs/llm_cache/<鍵>.json）は decompose.py と同じ形に planner_variant と thinking を足す
SYSTEM・SCHEMA・PROMPT_VERSION・user_message・check・fallback_reply・client・CACHE は decompose.py のものをそのまま使う。
読むもの: decompose.py（import）、キャッシュ。書くもの: キャッシュ（outputs/llm_cache/）。鍵は decompose.client() が読む
ANTHROPIC_API_KEY（環境変数かユーザー環境変数）。
"""
import hashlib
import json
import time

from recovla.planner import decompose as _legacy
from recovla.planner.decompose import (CACHE, PROMPT_VERSION, SCHEMA, SYSTEM, check, client, fallback_reply,  # noqa: F401
                                       user_message)

DEFAULT_MODEL = "claude-haiku-5-5"
PLANNER_VARIANT = "s4_haiku55"
THINKING = {"type": "disabled"}


def cache_key(model: str, msg: str) -> str:
    """キャッシュの鍵。decompose.py の鍵の項目に、送らない温度（None）と思考の設定を足す（Haiku 4.5 の鍵とは必ず違う）。"""
    key_src = json.dumps({"model": model, "temperature": None, "thinking": THINKING, "system": SYSTEM, "schema": SCHEMA,
                          "user": msg, "prompt_version": PROMPT_VERSION}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(key_src.encode("utf-8")).hexdigest()


def decompose(text: str, table_colors, box_colors, cli=None, use_cache: bool = True, model: str = DEFAULT_MODEL) -> dict:
    p = _legacy._CFG["planner"]
    msg = user_message(text, table_colors, box_colors)
    key = cache_key(model, msg)
    path = CACHE / f"{key}.json"
    if use_cache and path.is_file():
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["from_cache"] = True
    else:
        cli = cli or client()
        t0 = time.perf_counter()
        # 温度は送らない（extra_body も使わない）。思考は disabled を明示する
        r = cli.messages.create(model=model, max_tokens=int(p["max_tokens"]),
                                system=SYSTEM, messages=[{"role": "user", "content": msg}],
                                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                                thinking=dict(THINKING))
        raw = next((b.text for b in r.content if b.type == "text"), "")
        rec = {"key": key, "model": r.model, "request": {"system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                                                          "user": msg, "prompt_version": PROMPT_VERSION,
                                                          "requested_model": model, "temperature": None,
                                                          "thinking": dict(THINKING)},
               "raw": raw, "stop_reason": r.stop_reason, "request_id": getattr(r, "_request_id", None),
               "usage": {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens},
               "wall_s": round(time.perf_counter() - t0, 3), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
               "from_cache": False, "planner_variant": PLANNER_VARIANT}
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        data = json.loads(rec["raw"])
        steps, reply = list(data["steps"]), str(data["reply"])
        parsed = True
    except (ValueError, KeyError, TypeError):
        steps, reply, parsed = [], "指示をうまく読み取れませんでした。もう一度教えてください。", False
    checks = {"parsed": parsed, **check(steps, table_colors, box_colors)}
    valid = all(checks.values())
    return {"steps": steps if valid else [], "reply": reply if valid else fallback_reply(steps, table_colors, box_colors),
            "llm_reply": reply,
            "llm_steps": steps, "valid": valid, "checks": checks, "cache": rec["key"], "from_cache": rec["from_cache"],
            "usage": rec.get("usage"), "stop_reason": rec.get("stop_reason"),
            "planner_variant": PLANNER_VARIANT, "model": rec.get("model")}
