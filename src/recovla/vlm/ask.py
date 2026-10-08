"""VLM の試験台の問い合わせ（anthropic SDK 1.x。クライアントは planner/decompose.py の client() を使う）。

問いの文・答えの形・モデルの設定は結果を見る前に固定する（docs/stage4/vlm_bench_protocol.md の 5・6 節）。
  - 5 世代のモデルは temperature を受け付けないので送らない
  - 構造化出力 output_config.format = json_schema（decompose.py と同じ形）
  - 思考: Haiku 5.5 は thinking disabled、Sonnet 5.5 は thinking between_tools ＋ effort low、Opus 5.5 は thinking を送らず
    （切れない）effort low。設定ごとに smoke で 1 回ずつ通ったことを確かめた形だけを使う（MODEL_SETTINGS）
  - 拒否の代わりのモデル（fallbacks）は使わない（名前を挙げたモデルの答えだけを測る。Batch でも使えない）
キャッシュ: outputs/s4/vlm/cache/<鍵>.json。鍵 = SHA-256(モデル・設定・system の SHA-256・問いの文・画像の SHA-256 の並び・
  schema の SHA-256・PROMPT_VERSION)。同じ鍵は二度呼ばない。usage（入力・出力・キャッシュのトークン）・かかった時間
  （同期はその呼び出しの秒、Batch は作ってから終わるまでの秒）・応答のモデル名・request_id を残す。
大量の問いは Message Batches（半額）: create → retrieve で processing_status が "ended" まで待つ → results を custom_id で
  突き合わせる。出した Batch は outputs/s4/vlm/batches/<id>.json に custom_id と鍵の対応を先に書いてから待つ（落ちても回収できる）。
"""
import base64
import hashlib
import json
import pathlib
import time

from recovla.common import config
from recovla.vlm import cost as C
from recovla.vlm import labels as L

PROMPT_VERSION = 1
OUT = config.path(config.load()["paths"]["outputs"]) / "s4" / "vlm"
CACHE = OUT / "cache"
BATCHES = OUT / "batches"

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
MODEL_SETTINGS = {
    # 送る引数のうち、問いによらない部分（output_config の format は問いごとに足す）。smoke で通った形（手順書 6 節）
    "claude-haiku-5-5": {"max_tokens": 1024, "thinking": {"type": "disabled"}, "output_config": {}},
    "claude-sonnet-5-5": {"max_tokens": 2048, "thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}},
    "claude-opus-5-5": {"max_tokens": 4096, "output_config": {"effort": "low"}},
}
SYNC_TIMEOUT_S = 180.0
BATCH_MAX_REQUESTS = 5000
BATCH_MAX_BYTES = 100 * 1024 * 1024        # API の上限 256 MB より小さく切る（メモリにも 1 塊ずつしか載せない）
POLL_S = 30.0

COLOR_JA = {"red": "赤", "green": "緑", "blue": "青"}

SYSTEM_COMPLETION = """あなたはロボットの作業の結果を確かめる係です。画像は、机の上の作業場を斜め上から撮ったカメラの 1 枚です（ロボットの腕が写ることがあります）。
机の上には赤・緑・青の立方体（一辺 4 cm）と、上が開いた四角い箱が 1 個あります。
問われた色の立方体が「箱の中に置かれている」かを答えてください。
- 箱の中に置かれている: 立方体が箱の 4 枚の壁の内側にあり、箱の底の上にある。
- 次は no: 箱の外の机の上にある、箱の壁や縁の上に乗っている、ロボットの指に挟まれて持たれている（箱の上や中にあっても）。
- 画像から決められないとき（隠れて見えないなど）は unsure を選んでください。
答えは指定の JSON の形だけで返してください。reason は日本語で 40 字以内にしてください。"""

SYSTEM_FAILURE = """あなたはロボットの作業の記録を見て、何が起きたかを分類する係です。
ロボットの腕が、机の上の指定された色の立方体（一辺 4 cm）をつかみ、上が開いた四角い箱に入れようとしています。
画像は同じカメラで撮った 4 枚で、時刻の順に並んでいます（各画像の前に、1 枚目からの経過秒を書きます）。
4 枚の間に、指定された色の立方体について起きたことを、次の 5 つから 1 つ選んでください。
- grasp_miss（掴み損ね）: 指を閉じたが立方体をつかめず、立方体は持ち上がらなかった
- drop（持ち上げた後に落とした）: 立方体を持ち上げたが、運んでいる途中など、下ろす前に手から離れて落ちた
- misplace（箱の縁などに置き損ねた）: 立方体を下ろしながら、または止まって放したが、箱の中に入らなかった（箱の縁・壁の上、箱の外の机の上など）
- stall（止まった・動けない）: 腕がほとんど動かず、作業が進んでいない
- success（成功）: 立方体を放して、箱の中に入れた
答えは指定の JSON の形だけで返してください。reason は日本語で 60 字以内にしてください。"""

SCHEMA_COMPLETION = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unsure"]}, "reason": {"type": "string"}},
    "required": ["answer", "reason"],
    "additionalProperties": False,
}
SCHEMA_FAILURE = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": list(L.CLASSES)}, "reason": {"type": "string"}},
    "required": ["answer", "reason"],
    "additionalProperties": False,
}
SYSTEM = {"completion": SYSTEM_COMPLETION, "failure": SYSTEM_FAILURE}
SCHEMA = {"completion": SCHEMA_COMPLETION, "failure": SCHEMA_FAILURE}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_json(o) -> str:
    return sha256_text(json.dumps(o, ensure_ascii=False, sort_keys=True))


# ---------------------------------------------------------------- 問い
def question_texts(item: dict, files: list) -> list:
    """画像の前に置く文の並び（画像と同じ数）と、最後の問いの文。返り値 [前置きの文..., 問いの文]。"""
    if item["task"] == "completion":
        return ["", f"質問: {COLOR_JA[item['color']]}の立方体は箱の中に置かれていますか。"]
    t0 = files[0]["state"]["t_render"]
    pre = [f"{k + 1} 枚目（{f['state']['t_render'] - t0:.1f} 秒）" for k, f in enumerate(files)]
    return pre + [f"指定された色: {COLOR_JA[item['target']]}。この 4 枚の間に、{COLOR_JA[item['target']]}の立方体について起きたことを選んでください。"]


def build_content(item: dict, files: list, image_bytes: list) -> list:
    texts = question_texts(item, files)
    content = []
    for k, b in enumerate(image_bytes):
        if texts[k]:
            content.append({"type": "text", "text": texts[k]})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                     "data": base64.standard_b64encode(b).decode("ascii")}})
    content.append({"type": "text", "text": texts[-1]})
    return content


def build_params(task: str, model: str, content: list) -> dict:
    s = MODEL_SETTINGS[model]
    oc = dict(s.get("output_config") or {})
    oc["format"] = {"type": "json_schema", "schema": SCHEMA[task]}
    p = {"model": model, "max_tokens": int(s["max_tokens"]), "system": SYSTEM[task],
         "messages": [{"role": "user", "content": content}], "output_config": oc}
    if s.get("thinking") is not None:
        p["thinking"] = dict(s["thinking"])
    return p


def cache_key(task: str, model: str, item: dict, files: list) -> str:
    texts = question_texts(item, files)
    return sha256_json({"model": model, "settings": MODEL_SETTINGS[model], "system_sha256": sha256_text(SYSTEM[task]),
                        "question": texts, "images_sha256": [f["sha256"] for f in files],
                        "schema_sha256": sha256_json(SCHEMA[task]), "prompt_version": PROMPT_VERSION})


def custom_id(key: str) -> str:
    return "k" + key[:48]


# ---------------------------------------------------------------- キャッシュ
def cache_path(key: str, cache_dir=None) -> pathlib.Path:
    return pathlib.Path(cache_dir or CACHE) / f"{key}.json"


def cache_get(key: str, cache_dir=None):
    p = cache_path(key, cache_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def cache_put(rec: dict, cache_dir=None) -> None:
    p = cache_path(rec["key"], cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def _usage(u) -> dict:
    if u is None:
        return {}
    g = (lambda k: u.get(k)) if isinstance(u, dict) else (lambda k: getattr(u, k, None))
    return {k: g(k) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
            if g(k) is not None}


def parse_answer(task: str, raw: str):
    try:
        d = json.loads(raw)
    except (TypeError, ValueError):
        return None
    allowed = SCHEMA[task]["properties"]["answer"]["enum"]
    if not isinstance(d, dict) or d.get("answer") not in allowed:
        return None
    return {"answer": d["answer"], "reason": str(d.get("reason", ""))}


def record_from_message(msg, key: str, meta: dict, mode: str, wall_s, extra: dict = None) -> dict:
    raw = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), "")
    usage = _usage(getattr(msg, "usage", None))
    model = meta["model"]
    rec = {"key": key, "item_id": meta["item_id"], "task": meta["task"], "view": meta["view"], "model_requested": model,
           "model": getattr(msg, "model", None), "mode": mode, "raw": raw, "parsed": parse_answer(meta["task"], raw),
           "stop_reason": getattr(msg, "stop_reason", None), "usage": usage,
           "thinking_blocks": sum(1 for b in msg.content if getattr(b, "type", None) in ("thinking", "redacted_thinking")),
           "usd": round(C.cost_usd(model, usage, batch=(mode == "batch")), 8), "wall_s": wall_s,
           "request_id": getattr(msg, "_request_id", None), "settings": MODEL_SETTINGS[model],
           "prompt_version": PROMPT_VERSION, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    rec.update(extra or {})
    return rec


# ---------------------------------------------------------------- 問いの組み立て（画像は描き直しの目録から）
def load_manifest(task: str, view: str, out=None) -> dict:
    p = pathlib.Path(out or OUT) / "images" / task / view / "manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p} が無い（先に render）")
    return json.loads(p.read_text(encoding="utf-8"))


def questions(task: str, view: str, model: str, items: list, manifest: dict, out=None) -> list:
    """描いた項目ごとの (鍵, 項目, 目録の画像)。"""
    qs = []
    for it in items:
        files = manifest["items"].get(it["item_id"])
        if not files:
            continue
        qs.append({"key": cache_key(task, model, it, files), "item": it, "files": files, "task": task, "view": view,
                   "model": model})
    return qs


def load_images(q: dict, out=None) -> list:
    d = pathlib.Path(out or OUT) / "images" / q["task"] / q["view"]
    bs = []
    for f in q["files"]:
        b = (d / f["file"]).read_bytes()
        if hashlib.sha256(b).hexdigest() != f["sha256"]:
            raise RuntimeError(f"画像の SHA-256 が目録と違う: {f['file']}")
        bs.append(b)
    return bs


def spent_usd(cache_dir=None) -> float:
    d = pathlib.Path(cache_dir or CACHE)
    if not d.is_dir():
        return 0.0
    tot = 0.0
    for p in d.glob("*.json"):
        try:
            tot += float(json.loads(p.read_text(encoding="utf-8")).get("usd") or 0.0)
        except (ValueError, OSError):
            pass
    return tot


# ---------------------------------------------------------------- 同期の呼び出し
def ask_sync(qs: list, cli, cache_dir=None, out=None, log=print) -> list:
    recs = []
    for q in qs:
        hit = cache_get(q["key"], cache_dir)
        if hit is not None:
            recs.append(dict(hit, from_cache=True))
            continue
        params = build_params(q["task"], q["model"], build_content(q["item"], q["files"], load_images(q, out)))
        t0 = time.perf_counter()
        msg = cli.messages.create(**params)
        wall = round(time.perf_counter() - t0, 3)
        rec = record_from_message(msg, q["key"], {"item_id": q["item"]["item_id"], "task": q["task"], "view": q["view"],
                                                  "model": q["model"]}, "sync", wall)
        cache_put(rec, cache_dir)
        recs.append(dict(rec, from_cache=False))
        if log:
            log(f"[ask] {q['model']} {q['item']['item_id']} {rec['parsed']} {rec['usage']} {wall}s")
    return recs


# ---------------------------------------------------------------- Message Batches
def request_bytes_estimate(q: dict, out=None) -> int:
    """1 問の要求の大きさの見込み（画像のファイルの大きさ x 4/3 ＋ 文の分）。画像は読み込まない。"""
    d = pathlib.Path(out or OUT) / "images" / q["task"] / q["view"]
    return sum(int((d / f["file"]).stat().st_size * 4 / 3) + 200 for f in q["files"]) + 8192


def batch_plan(qs: list, out=None) -> list:
    """問いの塊の並び（1 塊は BATCH_MAX_REQUESTS 件・BATCH_MAX_BYTES 以下）。画像はまだ読まない（メモリを使わない）。"""
    chunks, cur, size = [], [], 0
    for q in qs:
        n = request_bytes_estimate(q, out)
        if cur and (len(cur) >= BATCH_MAX_REQUESTS or size + n > BATCH_MAX_BYTES):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(q)
        size += n
    if cur:
        chunks.append(cur)
    return chunks


def submit_batches(qs: list, cli, batches_dir=None, cache_dir=None, out=None, log=print) -> list:
    """キャッシュに無い問いだけを Batch に出す。返り値は出した Batch の記録の並び。"""
    todo = [q for q in qs if cache_get(q["key"], cache_dir) is None]
    seen, uniq = set(), []
    for q in todo:                                   # 同じ鍵は 1 回だけ
        if q["key"] not in seen:
            seen.add(q["key"])
            uniq.append(q)
    bdir = pathlib.Path(batches_dir or BATCHES)
    bdir.mkdir(parents=True, exist_ok=True)
    out_recs = []
    for chunk in batch_plan(uniq, out):
        reqs = [{"custom_id": custom_id(q["key"]),
                 "params": build_params(q["task"], q["model"], build_content(q["item"], q["files"], load_images(q, out)))}
                for q in chunk]
        b = cli.messages.batches.create(requests=reqs)
        del reqs
        rec = {"batch_id": b.id, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "t_created": time.time(),
               "n": len(chunk), "status": getattr(b, "processing_status", None),
               "map": {custom_id(q["key"]): {"key": q["key"], "item_id": q["item"]["item_id"], "task": q["task"],
                                             "view": q["view"], "model": q["model"]} for q in chunk}}
        (bdir / f"{b.id}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        out_recs.append(rec)
        if log:
            log(f"[batch] 出した {b.id}: {len(chunk)} 件")
    return out_recs


def collect_batch(batch_id: str, cli, batches_dir=None, cache_dir=None, wait: bool = True, poll_s: float = POLL_S,
                  log=print, sleep=time.sleep) -> dict:
    """Batch が終わるまで待ち（wait）、結果を custom_id で突き合わせてキャッシュに入れる。"""
    bdir = pathlib.Path(batches_dir or BATCHES)
    path = bdir / f"{batch_id}.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    while True:
        b = cli.messages.batches.retrieve(batch_id)
        st = getattr(b, "processing_status", None)
        if st == "ended" or not wait:
            break
        if log:
            log(f"[batch] {batch_id} {st} {getattr(b, 'request_counts', None)}")
        sleep(poll_s)
    rec["status"] = st
    if st != "ended":
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec
    elapsed = round(time.time() - float(rec.get("t_created") or time.time()), 1)
    counts = {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0, "unknown_custom_id": 0}
    errors = []
    for r in cli.messages.batches.results(batch_id):
        m = rec["map"].get(r.custom_id)
        typ = r.result.type
        if m is None:
            counts["unknown_custom_id"] += 1
            continue
        counts[typ] = counts.get(typ, 0) + 1
        if typ == "succeeded":
            c = record_from_message(r.result.message, m["key"], m, "batch", None,
                                    {"batch_id": batch_id, "batch_elapsed_s": elapsed, "custom_id": r.custom_id})
            cache_put(c, cache_dir)
        else:
            err = getattr(r.result, "error", None)
            errors.append({"custom_id": r.custom_id, "type": typ, "error": str(err)[:500] if err is not None else None})
    rec.update(ended=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_s=elapsed, counts=counts, errors=errors)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    if log:
        log(f"[batch] {batch_id} 終わり {counts}（{elapsed} s）")
    return rec
