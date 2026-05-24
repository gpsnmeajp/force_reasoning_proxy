# ─────────────────────────────────────────────────────────────────────────────
# force_reasoning_proxy — アップストリームの LLM に推論（reasoning）を強制する
# リバースプロキシです。
# クライアントからのリクエストを受け取り、推論コンテンツが含まれるまで
# 自動的にリトライを行います。
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import argparse
import copy
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Literal, cast

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# ── 設定定数 ──────────────────────────────────────────────────────────────────
# アップストリームサーバーのベース URL です。
# 起動引数 --upstream で上書きできます。
UPSTREAM_BASE_URL = "http://localhost:8080/v1"

# ストリーミング時にバッファリングするチャンク数です。
# 少なくともこの数まではクライアントへの送信を保留し、推論が確認できた後に送信を始めます。
BUFFER_CHUNKS = 5

# 推論コンテンツが得られない場合の最大リトライ回数です。
MAX_RETRIES = 100

# このリトライ回数を超えた場合、プロンプトに THINK_TOKEN を付加し始めます。
THINK_THRESHOLD = 5

# 推論を促すために末尾に付加するトークンです。
THINK_TOKEN = "<|think|>"

# --model 引数で指定された場合に、クライアントのモデル名を上書きします。
# None のときは上書きしません。
FORCE_MODEL: str | None = None

# 連続的推論サポート: クライアントから送られてくる assistant メッセージの
# reasoning_content / reasoning を最新 N 件のみ保持し、古いものは削除します。
# 0 はすべて削除、負の値は無制限に保持します。
KEEP_REASONING_N: int = 5

# チャンク途絶タイムアウト（秒）。推論中・生成中に意味のあるチャンクが届かない時間が
# この値を超えた場合、生成をキャンセルしてリトライします。0 以下で無効。
CHUNK_TIMEOUT: float = 10.0

# Reasoning タイムアウト（秒）。Reasoning フェーズがこの時間を超えた場合、
# 生成をキャンセルしてリトライします。0 以下で無効。
REASONING_TIMEOUT: float = 600.0

# 生成タイムアウト（秒）。コンテンツ生成フェーズがこの時間を超えた場合、
# 生成をキャンセルしてリトライします。0 以下で無効。
GENERATION_TIMEOUT: float = 600.0


# ── アプリケーションライフサイクル ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリ起動時に共有の HTTP クライアントを生成し、終了時にクローズします。"""
    # アプリケーション全体で再利用する非同期 HTTP クライアントを初期化します。
    app.state.http_client = httpx.AsyncClient()
    logger.info(
        "proxy startup upstream=%s buffer_chunks=%s max_retries=%s think_threshold=%s force_model=%s keep_reasoning_n=%s chunk_timeout=%s reasoning_timeout=%s generation_timeout=%s",
        UPSTREAM_BASE_URL,
        BUFFER_CHUNKS,
        MAX_RETRIES,
        THINK_THRESHOLD,
        FORCE_MODEL or "-",
        KEEP_REASONING_N,
        CHUNK_TIMEOUT,
        REASONING_TIMEOUT,
        GENERATION_TIMEOUT,
    )
    yield
    # アプリケーション終了時にクライアントを適切にクローズします。
    logger.info("proxy shutdown")
    await app.state.http_client.aclose()


app = FastAPI(lifespan=lifespan)
logger = logging.getLogger("uvicorn.error")


def _make_request_id(request: Request) -> str:
    """受信したリクエストに対するログ相関 ID を返します。"""
    incoming = request.headers.get("x-request-id")
    if incoming:
        return incoming
    return uuid.uuid4().hex[:8]


def _preview_bytes(data: bytes, limit: int = 200) -> str:
    """ログ出力用にレスポンス本文を短く整形します。"""
    text = data.decode("utf-8", errors="replace")
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _elapsed_ms(started_at: float) -> int:
    """開始時刻からの経過時間をミリ秒で返します。"""
    return int((time.perf_counter() - started_at) * 1000)


def _summarize_message_roles(messages: Any) -> str:
    """メッセージ列の role 内訳を短い文字列に要約します。"""
    if not isinstance(messages, list) or not messages:
        return "-"

    counts: dict[str, int] = {}
    for item in messages:
        role = item.get("role") if isinstance(item, dict) else None
        role_key = role if isinstance(role, str) and role else "?"
        counts[role_key] = counts.get(role_key, 0) + 1
    return ",".join(f"{role}:{counts[role]}" for role in sorted(counts))


def _trim_old_reasoning(messages: list, keep_n: int) -> list:
    """メッセージリスト内の古い reasoning_content / reasoning を削除します。

    assistant メッセージの reasoning_content / reasoning を最新 keep_n 件のみ残します。
    keep_n が 0 の場合はすべて削除します。
    keep_n が負の場合は何も削除しません。
    """
    if keep_n < 0:
        return messages

    reasoning_keys = {"reasoning_content", "reasoning"}

    # reasoning_content / reasoning を持つ assistant メッセージのインデックスを収集します。
    reasoning_indices = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict)
        and m.get("role") == "assistant"
        and any(key in m for key in reasoning_keys)
    ]

    if len(reasoning_indices) <= keep_n:
        # 削除対象がないのでそのまま返します。
        return messages

    # 古い方から削除対象のインデックスセットを決定します。
    remove_set = set(reasoning_indices[: len(reasoning_indices) - keep_n])

    result = []
    for i, msg in enumerate(messages):
        if i in remove_set:
            # 推論フィールドのみ除いた新しい dict を生成します。
            msg = {k: v for k, v in msg.items() if k not in reasoning_keys}
        result.append(msg)
    return result


def _summarize_chat_body(body: dict) -> str:
    """チャット補完リクエストの機微でない情報だけを要約します。"""
    messages = body.get("messages")
    message_count = len(messages) if isinstance(messages, list) else 0
    tools = body.get("tools")
    tool_count = len(tools) if isinstance(tools, list) else 0

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict):
        tool_choice_label = cast(str, tool_choice.get("type") or "object")
    elif tool_choice is None:
        tool_choice_label = "-"
    else:
        tool_choice_label = str(tool_choice)

    response_format = body.get("response_format")
    if isinstance(response_format, dict):
        response_format_label = cast(str, response_format.get("type") or "object")
    elif response_format is None:
        response_format_label = "-"
    else:
        response_format_label = str(response_format)

    return (
        f"messages={message_count} roles={_summarize_message_roles(messages)} "
        f"tools={tool_count} tool_choice={tool_choice_label} "
        f"response_format={response_format_label}"
    )


def _with_request_id_header(response: Response, request_id: str) -> Response:
    """レスポンスに request_id を付与して返します。"""
    response.headers["X-Proxy-Request-Id"] = request_id
    return response


# ── SSE（Server-Sent Events）パーサー ────────────────────────────────────────

def parse_sse_line(line: bytes) -> bytes | None:
    """SSE の 1 行を受け取り、'data:' フィールドの値を返します。

    'data:' で始まらない行（イベント名・コメント等）は None を返します。
    """
    if not line.startswith(b"data:"):
        return None
    # 先頭の 'data:' 部分を除去し、余分な空白を取り除いて返します。
    return line[5:].lstrip()


# ── チャンク検査ユーティリティ ───────────────────────────────────────────────

def chunk_has_content(obj: dict) -> bool:
    """チャンクオブジェクトに通常コンテンツ（content）が含まれるか検査します。

    推論なしにコンテンツが先に届いた場合はリトライが必要と判断するために使用します。
    """
    for ch in obj.get("choices", []) or []:
        delta = ch.get("delta") or {}
        c = delta.get("content")
        if isinstance(c, str) and c != "":
            return True
    return False


def chunk_has_reasoning(obj: dict) -> bool:
    """チャンクオブジェクトに推論コンテンツが含まれるか検査します。

    サーバーによって 'reasoning_content' または 'reasoning' キーが使われるため、
    両方を確認します。
    """
    for ch in obj.get("choices", []) or []:
        delta = ch.get("delta") or {}
        # 標準的な推論フィールドを確認します。
        r = delta.get("reasoning_content")
        if isinstance(r, str) and r != "":
            return True
        # 一部のサーバーでは 'reasoning' キーが使われます。
        r2 = delta.get("reasoning")
        if isinstance(r2, str) and r2 != "":
            return True
    return False


def chunk_is_finish(obj: dict) -> bool:
    """チャンクオブジェクトがストリームの終了チャンクか検査します。

    finish_reason が設定されているチョイスが 1 つでもあれば True を返します。
    """
    for ch in obj.get("choices", []) or []:
        if ch.get("finish_reason"):
            return True
    return False


def format_sse_comment(message: str) -> bytes:
    """SSE コメント行を生成します。"""
    sanitized = message.replace("\r", " ").replace("\n", " ")
    return f": {sanitized}".encode("utf-8")


def build_retry_payload(original_body: dict, retry_count: int) -> dict:
    """リトライ用リクエストボディを構築します。

    retry_count が THINK_THRESHOLD を超えた場合は、最後のユーザーメッセージの
    末尾に THINK_TOKEN を付加し、モデルに推論を促します。
    閾値を超えるほど多くのトークンを付加します。
    """
    # 元のボディを深くコピーして元データを保護します。
    body = copy.deepcopy(original_body)
    if retry_count > THINK_THRESHOLD:
        # 閾値を超えたリトライ回数分だけ THINK_TOKEN を付加します。
        extra_n = retry_count - THINK_THRESHOLD
        suffix = THINK_TOKEN * (extra_n // 5) # 回数そのままだとトークンが多すぎるため、5 回ごとに 1 トークン付加するように調整します。
        messages = body.get("messages") or []
        # メッセージリストを末尾から検索し、最後のユーザーメッセージを見つけます。
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                msg = messages[i]
                content = msg.get("content")
                if isinstance(content, str):
                    # 通常のテキストメッセージの場合は末尾に追加します。
                    msg["content"] = content + suffix
                elif isinstance(content, list):
                    # OpenAI のビジョン形式（コンテンツ配列）の場合は
                    # 新しいテキストパートとして追加します。
                    msg["content"] = content + [{"type": "text", "text": suffix}]
                else:
                    msg["content"] = suffix
                break
    # アップストリームへは常にストリーミングリクエストとして送信します。
    body["stream"] = True
    return body


def _merge_stream_value(existing: Any, new_value: Any) -> Any:
    """ストリーミングで分割された値を非ストリーミング向けに結合します。"""
    if existing is None:
        return copy.deepcopy(new_value)
    if new_value is None:
        return copy.deepcopy(existing)
    if isinstance(existing, str) and isinstance(new_value, str):
        return existing + new_value
    if isinstance(existing, list) and isinstance(new_value, list):
        return existing + copy.deepcopy(new_value)
    if isinstance(existing, dict) and isinstance(new_value, dict):
        merged = copy.deepcopy(existing)
        for key, value in new_value.items():
            if key in merged:
                merged[key] = _merge_stream_value(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(new_value)


def assemble_non_streaming_response(
    chunks: list[dict], model_hint: str | None
) -> dict:
    """ストリーミングチャンクをまとめて、非ストリーミング形式のレスポンスを組み立てます。

    クライアントが非ストリーミング（stream=false）を要求した場合に、
    アップストリームから受け取った複数のチャンクを 1 つのレスポンスに結合します。
    """
    # 複数のチョイス（候補応答）を番号ごとに蓄積します。
    result: dict[str, Any] = {}
    choices: dict[int, dict[str, Any]] = {}

    # 各チャンクを順に処理してメタデータとコンテンツを蓄積します。
    for obj in chunks:
        for key, value in obj.items():
            if key == "choices":
                continue
            if key == "object" and isinstance(value, str) and value.endswith(".chunk"):
                result[key] = value[: -len(".chunk")]
            else:
                result[key] = copy.deepcopy(value)

        for ch in obj.get("choices", []) or []:
            idx = ch.get("index", 0)
            entry = choices.setdefault(
                idx,
                {
                    "index": idx,
                    "message": {"role": "assistant"},
                    "finish_reason": None,
                },
            )
            delta = ch.get("delta") or {}
            msg = entry["message"]
            # ロールが存在する場合は上書きします。
            if "role" in delta and delta["role"]:
                msg["role"] = delta["role"]
            # 通常コンテンツを文字列として結合します。
            if "content" in delta:
                if isinstance(delta["content"], str):
                    existing_content = msg.get("content")
                    if isinstance(existing_content, str):
                        msg["content"] = existing_content + delta["content"]
                    else:
                        msg["content"] = delta["content"]
                elif delta["content"] is None and "content" not in msg:
                    msg["content"] = None
            # 推論コンテンツ（reasoning_content）を結合します。
            if "reasoning_content" in delta and isinstance(
                delta["reasoning_content"], str
            ):
                msg["reasoning_content"] = (
                    msg.get("reasoning_content") or ""
                ) + delta["reasoning_content"]
            # 一部サーバー用の 'reasoning' キーも同様に結合します。
            if "reasoning" in delta and isinstance(delta["reasoning"], str):
                msg["reasoning"] = (msg.get("reasoning") or "") + delta["reasoning"]
            # ツール呼び出しのデルタを蓄積します。
            if "tool_calls" in delta and delta["tool_calls"]:
                existing = msg.setdefault("tool_calls", [])
                for tc in delta["tool_calls"]:
                    tc_idx = tc.get("index", 0)
                    # スロットが足りない場合は初期化して追加します。
                    while len(existing) <= tc_idx:
                        existing.append(
                            {
                                "index": len(existing),
                                "id": None,
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        )
                    slot = existing[tc_idx]
                    slot["index"] = tc_idx
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    if tc.get("type"):
                        slot["type"] = tc["type"]
                    fn = tc.get("function") or {}
                    # 関数名と引数を結合します。
                    if "name" in fn and fn["name"]:
                        slot["function"]["name"] = (
                            slot["function"].get("name") or ""
                        ) + fn["name"]
                    if "arguments" in fn and isinstance(fn["arguments"], str):
                        slot["function"]["arguments"] = (
                            slot["function"].get("arguments") or ""
                        ) + fn["arguments"]
                    for key, value in tc.items():
                        if key in {"index", "id", "type", "function"}:
                            continue
                        slot[key] = _merge_stream_value(slot.get(key), value)
            # 既知のキー以外の未知のデルタフィールドはそのまま透過させます。
            for k, v in delta.items():
                if k in {
                    "role",
                    "content",
                    "reasoning_content",
                    "reasoning",
                    "tool_calls",
                }:
                    continue
                msg[k] = _merge_stream_value(msg.get(k), v)
            if "finish_reason" in ch:
                entry["finish_reason"] = ch["finish_reason"]
            # 既知のキー以外の未知のチョイスフィールドも透過させます。
            for k, v in ch.items():
                if k in {"index", "delta", "finish_reason"}:
                    continue
                entry[k] = _merge_stream_value(entry.get(k), v)

    # 最終的なレスポンスオブジェクトを組み立てます。
    # id が取得できなかった場合は UUID で生成します。
    result.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
    result.setdefault("object", "chat.completion")
    result.setdefault("created", int(time.time()))
    if "model" not in result and model_hint is not None:
        result["model"] = model_hint
    result["choices"] = [choices[i] for i in sorted(choices.keys())]
    return result


# ── アップストリーム通信 ──────────────────────────────────────────────────────

async def stream_attempt(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    body: dict,
    params: tuple[tuple[str, str], ...] | None = None,
    request_id: str = "-",
) -> AsyncGenerator[
    tuple[Literal["chunk"], bytes, dict]
    | tuple[Literal["raw"], bytes, None]
    | tuple[Literal["done"], None, None]
    | tuple[Literal["end"], None, None],
    None,
]:
    """アップストリームへの 1 回の接続試行を行い、SSE イベントを非同期ジェネレータで返します。

    生成するタプルの形式:
      ("chunk", raw_line_bytes, parsed_obj)  — SSE データ行（JSON パース済み）
            ("raw",   raw_line_bytes, None)        — data 以外の SSE 行または JSON 解析できない行（透過）
      ("done",  None, None)                  — [DONE] センチネルを受信
      ("end",   None, None)                  — ストリームが正常終了

    トランスポートエラーが発生した場合は httpx の例外を送出します。
    """
    started_at = time.perf_counter()
    raw_line_count = 0
    json_error_count = 0
    first_raw_line: bytes | None = None
    first_json_error_line: bytes | None = None
    terminal_event = "closed"

    try:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=body,
            params=params,
            timeout=None,  # タイムアウトなし（長時間の推論に対応するため）
        ) as resp:
            # HTTP エラーレスポンスは UpstreamError として送出します。
            if resp.status_code >= 400:
                terminal_event = f"http_{resp.status_code}"
                body_bytes = await resp.aread()
                logger.warning(
                    "[%s] upstream error status=%s url=%s body=%s",
                    request_id,
                    resp.status_code,
                    url,
                    _preview_bytes(body_bytes),
                )
                raise UpstreamError(
                    resp.status_code,
                    body_bytes,
                    _filter_response_headers(dict(resp.headers.items())),
                )
            buf = b""
            # レスポンスボディをバイト単位で受信し、改行ごとに処理します。
            async for raw in resp.aiter_raw():
                buf += raw
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line = buf[:nl]
                    buf = buf[nl + 1 :]
                    line = line.rstrip(b"\r")
                    # 空行は SSE のフィールド区切りなのでスキップします。
                    if not line:
                        continue
                    data = parse_sse_line(line)
                    if data is None:
                        # 'data:' 以外の SSE フィールド（event:, id:, コメントなど）は透過します。
                        raw_line_count += 1
                        if first_raw_line is None:
                            first_raw_line = line
                        yield ("raw", line, None)
                        continue
                    # ストリーム終了センチネルを受信したら終了します。
                    if data == b"[DONE]":
                        terminal_event = "done"
                        yield ("done", None, None)
                        return
                    try:
                        obj = json.loads(data.decode("utf-8"))
                    except Exception:
                        # JSON パースに失敗した場合は生データとして透過します。
                        json_error_count += 1
                        if first_json_error_line is None:
                            first_json_error_line = line
                        yield ("raw", line, None)
                        continue
                    yield ("chunk", line, obj)
            # バッファに残ったデータをフラッシュします。
            if buf.strip():
                line = buf.strip().rstrip(b"\r")
                data = parse_sse_line(line)
                if data is not None and data != b"[DONE]":
                    try:
                        obj = json.loads(data.decode("utf-8"))
                        yield ("chunk", line, obj)
                    except Exception:
                        json_error_count += 1
                        if first_json_error_line is None:
                            first_json_error_line = line
            terminal_event = "end"
            yield ("end", None, None)
    finally:
        if raw_line_count or json_error_count or terminal_event in {"closed", "end"}:
            log = logger.warning if json_error_count or terminal_event in {"closed", "end"} else logger.info
            log(
                "[%s] stream attempt summary url=%s terminal=%s raw_lines=%s json_errors=%s first_raw=%s first_bad_json=%s duration_ms=%s",
                request_id,
                url,
                terminal_event,
                raw_line_count,
                json_error_count,
                _preview_bytes(first_raw_line) if first_raw_line is not None else "-",
                _preview_bytes(first_json_error_line) if first_json_error_line is not None else "-",
                _elapsed_ms(started_at),
            )


# ── 例外クラス ────────────────────────────────────────────────────────────────

class UpstreamError(Exception):
    """アップストリームサーバーが 4xx/5xx エラーを返した場合に送出される例外です。"""

    def __init__(self, status: int, body: bytes, headers: dict[str, str]):
        self.status = status
        self.body = body
        self.headers = headers
        preview = body.decode("utf-8", errors="replace")
        super().__init__(f"upstream {status}: {preview[:200]}")


class RetryNeeded(Exception):
    """リトライが必要な状態を示すセンチネル例外です。"""
    pass


class ChunkTimeoutError(Exception):
    """チャンク途絶タイムアウトを示す例外です。"""
    pass


class ReasoningTimeoutError(Exception):
    """Reasoning タイムアウトを示す例外です。"""
    pass


class GenerationTimeoutError(Exception):
    """生成タイムアウトを示す例外です。"""
    pass


# ── リトライループ ────────────────────────────────────────────────────────────

async def _anext_or_none(gen: AsyncGenerator) -> Any:
    """StopAsyncIteration を None に変換します。"""
    try:
        return await gen.__anext__()
    except StopAsyncIteration:
        return None


async def _iter_with_timeouts(
    gen: AsyncGenerator,
    chunk_timeout: float,
    reasoning_timeout: float,
    generation_timeout: float,
    request_id: str = "-",
) -> AsyncGenerator:
    """タイムアウト監視付きストリームラッパーです。

    以下の 3 種類のタイムアウトを監視します。

     1. チャンク途絶タイムアウト: 最初の意味のあるチャンクを受信した後に、
         意味のあるチャンクが chunk_timeout 秒以上届かない場合。
         「意味のある」とは reasoning_content または content に非空文字列を含む場合です。
    2. Reasoning タイムアウト: Reasoning フェーズが reasoning_timeout 秒を超えた場合。
    3. 生成タイムアウト: コンテンツ生成フェーズが generation_timeout 秒を超えた場合。

    タイムアウト時は対応する例外を送出します。
    timeout 値が 0 以下の場合は、そのタイムアウトは無効になります。
    """
    # 最初の意味のあるチャンクを受信するまではチャンク途絶タイマーを開始しません。
    last_meaningful_at: float | None = None
    reasoning_started_at: float | None = None
    generation_started_at: float | None = None
    try:
        while True:
            now = time.monotonic()

            # チャンク途絶タイムアウト: 次のチャンクを受信する（タイムアウト付き）
            if chunk_timeout > 0 and last_meaningful_at is not None:
                try:
                    item = await asyncio.wait_for(
                        _anext_or_none(gen), timeout=chunk_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[%s] chunk timeout: no chunk arrived for %.1fs",
                        request_id,
                        chunk_timeout,
                    )
                    raise ChunkTimeoutError()
            else:
                item = await _anext_or_none(gen)

            if item is None:
                # ストリームが正常終了しました。
                return

            ev, line, obj = item
            now = time.monotonic()

            if ev == "chunk":
                chunk_obj = cast(dict, obj)
                has_reasoning = chunk_has_reasoning(chunk_obj)
                has_content = chunk_has_content(chunk_obj)
                is_meaningful = has_reasoning or has_content

                if has_reasoning and reasoning_started_at is None:
                    reasoning_started_at = now
                if has_content and generation_started_at is None:
                    generation_started_at = now

                if is_meaningful:
                    last_meaningful_at = now
                else:
                    # 意味のないチャンクが届き続けている場合もチェック
                    if (
                        chunk_timeout > 0
                        and last_meaningful_at is not None
                        and (now - last_meaningful_at) > chunk_timeout
                    ):
                        logger.warning(
                            "[%s] chunk timeout: only empty chunks for %.1fs",
                            request_id,
                            now - last_meaningful_at,
                        )
                        raise ChunkTimeoutError()

            # Reasoning タイムアウトチェック（Reasoning フェーズ中のみ）
            if (
                reasoning_timeout > 0
                and reasoning_started_at is not None
                and generation_started_at is None
            ):
                elapsed = now - reasoning_started_at
                if elapsed > reasoning_timeout:
                    logger.warning(
                        "[%s] reasoning timeout: %.1fs elapsed (limit=%.1fs)",
                        request_id,
                        elapsed,
                        reasoning_timeout,
                    )
                    raise ReasoningTimeoutError()

            # 生成タイムアウトチェック（コンテンツ生成フェーズ中のみ）
            if generation_timeout > 0 and generation_started_at is not None:
                elapsed = now - generation_started_at
                if elapsed > generation_timeout:
                    logger.warning(
                        "[%s] generation timeout: %.1fs elapsed (limit=%.1fs)",
                        request_id,
                        elapsed,
                        generation_timeout,
                    )
                    raise GenerationTimeoutError()

            yield ev, line, obj
    finally:
        await gen.aclose()


async def run_with_retries(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    original_body: dict,
    client_streaming: bool,
    params: tuple[tuple[str, str], ...] | None = None,
    request_id: str = "-",
) -> (
    tuple[Literal["error"], int, bytes, dict[str, str]]
    | tuple[Literal["stream"], AsyncGenerator[bytes, None]]
    | tuple[Literal["full"], dict]
):
    """推論コンテンツが得られるまでリトライループを駆動します。

    戻り値:
      ("stream", async_generator) — クライアントがストリーミングを要求した場合
      ("full",   dict)            — クライアントが非ストリーミングを要求した場合
            ("error",  status, body, headers) — upstream または proxy 側のエラーが発生した場合
    """
    retry_count = 0
    started_at = time.perf_counter()
    while True:
        # 最大リトライ回数を超えた場合はエラーを返します。
        if retry_count > MAX_RETRIES:
            logger.warning(
                "[%s] retries exhausted max_retries=%s upstream_url=%s duration_ms=%s",
                request_id,
                MAX_RETRIES,
                url,
                _elapsed_ms(started_at),
            )
            return (
                "error",
                500,
                json.dumps(
                    {"error": {"message": "Exceeded max retries to force reasoning content"}}
                ).encode("utf-8"),
                {"content-type": "application/json"},
            )

        # リトライ回数に応じたリクエストボディを生成します。
        body = build_retry_payload(original_body, retry_count)
        attempt_started_at = time.perf_counter()
        logger.info(
            "[%s] chat attempt=%s client_streaming=%s upstream_url=%s",
            request_id,
            retry_count + 1,
            client_streaming,
            url,
        )

        all_chunks: list[dict] = []  # 非ストリーミング組み立て用バッファ
        saw_reasoning = False
        cancel_retry = False
        retry_reason: str | None = None

        # ストリーミングクライアントの場合:
        # すでにバイトを送信し始めた後でリトライを決定することはできません。
        # そのため BUFFER_CHUNKS 分のチャンクを先読みして推論の有無を確認し、
        # 問題なければそのまま残りをストリーミングします。

        if client_streaming:
            # ストリーミングクライアント向けの処理を実行します。
            result = await _attempt_streaming_client(
                client, url, headers, body, params, request_id
            )
            if result[0] == "retry":
                logger.info(
                    "[%s] retrying attempt=%s reason=%s duration_ms=%s",
                    request_id,
                    retry_count + 1,
                    result[1],
                    _elapsed_ms(attempt_started_at),
                )
                retry_count += 1
                continue
            if result[0] == "error":
                logger.warning(
                    "[%s] chat attempt failed attempt=%s status=%s duration_ms=%s",
                    request_id,
                    retry_count + 1,
                    result[1],
                    _elapsed_ms(attempt_started_at),
                )
                return ("error", result[1], result[2], result[3])
            # コミット済み: ストリームジェネレータをそのまま返します。
            logger.info(
                "[%s] chat attempts complete mode=stream attempts=%s duration_ms=%s",
                request_id,
                retry_count + 1,
                _elapsed_ms(started_at),
            )
            return ("stream", result[1])
        else:
            # 非ストリーミングクライアントの場合: 全チャンクを収集してから組み立てます。
            _ns_gen = _iter_with_timeouts(
                stream_attempt(client, url, headers, body, params, request_id),
                CHUNK_TIMEOUT,
                REASONING_TIMEOUT,
                GENERATION_TIMEOUT,
                request_id,
            )
            try:
                async for ev, line, obj in _ns_gen:
                    if ev == "chunk":
                        chunk_obj = cast(dict, obj)
                        all_chunks.append(chunk_obj)
                        if chunk_has_reasoning(chunk_obj):
                            saw_reasoning = True
                        # 推論なしにコンテンツが届いた場合はリトライします。
                        if not saw_reasoning and chunk_has_content(chunk_obj):
                            cancel_retry = True
                            retry_reason = "content_before_reasoning"
                            break
                        # 推論なしにストリームが終了した場合もリトライします。
                        if chunk_is_finish(chunk_obj) and not saw_reasoning:
                            cancel_retry = True
                            retry_reason = "finish_before_reasoning"
                            break
                    elif ev == "done":
                        if not saw_reasoning:
                            cancel_retry = True
                            retry_reason = "done_before_reasoning"
                        break
                    elif ev == "end":
                        if not saw_reasoning:
                            cancel_retry = True
                            retry_reason = "end_before_reasoning"
                        break
                    else:
                        # raw 行は非ストリーミング組み立てには不要なため無視します。
                        pass
            except (ChunkTimeoutError, ReasoningTimeoutError, GenerationTimeoutError) as e:
                cancel_retry = True
                retry_reason = type(e).__name__
            except UpstreamError as e:
                return ("error", e.status, e.body, e.headers)
            except httpx.HTTPError as e:
                return (
                    "error",
                    502,
                    json.dumps(
                        {"error": {"message": f"upstream connection error: {e}"}}
                    ).encode("utf-8"),
                    {"content-type": "application/json"},
                )
            finally:
                # ジェネレータを確実にクローズしてリソースを解放します。
                await _ns_gen.aclose()

            if cancel_retry:
                logger.info(
                    "[%s] retrying attempt=%s reason=%s chunk_count=%s duration_ms=%s",
                    request_id,
                    retry_count + 1,
                    retry_reason or "unknown",
                    len(all_chunks),
                    _elapsed_ms(attempt_started_at),
                )
                retry_count += 1
                continue
            # 成功: 収集したチャンクから非ストリーミングレスポンスを組み立てます。
            assembled = assemble_non_streaming_response(
                all_chunks, original_body.get("model")
            )
            logger.info(
                "[%s] chat attempts complete mode=full attempts=%s chunk_count=%s saw_reasoning=%s duration_ms=%s",
                request_id,
                retry_count + 1,
                len(all_chunks),
                saw_reasoning,
                _elapsed_ms(started_at),
            )
            return ("full", assembled)


# ── ストリーミングクライアント向け処理 ──────────────────────────────────────

async def _attempt_streaming_client(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    body: dict,
    params: tuple[tuple[str, str], ...] | None = None,
    request_id: str = "-",
) -> (
    tuple[Literal["retry"], str]
    | tuple[Literal["error"], int, bytes, dict[str, str]]
    | tuple[Literal["commit"], AsyncGenerator[bytes, None]]
):
    """ストリーミングクライアント向けの 1 回の接続試行を行います。

    戻り値:
      ("retry",)                 — リトライが必要
    ("error",  status, body, headers) — エラー
      ("commit", async_gen)      — コミット完了。gen が SSE バイトを生成します

    BUFFER_CHUNKS 分のチャンクを先読みし、推論の有無を確認します。
    問題が検出された場合はストリームをキャンセルしてリトライを指示します。
    問題がなければバッファの内容を流し、残りのストリームを継続します。
    """
    gen = _iter_with_timeouts(
        stream_attempt(client, url, headers, body, params, request_id),
        CHUNK_TIMEOUT,
        REASONING_TIMEOUT,
        GENERATION_TIMEOUT,
        request_id,
    )
    buffered_lines: list[bytes] = []
    saw_reasoning = False
    chunk_count = 0
    stream_ended = False
    ended_with_done = False
    started_at = time.perf_counter()

    try:
        # バッファリングフェーズ: 判定に必要なチャンクを先読みします。
        async for ev, line, obj in gen:
            if ev == "raw":
                # 未知の SSE フィールドはそのままバッファに追加します。
                buffered_lines.append(cast(bytes, line))
                continue
            if ev == "done":
                if not saw_reasoning:
                    await gen.aclose()
                    logger.info(
                        "[%s] stream retry trigger reason=done_before_reasoning chunk=%s buffered_lines=%s duration_ms=%s",
                        request_id,
                        chunk_count,
                        len(buffered_lines),
                        _elapsed_ms(started_at),
                    )
                    return ("retry", "done_before_reasoning")
                # ストリーム終了センチネルをバッファに記録して終了します。
                buffered_lines.append(b"data: [DONE]")
                ended_with_done = True
                stream_ended = True
                break
            if ev == "end":
                if not saw_reasoning:
                    await gen.aclose()
                    logger.info(
                        "[%s] stream retry trigger reason=end_before_reasoning chunk=%s buffered_lines=%s duration_ms=%s",
                        request_id,
                        chunk_count,
                        len(buffered_lines),
                        _elapsed_ms(started_at),
                    )
                    return ("retry", "end_before_reasoning")
                stream_ended = True
                break
            if ev == "chunk":
                chunk_count += 1
                chunk_obj = cast(dict, obj)
                if chunk_has_reasoning(chunk_obj):
                    if not saw_reasoning:
                        logger.info(
                            "[%s] stream reasoning detected chunk=%s buffered_lines=%s duration_ms=%s",
                            request_id,
                            chunk_count,
                            len(buffered_lines),
                            _elapsed_ms(started_at),
                        )
                    saw_reasoning = True
                # 推論なしにコンテンツが届いた場合は不正な状態です。リトライします。
                if not saw_reasoning and chunk_has_content(chunk_obj):
                    await gen.aclose()
                    logger.info(
                        "[%s] stream retry trigger reason=content_before_reasoning chunk=%s buffered_lines=%s duration_ms=%s",
                        request_id,
                        chunk_count,
                        len(buffered_lines),
                        _elapsed_ms(started_at),
                    )
                    return ("retry", "content_before_reasoning")
                # 推論なしにストリームが終了した場合もリトライします。
                if chunk_is_finish(chunk_obj) and not saw_reasoning:
                    await gen.aclose()
                    logger.info(
                        "[%s] stream retry trigger reason=finish_before_reasoning chunk=%s buffered_lines=%s duration_ms=%s",
                        request_id,
                        chunk_count,
                        len(buffered_lines),
                        _elapsed_ms(started_at),
                    )
                    return ("retry", "finish_before_reasoning")
                buffered_lines.append(cast(bytes, line))
                # 最初の BUFFER_CHUNKS チャンクは必ずバッファし、その後も推論確認まではコミットしません。
                if chunk_count > BUFFER_CHUNKS and saw_reasoning:
                    # 推論が確認されたのでコミットします。
                    logger.info(
                        "[%s] stream commit ready chunk_count=%s buffered_lines=%s duration_ms=%s",
                        request_id,
                        chunk_count,
                        len(buffered_lines),
                        _elapsed_ms(started_at),
                    )
                    break
    except (ChunkTimeoutError, ReasoningTimeoutError, GenerationTimeoutError) as e:
        await gen.aclose()
        timeout_reason = type(e).__name__
        logger.info(
            "[%s] stream retry trigger reason=%s chunk=%s buffered_lines=%s duration_ms=%s",
            request_id,
            timeout_reason,
            chunk_count,
            len(buffered_lines),
            _elapsed_ms(started_at),
        )
        return ("retry", timeout_reason)
    except UpstreamError as e:
        await gen.aclose()
        return ("error", e.status, e.body, e.headers)
    except httpx.HTTPError as e:
        await gen.aclose()
        return (
            "error",
            502,
            json.dumps(
                {"error": {"message": f"upstream connection error: {e}"}}
            ).encode("utf-8"),
            {"content-type": "application/json"},
        )

    # バッファウィンドウ内でストリームが終了した場合:
    # 推論が確認されていれば正常終了です。バッファ内容のみを送信します。
    # 推論なしで終了した場合は上のチェックで既にリトライが発生しています。
    if stream_ended:
        logger.info(
            "[%s] stream ended before commit chunk_count=%s buffered_lines=%s saw_reasoning=%s ended_with_done=%s duration_ms=%s",
            request_id,
            chunk_count,
            len(buffered_lines),
            saw_reasoning,
            ended_with_done,
            _elapsed_ms(started_at),
        )

        async def short_gen():
            for ln in buffered_lines:
                yield ln + b"\n\n"
            # [DONE] が含まれていない場合は補完します。
            if not ended_with_done:
                yield b"data: [DONE]\n\n"
        return ("commit", short_gen())

    # ストリームの途中でコミット: バッファを送信後、残りのストリームを転送します。
    async def forward_gen():
        forward_started_at = time.perf_counter()
        try:
            # バッファに蓄積した先読みチャンクを先に送信します。
            for ln in buffered_lines:
                yield ln + b"\n\n"
            # 残りのストリームをそのまま転送します。
            async for ev, line, obj in gen:
                if ev == "chunk" or ev == "raw":
                    yield cast(bytes, line) + b"\n\n"
                elif ev == "done":
                    logger.info(
                        "[%s] stream forward completed end_reason=done duration_ms=%s",
                        request_id,
                        _elapsed_ms(forward_started_at),
                    )
                    yield b"data: [DONE]\n\n"
                    return
                elif ev == "end":
                    logger.warning(
                        "[%s] stream forward completed end_reason=end_without_done duration_ms=%s",
                        request_id,
                        _elapsed_ms(forward_started_at),
                    )
                    yield b"data: [DONE]\n\n"
                    return
        except asyncio.CancelledError:
            logger.info(
                "[%s] stream forward cancelled duration_ms=%s",
                request_id,
                _elapsed_ms(forward_started_at),
            )
            raise
        except (ChunkTimeoutError, ReasoningTimeoutError, GenerationTimeoutError) as e:
            logger.warning(
                "[%s] stream forward timed out reason=%s duration_ms=%s",
                request_id,
                type(e).__name__,
                _elapsed_ms(forward_started_at),
            )
            yield format_sse_comment(
                f"proxy stream cancelled after timeout: {type(e).__name__}"
            ) + b"\n\n"
            yield b"data: [DONE]\n\n"
            return
        except Exception:
            logger.exception(
                "[%s] stream forward failed duration_ms=%s",
                request_id,
                _elapsed_ms(forward_started_at),
            )
            raise
        finally:
            # 確実にジェネレータをクローズしてリソースを解放します。
            await gen.aclose()

    return ("commit", forward_gen())


# ── ヘッダーフィルタリング ────────────────────────────────────────────────────

def _filter_hop_headers(headers: dict) -> dict:
    """HTTP ホップバイホップヘッダーを除去し、アップストリームへ転送するヘッダーを返します。

    これらのヘッダーはプロキシ間の通信にのみ意味を持ち、
    エンドツーエンドで転送すべきではありません。
    """
    # RFC 2616 で定義されているホップバイホップヘッダーのセットです。
    hop = {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "accept-encoding",
    }
    return {k: v for k, v in headers.items() if k.lower() not in hop}


def _filter_response_headers(headers: dict) -> dict:
    """プロキシがそのまま返すのに不適切なレスポンスヘッダーを除去します。"""
    excluded = {"content-encoding", "transfer-encoding", "content-length", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in excluded}


def _build_upstream_url(full_path: str) -> str:
    """受信パスの /v1 有無を保ったままアップストリーム URL を構築します。"""
    base = UPSTREAM_BASE_URL.rstrip("/")
    upstream_root = base[:-3] if base.endswith("/v1") else base
    normalized_path = full_path.lstrip("/")
    if not normalized_path:
        return upstream_root
    return f"{upstream_root}/{normalized_path}"


# ── FastAPI エンドポイント ────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    """チャット補完リクエストを受け取り、アップストリームへ転送します。

    /v1/chat/completions と /chat/completions の両方のパスに対応します。
    推論コンテンツが得られない場合は自動的にリトライします。
    """
    request_id = _make_request_id(request)
    request_started_at = time.perf_counter()

    # リクエストボディを JSON として解析します。
    try:
        body = await request.json()
    except Exception:
        logger.warning("[%s] invalid JSON body path=%s", request_id, request.url.path)
        return _with_request_id_header(
            JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400),
            request_id,
        )

    if not isinstance(body, dict):
        logger.warning("[%s] invalid JSON type path=%s type=%s", request_id, request.url.path, type(body).__name__)
        return _with_request_id_header(
            JSONResponse(
                {"error": {"message": "body must be a JSON object"}}, status_code=400
            ),
            request_id,
        )

    # クライアントがストリーミングを要求しているか確認します。
    client_streaming = bool(body.get("stream", False))

    # --model 引数が指定されている場合はモデル名を上書きします。
    if FORCE_MODEL is not None:
        body["model"] = FORCE_MODEL

    # 古い reasoning_content / reasoning を削除します（連続的推論サポート）。
    if KEEP_REASONING_N >= 0 and isinstance(body.get("messages"), list):
        body["messages"] = _trim_old_reasoning(body["messages"], KEEP_REASONING_N)

    # ホップバイホップヘッダーを除去してからアップストリームへ転送するヘッダーを構築します。
    fwd_headers = _filter_hop_headers(dict(request.headers.items()))
    fwd_headers["accept"] = "text/event-stream"
    fwd_headers["content-type"] = "application/json"

    # 転送先 URL を組み立てます。
    url = _build_upstream_url(request.url.path)
    params = tuple(request.query_params.multi_items())
    logger.info(
        "[%s] chat request start path=%s client=%s stream=%s model=%s upstream_url=%s summary=%s",
        request_id,
        request.url.path,
        request.client.host if request.client is not None else "-",
        client_streaming,
        body.get("model"),
        url,
        _summarize_chat_body(body),
    )

    client = request.app.state.http_client
    try:
        outcome = await run_with_retries(
            client, url, fwd_headers, body, client_streaming, params, request_id
        )
    except Exception as e:
        logger.exception("[%s] chat request crashed path=%s", request_id, request.url.path)
        return _with_request_id_header(
            JSONResponse(
                {"error": {"message": f"proxy error: {e}"}}, status_code=500
            ),
            request_id,
        )

    # エラーの場合は、upstream の応答か proxy 自身の生成エラーをそのまま返します。
    if outcome[0] == "error":
        _, status, body_bytes, headers = outcome
        logger.warning(
            "[%s] chat request returning error status=%s path=%s duration_ms=%s",
            request_id,
            status,
            request.url.path,
            _elapsed_ms(request_started_at),
        )
        return _with_request_id_header(
            Response(content=body_bytes, status_code=status, headers=headers),
            request_id,
        )

    # 非ストリーミング: 組み立て済みのレスポンス JSON をそのまま返します。
    if outcome[0] == "full":
        logger.info(
            "[%s] chat request completed mode=full path=%s choices=%s duration_ms=%s",
            request_id,
            request.url.path,
            len(outcome[1].get("choices") or []),
            _elapsed_ms(request_started_at),
        )
        return _with_request_id_header(JSONResponse(outcome[1]), request_id)

    # ストリーミング: SSE バイトをクライアントへ逐次送信します。
    if outcome[0] == "stream":
        gen = outcome[1]
        logger.info(
            "[%s] chat request committed mode=stream path=%s duration_ms=%s",
            request_id,
            request.url.path,
            _elapsed_ms(request_started_at),
        )

        async def wrapper():
            """ジェネレータをラップし、終了時に確実にクローズします。"""
            try:
                async for chunk in gen:
                    yield chunk
            except asyncio.CancelledError:
                logger.info(
                    "[%s] client disconnected while streaming path=%s",
                    request_id,
                    request.url.path,
                )
                raise
            except Exception:
                logger.exception(
                    "[%s] streaming response failed path=%s",
                    request_id,
                    request.url.path,
                )
                raise
            finally:
                # クライアントが切断した場合もリソースを解放します。
                aclose = getattr(gen, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        logger.exception(
                            "[%s] streaming generator close failed path=%s",
                            request_id,
                            request.url.path,
                        )

        return _with_request_id_header(
            StreamingResponse(
                wrapper(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",   # キャッシュを無効化します。
                    "Connection": "keep-alive",     # 接続を維持します。
                    "X-Accel-Buffering": "no",      # nginx のバッファリングを無効化します。
                },
            ),
            request_id,
        )

    # 想定外の結果タイプが返された場合のフォールバックです。
    logger.error("[%s] unknown proxy outcome path=%s", request_id, request.url.path)
    return _with_request_id_header(
        JSONResponse({"error": {"message": "unknown proxy outcome"}}, status_code=500),
        request_id,
    )


# Pass-through endpoints commonly used (models list etc.)
@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def passthrough(full_path: str, request: Request):
    request_id = _make_request_id(request)
    request_started_at = time.perf_counter()

    # Don't intercept the chat completions route (handled above)
    if full_path in ("v1/chat/completions", "chat/completions"):
        # Shouldn't reach here due to specific routes above, but safety
        logger.error(
            "[%s] passthrough hit guarded chat route path=%s",
            request_id,
            request.url.path,
        )
        return _with_request_id_header(
            JSONResponse({"error": {"message": "internal routing error"}}, status_code=500),
            request_id,
        )

    url = _build_upstream_url(full_path)

    method = request.method
    body_bytes = await request.body()
    fwd_headers = _filter_hop_headers(dict(request.headers.items()))
    logger.info(
        "[%s] passthrough start method=%s path=%s upstream_url=%s",
        request_id,
        method,
        request.url.path,
        url,
    )

    client = request.app.state.http_client
    try:
        resp = await client.request(
            method,
            url,
            headers=fwd_headers,
            content=body_bytes,
            params=tuple(request.query_params.multi_items()),
            timeout=None,
        )
    except httpx.HTTPError as e:
        logger.warning(
            "[%s] passthrough upstream connection error method=%s path=%s error=%s duration_ms=%s",
            request_id,
            method,
            request.url.path,
            e,
            _elapsed_ms(request_started_at),
        )
        return _with_request_id_header(
            JSONResponse(
                {"error": {"message": f"upstream connection error: {e}"}},
                status_code=502,
            ),
            request_id,
        )

    resp_headers = _filter_response_headers(dict(resp.headers.items()))
    log = logger.warning if resp.status_code >= 400 else logger.info
    log(
        "[%s] passthrough response method=%s path=%s upstream_status=%s upstream_url=%s content_type=%s duration_ms=%s body=%s",
        request_id,
        method,
        request.url.path,
        resp.status_code,
        url,
        resp.headers.get("content-type", "-"),
        _elapsed_ms(request_started_at),
        _preview_bytes(resp.content) if resp.status_code >= 400 else "-",
    )

    return _with_request_id_header(
        Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=resp.headers.get("content-type"),
        ),
        request_id,
    )


if __name__ == "__main__":
    import uvicorn

    # ── コマンドライン引数の定義 ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="force_reasoning_proxy — 推論を強制するリバースプロキシ"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="リッスンするホスト（デフォルト: 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="リッスンするポート番号（デフォルト: 8000）",
    )
    parser.add_argument(
        "--upstream",
        default=UPSTREAM_BASE_URL,
        help=f"アップストリームサーバーのベース URL（デフォルト: {UPSTREAM_BASE_URL}）",
    )
    parser.add_argument(
        "--buffer-chunks",
        type=int,
        default=BUFFER_CHUNKS,
        help=f"ストリーミング時の先読みチャンク数（デフォルト: {BUFFER_CHUNKS}）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"最大リトライ回数（デフォルト: {MAX_RETRIES}）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="クライアントのモデル名を常にこの値で上書きします（省略時は上書きしない）",
    )
    parser.add_argument(
        "--keep-reasoning",
        type=int,
        default=KEEP_REASONING_N,
        help=f"保持する reasoning_content / reasoning の最大件数（デフォルト: {KEEP_REASONING_N}、0 はすべて削除、負の値は無制限）",
    )
    parser.add_argument(
        "--chunk-timeout",
        type=float,
        default=CHUNK_TIMEOUT,
        help=f"チャンク途絶タイムアウト（秒）。意味のあるチャンクがこの秒数届かない場合にリトライします（デフォルト: {CHUNK_TIMEOUT}、0 以下で無効）",
    )
    parser.add_argument(
        "--reasoning-timeout",
        type=float,
        default=REASONING_TIMEOUT,
        help=f"Reasoning タイムアウト（秒）。Reasoning フェーズがこの秒数を超えた場合にリトライします（デフォルト: {REASONING_TIMEOUT}、0 以下で無効）",
    )
    parser.add_argument(
        "--generation-timeout",
        type=float,
        default=GENERATION_TIMEOUT,
        help=f"生成タイムアウト（秒）。コンテンツ生成フェーズがこの秒数を超えた場合にリトライします（デフォルト: {GENERATION_TIMEOUT}、0 以下で無効）",
    )
    args = parser.parse_args()

    # 引数でグローバル設定を上書きします。
    UPSTREAM_BASE_URL = args.upstream
    BUFFER_CHUNKS = args.buffer_chunks
    MAX_RETRIES = args.max_retries
    FORCE_MODEL = args.model
    KEEP_REASONING_N = args.keep_reasoning
    CHUNK_TIMEOUT = args.chunk_timeout
    REASONING_TIMEOUT = args.reasoning_timeout
    GENERATION_TIMEOUT = args.generation_timeout

    uvicorn.run(app, host=args.host, port=args.port)
