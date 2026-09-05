# force_reasoning_proxy

> 日本語版: [README_ja.md](README_ja.md)

An OpenAI-compatible reverse proxy that solves the reasoning-skip problem common in local LLMs, especially Gemma4.

Some models (Gemma4 in particular) probabilistically skip `reasoning_content` generation and output `content` directly — especially when the system prompt or conversation history grows long — even when Thinking is explicitly enabled. In most cases, this results in outputs that merely mimic prior turns or produce corrupted responses, making such models unsuitable for reliability-critical use cases.  
Manual retries can work around this, but the probability of skipping reasoning increases as the conversation grows longer.

This proxy automatically retries when no reasoning content is detected, ensuring that every response returned to the client is accompanied by reasoning.

~~This is a brute-force workaround. If there’s actually a setting that solves this, I’d appreciate it if someone could let me know...~~

**Update:**

By modifying the Jinja template to always start generation with `<|channel>thought\n* `, the improvement was dramatic — retries became almost unnecessary.  
Gemma 4 generally starts bullet points with `*`, so this format was chosen, but you can also insert other instructions at the beginning of the thought, which opens up interesting possibilities.

This means stable Reasoning is now achievable without the proxy, but since the proxy has other processing features added, it can still be used as a safety net and for convenient response transformations.

```gemma4_force_think_chat_template.jinja
{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- '<|turn>model\n' -}}
        {%- if not enable_thinking | default(false) -%}
            {{- '<|channel>thought\n<channel|>' -}}
        {%- else -%}
            {{- '<|channel>thought\n* ' -}}
        {%- endif -%}
    {%- endif -%}
{%- endif -%}
```

## Chat Template License

The chat templates in this repository are derived from Google Gemma4 and retain its Apache License 2.0.

## How It Works

1. Forwards requests from the client to the upstream LLM server.
2. Always sends requests to the upstream using **streaming**, regardless of whether the client requested streaming. For non-streaming clients, the full response is buffered and returned as a single non-streaming response.
3. Monitors the stream; if `content` arrives before `reasoning_content` (or `reasoning`) is generated, the request is cancelled immediately and retried.
4. The first 5 chunks are held in a buffer. Once reasoning is confirmed, the buffer is flushed and streaming to the client begins (buffer is cleared on retry).
5. If the retry count exceeds **5**, a `<|think|>` token is appended to the end of the last user message on retry (one token added per 5 retries).
6. If the retry count exceeds **100**, the proxy aborts and returns a `500` error to the client.
7. Retry interval is 0 seconds (designed for local LLMs).
8. Past `reasoning_content` or `reasoning` entries sent by the client (continuous reasoning support) are trimmed to keep only the latest **5** before forwarding to the upstream (configurable with `--keep-reasoning`).
9. **Chunk interruption timeout**: If no meaningful chunk (non-empty `reasoning_content` or `content`) arrives for a set duration (default **10 seconds**) during reasoning or generation, the request is cancelled and retried. Also handles the case where only empty chunks keep flowing. Configurable with `--chunk-timeout` (disabled when ≤ 0).
10. **Reasoning timeout**: If the reasoning phase exceeds a set duration (default **600 seconds**), the request is cancelled and retried. Configurable with `--reasoning-timeout` (disabled when ≤ 0).
11. **Generation timeout**: If the content generation phase exceeds a set duration (default **600 seconds**), the request is cancelled and retried. Configurable with `--generation-timeout` (disabled when ≤ 0).

## Requirements

- Upstream: llama.cpp is assumed
- Python 3.11 or higher
- Dependencies (see `requirements.txt`)

```
fastapi
uvicorn
httpx
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python proxy.py
```

### Options

| Option        | Default                       | Description                                              |
|---------------|-------------------------------|----------------------------------------------------------|
| `--upstream`  | `http://localhost:8080/`      | Base URL of the upstream LLM server                     |
| `--host`      | `0.0.0.0`                     | Host to bind                                             |
| `--port`      | `8000`                        | Port to bind                                             |
| `--model`     | (none)                        | Override the client's model name with this value         |
| `--keep-reasoning` | `5`                      | Maximum number of `reasoning_content` / `reasoning` entries to keep (0 removes all, negative keeps all) |
| `--chunk-timeout` | `10`                      | Chunk interruption timeout in seconds. Retries if no meaningful chunk arrives within this duration (disabled when ≤ 0) |
| `--reasoning-timeout` | `600`               | Reasoning phase timeout in seconds. Retries if reasoning exceeds this duration (disabled when ≤ 0) |
| `--generation-timeout` | `600`              | Generation phase timeout in seconds. Retries if content generation exceeds this duration (disabled when ≤ 0) |

```bash
python proxy.py --upstream http://localhost:11434/ --port 8000
```

## Client Usage

After starting the proxy, simply point your OpenAI client's endpoint to the proxy.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # Any value works for local LLMs
)

response = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

llama-swap config example

```yaml
groups:
  "forever":
    persistent: true
    swap: false
    exclusive: false
    members:
      - "gemma4_proxy"

models:
  gemma4_proxy:
    cmd: |
      /home/{user}/force_reasoning_proxy/venv/bin/python3 /home/{user}/force_reasoning_proxy/proxy.py
      --port ${PORT}
      --upstream http://localhost:11400/
      --model gemma4-26B-A4B
```
