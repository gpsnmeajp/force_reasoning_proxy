# force_reasoning_proxy

> 日本語版: [README_ja.md](README_ja.md)

An OpenAI-compatible reverse proxy that solves the reasoning-skip problem common in local LLMs, especially Gemma4.

Some models (Gemma4 in particular) probabilistically skip `reasoning_content` generation and output `content` directly — especially when the system prompt or conversation history grows long — even when Thinking is explicitly enabled. In most cases, this results in outputs that merely mimic prior turns or produce corrupted responses, making such models unsuitable for reliability-critical use cases.  
Manual retries can work around this, but the probability of skipping reasoning increases as the conversation grows longer.

This proxy automatically retries when no reasoning content is detected, ensuring that every response returned to the client is accompanied by reasoning.

## How It Works

1. Forwards requests from the client to the upstream LLM server.
2. Always sends requests to the upstream using **streaming**, regardless of whether the client requested streaming. For non-streaming clients, the full response is buffered and returned as a single non-streaming response.
3. Monitors the stream; if `content` arrives before `reasoning_content` (or `reasoning`) is generated, the request is cancelled immediately and retried.
4. The first 5 chunks are held in a buffer. Once reasoning is confirmed, the buffer is flushed and streaming to the client begins (buffer is cleared on retry).
5. If the retry count exceeds **5**, a `<|think|>` token is appended to the end of the last user message on retry (one token added per 5 retries).
6. If the retry count exceeds **100**, the proxy aborts and returns a `500` error to the client.
7. Retry interval is 0 seconds (designed for local LLMs).

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
models:
  gemma4_proxy:
    cmd: |
      /home/{user}/force_reasoning_proxy/venv/bin/python3 /home/{user}/force_reasoning_proxy/proxy.py
      --port ${PORT}
      --upstream http://localhost:11400/
      --model gemma4-26B-A4B
```