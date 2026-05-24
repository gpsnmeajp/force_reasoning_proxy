# force_reasoning_proxy

> English version: [README.md](README.md)

Gemma4で主に発生する、ローカル LLM の推論（reasoning）スキップ問題を解決する OpenAI 互換のリバースプロキシです。

一部のモデル(特にGemma4)は特にシステムプロンプトやターン数が多くなってきた際、明示的にThinkingを指定しているにも関わらず、確率的に `reasoning_content` を生成せずに 直接 `content` を出力し始めることがあります。基本的にその場合、直前までの出力の模倣や、崩壊した出力になっていることが多く、信頼性の必要な用途には向きません。  
手動でリトライすることで対応はできますが、ターンが長くなればなるほど生成確率も低下していきます。

このプロキシは、推論コンテンツが確認できない場合に自動でリトライを行い、常に推論を伴う応答をクライアントに返します。

~~※力技です。それ実は設定で解決するよ、っていうのがありましたら誰か教えてください...~~

**追記: **

jinja templateを加工し、常に `<|channel>thought\n* ` から始めるようにしたところ、リトライがほぼ不要になるほど劇的に改善しました。  
一般にGemma 4は「*」から箇条書きを開始するのでこの形にしていますが、他にも思考の先頭に指示を挿入できるので、色々面白いです。  

これによりproxyがなくても安定してReasoningができるようになりましたが、proxyには他にも加工機能を追加していますので、保険 & 便利加工処理として使えます。

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


## 動作概要

1. クライアントからのリクエストをアップストリーム LLM サーバーへ転送します。
2. アップストリームへは**常にストリーミング**でリクエストを送信します。クライアントが non-streaming を要求した場合は、受信完了後にまとめて non-streaming レスポンスとして返します。
3. ストリームを監視し、`reasoning_content`（または `reasoning`）が生成される前に `content` が来た場合、即座にキャンセルしてリトライします。
4. 最初の 5 チャンクはバッファに保持し、推論の確認後にクライアントへの送信を開始します（リトライ時はバッファをクリア）。
5. リトライ回数が **5 回**を超えた場合、最後のユーザーメッセージ末尾に `<|think|>` トークンを付加してリトライします（5 回ごとに 1 トークン追加）。
6. リトライ回数が **100 回**を超えた場合は中止し、クライアントに `500` エラーを返します。
7. リトライ間隔は 0 秒です（ローカル LLM 向け）。
8. クライアントが送信した過去の `reasoning_content` / `reasoning`（連続的推論サポート）は、最新 **5 件**のみ保持し、それより古いものはアップストリームへの転送前に削除します（`--keep-reasoning` で件数を変更可能）。

## 必要要件

- upstreamは、llama.cppを想定
- Python 3.11 以上
- 依存ライブラリ（`requirements.txt` 参照）

```
fastapi
uvicorn
httpx
```

## インストール

```bash
pip install -r requirements.txt
```

## 起動

```bash
python proxy.py
```

### オプション

| オプション      | デフォルト値                    | 説明                                                     |
|---------------|-------------------------------|--------------------------------------------------------|
| `--upstream`  | `http://localhost:8080/`    | アップストリーム LLM サーバーのベース URL                  |
| `--host`      | `0.0.0.0`                     | バインドするホスト                                        |
| `--port`      | `8000`                        | バインドするポート                                        |
| `--model`     | （なし）                        | クライアントのモデル名をこの値に上書きする                  |
| `--keep-reasoning` | `5`                      | 保持する `reasoning_content` / `reasoning` の最大件数（0 はすべて削除、負の値は無制限） |

```bash
python proxy.py --upstream http://localhost:11434/ --port 8000
```

## 使い方

プロキシ起動後、OpenAI クライアントのエンドポイントをプロキシに向けるだけです。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # ローカル LLM のため任意の値で構いません
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