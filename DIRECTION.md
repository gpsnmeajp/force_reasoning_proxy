作成するもの
+ Pythonで実装
+ OpenAI互換の chat completion　API Proxy
+ non-streaming, streaming両対応
+ すべてのリクエストはパススルーする(未知のパラメータも含めて)
+ すべての応答はパススルーする(未知のパラメータも含めて)

但し、以下の挙動をする。

+ 現在LLM側でReasoningが確率的スキップされる問題が発生している。
+ このProxyは、常にstreamingで要求を出し、仮にクライアントがnon-streamingの場合は、LLM応答を受領完了してからnon-streamingとして応答する。
+ このProxyは、streamingを監視し、reasoning_contentが生成されないままcontentの生成が始まった場合、即座にキャンセルし、リトライを行う。(そのため、最初の5回までのstreamingはバッファし、5回を超えた時点でクライアントに送信を始める。もちろんリトライ時にはバッファをクリアする)
+ リトライ回数が5回を超えた場合、リクエストの最後のuser roleのcontentに、"\n<|think|>"という文字列を、以降のリトライ回数分追加してリトライする。
+ リトライ回数が100回を超えた場合、中止し、クライアントには500エラーを返す。
+ これはローカルLLM環境で使用するため、リトライ間隔は0秒で良い。

# 正しく処理が始まる例

```
data: {"choices":[{"finish_reason":null,"index":0,"delta":{"role":"assistant","content":null}}],"created":1779518272,"id":"chatcmpl-hLtRN5vT0if18oOj9MrOcL0I6oNf0Jh4","model":"unsloth/gemma-4-26B-A4B-it-GGUF:Q6_K","system_fingerprint":"b9254-e94722822","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":"*"}}],"created":1779518272,"id":"chatcmpl-hLtRN5vT0if18oOj9MrOcL0I6oNf0Jh4","model":"unsloth/gemma-4-26B-A4B-it-GGUF:Q6_K","system_fingerprint":"b9254-e94722822","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":"   "}}],"created":1779518272,"id":"chatcmpl-hLtRN5vT0if18oOj9MrOcL0I6oNf0Jh4","model":"unsloth/gemma-4-26B-A4B-it-GGUF:Q6_K","system_fingerprint":"b9254-e94722822","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":"Current"}}],"created":1779518272,"id":"chatcmpl-hLtRN5vT0if18oOj9MrOcL0I6oNf0Jh4","model":"unsloth/gemma-4-26B-A4B-it-GGUF:Q6_K","system_fingerprint":"b9254-e94722822","object":"chat.completion.chunk"}

...

# Reasoningがスキップされる例

```

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"role":"assistant","content":null}}],"created":1779518360,"id":"chatcmpl-c2PLrEaGTr4itu907yXwzGqtp9Hl787w","model":"unsloth/gemma-4-26B-A4B-it-GGUF:Q6_K","system_fingerprint":"b9254-e94722822","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"{"}}],"created":1779518360,"id":"chatcmpl-c2PLrEaGTr4itu907yXwzGqtp9Hl787w","model":"unsloth/gemma-4-26B-A4B-it-GGUF:Q6_K","system_fingerprint":"b9254-e94722822","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"\n"}}],"created":1779518360,"id":"chatcmpl-c2PLrEaGTr4itu907yXwzGqtp9Hl787w","model":"unsloth/gemma-4-26B-A4B-it-GGUF:Q6_K","system_fingerprint":"b9254-e94722822","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"\t"}}],"created":1779518360,"id":"chatcmpl-c2PLrEaGTr4itu907yXwzGqtp9Hl787w","model":"unsloth/gemma-4-26B-A4B-it-GGUF:Q6_K","system_fingerprint":"b9254-e94722822","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"\""}}],"created":1779518360,"id":"chatcmpl-c2PLrEaGTr4itu907yXwzGqtp9Hl787w","model":"unsloth/gemma-4-26B-A4B-it-GGUF:Q6_K","system_fingerprint":"b9254-e94722822","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"response"}}],"created":1779518360,"id":"chatcmpl-c2PLrEaGTr4itu907yXwzGqtp9Hl787w","model":"unsloth/gemma-4-26B-A4B-it-GGUF:Q6_K","system_fingerprint":"b9254-e94722822","object":"chat.completion.chunk"}

```

---

# 機能追加
クライアントからは、過去のすべてのreasoning_contentが送られてきます。(連続的推論サポート)
これを、最新N件を残してそれより過去を削除するようにしてください。(既定でN=5, 引数で設定可能)

```
    {
      "role": "user",
      "content": "xxxxxxxxxxxxxx"
    },
    {
      "role": "assistant",
      "content": "xxxxxxxxxxxxxxx",
      "reasoning_content": "xxxxxxxxxxxxxxxxxxxxxx",
      "reasoningTimeMs": 70222
    },
    {
      "role": "user",
      "content": "xxxxxxxxxxxxxxxxxxx"
    },
```

# チャンク途絶タイムアウト
推論の途中(Reasoningや最終生成中)に、生成が中断(スタック)することがあります。
途中で突如追加のチャンクが来なくなり(あるいは空のチャンクだけが来て)、N秒(既定でN=10, 引数で設定可能)経過した場合、生成を強制的にキャンセルしてください。

チャンクが再開した場合、経過時間はリセットします。

# Reasoningタイムアウトと、生成タイムアウト
Reasoningが長時間続いたり、生成が長時間続く場合があります。それぞれに対して秒数でタイムアウトを設定できるようにしてください。
タイムアウトした場合、生成を強制的にキャンセルしてください。(既定で600秒)
