#!/usr/bin/env bash
set -euo pipefail

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv-lm-7.2b",
    "messages": [{"role": "user", "content": "你好，做个自我介绍。"}],
    "stream": false,
    "max_tokens": 256,
    "temperature": 1.0,
    "top_k": 500,
    "top_p": 0.3
  }'
