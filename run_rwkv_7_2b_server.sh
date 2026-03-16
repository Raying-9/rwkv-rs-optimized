#!/usr/bin/env bash
set -euo pipefail

cd /home/asturian/rwkv-rs-optimized
source /home/asturian/use_cuda_12_8.sh >/dev/null

cargo run -p rwkv-lm --example rwkv-lm-infer --release --no-default-features --features inferring,cuda,trace -- \
  --config-dir examples/rwkv-lm/config \
  --infer-cfg rwkv-lm-7.2b
