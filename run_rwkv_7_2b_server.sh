#!/usr/bin/env bash
set -euo pipefail

export HOME=/public/home/ssjxusr
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export CUDARC_CUDA_VERSION=12080
export PATH=/public/home/ssjxusr/.cargo/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda-12.8/nvvm/lib64:/usr/local/lib:/usr/lib64:/usr/lib:/lib64:/lib

cd /public/home/ssjxusr/Asturian/rwkv-rs-optimized-opti

cargo run -p rwkv-lm --example rwkv-lm-infer --release --no-default-features --features inferring,cuda,trace --   --config-dir examples/rwkv-lm/config   --infer-cfg rwkv-lm-7.2b
