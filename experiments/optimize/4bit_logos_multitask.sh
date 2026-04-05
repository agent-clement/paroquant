#!/usr/bin/env bash
set -euo pipefail

shards=${1:-1}
train_size=${2:-1024}

python3 -m paroquant.cli.optimize \
    --model familiar-ai/logos-multitask-qwen3.5-2026-03-31-best \
    --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6" \
    --epochs 10 10 \
    --group-size 128 \
    --n-bit 4 \
    --num-rotations 8 \
    --datasets \
    "hf://familiar-ai/logos-approach-2026-03-26?split=train&text_key=prompt,answer&weight=8000&repeat=true" \
    "hf://familiar-ai/logos-observe-balanced-2026-03-27?split=train&text_key=input,output&weight=3600&repeat=true" \
    "hf://familiar-ai/Flux11?split=train&text_key=prompt&weight=11000&repeat=true" \
    --skipped-modules "linear_attn.in_proj_a" "linear_attn.in_proj_b" \
    --val-datasets \
    "hf://familiar-ai/logos-approach-2026-03-26?split=validation&text_key=prompt,answer&weight=8000" \
    "hf://familiar-ai/logos-observe-balanced-2026-03-27?split=eval&text_key=input,output&weight=3600" \
    "hf://familiar-ai/Flux11?split=validation&text_key=prompt&weight=11000" \
    --train-size "$train_size" \
    --validation-size 64 \
    --batch-size 16 \
    --seqlen 2048 \
    --cache-shards "$shards" \
    --output-dir ./output \
    --resume \
    --seed 0
