#!/usr/bin/env bash
set -euo pipefail

: "${PREFILL_HOST:?请设置 Prefill 监听地址 PREFILL_HOST}"
: "${PREFILL_PORT:?请设置 Prefill 服务端口 PREFILL_PORT}"
: "${PREFILL_BOOTSTRAP_PORT:?请设置 Prefill bootstrap 端口 PREFILL_BOOTSTRAP_PORT}"

source "$(dirname -- "${BASH_SOURCE[0]}")/server_log.sh" prefill
export SGLANG_MOONCAKE_DCP_PACK_BUFFER_MB=${SGLANG_MOONCAKE_DCP_PACK_BUFFER_MB:-128}
export SGLANG_MOONCAKE_SEND_AUX_TCP=1
GLOG_logtostderr=1 \
GLOG_minloglevel=1 \
SGLANG_USE_BREAKABLE_CUDA_GRAPH=1 \
SGLANG_K3_ATTN_RES_MODE=jit \
SGLANG_MOE_FUSED_GATE_RADIX=1 \
SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 \
SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=16 \
SGLANG_DISAGGREGATION_QUEUE_SIZE=4 \
UCX_TLS=rc,cuda_copy,cuda_ipc,self,sm \
MOONCAKE_GLOBAL_SEGMENT_SIZE=0 \
SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1 \
nohup python -m sglang.launch_server \
    --model-path /model/Kimi-K3 \
    --served-model-name base_model \
    --attention-backend cutedsl_mla \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port "$PREFILL_BOOTSTRAP_PORT" \
    --host "$PREFILL_HOST" \
    --port "$PREFILL_PORT" \
    --cuda-graph-backend-prefill breakable \
    --enable-symm-mem \
    --enable-hierarchical-cache \
    --hicache-ratio 1.5 \
    --hicache-mem-layout page_first_direct \
    --trust-remote-code \
    --tp-size 1 \
    --pp-size 8 \
    --page-size 64 \
    --mem-fraction-static 0.85 \
    --mamba-full-memory-ratio 1.02 \
    --chunked-prefill-size 16384 \
    --max-prefill-tokens 16384 \
    --context-length 1048576 \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-ssm-dtype bfloat16 \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --disaggregation-transfer-backend mooncake \
    --load-format runai_streamer \
    --disaggregation-ib-device mlx5_000,mlx5_001,mlx5_002,mlx5_003,mlx5_004,mlx5_005,mlx5_006,mlx5_007,mlx5_008,mlx5_009,mlx5_010,mlx5_011,mlx5_012,mlx5_013,mlx5_014,mlx5_015 \
    --max-running-requests 64 \
    --mm-feature-transport cuda_ipc \
    >> "$SERVER_LOG" 2>&1 &
