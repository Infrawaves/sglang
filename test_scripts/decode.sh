#!/usr/bin/env bash
set -euo pipefail

: "${DECODE_HOST:?请设置 Decode 监听地址 DECODE_HOST}"
: "${DECODE_PORT:?请设置 Decode 服务端口 DECODE_PORT}"
: "${DECODE_BOOTSTRAP_PORT:?请设置 Decode bootstrap 端口 DECODE_BOOTSTRAP_PORT}"

source "$(dirname -- "${BASH_SOURCE[0]}")/server_log.sh" decode
export SGLANG_MOONCAKE_DCP_PACK_BUFFER_MB=${SGLANG_MOONCAKE_DCP_PACK_BUFFER_MB:-128}
GLOG_logtostderr=1 \
GLOG_minloglevel=1 \
MC_USE_DATA_DIRECT=1 \
WITH_NVIDIA_PEERMEM=0 \
MC_TE_METRIC=1 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
SGLANG_LOG_LEVEL=debug \
SGLANG_LOG_REQUEST_HEADERS=venus-request-id \
SGLANG_K3_GEMM_AR=0 \
nohup python -m sglang.launch_server \
    --model-path /model/Kimi-K3 \
    --served-model-name base_model \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port "$DECODE_BOOTSTRAP_PORT" \
    --host "$DECODE_HOST" \
    --port "$DECODE_PORT" \
    --trust-remote-code \
    --tp-size 8 \
    --dcp-size 8 \
    --mem-fraction-static 0.92 \
    --max-running-requests 64 \
    --disaggregation-decode-extra-slots 16 \
    --mamba-full-memory-ratio 0.062 \
    --context-length 1048576 \
    --page-size 64 \
    --load-format runai_streamer \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-ssm-dtype bfloat16 \
    --attention-backend tokenspeed_mla \
    --disable-radix-cache \
    --cuda-graph-max-bs 64 \
    --watchdog-timeout 3600 \
    --dist-timeout 3600 \
    --enable-metrics \
    --enable-mfu-metrics \
    --enable-cache-report \
    --detokenizer-worker-num 4 \
    --tokenizer-worker-num 4 \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device mlx5_000,mlx5_001,mlx5_002,mlx5_003,mlx5_004,mlx5_005,mlx5_006,mlx5_007,mlx5_008,mlx5_009,mlx5_010,mlx5_011,mlx5_012,mlx5_013,mlx5_014,mlx5_015 \
    >> "$SERVER_LOG" 2>&1 &
