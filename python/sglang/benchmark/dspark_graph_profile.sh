#!/usr/bin/env bash
# Kimi-K3 / GB300, TP32 EP32 DP8 DCP4, gamma=5, 64K graph SPS profiling.
# Use this branch's source and the same CUDA/runtime image on every node.
#
# On each of the 8 nodes (4 GPUs/node), set its NODE_RANK to 0..7:
#   MASTER_ADDR=<node-0-ip> NODE_RANK=0 bash python/sglang/benchmark/dspark_graph_profile.sh serve
# After all nodes are ready, run once on a host with /model/Kimi-K3 available:
#   BASE_URL=http://<node-0-ip>:30000 PROFILE_OUT=/mnt/nfs/kimi_k3_dspark_sps_graph_64k.json \
#     bash python/sglang/benchmark/dspark_graph_profile.sh profile
# For 8-GPU nodes, set NNODES=4 on every server and use NODE_RANK=0..3.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

PROFILE_PYTHON=${PROFILE_PYTHON:-python3}
MODEL_PATH=${MODEL_PATH:-/model/Kimi-K3}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-/model/Kimi-K3-DSpark}
SERVER_PORT=${SERVER_PORT:-30000}
DIST_PORT=${DIST_PORT:-5777}
NNODES=${NNODES:-8}
case "${1:-}" in
  serve)
    shift
    : "${MASTER_ADDR:?Set MASTER_ADDR to the reachable node 0 IP or hostname}"
    : "${NODE_RANK:?Set NODE_RANK to the node rank in 0..NNODES-1}"
    if ! [[ "${NNODES}" =~ ^[1-9][0-9]*$ && "${NODE_RANK}" =~ ^[0-9]+$ ]] ||
       (( 32 % NNODES != 0 || NODE_RANK >= NNODES )); then
      echo "NNODES must divide 32; NODE_RANK must be in 0..NNODES-1." >&2
      exit 2
    fi

    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond2}
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond2}
    export SGLANG_RAGGED_VERIFY_MODE=compact
    export SGLANG_DSPARK_ENABLE_COMPACT_DP_TIER=1
    export SGLANG_DSPARK_ENABLE_SPS_RECORD=1
    export SGLANG_SIMULATE_ACC_LEN=1.0
    export SGLANG_DSPARK_DEBUG_DUMP=""
    export SGLANG_DSPARK_LOG_SPS_PRED_INTERVAL=0
    export SGLANG_CUTEDSL_MLA_NUM_KV_SPLITS=4
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=1920
    export SGLANG_K3_SP_COLLECTIVE=1

    exec "${PROFILE_PYTHON}" -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --speculative-algorithm DSPARK \
      --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
      --speculative-dspark-block-size 5 \
      --speculative-attention-mode decode \
      --speculative-draft-attention-backend trtllm_mha \
      --speculative-dspark-align-verify-tokens-to-graph-tier \
      --attention-backend cutedsl_mla \
      --tp-size 32 --ep-size 32 --dp-size 8 \
      --enable-dp-attention --enable-dp-lm-head \
      --moe-a2a-backend megamoe \
      --dcp-size 4 --dcp-kv-layout page \
      --nnodes "${NNODES}" --node-rank "${NODE_RANK}" \
      --dist-init-addr "${MASTER_ADDR}:${DIST_PORT}" \
      --host 0.0.0.0 --port "${SERVER_PORT}" \
      --disaggregation-mode decode \
      --disaggregation-transfer-backend fake \
      --disaggregation-decode-extra-slots 16 \
      --enable-linear-replayssm-spec \
      --mamba-full-memory-ratio 0.25 \
      --mamba-ssm-dtype bfloat16 \
      --max-mamba-cache-size 2176 \
      --kv-cache-dtype fp8_e4m3 --page-size 64 \
      --mem-fraction-static 0.835 \
      --cuda-graph-max-bs-decode 256 \
      --max-running-requests 2048 --max-batch-outputs 256 \
      --context-length 1048576 \
      --disable-radix-cache \
      --load-balance-method context_bucket \
      --load-snapshot-publish-interval 1 \
      --detokenizer-worker-num 4 --tokenizer-worker-num 8 \
      --watchdog-timeout 3600 --dist-timeout 3600 \
      --trust-remote-code "$@"
    ;;
  profile)
    shift
    BASE_URL=${BASE_URL:-"http://${MASTER_ADDR:?Set BASE_URL or MASTER_ADDR}:${SERVER_PORT}"}
    PROFILE_OUT=${PROFILE_OUT:-"${PWD}/kimi_k3_dspark_sps_graph_64k.json"}
    exec "${PROFILE_PYTHON}" -m sglang.benchmark.dspark_sps_profiler all \
      --graph-table \
      --base-url "${BASE_URL}" \
      --local-tokenizer-path "${MODEL_PATH}" \
      --batch-size 1 2 4 8 16 32 64 96 128 192 256 \
      --input-len 65536 --temperature 1.0 \
      --repeats 2 \
      --min-steady-steps 128 --min-steady-seconds 20 \
      --round-timeout 600 --ramp-token-slack 1000 \
      --no-plot --out "${PROFILE_OUT}" "$@"
    ;;
  *)
    echo "Usage: MASTER_ADDR=<node-0-ip> [NODE_RANK=0] bash $0 {serve|profile} [extra CLI args]" >&2
    exit 2
    ;;
esac
