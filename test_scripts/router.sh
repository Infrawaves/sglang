#!/usr/bin/env bash
set -euo pipefail

: "${PREFILL_IP:?请设置 Prefill 服务地址 PREFILL_IP}"
: "${PREFILL_PORT:?请设置 Prefill 服务端口 PREFILL_PORT}"
: "${PREFILL_BOOTSTRAP_PORT:?请设置 Prefill bootstrap 端口 PREFILL_BOOTSTRAP_PORT}"
: "${DECODE_IP:?请设置 Decode 服务地址 DECODE_IP}"
: "${DECODE_PORT:?请设置 Decode 服务端口 DECODE_PORT}"
: "${ROUTER_HOST:?请设置路由器监听地址 ROUTER_HOST}"
: "${ROUTER_PORT:?请设置路由器监听端口 ROUTER_PORT}"

python3 -m sglang_router.launch_router \
    --host "$ROUTER_HOST" \
    --port "$ROUTER_PORT" \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:${PREFILL_PORT}" "$PREFILL_BOOTSTRAP_PORT" \
    --decode "http://${DECODE_IP}:${DECODE_PORT}" \
    --mini-lb \
    > router.log 2>&1 &
