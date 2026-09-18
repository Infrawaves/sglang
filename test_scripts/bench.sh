#!/usr/bin/env bash
set -euo pipefail

: "${MODE:?请设置 MODE=A 或 MODE=B}"
: "${BASE:?请设置 Router 地址 BASE，例如 http://10.51.1.5:8848}"
case "$MODE" in
  A|B) ;;
  *) echo "MODE 必须是 A 或 B" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cmd=(
  "${PYTHON:-python3}" "$SCRIPT_DIR/bench_gpu_prefix.py" run
  --mode "$MODE"
  --base "$BASE"
  --model "${MODEL_PATH:-/model/Kimi-K3}"
  --prefix-k "${PREFIX_K:-512}"
  --suffix-k "${SUFFIX_K:-16}"
  --requests "${REQUESTS:-8}"
  --output-tokens "${OUTPUT_TOKENS:-8}"
  --results "${RESULT_DIR:-ab-results/gpu-prefix}"
  --prefill-log "${PREFILL_LOG:-${LOG_DIR:-log}/prefill-${MODE}-latest.log}"
)
if [[ -n ${PROFILE_PREFILL_URL:-} ]]; then
  cmd+=(--profile-prefill "$PROFILE_PREFILL_URL")
fi

echo "GPU-prefix bench: MODE=$MODE; 先清缓存、建立前缀、预热，再正式测量。"
echo "MODE 不会修改服务配置；请确认 P/D 已以同一 A/B 模式启动。"
exec "${cmd[@]}"
