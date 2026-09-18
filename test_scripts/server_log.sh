#!/usr/bin/env bash
# Sourced by prefill.sh/decode.sh with the role as the first argument.
case "${SGLANG_MOONCAKE_DCP_WINDOW_PACK:-0}" in
  1|true|True|TRUE) server_mode=B ;;
  0|false|False|FALSE) server_mode=A ;;
  *) echo "请将 SGLANG_MOONCAKE_DCP_WINDOW_PACK 设置为 0 或 1" >&2; exit 2 ;;
esac
if [[ -n ${MODE:-} && $MODE != "$server_mode" ]]; then
  echo "MODE=$MODE 与实际窗口化配置（$server_mode）不一致，拒绝启动" >&2
  exit 2
fi
mkdir -p "${LOG_DIR:-log}"
server_log_dir=$(cd -- "${LOG_DIR:-log}" && pwd)
# mktemp prevents overwrite even when two starts occur in the same second.
SERVER_LOG=$(mktemp "$server_log_dir/$1-$server_mode-$(date +%Y%m%d-%H%M%S).XXXXXX.log")
ln -sfn -- "$SERVER_LOG" "$server_log_dir/$1-$server_mode-latest.log"
echo "服务模式=$server_mode 日志=$SERVER_LOG"
