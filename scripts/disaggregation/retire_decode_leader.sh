#!/bin/sh
# preStop hook of a PD decode leader pod: hand its running requests off to other
# decode instances (decode hot migration), then let the pod terminate.
#
#   1. Sleep GRACE seconds: short requests finish and routers drop this pod.
#   2. POST /pause_generation {"mode": "migrate"}: new requests get a 503 with
#      X-SGLang-Draining: 1, running ones are stored and handed off.
#   3. Poll /server_info until every DP rank reports migration_done, or DRAIN
#      seconds pass.
#
# Always exits 0; terminationGracePeriodSeconds bounds the whole hook, so keep
# GRACE + FLUSH + DRAIN below it. Worker pods of the group run
# retire_decode_worker.sh, which waits for migration_done.
#
# Environment: PORT (30000), GRACE (60), FLUSH (30), DRAIN (300).

PORT="${PORT:-30000}"
GRACE="${GRACE:-60}"
FLUSH="${FLUSH:-30}"
DRAIN="${DRAIN:-300}"
URL="http://127.0.0.1:${PORT}"

log() { echo "[retire_decode_leader] $*"; }

sleep "$GRACE"

code=$(curl -s -o /dev/null -w '%{http_code}' -m 30 -X POST "$URL/pause_generation" \
  -H 'Content-Type: application/json' \
  -d "{\"mode\":\"migrate\",\"migrate_flush_timeout_s\":${FLUSH}}")
if [ "$code" != "200" ]; then
  # Not a migration-capable decode: its remaining requests fail on shutdown.
  log "pause_generation(mode=migrate) returned HTTP ${code}; exiting"
  exit 0
fi
log "migrate pause issued"

end=$(( $(date +%s) + DRAIN ))
while [ "$(date +%s)" -lt "$end" ]; do
  # fully_idle stands in for migration_done on an engine that predates it.
  if curl -s -f -m 5 "$URL/server_info" | python3 -c '
import json, sys
states = json.load(sys.stdin).get("internal_states") or []
done = [s.get("migration_done", s.get("fully_idle")) is True for s in states]
sys.exit(0 if done and all(done) else 1)
' 2>/dev/null; then
    log "migration done"
    exit 0
  fi
  sleep 1
done
log "timed out after ${DRAIN}s waiting for migration_done"
exit 0
