#!/bin/sh
# preStop hook of a PD decode worker (non-leader) pod of an LWS group. The
# worker hosts ranks the leader's hand-off still runs collectives with, so it
# must outlive the migration retire_decode_leader.sh drives:
#
#   wait until every DP rank of the leader reports migration_done, or the
#   leader is unreachable MAX_FAILS times in a row (it is gone, and so is the
#   migration), or LIMIT seconds pass.
#
# Only migration_done counts: fully_idle is also true on an idle leader that has
# not started migrating yet. Always exits 0.
#
# Environment: LWS_LEADER_ADDRESS (required), PORT (30000), LIMIT (560),
# MAX_FAILS (10).

PORT="${PORT:-30000}"
LIMIT="${LIMIT:-560}"
MAX_FAILS="${MAX_FAILS:-10}"

log() { echo "[retire_decode_worker] $*"; }

if [ -z "$LWS_LEADER_ADDRESS" ]; then
  log "LWS_LEADER_ADDRESS is not set; exiting"
  exit 0
fi
LEADER="http://${LWS_LEADER_ADDRESS}:${PORT}"

end=$(( $(date +%s) + LIMIT ))
fails=0
while [ "$(date +%s)" -lt "$end" ]; do
  if out=$(curl -s -f -m 5 "$LEADER/server_info"); then
    fails=0
    if printf '%s' "$out" | python3 -c '
import json, sys
states = json.load(sys.stdin).get("internal_states") or []
sys.exit(0 if states and all(s.get("migration_done") is True for s in states) else 1)
' 2>/dev/null; then
      log "leader reports migration done"
      exit 0
    fi
  else
    fails=$((fails + 1))
    if [ "$fails" -ge "$MAX_FAILS" ]; then
      log "leader unreachable ${fails} times in a row; exiting"
      exit 0
    fi
  fi
  sleep 1
done
log "timed out after ${LIMIT}s waiting for the leader"
exit 0
