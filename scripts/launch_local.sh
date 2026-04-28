#!/usr/bin/env bash
# Convenience launcher for local single-host workers.
#
# Usage:
#     bash scripts/launch.sh           # 1 worker
#     bash scripts/launch.sh 4         # 4 parallel workers (each grabs
#                                      #   a different job via lease)
#
# Each worker is a background process whose log is tee'd to
# logs/worker-<i>.log. The script returns immediately after spawning;
# `tail -f logs/worker-1.log` to follow.
#
# For multi-host see scripts/orchestrate.sh §"Remote".

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

N="${1:-1}"
mkdir -p "$REPO_ROOT/logs"

for i in $(seq 1 "$N"); do
    log="$REPO_ROOT/logs/worker-${i}.log"
    echo "▶ spawning worker-${i} (log: $log)"
    nohup bash "$REPO_ROOT/scripts/orchestrate.sh" >"$log" 2>&1 &
done

cat <<EOF

${N} worker(s) spawned. Useful follow-ups:
    tail -f logs/worker-1.log
    ls checkpoints/                         # one dir per <exp>
    find checkpoints -name complete         # finished runs
EOF
