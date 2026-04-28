#!/usr/bin/env bash
# Bare-EC2 launch path. Loads .env and exec's the watchdog, which keeps
# TARGET_WORKERS spot instances alive until every experiment has a
# `complete` sentinel in S3.
#
# Use this when you don't want dstack in the loop — the watchdog calls
# `aws ec2 run-instances` directly and self-cleans on completion.
#
# Stop with Ctrl-C. Watchdog will exit on its own when the queue drains.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    # Preserve caller-provided overrides ("TARGET_WORKERS=1 bash …")
    # so the .env doesn't silently clobber CLI-style invocation.
    _caller_TARGET_WORKERS="${TARGET_WORKERS-}"
    _caller_INSTANCE_TYPE="${INSTANCE_TYPE-}"
    _caller_JOB_ORDER="${JOB_ORDER-}"
    _caller_CAMPAIGN="${CAMPAIGN-}"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    [[ -n "$_caller_TARGET_WORKERS" ]] && TARGET_WORKERS="$_caller_TARGET_WORKERS"
    [[ -n "$_caller_INSTANCE_TYPE"  ]] && INSTANCE_TYPE="$_caller_INSTANCE_TYPE"
    [[ -n "$_caller_JOB_ORDER"      ]] && JOB_ORDER="$_caller_JOB_ORDER"
    [[ -n "$_caller_CAMPAIGN"       ]] && CAMPAIGN="$_caller_CAMPAIGN"
else
    echo "FATAL: $ENV_FILE not found. Copy .env.example → .env and fill it in." >&2
    exit 2
fi

: "${CAMPAIGN_AZS:?CAMPAIGN_AZS missing from .env}"
: "${TARGET_WORKERS:=2}"
: "${CAMPAIGN:=dyf}"
export CAMPAIGN

# Derive JOB_ORDER from the flat-file default unless the caller set it.
if [[ -z "${JOB_ORDER:-}" ]]; then
    JOB_ORDER=$(grep -v '^\s*#' "$REPO_ROOT/scripts/infra/default_job_order.txt" \
        | grep -v '^\s*$' | tr '\n' ' ' | sed 's/  */ /g' | sed 's/^ //;s/ $//')
    echo "JOB_ORDER not set — parsed default_job_order.txt"
fi
export JOB_ORDER

cat <<EOF
Campaign starting:
  CAMPAIGN       = $CAMPAIGN
  TARGET_WORKERS = $TARGET_WORKERS
  INSTANCE_TYPE  = ${INSTANCE_TYPE:-p5.48xlarge}
  CAMPAIGN_AZS   = $CAMPAIGN_AZS
  CKPT_BUCKET    = $CKPT_BUCKET
  JOB_ORDER      = $(echo "$JOB_ORDER" | tr ' ' '\n' | wc -l | tr -d ' ') jobs

EOF

exec bash "$REPO_ROOT/scripts/infra/watchdog.sh"
