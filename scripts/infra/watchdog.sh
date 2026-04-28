#!/usr/bin/env bash
# Spot-fleet watchdog. Keeps TARGET_WORKERS live in the AZs listed in
# CAMPAIGN_AZS until every experiment has a `complete` sentinel in S3.
#
# Behavior:
#   * Maintain TARGET_WORKERS spot instances tagged Project=dyf,
#     Role=worker, Campaign=$CAMPAIGN. If fewer are live, launch one
#     by rotating through CAMPAIGN_AZS in user-specified order.
#   * Completion: when every experiment key in JOB_ORDER has a
#     `complete` object under $CKPT_BUCKET/<exp>/, exit 0.
#   * Capacity retry: if run-instances fails in all CAMPAIGN_AZS, sleep
#     POLL_INTERVAL_SEC and try again — forever. Never escalates to
#     on-demand, never gives up.
#
# Required env (.env):
#   CKPT_BUCKET, WANDB_API_KEY, JOB_ORDER, CAMPAIGN_AZS,
#   AMI, SG, IAM_PROFILE, DATA_SNAPSHOT, SUBNET_<AZ>
# Optional:
#   TARGET_WORKERS=2
#   INSTANCE_TYPE=p5.48xlarge
#   POLL_INTERVAL_SEC=60
#   CAPACITY_SLEEP_SEC=60
#   CAMPAIGN=dyf
set -euo pipefail

: "${CKPT_BUCKET:?CKPT_BUCKET required}"
: "${WANDB_API_KEY:?WANDB_API_KEY required (empty string to disable W&B)}"
: "${JOB_ORDER:?JOB_ORDER required}"
: "${CAMPAIGN_AZS:?CAMPAIGN_AZS required}"
: "${TARGET_WORKERS:=2}"
: "${INSTANCE_TYPE:=p5.48xlarge}"
: "${POLL_INTERVAL_SEC:=60}"
: "${CAPACITY_SLEEP_SEC:=60}"
: "${CAMPAIGN:=dyf}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
export CKPT_BUCKET WANDB_API_KEY CAMPAIGN AWS_DEFAULT_REGION

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_WORKER="$SCRIPT_DIR/run_worker.sh"

log() { printf '[watchdog %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

bucket_root="${CKPT_BUCKET%/}"
bucket_name=$(echo "$bucket_root" | sed -E 's|^s3://([^/]+).*|\1|')
bucket_prefix=$(echo "$bucket_root" | sed -E 's|^s3://[^/]+/?||')
[[ -n "$bucket_prefix" ]] && bucket_prefix="${bucket_prefix}/"

exp_complete() {
    # True iff $1 (an experiment basename) has a `complete` sentinel.
    # orchestrate.sh writes it directly at <CKPT_BUCKET>/<exp>/complete
    # (no nested-experiment subdir like timm does in NELU). We tolerate
    # both layouts in case main.py's CheckpointSaver structure changes.
    local exp="$1"
    local key_flat="${bucket_prefix}${exp}/complete"
    local key_nested="${bucket_prefix}${exp}/${exp}/complete"
    aws s3api head-object --bucket "$bucket_name" --key "$key_flat" \
        >/dev/null 2>&1 && return 0
    aws s3api head-object --bucket "$bucket_name" --key "$key_nested" \
        >/dev/null 2>&1 && return 0
    return 1
}

_exp_from_entry() {
    local entry="$1" cfg act
    IFS=: read -r cfg act <<<"$entry"
    local base
    base=$(basename "${cfg%.yaml}")
    echo "${base}-${act}"
}

all_done() {
    for entry in $JOB_ORDER; do
        local exp; exp=$(_exp_from_entry "$entry")
        exp_complete "$exp" || return 1
    done
    return 0
}

count_incomplete() {
    local n=0
    for entry in $JOB_ORDER; do
        local exp; exp=$(_exp_from_entry "$entry")
        exp_complete "$exp" || n=$((n + 1))
    done
    echo "$n"
}

count_live_workers() {
    # Filter on Campaign so multiple campaigns can coexist. Without it
    # a second campaign's TARGET_WORKERS=1 would see this campaign's
    # fleet as already-live and never relaunch on its own.
    aws ec2 describe-instances \
        --region "$AWS_DEFAULT_REGION" \
        --filters \
            "Name=tag:Project,Values=dyf" \
            "Name=tag:Role,Values=worker" \
            "Name=tag:Campaign,Values=${CAMPAIGN}" \
            "Name=instance-state-name,Values=pending,running" \
        --query 'length(Reservations[].Instances[])' \
        --output text
}

launch_one() {
    local az_cycle=($CAMPAIGN_AZS)
    local cycle=0
    while :; do
        for az in "${az_cycle[@]}"; do
            local suffix
            suffix="$(date -u +%Y%m%dT%H%M%S)-${az##*-}"
            log "launching worker in $az ($INSTANCE_TYPE)"
            if bash "$RUN_WORKER" "$az" "$INSTANCE_TYPE" "$suffix" \
                    > /tmp/dyf-launch-$$.log 2>&1; then
                log "launch OK"
                head -10 /tmp/dyf-launch-$$.log
                rm -f /tmp/dyf-launch-$$.log
                return 0
            fi
            local err; err=$(tail -5 /tmp/dyf-launch-$$.log | tr '\n' ' ')
            log "launch in $az failed: $err"
        done
        cycle=$((cycle + 1))
        log "AZ cycle $cycle exhausted — sleeping ${CAPACITY_SLEEP_SEC}s and retrying"
        sleep "$CAPACITY_SLEEP_SEC"
    done
}

log "starting. campaign=$CAMPAIGN target=$TARGET_WORKERS type=$INSTANCE_TYPE AZs=($CAMPAIGN_AZS)"
while :; do
    if all_done; then
        log "queue drained — every experiment has a complete sentinel. exiting."
        break
    fi

    # Authoritative live count from AWS at the top of each polling pass.
    # Inside the inner launch loop we increment locally rather than
    # re-querying — DescribeInstances has 1-3 s eventual-consistency
    # after RunInstances so successive calls would still report the old
    # count and we'd double-launch. The next outer pass corrects drift.
    live=$(count_live_workers)
    remaining=$(count_incomplete)
    effective_target=$TARGET_WORKERS
    (( remaining < effective_target )) && effective_target=$remaining
    log "live: $live / $effective_target  (TARGET=$TARGET_WORKERS, remaining_jobs=$remaining)"

    while (( live < effective_target )); do
        if launch_one; then
            live=$((live + 1))
        fi
    done

    sleep "$POLL_INTERVAL_SEC"
done
