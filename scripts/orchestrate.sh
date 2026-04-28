#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# DyF training-campaign worker.
#
# A single VM (spot or otherwise) runs this script. The worker pulls one
# training job at a time from an S3-backed queue and exits when the
# queue is drained. dstack / bare-EC2 launchers do not relaunch it.
#
# Flow per VM:
#   1. Sanity-check /data mount (caller is responsible for actually
#      mounting it — dstack volumes: or bootstrap.sh).
#   2. Start a preempt watcher that polls IMDSv2's spot/instance-action
#      endpoint and SIGTERMs the trainer on notification.
#   3. Loop the JOB_ORDER queue:
#        * Skip if s3://CKPT_BUCKET/<exp>/complete exists.
#        * Skip if s3://CKPT_BUCKET/<exp>/lease is fresh (<LEASE_TTL).
#        * Else acquire lease (S3 conditional PUT + readback), pull
#          prior state, run main.py, push state back, release lease.
#   4. After MAX_IDLE_PASSES empty passes → exit cleanly.
#
# This script is the cloud-multi-host counterpart to a local single-host
# loop: the lease + sentinel layer lives on S3 so multiple workers in
# different regions/AZs can race for jobs without trampling each other.
#
# Required env (from dstack task or render_user_data.sh):
#   CKPT_BUCKET         e.g. s3://dyf-checkpoints
#   AWS_DEFAULT_REGION  e.g. us-west-2
#
# Optional env:
#   WANDB_API_KEY       enables W&B logging
#   WANDB_PROJECT       default "dyf"
#   WANDB_ENTITY        default unset
#   JOB_ORDER           space-separated <cfg>:<act> pairs; if unset,
#                       parsed from scripts/infra/default_job_order.txt
#   LEASE_TTL           seconds (default 600)
#   HEARTBEAT_EVERY     seconds (default 60)
#   MAX_IDLE_PASSES     default 1
#   DATA_MOUNT          where the dataset lives (default /data)
#   NPROC_OVERRIDE      override --nproc_per_node (rare)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

log() { printf '[orchestrate %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

: "${CKPT_BUCKET:?CKPT_BUCKET not set (e.g. s3://dyf-checkpoints)}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${WANDB_PROJECT:=dyf}"
: "${WANDB_ENTITY:=}"
: "${LEASE_TTL:=600}"
: "${HEARTBEAT_EVERY:=60}"
: "${MAX_IDLE_PASSES:=1}"
: "${DATA_MOUNT:=/data}"
export AWS_DEFAULT_REGION

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Job queue ──────────────────────────────────────────────────────
if [[ -z "${JOB_ORDER:-}" ]]; then
    job_file="$REPO_ROOT/scripts/infra/default_job_order.txt"
    if [[ -f "$job_file" ]]; then
        JOB_ORDER=$(grep -v '^\s*#' "$job_file" | grep -v '^\s*$' | tr '\n' ' ')
    fi
fi
: "${JOB_ORDER:?JOB_ORDER empty and default_job_order.txt missing}"

# Activation → CLI flag mapping.
# shellcheck source=infra/activation_flags.sh
source "$REPO_ROOT/scripts/infra/activation_flags.sh"

# ── IMDSv2 helpers ─────────────────────────────────────────────────
imds() {
    local token
    token=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
    curl -sS -H "X-aws-ec2-metadata-token: $token" \
         "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null || true
}

AZ=$(imds placement/availability-zone)
INSTANCE_ID=$(imds instance-id)
[[ -z "$INSTANCE_ID" ]] && INSTANCE_ID="$(hostname)-$$"
log "AZ=${AZ:-unknown} instance=$INSTANCE_ID"

# ── Data sanity ────────────────────────────────────────────────────
# We don't mount /data ourselves — dstack `volumes:` or bootstrap.sh do.
# Fail fast if the layout doesn't match _base.yaml's data_path so we
# don't burn spot $ on a doomed main.py launch.
#
# Expected layout (NELU-compatible):
#     $DATA_MOUNT/imagenet/train/<class>/<img>.JPEG    (1000 classes)
#     $DATA_MOUNT/imagenet/val/<class>/<img>.JPEG      (1000 classes)
check_data_mount() {
    local root="$DATA_MOUNT/imagenet"
    if [[ ! -d "$root/train" || ! -d "$root/val" ]]; then
        log "FATAL: expected $root/{train,val} — neither exists."
        log "  Top of $DATA_MOUNT:"; ls "$DATA_MOUNT" 2>&1 | head -10 >&2 || true
        exit 4
    fi
    # Count classes — both splits should have 1000 for ImageNet-1k.
    local n_train n_val
    n_train=$(ls "$root/train" 2>/dev/null | wc -l | tr -d ' ')
    n_val=$(ls "$root/val" 2>/dev/null | wc -l | tr -d ' ')
    if (( n_train < 1000 || n_val < 1000 )); then
        log "FATAL: $root looks malformed (train=$n_train, val=$n_val classes; want 1000)."
        log "  Often means val/ wasn't sorted into class subfolders. Fix the volume."
        exit 4
    fi
    log "data mount OK: $root (train=$n_train, val=$n_val classes)"
    log "  free space: $(df -h "$DATA_MOUNT" 2>/dev/null | tail -1 || echo n/a)"
}

# ── Preempt watcher ────────────────────────────────────────────────
# Polls the IMDS spot/instance-action endpoint. On notification it
# SIGTERMs the trainer process group (pgid in /tmp/trainer.pgid).
PREEMPT_WATCHER_PID=""
start_preempt_watcher() {
    (
        while :; do
            local token code
            token=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
                    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
            code=$(curl -sS -o /dev/null -w '%{http_code}' \
                   -H "X-aws-ec2-metadata-token: $token" \
                   "http://169.254.169.254/latest/meta-data/spot/instance-action" \
                   2>/dev/null || echo "000")
            if [[ "$code" == "200" ]]; then
                if [[ -f /tmp/trainer.pgid ]]; then
                    pgid=$(cat /tmp/trainer.pgid)
                    log "preempt notice — SIGTERM to pgid=$pgid"
                    kill -TERM -"$pgid" 2>/dev/null || true
                fi
                break
            fi
            sleep 5
        done
    ) &
    PREEMPT_WATCHER_PID=$!
    log "preempt watcher pid=$PREEMPT_WATCHER_PID"
}

# ── S3 helpers ─────────────────────────────────────────────────────
s3_exists() { aws s3 ls "$1" >/dev/null 2>&1; }

exp_key() {
    local cfg="$1" act="$2" base
    base=$(basename "${cfg%.yaml}")
    echo "${base}-${act}"
}

# Lease layout under $CKPT_BUCKET/<exp>/:
#   complete   — sentinel, presence ⇒ done. Never re-run.
#   lease      — content "<owner> <unix-ts>". TTL = LEASE_TTL.
#   ckpt-*.pth, log.txt, args.json, wandb_run_id.json
lease_claim() {
    # Optimistic claim with read-back confirmation:
    #   1. Skip if a fresh lease exists.
    #   2. PUT our own (owner, ts).
    #   3. Sleep tiny jitter, GET. If another worker PUT after us they
    #      win and we skip. Catches the read-after-write race that S3
    #      cannot protect against (both workers see "no lease" before
    #      either PUTs).
    local exp="$1"
    local key="${CKPT_BUCKET}/${exp}/lease"
    local now owner ts age
    now=$(date +%s)
    if s3_exists "$key"; then
        read -r owner ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
        ts=${ts:-0}
        age=$((now - ts))
        if (( age < LEASE_TTL )); then
            return 1
        fi
        log "stealing stale lease on $exp (age=${age}s, owner=$owner)"
    fi
    echo "$INSTANCE_ID $now" | aws s3 cp - "$key" >/dev/null
    sleep "$(awk "BEGIN{print 0.5+rand()}")"
    local winner
    read -r winner _ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
    if [[ "$winner" != "$INSTANCE_ID" ]]; then
        log "lost lease race on $exp (winner=$winner)"
        return 1
    fi
    return 0
}

lease_refresh() {
    local exp="$1"
    local key="${CKPT_BUCKET}/${exp}/lease"
    local now owner
    now=$(date +%s)
    read -r owner _ts < <(aws s3 cp "$key" - 2>/dev/null || echo "- 0")
    [[ "$owner" == "$INSTANCE_ID" ]] || return 1
    echo "$INSTANCE_ID $now" | aws s3 cp - "$key" >/dev/null
}

lease_release() {
    aws s3 rm "${CKPT_BUCKET}/$1/lease" >/dev/null 2>&1 || true
}

# ── Run one job ────────────────────────────────────────────────────
run_job() {
    local cfg="$1" act="$2"
    local exp s3_prefix
    exp=$(exp_key "$cfg" "$act")
    s3_prefix="${CKPT_BUCKET}/${exp}"

    if s3_exists "${s3_prefix}/complete"; then
        log "skip $exp (complete)"
        return 3
    fi

    if ! lease_claim "$exp"; then
        log "skip $exp (fresh lease held)"
        return 2
    fi

    log "▶ running $exp"

    local outdir="/tmp/runs/${exp}"
    mkdir -p "$outdir"

    # Pull prior state. main.py's auto_resume globs checkpoint-*.pth
    # under --output_dir, so we must include them. save_ckpt_num=3 in
    # _base.yaml caps the rolling history at 3 files, keeping transfer
    # bounded.
    aws s3 sync "${s3_prefix}/" "${outdir}/" \
        --exclude "lease" --exclude "complete" \
        --exact-timestamps --only-show-errors || true

    # Build CLI args from YAML + activation variant.
    local cfg_args act_args
    cfg_args=$(python "$REPO_ROOT/scripts/infra/yaml_to_args.py" "$cfg")
    act_args=$(activation_flags "$act")

    local wandb_args=()
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        wandb_args+=(--enable_wandb true --project "$WANDB_PROJECT")
        if command -v wandb >/dev/null 2>&1; then
            wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
        fi
    fi

    # Heartbeat + interim S3 push every HEARTBEAT_EVERY seconds.
    (
        while sleep "$HEARTBEAT_EVERY"; do
            lease_refresh "$exp" || exit 0
            aws s3 sync "${outdir}/" "${s3_prefix}/" \
                --exclude "*.tmp" --exclude "lease" \
                --only-show-errors || true
        done
    ) &
    local heartbeat_pid=$!

    # Detect GPUs.
    local gpus
    if [[ -n "${NPROC_OVERRIDE:-}" ]]; then
        gpus="$NPROC_OVERRIDE"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    else
        gpus=1
    fi
    [[ -z "$gpus" || "$gpus" == "0" ]] && gpus=1

    # Launch trainer in its own session so the preempt watcher can
    # SIGTERM the whole tree by pgid.
    local trainer_rc=0
    set +e
    if (( gpus > 1 )); then
        # shellcheck disable=SC2086
        setsid torchrun --standalone --nproc_per_node="$gpus" \
            "$REPO_ROOT/main.py" \
            $cfg_args $act_args \
            --output_dir "$outdir" \
            --log_dir "$outdir/tb" \
            "${wandb_args[@]}" \
            >>"$outdir/log.txt" 2>&1 &
    else
        # shellcheck disable=SC2086
        setsid python "$REPO_ROOT/main.py" \
            $cfg_args $act_args \
            --output_dir "$outdir" \
            --log_dir "$outdir/tb" \
            "${wandb_args[@]}" \
            >>"$outdir/log.txt" 2>&1 &
    fi
    local trainer_pid=$!
    echo "$trainer_pid" >/tmp/trainer.pgid
    wait "$trainer_pid"
    trainer_rc=$?
    set -e

    kill "$heartbeat_pid" 2>/dev/null || true
    rm -f /tmp/trainer.pgid

    # Final sync regardless of exit code — preserve whatever the trainer
    # checkpointed before crash/preempt.
    aws s3 sync "${outdir}/" "${s3_prefix}/" --exclude "lease" \
        --only-show-errors || true

    # Sentinel: only on clean exit. main.py writes a marker we can
    # detect; absent that, treat exit-code 0 as completion. Pause-on-
    # preempt is the SIGTERM path which exits non-zero through Python's
    # default handler — heartbeat+sync above already pushed the latest
    # checkpoint so the next worker can resume.
    if (( trainer_rc == 0 )); then
        date -u +%Y-%m-%dT%H:%M:%SZ \
            | aws s3 cp - "${s3_prefix}/complete" >/dev/null
        log "✓ $exp complete"
    elif (( trainer_rc == 143 || trainer_rc == 137 )); then
        log "⏸ $exp paused (signal exit rc=$trainer_rc, will resume)"
    else
        log "✗ $exp failed (rc=$trainer_rc) — see ${s3_prefix}/log.txt"
    fi

    lease_release "$exp"
    rm -rf "$outdir"
    return 0
}

# ── Main ───────────────────────────────────────────────────────────
check_data_mount
start_preempt_watcher

idle_passes=0
while (( idle_passes < MAX_IDLE_PASSES )); do
    ran_any=0
    for pair in $JOB_ORDER; do
        IFS=: read -r cfg act <<<"$pair"
        rc=0
        run_job "$cfg" "$act" || rc=$?
        # rc=0 ⇒ trainer actually ran (counts as work).
        # rc=2 (lease) and rc=3 (complete) MUST NOT reset idle counter,
        # otherwise a fully-drained queue loops forever burning spot $.
        [[ $rc -eq 0 ]] && ran_any=1
    done
    if (( ran_any == 0 )); then
        idle_passes=$((idle_passes + 1))
        log "no work in this pass (${idle_passes}/${MAX_IDLE_PASSES})"
        sleep 10
    else
        idle_passes=0
    fi
done

log "queue drained — worker exiting"
