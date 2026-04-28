#!/usr/bin/env bash
# Bare-EC2 worker bootstrap. Invoked by the VM's user-data at first boot.
#
# Pre-conditions (handled by the launch template / run_worker.sh):
#   * /dev/sdg is an EBS volume cloned from $DATA_SNAPSHOT, mounted at /data
#     (we mount it here, not in user-data, so we can log the mount).
#   * IAM instance profile `dyf-worker-profile` is attached.
#   * Ubuntu 22.04 + NVIDIA driver pre-installed (AWS Deep Learning Base
#     AMI works; vanilla Ubuntu needs a separate driver install step).
#
# Responsibilities:
#   1. Mount the dataset volume at /data (read-only).
#   2. Clone the repo at /workspace and check out $REPO_REF.
#   3. Install minimal Python deps into the system Python (timm, wandb,
#      pyyaml, awscli). PyTorch is assumed pre-installed by the AMI.
#   4. Hand off to scripts/orchestrate.sh.
#   5. On orchestrator exit, self-terminate via ec2:TerminateInstances
#      so the spot VM stops billing.
#
# Logs go to /var/log/dyf/bootstrap.log AND /dev/console so
# `aws ec2 get-console-output` is useful when SSH is unreachable.
#
# Required env (exported by user-data wrapper):
#   REPO_URL            https://github.com/<user>/DyF.git
#   REPO_REF            branch / tag / commit
#   CKPT_BUCKET         s3://dyf-checkpoints
#   AWS_DEFAULT_REGION  us-west-2
# Optional:
#   WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY
#   JOB_ORDER           override default queue
#   ENTRY_SCRIPT        defaults to scripts/orchestrate.sh

LOGDIR=/var/log/dyf
mkdir -p "$LOGDIR"
LOG="$LOGDIR/bootstrap.log"
exec > >(tee -a "$LOG" | tee /dev/console) 2>&1

echo "[bootstrap] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting on $(hostname)"

: "${REPO_URL:?REPO_URL required}"
: "${REPO_REF:=main}"
: "${CKPT_BUCKET:?CKPT_BUCKET required}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${WORKSPACE:=/workspace}"
# Pre-built venv tarball: contains Python 3.10 + PyTorch (CUDA 13 build)
# + timm + wandb + everything else needed for training. Reused from
# NELU — same Python/torch versions, identical dependency surface.
: "${VENV_S3_URL:=s3://nelu-datasets/env/nelu-venv-py310-cu130.tar.gz}"
: "${VENV_ROOT:=/opt/dyf-venv}"
export AWS_DEFAULT_REGION

die() {
    echo "[bootstrap] FATAL: $1 (rc=${2:-99})"
    exit "${2:-99}"
}

step() {
    local name="$1"; shift
    local logfile="${LOGDIR}/${name}.log"
    echo "[bootstrap] ▶ $name"
    # Capture rc immediately — `local rc=$?` inside `if` would record
    # `local`'s exit (0), masking the real failure. Same trap with `set
    # -e`, which we don't use, but keep this idiom anyway.
    "$@" > "$logfile" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        echo "[bootstrap] ✗ $name failed (rc=$rc). Last 40 lines of $logfile:"
        tail -40 "$logfile"
        return $rc
    fi
    echo "[bootstrap] ✓ $name ok"
}

# ── 1. Mount /data ────────────────────────────────────────────────
mount_data() {
    if mountpoint -q /data; then
        echo "[bootstrap] /data already mounted"; return 0
    fi
    # The data volume is the EBS disk attached as /dev/sdg — but NVMe
    # remaps sdg to /dev/nvme*n1 in unpredictable enumeration order, and
    # some p5 AMIs return only the device-name (not vol-id) from the
    # IMDS block-device-mapping path. We pick by structural properties:
    #
    #   - is an nvme disk (lsblk TYPE=disk)
    #   - has SERIAL starting with "vol" (EBS, not "AWS…" instance store)
    #   - has NO mounted partition and is NOT the root device
    #
    # That uniquely matches the dataset volume on every p5/g5 AMI we've
    # seen, regardless of NVMe ordering.
    local dev="" root_disk
    root_disk=$(findmnt -no SOURCE / | sed -E 's|p?[0-9]+$||')   # /dev/nvme0n1
    for _ in {1..30}; do
        # List nvme disks; for each, check serial prefix and skip root.
        while read -r name serial; do
            local devpath="/dev/$name"
            [[ "$devpath" == "$root_disk" ]] && continue
            [[ "$serial" != vol* ]] && continue
            # Skip if any partition exists or it's already mounted.
            if lsblk -no MOUNTPOINT "$devpath" | grep -qE '\S'; then
                continue
            fi
            dev="$devpath"
            break
        done < <(lsblk -dno NAME,SERIAL,TYPE | awk '$3=="disk" {print $1, $2}')
        [[ -n "$dev" ]] && break
        sleep 1
    done
    [[ -z "$dev" ]] && die "could not resolve /data device (no unmounted EBS disk found)" 2
    echo "[bootstrap] mounting $dev at /data (ro)"
    mkdir -p /data
    mount -o ro "$dev" /data || die "mount $dev /data failed" 3
    df -h /data
}
mount_data || die "/data mount failed — refusing to continue without dataset" 9

# ── 2. Clone repo ─────────────────────────────────────────────────
if [[ -d "$WORKSPACE/.git" ]]; then
    step repo-update bash -c "cd $WORKSPACE && git fetch --all -q && \
        git checkout $REPO_REF -q && git pull -q" || die "repo-update failed" 6
else
    step repo-clone git clone -q --branch "$REPO_REF" "$REPO_URL" "$WORKSPACE" \
        || die "repo-clone failed" 6
fi

# ── 3. Install Python env ────────────────────────────────────────
# AWS DL Base AMI gives us NVIDIA driver + CUDA but NOT PyTorch. We
# fetch the pre-built venv tarball NELU uses (it has Python 3.10 +
# torch + timm + wandb + everything). Two gotchas baked into that
# tarball that we have to work around:
#
#   (a) the venv was built at /opt/nelu-venv, and `bin/activate` has
#       that path HARD-CODED. Renaming the directory breaks PATH.
#       Workaround: leave the directory at /opt/nelu-venv, don't
#       try to rename to our project name.
#
#   (b) the venv ships only a `python3.10` binary in bin/, no `python`
#       symlink. After activate, the PATH lookup falls through to
#       /usr/bin/python3 (system) and `python` is missing entirely.
#       Workaround: make the symlink ourselves before activating.
VENV_ROOT=/opt/nelu-venv

if [[ -x "${VENV_ROOT}/bin/python3.10" || -x "${VENV_ROOT}/bin/python3" ]]; then
    echo "[bootstrap] reusing existing ${VENV_ROOT}"
else
    step fetch-venv aws s3 cp "$VENV_S3_URL" /tmp/dyf-venv.tar.gz \
        || die "fetch-venv failed" 4
    step extract-venv bash -c \
        "mkdir -p /opt && tar xzf /tmp/dyf-venv.tar.gz -C /opt && rm -f /tmp/dyf-venv.tar.gz" \
        || die "extract-venv failed" 5
fi

# Make sure `python` exists in the venv bin/. NELU's venv only has
# python3.10; without this symlink, every script calling bare `python`
# (orchestrate.sh, yaml_to_args.py, dryrun.sh) crashes.
if [[ ! -e "${VENV_ROOT}/bin/python" ]]; then
    if [[ -x "${VENV_ROOT}/bin/python3.10" ]]; then
        ln -sf python3.10 "${VENV_ROOT}/bin/python"
    elif [[ -x "${VENV_ROOT}/bin/python3" ]]; then
        ln -sf python3 "${VENV_ROOT}/bin/python"
    fi
fi

# Activate the venv. PATH-prepend, sourced so child shells inherit it.
# shellcheck disable=SC1090
source "${VENV_ROOT}/bin/activate"

# Sanity: torch + torchrun must work from the activated venv.
step verify-env python -c \
    "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || die "torch import failed in venv" 7
command -v torchrun >/dev/null 2>&1 || die "torchrun not on PATH after venv activate" 8

# Top-up: pyyaml + awscli are tiny and may have been pinned in the
# venv. Install into venv so orchestrate.sh's helpers don't fall back
# to the system Python (which has no torch). Non-fatal — both are
# usually present already.
python -m pip install --quiet --upgrade pyyaml awscli >/dev/null 2>&1 || true

# ── 4. Orchestrate ────────────────────────────────────────────────
: "${ENTRY_SCRIPT:=scripts/orchestrate.sh}"
echo "[bootstrap] handing off to ${ENTRY_SCRIPT}"
cd "$WORKSPACE"
bash "$ENTRY_SCRIPT"
ORC_RC=$?
echo "[bootstrap] ${ENTRY_SCRIPT} exited rc=$ORC_RC"

# ── 5. Self-terminate ────────────────────────────────────────────
# Without this the spot VM sits idle billing $14-21/h and confuses
# the watchdog's live-worker count. Primary path: ec2:TerminateInstances.
# Fallback: OS halt — AWS reaps halted spot VMs shortly after.
echo "[bootstrap] self-terminating spot instance"
TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
IID=$(curl -sSH "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)
terminated=0
if [[ -n "$IID" ]]; then
    if aws ec2 terminate-instances --instance-ids "$IID" \
            --region "$AWS_DEFAULT_REGION" >>"$LOG" 2>&1; then
        echo "[bootstrap] terminate-instances OK for $IID"
        terminated=1
    else
        echo "[bootstrap] terminate-instances FAILED — scheduling OS halt"
    fi
fi
if (( terminated == 0 )); then
    shutdown -h +2 "dyf worker self-terminate" >/dev/null 2>&1 || \
        (sleep 120 && halt -p) &
fi
exit $ORC_RC
