#!/usr/bin/env bash
# VM user-data: runs as root at first boot. Hands off to bootstrap.sh.
# Intentionally minimal — anything substantive belongs in bootstrap.sh.
#
# @@VAR@@ placeholders are substituted by render_user_data.sh before
# the launch template gets this script.

set -euo pipefail
exec > /var/log/user-data.log 2>&1
echo "[user-data] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting"

export REPO_URL="@@REPO_URL@@"
export REPO_REF="@@REPO_REF@@"
export CKPT_BUCKET="@@CKPT_BUCKET@@"
export WANDB_PROJECT="@@WANDB_PROJECT@@"
export WANDB_ENTITY="@@WANDB_ENTITY@@"
export WANDB_API_KEY="@@WANDB_API_KEY@@"
export AWS_DEFAULT_REGION="@@AWS_DEFAULT_REGION@@"
export JOB_ORDER="@@JOB_ORDER@@"
export ENTRY_SCRIPT="@@ENTRY_SCRIPT@@"
export VENV_S3_URL="@@VENV_S3_URL@@"

# Throwaway clone just to pick up scripts/bootstrap.sh. The real
# /workspace is set up by bootstrap.sh itself.
mkdir -p /opt/bootstrap
git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" /opt/bootstrap/dyf
exec bash /opt/bootstrap/dyf/scripts/bootstrap.sh
