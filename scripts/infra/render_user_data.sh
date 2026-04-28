#!/usr/bin/env bash
# Render scripts/infra/user-data.sh with env-var substitution, output
# to stdout. Python handles substitution so arbitrary characters in
# values (slashes, pipes, newlines) pass through safely.
#
# Required env (usually sourced from .env):
#   REPO_URL, REPO_REF, CKPT_BUCKET, WANDB_API_KEY,
#   WANDB_PROJECT, WANDB_ENTITY, AWS_DEFAULT_REGION
# Optional:
#   JOB_ORDER       empty → orchestrate.sh uses default_job_order.txt
#   ENTRY_SCRIPT    empty → defaults to scripts/orchestrate.sh

set -euo pipefail

: "${REPO_URL:?REPO_URL required}"
: "${REPO_REF:=main}"
: "${CKPT_BUCKET:?CKPT_BUCKET required}"
: "${WANDB_API_KEY:?WANDB_API_KEY required (use empty string to disable W&B)}"
: "${WANDB_PROJECT:=dyf}"
: "${WANDB_ENTITY:=}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
: "${JOB_ORDER:=}"
: "${ENTRY_SCRIPT:=}"
: "${VENV_S3_URL:=s3://nelu-datasets/env/nelu-venv-py310-cu130.tar.gz}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
template="$SCRIPT_DIR/user-data.sh"

export REPO_URL REPO_REF CKPT_BUCKET WANDB_API_KEY WANDB_PROJECT \
       WANDB_ENTITY AWS_DEFAULT_REGION JOB_ORDER ENTRY_SCRIPT VENV_S3_URL

python3 - "$template" <<'PY'
import os, sys, pathlib
tpl = pathlib.Path(sys.argv[1]).read_text()
keys = [
    "REPO_URL", "REPO_REF", "CKPT_BUCKET",
    "WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY",
    "AWS_DEFAULT_REGION", "JOB_ORDER", "ENTRY_SCRIPT",
    "VENV_S3_URL",
]
for k in keys:
    tpl = tpl.replace(f"@@{k}@@", os.environ.get(k, ""))
sys.stdout.write(tpl)
PY
