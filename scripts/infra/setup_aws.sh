#!/usr/bin/env bash
# One-shot AWS setup for the DyF campaign.
#
# Idempotent — re-running is safe; it skips anything that already exists.
# Creates:
#   1. CKPT_BUCKET (s3://dyf-checkpoints by default)
#   2. IAM role `dyf-worker-role` with the trust + worker policies
#   3. IAM instance profile `dyf-worker-profile` with that role attached
#
# Pre-conditions:
#   * AWS CLI v2 installed and configured (`aws sts get-caller-identity` works)
#   * Caller has IAM + S3 admin permissions (one-time setup)
#
# Usage:
#   source .env       # for CKPT_BUCKET, AWS_DEFAULT_REGION
#   bash scripts/infra/setup_aws.sh
#
# After this, scripts/infra/run_worker.sh and the dstack worker task
# can both reference `dyf-worker-profile` for AWS access.

set -euo pipefail

: "${CKPT_BUCKET:?CKPT_BUCKET required (e.g. s3://dyf-checkpoints)}"
: "${AWS_DEFAULT_REGION:=us-west-2}"
export AWS_DEFAULT_REGION

ROLE_NAME="${ROLE_NAME:-dyf-worker-role}"
PROFILE_NAME="${PROFILE_NAME:-dyf-worker-profile}"
POLICY_NAME="${POLICY_NAME:-dyf-worker-policy}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRUST="$SCRIPT_DIR/trust-policy.json"
POLICY="$SCRIPT_DIR/worker-policy.json"

bucket=$(echo "$CKPT_BUCKET" | sed -E 's|^s3://([^/]+).*|\1|')

log() { printf '[setup_aws %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# 1. Bucket
if aws s3api head-bucket --bucket "$bucket" 2>/dev/null; then
    log "bucket s3://$bucket exists — skipping"
else
    log "creating bucket s3://$bucket in $AWS_DEFAULT_REGION"
    if [[ "$AWS_DEFAULT_REGION" == "us-east-1" ]]; then
        aws s3api create-bucket --bucket "$bucket"
    else
        aws s3api create-bucket --bucket "$bucket" \
            --create-bucket-configuration "LocationConstraint=$AWS_DEFAULT_REGION"
    fi
fi

# 2. Role
if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    log "role $ROLE_NAME exists — skipping create"
else
    log "creating role $ROLE_NAME"
    aws iam create-role --role-name "$ROLE_NAME" \
        --assume-role-policy-document "file://$TRUST" >/dev/null
fi

log "(re)attaching inline policy $POLICY_NAME"
aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name "$POLICY_NAME" \
    --policy-document "file://$POLICY"

# 3. Instance profile
if aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" \
        >/dev/null 2>&1; then
    log "instance profile $PROFILE_NAME exists — skipping create"
else
    log "creating instance profile $PROFILE_NAME"
    aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
fi

# add-role-to-instance-profile is idempotent only via try/catch — it
# returns LimitExceeded if the role is already attached. Detect first.
existing=$(aws iam get-instance-profile \
    --instance-profile-name "$PROFILE_NAME" \
    --query 'InstanceProfile.Roles[0].RoleName' \
    --output text 2>/dev/null || echo "None")
if [[ "$existing" == "$ROLE_NAME" ]]; then
    log "role already attached to instance profile — skipping"
else
    log "attaching role $ROLE_NAME to profile $PROFILE_NAME"
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$PROFILE_NAME" \
        --role-name "$ROLE_NAME"
fi

cat <<EOF

✓ AWS setup complete.
    Bucket:           s3://$bucket
    IAM role:         $ROLE_NAME
    Instance profile: $PROFILE_NAME

Next:
    bash scripts/launch_workers.sh 2          # dstack-managed
  or
    bash scripts/launch_campaign.sh           # bare-EC2 + watchdog
EOF
