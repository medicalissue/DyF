#!/usr/bin/env bash
# Launch ONE bare-EC2 spot worker.
#
# Usage:
#   source .env
#   bash scripts/infra/run_worker.sh <az> [<instance-type>] [<name-suffix>]
#
# Positional:
#   az              us-west-2a|b|c|d. Picks the matching subnet.
#   instance-type   Defaults to $INSTANCE_TYPE (.env), then p5.48xlarge.
#   name-suffix     Tag suffix; defaults to current UTC timestamp.
#
# /data is provisioned in-place from $DATA_SNAPSHOT via a
# BlockDeviceMapping on run-instances — no pre-create step.
# DeleteOnTermination=true cleans the volume on spot preemption.
#
# Required env (.env):
#   AMI                 AMI ID (Deep Learning Base AMI for your region)
#   SG                  security group id (allow ssh 22 if you want SSH)
#   KEY                 EC2 key pair name (omit for IMDS-only access)
#   IAM_PROFILE         dyf-worker-profile
#   DATA_SNAPSHOT       snap-... (dataset snapshot)
#   SUBNET_<AZ>         subnet id per AZ (e.g. SUBNET_us_west_2d=subnet-…)
#   REPO_URL, REPO_REF, CKPT_BUCKET, WANDB_*  — for render_user_data.sh

set -euo pipefail

AZ="${1:?az required (e.g. us-west-2d)}"
INSTANCE_TYPE="${2:-${INSTANCE_TYPE:-p5.48xlarge}}"
NAME_SUFFIX="${3:-$(date -u +%Y%m%dT%H%M%S)}"

: "${AMI:?AMI required (Deep Learning Base AMI ID for your region)}"
: "${SG:?SG required (security group id)}"
: "${IAM_PROFILE:=dyf-worker-profile}"
: "${DATA_SNAPSHOT:?DATA_SNAPSHOT required (e.g. snap-0adfaa42ce378623c)}"
: "${REGION:=${AWS_DEFAULT_REGION:-us-west-2}}"

# Subnet lookup: SUBNET_us_west_2d, SUBNET_us_west_2c, …
subnet_var="SUBNET_$(echo "$AZ" | tr '-' '_')"
SUBNET="${!subnet_var:-}"
[[ -z "$SUBNET" ]] && {
    echo "ERROR: $subnet_var not set in .env (subnet for $AZ)" >&2; exit 2;
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DATA_B64=$(bash "$SCRIPT_DIR/render_user_data.sh" | base64 | tr -d '\n')

NAME="dyf-worker-${NAME_SUFFIX}"

echo "▶ launching ${NAME} in ${AZ} (${INSTANCE_TYPE})"
key_args=()
if [[ -n "${KEY:-}" ]]; then
    key_args=(--key-name "$KEY")
fi

aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    "${key_args[@]}" \
    --subnet-id "$SUBNET" \
    --security-group-ids "$SG" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --instance-market-options 'MarketType=spot' \
    --block-device-mappings '[
        {"DeviceName":"/dev/sda1",
         "Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true}},
        {"DeviceName":"/dev/sdg",
         "Ebs":{"SnapshotId":"'"$DATA_SNAPSHOT"'","VolumeSize":500,"VolumeType":"gp3",
                "Iops":16000,"Throughput":1000,"DeleteOnTermination":true}}
    ]' \
    --user-data "$USER_DATA_B64" \
    --tag-specifications "ResourceType=instance,Tags=[
        {Key=Name,Value=$NAME},
        {Key=Project,Value=dyf},
        {Key=Role,Value=worker},
        {Key=Campaign,Value=${CAMPAIGN:-dyf}}
    ]" \
    --query 'Instances[0].{InstanceId:InstanceId,AZ:Placement.AvailabilityZone,State:State.Name}' \
    --output json
