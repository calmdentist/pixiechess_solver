#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Launch a PixieChess proof EC2 instance with AWS CLI, upload the current checkout to S3,
and bootstrap the host so it can run scripts/aws_run_proof.sh via systemd.

Usage:
  scripts/aws_launch_training_instance.sh [options]

Options:
  --region REGION                    AWS region. Defaults to AWS_REGION, AWS_DEFAULT_REGION, or us-east-1.
  --instance-type TYPE               EC2 instance type. Default: auto (g6e.16xlarge -> g5.16xlarge -> g6.16xlarge -> g5.8xlarge)
  --volume-size-gb SIZE              Root EBS volume size. Default: 500
  --volume-iops IOPS                 gp3 IOPS. Default: 6000
  --volume-throughput MBPS           gp3 throughput in MiB/s. Default: 250
  --subnet-id SUBNET                 Subnet to launch into. Defaults to the default subnet in the default VPC.
  --bucket NAME                      S3 bucket for source bundle and run artifacts.
  --bucket-prefix PREFIX             Prefix under the bucket. Default: pixie-proof
  --run-name NAME                    Instance/run name tag. Default: pixie-proof-YYYYMMDD-HHMMSS
  --ami-parameter PARAM              Public SSM parameter for the DLAMI.
  --role-name NAME                   IAM role name. Default: pixie-training-ec2-role
  --instance-profile-name NAME       IAM instance profile name. Default: pixie-training-ec2-profile
  --security-group-name NAME         Security group name. Default: pixie-training-ec2
  --auto-start                       Start the proof run automatically after bootstrap.
  --no-auto-start                    Do not start the proof run automatically. Default behavior.
  -h, --help                         Show this help.

Training config:
  Export the same env vars used by scripts/aws_run_proof.sh before launching to override
  the default serious-run workload, for example:

    WORKERS=auto STRATEGY_PROVIDER=llm STRATEGY_CACHE_SCOPE=world_phase \\
    STRATEGY_REFRESH_ON_UNCERTAINTY=1 CURRICULUM_PROVIDER_MODE=live_llm \\
    RANDOMIZE_HANDAUTHORED_SPECIALS=0 TRAIN_GAMES=384 CYCLES=8 \\
    BENCHMARK_MANIFEST=data/benchmarks/frozen/phase0_serious_v0/manifest.json \\
    scripts/aws_launch_training_instance.sh --auto-start
EOF
}

log() {
  printf '[aws-launch] %s\n' "$*" >&2
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

required_vcpus_for_instance_type() {
  case "$1" in
    g6e.16xlarge|g5.16xlarge|g6.16xlarge)
      echo "64"
      ;;
    g5.8xlarge)
      echo "32"
      ;;
    *)
      echo ""
      ;;
  esac
}

gpu_vt_quota_value() {
  aws service-quotas list-service-quotas \
    --service-code ec2 \
    --region "$REGION" \
    --query "Quotas[?QuotaCode=='L-DB2E81BA'].Value | [0]" \
    --output text 2>/dev/null || true
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
INSTANCE_TYPE=""
INSTANCE_TYPE_SPECIFIED="0"
VOLUME_SIZE_GB="500"
VOLUME_IOPS="6000"
VOLUME_THROUGHPUT="250"
SUBNET_ID=""
BUCKET=""
BUCKET_PREFIX="pixie-proof"
RUN_NAME="pixie-proof-$(date -u +%Y%m%d-%H%M%S)"
AMI_PARAMETER="/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id"
ROLE_NAME="pixie-training-ec2-role"
INSTANCE_PROFILE_NAME="pixie-training-ec2-profile"
SECURITY_GROUP_NAME="pixie-training-ec2"
AUTO_START="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="${2:?missing value for --region}"
      shift 2
      ;;
    --instance-type)
      INSTANCE_TYPE="${2:?missing value for --instance-type}"
      INSTANCE_TYPE_SPECIFIED="1"
      shift 2
      ;;
    --volume-size-gb)
      VOLUME_SIZE_GB="${2:?missing value for --volume-size-gb}"
      shift 2
      ;;
    --volume-iops)
      VOLUME_IOPS="${2:?missing value for --volume-iops}"
      shift 2
      ;;
    --volume-throughput)
      VOLUME_THROUGHPUT="${2:?missing value for --volume-throughput}"
      shift 2
      ;;
    --subnet-id)
      SUBNET_ID="${2:?missing value for --subnet-id}"
      shift 2
      ;;
    --bucket)
      BUCKET="${2:?missing value for --bucket}"
      shift 2
      ;;
    --bucket-prefix)
      BUCKET_PREFIX="${2:?missing value for --bucket-prefix}"
      shift 2
      ;;
    --run-name)
      RUN_NAME="${2:?missing value for --run-name}"
      shift 2
      ;;
    --ami-parameter)
      AMI_PARAMETER="${2:?missing value for --ami-parameter}"
      shift 2
      ;;
    --role-name)
      ROLE_NAME="${2:?missing value for --role-name}"
      shift 2
      ;;
    --instance-profile-name)
      INSTANCE_PROFILE_NAME="${2:?missing value for --instance-profile-name}"
      shift 2
      ;;
    --security-group-name)
      SECURITY_GROUP_NAME="${2:?missing value for --security-group-name}"
      shift 2
      ;;
    --auto-start)
      AUTO_START="1"
      shift
      ;;
    --no-auto-start)
      AUTO_START="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

require_cmd aws
require_cmd tar

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
if [[ -z "$ACCOUNT_ID" || "$ACCOUNT_ID" == "None" ]]; then
  fail "unable to determine AWS account id from sts get-caller-identity"
fi

if [[ -z "$BUCKET" ]]; then
  BUCKET="pixiechess-proof-${ACCOUNT_ID}-${REGION}"
fi

ARTIFACT_PREFIX="${BUCKET_PREFIX%/}/$RUN_NAME"
SOURCE_KEY="$ARTIFACT_PREFIX/source/source.tar.gz"
OUTPUT_PREFIX="$ARTIFACT_PREFIX/output"
OUTPUT_URI="s3://$BUCKET/$OUTPUT_PREFIX"
SHARED_ARTIFACT_PREFIX="${BUCKET_PREFIX%/}"

log "bundling current checkout from $REPO_ROOT"
ARCHIVE_PATH="$TMP_DIR/source.tar.gz"
tar \
  -C "$REPO_ROOT" \
  --exclude=".git" \
  --exclude=".context" \
  --exclude=".venv" \
  --exclude=".pytest_cache" \
  --exclude=".mypy_cache" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  --exclude="runs" \
  -czf "$ARCHIVE_PATH" \
  .

if ! aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
  log "creating bucket $BUCKET"
  if [[ "$REGION" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket \
      --bucket "$BUCKET" \
      --region "$REGION" \
      --create-bucket-configuration "LocationConstraint=$REGION" \
      >/dev/null
  fi
fi

log "uploading source bundle to s3://$BUCKET/$SOURCE_KEY"
aws s3 cp "$ARCHIVE_PATH" "s3://$BUCKET/$SOURCE_KEY" --only-show-errors >/dev/null

ASSUME_ROLE_FILE="$TMP_DIR/assume-role.json"
cat >"$ASSUME_ROLE_FILE" <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  log "creating IAM role $ROLE_NAME"
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "file://$ASSUME_ROLE_FILE" \
    >/dev/null
fi

aws iam attach-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore \
  >/dev/null

S3_POLICY_FILE="$TMP_DIR/s3-policy.json"
cat >"$S3_POLICY_FILE" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListRunPrefix",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::$BUCKET"
      ],
      "Condition": {
        "StringLike": {
          "s3:prefix": [
            "$SHARED_ARTIFACT_PREFIX",
            "$SHARED_ARTIFACT_PREFIX/*"
          ]
        }
      }
    },
    {
      "Sid": "ReadWriteRunObjects",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::$BUCKET/$SHARED_ARTIFACT_PREFIX/*"
      ]
    }
  ]
}
EOF

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "${ROLE_NAME}-run-bucket" \
  --policy-document "file://$S3_POLICY_FILE" \
  >/dev/null

if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" >/dev/null 2>&1; then
  log "creating instance profile $INSTANCE_PROFILE_NAME"
  aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" >/dev/null
fi

PROFILE_ROLE_COUNT="$(aws iam get-instance-profile \
  --instance-profile-name "$INSTANCE_PROFILE_NAME" \
  --query "InstanceProfile.Roles[?RoleName=='$ROLE_NAME'] | length(@)" \
  --output text)"
if [[ "$PROFILE_ROLE_COUNT" == "0" ]]; then
  log "adding role $ROLE_NAME to instance profile $INSTANCE_PROFILE_NAME"
  aws iam add-role-to-instance-profile \
    --instance-profile-name "$INSTANCE_PROFILE_NAME" \
    --role-name "$ROLE_NAME" \
    >/dev/null
fi

sleep 10

if [[ -n "$SUBNET_ID" ]]; then
  VPC_ID="$(aws ec2 describe-subnets \
    --region "$REGION" \
    --subnet-ids "$SUBNET_ID" \
    --query 'Subnets[0].VpcId' \
    --output text)"
  CANDIDATE_SUBNET_IDS=("$SUBNET_ID")
else
  VPC_ID="$(aws ec2 describe-vpcs \
    --region "$REGION" \
    --filters Name=isDefault,Values=true \
    --query 'Vpcs[0].VpcId' \
    --output text)"
  if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
    fail "no default VPC found in $REGION; rerun with --subnet-id"
  fi
  CANDIDATE_SUBNET_IDS=()
  while IFS= read -r candidate_subnet_id; do
    if [[ -n "$candidate_subnet_id" ]]; then
      CANDIDATE_SUBNET_IDS+=("$candidate_subnet_id")
    fi
  done < <(
    aws ec2 describe-subnets \
      --region "$REGION" \
      --filters Name=vpc-id,Values="$VPC_ID" Name=default-for-az,Values=true \
      --query 'Subnets[].[SubnetId,AvailabilityZone]' \
      --output text \
      | sort -k2,2 \
      | awk '{print $1}'
  )
fi

if [[ "${#CANDIDATE_SUBNET_IDS[@]}" -eq 0 ]]; then
  fail "unable to determine subnet id"
fi

SG_ID="$(aws ec2 describe-security-groups \
  --region "$REGION" \
  --filters Name=vpc-id,Values="$VPC_ID" Name=group-name,Values="$SECURITY_GROUP_NAME" \
  --query 'SecurityGroups[0].GroupId' \
  --output text 2>/dev/null || true)"
if [[ -z "$SG_ID" || "$SG_ID" == "None" ]]; then
  log "creating security group $SECURITY_GROUP_NAME"
  SG_ID="$(aws ec2 create-security-group \
    --region "$REGION" \
    --vpc-id "$VPC_ID" \
    --group-name "$SECURITY_GROUP_NAME" \
    --description "PixieChess training access via SSM only" \
    --query 'GroupId' \
    --output text)"
fi

AMI_ID="$(aws ssm get-parameter \
  --region "$REGION" \
  --name "$AMI_PARAMETER" \
  --query 'Parameter.Value' \
  --output text)"
if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
  fail "unable to resolve AMI from $AMI_PARAMETER"
fi

PASSTHROUGH_ENV_VARS=(
  RUN_PROFILE
  DEVICE
  SELFPLAY_DEVICE
  INFERENCE_DEVICE
  WORKERS
  SEED
  STRESS_GAMES
  STRESS_MAX_PLIES
  CYCLES
  TRAIN_GAMES
  VAL_GAMES
  SIMULATIONS
  MAX_PLIES
  EPOCHS_PER_CYCLE
  BATCH_SIZE
  INFERENCE_BATCH_SIZE
  INFERENCE_MAX_WAIT_MS
  ARENA_GAMES
  ARENA_SIMULATIONS
  ARENA_MAX_PLIES
  PROMOTION_SCORE_THRESHOLD
  BATCHED_INFERENCE
  PROMOTION_GATE
  ANALYZE_AFTER_RUN
  MODEL_ARCHITECTURE
  ROOT_VALUE_TARGET_WEIGHT
  OUTCOME_TARGET_WEIGHT
  UNCERTAINTY_WEIGHT
  ADAPTIVE_SEARCH
  ADAPTIVE_MIN_SIMULATIONS
  ADAPTIVE_MAX_SIMULATIONS
  CURRICULUM_FOUNDATION_WEIGHT
  CURRICULUM_KNOWN_WEIGHT
  CURRICULUM_RECENT_WEIGHT
  CURRICULUM_COMPOSITION_WEIGHT
  CURRICULUM_RECENT_WINDOW
  STRATEGY_PROVIDER
  STRATEGY_FILE
  STRATEGY_CACHE_SCOPE
  STRATEGY_REFRESH_ON_UNCERTAINTY
  STRATEGY_REFRESH_UNCERTAINTY_THRESHOLD
  BENCHMARK_MANIFEST
  BENCHMARK_DIR
  PIECE_REGISTRY
  USE_VERIFIED_PIECES
  SPECIAL_PIECE_INCLUSION_PROBABILITY
  RANDOMIZE_HANDAUTHORED_SPECIALS
  REPLAY_WINDOW_CYCLES
  ENABLE_SCHEDULED_CURRICULUM
  CURRICULUM_PROVIDER_MODE
  CURRICULUM_TASKS
  RUN_THROUGHPUT_PREFLIGHT
  AUTO_TUNE_WORKERS
  THROUGHPUT_DIR
  THROUGHPUT_GAMES
  THROUGHPUT_REPEATS
  THROUGHPUT_WARMUP_RUNS
  THROUGHPUT_SIMULATIONS
  THROUGHPUT_MAX_PLIES
  THROUGHPUT_CHECKPOINT
  PERIODIC_SYNC
  SYNC_INTERVAL_SECONDS
  D_MODEL
  NUM_HEADS
  NUM_LAYERS
  DROPOUT
  FEEDFORWARD_MULTIPLIER
  RESUME_CHECKPOINT
  BEST_CHECKPOINT
  BEST_CHECKPOINT_OUT
  TRAIN_LOOP_EXTRA_ARGS
)
PASSTHROUGH_ENV_LINES=""
for var_name in "${PASSTHROUGH_ENV_VARS[@]}"; do
  if [[ -n "${!var_name:-}" ]]; then
    printf -v PASSTHROUGH_ENV_LINES '%s%s=%q\n' \
      "$PASSTHROUGH_ENV_LINES" \
      "$var_name" \
      "${!var_name}"
  fi
done

ENV_FILE_CONTENT=""
printf -v ENV_FILE_CONTENT '%s\n' \
  "RUN_NAME=$RUN_NAME" \
  "RUNS_ROOT=/opt/pixie_runs" \
  "OUTPUT_DIR=/opt/pixie_runs/$RUN_NAME" \
  "PIXIE_S3_OUTPUT_URI=$OUTPUT_URI" \
  "AWS_DEFAULT_REGION=$REGION" \
  "PYTHON_BIN=/opt/pytorch/bin/python"
ENV_FILE_CONTENT+=$'\n'"$PASSTHROUGH_ENV_LINES"

USER_DATA_FILE="$TMP_DIR/user-data.sh"
cat >"$USER_DATA_FILE" <<EOF
#!/bin/bash
set -euxo pipefail

exec > >(tee /var/log/pixie-user-data.log | logger -t pixie-user-data) 2>&1

SOURCE_URI="s3://$BUCKET/$SOURCE_KEY"
AUTO_START="$AUTO_START"
PYTHON_BIN="/opt/pytorch/bin/python"
export AWS_DEFAULT_REGION="$REGION"

if ! command -v aws >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y awscli
fi

mkdir -p /opt/pixie_solver /opt/pixie_runs /opt/pixie_bootstrap
aws s3 cp --region "\$AWS_DEFAULT_REGION" "\$SOURCE_URI" /opt/pixie_bootstrap/source.tar.gz
tar -xzf /opt/pixie_bootstrap/source.tar.gz -C /opt/pixie_solver --strip-components=1

if [[ ! -x "\$PYTHON_BIN" ]]; then
  echo "expected DLAMI python at \$PYTHON_BIN" >&2
  exit 1
fi

"\$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ required, found {sys.version}")
print(sys.version)
PY

"\$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
cd /opt/pixie_solver
"\$PYTHON_BIN" -m pip install -e .
"\$PYTHON_BIN" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

cat >/etc/pixie-proof.env <<'ENVEOF'
$ENV_FILE_CONTENT
ENVEOF

cat >/etc/systemd/system/pixie-proof.service <<'SERVICEEOF'
[Unit]
Description=PixieChess proof training run
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/pixie_solver
EnvironmentFile=/etc/pixie-proof.env
ExecStart=/bin/bash -lc 'cd /opt/pixie_solver && exec /opt/pixie_solver/scripts/aws_run_proof.sh >> /var/log/pixie-proof.log 2>&1'
Restart=no

[Install]
WantedBy=multi-user.target
SERVICEEOF

touch /var/log/pixie-proof.log
chown ubuntu:ubuntu /var/log/pixie-proof.log
chown -R ubuntu:ubuntu /opt/pixie_solver /opt/pixie_runs

systemctl daemon-reload
if [[ "\$AUTO_START" == "1" ]]; then
  systemctl enable --now pixie-proof.service
else
  systemctl enable pixie-proof.service
fi
EOF

BLOCK_DEVICE_FILE="$TMP_DIR/block-device.json"
cat >"$BLOCK_DEVICE_FILE" <<EOF
[
  {
    "DeviceName": "/dev/sda1",
    "Ebs": {
      "VolumeSize": $VOLUME_SIZE_GB,
      "VolumeType": "gp3",
      "Iops": $VOLUME_IOPS,
      "Throughput": $VOLUME_THROUGHPUT,
      "DeleteOnTermination": true,
      "Encrypted": true
    }
  }
]
EOF

if [[ "$INSTANCE_TYPE_SPECIFIED" == "1" ]]; then
  INSTANCE_TYPE_CANDIDATES=("$INSTANCE_TYPE")
else
  INSTANCE_TYPE_CANDIDATES=(g6e.16xlarge g5.16xlarge g6.16xlarge g5.8xlarge)
fi

ALL_CANDIDATES_USE_GPU_VT_BUCKET="1"
MIN_REQUIRED_VCPUS=""
MIN_REQUIRED_INSTANCE_TYPE=""
for candidate_instance_type in "${INSTANCE_TYPE_CANDIDATES[@]}"; do
  case "$candidate_instance_type" in
    g*|vt*)
      ;;
    *)
      ALL_CANDIDATES_USE_GPU_VT_BUCKET="0"
      ;;
  esac
  candidate_vcpus="$(required_vcpus_for_instance_type "$candidate_instance_type")"
  if [[ -n "$candidate_vcpus" ]]; then
    if [[ -z "$MIN_REQUIRED_VCPUS" || "$candidate_vcpus" -lt "$MIN_REQUIRED_VCPUS" ]]; then
      MIN_REQUIRED_VCPUS="$candidate_vcpus"
      MIN_REQUIRED_INSTANCE_TYPE="$candidate_instance_type"
    fi
  fi
done

if [[ "$ALL_CANDIDATES_USE_GPU_VT_BUCKET" == "1" ]]; then
  GPU_VT_QUOTA="$(gpu_vt_quota_value)"
  case "$GPU_VT_QUOTA" in
    ''|None|null)
      ;;
    [0-9]*|[0-9]*.[0-9]*)
      GPU_VT_QUOTA_INT="${GPU_VT_QUOTA%%.*}"
      if [[ -z "$GPU_VT_QUOTA_INT" ]]; then
        GPU_VT_QUOTA_INT="0"
      fi
      if [[ -n "$MIN_REQUIRED_VCPUS" && "$GPU_VT_QUOTA_INT" -lt "$MIN_REQUIRED_VCPUS" ]]; then
        fail \
"EC2 quota preflight failed in $REGION: Running On-Demand G and VT instances (quota code L-DB2E81BA) is $GPU_VT_QUOTA vCPU. The smallest configured candidate requires $MIN_REQUIRED_VCPUS vCPU ($MIN_REQUIRED_INSTANCE_TYPE). Request at least 32 vCPU to allow g5.8xlarge, or 64 vCPU for the default serious-run targets."
      fi
      ;;
  esac
fi

INSTANCE_ID=""
SELECTED_INSTANCE_TYPE=""
SELECTED_SUBNET_ID=""
LAST_RUN_INSTANCES_ERROR_FILE="$TMP_DIR/run-instances-last-error.log"
LAST_RUN_INSTANCES_ERROR=""

for candidate_instance_type in "${INSTANCE_TYPE_CANDIDATES[@]}"; do
  for candidate_subnet_id in "${CANDIDATE_SUBNET_IDS[@]}"; do
    log "launch attempt: instance_type=$candidate_instance_type subnet_id=$candidate_subnet_id"
    if INSTANCE_ID="$(aws ec2 run-instances \
      --region "$REGION" \
      --image-id "$AMI_ID" \
      --instance-type "$candidate_instance_type" \
      --count 1 \
      --iam-instance-profile "Name=$INSTANCE_PROFILE_NAME" \
      --subnet-id "$candidate_subnet_id" \
      --security-group-ids "$SG_ID" \
      --metadata-options HttpTokens=required,HttpEndpoint=enabled \
      --block-device-mappings "file://$BLOCK_DEVICE_FILE" \
      --tag-specifications \
        "ResourceType=instance,Tags=[{Key=Name,Value=$RUN_NAME},{Key=Project,Value=PixieChess},{Key=ManagedBy,Value=aws_launch_training_instance.sh}]" \
        "ResourceType=volume,Tags=[{Key=Name,Value=$RUN_NAME},{Key=Project,Value=PixieChess},{Key=ManagedBy,Value=aws_launch_training_instance.sh}]" \
      --user-data "file://$USER_DATA_FILE" \
      --query 'Instances[0].InstanceId' \
      --output text 2>"$LAST_RUN_INSTANCES_ERROR_FILE")"; then
      SELECTED_INSTANCE_TYPE="$candidate_instance_type"
      SELECTED_SUBNET_ID="$candidate_subnet_id"
      break 2
    fi
    LAST_RUN_INSTANCES_ERROR="$(<"$LAST_RUN_INSTANCES_ERROR_FILE")"
    log "launch attempt failed for instance_type=$candidate_instance_type subnet_id=$candidate_subnet_id"
  done
done

if [[ -z "$INSTANCE_ID" ]]; then
  if [[ "$LAST_RUN_INSTANCES_ERROR" == *"VcpuLimitExceeded"* ]]; then
    GPU_VT_QUOTA="$(gpu_vt_quota_value)"
    GPU_QUOTA_GUIDANCE=" Request at least 32 vCPU in Running On-Demand G and VT instances (quota code L-DB2E81BA) to allow g5.8xlarge, or 64 vCPU for the default serious-run targets."
    case "$GPU_VT_QUOTA" in
      ''|None|null)
        ;;
      *)
        GPU_QUOTA_GUIDANCE=" Running On-Demand G and VT instances (quota code L-DB2E81BA) is currently $GPU_VT_QUOTA vCPU in $REGION.$GPU_QUOTA_GUIDANCE"
        ;;
    esac
    fail "unable to launch an instance in $REGION using candidate types [${INSTANCE_TYPE_CANDIDATES[*]}]: $LAST_RUN_INSTANCES_ERROR$GPU_QUOTA_GUIDANCE"
  fi
  if [[ -n "$LAST_RUN_INSTANCES_ERROR" ]]; then
    fail "unable to launch an instance in $REGION using candidate types [${INSTANCE_TYPE_CANDIDATES[*]}]: $LAST_RUN_INSTANCES_ERROR"
  fi
  fail "unable to launch an instance in $REGION using candidate types [${INSTANCE_TYPE_CANDIDATES[*]}]"
fi

INSTANCE_TYPE="$SELECTED_INSTANCE_TYPE"
SUBNET_ID="$SELECTED_SUBNET_ID"

aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PRIVATE_IP="$(aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text)"

SSM_STATUS="unknown"
for _ in $(seq 1 40); do
  SSM_STATUS="$(aws ssm describe-instance-information \
    --region "$REGION" \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text 2>/dev/null || true)"
  if [[ "$SSM_STATUS" == "Online" ]]; then
    break
  fi
  sleep 15
done

cat <<EOF
instance_id=$INSTANCE_ID
region=$REGION
instance_type=$INSTANCE_TYPE
subnet_id=$SUBNET_ID
private_ip=$PRIVATE_IP
ssm_status=$SSM_STATUS
run_name=$RUN_NAME
bucket=$BUCKET
source_uri=s3://$BUCKET/$SOURCE_KEY
output_uri=$OUTPUT_URI
auto_start=$AUTO_START

Connect:
  aws ssm start-session --region $REGION --target $INSTANCE_ID

Bootstrap log:
  aws ssm send-command --region $REGION --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript --comment "Show bootstrap log" --parameters commands='["sudo tail -n 200 /var/log/pixie-user-data.log"]'

Training log:
  aws ssm send-command --region $REGION --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript --comment "Show training log" --parameters commands='["sudo tail -n 200 /var/log/pixie-proof.log"]'

Start training manually:
  aws ssm send-command --region $REGION --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript --comment "Start Pixie proof run" --parameters commands='["sudo systemctl start pixie-proof.service"]'

Service status:
  aws ssm send-command --region $REGION --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript --comment "Show service status" --parameters commands='["sudo systemctl status pixie-proof.service --no-pager"]'
EOF
