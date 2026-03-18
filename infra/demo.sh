#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${AWS_REGION:?set AWS_REGION}"
: "${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID}"
: "${RCA_CLUSTER_NAME:?set RCA_CLUSTER_NAME}"
: "${RCA_SUBNETS:?set RCA_SUBNETS (comma-separated)}"
: "${RCA_SECURITY_GROUPS:?set RCA_SECURITY_GROUPS (comma-separated)}"
: "${RCA_EXECUTION_ROLE_ARN:?set RCA_EXECUTION_ROLE_ARN}"
: "${RCA_TASK_ROLE_ARN:?set RCA_TASK_ROLE_ARN}"
: "${RCA_OPENAI_API_KEY_SECRET_ARN:?set RCA_OPENAI_API_KEY_SECRET_ARN}"

export RCA_ENABLE_FILESYSTEM_TOOLS="${RCA_ENABLE_FILESYSTEM_TOOLS:-false}"
export RCA_FILESYSTEM_ROOT="${RCA_FILESYSTEM_ROOT:-/app}"

BUILD_OUTPUT="$("${SCRIPT_DIR}/build_and_push.sh")"
export RCA_IMAGE_URI
RCA_IMAGE_URI="$(printf '%s\n' "${BUILD_OUTPUT}" | grep '^RCA_IMAGE_URI=' | tail -1 | cut -d= -f2-)"

TASK_DEF_RENDERED="$(mktemp)"
python3 "${SCRIPT_DIR}/render_template.py" "${SCRIPT_DIR}/ecs-task-def.json" "${TASK_DEF_RENDERED}"

TASK_DEF_ARN="$(aws ecs register-task-definition \
  --region "${AWS_REGION}" \
  --cli-input-json "file://${TASK_DEF_RENDERED}" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)"

IFS=',' read -r -a SUBNETS <<<"${RCA_SUBNETS}"
IFS=',' read -r -a SECURITY_GROUPS <<<"${RCA_SECURITY_GROUPS}"
SUBNETS_JSON="$(printf '"%s",' "${SUBNETS[@]}")"
SUBNETS_JSON="[${SUBNETS_JSON%,}]"
SECURITY_GROUPS_JSON="$(printf '"%s",' "${SECURITY_GROUPS[@]}")"
SECURITY_GROUPS_JSON="[${SECURITY_GROUPS_JSON%,}]"

TASK_ARN="$(aws ecs run-task \
  --region "${AWS_REGION}" \
  --cluster "${RCA_CLUSTER_NAME}" \
  --launch-type FARGATE \
  --task-definition "${TASK_DEF_ARN}" \
  --network-configuration "awsvpcConfiguration={subnets=${SUBNETS_JSON},securityGroups=${SECURITY_GROUPS_JSON},assignPublicIp=ENABLED}" \
  --query 'tasks[0].taskArn' \
  --output text)"

cleanup() {
  if [[ -n "${TASK_ARN:-}" ]]; then
    aws ecs stop-task \
      --region "${AWS_REGION}" \
      --cluster "${RCA_CLUSTER_NAME}" \
      --task "${TASK_ARN}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Waiting for task to reach RUNNING"
aws ecs wait tasks-running --region "${AWS_REGION}" --cluster "${RCA_CLUSTER_NAME}" --tasks "${TASK_ARN}"

ENI_ID="$(aws ecs describe-tasks \
  --region "${AWS_REGION}" \
  --cluster "${RCA_CLUSTER_NAME}" \
  --tasks "${TASK_ARN}" \
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value | [0]" \
  --output text)"

PUBLIC_IP="$(aws ec2 describe-network-interfaces \
  --region "${AWS_REGION}" \
  --network-interface-ids "${ENI_ID}" \
  --query 'NetworkInterfaces[0].Association.PublicIp' \
  --output text)"

URL="http://${PUBLIC_IP}:8501"
echo "Waiting for Streamlit health check at ${URL}/_stcore/health"
for _ in $(seq 1 60); do
  if curl -fsS "${URL}/_stcore/health" >/dev/null 2>&1; then
    echo "Demo URL: ${URL}"
    read -r -p "Press enter to tear down the task..."
    break
  fi
  sleep 5
done

echo "Stopping task"
aws ecs stop-task \
  --region "${AWS_REGION}" \
  --cluster "${RCA_CLUSTER_NAME}" \
  --task "${TASK_ARN}" >/dev/null
aws ecs wait tasks-stopped --region "${AWS_REGION}" --cluster "${RCA_CLUSTER_NAME}" --tasks "${TASK_ARN}"
trap - EXIT
echo "Task stopped. Costs stop when the task stops; the pushed image remains in ECR."
