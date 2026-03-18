#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${AWS_REGION:?set AWS_REGION}"
: "${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID}"
: "${RCA_CLUSTER_NAME:?set RCA_CLUSTER_NAME}"
: "${RCA_SERVICE_NAME:?set RCA_SERVICE_NAME}"
: "${RCA_SUBNETS:?set RCA_SUBNETS (comma-separated)}"
: "${RCA_SECURITY_GROUPS:?set RCA_SECURITY_GROUPS (comma-separated)}"
: "${RCA_TARGET_GROUP_ARN:?set RCA_TARGET_GROUP_ARN}"
: "${RCA_EXECUTION_ROLE_ARN:?set RCA_EXECUTION_ROLE_ARN}"
: "${RCA_TASK_ROLE_ARN:?set RCA_TASK_ROLE_ARN}"
: "${RCA_OPENAI_API_KEY_SECRET_ARN:?set RCA_OPENAI_API_KEY_SECRET_ARN}"

BUILD_OUTPUT="$("${SCRIPT_DIR}/build_and_push.sh")"
export RCA_IMAGE_URI
RCA_IMAGE_URI="$(printf '%s\n' "${BUILD_OUTPUT}" | grep '^RCA_IMAGE_URI=' | tail -1 | cut -d= -f2-)"

TASK_DEF_RENDERED="$(mktemp)"
SERVICE_RENDERED="$(mktemp)"
python3 "${SCRIPT_DIR}/render_template.py" "${SCRIPT_DIR}/ecs-task-def.json" "${TASK_DEF_RENDERED}"
python3 "${SCRIPT_DIR}/render_template.py" "${SCRIPT_DIR}/ecs-service.json" "${SERVICE_RENDERED}"

TASK_DEF_ARN="$(aws ecs register-task-definition \
  --region "${AWS_REGION}" \
  --cli-input-json "file://${TASK_DEF_RENDERED}" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text)"

SERVICE_STATUS="$(aws ecs describe-services \
  --region "${AWS_REGION}" \
  --cluster "${RCA_CLUSTER_NAME}" \
  --services "${RCA_SERVICE_NAME}" \
  --query 'services[0].status' \
  --output text 2>/dev/null || true)"

if [[ -n "${SERVICE_STATUS}" && "${SERVICE_STATUS}" != "None" ]]; then
  aws ecs update-service \
    --region "${AWS_REGION}" \
    --cluster "${RCA_CLUSTER_NAME}" \
    --service "${RCA_SERVICE_NAME}" \
    --task-definition "${TASK_DEF_ARN}" \
    --desired-count 1 >/dev/null
else
  aws ecs create-service \
    --region "${AWS_REGION}" \
    --cluster "${RCA_CLUSTER_NAME}" \
    --service-name "${RCA_SERVICE_NAME}" \
    --task-definition "${TASK_DEF_ARN}" \
    --cli-input-json "file://${SERVICE_RENDERED}" >/dev/null
fi

aws ecs wait services-stable \
  --region "${AWS_REGION}" \
  --cluster "${RCA_CLUSTER_NAME}" \
  --services "${RCA_SERVICE_NAME}"

echo "Service is stable."
if [[ -n "${RCA_ALB_URL:-}" ]]; then
  echo "URL: ${RCA_ALB_URL}"
else
  echo "Set RCA_ALB_URL to print the final load balancer URL automatically."
fi
echo "To avoid charges, scale the service down when you're done:"
echo "  ./infra/teardown_service.sh"
