#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION}"
: "${RCA_CLUSTER_NAME:?set RCA_CLUSTER_NAME}"
: "${RCA_SERVICE_NAME:?set RCA_SERVICE_NAME}"

aws ecs update-service \
  --region "${AWS_REGION}" \
  --cluster "${RCA_CLUSTER_NAME}" \
  --service "${RCA_SERVICE_NAME}" \
  --desired-count 0 >/dev/null

aws ecs wait services-stable \
  --region "${AWS_REGION}" \
  --cluster "${RCA_CLUSTER_NAME}" \
  --services "${RCA_SERVICE_NAME}"

echo "Service scaled to 0."
