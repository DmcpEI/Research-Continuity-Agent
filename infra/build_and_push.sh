#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${AWS_REGION:?set AWS_REGION}"
: "${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID}"
: "${RCA_ECR_REPOSITORY:?set RCA_ECR_REPOSITORY}"

IMAGE_TAG="${IMAGE_TAG:-$(git -C "${REPO_ROOT}" rev-parse --short HEAD)}"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${RCA_ECR_REPOSITORY}"
IMAGE_URI="${ECR_URI}:${IMAGE_TAG}"

echo "Ensuring ECR repository exists: ${RCA_ECR_REPOSITORY}"
if ! aws ecr describe-repositories --repository-names "${RCA_ECR_REPOSITORY}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "${RCA_ECR_REPOSITORY}" --region "${AWS_REGION}" >/dev/null
fi

echo "Logging into ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Building image ${IMAGE_URI}"
docker build -t "${IMAGE_URI}" "${REPO_ROOT}"

echo "Pushing image ${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo "RCA_IMAGE_URI=${IMAGE_URI}"
