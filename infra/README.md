# AWS Demo Deployment

These artifacts are intentionally **deployment-ready**, not "always on by default."

The cheapest useful flow for a student portfolio is:
- bake the current `.rca/` corpus into the image
- push to ECR
- run a one-off Fargate task with a public IP
- demo the app
- tear the task down immediately

That is what [`demo.sh`](./demo.sh) automates.

## Modes

### 1. Cheap demo mode

`./infra/demo.sh`

- builds and pushes the image to ECR
- registers a fresh ECS task definition
- runs a single Fargate task with a public IP
- waits for Streamlit `/_stcore/health`
- prints the demo URL
- tears the task down when you press enter

This mode is the best fit for interview screenshots or short demos because it avoids keeping an ALB and ECS service running.

### 2. Service mode

`./infra/deploy.sh`

- builds and pushes the image
- registers a task definition
- creates or updates an ECS service
- expects an existing cluster, subnets, security groups, and target group

Use this only if you deliberately want a longer-lived deployment.

## Required environment variables

Common:
- `AWS_REGION`
- `AWS_ACCOUNT_ID`
- `RCA_ECR_REPOSITORY`
- `RCA_EXECUTION_ROLE_ARN`
- `RCA_TASK_ROLE_ARN`
- `RCA_OPENAI_API_KEY_SECRET_ARN`

Demo mode:
- `RCA_CLUSTER_NAME`
- `RCA_SUBNETS` comma-separated
- `RCA_SECURITY_GROUPS` comma-separated

Service mode adds:
- `RCA_SERVICE_NAME`
- `RCA_TARGET_GROUP_ARN`
- `RCA_ALB_URL` optional, for pretty output

Optional overrides:
- `IMAGE_TAG`
- `RCA_TASK_FAMILY`
- `RCA_TASK_CPU`
- `RCA_TASK_MEMORY`
- `RCA_OPENAI_BASE_URL`
- `RCA_OPENAI_CHAT_MODEL`
- `RCA_OPENAI_EMBED_MODEL`
- `RCA_ENABLE_FILESYSTEM_TOOLS`
- `RCA_FILESYSTEM_ROOT`

## Production defaults baked into the image

The production image defaults to:
- `RCA_LLM_BACKEND=openai_compatible`
- `RCA_ENABLE_FILESYSTEM_TOOLS=false`
- baked-in `.rca/` demo corpus

Local Docker Compose overrides the backend back to `ollama`.

## Cost notes

- A one-off task still costs money while it is running, but much less than leaving an ECS service plus ALB up full time.
- `demo.sh` is designed to keep the runtime short and explicit.
- The image stored in ECR may still incur small storage costs until deleted.
