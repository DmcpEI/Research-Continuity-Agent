#!/usr/bin/env python3
"""Render simple placeholder templates for AWS deployment artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _json_list_from_env(name: str) -> str:
    raw = os.environ.get(name, "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return json.dumps(values)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: render_template.py <template> <output>", file=sys.stderr)
        return 1

    template_path = Path(argv[1])
    output_path = Path(argv[2])
    template = template_path.read_text(encoding="utf-8")

    replacements = {
        "__AWS_REGION__": os.environ.get("AWS_REGION", ""),
        "__RCA_EXECUTION_ROLE_ARN__": os.environ.get("RCA_EXECUTION_ROLE_ARN", ""),
        "__RCA_TASK_ROLE_ARN__": os.environ.get("RCA_TASK_ROLE_ARN", ""),
        "__RCA_IMAGE_URI__": os.environ.get("RCA_IMAGE_URI", ""),
        "__RCA_TASK_FAMILY__": os.environ.get("RCA_TASK_FAMILY", "rca-streamlit"),
        "__RCA_TASK_CPU__": os.environ.get("RCA_TASK_CPU", "1024"),
        "__RCA_TASK_MEMORY__": os.environ.get("RCA_TASK_MEMORY", "2048"),
        "__RCA_LOG_GROUP__": os.environ.get("RCA_LOG_GROUP", "/ecs/rca-streamlit"),
        "__RCA_OPENAI_BASE_URL__": os.environ.get(
            "RCA_OPENAI_BASE_URL", "https://api.openai.com/v1"
        ),
        "__RCA_OPENAI_CHAT_MODEL__": os.environ.get("RCA_OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "__RCA_OPENAI_EMBED_MODEL__": os.environ.get(
            "RCA_OPENAI_EMBED_MODEL", "text-embedding-3-small"
        ),
        "__RCA_OPENAI_API_KEY_SECRET_ARN__": os.environ.get(
            "RCA_OPENAI_API_KEY_SECRET_ARN", ""
        ),
        "__RCA_ENABLE_FILESYSTEM_TOOLS__": os.environ.get(
            "RCA_ENABLE_FILESYSTEM_TOOLS", "false"
        ).lower(),
        "__RCA_FILESYSTEM_ROOT__": os.environ.get("RCA_FILESYSTEM_ROOT", "/app"),
        "__RCA_SERVICE_NAME__": os.environ.get("RCA_SERVICE_NAME", "rca-streamlit"),
        "__RCA_SUBNETS_JSON__": _json_list_from_env("RCA_SUBNETS"),
        "__RCA_SECURITY_GROUPS_JSON__": _json_list_from_env("RCA_SECURITY_GROUPS"),
        "__RCA_TARGET_GROUP_ARN__": os.environ.get("RCA_TARGET_GROUP_ARN", ""),
    }

    rendered = template
    missing = []
    for token, value in replacements.items():
        if token in rendered and value == "":
            missing.append(token)
        rendered = rendered.replace(token, value)

    if missing:
        print(
            "missing values for: " + ", ".join(sorted(missing)),
            file=sys.stderr,
        )
        return 2

    output_path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
