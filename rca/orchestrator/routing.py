"""Intent routing and tool-policy enforcement."""

from __future__ import annotations

from enum import Enum

try:
    import yaml
except ImportError:
    yaml = None

from rca.config.settings import Settings, get_settings


class Intent(str, Enum):
    ingest = "ingest"
    retrieve = "retrieve"
    generate = "generate"


class Router:
    """Route user requests to flows and enforce configured tool policy."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def detect_intent(self, text: str) -> Intent:
        lowered = text.lower().strip()
        if lowered.startswith("ingest ") or lowered.startswith("index "):
            return Intent.ingest
        if "digest" in lowered or "weekly summary" in lowered:
            return Intent.generate
        return Intent.retrieve

    def flow_for_intent(self, intent: Intent) -> str:
        mapping = {
            Intent.ingest: "ingest_flow",
            Intent.retrieve: "retrieve_flow",
            Intent.generate: "generate_flow",
        }
        return mapping[intent]

    def load_tool_policies(self) -> dict[str, set[str]]:
        policy_text = self.settings.tool_policy_path.read_text(encoding="utf-8")
        if yaml is not None:
            raw = yaml.safe_load(policy_text) or {}
            flow_section = raw.get("flows", {})
            return {
                flow_name: set(flow_config.get("allowed_tools", []))
                for flow_name, flow_config in flow_section.items()
            }
        return self._parse_simple_yaml(policy_text)

    def enforce_tool_policy(self, flow_name: str, requested_tools: list[str]) -> list[str]:
        allowed_tools = self.load_tool_policies().get(flow_name, set())
        blocked_tools = [tool for tool in requested_tools if tool not in allowed_tools]
        if blocked_tools:
            blocked_list = ", ".join(blocked_tools)
            raise PermissionError(f"Flow '{flow_name}' is not allowed to call: {blocked_list}")
        return sorted(tool for tool in requested_tools if tool in allowed_tools)

    @staticmethod
    def _parse_simple_yaml(policy_text: str) -> dict[str, set[str]]:
        policies: dict[str, set[str]] = {}
        current_flow: str | None = None
        in_allowed_tools = False

        for line in policy_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))

            if indent == 2 and stripped.endswith(":") and stripped != "flows:":
                current_flow = stripped[:-1]
                policies[current_flow] = set()
                in_allowed_tools = False
                continue

            if current_flow and stripped == "allowed_tools:":
                in_allowed_tools = True
                continue

            if current_flow and in_allowed_tools and stripped.startswith("- "):
                policies[current_flow].add(stripped[2:].strip())

        return policies
