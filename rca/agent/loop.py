"""Multi-turn MCP-backed agent loop."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from rca.agent.contracts import AgentResult, AgentTrace, ToolCallTrace
from rca.agent.tools import ToolRegistry
from rca.config.settings import Settings, get_settings
from rca.llm.client import EchoLLMClient, LLMClient, OllamaLLMClient

MAX_ITERATIONS = 8

SYSTEM_PROMPT = """You are RCA — a research continuity agent with access to tools.

You can:
- search the research knowledge base through a native adapter
- read and search files through MCP filesystem tools
- inspect experiment records through MCP experiments tools

Rules:
- For research questions, search the knowledge base before answering
- Use filesystem tools for local file inspection and text search
- Use experiment tools only for existing run inspection in this v1
- When using knowledge-base evidence, cite source IDs exactly as they appear, using [[source_id]]
- If tools do not provide enough evidence, say so clearly instead of guessing
- Be concise, useful, and grounded in the retrieved evidence"""


class AgentLoop:
    def __init__(
        self,
        settings: Settings | None = None,
        registry: ToolRegistry | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or ToolRegistry(self.settings)

        if llm_client is not None:
            self.llm = llm_client
        elif self.settings.generation_model:
            self.llm = OllamaLLMClient(
                base_url=self.settings.llm_base_url,
                model=self.settings.generation_model,
                api_key=self.settings.llm_api_key,
            )
        else:
            self.llm = EchoLLMClient()

    def run(self, query: str) -> AgentResult:
        trace = AgentTrace(
            query=query, model=getattr(self.llm, "model", self.llm.__class__.__name__)
        )
        started_total = perf_counter()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        tools = self.registry.ollama_tool_definitions()

        try:
            for iteration in range(MAX_ITERATIONS):
                trace.iterations = iteration + 1
                response = self._llm_call(messages, tools, trace)

                tool_calls = response.tool_calls or []
                content = response.text.strip()

                if not tool_calls:
                    trace.stopped_reason = "final_answer"
                    trace.total_latency_ms = (perf_counter() - started_total) * 1000.0
                    answer = content or "I could not produce an answer from the available evidence."
                    return AgentResult(query=query, answer=answer, trace=trace)

                assistant_message = {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": tool_calls,
                }
                messages.append(assistant_message)

                decoded_calls = self._decode_tool_calls(tool_calls, trace)
                if not decoded_calls:
                    trace.stopped_reason = "final_answer"
                    trace.total_latency_ms = (perf_counter() - started_total) * 1000.0
                    fallback = (
                        content or "I could not parse the tool request produced by the model."
                    )
                    return AgentResult(query=query, answer=fallback, trace=trace)

                for tool_name, arguments in decoded_calls:
                    tool_trace = self.registry.call(tool_name, arguments)
                    trace.tool_calls.append(tool_trace)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": tool_trace.output,
                        }
                    )

            trace.stopped_reason = "max_iterations"
            trace.total_latency_ms = (perf_counter() - started_total) * 1000.0
            return AgentResult(
                query=query,
                answer=(
                    "I reached the maximum number of reasoning steps. "
                    "Here is what I found so far:\n\n"
                    + self._summarize_tool_results(trace.tool_calls)
                ),
                trace=trace,
            )
        except Exception as exc:
            trace.stopped_reason = "error"
            trace.total_latency_ms = (perf_counter() - started_total) * 1000.0
            return AgentResult(
                query=query,
                answer=f"Agent error: {exc}",
                trace=trace,
                error=str(exc),
            )

    def close(self) -> None:
        self.registry.close()

    def _llm_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        trace: AgentTrace,
    ):
        response = self.llm.chat_with_tools(messages, tools)
        self._accumulate_usage(trace, response.raw)
        return response

    @staticmethod
    def _decode_tool_calls(
        tool_calls: list[dict[str, Any]],
        trace: AgentTrace,
    ) -> list[tuple[str, dict[str, Any]]]:
        decoded: list[tuple[str, dict[str, Any]]] = []

        for call in tool_calls:
            function = call.get("function", {})
            if not isinstance(function, dict):
                trace.warnings.append("ignored malformed tool call payload")
                continue

            tool_name = function.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                trace.warnings.append("ignored tool call without a name")
                continue

            arguments_payload = function.get("arguments", {})
            if isinstance(arguments_payload, str):
                try:
                    arguments = json.loads(arguments_payload)
                except Exception:
                    trace.warnings.append(f"tool call arguments for {tool_name} were malformed")
                    continue
            elif isinstance(arguments_payload, dict):
                arguments = arguments_payload
            else:
                trace.warnings.append(f"tool call arguments for {tool_name} had unsupported type")
                continue

            decoded.append((tool_name, arguments))

        return decoded

    @staticmethod
    def _summarize_tool_results(tool_calls: list[ToolCallTrace]) -> str:
        if not tool_calls:
            return "(no tool results)"

        lines = []
        for tool_call in tool_calls:
            preview = tool_call.output[:200]
            suffix = "…" if len(tool_call.output) > 200 else ""
            lines.append(f"**{tool_call.tool_name}**: {preview}{suffix}")
        return "\n\n".join(lines)

    @staticmethod
    def _accumulate_usage(trace: AgentTrace, raw: dict[str, Any]) -> None:
        prompt_tokens = 0
        completion_tokens = 0

        usage = raw.get("usage")
        if isinstance(usage, dict):
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )

        prompt_tokens += int(raw.get("prompt_tokens", raw.get("prompt_eval_count", 0)) or 0)
        completion_tokens += int(raw.get("completion_tokens", raw.get("eval_count", 0)) or 0)

        trace.prompt_tokens += prompt_tokens
        trace.completion_tokens += completion_tokens
