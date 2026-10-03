"""Multi-turn MCP-backed agent loop."""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

from rca.agent.contracts import AgentResult, AgentTrace, ToolCallTrace
from rca.agent.tools import ToolRegistry
from rca.config.settings import Settings, get_settings
from rca.flows.retrieve_flow import RetrieveFlow
from rca.llm.client import ChatMessage, EchoLLMClient, LLMClient
from rca.llm.factory import get_llm_client

MAX_ITERATIONS = 10

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
- Be concise, useful, and grounded in the retrieved evidence
- Do not answer with only a tool call string; execute tools, then provide a final answer
- For search-style requests, return a ranked list with one bullet per source
- Each ranked bullet must include: title, relevance summary, score when available, and [[source_id]]
- Put caveats or uncertainty after the ranked findings, never before them
- Do not ask the user to rephrase unless no relevant results were found"""

TEXT_TOOL_CALL_PATTERN = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((\{.*\})\)\s*$", re.DOTALL)
SPECIFIC_QUERY_PATTERN = re.compile(r"\b(my|paper|thesis|pic2)\b", re.IGNORECASE)


class AgentLoop:
    def __init__(
        self,
        settings: Settings | None = None,
        registry: ToolRegistry | None = None,
        llm_client: LLMClient | None = None,
        retrieve_flow: RetrieveFlow | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if registry is not None and retrieve_flow is not None:
            raise ValueError("Pass registry or retrieve_flow, not both")
        self.registry = registry or ToolRegistry(self.settings, retrieve_flow=retrieve_flow)

        if llm_client is not None:
            self.llm = llm_client
        elif self.settings.agent_model:
            self.llm = get_llm_client(
                self.settings.model_copy(
                    update={
                        "generation_model": self.settings.agent_model,
                        "openai_chat_model": self.settings.agent_model,
                    }
                )
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

                decoded_calls = self._decode_tool_calls(tool_calls, trace)
                if not decoded_calls:
                    textual_call = self._decode_text_tool_call(content, trace)
                    if textual_call is not None:
                        decoded_calls = [textual_call]
                        trace.warnings.append("parsed textual tool call from model output")

                if not tool_calls and not decoded_calls:
                    # An empty final turn is a model failure, not an answer.
                    trace.stopped_reason = "final_answer" if content else "empty_final_answer"
                    if not content:
                        self._note_empty_answer(response.raw, trace)
                    trace.total_latency_ms = (perf_counter() - started_total) * 1000.0
                    answer = content or "I could not produce an answer from the available evidence."
                    return AgentResult(query=query, answer=answer, trace=trace)

                assistant_message = {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": tool_calls,
                }
                messages.append(assistant_message)

                if not decoded_calls:
                    trace.stopped_reason = "final_answer" if content else "unparsed_tool_call"
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

                    clarification = self._build_clarification_if_needed(
                        tool_name,
                        arguments,
                        tool_trace.output,
                    )
                    if clarification is not None:
                        trace.stopped_reason = "clarification_needed"
                        trace.total_latency_ms = (perf_counter() - started_total) * 1000.0
                        return AgentResult(query=query, answer=clarification, trace=trace)

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
            if self._is_tool_calling_unsupported(exc):
                trace.stopped_reason = "error"
                trace.total_latency_ms = (perf_counter() - started_total) * 1000.0
                trace.warnings.append(f"tool-calling unsupported for model: {exc}")
                return AgentResult(
                    query=query,
                    answer=(
                        "This model does not support agent mode tool-calling. "
                        "Please select a tool-capable agent model."
                    ),
                    trace=trace,
                    error=str(exc),
                )

            fallback_result = self._fallback_without_tool_calling(query, trace, started_total, exc)
            if fallback_result is not None:
                return fallback_result

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
    def _decode_text_tool_call(
        content: str,
        trace: AgentTrace,
    ) -> tuple[str, dict[str, Any]] | None:
        if not content:
            return None

        match = TEXT_TOOL_CALL_PATTERN.match(content)
        if match is None:
            return None

        tool_name = match.group(1)
        arguments_raw = match.group(2)
        try:
            arguments = json.loads(arguments_raw)
        except Exception:
            trace.warnings.append(f"text tool call arguments for {tool_name} were malformed")
            return None

        if not isinstance(arguments, dict):
            trace.warnings.append(f"text tool call arguments for {tool_name} were not an object")
            return None

        return tool_name, arguments

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

    @staticmethod
    def _note_empty_answer(raw: Any, trace: AgentTrace) -> None:
        """Say why the final turn was empty when the response shows it."""
        message = raw.get("message", {}) if isinstance(raw, dict) else {}
        if isinstance(message, dict) and message.get("thinking"):
            trace.warnings.append(
                "model returned reasoning but no answer text (thinking enabled or unsupported)"
            )
        if isinstance(raw, dict) and raw.get("done_reason") == "length":
            trace.warnings.append("model output hit the token limit before answering")

    @staticmethod
    def _is_tool_calling_unsupported(exc: Exception) -> bool:
        text = str(exc)
        # A 400 for an unsupported "think" flag is a config problem, not missing tool support.
        return "400" in text and "Bad Request" in text and "support thinking" not in text

    @staticmethod
    def _build_clarification_if_needed(
        tool_name: str,
        arguments: dict[str, Any],
        tool_output: str,
    ) -> str | None:
        if tool_name != "search_knowledge_base":
            return None

        query_value = arguments.get("query")
        query_text = query_value.strip() if isinstance(query_value, str) else ""
        if not query_text:
            return None

        if not AgentLoop._looks_like_specific_query(query_text):
            return None

        if "No relevant documents found." in tool_output:
            return (
                f"I couldn't find a paper matching '{query_text}' with high confidence. "
                "Could you provide the full title or author names?"
            )

        max_score = AgentLoop._extract_max_score(tool_output)
        has_results = "RESULT " in tool_output or "[src:" in tool_output or "[chk:" in tool_output
        if has_results and (max_score is None or max_score < 0.6):
            return (
                f"I couldn't find a paper matching '{query_text}' with high confidence. "
                "Could you provide the full title or author names?"
            )

        return None

    @staticmethod
    def _looks_like_specific_query(query_text: str) -> bool:
        if SPECIFIC_QUERY_PATTERN.search(query_text):
            return True
        words = [token for token in re.split(r"\s+", query_text.strip()) if token]
        return 0 < len(words) <= 3

    @staticmethod
    def _extract_max_score(tool_output: str) -> float | None:
        matches = re.findall(r"score\s*[:=]\s*([0-9]*\.?[0-9]+)", tool_output)
        if not matches:
            return None

        best = None
        for match in matches:
            try:
                value = float(match)
            except ValueError:
                continue
            if best is None or value > best:
                best = value
        return best

    def _fallback_without_tool_calling(
        self,
        query: str,
        trace: AgentTrace,
        started_total: float,
        exc: Exception,
    ) -> AgentResult | None:
        """Fallback path when the selected model rejects tool-calling payloads."""
        if trace.tool_calls:
            return None

        error_text = str(exc)
        if "400" in error_text and "Bad Request" in error_text:
            return None

        if not error_text:
            return None

        trace.warnings.append(f"tool-calling unavailable: {error_text}")

        kb_trace = self.registry.call("search_knowledge_base", {"query": query, "limit": 5})
        trace.tool_calls.append(kb_trace)

        try:
            fallback_messages = [
                ChatMessage(
                    role="system",
                    content=(
                        "You are RCA. Tool-calling is unavailable, but evidence was retrieved. "
                        "Answer using only the evidence below. If insufficient, say so clearly. "
                        "When evidence exists, provide ranked findings first and caveats last."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=(
                        f"Question: {query}\n\n"
                        f"Retrieved evidence:\n{kb_trace.output}\n\n"
                        "Provide a concise answer and preserve source IDs like [[src:...]] when present."
                    ),
                ),
            ]
            response = self.llm.chat(fallback_messages)
            self._accumulate_usage(trace, response.raw)
            answer = response.text.strip() or kb_trace.output
        except Exception as fallback_exc:  # pragma: no cover - defensive path
            trace.warnings.append(f"fallback generation failed: {fallback_exc}")
            answer = (
                "Tool-calling is unavailable for the selected model. "
                "Here is the retrieved evidence:\n\n"
                f"{kb_trace.output}"
            )

        trace.stopped_reason = "fallback_no_tools"
        trace.total_latency_ms = (perf_counter() - started_total) * 1000.0
        trace.iterations = max(trace.iterations, 1)
        return AgentResult(query=query, answer=answer, trace=trace, error=None)
