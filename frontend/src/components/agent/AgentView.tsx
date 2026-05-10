import { FormEvent, useEffect, useMemo, useState } from 'react';
import {
  type ApiAgentResponse,
  type ApiAgentToolCall,
  type ApiAgentTrace,
  sendAgent,
} from '../../api/client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { CitationChip } from '../chat/CitationChip';

type AgentViewProps = {
  conversationId?: string;
};

type AgentMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  trace?: ApiAgentTrace | null;
  error?: string | null;
};

function _normalizeSourceId(raw: string): string {
  return raw.replace(/[),.;:!?]+$/g, '').trim();
}

function _extractSourceIds(text: string): string[] {
  const matches = text.match(/(?:src|chk):[^\s\]]+/g) ?? [];
  const unique = new Set<string>();
  for (const value of matches) {
    const normalized = _normalizeSourceId(value);
    if (normalized.startsWith('src:') || normalized.startsWith('chk:')) {
      unique.add(normalized);
    }
  }
  return Array.from(unique);
}

function _toPaperSourceId(sourceId: string): string {
  if (sourceId.startsWith('src:')) {
    return sourceId;
  }
  if (!sourceId.startsWith('chk:')) {
    return sourceId;
  }
  return sourceId.replace(/^chk:/, 'src:').replace(/:\d+$/, '');
}

function _dedupePaperSourceIds(sourceIds: string[]): string[] {
  const seen = new Set<string>();
  const deduped: string[] = [];

  for (const sourceId of sourceIds) {
    const paperId = _toPaperSourceId(sourceId);
    if (seen.has(paperId)) {
      continue;
    }
    seen.add(paperId);
    deduped.push(paperId);
  }

  return deduped;
}

function _dedupeResultsContentByPaperId(content: string): string {
  const seen = new Set<string>();
  const keptLines: string[] = [];

  for (const line of content.split('\n')) {
    const sourceIds = _extractSourceIds(line);
    if (sourceIds.length === 0) {
      keptLines.push(line);
      continue;
    }

    let hasNewPaper = false;
    for (const sourceId of sourceIds) {
      const paperId = _toPaperSourceId(sourceId);
      if (!seen.has(paperId)) {
        seen.add(paperId);
        hasNewPaper = true;
      }
    }

    if (hasNewPaper) {
      keptLines.push(line);
    }
  }

  return keptLines.join('\n');
}

function ToolCallRow({ toolCall }: { toolCall: ApiAgentToolCall }) {
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    setExpanded(false);
  }, [toolCall]);

  const summary = `${toolCall.tool_name} · ${toolCall.status.toUpperCase()} · ${Math.round(toolCall.duration_ms)}ms`;

  return (
    <div className="agent-tool-call-row">
      <button className="agent-tool-toggle" onClick={() => setExpanded((value) => !value)} type="button">
        <span className="query-trace-chevron">{expanded ? '▾' : '›'}</span>
        <span>{summary}</span>
      </button>
      {expanded ? (
        <div className="agent-tool-panel">
          <div className="agent-tool-block">
            <p>input</p>
            <pre>{JSON.stringify(toolCall.input, null, 2)}</pre>
          </div>
          <div className="agent-tool-block">
            <p>output</p>
            <pre>{toolCall.output}</pre>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function TracePanel({ trace }: { trace: ApiAgentTrace }) {
  return (
    <aside className="agent-trace-panel">
      <h3>Trace</h3>
      <div className="agent-trace-kv">Iterations: {trace.iterations}</div>
      <div className="agent-trace-kv">Model: {trace.model || '--'}</div>
      <div className="agent-trace-kv">Latency: {(trace.total_latency_ms / 1000).toFixed(1)} s</div>
      <div className="agent-trace-kv">Stop reason: {trace.stopped_reason || '--'}</div>
      <div className="agent-trace-kv">Tool calls: {trace.tool_calls.length}</div>
      {trace.warnings.length > 0 ? (
        <div className="agent-trace-warning-list">
          {trace.warnings.map((warning) => (
            <div key={warning} className="agent-trace-warning-item">
              {warning}
            </div>
          ))}
        </div>
      ) : null}
    </aside>
  );
}

export function AgentView({ conversationId }: AgentViewProps) {
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const latestTrace = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'assistant' && messages[i].trace) {
        return messages[i].trace;
      }
    }
    return null;
  }, [messages]);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || isLoading) {
      return;
    }

    const userMessage: AgentMessage = {
      id: `agent-user-${Date.now()}`,
      role: 'user',
      content: trimmed,
    };
    const history = [...messages, userMessage].map((message) => ({
      role: message.role,
      content: message.content,
    }));

    setQuery('');
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response: ApiAgentResponse = await sendAgent(trimmed, conversationId, history);
      const assistantMessage: AgentMessage = {
        id: `agent-assistant-${Date.now()}`,
        role: 'assistant',
        content: response.answer,
        trace: response.trace,
        error: response.error,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: `agent-error-${Date.now()}`,
          role: 'assistant',
          content: `Agent request failed: ${String(error)}`,
          trace: null,
          error: String(error),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section className="agent-view">
      <div className="agent-main-column">
        <div className="agent-task-header">
          <div className="agent-task-title">Task Runner</div>
          <div className="agent-task-subtitle">Plan, execute tools, then return grounded results.</div>
        </div>
        <div className="thread-body">
          {messages.length === 0 ? (
            <div className="thread-empty-state">
              <div className="thread-empty-logo" aria-hidden="true">
                AG
              </div>
              <p className="thread-empty-title">Describe a multi-step task</p>
              <div className="thread-empty-chip-row">
                <span className="thread-empty-chip">Inspect experiment runs and rank best precision</span>
                <span className="thread-empty-chip">Read notes and summarize open research questions</span>
                <span className="thread-empty-chip">Combine local files with knowledge-base evidence</span>
              </div>
            </div>
          ) : null}

          {messages.map((message) => (
            message.role === 'assistant' ? (
              <article className="answer-bubble" key={message.id}>
                {(() => {
                  const displayContent = _dedupeResultsContentByPaperId(message.content)
                    .replace(/\[\[[\w:/.-]+\]\]/g, '')
                    .split('\n')
                    .map((line) => line.trimEnd())
                    .join('\n')
                    .trim();
                  const sourceIds = _dedupePaperSourceIds(_extractSourceIds(message.content));

                  return (
                    <>
                      <div className="answer-prose">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayContent}</ReactMarkdown>
                      </div>
                      {sourceIds.length > 0 ? (
                        <div className="citations-row">
                          {sourceIds.map((sourceId) => (
                            <CitationChip key={`${message.id}-${sourceId}`} sourceId={sourceId} />
                          ))}
                        </div>
                      ) : null}
                    </>
                  );
                })()}
                {message.error ? <div className="agent-error-line">{message.error}</div> : null}
                {message.trace ? (
                  <div className="agent-tool-call-list">
                    {message.trace.tool_calls.map((toolCall, index) => (
                      <ToolCallRow key={`${message.id}-${toolCall.tool_name}-${index}`} toolCall={toolCall} />
                    ))}
                  </div>
                ) : null}
              </article>
            ) : (
              <div className="user-message-row" key={message.id}>
                <div className="user-message-bubble">{message.content}</div>
              </div>
            )
          ))}

          {isLoading ? (
            <div className="thread-thinking" aria-live="polite">
              <span>Agent reasoning</span>
              <span className="thread-thinking-dots" aria-hidden="true">
                <span>.</span>
                <span>.</span>
                <span>.</span>
              </span>
            </div>
          ) : null}
        </div>

        <form className="thread-input-bar" onSubmit={handleSubmit}>
          <input
            aria-label="Describe a task"
            className="thread-input"
            disabled={isLoading}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Describe a task involving papers, files, or experiments..."
            type="text"
            value={query}
          />
          <button className="thread-send-btn" disabled={isLoading || !query.trim()} type="submit">
            Run Task
          </button>
        </form>
      </div>

      <div className="agent-trace-column">
        {latestTrace ? (
          <TracePanel trace={latestTrace} />
        ) : (
          <aside className="agent-trace-panel is-empty">
            <h3>Trace</h3>
            <p>Run the agent to see iterations, tool calls, and stop reason.</p>
          </aside>
        )}
      </div>
    </section>
  );
}
