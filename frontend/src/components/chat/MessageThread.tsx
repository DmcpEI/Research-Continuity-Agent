import { FormEvent, useEffect, useRef, useState } from 'react';
import type { ChatStage } from '../../api/client';
import { AnswerBubble } from './AnswerBubble';
import { isFailedRequest, type Message } from '../../hooks/useChat';

type MessageThreadProps = {
  conversationId: string;
  messages: Message[];
  isLoading: boolean;
  onSend: (query: string) => Promise<void>;
  onRetry?: (errorId: string) => void;
  onCitationClick?: (sourceId: string) => void;
  showWelcomeEmptyState?: boolean;
  inputDisabledReason?: string | null;
  // Questions about papers in this library (see suggestions.ts); none while it is loading.
  suggestions?: string[];
  libraryEmpty?: boolean;
  // While a question runs: the server's current step, when it was sent, and how to stop it.
  stage?: ChatStage | null;
  startedAt?: number | null;
  onStop?: () => void;
};

const STAGE_TEXT: Record<ChatStage, string> = {
  retrieving: 'Searching your library',
  generating: 'Writing the answer',
  checking: 'Checking the answer',
};

function PendingBlock({ stage, startedAt, onStop }: { stage: ChatStage | null; startedAt: number | null; onStop?: () => void }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const seconds = startedAt === null ? 0 : Math.max(0, Math.floor((now - startedAt) / 1000));
  return (
    <div className="thread-pending" aria-live="polite">
      <span>{stage ? STAGE_TEXT[stage] : 'Sending the question'}</span>
      <span className="thread-pending-elapsed">{seconds} s</span>
      {onStop ? (
        <button className="thread-pending-stop" type="button" onClick={onStop}>
          Stop
        </button>
      ) : null}
    </div>
  );
}

export function MessageThread({
  conversationId,
  messages,
  isLoading,
  onSend,
  onRetry,
  onCitationClick,
  showWelcomeEmptyState = false,
  inputDisabledReason = null,
  suggestions = [],
  libraryEmpty = false,
  stage = null,
  startedAt = null,
  onStop,
}: MessageThreadProps) {
  const [query, setQuery] = useState('');
  const endRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const inputDisabled = isLoading || Boolean(inputDisabledReason);
  const inputTitle = inputDisabledReason ?? undefined;

  useEffect(() => {
    setQuery('');
  }, [conversationId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [isLoading, messages]);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
        if (!query.trim() || inputDisabled) {
      return;
    }

    const nextQuery = query;
    setQuery('');
    try {
      await onSend(nextQuery);
    } catch {
      // Sending failures already append an assistant-side error bubble.
    }
  };

  const shouldShowEmptyState = showWelcomeEmptyState && messages.length === 0 && !isLoading;

  // Only the latest turn can be retried, so a retry never reorders the conversation.
  const latest = [...messages].reverse().find((message) => message.role !== 'system');
  const retryId =
    onRetry && !inputDisabled && latest?.role === 'assistant' && isFailedRequest(latest) ? latest.id : null;

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    inputRef.current?.focus();
  };

  return (
    <div className="message-thread">
      <div className="thread-topbar" aria-hidden="true" />
      <div className="thread-body">
        {shouldShowEmptyState ? (
          <div className="thread-empty-state">
            <div className="thread-empty-logo" aria-hidden="true">
              RCA
            </div>
            <p className="thread-empty-title">Research Continuity Agent</p>
            {libraryEmpty ? (
              <p className="thread-empty-hint">Your library is empty. Add PDFs in Library, then ask about them.</p>
            ) : null}
            <div className="thread-empty-chip-row">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  className="thread-empty-chip"
                  type="button"
                  onClick={() => handleSuggestionClick(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {messages.map((message) =>
          message.role === 'assistant' ? (
            <AnswerBubble
              citations={message.citations ?? []}
              content={message.content}
              grounded={Boolean(message.grounded)}
              key={message.id}
              outcome={message.outcome}
              onCitationClick={onCitationClick}
              onRetry={message.id === retryId ? () => onRetry?.(message.id) : undefined}
              trace={message.trace ?? null}
            />
          ) : message.role === 'system' ? (
            <div className="model-switch-divider" key={message.id}>
              <span>{message.content}</span>
            </div>
          ) : (
            <div className="user-message-row" key={message.id}>
              <div className="user-message-bubble">{message.content}</div>
            </div>
          ),
        )}

        {isLoading ? <PendingBlock stage={stage} startedAt={startedAt} onStop={onStop} /> : null}
        <div ref={endRef} />
      </div>

      <form className="thread-input-bar" onSubmit={handleSubmit} title={inputTitle}>
        <input
          aria-label="Ask a question"
          className="thread-input"
          disabled={inputDisabled}
          ref={inputRef}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Ask a question about your research…"
          type="text"
          title={inputTitle}
          value={query}
        />
        <button
          className="thread-send-btn"
          disabled={inputDisabled || !query.trim()}
          type="submit"
          title={inputTitle}
        >
          Send
        </button>
      </form>
    </div>
  );
}
