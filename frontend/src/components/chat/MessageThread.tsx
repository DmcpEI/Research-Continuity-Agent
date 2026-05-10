import { FormEvent, useEffect, useRef, useState } from 'react';
import { AnswerBubble } from './AnswerBubble';
import type { Message } from '../../hooks/useChat';

type MessageThreadProps = {
  conversationId: string;
  messages: Message[];
  isLoading: boolean;
  onSend: (query: string) => Promise<void>;
  onCitationClick?: (sourceId: string) => void;
  showWelcomeEmptyState?: boolean;
  inputDisabledReason?: string | null;
};

const SUGGESTED_QUESTIONS = [
  'What changed in my retrieval quality over the last week?',
  'Summarize unresolved TODOs from recent experiments.',
  'Which sources most strongly support my current hypothesis?',
];

export function MessageThread({
  conversationId,
  messages,
  isLoading,
  onSend,
  onCitationClick,
  showWelcomeEmptyState = false,
  inputDisabledReason = null,
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
            <div className="thread-empty-chip-row">
              {SUGGESTED_QUESTIONS.map((suggestion) => (
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
              onCitationClick={onCitationClick}
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

        {isLoading ? (
          <div className="thread-thinking" aria-live="polite">
            <span>Thinking</span>
            <span className="thread-thinking-dots" aria-hidden="true">
              <span>.</span>
              <span>.</span>
              <span>.</span>
            </span>
          </div>
        ) : null}
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
