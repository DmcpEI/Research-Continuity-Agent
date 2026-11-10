import { fireEvent, render, screen } from '@testing-library/react';
import type { Message } from '../../hooks/useChat';
import { MessageThread } from './MessageThread';

const at = new Date('2026-11-10T12:00:00Z');
const failed: Message[] = [
  { id: 'u1', role: 'user', content: 'What is JamPacker?', timestamp: at },
  { id: 'e1', role: 'assistant', content: 'I could not complete the request.', outcome: 'error', timestamp: at },
];

describe('MessageThread retry', () => {
  // jsdom has no scrollIntoView; the thread calls it after every render.
  const original = window.HTMLElement.prototype.scrollIntoView;
  beforeEach(() => {
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });
  afterEach(() => {
    window.HTMLElement.prototype.scrollIntoView = original;
  });

  it('offers a retry on a failed request and passes its id', () => {
    const onRetry = vi.fn();
    render(
      <MessageThread conversationId="c1" isLoading={false} messages={failed} onRetry={onRetry} onSend={async () => undefined} />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));

    expect(onRetry).toHaveBeenCalledWith('e1');
  });

  it('offers a retry on a failed request saved before outcomes existed', () => {
    const onRetry = vi.fn();
    const oldError: Message = {
      id: 'msg-error-1700000000000',
      role: 'assistant',
      content: 'I could not complete the request.',
      grounded: false,
      timestamp: at,
    };
    render(
      <MessageThread
        conversationId="c1"
        isLoading={false}
        messages={[failed[0], oldError]}
        onRetry={onRetry}
        onSend={async () => undefined}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));

    expect(onRetry).toHaveBeenCalledWith('msg-error-1700000000000');
  });

  it('offers no retry while a request is running', () => {
    render(
      <MessageThread conversationId="c1" isLoading messages={failed} onRetry={vi.fn()} onSend={async () => undefined} />,
    );

    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
  });

  it('offers no retry on an older failure once the conversation moved on', () => {
    const movedOn: Message[] = [
      ...failed,
      { id: 'u2', role: 'user', content: 'Something else?', timestamp: at },
      { id: 'a2', role: 'assistant', content: 'An answer.', outcome: 'answered', timestamp: at },
    ];
    render(
      <MessageThread conversationId="c1" isLoading={false} messages={movedOn} onRetry={vi.fn()} onSend={async () => undefined} />,
    );

    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
  });

  it('offers no retry on a refusal: it is an answer, not a failure', () => {
    render(
      <MessageThread
        conversationId="c1"
        isLoading={false}
        messages={[failed[0], { ...failed[1], outcome: 'not_found' }]}
        onRetry={vi.fn()}
        onSend={async () => undefined}
      />,
    );

    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
  });
});
