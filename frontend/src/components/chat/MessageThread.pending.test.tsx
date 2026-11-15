import { act, render, screen } from '@testing-library/react';
import { MessageThread } from './MessageThread';

describe('Pending block while a question runs', () => {
  const original = window.HTMLElement.prototype.scrollIntoView;
  beforeEach(() => {
    vi.useFakeTimers();
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });
  afterEach(() => {
    vi.useRealTimers();
    window.HTMLElement.prototype.scrollIntoView = original;
  });

  it('shows the current stage in words, the seconds so far, and a Stop button', () => {
    const onStop = vi.fn();
    const started = Date.now();
    const { rerender } = render(
      <MessageThread conversationId="c1" messages={[]} isLoading onSend={vi.fn()} stage="retrieving" startedAt={started} onStop={onStop} />,
    );
    expect(screen.getByText('Searching your library')).toBeInTheDocument();
    expect(screen.getByText('0 s')).toBeInTheDocument();

    act(() => vi.advanceTimersByTime(3000));
    rerender(
      <MessageThread conversationId="c1" messages={[]} isLoading onSend={vi.fn()} stage="generating" startedAt={started} onStop={onStop} />,
    );
    expect(screen.getByText('Writing the answer')).toBeInTheDocument();
    expect(screen.getByText('3 s')).toBeInTheDocument();
    expect(screen.queryByText(/Thinking/)).not.toBeInTheDocument();

    screen.getByRole('button', { name: 'Stop' }).click();
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it('shows a stage name it does not know as plain progress, not an empty label', () => {
    render(
      <MessageThread conversationId="c1" messages={[]} isLoading onSend={vi.fn()} stage={'reranking' as never} startedAt={Date.now()} />,
    );
    expect(screen.getByText('Working')).toBeInTheDocument();
  });
});
