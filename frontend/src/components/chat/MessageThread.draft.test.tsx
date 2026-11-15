import { fireEvent, render, screen } from '@testing-library/react';
import { MessageThread } from './MessageThread';
import type { Message } from '../../hooks/useChat';

// Counts Markdown renders, so a test can see which answers re-render while the draft grows.
const rendered: string[] = [];
vi.mock('react-markdown', () => ({
  default: ({ children }: { children: string }) => {
    rendered.push(children);
    return <p>{children}</p>;
  },
}));

const history: Message[] = [
  { id: 'u1', role: 'user', content: 'What is JamPacker?', timestamp: new Date(0) },
  { id: 'a1', role: 'assistant', content: 'JamPacker packs boxes.', outcome: 'answered', grounded: true, timestamp: new Date(0) },
  { id: 'u2', role: 'user', content: 'And SayPlan?', timestamp: new Date(0) },
];
const onSend = vi.fn();
const onCitationClick = vi.fn();
const onRetry = vi.fn();

function thread(props: { draft?: string; isLoading?: boolean; onStop?: () => void }) {
  return (
    <MessageThread
      conversationId="c1"
      messages={history}
      isLoading={props.isLoading ?? true}
      onSend={onSend}
      onCitationClick={onCitationClick}
      onRetry={onRetry}
      stage="generating"
      startedAt={0}
      draft={props.draft ?? ''}
      onStop={props.onStop}
    />
  );
}

describe('While the draft grows', () => {
  const original = window.HTMLElement.prototype.scrollIntoView;
  beforeEach(() => {
    rendered.length = 0;
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });
  afterEach(() => {
    window.HTMLElement.prototype.scrollIntoView = original;
  });

  it('re-renders only the draft, not the answers above it', () => {
    const { rerender } = render(thread({ draft: 'Say' }));
    rendered.length = 0;
    rerender(thread({ draft: 'SayPlan uses' }));
    expect(rendered).toEqual(['SayPlan uses']);
  });

  it('does not scroll a conversation that is not waiting when the draft changes', () => {
    const scroll = vi.fn();
    window.HTMLElement.prototype.scrollIntoView = scroll;
    const { rerender } = render(thread({ isLoading: false, draft: 'Say' }));
    scroll.mockClear();
    rerender(thread({ isLoading: false, draft: 'SayPlan uses' }));
    expect(scroll).not.toHaveBeenCalled();
  });
});

describe('Stop in the input bar', () => {
  beforeEach(() => {
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it('ignores the second click of a double-click on Send', () => {
    const onStop = vi.fn();
    render(thread({ onStop }));
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }), { detail: 2 });
    expect(onStop).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }), { detail: 1 });
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it('does not inherit the focus Send had, so a second Enter cannot stop the question', () => {
    const onStop = vi.fn();
    const { rerender } = render(thread({ isLoading: false, onStop }));
    const send = screen.getByRole('button', { name: 'Send' });
    send.removeAttribute('disabled');
    send.focus();
    rerender(thread({ isLoading: true, onStop }));
    expect(document.activeElement).not.toBe(screen.getByRole('button', { name: 'Stop' }));
  });
});
