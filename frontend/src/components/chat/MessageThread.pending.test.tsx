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

    // In the input bar, in place of Send, so it stays put while the draft grows above it.
    const stopButton = screen.getByRole('button', { name: 'Stop' });
    expect(stopButton.closest('form')).not.toBeNull();
    expect(screen.queryByRole('button', { name: 'Send' })).not.toBeInTheDocument();
    stopButton.click();
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it('shows a stage name it does not know as plain progress, not an empty label', () => {
    render(
      <MessageThread conversationId="c1" messages={[]} isLoading onSend={vi.fn()} stage={'reranking' as never} startedAt={Date.now()} />,
    );
    expect(screen.getByText('Working')).toBeInTheDocument();
  });
});

describe('Draft while the answer is written', () => {
  const original = window.HTMLElement.prototype.scrollIntoView;
  beforeEach(() => {
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });
  afterEach(() => {
    window.HTMLElement.prototype.scrollIntoView = original;
  });

  it('shows the draft under the stage, without citation markers', () => {
    render(
      <MessageThread
        conversationId="c1"
        messages={[]}
        isLoading
        onSend={vi.fn()}
        stage="generating"
        startedAt={Date.now()}
        draft="JamPacker packs boxes. [[chk:pdf/jampacker:0003]] It [src:pdf/jampacker] plans [[chk:pdf/jam"
      />,
    );
    expect(screen.getByText('Writing the answer')).toBeInTheDocument();
    const draft = screen.getByTestId('thread-draft');
    expect(draft).toHaveTextContent('JamPacker packs boxes. It plans');
    expect(draft.textContent).not.toContain('chk:');
    expect(draft.textContent).not.toContain('src:');
  });

  it('shows no draft block before the first piece arrives', () => {
    render(<MessageThread conversationId="c1" messages={[]} isLoading onSend={vi.fn()} stage="generating" startedAt={Date.now()} draft="" />);
    expect(screen.queryByTestId('thread-draft')).not.toBeInTheDocument();
  });
});

describe('Following the draft', () => {
  const original = window.HTMLElement.prototype.scrollIntoView;
  afterEach(() => {
    window.HTMLElement.prototype.scrollIntoView = original;
  });

  const noMessages: never[] = []; // stable, as the app's state is between pieces

  function renderDraft(draft: string) {
    return (
      <MessageThread conversationId="c1" messages={noMessages} isLoading onSend={vi.fn()} stage="generating" startedAt={Date.now()} draft={draft} />
    );
  }

  function placeBody(container: HTMLElement, distanceFromBottom: number) {
    const body = container.querySelector('.thread-body') as HTMLElement;
    Object.defineProperty(body, 'scrollHeight', { configurable: true, value: 2000 });
    Object.defineProperty(body, 'clientHeight', { configurable: true, value: 500 });
    body.scrollTop = 1500 - distanceFromBottom;
  }

  it('keeps the newest text in view while the reader is at the bottom', () => {
    const scroll = vi.fn();
    window.HTMLElement.prototype.scrollIntoView = scroll;
    const { container, rerender } = render(renderDraft('Jam'));
    placeBody(container, 0);
    scroll.mockClear();
    rerender(renderDraft('JamPacker packs'));
    expect(scroll).toHaveBeenCalled();
  });

  it('leaves the reader alone after they scroll up', () => {
    const scroll = vi.fn();
    window.HTMLElement.prototype.scrollIntoView = scroll;
    const { container, rerender } = render(renderDraft('Jam'));
    placeBody(container, 900);
    scroll.mockClear();
    rerender(renderDraft('JamPacker packs'));
    expect(scroll).not.toHaveBeenCalled();
  });
});
