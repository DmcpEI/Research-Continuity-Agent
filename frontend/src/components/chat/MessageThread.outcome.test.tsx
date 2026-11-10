import { render, screen } from '@testing-library/react';
import { MessageThread } from './MessageThread';

describe('MessageThread answer outcome', () => {
  // jsdom has no scrollIntoView; the thread calls it after every render.
  const original = window.HTMLElement.prototype.scrollIntoView;
  beforeEach(() => {
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });
  afterEach(() => {
    window.HTMLElement.prototype.scrollIntoView = original;
  });

  it('passes a stored refusal through to the badge', () => {
    render(
      <MessageThread
        conversationId="c1"
        isLoading={false}
        messages={[
          {
            id: 'm1',
            role: 'assistant',
            content: "I couldn't find evidence for this in your knowledge base.",
            outcome: 'not_found',
            grounded: false,
            timestamp: new Date(),
          },
        ]}
        onSend={async () => undefined}
      />,
    );

    expect(screen.getByText('No evidence found')).toBeInTheDocument();
  });
});
