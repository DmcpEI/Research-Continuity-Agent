import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { sendChat } from '../../api/client';
import { ChatView } from './ChatView';

vi.mock('../../api/client', () => ({ sendChat: vi.fn(), getSources: vi.fn().mockResolvedValue([]) }));

describe('ChatView retry after a failed request', () => {
  // jsdom has no scrollIntoView; the thread calls it after every render.
  const original = window.HTMLElement.prototype.scrollIntoView;
  beforeEach(() => {
    window.localStorage.clear();
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
  });
  afterEach(() => {
    window.HTMLElement.prototype.scrollIntoView = original;
    vi.mocked(sendChat).mockReset();
  });

  it('sends the same question again and leaves one question and its answer', async () => {
    vi.mocked(sendChat).mockRejectedValueOnce(new Error('API 502')).mockResolvedValueOnce({
      answer: 'JamPacker recovers from failed placements.',
      citations: [],
      outcome: 'unverified',
      grounded: false,
      trace: null,
    });
    const user = userEvent.setup();
    const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <ChatView />
      </QueryClientProvider>,
    );

    await user.type(screen.getByPlaceholderText('Ask a question about your research…'), 'What does JamPacker recover from?{Enter}');
    await user.click(await screen.findByRole('button', { name: 'Retry' }));

    await screen.findByText('JamPacker recovers from failed placements.');
    expect(vi.mocked(sendChat).mock.calls.map((call) => call[0])).toEqual([
      'What does JamPacker recover from?',
      'What does JamPacker recover from?',
    ]);
    expect(vi.mocked(sendChat).mock.calls[1][3]).toEqual([
      { role: 'user', content: 'What does JamPacker recover from?' },
    ]);
    expect(screen.getAllByText('What does JamPacker recover from?')).toHaveLength(1);
    await waitFor(() => expect(screen.queryByText('Request failed')).not.toBeInTheDocument());
    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
  });
});
