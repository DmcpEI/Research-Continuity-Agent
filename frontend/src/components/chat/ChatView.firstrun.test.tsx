import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { type ApiSourceSummary, getSources, sendChat } from '../../api/client';
import { ChatView } from './ChatView';

vi.mock('../../api/client', () => ({ sendChat: vi.fn(), getSources: vi.fn() }));

const SEED_TITLES = [
  'Bin packing retrieval quality checks',
  'Abstention calibration notes',
  'Agent loop follow-up actions',
];

function paper(title: string): ApiSourceSummary {
  return {
    id: `src:pdf/${title.toLowerCase()}`,
    title,
    kind: 'paper',
    chunk_count: 2,
    latest_path: null,
    content_sha256: null,
    latest_revision_created_at: null,
  };
}

function renderChat() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ChatView />
    </QueryClientProvider>,
  );
}

function storedConversations(): { id: string; title: string }[] {
  const raw = window.localStorage.getItem('rca-conversations');
  return raw ? JSON.parse(raw).conversations : [];
}

function storeState(conversations: object[], messagesByConversation: object, activeConversationId: string) {
  window.localStorage.setItem(
    'rca-conversations',
    JSON.stringify({ conversations, messagesByConversation, activeConversationId }),
  );
}

const stamp = new Date().toISOString();
const realConversation = { id: 'conv-77', title: 'My real question', timestamp: stamp, messageCount: 2 };
const realMessages = {
  'conv-77': [
    { id: 'msg-user-1', role: 'user', content: 'My real question', timestamp: stamp },
    { id: 'msg-assistant-1', role: 'assistant', content: 'A real answer.', outcome: 'answered', timestamp: stamp },
  ],
};

describe('ChatView on first run', () => {
  const original = window.HTMLElement.prototype.scrollIntoView;
  beforeEach(() => {
    window.localStorage.clear();
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
    vi.mocked(getSources).mockResolvedValue([]);
  });
  afterEach(() => {
    window.HTMLElement.prototype.scrollIntoView = original;
    vi.mocked(getSources).mockReset();
  });

  it('starts with one empty conversation and no made-up history', async () => {
    renderChat();
    await waitFor(() => expect(storedConversations().map((c) => c.title)).toEqual(['New conversation']));
  });

  it('drops the old fake conversations a browser saved, but keeps real ones', async () => {
    storeState(
      [
        ...SEED_TITLES.map((title, index) => ({ id: `conv-${index + 1}`, title, timestamp: stamp, messageCount: 0 })),
        realConversation,
      ],
      realMessages,
      'conv-1',
    );
    renderChat();
    await waitFor(() => expect(storedConversations().map((c) => c.id)).toEqual(['conv-77']));
    expect(await screen.findByText('A real answer.')).toBeInTheDocument();
  });

  it('never deletes stored messages on load, even when only seeds are listed', async () => {
    storeState(
      SEED_TITLES.map((title, index) => ({ id: `conv-${index + 1}`, title, timestamp: stamp, messageCount: 0 })),
      { 'conv-9': [{ id: 'msg-user-9', role: 'user', content: 'Lost question', timestamp: stamp }] },
      'conv-9',
    );
    renderChat();
    await waitFor(() => expect(storedConversations().map((c) => c.title)).toEqual(['New conversation']));
    expect(window.localStorage.getItem('rca-conversations')).toContain('Lost question');
  });

  it('opens a new empty conversation when the last one is deleted', async () => {
    storeState([realConversation], realMessages, 'conv-77');
    const user = userEvent.setup();
    renderChat();
    await user.click(await screen.findByRole('button', { name: 'Conversation actions' }));
    await user.click(screen.getByRole('menuitem', { name: 'Delete' }));
    // Before, the deleted conversation stayed active: no welcome state, and new messages went to
    // a conversation that no list showed.
    expect(await screen.findByText(/Add PDFs in Library/)).toBeInTheDocument();
    await waitFor(() => expect(storedConversations().map((c) => c.title)).toEqual(['New conversation']));

    vi.mocked(sendChat).mockResolvedValueOnce({
      answer: 'AutoBag opens bags.',
      citations: [],
      outcome: 'unverified',
      grounded: false,
      trace: null,
    });
    await user.type(screen.getByPlaceholderText('Ask a question about your research…'), 'What does AutoBag do?{Enter}');
    const list = screen.getByRole('complementary', { name: 'Conversation list' });
    await waitFor(() => expect(within(list).getByText('What does AutoBag do')).toBeInTheDocument());
  });

  it('suggests questions about papers in the library, and a click fills the box', async () => {
    vi.mocked(getSources).mockResolvedValue([paper('AutoBag_Learning_to_Open_Plastic_Bags')]);
    const user = userEvent.setup();
    renderChat();
    const chip = await screen.findByRole('button', {
      name: 'What problem does “AutoBag Learning to Open Plastic Bags” address, and how?',
    });
    expect(screen.queryByText(/retrieval quality over the last week/)).not.toBeInTheDocument();
    await user.click(chip);
    expect(screen.getByPlaceholderText('Ask a question about your research…')).toHaveValue(
      'What problem does “AutoBag Learning to Open Plastic Bags” address, and how?',
    );
  });

  it('points an empty library to Library instead of suggesting questions', async () => {
    renderChat();
    const empty = await screen.findByText(/Add PDFs in Library/);
    expect(within(empty.closest('.thread-empty-state') as HTMLElement).queryAllByRole('button')).toHaveLength(0);
  });
});
