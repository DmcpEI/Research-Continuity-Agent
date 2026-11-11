import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChatView } from './ChatView';

vi.mock('./ConversationList', () => ({
  ConversationList: ({ conversations, onConversationSelect }: any) => (
    <div>
      {conversations.map((conversation: any) => (
        <button key={conversation.id} type="button" onClick={() => onConversationSelect(conversation.id)}>
          {conversation.id}
        </button>
      ))}
    </div>
  ),
}));

vi.mock('./MessageThread', () => ({
  MessageThread: ({ conversationId, onCitationClick }: any) => (
    <button type="button" onClick={() => onCitationClick?.(`src:${conversationId}`)}>
      cite-{conversationId}
    </button>
  ),
}));

// ChatView asks the library for suggestions; these tests do not render a QueryClient.
vi.mock('../../hooks/useSources', () => ({ useSources: () => ({ data: [], isSuccess: true }) }));

vi.mock('../../hooks/useChat', async () => {
  const actual = await vi.importActual('../../hooks/useChat');
  return {
    ...actual,
    useChat: () => ({
      send: vi.fn(),
      isLoading: false,
    }),
  };
});

describe('Conversation-scoped citation state', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('CIT-010: keeps selected source linkage scoped by conversation', async () => {
    const user = userEvent.setup();
    const onCitationClick = vi.fn();
    // Two saved conversations (the build used to seed conv-1..conv-3 itself).
    const stamp = new Date().toISOString();
    window.localStorage.setItem(
      'rca-conversations',
      JSON.stringify({
        conversations: ['conv-a', 'conv-b'].map((id) => ({ id, title: id, timestamp: stamp, messageCount: 0 })),
        messagesByConversation: {},
        activeConversationId: 'conv-a',
      }),
    );

    render(<ChatView onCitationClick={onCitationClick} />);

    await user.click(screen.getByRole('button', { name: 'cite-conv-a' }));
    expect(onCitationClick).toHaveBeenLastCalledWith('src:conv-a');

    await user.click(screen.getByRole('button', { name: 'conv-b' }));
    expect(onCitationClick).toHaveBeenLastCalledWith(null);

    await user.click(screen.getByRole('button', { name: 'cite-conv-b' }));
    expect(onCitationClick).toHaveBeenLastCalledWith('src:conv-b');

    await user.click(screen.getByRole('button', { name: 'conv-a' }));
    expect(onCitationClick).toHaveBeenLastCalledWith('src:conv-a');
  });

  it('CIT-011: restores graph focus from assistant active source IDs', async () => {
    const onGraphFocusChange = vi.fn();
    window.localStorage.setItem(
      'rca-conversations',
      JSON.stringify({
        conversations: [
          {
            id: 'conv-active',
            title: 'Active',
            timestamp: new Date().toISOString(),
            messageCount: 1,
            model: 'llama3.1:8b',
          },
        ],
        messagesByConversation: {
          'conv-active': [
            {
              id: 'msg-assistant',
              role: 'assistant',
              content: 'Answer',
              citations: [],
              activeSourceIds: ['src:pdf/paper_a'],
              grounded: true,
              timestamp: new Date().toISOString(),
            },
          ],
        },
        selectedSourceByConversation: {},
        activeConversationId: 'conv-active',
      }),
    );

    render(<ChatView onGraphFocusChange={onGraphFocusChange} />);

    await waitFor(() =>
      expect(onGraphFocusChange).toHaveBeenLastCalledWith(['src:pdf/paper_a']),
    );
  });
});
