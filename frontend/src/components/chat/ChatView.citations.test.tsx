import { render, screen } from '@testing-library/react';
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
  it('CIT-010: keeps selected source linkage scoped by conversation', async () => {
    const user = userEvent.setup();
    const onCitationClick = vi.fn();

    render(<ChatView onCitationClick={onCitationClick} />);

    await user.click(screen.getByRole('button', { name: 'cite-conv-1' }));
    expect(onCitationClick).toHaveBeenLastCalledWith('src:conv-1');

    await user.click(screen.getByRole('button', { name: 'conv-2' }));
    expect(onCitationClick).toHaveBeenLastCalledWith(null);

    await user.click(screen.getByRole('button', { name: 'cite-conv-2' }));
    expect(onCitationClick).toHaveBeenLastCalledWith('src:conv-2');

    await user.click(screen.getByRole('button', { name: 'conv-1' }));
    expect(onCitationClick).toHaveBeenLastCalledWith('src:conv-1');
  });
});
