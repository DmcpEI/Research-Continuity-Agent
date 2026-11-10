import { useState, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook } from '@testing-library/react';
import { sendChat } from '../api/client';
import { type MessagesByConversation, useChat } from './useChat';

vi.mock('../api/client', () => ({ sendChat: vi.fn() }));

function renderChat() {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  return renderHook(
    () => {
      const [messagesByConversation, setMessagesByConversation] = useState<MessagesByConversation>({});
      return useChat({ conversationId: 'c1', messagesByConversation, setMessagesByConversation });
    },
    { wrapper },
  );
}

describe('useChat answer outcome', () => {
  afterEach(() => {
    vi.mocked(sendChat).mockReset();
  });

  it('keeps the outcome the API returned on the assistant message', async () => {
    vi.mocked(sendChat).mockResolvedValue({
      answer: "I couldn't find evidence for this in your knowledge base.",
      citations: [],
      outcome: 'not_found',
      grounded: false,
      trace: null,
    });
    const { result } = renderChat();

    await act(async () => {
      await result.current.send('What is the capital of France?');
    });

    expect(result.current.messages.slice(-1)[0]).toMatchObject({ role: 'assistant', outcome: 'not_found' });
  });

  it('marks a failed request as an error, not as an answer', async () => {
    vi.mocked(sendChat).mockRejectedValue(new Error('API 502'));
    const { result } = renderChat();

    await act(async () => {
      await result.current.send('anything').catch(() => undefined);
    });

    expect(result.current.messages.slice(-1)[0]).toMatchObject({ role: 'assistant', outcome: 'error' });
  });
});
