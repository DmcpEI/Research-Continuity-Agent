import { useState, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook } from '@testing-library/react';
import { sendChat } from '../api/client';
import { type Message, type MessagesByConversation, toApiHistory, useChat } from './useChat';

vi.mock('../api/client', () => ({ sendChat: vi.fn() }));

const at = new Date('2026-11-10T12:00:00Z');
const user = (id: string, content: string): Message => ({ id, role: 'user', content, timestamp: at });
const reply = (id: string, content: string, outcome: Message['outcome']): Message => ({
  id,
  role: 'assistant',
  content,
  outcome,
  timestamp: at,
});

describe('toApiHistory', () => {
  it('leaves out failed requests and the question they failed on', () => {
    const history = toApiHistory([
      user('u1', 'What is JamPacker?'),
      reply('a1', 'A bin-packing planner.', 'answered'),
      user('u2', 'And its recovery module?'),
      reply('e1', 'I could not complete the request. Error: API 502', 'error'),
      { id: 's1', role: 'system', content: 'Switched to qwen3.5:4b', timestamp: at },
      user('u3', 'What is the capital of France?'),
      reply('a3', "I couldn't find evidence for this in your knowledge base.", 'not_found'),
    ]);

    expect(history).toEqual([
      { role: 'user', content: 'What is JamPacker?' },
      { role: 'assistant', content: 'A bin-packing planner.' },
      { role: 'user', content: 'What is the capital of France?' },
      { role: 'assistant', content: "I couldn't find evidence for this in your knowledge base." },
    ]);
  });

  it('keeps messages saved before outcomes existed', () => {
    const old: Message = { id: 'a0', role: 'assistant', content: 'Old answer.', grounded: true, timestamp: at };

    expect(toApiHistory([user('u0', 'Old question?'), old])).toHaveLength(2);
  });

  it('also leaves out failed requests saved before outcomes existed', () => {
    // Older builds stored error bubbles with no outcome; their id prefix has always marked them.
    const oldError: Message = {
      id: 'msg-error-1700000000000',
      role: 'assistant',
      content: 'I could not complete the request. Error: API 502',
      grounded: false,
      timestamp: at,
    };

    expect(toApiHistory([user('u1', 'Old question?'), oldError, user('u2', 'New question?')])).toEqual([
      { role: 'user', content: 'New question?' },
    ]);
  });
});

function renderChat(initial: MessagesByConversation = {}) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  return renderHook(
    () => {
      const [messagesByConversation, setMessagesByConversation] = useState<MessagesByConversation>(initial);
      return useChat({ conversationId: 'c1', messagesByConversation, setMessagesByConversation });
    },
    { wrapper },
  );
}

describe('useChat after a failed request', () => {
  afterEach(() => {
    vi.mocked(sendChat).mockReset();
  });

  it('does not send the failed turn as history with the next question', async () => {
    vi.mocked(sendChat).mockRejectedValueOnce(new Error('API 502')).mockResolvedValueOnce({
      answer: 'It recovers from failed placements. [[src:pdf/jampacker]]',
      citations: [],
      outcome: 'answered',
      grounded: true,
      trace: null,
    });
    const { result } = renderChat();

    await act(async () => {
      await result.current.send('What does JamPacker recover from?').catch(() => undefined);
    });
    await act(async () => {
      await result.current.send('What does JamPacker recover from, briefly?');
    });

    const history = vi.mocked(sendChat).mock.calls[1][3];
    expect(history).toEqual([{ role: 'user', content: 'What does JamPacker recover from, briefly?' }]);
  });

  it('takes the failed question back out of the thread so it can be sent again', () => {
    const { result } = renderChat({
      c1: [
        user('u1', 'What is JamPacker?'),
        reply('a1', 'A bin-packing planner.', 'answered'),
        user('u2', 'And its recovery module?'),
        reply('e1', 'I could not complete the request.', 'error'),
      ],
    });

    let question: string | undefined;
    act(() => {
      question = result.current.takeFailedQuestion('e1');
    });

    expect(question).toBe('And its recovery module?');
    expect(result.current.messages.map((message) => message.id)).toEqual(['u1', 'a1']);
  });

  it('takes nothing for a message that is not a failed request', () => {
    const { result } = renderChat({ c1: [user('u1', 'Q?'), reply('a1', 'A.', 'answered')] });

    let question: string | undefined;
    act(() => {
      question = result.current.takeFailedQuestion('a1');
    });

    expect(question).toBeUndefined();
    expect(result.current.messages).toHaveLength(2);
  });
});
