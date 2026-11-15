import { useState, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { type ChatStreamOptions, sendChat } from '../api/client';
import { isFailedRequest, type MessagesByConversation, toApiHistory, useChat } from './useChat';

vi.mock('../api/client', () => ({ sendChat: vi.fn() }));

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe('Stopping a question', () => {
  afterEach(() => vi.mocked(sendChat).mockReset());

  it('reports the stage, then turns a stop into a retryable "stopped" turn kept out of history', async () => {
    vi.mocked(sendChat).mockImplementation(
      (_q, _c, _m, _h, options?: ChatStreamOptions) =>
        new Promise((_resolve, reject) => {
          options?.onStage?.('retrieving');
          options?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
        }),
    );
    const { result } = renderHook(
      () => {
        const [byConversation, setByConversation] = useState<MessagesByConversation>({});
        return useChat({ conversationId: 'c1', messagesByConversation: byConversation, setMessagesByConversation: setByConversation });
      },
      { wrapper },
    );

    let sent: Promise<unknown> = Promise.resolve();
    act(() => {
      sent = result.current.send('What is JamPacker?').catch(() => undefined);
    });
    await waitFor(() => expect(result.current.stage).toBe('retrieving'));
    expect(result.current.startedAt).toEqual(expect.any(Number));

    act(() => result.current.stop());
    await act(async () => {
      await sent;
    });

    const last = result.current.messages[result.current.messages.length - 1];
    expect(last.outcome).toBe('stopped');
    expect(last.content).toBe('You stopped this request.');
    expect(isFailedRequest(last)).toBe(true);
    expect(toApiHistory(result.current.messages)).toEqual([]);
    expect(result.current.stage).toBeNull();
    expect(result.current.isLoading).toBe(false);
  });
});

describe('The draft', () => {
  afterEach(() => vi.mocked(sendChat).mockReset());

  it('collects the pieces as they arrive and is dropped when the request is stopped', async () => {
    vi.mocked(sendChat).mockImplementation(
      (_q, _c, _m, _h, options?: ChatStreamOptions) =>
        new Promise((_resolve, reject) => {
          options?.onStage?.('generating');
          options?.onToken?.('Jam');
          options?.onToken?.('Packer');
          options?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
        }),
    );
    const { result } = renderHook(
      () => {
        const [byConversation, setByConversation] = useState<MessagesByConversation>({});
        return useChat({ conversationId: 'c1', messagesByConversation: byConversation, setMessagesByConversation: setByConversation });
      },
      { wrapper },
    );

    let sent: Promise<unknown> = Promise.resolve();
    act(() => {
      sent = result.current.send('What is JamPacker?').catch(() => undefined);
    });
    await waitFor(() => expect(result.current.draft).toBe('JamPacker'));

    act(() => result.current.stop());
    await act(async () => {
      await sent;
    });

    expect(result.current.draft).toBe('');
    const last = result.current.messages[result.current.messages.length - 1];
    expect(last.content).toBe('You stopped this request.');
  });

  it('starts empty for the next question after an answer', async () => {
    vi.mocked(sendChat).mockImplementation(async (_q, _c, _m, _h, options?: ChatStreamOptions) => {
      options?.onToken?.('Draft text');
      return { answer: 'Final text', citations: [], outcome: 'answered', grounded: true, trace: null };
    });
    const { result } = renderHook(
      () => {
        const [byConversation, setByConversation] = useState<MessagesByConversation>({});
        return useChat({ conversationId: 'c1', messagesByConversation: byConversation, setMessagesByConversation: setByConversation });
      },
      { wrapper },
    );
    await act(async () => {
      await result.current.send('What is JamPacker?');
    });
    expect(result.current.draft).toBe('');
    expect(result.current.messages[result.current.messages.length - 1].content).toBe('Final text');
  });
});
