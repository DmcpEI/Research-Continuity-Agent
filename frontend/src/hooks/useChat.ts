import type { Dispatch, SetStateAction } from 'react';
import { useMutation } from '@tanstack/react-query';
import { type ApiCitation, type ApiChatMessage, sendChat } from '../api/client';

export type Message = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  citations?: ApiCitation[];
  grounded?: boolean;
  trace?: Record<string, unknown> | null;
  timestamp: Date;
};

export type MessagesByConversation = Record<string, Message[]>;

type UseChatArgs = {
  conversationId?: string;
  messagesByConversation: MessagesByConversation;
  setMessagesByConversation: Dispatch<SetStateAction<MessagesByConversation>>;
};

export function useChat({
  conversationId,
  messagesByConversation,
  setMessagesByConversation,
}: UseChatArgs) {

  const key = conversationId ?? '__default__';
  const messages = messagesByConversation[key] ?? [];

  const mutation = useMutation({
    mutationFn: ({
      query,
      cid,
      history,
    }: {
      query: string;
      cid?: string;
      history: ApiChatMessage[];
    }) => sendChat(query, cid, undefined, history),
  });

  const send = async (query: string) => {
    const trimmed = query.trim();
    if (!trimmed) {
      return;
    }

    const now = new Date();
    const userMessage: Message = {
      id: `msg-user-${now.getTime()}`,
      role: 'user',
      content: trimmed,
      timestamp: now,
    };
    const outboundHistory: ApiChatMessage[] = [...messages, userMessage]
      .filter((message): message is Message & { role: 'user' | 'assistant' } => message.role !== 'system')
      .map((message) => ({
        role: message.role,
        content: message.content,
      }));

    setMessagesByConversation((previous) => ({
      ...previous,
      [key]: [...(previous[key] ?? []), userMessage],
    }));

    try {
      const response = await mutation.mutateAsync({
        query: trimmed,
        cid: conversationId,
        history: outboundHistory,
      });
      const assistantMessage: Message = {
        id: `msg-assistant-${Date.now()}`,
        role: 'assistant',
        content: response.answer,
        citations: response.citations,
        grounded: response.grounded,
        trace: response.trace,
        timestamp: new Date(),
      };

      setMessagesByConversation((previous) => ({
        ...previous,
        [key]: [...(previous[key] ?? []), assistantMessage],
      }));
    } catch (error) {
      const errorMessage: Message = {
        id: `msg-error-${Date.now()}`,
        role: 'assistant',
        content: `I could not complete the request. ${String(error)}`,
        grounded: false,
        citations: [],
        trace: null,
        timestamp: new Date(),
      };

      setMessagesByConversation((previous) => ({
        ...previous,
        [key]: [...(previous[key] ?? []), errorMessage],
      }));
      throw error;
    }
  };

  return {
    messagesByConversation,
    messages,
    send,
    isLoading: mutation.isPending,
    error: mutation.error,
  };
}
