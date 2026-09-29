import type { Dispatch, SetStateAction } from 'react';
import { useMutation } from '@tanstack/react-query';
import { type ApiCitation, type ApiChatMessage, sendChat } from '../api/client';

export type Message = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  citations?: ApiCitation[];
  activeSourceIds?: string[];
  grounded?: boolean;
  model?: string;
  trace?: Record<string, unknown> | null;
  timestamp: Date;
};

export type MessagesByConversation = Record<string, Message[]>;

type UseChatArgs = {
  conversationId?: string;
  activeModel?: string;
  messagesByConversation: MessagesByConversation;
  setMessagesByConversation: Dispatch<SetStateAction<MessagesByConversation>>;
};

export function useChat({
  conversationId,
  activeModel,
  messagesByConversation,
  setMessagesByConversation,
}: UseChatArgs) {
  const key = conversationId ?? '__default__';
  const messages = messagesByConversation[key] ?? [];

  const mutation = useMutation({
    mutationFn: ({
      query,
      cid,
      model,
      history,
    }: {
      query: string;
      cid?: string;
      model?: string;
      history: ApiChatMessage[];
    }) => sendChat(query, cid, model, history),
  });

  const send = async (query: string, modelOverride?: string) => {
    const trimmed = query.trim();
    if (!trimmed) {
      return;
    }
    const model = modelOverride ?? activeModel;

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
        model,
        history: outboundHistory,
      });
      const assistantMessage: Message = {
        id: `msg-assistant-${Date.now()}`,
        role: 'assistant',
        content: response.answer,
        citations: response.citations,
        activeSourceIds: response.active_source_ids ?? [],
        grounded: response.grounded,
        model: response.model ?? model,
        trace: response.trace,
        timestamp: new Date(),
      };

      setMessagesByConversation((previous) => ({
        ...previous,
        [key]: [...(previous[key] ?? []), assistantMessage],
      }));

      return response;
    } catch (error) {
      const errorMessage: Message = {
        id: `msg-error-${Date.now()}`,
        role: 'assistant',
        content: `I could not complete the request. ${String(error)}`,
        grounded: false,
        citations: [],
        activeSourceIds: [],
        model,
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
