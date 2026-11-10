import type { Dispatch, SetStateAction } from 'react';
import { useMutation } from '@tanstack/react-query';
import { type AnswerOutcome, type ApiCitation, type ApiChatMessage, sendChat } from '../api/client';

export type Message = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  citations?: ApiCitation[];
  activeSourceIds?: string[];
  // 'error' is client-only: the request failed, so there is no answer to classify.
  outcome?: AnswerOutcome | 'error';
  grounded?: boolean;
  model?: string;
  trace?: Record<string, unknown> | null;
  timestamp: Date;
};

export type MessagesByConversation = Record<string, Message[]>;

// Builds before outcomes existed saved error bubbles with no outcome; their id prefix has always
// marked them, so it identifies them without guessing from the text.
export function isFailedRequest(message: Message): boolean {
  return message.outcome === 'error' || (message.outcome === undefined && message.id.startsWith('msg-error-'));
}

// The turns the model sees as history. A failed request is not an answer, so neither it nor the
// question it failed on is sent; refusals are answers and stay.
export function toApiHistory(messages: Message[]): ApiChatMessage[] {
  return messages
    .filter((message, index) => {
      if (message.role === 'system' || isFailedRequest(message)) {
        return false;
      }
      const next = messages.slice(index + 1).find((later) => later.role !== 'system');
      return !(message.role === 'user' && next !== undefined && isFailedRequest(next));
    })
    .map((message) => ({ role: message.role as 'user' | 'assistant', content: message.content }));
}

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
    const outboundHistory = toApiHistory([...messages, userMessage]);

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
        outcome: response.outcome,
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
        outcome: 'error',
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

  // Removes a failed request and the question it failed on, and returns that question so the
  // caller can send it again. Returns undefined for anything that is not a failed request.
  const takeFailedQuestion = (errorId: string): string | undefined => {
    const errorIndex = messages.findIndex((message) => message.id === errorId);
    if (errorIndex < 0 || !isFailedRequest(messages[errorIndex])) {
      return undefined;
    }
    const question = messages
      .slice(0, errorIndex)
      .reverse()
      .find((message) => message.role !== 'system');
    if (question?.role !== 'user') {
      return undefined;
    }
    const drop = new Set([errorId, question.id]);
    setMessagesByConversation((previous) => ({
      ...previous,
      [key]: (previous[key] ?? []).filter((message) => !drop.has(message.id)),
    }));
    return question.content;
  };

  return {
    messagesByConversation,
    messages,
    send,
    takeFailedQuestion,
    isLoading: mutation.isPending,
    error: mutation.error,
  };
}
