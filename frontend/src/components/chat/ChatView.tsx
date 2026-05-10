import { useEffect, useMemo, useState } from 'react';
import { ConversationList } from './ConversationList';
import { MessageThread } from './MessageThread';
import { type Message, type MessagesByConversation, useChat } from '../../hooks/useChat';

export type Conversation = {
  id: string;
  title: string;
  timestamp: string;
  messageCount: number;
  model?: string;
};

type ChatViewProps = {
  onCitationClick?: (sourceId: string | null) => void;
  onGraphFocusChange?: (sourceIds: string[]) => void;
};

const STORAGE_KEY = 'rca-conversations';
const CHAT_MODEL_STORAGE_KEY = 'rca-current-chat-model';

type PersistedMessage = Omit<Message, 'timestamp'> & {
  timestamp: string;
};

type PersistedState = {
  conversations: Conversation[];
  messagesByConversation: Record<string, PersistedMessage[]>;
  selectedSourceByConversation?: Record<string, string>;
  activeConversationId: string;
};

type ChatModelSelectedEventDetail = {
  model: string;
};

type ChatModelActiveEventDetail = {
  model: string;
};

function makeConversationTitleFromQuery(query: string): string {
  const normalized = query.replace(/\s+/g, ' ').trim();
  if (!normalized) {
    return 'New conversation';
  }

  const trimmed = normalized.replace(/[.!?]+$/, '');
  return trimmed.length > 56 ? `${trimmed.slice(0, 56)}...` : trimmed;
}

const initialConversations: Conversation[] = [
  {
    id: 'conv-1',
    title: 'Bin packing retrieval quality checks',
    timestamp: new Date().toISOString(),
    messageCount: 4,
  },
  {
    id: 'conv-2',
    title: 'Abstention calibration notes',
    timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    messageCount: 3,
  },
  {
    id: 'conv-3',
    title: 'Agent loop follow-up actions',
    timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    messageCount: 6,
  },
];

export function ChatView({ onCitationClick, onGraphFocusChange }: ChatViewProps) {
  const [conversations, setConversations] = useState<Conversation[]>(initialConversations);
  const [activeConversationId, setActiveConversationId] = useState<string>(
    initialConversations[0]?.id ?? '',
  );
  const [messagesByConversation, setMessagesByConversation] = useState<MessagesByConversation>({});
  const [selectedSourceByConversation, setSelectedSourceByConversation] = useState<
    Record<string, string>
  >({});
  const [isHydrated, setIsHydrated] = useState(false);
  const [pendingConversationId, setPendingConversationId] = useState<string | null>(null);

  const { send, isLoading } = useChat({
    conversationId: activeConversationId,
    messagesByConversation,
    setMessagesByConversation,
  });

  const computedConversations = useMemo(
    () =>
      conversations.map((conversation) => ({
        ...conversation,
        messageCount: messagesByConversation[conversation.id]?.length ?? 0,
      })),
    [conversations, messagesByConversation],
  );

  const activeMessages = messagesByConversation[activeConversationId] ?? [];

  const activeConversation = computedConversations.find(
    (conversation) => conversation.id === activeConversationId,
  );

  const hasGlobalPending = pendingConversationId !== null;
  const isWaitingOnOtherConversation =
    pendingConversationId !== null && pendingConversationId !== activeConversationId;

  const activeSelectedSourceId = selectedSourceByConversation[activeConversationId] ?? null;

  const handleNewConversation = () => {
    const now = new Date();
    const currentModel = window.localStorage.getItem(CHAT_MODEL_STORAGE_KEY) ?? undefined;
    const nextConversation: Conversation = {
      id: `conv-${now.getTime()}`,
      title: 'New conversation',
      timestamp: now.toISOString(),
      messageCount: 0,
      model: currentModel,
    };

    setConversations((previous) => [nextConversation, ...previous]);
    setActiveConversationId(nextConversation.id);
  };

  const handleDeleteConversation = (conversationId: string) => {
    setConversations((previous) => previous.filter((conversation) => conversation.id !== conversationId));
    setMessagesByConversation((previous) => {
      const next = { ...previous };
      delete next[conversationId];
      return next;
    });
    setSelectedSourceByConversation((previous) => {
      const next = { ...previous };
      delete next[conversationId];
      return next;
    });

    if (conversationId === activeConversationId) {
      const remaining = computedConversations.filter((conversation) => conversation.id !== conversationId);
      if (remaining.length > 0) {
        setActiveConversationId(remaining[0].id);
      }
    }
  };

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return;
      }

      const parsed = JSON.parse(raw) as PersistedState;
      if (
        !Array.isArray(parsed.conversations) ||
        typeof parsed.messagesByConversation !== 'object' ||
        parsed.messagesByConversation === null
      ) {
        return;
      }

      setConversations(parsed.conversations);

      const hydratedMessages = Object.fromEntries(
        Object.entries(parsed.messagesByConversation).map(([conversationKey, storedMessages]) => [
          conversationKey,
          (storedMessages ?? []).map((message) => ({
            ...message,
            timestamp: new Date(message.timestamp),
          })),
        ]),
      ) as MessagesByConversation;

      setMessagesByConversation(hydratedMessages);
      setSelectedSourceByConversation(parsed.selectedSourceByConversation ?? {});

      const nextActiveId = parsed.activeConversationId;
      const hasActiveConversation = parsed.conversations.some(
        (conversation) => conversation.id === nextActiveId,
      );

      if (hasActiveConversation) {
        setActiveConversationId(nextActiveId);
      } else if (parsed.conversations.length > 0) {
        setActiveConversationId(parsed.conversations[0].id);
      }
    } catch {
      // Ignore malformed localStorage payloads.
    } finally {
      setIsHydrated(true);
    }
  }, []);

  useEffect(() => {
    if (!isHydrated) {
      return;
    }

    const serializedMessages: Record<string, PersistedMessage[]> = Object.fromEntries(
      Object.entries(messagesByConversation).map(([conversationKey, messageList]) => [
        conversationKey,
        messageList.map((message) => ({
          ...message,
          timestamp: message.timestamp.toISOString(),
        })),
      ]),
    );

    const persistedState: PersistedState = {
      conversations: computedConversations,
      messagesByConversation: serializedMessages,
      selectedSourceByConversation,
      activeConversationId,
    };

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(persistedState));
    window.dispatchEvent(new CustomEvent('rca-conversations-updated'));
  }, [
    activeConversationId,
    computedConversations,
    isHydrated,
    messagesByConversation,
    selectedSourceByConversation,
  ]);

  useEffect(() => {
    if (!activeConversationId) {
      return;
    }

    const activeModel = computedConversations.find(
      (conversation) => conversation.id === activeConversationId,
    )?.model;

    if (!activeModel) {
      return;
    }

    window.localStorage.setItem(CHAT_MODEL_STORAGE_KEY, activeModel);
    window.dispatchEvent(
      new CustomEvent<ChatModelActiveEventDetail>('rca-chat-active-model', {
        detail: { model: activeModel },
      }),
    );
  }, [activeConversationId, computedConversations]);

  useEffect(() => {
    onCitationClick?.(activeSelectedSourceId);
  }, [activeSelectedSourceId, onCitationClick]);

  useEffect(() => {
    const citedSourceIds = activeMessages
      .filter((message) => message.role === 'assistant')
      .flatMap((message) => message.citations?.map((citation) => citation.source_id) ?? []);

    const focusSourceIds = Array.from(
      new Set(
        [activeSelectedSourceId, ...citedSourceIds].filter(
          (sourceId): sourceId is string => Boolean(sourceId),
        ),
      ),
    );

    onGraphFocusChange?.(focusSourceIds);
  }, [activeMessages, activeSelectedSourceId, onGraphFocusChange]);

  const handleCitationClick = (sourceId: string) => {
    setSelectedSourceByConversation((previous) => ({
      ...previous,
      [activeConversationId]: sourceId,
    }));
    onCitationClick?.(sourceId);
  };

  const handleSend = async (query: string) => {
    if (hasGlobalPending && isWaitingOnOtherConversation) {
      return;
    }

    const now = new Date();
    const currentModel = window.localStorage.getItem(CHAT_MODEL_STORAGE_KEY) ?? undefined;

    setConversations((previous) =>
      previous.map((conversation) =>
        conversation.id === activeConversationId
          ? {
              ...conversation,
              model: currentModel,
            }
          : conversation,
      ),
    );

    if (activeConversation && activeConversation.messageCount === 0) {
      setConversations((previous) =>
        previous.map((conversation) =>
          conversation.id === activeConversationId
            ? {
                ...conversation,
                title: makeConversationTitleFromQuery(query),
                timestamp: now.toISOString(),
              }
            : conversation,
        ),
      );
    } else {
      setConversations((previous) =>
        previous.map((conversation) =>
          conversation.id === activeConversationId
            ? {
                ...conversation,
                timestamp: now.toISOString(),
              }
            : conversation,
        ),
      );
    }

    setPendingConversationId(activeConversationId);
    try {
      await send(query);
    } finally {
      setPendingConversationId((current) =>
        current === activeConversationId ? null : current,
      );
    }
  };

  useEffect(() => {
    const handler = (event: Event) => {
      const customEvent = event as CustomEvent<ChatModelSelectedEventDetail>;
      const model = customEvent.detail?.model;
      if (!model || !activeConversationId) {
        return;
      }

      setConversations((previous) =>
        previous.map((conversation) =>
          conversation.id === activeConversationId ? { ...conversation, model } : conversation,
        ),
      );

      setMessagesByConversation((previous) => {
        const current = previous[activeConversationId] ?? [];
        const last = current[current.length - 1];
        if (last?.role === 'system' && last.content === `Switched to ${model}`) {
          return previous;
        }

        if (current.length === 0) {
          return previous;
        }

        return {
          ...previous,
          [activeConversationId]: [
            ...current,
            {
              id: `msg-system-${Date.now()}`,
              role: 'system',
              content: `Switched to ${model}`,
              timestamp: new Date(),
            },
          ],
        };
      });
    };

    window.addEventListener('rca-chat-model-selected', handler as EventListener);
    return () => window.removeEventListener('rca-chat-model-selected', handler as EventListener);
  }, [activeConversationId]);

  useEffect(() => {
    if (!isHydrated || computedConversations.length === 0) {
      return;
    }

    const hasActiveConversation = computedConversations.some(
      (conversation) => conversation.id === activeConversationId,
    );
    if (!hasActiveConversation) {
      setActiveConversationId(computedConversations[0].id);
    }
  }, [activeConversationId, computedConversations, isHydrated]);

  useEffect(() => {
    window.dispatchEvent(
      new CustomEvent('rca-chat-loading', {
        detail: { isLoading: hasGlobalPending },
      }),
    );
  }, [hasGlobalPending]);

  return (
    <section className="chat-view" aria-label="Chat workspace">
      <ConversationList
        activeConversationId={activeConversationId}
        conversations={computedConversations}
        onConversationDelete={handleDeleteConversation}
        onConversationSelect={setActiveConversationId}
        onNewConversation={handleNewConversation}
      />
      <MessageThread
        conversationId={activeConversationId}
        isLoading={isLoading}
        messages={activeMessages}
        onCitationClick={handleCitationClick}
        onSend={handleSend}
        inputDisabledReason={
          isWaitingOnOtherConversation
            ? 'Waiting for response in another conversation.'
            : null
        }
        showWelcomeEmptyState={Boolean(activeConversation && activeConversation.messageCount === 0)}
      />
    </section>
  );
}
