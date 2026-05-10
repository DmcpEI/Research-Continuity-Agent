import { useMemo, useState } from 'react';
import type { Conversation } from './ChatView';

function formatGroupLabel(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();

  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfConversationDay = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const dayDiff = Math.floor(
    (startOfToday.getTime() - startOfConversationDay.getTime()) / (24 * 60 * 60 * 1000),
  );

  if (dayDiff === 0) {
    return 'Today';
  }
  if (dayDiff === 1) {
    return 'Yesterday';
  }

  return date.toLocaleDateString(undefined, { weekday: 'long' });
}

function formatTimestamp(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

type ConversationListProps = {
  conversations: Conversation[];
  activeConversationId: string;
  onConversationDelete: (conversationId: string) => void;
  onConversationSelect: (conversationId: string) => void;
  onNewConversation: () => void;
};

export function ConversationList({
  conversations,
  activeConversationId,
  onConversationDelete,
  onConversationSelect,
  onNewConversation,
}: ConversationListProps) {
  const [openMenuConversationId, setOpenMenuConversationId] = useState<string | null>(null);

  const groupedConversations = useMemo(() => {
    const sorted = [...conversations]
      .filter((conversation) => conversation.messageCount > 0)
      .sort((left, right) => new Date(right.timestamp).getTime() - new Date(left.timestamp).getTime());

    const groups = new Map<string, Conversation[]>();
    for (const conversation of sorted) {
      const groupLabel = formatGroupLabel(conversation.timestamp);
      const existing = groups.get(groupLabel) ?? [];
      existing.push(conversation);
      groups.set(groupLabel, existing);
    }

    return Array.from(groups.entries());
  }, [conversations]);

  return (
    <aside className="conversation-list" aria-label="Conversation list">
      <button className="new-conversation-btn" type="button" onClick={onNewConversation}>
        <span aria-hidden="true">+</span>
        <span>New conversation</span>
      </button>

      <div className="conversation-groups">
        {groupedConversations.map(([groupLabel, items]) => (
          <section className="conversation-group" key={groupLabel}>
            <h3 className="conversation-group-label">{groupLabel}</h3>
            <div className="conversation-group-items">
              {items.map((conversation) => (
                <div className="conversation-row-wrap" key={conversation.id}>
                  <button
                    className={`conversation-row ${
                      conversation.id === activeConversationId ? 'active' : ''
                    }`}
                    onClick={() => {
                      onConversationSelect(conversation.id);
                      setOpenMenuConversationId(null);
                    }}
                    type="button"
                  >
                    <div className="conversation-title" title={conversation.title}>
                      {conversation.title}
                    </div>
                    <div className="conversation-meta">
                      <span>{formatTimestamp(conversation.timestamp)}</span>
                      <span>{conversation.messageCount} msgs</span>
                    </div>
                  </button>

                  <button
                    aria-label="Conversation actions"
                    className="conversation-row-menu-trigger"
                    type="button"
                    onClick={() =>
                      setOpenMenuConversationId((previous) =>
                        previous === conversation.id ? null : conversation.id,
                      )
                    }
                  >
                    ...
                  </button>

                  {openMenuConversationId === conversation.id ? (
                    <div className="conversation-row-menu" role="menu">
                      <button
                        className="conversation-row-menu-item"
                        role="menuitem"
                        type="button"
                        onClick={() => {
                          setOpenMenuConversationId(null);
                          onConversationDelete(conversation.id);
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </aside>
  );
}
