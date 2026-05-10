import type { MouseEvent } from 'react';
import type { ApiSourceSummary } from '../../api/client';

type SourceRowProps = {
  source: ApiSourceSummary;
  onOpen?: (sourceId: string) => void;
  selected?: boolean;
  onContextMenu?: (sourceId: string, event: MouseEvent<HTMLTableRowElement>) => void;
};

function formatRelativeTime(value: string | null): string {
  if (!value) {
    return 'Unknown';
  }

  const parsed = new Date(value);
  const timestamp = parsed.getTime();
  if (!Number.isFinite(timestamp)) {
    return 'Unknown';
  }

  const diffMs = Date.now() - timestamp;
  if (diffMs < 60_000) {
    return 'just now';
  }

  const minutes = Math.floor(diffMs / 60_000);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }

  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h ago`;
  }

  const days = Math.floor(hours / 24);
  if (days < 7) {
    return `${days}d ago`;
  }

  const weeks = Math.floor(days / 7);
  if (weeks < 5) {
    return `${weeks}w ago`;
  }

  const months = Math.floor(days / 30);
  if (months < 12) {
    return `${months}mo ago`;
  }

  const years = Math.floor(days / 365);
  return `${years}y ago`;
}

function isRecentlyUpdated(value: string | null): boolean {
  if (!value) {
    return false;
  }

  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) {
    return false;
  }

  return Date.now() - timestamp < 1000 * 60 * 60 * 24 * 3;
}

export function SourceRow({ source, onOpen, selected = false, onContextMenu }: SourceRowProps) {
  const recent = isRecentlyUpdated(source.latest_revision_created_at);
  const updatedLabel = formatRelativeTime(source.latest_revision_created_at);

  return (
    <tr
      className={`library-data-row ${selected ? 'is-selected' : ''}`}
      onClick={() => onOpen?.(source.id)}
      onContextMenu={(event) => {
        if (!onContextMenu) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        onContextMenu(source.id, event);
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onOpen?.(source.id);
        }
      }}
      aria-label={`Open ${source.title || source.id}`}
      aria-current={selected ? 'true' : undefined}
    >
      <td>
        <p className="library-source-title">{source.title || 'Untitled source'}</p>
        <p className="library-source-id">{source.id}</p>
      </td>
      <td>
        <span className="library-chunk-badge">{source.chunk_count}</span>
      </td>
      <td>
        <div className="library-updated-cell">
          <span
            aria-hidden="true"
            className={recent ? 'library-status-dot is-amber' : 'library-status-dot is-green'}
          />
          <span>{updatedLabel}</span>
        </div>
      </td>
    </tr>
  );
}
