import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { ApiSourceSummary } from '../../api/client';
import { SourceRow } from './SourceRow';

const source = (extra: Partial<ApiSourceSummary> = {}): ApiSourceSummary => ({
  id: 'src:pdf/paper',
  title: 'Paper',
  kind: 'paper',
  chunk_count: 3,
  latest_path: null,
  content_sha256: null,
  latest_revision_created_at: null,
  ...extra,
});

function renderRow(summary: ApiSourceSummary) {
  return render(
    <table>
      <tbody>
        <SourceRow source={summary} />
      </tbody>
    </table>,
  );
}

describe('SourceRow incomplete flag', () => {
  it('says a half-indexed source needs another upload', () => {
    renderRow(source({ incomplete: true }));

    expect(screen.getByText('Not fully indexed: upload it again to repair')).toBeTruthy();
  });

  it('says nothing for a complete source or a server without the flag', () => {
    renderRow(source({ incomplete: false }));
    renderRow(source());

    expect(screen.queryByText(/Not fully indexed/)).toBeNull();
  });
});

describe('SourceRow pages without text', () => {
  it('says how many pages could not be read', () => {
    renderRow(source({ page_count: 10, pages_without_text: 3 }));

    expect(screen.getByText('3 of 10 pages have no text (scanned?)')).toBeTruthy();
  });

  it('says nothing when every page has text, or for older servers', () => {
    renderRow(source({ page_count: 10, pages_without_text: 0 }));
    renderRow(source());

    expect(screen.queryByText(/pages have no text/)).toBeNull();
  });
});
