import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import { LibraryView } from './LibraryView';

type Reply = { status: number; body: unknown };

function renderLibrary(fetchReply: Reply) {
  const calls: { url: string; init?: RequestInit }[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      if (url === '/api/sources/fetch') {
        return new Response(JSON.stringify(fetchReply.body), { status: fetchReply.status });
      }
      return new Response(JSON.stringify([]), { status: 200 });
    }),
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <LibraryView />
    </QueryClientProvider>,
  );
  return calls;
}

function submit(value: string) {
  fireEvent.change(screen.getByLabelText('arXiv ID or DOI'), { target: { value } });
  fireEvent.click(screen.getByRole('button', { name: 'Fetch' }));
}

describe('LibraryView fetch by identifier', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('posts the identifier and says which paper was added', async () => {
    const calls = renderLibrary({
      status: 200,
      body: {
        source_id: 'src:pdf/jsonschemabench',
        chunk_ids: [],
        node_count: 1,
        edge_count: 0,
        revision_id: 'r1',
        ingest_status: 'created',
        metadata: { title: 'JSONSchemaBench_file_name' },
        title: 'JSONSchemaBench',
        origin: { kind: 'arxiv', identifier: '2501.10868v3' },
      },
    });

    submit(' 2501.10868 ');

    expect(await screen.findByText('Added “JSONSchemaBench” (arXiv 2501.10868v3).')).toBeTruthy();
    const post = calls.find((call) => call.url === '/api/sources/fetch');
    expect(post?.init?.method).toBe('POST');
    expect(JSON.parse(String(post?.init?.body))).toEqual({ identifier: '2501.10868' });
  });

  it('says when the paper was already in the library', async () => {
    renderLibrary({
      status: 200,
      body: {
        source_id: 'src:pdf/x',
        chunk_ids: [],
        node_count: 1,
        edge_count: 0,
        revision_id: 'r1',
        ingest_status: 'unchanged',
        metadata: { title: 'X_file' },
        title: 'X',
        origin: { kind: 'doi', identifier: '10.3390/x' },
      },
    });

    submit('10.3390/x');

    expect(await screen.findByText('“X” is already in your library.')).toBeTruthy();
  });

  it("shows the server's message when there is nothing to fetch", async () => {
    renderLibrary({ status: 404, body: { detail: 'No open-access PDF found for DOI 10.1109/x.' } });

    submit('10.1109/x');

    expect(await screen.findByText('No open-access PDF found for DOI 10.1109/x.')).toBeTruthy();
  });

  it('does not send an empty identifier', () => {
    const calls = renderLibrary({ status: 200, body: {} });

    submit('   ');

    expect(calls.some((call) => call.url === '/api/sources/fetch')).toBe(false);
  });
});
