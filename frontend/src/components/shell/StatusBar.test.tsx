import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { StatusBar } from './StatusBar';

const FALLBACK_MESSAGE = 'VectorStore using JSON fallback after Chroma failure: RuntimeError: down';

function renderWithStatus(status: Record<string, unknown>) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      const body = url === '/api/status' ? status : { models: [], recommended: [], other: [], current: '' };
      return new Response(JSON.stringify(body), { status: 200 });
    }),
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/library']}>
        <StatusBar />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const baseStatus = { papers: 1, chunks: 2, backend: 'ollama', model: 'gemma3:12b', ollama_connected: true };

describe('StatusBar vector-store warning', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('shows a degraded badge with the backend message when the API reports a fallback', async () => {
    renderWithStatus({ ...baseStatus, vector_backend: 'json', warnings: [FALLBACK_MESSAGE] });

    const summary = await screen.findByText('⚠ Vector search degraded');
    // Native disclosure: keyboard/touch reachable, full message available on expand.
    expect(summary.tagName).toBe('SUMMARY');
    expect(screen.getByText(FALLBACK_MESSAGE)).toBeInTheDocument();
    // Badge sits inside the polite live region, which has no aria-label override.
    const region = screen.getByRole('status');
    expect(region).toContainElement(summary);
    expect(region).not.toHaveAttribute('aria-label');
  });

  it('shows no badge when the API reports no warnings', async () => {
    renderWithStatus({ ...baseStatus, vector_backend: 'chroma', warnings: [] });
    await waitFor(() => expect(screen.getByText('● Ollama connected')).toBeInTheDocument());
    // Live region always exists (so later warnings are announced) but stays empty.
    expect(screen.getByRole('status')).toBeEmptyDOMElement();
  });

  it('shows no badge for an older API without the new fields', async () => {
    renderWithStatus(baseStatus);
    await waitFor(() => expect(screen.getByText('● Ollama connected')).toBeInTheDocument());
    // Live region always exists (so later warnings are announced) but stays empty.
    expect(screen.getByRole('status')).toBeEmptyDOMElement();
  });
});
