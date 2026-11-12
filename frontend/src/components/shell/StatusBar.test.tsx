import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, useNavigate } from 'react-router-dom';
import type { NavigateFunction } from 'react-router-dom';
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

describe('StatusBar agent model picker', () => {
  const NOT_CAPABLE = 'Selected model is not tool-calling capable for agent mode';
  let selectCalls: string[] = [];

  function renderAgentRoute() {
    selectCalls = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string, init?: RequestInit) => {
        if (url === '/api/agent/models/select') {
          selectCalls.push(JSON.parse(String(init?.body)).model);
          return new Response(JSON.stringify({ detail: NOT_CAPABLE }), { status: 400 });
        }
        const bodies: Record<string, unknown> = {
          '/api/status': { ...baseStatus, warnings: [] },
          '/api/agent/models': { models: ['qwen3.5:4b', 'gemma3:12b'], current: 'qwen3.5:4b', current_capable: true },
          '/api/agent/status': { model: 'qwen3.5:4b', tool_calling_capable: true, backend: 'ollama', ollama_connected: true },
          '/api/models': {
            models: ['qwen3.5:4b', 'gemma3:12b'],
            current: 'qwen3.5:4b',
            recommended: [
              { name: 'qwen3.5:4b', recommended: true, score: 30, reasons: [] },
              { name: 'gemma3:12b', recommended: true, score: 30, reasons: [] },
            ],
            other: [],
          },
        };
        return new Response(JSON.stringify(bodies[url] ?? {}), { status: 200 });
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    return render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={['/agent']}>
          <StatusBar />
          <NavigateGrabber />
        </MemoryRouter>
      </QueryClientProvider>,
    );
  }

  let navigate: NavigateFunction = () => undefined;
  function NavigateGrabber() {
    navigate = useNavigate();
    return null;
  }

  afterEach(() => {
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it('does not keep re-selecting a saved agent model the server refuses', async () => {
    window.localStorage.setItem('rca-current-agent-model', 'gemma3:12b');
    renderAgentRoute();
    await waitFor(() => expect(selectCalls).toEqual(['gemma3:12b']));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(selectCalls).toEqual(['gemma3:12b']);
    expect(window.localStorage.getItem('rca-current-agent-model')).toBeNull();
  });

  it('opens on request and shows why a model was refused', async () => {
    renderAgentRoute();
    await screen.findByText('qwen3.5:4b');
    window.dispatchEvent(new CustomEvent('rca-open-model-picker'));
    const option = await screen.findByRole('button', { name: 'gemma3:12b' });
    option.click();
    expect(await screen.findByText(`gemma3:12b: ${NOT_CAPABLE}`)).toBeInTheDocument();
  });

  it('forgets a refusal when the route changes', async () => {
    const { act } = await import('@testing-library/react');
    renderAgentRoute();
    await screen.findByText('qwen3.5:4b');
    window.dispatchEvent(new CustomEvent('rca-open-model-picker'));
    (await screen.findByRole('button', { name: 'gemma3:12b' })).click();
    await screen.findByText(`gemma3:12b: ${NOT_CAPABLE}`);
    act(() => navigate('/'));
    act(() => navigate('/agent'));
    window.dispatchEvent(new CustomEvent('rca-open-model-picker'));
    await screen.findByRole('button', { name: 'gemma3:12b' });
    expect(screen.queryByText(`gemma3:12b: ${NOT_CAPABLE}`)).not.toBeInTheDocument();
  });
});

describe('StatusBar agent model restore', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it('does not re-select a saved agent model the server already uses', async () => {
    const selects: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string, init?: RequestInit) => {
        if (url === '/api/agent/models/select') {
          selects.push(JSON.parse(String(init?.body)).model);
          return new Response(JSON.stringify({ model: 'qwen3.5:4b' }), { status: 200 });
        }
        const bodies: Record<string, unknown> = {
          '/api/status': { ...baseStatus, warnings: [] },
          '/api/agent/models': { models: ['qwen3.5:4b'], current: 'qwen3.5:4b', current_capable: true },
          '/api/agent/status': { model: 'qwen3.5:4b', tool_calling_capable: true, backend: 'ollama', ollama_connected: true },
          '/api/models': { models: ['qwen3.5:4b'], current: 'qwen3.5:4b', recommended: [], other: [] },
        };
        return new Response(JSON.stringify(bodies[url] ?? {}), { status: 200 });
      }),
    );
    window.localStorage.setItem('rca-current-agent-model', 'qwen3.5:4b');
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={['/agent']}>
          <StatusBar />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    await screen.findByText('qwen3.5:4b');
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(selects).toEqual([]);
  });
});

describe('StatusBar agent model restore after a server restart', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it('restores the saved model when the server comes back on another one', async () => {
    let serverModel = 'qwen3.5:9b';
    const selects: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string, init?: RequestInit) => {
        if (url === '/api/agent/models/select') {
          const model = JSON.parse(String(init?.body)).model;
          selects.push(model);
          serverModel = model;
          return new Response(JSON.stringify({ model }), { status: 200 });
        }
        const bodies: Record<string, unknown> = {
          '/api/status': { ...baseStatus, warnings: [] },
          '/api/agent/models': { models: ['qwen3.5:4b', 'qwen3.5:9b'], current: serverModel, current_capable: true },
          '/api/agent/status': { model: serverModel, tool_calling_capable: true, backend: 'ollama', ollama_connected: true },
          '/api/models': { models: [], current: '', recommended: [], other: [] },
        };
        return new Response(JSON.stringify(bodies[url] ?? {}), { status: 200 });
      }),
    );
    window.localStorage.setItem('rca-current-agent-model', 'qwen3.5:9b');
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={['/agent']}>
          <StatusBar />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    await screen.findByText('qwen3.5:9b');
    expect(selects).toEqual([]);

    serverModel = 'qwen3.5:4b'; // API restarted on its default
    await client.invalidateQueries();
    await waitFor(() => expect(selects).toEqual(['qwen3.5:9b']));
  });
});
