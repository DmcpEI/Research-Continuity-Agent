export type ApiCitation = {
  source_id: string;
  title: string;
  excerpt: string;
};

// Mirrors AnswerOutcome in rca/flows/generate_flow.py.
export type AnswerOutcome = 'answered' | 'unverified' | 'not_found' | 'conversational';

export type ApiChatResponse = {
  answer: string;
  citations: ApiCitation[];
  outcome?: AnswerOutcome; // absent from servers older than the outcome field
  grounded: boolean;
  failure_labels?: string[];
  active_source_ids?: string[];
  model?: string;
  trace: Record<string, unknown> | null;
};

export type ApiChatMessage = {
  role: 'user' | 'assistant';
  content: string;
};

export type ApiAgentToolCall = {
  tool_name: string;
  input: Record<string, unknown>;
  output: string;
  status: string;
  duration_ms: number;
};

export type ApiAgentTrace = {
  query: string;
  iterations: number;
  tool_calls: ApiAgentToolCall[];
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_latency_ms: number;
  stopped_reason: string | null;
  warnings: string[];
};

export type ApiAgentResponse = {
  answer: string;
  trace: ApiAgentTrace;
  error: string | null;
  model?: string;
};

export type ApiAgentModelsResponse = {
  models: string[];
  current: string;
  // null: Ollama could not be asked whether the current model can call tools.
  current_capable?: boolean | null;
};

export type ApiAgentStatus = {
  model: string;
  tool_calling_capable: boolean | null;
  backend: string;
  ollama_connected: boolean;
};

export type ApiSourceSummary = {
  id: string;
  title: string;
  kind: string;
  chunk_count: number;
  latest_path: string | null;
  content_sha256: string | null;
  latest_revision_created_at: string | null;
};

export type ApiRevision = {
  revision_id: string;
  source_id: string;
  revision_number: number;
  ingest_status: string;
  path: string;
  ingest_name: string;
  title: string;
  file_sha256: string | null;
  content_sha256: string;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type ApiSourceDetail = {
  id: string;
  title: string;
  kind: string;
  text: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  chunk_count: number;
  revisions: ApiRevision[];
};

export type ApiStatus = {
  papers: number;
  chunks: number;
  backend: string;
  model: string;
  ollama_connected: boolean;
  // Optional so the UI keeps working against an older API without these fields.
  vector_backend?: string;
  warnings?: string[];
};

export type ApiModelsResponse = {
  models: string[];
  current: string;
  recommended: ApiModelOption[];
  other: ApiModelOption[];
};

export type ApiModelOption = {
  name: string;
  recommended: boolean;
  score: number;
  reasons: string[];
};

export type ApiIngestResult = {
  source_id: string;
  chunk_ids: string[];
  node_count: number;
  edge_count: number;
  revision_id: string | null;
  ingest_status: string;
  metadata: Record<string, unknown>;
};

// A failed API call: `detail` is the server's own message (FastAPI's `detail`), for showing to people.
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    body: string,
  ) {
    super(`API ${status}: ${body}`);
    this.name = 'ApiError';
  }
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try {
      const parsed: unknown = JSON.parse(text);
      if (parsed && typeof parsed === 'object' && typeof (parsed as { detail?: unknown }).detail === 'string') {
        detail = (parsed as { detail: string }).detail;
      }
    } catch {
      // Not JSON: keep the raw text.
    }
    throw new ApiError(response.status, detail, text);
  }
  return (await response.json()) as T;
}

export async function getStatus(): Promise<ApiStatus> {
  const response = await fetch('/api/status');
  return parseJson<ApiStatus>(response);
}

export async function getSources(): Promise<ApiSourceSummary[]> {
  const response = await fetch('/api/sources');
  return parseJson<ApiSourceSummary[]>(response);
}

export async function getSource(id: string): Promise<ApiSourceDetail> {
  const response = await fetch(`/api/sources/${id}`);
  return parseJson<ApiSourceDetail>(response);
}

export async function deleteSource(id: string): Promise<{ deleted: string }> {
  const response = await fetch(`/api/sources/${id}`, {
    method: 'DELETE',
  });
  return parseJson<{ deleted: string }>(response);
}

export type ChatStage = 'retrieving' | 'generating' | 'checking';

export type ChatStreamOptions = {
  onStage?: (stage: ChatStage) => void;
  signal?: AbortSignal;
};

/**
 * Asks a question over `/api/chat/stream`: `onStage` hears each step as the server starts it, and
 * the promise resolves with the same payload `/api/chat` returns. Aborting `signal` stops the
 * request; the server then stops at its next step.
 */
export async function sendChat(
  query: string,
  conversationId?: string,
  model?: string,
  messages: ApiChatMessage[] = [],
  options: ChatStreamOptions = {},
): Promise<ApiChatResponse> {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      conversation_id: conversationId ?? null,
      model: model ?? null,
      messages,
    }),
    signal: options.signal,
  });
  if (!response.ok || !response.body) {
    return parseJson<ApiChatResponse>(response); // throws ApiError with the server's detail
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf('\n\n');
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf('\n\n');
      const fields: Record<string, string> = {};
      for (const line of block.split('\n')) {
        const colon = line.indexOf(': ');
        if (colon > 0) {
          fields[line.slice(0, colon)] = line.slice(colon + 2);
        }
      }
      const data: unknown = fields.data ? JSON.parse(fields.data) : {};
      if (fields.event === 'stage') {
        options.onStage?.((data as { stage: ChatStage }).stage);
      } else if (fields.event === 'final') {
        return data as ApiChatResponse;
      } else if (fields.event === 'error') {
        const detail = String((data as { detail?: unknown }).detail ?? 'Request failed');
        throw new ApiError(502, detail, fields.data);
      }
    }
  }
  throw new Error('The connection closed before an answer arrived.');
}

export async function sendAgent(query: string): Promise<ApiAgentResponse> {
  const response = await fetch('/api/agent', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  });

  return parseJson<ApiAgentResponse>(response);
}

export async function getAgentModels(): Promise<ApiAgentModelsResponse> {
  const response = await fetch('/api/agent/models');
  return parseJson<ApiAgentModelsResponse>(response);
}

export async function selectAgentModel(model: string): Promise<{ model: string }> {
  const response = await fetch('/api/agent/models/select', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ model }),
  });
  return parseJson<{ model: string }>(response);
}

export async function getAgentStatus(): Promise<ApiAgentStatus> {
  const response = await fetch('/api/agent/status');
  return parseJson<ApiAgentStatus>(response);
}

export async function getModels(): Promise<ApiModelsResponse> {
  const response = await fetch('/api/models');
  return parseJson<ApiModelsResponse>(response);
}

export async function selectModel(model: string): Promise<{ model: string }> {
  const response = await fetch('/api/models/select', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ model }),
  });
  return parseJson<{ model: string }>(response);
}

export async function ingestFile(file: File): Promise<ApiIngestResult> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/api/sources/ingest', {
    method: 'POST',
    body: formData,
  });

  return parseJson<ApiIngestResult>(response);
}

export function getPdfUrl(sourceId: string): string {
  return `/api/sources/${sourceId}/pdf#toolbar=0`;
}
