export type ApiCitation = {
  source_id: string;
  title: string;
  excerpt: string;
};

export type ApiChatResponse = {
  answer: string;
  citations: ApiCitation[];
  grounded: boolean;
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
  stopped_reason: string;
  warnings: string[];
};

export type ApiAgentResponse = {
  answer: string;
  trace: ApiAgentTrace;
  error: string | null;
};

export type ApiAgentModelsResponse = {
  models: string[];
  current: string;
};

export type ApiAgentStatus = {
  model: string;
  tool_calling_capable: boolean;
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

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API ${response.status}: ${text}`);
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

export async function sendChat(
  query: string,
  conversationId?: string,
  model?: string,
  messages: ApiChatMessage[] = [],
): Promise<ApiChatResponse> {
  const response = await fetch('/api/chat', {
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
  });

  return parseJson<ApiChatResponse>(response);
}

export async function sendAgent(
  query: string,
  conversationId?: string,
  messages: ApiChatMessage[] = [],
): Promise<ApiAgentResponse> {
  const response = await fetch('/api/agent', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      conversation_id: conversationId ?? null,
      messages,
    }),
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
