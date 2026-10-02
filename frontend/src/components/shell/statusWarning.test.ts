import type { ApiStatus } from '../../api/client';
import { statusWarning } from './statusWarning';

const base: ApiStatus = {
  papers: 1,
  chunks: 2,
  backend: 'ollama',
  model: 'gemma3:12b',
  ollama_connected: true,
};

describe('statusWarning', () => {
  it('returns null when status is missing, has no warnings, or only blank ones', () => {
    expect(statusWarning(undefined)).toBeNull();
    expect(statusWarning(base)).toBeNull();
    expect(statusWarning({ ...base, vector_backend: 'chroma', warnings: [] })).toBeNull();
    expect(statusWarning({ ...base, warnings: ['  '] })).toBeNull();
  });

  it('flags degraded vector search with the backend message as detail', () => {
    const warning = statusWarning({
      ...base,
      vector_backend: 'json',
      warnings: ['VectorStore using JSON fallback after Chroma failure: RuntimeError: down'],
    });
    expect(warning).toEqual({
      label: '⚠ Vector search degraded',
      detail: 'VectorStore using JSON fallback after Chroma failure: RuntimeError: down',
    });
  });

  it('uses a generic label for warnings unrelated to the vector backend', () => {
    expect(statusWarning({ ...base, vector_backend: 'chroma', warnings: ['a', 'b'] })).toEqual({
      label: '⚠ Backend warning',
      detail: 'a\nb',
    });
  });

  it('ignores malformed warnings from the API instead of crashing', () => {
    const malformed = [
      { ...base, warnings: 'not-a-list' },
      { ...base, warnings: [42, null, { text: 'x' }] },
      { ...base, warnings: null },
    ] as unknown as ApiStatus[];
    for (const status of malformed) {
      expect(statusWarning(status)).toBeNull();
    }
    expect(
      statusWarning({ ...base, vector_backend: 'json', warnings: [7, 'real message'] } as unknown as ApiStatus),
    ).toEqual({ label: '⚠ Vector search degraded', detail: 'real message' });
  });
});
