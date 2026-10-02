import type { ApiStatus } from '../../api/client';

export type StatusWarning = { label: string; detail: string };

/** Badge shown in the status bar when the backend reports degraded operation. */
export function statusWarning(status: ApiStatus | undefined): StatusWarning | null {
  // API payloads are outside data: tolerate a non-array or non-string entries.
  const raw: unknown = status?.warnings;
  const warnings = Array.isArray(raw)
    ? raw.filter((warning): warning is string => typeof warning === 'string' && warning.trim() !== '')
    : [];
  if (warnings.length === 0) {
    return null;
  }
  const label = status?.vector_backend === 'json' ? '⚠ Vector search degraded' : '⚠ Backend warning';
  return { label, detail: warnings.join('\n') };
}
