import { useQuery } from '@tanstack/react-query';
import { getStatus } from '../api/client';

export function useStatus() {
  return useQuery({
    queryKey: ['status'],
    queryFn: getStatus,
    staleTime: 30_000,
    // Poll so a mid-session vector-store fallback or Ollama outage shows up without a reload.
    refetchInterval: 30_000,
  });
}
