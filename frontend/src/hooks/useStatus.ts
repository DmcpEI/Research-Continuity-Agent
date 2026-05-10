import { useQuery } from '@tanstack/react-query';
import { getStatus } from '../api/client';

export function useStatus() {
  return useQuery({
    queryKey: ['status'],
    queryFn: getStatus,
    staleTime: 30_000,
  });
}
