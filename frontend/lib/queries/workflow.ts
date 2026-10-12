import { useQuery } from "@tanstack/react-query";

import { getWorkflowIntelligence } from "@/lib/api/workflow";

export function useWorkflowIntelligence(ticketId: string | null) {
  return useQuery({
    queryKey: ["workflow-intelligence", ticketId],
    queryFn: () => getWorkflowIntelligence(ticketId as string),
    enabled: Boolean(ticketId),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });
}