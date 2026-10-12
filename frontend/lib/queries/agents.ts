import { useQuery } from "@tanstack/react-query";

import { getAgentIntelligence } from "@/lib/api/agents";

export function useAgentIntelligence(agentId: string | null) {
  return useQuery({
    queryKey: ["agent-intelligence", agentId],
    queryFn: () => getAgentIntelligence(agentId as string),
    enabled: Boolean(agentId),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  });
}