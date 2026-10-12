import { useQuery } from "@tanstack/react-query";

import {
  getAgentIntelligence,
  getTicketIntelligence,
  getWorkflowIntelligence,
} from "@/lib/api/intelligence";

export function useTicketIntelligence(ticketId: string) {
  return useQuery({
    queryKey: ["ticket-intelligence", ticketId],
    queryFn: () => getTicketIntelligence(ticketId),
    enabled: Boolean(ticketId),
  });
}

export function useWorkflowIntelligence(ticketId: string) {
  return useQuery({
    queryKey: ["workflow-intelligence", ticketId],
    queryFn: () => getWorkflowIntelligence(ticketId),
    enabled: Boolean(ticketId),
  });
}

export function useAgentIntelligence(agentId: string) {
  return useQuery({
    queryKey: ["agent-intelligence", agentId],
    queryFn: () => getAgentIntelligence(agentId),
    enabled: Boolean(agentId),
  });
}