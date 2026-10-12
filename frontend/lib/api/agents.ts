import { apiGet } from "@/lib/api/client";

export type AgentIntelligence = {
  agent_id: string;
  name: string | null;
  team: string | null;
  active: boolean;

  assigned_ticket_count: number;
  open_ticket_count: number;
  closed_ticket_count: number;
  high_priority_ticket_count: number;
  waiting_ticket_count: number;

  total_lifecycle_seconds: number;
  average_lifecycle_seconds: number;

  total_waiting_seconds: number;
  average_waiting_seconds: number;

  total_transition_count: number;

  incoming_handoff_count: number;
  outgoing_handoff_count: number;

  unique_project_count: number;
  unique_issue_type_count: number;
};

export function getAgentIntelligence(agentId: string) {
  return apiGet<AgentIntelligence>(
    `/api/v1/agents/${encodeURIComponent(agentId)}/intelligence`,
  );
}