import { apiGet } from "@/lib/api/client";

export type TicketRecord = {
  ticket_id: string;
  project: string;
  issue_num: number;
  issue_type: string;
  priority: string;
  status: string;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
  resolution_date: string | null;
  resolution: string | null;
  reporter: string | null;
  assignee: string | null;
  contributor_count: number;
  comments_count: number;
  source: string;
  metadata: Record<string, unknown>;
};

export type TicketAnalytics = {
  ticket_id: string;
  project: string;
  issue_num: number;
  issue_type: string;
  priority: string;
  status: string;
  lifecycle_seconds: number;
  active_seconds: number;
  waiting_seconds: number;
  active_ratio: number;
  waiting_ratio: number;
  transition_count: number;
  unique_status_count: number;
  unique_assignee_count: number;
  handoff_count: number;
  is_high_priority: boolean;
  is_waiting: boolean;
};

export type WorkflowMetrics = {
  ticket_id: string;
  lifecycle_seconds: number;
  active_seconds: number;
  waiting_seconds: number;
  status_durations: Record<string, number>;
  interval_count: number;
  transition_count: number;
  unique_status_count: number;
  unique_assignee_count: number;
};

export type WorkflowEvent = {
  event_id: string;
  ticket_id: string;
  timestamp: string;
  activity: string;
  status: string | null;
  assignee: string | null;
  team: string | null;
  priority: string | null;
  category: string | null;
  source: string;
  metadata: Record<string, unknown>;
};

export type Handoff = Record<string, unknown>;

export type Bottleneck = {
  status: string;
  interval_count: number;
  ticket_count: number;
  total_seconds: number;
  average_seconds: number;
  volume_share: number;
  duration_share: number;
  recurrence_rate: number;
  bottleneck_score: number;
};

export type WorkflowVariant = {
  variant_id: string;
  path: string[];
  path_string: string;
  ticket_count: number;
  percentage: number;
};

export type TicketIntelligenceResponse = {
  ticket: TicketRecord;
  analytics: TicketAnalytics;
  workflow_metrics: WorkflowMetrics;
  workflow_events: WorkflowEvent[];
  handoffs: Handoff[];
};

export type WorkflowIntelligenceResponse = {
  ticket_id: string;
  events: WorkflowEvent[];
  metrics: WorkflowMetrics;
  handoffs: Handoff[];
  bottlenecks: Bottleneck[];
  variants: WorkflowVariant[];
};

export type AgentIntelligenceResponse = {
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

export function getTicketIntelligence(ticketId: string) {
  return apiGet<TicketIntelligenceResponse>(
    `/api/v1/tickets/${encodeURIComponent(ticketId)}/intelligence`,
  );
}

export function getWorkflowIntelligence(ticketId: string) {
  return apiGet<WorkflowIntelligenceResponse>(
    `/api/v1/workflow/${encodeURIComponent(ticketId)}/intelligence`,
  );
}

export function getAgentIntelligence(agentId: string) {
  return apiGet<AgentIntelligenceResponse>(
    `/api/v1/agents/${encodeURIComponent(agentId)}/intelligence`,
  );
}