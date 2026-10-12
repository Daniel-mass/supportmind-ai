import { apiGet } from "@/lib/api/client";

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

export type WorkflowHandoff = {
  [key: string]: unknown;
};

export type WorkflowBottleneck = {
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

export type WorkflowIntelligenceResponse = {
  ticket_id: string;
  events: WorkflowEvent[];
  metrics: WorkflowMetrics;
  handoffs: WorkflowHandoff[];
  bottlenecks: WorkflowBottleneck[];
  variants: WorkflowVariant[];
};

export function getWorkflowIntelligence(ticketId: string) {
  return apiGet<WorkflowIntelligenceResponse>(
    `/api/v1/workflow/${encodeURIComponent(ticketId)}/intelligence`,
  );
}