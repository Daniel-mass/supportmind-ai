import { apiGet } from "@/lib/api/client";

export type DistributionItem = {
  label: string;
  count: number;
};

export type TeamWorkload = {
  team: string;
  ticket_count: number;
  open_ticket_count: number;
  closed_ticket_count: number;
  high_priority_ticket_count: number;
  waiting_ticket_count: number;
  total_lifecycle_seconds: number;
  average_lifecycle_seconds: number;
  total_waiting_seconds: number;
  average_waiting_seconds: number;
  total_transition_count: number;
  total_handoff_count: number;
};

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

export type EmergingIssue = {
  label: string;
  count: number;
};

export type AgentAnalytics = {
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

export type TeamAnalytics = {
  [key: string]: unknown;
};

export type OrganizationMetrics = {
  agent_count: number;
  active_agent_count: number;
  team_count: number;
  total_ticket_count: number;
  open_ticket_count: number;
  closed_ticket_count: number;
  high_priority_ticket_count: number;
  waiting_ticket_count: number;
  total_lifecycle_seconds: number;
  average_lifecycle_seconds: number;
  total_waiting_seconds: number;
  average_waiting_seconds: number;
  total_transition_count: number;
  total_handoff_count: number;

  priority_distribution: DistributionItem[];
  status_distribution: DistributionItem[];
  team_workload: TeamWorkload[];
  emerging_issues: EmergingIssue[];
  bottlenecks: Bottleneck[];
  workflow_variants: WorkflowVariant[];
};

export type OrganizationIntelligenceResponse = {
  organization: OrganizationMetrics;
  agent_analytics: AgentAnalytics[];
  team_analytics: TeamAnalytics[];
  bottlenecks: Bottleneck[];
  workflow_variants: WorkflowVariant[];
  emerging_issues: EmergingIssue[];
};

export function getOrganizationIntelligence() {
  return apiGet<OrganizationIntelligenceResponse>(
    "/api/v1/organization/intelligence",
  );
}