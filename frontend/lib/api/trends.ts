
import { apiGet } from "@/lib/api/client";

export type MonthlyTrendPoint = {
  month: string;
  count: number;
};

export type TrendCategory = {
  month: string;
  category: string;
  count: number;
};

export type OrganizationTrendsResponse = {
  months: string[];
  ticket_volume: MonthlyTrendPoint[];
  issue_type_trends: TrendCategory[];
  priority_trends: TrendCategory[];
  status_trends: TrendCategory[];
  total_tickets: number;
  total_workflow_events: number;
};

export function getOrganizationTrends() {
  return apiGet<OrganizationTrendsResponse>(
    "/api/v1/organization/trends",
  );
}