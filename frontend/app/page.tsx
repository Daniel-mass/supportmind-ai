"use client";

import {
  Activity,
  AlertTriangle,
  Bot,
  CheckCircle2,
  Clock3,
  GitBranch,
  Ticket,
  Users,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { MetricCard } from "@/components/intelligence/MetricCard";
import { useOrganizationIntelligence } from "@/lib/queries/organization";

export default function Home() {
  const { data, isLoading, isError, error } =
    useOrganizationIntelligence();

  if (isLoading) {
    return <OverviewLoading />;
  }

  if (isError || !data) {
    return (
      <main className="min-h-screen bg-background p-6 lg:p-8">
        <div className="mx-auto max-w-7xl">
          <div className="rounded-xl border bg-card p-6 shadow-sm">
            <div className="flex items-start gap-3">
              <AlertTriangle className="mt-0.5 h-5 w-5 text-destructive" />

              <div>
                <h1 className="font-semibold">
                  Unable to load organization intelligence
                </h1>

                <p className="mt-1 text-sm text-muted-foreground">
                  {error instanceof Error
                    ? error.message
                    : "The SupportMind API could not be reached."}
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    );
  }

  const organization = data.organization;

  /*
   * The backend returns thousands of workflow variants.
   * For the Overview we only visualize the most important paths.
   */
  const workflowChartData = [...organization.workflow_variants]
    .sort((a, b) => b.ticket_count - a.ticket_count)
    .slice(0, 6)
    .map((variant) => ({
      path: variant.path.join(" → "),
      tickets: variant.ticket_count,
      percentage: variant.percentage,
    }));

  /*
   * Bottleneck data is already ranked by the backend analysis,
   * but sorting here makes the visualization deterministic.
   */
  const bottleneckChartData = [...organization.bottlenecks]
    .sort((a, b) => b.average_seconds - a.average_seconds)
    .slice(0, 8)
    .map((bottleneck) => ({
      status: bottleneck.status,
      average_seconds: bottleneck.average_seconds,
      ticket_count: bottleneck.ticket_count,
      bottleneck_score: bottleneck.bottleneck_score,
    }));

  return (
    <main className="min-h-screen bg-background p-6 lg:p-8">
      <div className="mx-auto max-w-7xl space-y-8">
        {/* Header */}
        <section>
          <div className="flex flex-col gap-2">
            <p className="text-sm font-medium text-muted-foreground">
              SupportMind AI
            </p>

            <h1 className="text-3xl font-semibold tracking-tight">
              Organization Overview
            </h1>

            <p className="max-w-3xl text-sm text-muted-foreground">
              Operational intelligence across tickets, agents,
              workflows, priorities, and support activity.
            </p>
          </div>
        </section>

        {/* KPI Grid */}
        <section>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              label="Total Tickets"
              value={formatNumber(organization.total_ticket_count)}
              description={`${formatNumber(
                organization.open_ticket_count,
              )} currently open`}
              icon={<Ticket className="h-5 w-5" />}
            />

            <MetricCard
              label="Closed Tickets"
              value={formatNumber(organization.closed_ticket_count)}
              description={formatPercentage(
                organization.closed_ticket_count,
                organization.total_ticket_count,
              )}
              icon={<CheckCircle2 className="h-5 w-5" />}
            />

            <MetricCard
              label="High Priority"
              value={formatNumber(
                organization.high_priority_ticket_count,
              )}
              description={formatPercentage(
                organization.high_priority_ticket_count,
                organization.total_ticket_count,
              )}
              icon={<AlertTriangle className="h-5 w-5" />}
            />

            <MetricCard
              label="Waiting Tickets"
              value={formatNumber(organization.waiting_ticket_count)}
              description="Tickets currently waiting"
              icon={<Clock3 className="h-5 w-5" />}
            />
          </div>
        </section>

        {/* Operational Metrics */}
        <section>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              label="Active Agents"
              value={formatNumber(organization.active_agent_count)}
              description={`${formatNumber(
                organization.agent_count,
              )} total agents`}
              icon={<Users className="h-5 w-5" />}
            />

            <MetricCard
              label="Workflow Transitions"
              value={formatNumber(
                organization.total_transition_count,
              )}
              description="Across all tickets"
              icon={<Activity className="h-5 w-5" />}
            />

            <MetricCard
              label="Handoffs"
              value={formatNumber(
                organization.total_handoff_count,
              )}
              description="Agent-to-agent transfers"
              icon={<GitBranch className="h-5 w-5" />}
            />

            <MetricCard
              label="Average Lifecycle"
              value={formatDuration(
                organization.average_lifecycle_seconds,
              )}
              description="Average ticket lifecycle"
              icon={<Clock3 className="h-5 w-5" />}
            />
          </div>
        </section>

        {/* Workflow Intelligence Charts */}
        <section>
          <div className="grid gap-6 xl:grid-cols-2">
            <WorkflowPathChart data={workflowChartData} />

            <BottleneckChart data={bottleneckChartData} />
          </div>
        </section>

        {/* Distribution Panels */}
        <section className="grid gap-6 lg:grid-cols-2">
          <DistributionPanel
            title="Priority Distribution"
            description="Ticket volume by priority."
            items={organization.priority_distribution}
          />

          <DistributionPanel
            title="Status Distribution"
            description="Current ticket distribution across workflow statuses."
            items={organization.status_distribution}
          />
        </section>

        {/* Intelligence Summary */}
        <section className="grid gap-6 lg:grid-cols-3">
          <SummaryPanel
            title="Bottlenecks"
            value={organization.bottlenecks.length}
            description="Detected workflow bottleneck states."
            icon={<AlertTriangle className="h-5 w-5" />}
          />

          <SummaryPanel
            title="Workflow Variants"
            value={organization.workflow_variants.length}
            description="Distinct workflow paths identified."
            icon={<GitBranch className="h-5 w-5" />}
          />

          <SummaryPanel
            title="Agents"
            value={data.agent_analytics.length}
            description="Agent-level intelligence records."
            icon={<Bot className="h-5 w-5" />}
          />
        </section>
      </div>
    </main>
  );
}

/* -------------------------------------------------------------------------- */
/* Workflow Path Chart                                                        */
/* -------------------------------------------------------------------------- */

type WorkflowChartRow = {
  path: string;
  tickets: number;
  percentage: number;
};

function WorkflowPathChart({
  data,
}: {
  data: WorkflowChartRow[];
}) {
  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm">
      <div className="mb-5">
        <h2 className="text-base font-semibold">
          Top Workflow Paths
        </h2>

        <p className="mt-1 text-sm text-muted-foreground">
          Most common ticket journeys through the support workflow.
        </p>
      </div>

      {data.length === 0 ? (
        <EmptyChartState message="No workflow path data available." />
      ) : (
        <div className="h-[360px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              layout="vertical"
              margin={{
                top: 8,
                right: 24,
                bottom: 8,
                left: 12,
              }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                horizontal={false}
              />

              <XAxis
                type="number"
                tickFormatter={formatCompactNumber}
              />

              <YAxis
                type="category"
                dataKey="path"
                width={230}
                tick={{ fontSize: 11 }}
              />

              <Tooltip
                formatter={(value, name) => {
                  if (name === "tickets") {
                    return [
                      formatNumber(Number(value)),
                      "Tickets",
                    ];
                  }

                  return [value, name];
                }}
                labelFormatter={(label) =>
                  `Workflow: ${label}`
                }
              />

              <Bar
                dataKey="tickets"
                name="Tickets"
                radius={[0, 6, 6, 0]}
                barSize={28}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Bottleneck Chart                                                           */
/* -------------------------------------------------------------------------- */

type BottleneckChartRow = {
  status: string;
  average_seconds: number;
  ticket_count: number;
  bottleneck_score: number;
};

function BottleneckChart({
  data,
}: {
  data: BottleneckChartRow[];
}) {
  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm">
      <div className="mb-5">
        <h2 className="text-base font-semibold">
          Workflow Bottleneck Duration
        </h2>

        <p className="mt-1 text-sm text-muted-foreground">
          Average time spent in the most time-consuming workflow states.
        </p>
      </div>

      {data.length === 0 ? (
        <EmptyChartState message="No bottleneck data available." />
      ) : (
        <div className="h-[360px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              layout="vertical"
              margin={{
                top: 8,
                right: 24,
                bottom: 8,
                left: 12,
              }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                horizontal={false}
              />

              <XAxis
                type="number"
                tickFormatter={(value) =>
                  formatCompactDuration(Number(value))
                }
              />

              <YAxis
                type="category"
                dataKey="status"
                width={150}
                tick={{ fontSize: 12 }}
              />

              <Tooltip
                formatter={(value, name) => {
                  if (name === "average_seconds") {
                    return [
                      formatDuration(Number(value)),
                      "Average duration",
                    ];
                  }

                  return [value, name];
                }}
                labelFormatter={(label) =>
                  `Status: ${label}`
                }
              />

              <Bar
                dataKey="average_seconds"
                name="average_seconds"
                radius={[0, 6, 6, 0]}
                barSize={28}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Distribution                                                               */
/* -------------------------------------------------------------------------- */

type DistributionPanelProps = {
  title: string;
  description: string;
  items: {
    label: string;
    count: number;
  }[];
};

function DistributionPanel({
  title,
  description,
  items,
}: DistributionPanelProps) {
  const total = items.reduce((sum, item) => sum + item.count, 0);

  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm">
      <div className="mb-5">
        <h2 className="text-base font-semibold">{title}</h2>

        <p className="mt-1 text-sm text-muted-foreground">
          {description}
        </p>
      </div>

      {items.length === 0 ? (
        <div className="flex h-48 items-center justify-center rounded-lg border border-dashed">
          <p className="text-sm text-muted-foreground">
            No data available.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {items.map((item) => {
            const percentage =
              total > 0 ? (item.count / total) * 100 : 0;

            return (
              <div key={item.label}>
                <div className="mb-1.5 flex items-center justify-between gap-4">
                  <span className="truncate text-sm font-medium">
                    {item.label}
                  </span>

                  <span className="shrink-0 text-sm text-muted-foreground">
                    {formatNumber(item.count)} ·{" "}
                    {percentage.toFixed(1)}%
                  </span>
                </div>

                <div className="h-2 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-primary transition-all"
                    style={{
                      width: `${Math.min(percentage, 100)}%`,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Summary                                                                    */
/* -------------------------------------------------------------------------- */

type SummaryPanelProps = {
  title: string;
  value: number;
  description: string;
  icon: React.ReactNode;
};

function SummaryPanel({
  title,
  value,
  description,
  icon,
}: SummaryPanelProps) {
  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-muted-foreground">
            {title}
          </p>

          <p className="mt-2 text-3xl font-semibold tracking-tight">
            {formatNumber(value)}
          </p>

          <p className="mt-1 text-sm text-muted-foreground">
            {description}
          </p>
        </div>

        <div className="rounded-lg border bg-muted/40 p-2">
          {icon}
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Loading                                                                    */
/* -------------------------------------------------------------------------- */

function OverviewLoading() {
  return (
    <main className="min-h-screen bg-background p-6 lg:p-8">
      <div className="mx-auto max-w-7xl space-y-8">
        <section>
          <div className="space-y-3">
            <div className="h-4 w-32 animate-pulse rounded bg-muted" />
            <div className="h-9 w-72 animate-pulse rounded bg-muted" />
            <div className="h-4 w-full max-w-2xl animate-pulse rounded bg-muted" />
          </div>
        </section>

        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 8 }).map((_, index) => (
            <div
              key={index}
              className="h-32 animate-pulse rounded-xl border bg-card"
            />
          ))}
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <div className="h-80 animate-pulse rounded-xl border bg-card" />
          <div className="h-80 animate-pulse rounded-xl border bg-card" />
        </section>
      </div>
    </main>
  );
}

/* -------------------------------------------------------------------------- */
/* Empty Chart                                                                */
/* -------------------------------------------------------------------------- */

function EmptyChartState({
  message,
}: {
  message: string;
}) {
  return (
    <div className="flex h-[360px] items-center justify-center rounded-lg border border-dashed">
      <p className="text-sm text-muted-foreground">
        {message}
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Formatting                                                                 */
/* -------------------------------------------------------------------------- */

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatCompactNumber(value: number) {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }

  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }

  return String(Math.round(value));
}

function formatPercentage(
  value: number,
  total: number,
) {
  if (total === 0) {
    return "0.0%";
  }

  return `${((value / total) * 100).toFixed(1)}%`;
}

function formatDuration(seconds: number) {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }

  const minutes = Math.floor(seconds / 60);

  if (minutes < 60) {
    return `${minutes}m`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;

  if (hours < 24) {
    return remainingMinutes > 0
      ? `${hours}h ${remainingMinutes}m`
      : `${hours}h`;
  }

  const days = Math.floor(hours / 24);
  const remainingHours = hours % 24;

  return remainingHours > 0
    ? `${days}d ${remainingHours}h`
    : `${days}d`;
}

function formatCompactDuration(seconds: number) {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }

  const minutes = Math.round(seconds / 60);

  if (minutes < 60) {
    return `${minutes}m`;
  }

  const hours = Math.round(minutes / 60);

  if (hours < 24) {
    return `${hours}h`;
  }

  const days = Math.round(hours / 24);

  return `${days}d`;
}