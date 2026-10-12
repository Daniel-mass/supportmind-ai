"use client";

import { useMemo } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Clock3,
  GitBranch,
  Layers3,
  RefreshCw,
  Timer,
  TrendingUp,
  Users,
  Workflow,
} from "lucide-react";

import { useOrganizationIntelligence } from "@/lib/queries/organization";

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0m";
  }

  const totalMinutes = Math.round(seconds / 60);
  const days = Math.floor(totalMinutes / (60 * 24));
  const hours = Math.floor((totalMinutes % (60 * 24)) / 60);
  const minutes = totalMinutes % 60;

  if (days > 0) {
    return `${days}d ${hours}h`;
  }

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  return `${minutes}m`;
}

function formatPercent(value: number): string {
  if (!Number.isFinite(value)) {
    return "0.0%";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function humanizeStatus(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function getSeverity(score: number): "critical" | "high" | "moderate" | "low" {
  if (score >= 0.2) {
    return "critical";
  }

  if (score >= 0.1) {
    return "high";
  }

  if (score > 0) {
    return "moderate";
  }

  return "low";
}

function severityClasses(
  severity: "critical" | "high" | "moderate" | "low",
): string {
  switch (severity) {
    case "critical":
      return "border-red-500/20 bg-red-500/10 text-red-300";
    case "high":
      return "border-orange-500/20 bg-orange-500/10 text-orange-300";
    case "moderate":
      return "border-yellow-500/20 bg-yellow-500/10 text-yellow-300";
    case "low":
      return "border-slate-700 bg-slate-800/60 text-slate-300";
  }
}

function MetricCard({
  label,
  value,
  description,
  icon: Icon,
}: {
  label: string;
  value: string;
  description: string;
  icon: typeof Activity;
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-slate-400">{label}</p>
          <p className="mt-2 text-2xl font-semibold tracking-tight text-white">
            {value}
          </p>
          <p className="mt-1 text-xs text-slate-500">{description}</p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900 p-2.5">
          <Icon className="h-5 w-5 text-slate-300" />
        </div>
      </div>
    </div>
  );
}

export default function OperationalInsightsPage() {
  const { data, isLoading, isError, error, refetch } =
    useOrganizationIntelligence();

  const organization = data?.organization;

  const topBottlenecks = useMemo(() => {
    return [...(data?.bottlenecks ?? [])]
      .sort((first, second) => {
        return second.bottleneck_score - first.bottleneck_score;
      })
      .slice(0, 8);
  }, [data?.bottlenecks]);

  const topWorkflowVariants = useMemo(() => {
    return [...(data?.workflow_variants ?? [])]
      .sort((first, second) => {
        return second.ticket_count - first.ticket_count;
      })
      .slice(0, 8);
  }, [data?.workflow_variants]);

  const priorityPressure = useMemo(() => {
    const distribution = organization?.priority_distribution ?? [];

    const highPriority = distribution
      .filter((item) => {
        const normalized = item.label.toLowerCase();
        return normalized === "high" || normalized === "highest";
      })
      .reduce((total, item) => total + item.count, 0);

    const blocker = distribution
      .filter((item) => item.label.toLowerCase() === "blocker")
      .reduce((total, item) => total + item.count, 0);

    return {
      highPriority,
      blocker,
    };
  }, [organization?.priority_distribution]);

  const statusPressure = useMemo(() => {
    const distribution = organization?.status_distribution ?? [];

    const waiting =
      distribution.find((item) => item.label.toLowerCase() === "waiting")
        ?.count ?? 0;

    const open =
      distribution.find((item) => item.label.toLowerCase() === "open")?.count ??
      0;

    const inProgress =
      distribution.find(
        (item) => item.label.toLowerCase() === "in_progress",
      )?.count ?? 0;

    return {
      waiting,
      open,
      inProgress,
    };
  }, [organization?.status_distribution]);

  const dominantBottleneck = topBottlenecks[0];

  if (isLoading) {
    return (
      <main className="min-h-screen bg-slate-950 px-6 py-8 text-white">
        <div className="mx-auto max-w-7xl">
          <div className="animate-pulse space-y-6">
            <div className="h-8 w-72 rounded bg-slate-800" />
            <div className="h-4 w-[32rem] max-w-full rounded bg-slate-800" />

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {Array.from({ length: 4 }).map((_, index) => (
                <div
                  key={index}
                  className="h-32 rounded-2xl border border-slate-800 bg-slate-900"
                />
              ))}
            </div>

            <div className="grid gap-6 xl:grid-cols-2">
              <div className="h-96 rounded-2xl border border-slate-800 bg-slate-900" />
              <div className="h-96 rounded-2xl border border-slate-800 bg-slate-900" />
            </div>
          </div>
        </div>
      </main>
    );
  }

  if (isError || !organization) {
    return (
      <main className="min-h-screen bg-slate-950 px-6 py-8 text-white">
        <div className="mx-auto max-w-3xl">
          <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-6">
            <div className="flex items-start gap-4">
              <div className="rounded-xl bg-red-500/10 p-3">
                <AlertTriangle className="h-6 w-6 text-red-400" />
              </div>

              <div className="flex-1">
                <h1 className="text-lg font-semibold">
                  Unable to load operational intelligence
                </h1>

                <p className="mt-2 text-sm text-slate-400">
                  {error instanceof Error
                    ? error.message
                    : "The organization intelligence API did not return usable data."}
                </p>

                <button
                  type="button"
                  onClick={() => void refetch()}
                  className="mt-4 inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800"
                >
                  <RefreshCw className="h-4 w-4" />
                  Retry
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-8 text-white">
      <div className="mx-auto max-w-7xl space-y-8">
        <header className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="mb-3 flex items-center gap-2 text-sm text-slate-500">
              <span>Organization</span>
              <span>/</span>
              <span className="text-slate-300">Operational Insights</span>
            </div>

            <h1 className="text-3xl font-semibold tracking-tight">
              Operational Insights
            </h1>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              Understand where the support organization spends time, where
              workflows become expensive, and how operational complexity is
              distributed across real ticket activity.
            </p>
          </div>

          <button
            type="button"
            onClick={() => void refetch()}
            className="inline-flex w-fit items-center gap-2 rounded-xl border border-slate-800 bg-slate-900 px-4 py-2.5 text-sm font-medium text-slate-200 transition hover:border-slate-700 hover:bg-slate-800"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh intelligence
          </button>
        </header>

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            label="Average lifecycle"
            value={formatDuration(organization.average_lifecycle_seconds)}
            description="Average end-to-end ticket lifecycle"
            icon={Clock3}
          />

          <MetricCard
            label="Waiting time"
            value={formatDuration(
              organization.waiting_ticket_count > 0
                ? organization.total_lifecycle_seconds
                : 0,
            )}
            description={`${formatCount(organization.waiting_ticket_count)} tickets currently waiting`}
            icon={Timer}
          />

          <MetricCard
            label="Workflow transitions"
            value={formatCount(organization.total_transition_count)}
            description="Recorded status transitions"
            icon={Workflow}
          />

          <MetricCard
            label="Handoffs"
            value={formatCount(organization.total_handoff_count)}
            description="Recorded ownership transitions"
            icon={Users}
          />
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.35fr_0.65fr]">
          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-orange-400" />
                  <h2 className="text-lg font-semibold">
                    Primary operational bottleneck
                  </h2>
                </div>

                <p className="mt-2 text-sm text-slate-400">
                  The highest-scoring bottleneck returned by the organization
                  intelligence service.
                </p>
              </div>

              {dominantBottleneck && (
                <span
                  className={`rounded-full border px-3 py-1 text-xs font-medium ${severityClasses(
                    getSeverity(dominantBottleneck.bottleneck_score),
                  )}`}
                >
                  {getSeverity(dominantBottleneck.bottleneck_score)}
                </span>
              )}
            </div>

            {dominantBottleneck ? (
              <div className="mt-6 grid gap-6 md:grid-cols-2">
                <div>
                  <p className="text-sm text-slate-500">State</p>
                  <p className="mt-1 text-2xl font-semibold text-white">
                    {humanizeStatus(dominantBottleneck.status)}
                  </p>

                  <p className="mt-4 text-sm leading-6 text-slate-400">
                    This state accounts for{" "}
                    <span className="font-medium text-slate-200">
                      {formatPercent(dominantBottleneck.duration_share)}
                    </span>{" "}
                    of the analyzed lifecycle duration and recurs across the
                    organization&apos;s workflow observations.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                    <p className="text-xs text-slate-500">Intervals</p>
                    <p className="mt-2 text-xl font-semibold">
                      {formatCount(dominantBottleneck.interval_count)}
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                    <p className="text-xs text-slate-500">Average duration</p>
                    <p className="mt-2 text-xl font-semibold">
                      {formatDuration(dominantBottleneck.average_seconds)}
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                    <p className="text-xs text-slate-500">Duration share</p>
                    <p className="mt-2 text-xl font-semibold">
                      {formatPercent(dominantBottleneck.duration_share)}
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
                    <p className="text-xs text-slate-500">Bottleneck score</p>
                    <p className="mt-2 text-xl font-semibold">
                      {dominantBottleneck.bottleneck_score.toFixed(3)}
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <p className="mt-6 text-sm text-slate-500">
                No bottleneck records were returned.
              </p>
            )}
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-6">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-slate-300" />
              <h2 className="text-lg font-semibold">Pressure signals</h2>
            </div>

            <div className="mt-5 space-y-4">
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                <div className="flex items-center justify-between gap-4">
                  <span className="text-sm text-slate-400">
                    High / highest priority
                  </span>
                  <span className="font-semibold text-white">
                    {formatCount(priorityPressure.highPriority)}
                  </span>
                </div>

                <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full rounded-full bg-orange-400"
                    style={{
                      width: `${Math.min(
                        100,
                        organization.total_ticket_count > 0
                          ? (priorityPressure.highPriority /
                              organization.total_ticket_count) *
                              100
                          : 0,
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                <div className="flex items-center justify-between gap-4">
                  <span className="text-sm text-slate-400">
                    Blocker tickets
                  </span>
                  <span className="font-semibold text-white">
                    {formatCount(priorityPressure.blocker)}
                  </span>
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                <div className="flex items-center justify-between gap-4">
                  <span className="text-sm text-slate-400">Waiting</span>
                  <span className="font-semibold text-white">
                    {formatCount(statusPressure.waiting)}
                  </span>
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                <div className="flex items-center justify-between gap-4">
                  <span className="text-sm text-slate-400">In progress</span>
                  <span className="font-semibold text-white">
                    {formatCount(statusPressure.inProgress)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-2">
          <div className="rounded-2xl border border-slate-800 bg-slate-950/70">
            <div className="border-b border-slate-800 p-6">
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-orange-400" />
                <div>
                  <h2 className="text-lg font-semibold">
                    Workflow bottlenecks
                  </h2>
                  <p className="mt-1 text-sm text-slate-500">
                    States consuming the most lifecycle time or recurring most
                    frequently.
                  </p>
                </div>
              </div>
            </div>

            <div className="divide-y divide-slate-800">
              {topBottlenecks.map((bottleneck) => {
                const severity = getSeverity(bottleneck.bottleneck_score);

                return (
                  <div
                    key={bottleneck.status}
                    className="p-5 transition hover:bg-slate-900/40"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="font-medium text-white">
                          {humanizeStatus(bottleneck.status)}
                        </p>
                        <p className="mt-1 text-xs text-slate-500">
                          {formatCount(bottleneck.interval_count)} intervals
                          {" · "}
                          {formatDuration(bottleneck.average_seconds)} average
                        </p>
                      </div>

                      <span
                        className={`rounded-full border px-2.5 py-1 text-[11px] font-medium ${severityClasses(
                          severity,
                        )}`}
                      >
                        {bottleneck.bottleneck_score.toFixed(3)}
                      </span>
                    </div>

                    <div className="mt-4">
                      <div className="mb-2 flex items-center justify-between text-xs">
                        <span className="text-slate-500">
                          Lifecycle duration share
                        </span>
                        <span className="font-medium text-slate-300">
                          {formatPercent(bottleneck.duration_share)}
                        </span>
                      </div>

                      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                        <div
                          className="h-full rounded-full bg-slate-300"
                          style={{
                            width: `${Math.min(
                              100,
                              bottleneck.duration_share * 100,
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}

              {topBottlenecks.length === 0 && (
                <div className="p-6 text-sm text-slate-500">
                  No bottlenecks were returned by the API.
                </div>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/70">
            <div className="border-b border-slate-800 p-6">
              <div className="flex items-center gap-2">
                <GitBranch className="h-5 w-5 text-slate-300" />
                <div>
                  <h2 className="text-lg font-semibold">
                    Workflow complexity
                  </h2>
                  <p className="mt-1 text-sm text-slate-500">
                    The most common workflow variants across the organization.
                  </p>
                </div>
              </div>
            </div>

            <div className="divide-y divide-slate-800">
              {topWorkflowVariants.map((variant, index) => (
                <div
                  key={variant.variant_id}
                  className="p-5 transition hover:bg-slate-900/40"
                >
                  <div className="flex items-start gap-4">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-900 text-xs font-semibold text-slate-400">
                      {index + 1}
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <p className="text-sm font-medium text-white">
                            {variant.variant_id}
                          </p>

                          <p className="mt-1 break-words text-xs leading-5 text-slate-500">
                            {variant.path
                              .map((status) => humanizeStatus(status))
                              .join(" → ")}
                          </p>
                        </div>

                        <div className="shrink-0 text-right">
                          <p className="font-semibold text-white">
                            {formatCount(variant.ticket_count)}
                          </p>
                          <p className="text-xs text-slate-500">
                            {variant.percentage.toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {topWorkflowVariants.length === 0 && (
                <div className="p-6 text-sm text-slate-500">
                  No workflow variants were returned by the API.
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-5">
            <div className="flex items-center gap-3">
              <Layers3 className="h-5 w-5 text-slate-400" />
              <span className="text-sm text-slate-400">
                Workflow variants
              </span>
            </div>

            <p className="mt-4 text-2xl font-semibold">
              {formatCount(organization.workflow_variants.length)}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              Distinct workflow paths observed
            </p>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-5">
            <div className="flex items-center gap-3">
              <Users className="h-5 w-5 text-slate-400" />
              <span className="text-sm text-slate-400">Active agents</span>
            </div>

            <p className="mt-4 text-2xl font-semibold">
              {formatCount(organization.active_agent_count)}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              Of {formatCount(organization.agent_count)} total agents
            </p>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-5">
            <div className="flex items-center gap-3">
              <Activity className="h-5 w-5 text-slate-400" />
              <span className="text-sm text-slate-400">
                Average transitions
              </span>
            </div>

            <p className="mt-4 text-2xl font-semibold">
              {organization.total_ticket_count > 0
                ? (
                    organization.total_transition_count /
                    organization.total_ticket_count
                  ).toFixed(1)
                : "0.0"}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              Transitions per ticket
            </p>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-5">
            <div className="flex items-center gap-3">
              <TrendingUp className="h-5 w-5 text-slate-400" />
              <span className="text-sm text-slate-400">
                Handoff intensity
              </span>
            </div>

            <p className="mt-4 text-2xl font-semibold">
              {organization.total_ticket_count > 0
                ? (
                    organization.total_handoff_count /
                    organization.total_ticket_count
                  ).toFixed(2)
                : "0.00"}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              Handoffs per ticket
            </p>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-950/70 p-6">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="flex items-center gap-2">
                <ArrowDownRight className="h-5 w-5 text-slate-400" />
                <h2 className="text-lg font-semibold">
                  Operational interpretation
                </h2>
              </div>

              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
                These observations are derived directly from the organization
                intelligence dataset. They describe where time and workflow
                activity are concentrated; they do not represent automated
                business recommendations.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3 text-center">
                <p className="text-lg font-semibold">
                  {formatCount(organization.total_ticket_count)}
                </p>
                <p className="text-[11px] text-slate-500">Tickets</p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3 text-center">
                <p className="text-lg font-semibold">
                  {formatCount(organization.total_transition_count)}
                </p>
                <p className="text-[11px] text-slate-500">Transitions</p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3 text-center">
                <p className="text-lg font-semibold">
                  {formatCount(organization.total_handoff_count)}
                </p>
                <p className="text-[11px] text-slate-500">Handoffs</p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3 text-center">
                <p className="text-lg font-semibold">
                  {formatCount(organization.bottlenecks.length)}
                </p>
                <p className="text-[11px] text-slate-500">Bottlenecks</p>
              </div>
            </div>
          </div>
        </section>

        <footer className="flex flex-col gap-2 border-t border-slate-800 pt-5 text-xs text-slate-600 sm:flex-row sm:items-center sm:justify-between">
          <span>
            Source: GET /api/v1/organization/intelligence
          </span>

          <span>
            Real SupportMind organization intelligence · No synthetic metrics
          </span>
        </footer>
      </div>
    </main>
  );
}