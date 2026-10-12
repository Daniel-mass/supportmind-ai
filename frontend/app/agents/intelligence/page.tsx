"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  ArrowDownLeft,
  ArrowUpRight,
  Bot,
  BriefcaseBusiness,
  CheckCircle2,
  Clock3,
  FolderKanban,
  GitBranch,
  Loader2,
  Search,
  TicketCheck,
  Timer,
  Users,
  XCircle,
} from "lucide-react";

import { useAgentIntelligence } from "@/lib/queries/agents";
import { useOrganizationIntelligence } from "@/lib/queries/organization";
import type { AgentAnalytics } from "@/lib/api/organization";

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(Math.round(value));
}

function formatPercentage(value: number) {
  return `${value.toFixed(1)}%`;
}

function formatDuration(totalSeconds: number) {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) {
    return "—";
  }

  const seconds = Math.round(totalSeconds);

  const days = Math.floor(seconds / 86_400);
  const hours = Math.floor((seconds % 86_400) / 3_600);
  const minutes = Math.floor((seconds % 3_600) / 60);

  if (days > 0) {
    return `${days}d ${hours}h`;
  }

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  return `${minutes}m`;
}

function getPercentage(part: number, total: number) {
  if (!total) {
    return 0;
  }

  return (part / total) * 100;
}

function StatCard({
  icon: Icon,
  label,
  value,
  description,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  description?: string;
}) {
  return (
    <div className="rounded-2xl border bg-card p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div className="flex size-10 items-center justify-center rounded-xl bg-muted">
          <Icon className="size-5 text-muted-foreground" />
        </div>
      </div>

      <p className="mt-5 text-sm text-muted-foreground">{label}</p>

      <p className="mt-1 text-2xl font-semibold tracking-tight">{value}</p>

      {description ? (
        <p className="mt-1 text-xs text-muted-foreground">{description}</p>
      ) : null}
    </div>
  );
}

function MetricRow({
  label,
  value,
  secondary,
}: {
  label: string;
  value: string;
  secondary?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 border-b py-3 last:border-b-0">
      <span className="text-sm text-muted-foreground">{label}</span>

      <div className="text-right">
        <p className="text-sm font-medium">{value}</p>

        {secondary ? (
          <p className="text-xs text-muted-foreground">{secondary}</p>
        ) : null}
      </div>
    </div>
  );
}

function CoverageBar({
  label,
  value,
  max,
}: {
  label: string;
  value: number;
  max: number;
}) {
  const percentage = max > 0 ? Math.min((value / max) * 100, 100) : 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium">{formatNumber(value)}</span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary transition-all"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default function AgentIntelligencePage() {
  const {
    data: organizationData,
    isLoading: organizationLoading,
    isError: organizationError,
  } = useOrganizationIntelligence();

  const [search, setSearch] = useState("");
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);

  const agents = organizationData?.agent_analytics ?? [];

  const filteredAgents = useMemo(() => {
    const normalized = search.trim().toLowerCase();

    const filtered = normalized
      ? agents.filter((agent) => {
          const haystack = [
            agent.agent_id,
            agent.name ?? "",
            agent.team ?? "",
          ]
            .join(" ")
            .toLowerCase();

          return haystack.includes(normalized);
        })
      : agents;

    return [...filtered].sort(
      (a, b) => b.assigned_ticket_count - a.assigned_ticket_count,
    );
  }, [agents, search]);

  const effectiveAgentId =
    selectedAgentId ??
    filteredAgents[0]?.agent_id ??
    agents[0]?.agent_id ??
    null;

  const {
    data: agent,
    isLoading: agentLoading,
    isError: agentError,
    error,
  } = useAgentIntelligence(effectiveAgentId);

  const selectedAgentSummary = useMemo(
    () =>
      agents.find(
        (candidate) => candidate.agent_id === effectiveAgentId,
      ),
    [agents, effectiveAgentId],
  );

  const closureRate = agent
    ? getPercentage(
        agent.closed_ticket_count,
        agent.assigned_ticket_count,
      )
    : 0;

  const openShare = agent
    ? getPercentage(
        agent.open_ticket_count,
        agent.assigned_ticket_count,
      )
    : 0;

  const highPriorityShare = agent
    ? getPercentage(
        agent.high_priority_ticket_count,
        agent.assigned_ticket_count,
      )
    : 0;

  const handoffTotal = agent
    ? agent.incoming_handoff_count + agent.outgoing_handoff_count
    : 0;

  const handoffBalance = agent
    ? agent.incoming_handoff_count - agent.outgoing_handoff_count
    : 0;

  const maxCoverage = agent
    ? Math.max(
        agent.unique_project_count,
        agent.unique_issue_type_count,
        1,
      )
    : 1;

  if (organizationLoading) {
    return (
      <main className="min-h-screen bg-background">
        <div className="mx-auto max-w-7xl px-6 py-10">
          <div className="flex min-h-[60vh] items-center justify-center">
            <div className="flex items-center gap-3 text-sm text-muted-foreground">
              <Loader2 className="size-5 animate-spin" />
              Loading agent intelligence...
            </div>
          </div>
        </div>
      </main>
    );
  }

  if (organizationError) {
    return (
      <main className="min-h-screen bg-background">
        <div className="mx-auto max-w-7xl px-6 py-10">
          <div className="rounded-2xl border border-destructive/30 bg-destructive/5 p-6">
            <div className="flex items-start gap-3">
              <XCircle className="mt-0.5 size-5 text-destructive" />

              <div>
                <h1 className="font-semibold">
                  Unable to load agent directory
                </h1>

                <p className="mt-1 text-sm text-muted-foreground">
                  The organization intelligence API could not provide the
                  available agents.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
        <div className="flex flex-col gap-6 border-b pb-7 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
              <Bot className="size-4" />
              Agent Operations
            </div>

            <h1 className="text-3xl font-semibold tracking-tight">
              Agent Intelligence
            </h1>

            <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">
              Inspect workload, lifecycle performance, handoffs and operational
              coverage for individual SupportMind agents.
            </p>
          </div>

          {agent ? (
            <div className="flex items-center gap-3 rounded-xl border bg-card px-4 py-3 shadow-sm">
              <div
                className={`size-2.5 rounded-full ${
                  agent.active
                    ? "bg-emerald-500"
                    : "bg-muted-foreground"
                }`}
              />

              <div>
                <p className="text-xs text-muted-foreground">
                  Selected agent
                </p>

                <p className="font-mono text-sm font-medium">
                  {agent.agent_id}
                </p>
              </div>
            </div>
          ) : null}
        </div>

        <section className="mt-6 rounded-2xl border bg-card p-5 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />

                <input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search agent ID, name or team..."
                  className="h-10 w-full rounded-lg border bg-background pl-9 pr-4 text-sm outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/10"
                />
              </div>
            </div>

            <div className="text-sm text-muted-foreground">
              {formatNumber(agents.length)} agents
            </div>
          </div>

          <div className="mt-4 max-h-64 overflow-y-auto rounded-xl border">
            {filteredAgents.length === 0 ? (
              <div className="p-6 text-center text-sm text-muted-foreground">
                No agents match your search.
              </div>
            ) : (
              <div className="divide-y">
                {filteredAgents.slice(0, 100).map((candidate) => {
                  const active =
                    candidate.agent_id === effectiveAgentId;

                  return (
                    <button
                      key={candidate.agent_id}
                      type="button"
                      onClick={() =>
                        setSelectedAgentId(candidate.agent_id)
                      }
                      className={`flex w-full items-center justify-between gap-4 px-4 py-3 text-left transition ${
                        active
                          ? "bg-accent"
                          : "hover:bg-muted/50"
                      }`}
                    >
                      <div className="flex min-w-0 items-center gap-3">
                        <div
                          className={`flex size-8 shrink-0 items-center justify-center rounded-lg ${
                            active
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted"
                          }`}
                        >
                          <Bot className="size-4" />
                        </div>

                        <div className="min-w-0">
                          <p className="truncate font-mono text-sm font-medium">
                            {candidate.agent_id}
                          </p>

                          <p className="truncate text-xs text-muted-foreground">
                            {candidate.name ||
                              candidate.team ||
                              "Identity metadata not provided"}
                          </p>
                        </div>
                      </div>

                      <div className="shrink-0 text-right">
                        <p className="text-sm font-medium">
                          {formatNumber(
                            candidate.assigned_ticket_count,
                          )}
                        </p>

                        <p className="text-xs text-muted-foreground">
                          tickets
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {filteredAgents.length > 100 ? (
            <p className="mt-3 text-xs text-muted-foreground">
              Showing the first 100 matching agents, ranked by assigned
              ticket volume.
            </p>
          ) : null}
        </section>

        {agentLoading ? (
          <div className="mt-6 flex min-h-80 items-center justify-center rounded-2xl border bg-card">
            <div className="flex items-center gap-3 text-sm text-muted-foreground">
              <Loader2 className="size-5 animate-spin" />
              Loading intelligence for{" "}
              <span className="font-mono font-medium text-foreground">
                {effectiveAgentId}
              </span>
              ...
            </div>
          </div>
        ) : agentError ? (
          <div className="mt-6 rounded-2xl border border-destructive/30 bg-destructive/5 p-6">
            <div className="flex items-start gap-3">
              <XCircle className="mt-0.5 size-5 text-destructive" />

              <div>
                <h2 className="font-semibold">
                  Unable to load agent intelligence
                </h2>

                <p className="mt-1 text-sm text-muted-foreground">
                  {error instanceof Error
                    ? error.message
                    : "The agent intelligence API returned an error."}
                </p>
              </div>
            </div>
          </div>
        ) : agent ? (
          <>
            <section className="mt-6 rounded-2xl border bg-card p-6 shadow-sm">
              <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
                <div className="flex items-center gap-4">
                  <div className="flex size-14 items-center justify-center rounded-2xl bg-primary/10">
                    <Bot className="size-7 text-primary" />
                  </div>

                  <div>
                    <div className="flex flex-wrap items-center gap-3">
                      <h2 className="font-mono text-xl font-semibold">
                        {agent.agent_id}
                      </h2>

                      <span
                        className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                          agent.active
                            ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                            : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {agent.active ? "Active" : "Inactive"}
                      </span>
                    </div>

                    <p className="mt-1 text-sm text-muted-foreground">
                      {agent.name || "Agent name not provided"}
                      {" · "}
                      {agent.team || "Team not provided"}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div className="rounded-xl bg-muted/50 px-4 py-3">
                    <p className="text-xs text-muted-foreground">
                      Projects
                    </p>

                    <p className="mt-1 text-lg font-semibold">
                      {formatNumber(agent.unique_project_count)}
                    </p>
                  </div>

                  <div className="rounded-xl bg-muted/50 px-4 py-3">
                    <p className="text-xs text-muted-foreground">
                      Issue types
                    </p>

                    <p className="mt-1 text-lg font-semibold">
                      {formatNumber(agent.unique_issue_type_count)}
                    </p>
                  </div>

                  <div className="rounded-xl bg-muted/50 px-4 py-3">
                    <p className="text-xs text-muted-foreground">
                      Transitions
                    </p>

                    <p className="mt-1 text-lg font-semibold">
                      {formatNumber(agent.total_transition_count)}
                    </p>
                  </div>

                  <div className="rounded-xl bg-muted/50 px-4 py-3">
                    <p className="text-xs text-muted-foreground">
                      Handoffs
                    </p>

                    <p className="mt-1 text-lg font-semibold">
                      {formatNumber(handoffTotal)}
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <section className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
              <StatCard
                icon={TicketCheck}
                label="Assigned tickets"
                value={formatNumber(agent.assigned_ticket_count)}
                description="Total assigned workload"
              />

              <StatCard
                icon={CheckCircle2}
                label="Closed tickets"
                value={formatNumber(agent.closed_ticket_count)}
                description={`${formatPercentage(closureRate)} closure rate`}
              />

              <StatCard
                icon={BriefcaseBusiness}
                label="Open tickets"
                value={formatNumber(agent.open_ticket_count)}
                description={`${formatPercentage(openShare)} of assigned`}
              />

              <StatCard
                icon={Timer}
                label="High priority"
                value={formatNumber(agent.high_priority_ticket_count)}
                description={`${formatPercentage(highPriorityShare)} of assigned`}
              />

              <StatCard
                icon={Clock3}
                label="Waiting tickets"
                value={formatNumber(agent.waiting_ticket_count)}
                description="Currently waiting"
              />
            </section>

            <section className="mt-6 grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl border bg-card p-6 shadow-sm">
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="font-semibold">
                      Lifecycle performance
                    </h2>

                    <p className="mt-1 text-sm text-muted-foreground">
                      Time spent across the agent&apos;s assigned workload.
                    </p>
                  </div>

                  <Clock3 className="size-5 text-muted-foreground" />
                </div>

                <div className="mt-5">
                  <MetricRow
                    label="Average lifecycle"
                    value={formatDuration(
                      agent.average_lifecycle_seconds,
                    )}
                  />

                  <MetricRow
                    label="Total lifecycle"
                    value={formatDuration(
                      agent.total_lifecycle_seconds,
                    )}
                  />

                  <MetricRow
                    label="Average waiting"
                    value={formatDuration(
                      agent.average_waiting_seconds,
                    )}
                  />

                  <MetricRow
                    label="Total waiting"
                    value={formatDuration(
                      agent.total_waiting_seconds,
                    )}
                  />
                </div>
              </div>

              <div className="rounded-2xl border bg-card p-6 shadow-sm">
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="font-semibold">
                      Workload composition
                    </h2>

                    <p className="mt-1 text-sm text-muted-foreground">
                      Distribution of the agent&apos;s current workload.
                    </p>
                  </div>

                  <Activity className="size-5 text-muted-foreground" />
                </div>

                <div className="mt-6 space-y-5">
                  <div>
                    <div className="mb-2 flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">
                        Closed
                      </span>

                      <span className="font-medium">
                        {formatPercentage(closureRate)}
                      </span>
                    </div>

                    <div className="h-2 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-emerald-500"
                        style={{
                          width: `${Math.min(closureRate, 100)}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="mb-2 flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">
                        Open
                      </span>

                      <span className="font-medium">
                        {formatPercentage(openShare)}
                      </span>
                    </div>

                    <div className="h-2 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{
                          width: `${Math.min(openShare, 100)}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="mb-2 flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">
                        High priority
                      </span>

                      <span className="font-medium">
                        {formatPercentage(highPriorityShare)}
                      </span>
                    </div>

                    <div className="h-2 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-orange-500"
                        style={{
                          width: `${Math.min(highPriorityShare, 100)}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div className="rounded-xl bg-muted/50 p-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">
                        Workflow transitions
                      </span>

                      <span className="font-semibold">
                        {formatNumber(agent.total_transition_count)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section className="mt-6 rounded-2xl border bg-card p-6 shadow-sm">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className="font-semibold">Handoff intelligence</h2>

                  <p className="mt-1 text-sm text-muted-foreground">
                    How work enters and leaves this agent.
                  </p>
                </div>

                <div className="flex items-center gap-2 rounded-lg bg-muted/50 px-3 py-2 text-sm">
                  <GitBranch className="size-4 text-muted-foreground" />
                  {formatNumber(handoffTotal)} total handoffs
                </div>
              </div>

              <div className="mt-6 grid gap-4 md:grid-cols-3">
                <div className="rounded-xl border p-5">
                  <div className="flex items-center gap-3">
                    <div className="flex size-10 items-center justify-center rounded-lg bg-blue-500/10">
                      <ArrowDownLeft className="size-5 text-blue-600 dark:text-blue-400" />
                    </div>

                    <div>
                      <p className="text-sm text-muted-foreground">
                        Incoming
                      </p>

                      <p className="text-xl font-semibold">
                        {formatNumber(
                          agent.incoming_handoff_count,
                        )}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="rounded-xl border p-5">
                  <div className="flex items-center gap-3">
                    <div className="flex size-10 items-center justify-center rounded-lg bg-orange-500/10">
                      <ArrowUpRight className="size-5 text-orange-600 dark:text-orange-400" />
                    </div>

                    <div>
                      <p className="text-sm text-muted-foreground">
                        Outgoing
                      </p>

                      <p className="text-xl font-semibold">
                        {formatNumber(
                          agent.outgoing_handoff_count,
                        )}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="rounded-xl border p-5">
                  <div className="flex items-center gap-3">
                    <div className="flex size-10 items-center justify-center rounded-lg bg-muted">
                      <Users className="size-5 text-muted-foreground" />
                    </div>

                    <div>
                      <p className="text-sm text-muted-foreground">
                        Net flow
                      </p>

                      <p className="text-xl font-semibold">
                        {handoffBalance > 0 ? "+" : ""}
                        {formatNumber(handoffBalance)}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section className="mt-6 grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl border bg-card p-6 shadow-sm">
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="font-semibold">
                      Operational coverage
                    </h2>

                    <p className="mt-1 text-sm text-muted-foreground">
                      Breadth of projects and issue types handled.
                    </p>
                  </div>

                  <FolderKanban className="size-5 text-muted-foreground" />
                </div>

                <div className="mt-6 space-y-5">
                  <CoverageBar
                    label="Unique projects"
                    value={agent.unique_project_count}
                    max={maxCoverage}
                  />

                  <CoverageBar
                    label="Unique issue types"
                    value={agent.unique_issue_type_count}
                    max={maxCoverage}
                  />
                </div>
              </div>

              <div className="rounded-2xl border bg-card p-6 shadow-sm">
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="font-semibold">
                      Agent profile
                    </h2>

                    <p className="mt-1 text-sm text-muted-foreground">
                      Canonical identity and source information.
                    </p>
                  </div>

                  <Users className="size-5 text-muted-foreground" />
                </div>

                <div className="mt-5">
                  <MetricRow
                    label="Agent ID"
                    value={agent.agent_id}
                  />

                  <MetricRow
                    label="Name"
                    value={agent.name || "Not provided"}
                  />

                  <MetricRow
                    label="Team"
                    value={agent.team || "Not provided"}
                  />

                  <MetricRow
                    label="Status"
                    value={agent.active ? "Active" : "Inactive"}
                  />

                  <MetricRow
                    label="Directory match"
                    value={
                      selectedAgentSummary
                        ? "Organization analytics"
                        : "Direct intelligence"
                    }
                  />
                </div>
              </div>
            </section>

            <div className="mt-8 flex flex-col gap-2 border-t pt-5 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
              <span>
                Agent intelligence is sourced from the SupportMind backend.
              </span>

              <span className="font-mono">
                GET /api/v1/agents/{agent.agent_id}/intelligence
              </span>
            </div>
          </>
        ) : (
          <div className="mt-6 rounded-2xl border bg-card p-12 text-center">
            <Bot className="mx-auto size-8 text-muted-foreground" />

            <h2 className="mt-4 font-semibold">
              No agent selected
            </h2>

            <p className="mt-1 text-sm text-muted-foreground">
              Select an agent above to inspect its intelligence.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}