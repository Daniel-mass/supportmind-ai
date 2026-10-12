"use client";

import {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo, useState, type ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock3,
  GitBranch,
  History,
  RefreshCw,
  Search,
  Users,
  Workflow,
  XCircle,
} from "lucide-react";

import { useWorkflowIntelligence } from "@/lib/queries/workflow";

const DEFAULT_TICKET_ID = "C10grx1nv:214";

type WorkflowNodeData = {
  label: ReactNode;
};

type TransitionCount = {
  from: string;
  to: string;
  count: number;
};

type WorkflowNode = Node<WorkflowNodeData>;

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0m";
  }

  const totalMinutes = Math.floor(seconds / 60);
  const days = Math.floor(totalMinutes / 1440);
  const hours = Math.floor((totalMinutes % 1440) / 60);
  const minutes = totalMinutes % 60;

  if (days > 0) {
    return `${days}d ${hours}h`;
  }

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  return `${Math.max(minutes, 1)}m`;
}

function formatDate(timestamp: string): string {
  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }

  return date.toLocaleString();
}

function formatStatus(status: string | null): string {
  if (!status) {
    return "Unknown";
  }

  return status
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character: string) => character.toUpperCase());
}

function statusClass(status: string | null): string {
  switch (status) {
    case "waiting":
      return "border-amber-500/20 bg-amber-500/10 text-amber-300";

    case "in_progress":
      return "border-blue-500/20 bg-blue-500/10 text-blue-300";

    case "resolved":
    case "done":
    case "closed":
      return "border-emerald-500/20 bg-emerald-500/10 text-emerald-300";

    case "reopened":
      return "border-rose-500/20 bg-rose-500/10 text-rose-300";

    default:
      return "border-white/10 bg-white/[0.04] text-slate-300";
  }
}

function getStatusIcon(status: string | null): ReactNode {
  switch (status) {
    case "waiting":
      return <Clock3 className="h-4 w-4" />;

    case "in_progress":
      return <Activity className="h-4 w-4" />;

    case "resolved":
    case "done":
    case "closed":
      return <CheckCircle2 className="h-4 w-4" />;

    case "reopened":
      return <RefreshCw className="h-4 w-4" />;

    default:
      return <Workflow className="h-4 w-4" />;
  }
}

function MetricCard({
  label,
  value,
  icon,
  description,
}: {
  label: string;
  value: string;
  icon: ReactNode;
  description?: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] text-slate-300">
          {icon}
        </div>
      </div>

      <p className="text-xs font-medium uppercase tracking-[0.12em] text-slate-500">
        {label}
      </p>

      <p className="mt-2 text-2xl font-semibold tracking-tight text-white">
        {value}
      </p>

      {description ? (
        <p className="mt-1 text-xs text-slate-500">{description}</p>
      ) : null}
    </div>
  );
}

export default function AgentWorkflowPage() {
  const [ticketId, setTicketId] = useState<string>(DEFAULT_TICKET_ID);
  const [selectedTicketId, setSelectedTicketId] =
    useState<string>(DEFAULT_TICKET_ID);

  const {
    data,
    isLoading,
    isError,
    error,
    refetch,
  } = useWorkflowIntelligence(selectedTicketId);

  const topology = useMemo<{
    nodes: WorkflowNode[];
    edges: Edge[];
    transitionCounts: TransitionCount[];
  }>(() => {
    if (!data || data.events.length === 0) {
      return {
        nodes: [],
        edges: [],
        transitionCounts: [],
      };
    }

    const statusCounts = new Map<string, number>();

    const transitionMap = new Map<string, TransitionCount>();

    data.events.forEach((event) => {
      const status: string = event.status ?? "unknown";

      const currentCount: number = statusCounts.get(status) ?? 0;

      statusCounts.set(status, currentCount + 1);
    });

    for (let index = 1; index < data.events.length; index += 1) {
      const previousEvent = data.events[index - 1];
      const currentEvent = data.events[index];

      const from: string = previousEvent.status ?? "unknown";
      const to: string = currentEvent.status ?? "unknown";

      if (from === to) {
        continue;
      }

      const key: string = `${from}::${to}`;
      const existing: TransitionCount | undefined = transitionMap.get(key);

      if (existing) {
        transitionMap.set(key, {
          ...existing,
          count: existing.count + 1,
        });
      } else {
        transitionMap.set(key, {
          from,
          to,
          count: 1,
        });
      }
    }

    const statuses: string[] = Array.from(statusCounts.keys());

    const nodes: WorkflowNode[] = statuses.map(
      (status: string, index: number): WorkflowNode => {
        const column: number = Math.floor(index / 2);
        const row: number = index % 2;

        const eventCount: number = statusCounts.get(status) ?? 0;
        const duration: number =
          data.metrics.status_durations[status] ?? 0;

        return {
          id: `status-${status}`,
          position: {
            x: column * 320,
            y: row * 190,
          },
          data: {
            label: (
              <div className="min-w-[190px]">
                <div className="flex items-center gap-2">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/[0.08] text-slate-200">
                    {getStatusIcon(status)}
                  </div>

                  <div>
                    <p className="font-semibold text-white">
                      {formatStatus(status)}
                    </p>

                    <p className="text-[11px] text-slate-500">
                      {eventCount} events
                    </p>
                  </div>
                </div>

                <div className="mt-4 border-t border-white/10 pt-3">
                  <p className="text-[10px] uppercase tracking-[0.12em] text-slate-500">
                    Total duration
                  </p>

                  <p className="mt-1 text-sm font-medium text-slate-200">
                    {formatDuration(duration)}
                  </p>
                </div>
              </div>
            ),
          },
          style: {
            width: 245,
            borderRadius: 16,
            border: "1px solid rgba(255,255,255,0.10)",
            background: "rgba(15,23,42,0.96)",
            color: "white",
            padding: 16,
            boxShadow: "0 12px 30px rgba(0,0,0,0.20)",
          },
        };
      },
    );

    const transitionCounts: TransitionCount[] = Array.from(
      transitionMap.values(),
    ).sort(
      (first: TransitionCount, second: TransitionCount): number =>
        second.count - first.count,
    );

    const edges: Edge[] = transitionCounts.map(
      (transition: TransitionCount, index: number): Edge => {
        const isFrequent: boolean = transition.count >= 10;

        return {
          id: `transition-${index}-${transition.from}-${transition.to}`,
          source: `status-${transition.from}`,
          target: `status-${transition.to}`,
          type: "smoothstep",
          animated: isFrequent,
          label: `${transition.count}×`,
          markerEnd: {
            type: MarkerType.ArrowClosed,
            width: 18,
            height: 18,
          },
          style: {
            stroke: isFrequent
              ? "rgba(96,165,250,0.85)"
              : "rgba(148,163,184,0.45)",
            strokeWidth: isFrequent ? 2.5 : 1.5,
          },
          labelStyle: {
            fill: "#cbd5e1",
            fontSize: 11,
            fontWeight: 600,
          },
          labelBgStyle: {
            fill: "#0f172a",
            fillOpacity: 0.95,
          },
          labelBgPadding: [5, 3],
          labelBgBorderRadius: 6,
        };
      },
    );

    return {
      nodes,
      edges,
      transitionCounts,
    };
  }, [data]);

  const handleAnalyze = (): void => {
    const normalizedTicketId: string = ticketId.trim();

    if (!normalizedTicketId) {
      return;
    }

    setSelectedTicketId(normalizedTicketId);
  };

  const workflowPath: string[] = data?.variants?.[0]?.path ?? [];

  return (
    <main className="min-h-full bg-slate-950 text-white">
      <div className="mx-auto max-w-[1700px] space-y-6 p-6 lg:p-8">
        <section>
          <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.14em] text-slate-500">
                <GitBranch className="h-4 w-4" />
                Agent Operations
              </div>

              <h1 className="text-3xl font-semibold tracking-tight">
                Agent Workflow
              </h1>

              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
                Trace the actual lifecycle of a support ticket, including
                status transitions, repeated workflow cycles, bottlenecks,
                and handoff activity.
              </p>
            </div>

            <div className="flex flex-col gap-2 sm:flex-row">
              <div className="relative">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />

                <input
                  value={ticketId}
                  onChange={(event): void =>
                    setTicketId(event.target.value)
                  }
                  onKeyDown={(event): void => {
                    if (event.key === "Enter") {
                      handleAnalyze();
                    }
                  }}
                  placeholder="project:issue"
                  className="h-11 w-full rounded-xl border border-white/10 bg-white/[0.04] pl-10 pr-4 text-sm text-white outline-none placeholder:text-slate-600 focus:border-blue-500/50 sm:w-72"
                />
              </div>

              <button
                type="button"
                onClick={handleAnalyze}
                disabled={isLoading || !ticketId.trim()}
                className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-white px-5 text-sm font-semibold text-slate-950 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Analyzing
                  </>
                ) : (
                  <>
                    <Workflow className="h-4 w-4" />
                    Analyze
                  </>
                )}
              </button>
            </div>
          </div>
        </section>

        {isLoading ? (
          <section className="rounded-2xl border border-white/10 bg-white/[0.035] p-8">
            <div className="flex items-center gap-4">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-500/10 text-blue-300">
                <RefreshCw className="h-5 w-5 animate-spin" />
              </div>

              <div>
                <p className="font-medium text-white">
                  Loading workflow intelligence
                </p>

                <p className="mt-1 text-sm text-slate-500">
                  Reading the ticket&apos;s actual change-history events.
                </p>
              </div>
            </div>
          </section>
        ) : null}

        {isError ? (
          <section className="rounded-2xl border border-rose-500/20 bg-rose-500/[0.06] p-6">
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-rose-500/10 text-rose-300">
                <XCircle className="h-5 w-5" />
              </div>

              <div className="min-w-0 flex-1">
                <p className="font-semibold text-rose-200">
                  Workflow analysis failed
                </p>

                <p className="mt-1 break-words text-sm text-rose-300/70">
                  {error instanceof Error
                    ? error.message
                    : "Unable to load workflow intelligence."}
                </p>

                <button
                  type="button"
                  onClick={(): void => {
                    void refetch();
                  }}
                  className="mt-4 inline-flex items-center gap-2 rounded-lg border border-rose-400/20 bg-rose-400/10 px-3 py-2 text-xs font-medium text-rose-200 hover:bg-rose-400/15"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  Retry
                </button>
              </div>
            </div>
          </section>
        ) : null}

        {!isLoading && !isError && data ? (
          <>
            <section className="rounded-2xl border border-white/10 bg-white/[0.035] p-5">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <p className="text-xs font-medium uppercase tracking-[0.12em] text-slate-500">
                    Analyzed ticket
                  </p>

                  <div className="mt-2 flex flex-wrap items-center gap-3">
                    <code className="rounded-lg border border-white/10 bg-black/20 px-3 py-1.5 text-sm font-medium text-slate-200">
                      {data.ticket_id}
                    </code>

                    {data.events[0]?.priority ? (
                      <span className="rounded-full border border-rose-500/20 bg-rose-500/10 px-3 py-1 text-xs font-medium text-rose-300">
                        {data.events[0].priority} priority
                      </span>
                    ) : null}

                    {data.events[0]?.category ? (
                      <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs font-medium text-slate-300">
                        {data.events[0].category}
                      </span>
                    ) : null}
                  </div>
                </div>

                <div className="text-left lg:text-right">
                  <p className="text-xs text-slate-500">
                    Workflow events
                  </p>

                  <p className="mt-1 text-lg font-semibold text-white">
                    {data.events.length}
                  </p>
                </div>
              </div>
            </section>

            <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                label="Lifecycle"
                value={formatDuration(data.metrics.lifecycle_seconds)}
                description="Total ticket lifecycle"
                icon={<Clock3 className="h-4 w-4" />}
              />

              <MetricCard
                label="Active time"
                value={formatDuration(data.metrics.active_seconds)}
                description="Time in active states"
                icon={<Activity className="h-4 w-4" />}
              />

              <MetricCard
                label="Transitions"
                value={data.metrics.transition_count.toLocaleString()}
                description={`${data.metrics.unique_status_count} unique statuses`}
                icon={<GitBranch className="h-4 w-4" />}
              />

              <MetricCard
                label="Handoffs"
                value={data.handoffs.length.toLocaleString()}
                description={`${data.metrics.unique_assignee_count} unique assignees`}
                icon={<Users className="h-4 w-4" />}
              />
            </section>

            <section className="overflow-hidden rounded-2xl border border-white/10 bg-white/[0.035]">
              <div className="border-b border-white/10 px-6 py-5">
                <div className="flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <Workflow className="h-4 w-4 text-blue-300" />

                      <h2 className="font-semibold text-white">
                        Workflow topology
                      </h2>
                    </div>

                    <p className="mt-1 text-sm text-slate-500">
                      Unique workflow states and the real transitions observed
                      between them.
                    </p>
                  </div>

                  <div className="text-xs text-slate-500">
                    {topology.nodes.length} states ·{" "}
                    {topology.edges.length} transition types
                  </div>
                </div>
              </div>

              <div className="h-[560px] bg-slate-950/60">
                {topology.nodes.length > 0 ? (
                  <ReactFlow
                    nodes={topology.nodes}
                    edges={topology.edges}
                    fitView
                    fitViewOptions={{
                      padding: 0.2,
                    }}
                    nodesDraggable
                    nodesConnectable={false}
                    elementsSelectable
                    proOptions={{
                      hideAttribution: true,
                    }}
                  >
                    <MiniMap
                      pannable
                      zoomable
                      nodeColor="#334155"
                      maskColor="rgba(2,6,23,0.75)"
                    />

                    <Controls />

                    <Background
                      gap={24}
                      size={1}
                      color="rgba(148,163,184,0.10)"
                    />
                  </ReactFlow>
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-slate-500">
                    No workflow states were returned.
                  </div>
                )}
              </div>
            </section>

            <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
              <div className="rounded-2xl border border-white/10 bg-white/[0.035]">
                <div className="border-b border-white/10 px-6 py-5">
                  <div className="flex items-center gap-2">
                    <ArrowRight className="h-4 w-4 text-blue-300" />

                    <h2 className="font-semibold text-white">
                      Transition summary
                    </h2>
                  </div>

                  <p className="mt-1 text-sm text-slate-500">
                    Repeated transitions are consolidated here so the topology
                    stays readable.
                  </p>
                </div>

                <div className="divide-y divide-white/5">
                  {topology.transitionCounts.length > 0 ? (
                    topology.transitionCounts.map(
                      (transition: TransitionCount) => (
                        <div
                          key={`${transition.from}-${transition.to}`}
                          className="flex items-center justify-between gap-4 px-6 py-4"
                        >
                          <div className="flex min-w-0 items-center gap-3">
                            <span
                              className={`rounded-lg border px-2.5 py-1.5 text-xs font-medium ${statusClass(
                                transition.from,
                              )}`}
                            >
                              {formatStatus(transition.from)}
                            </span>

                            <ArrowRight className="h-4 w-4 shrink-0 text-slate-600" />

                            <span
                              className={`rounded-lg border px-2.5 py-1.5 text-xs font-medium ${statusClass(
                                transition.to,
                              )}`}
                            >
                              {formatStatus(transition.to)}
                            </span>
                          </div>

                          <span className="shrink-0 rounded-full border border-white/10 bg-white/[0.04] px-2.5 py-1 text-xs font-semibold text-slate-300">
                            {transition.count}×
                          </span>
                        </div>
                      ),
                    )
                  ) : (
                    <div className="px-6 py-10 text-center text-sm text-slate-500">
                      No status transitions were recorded.
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-white/10 bg-white/[0.035]">
                <div className="border-b border-white/10 px-6 py-5">
                  <div className="flex items-center gap-2">
                    <Clock3 className="h-4 w-4 text-amber-300" />

                    <h2 className="font-semibold text-white">
                      Time by status
                    </h2>
                  </div>

                  <p className="mt-1 text-sm text-slate-500">
                    Actual lifecycle time accumulated in each state.
                  </p>
                </div>

                <div className="space-y-4 p-6">
                  {Object.entries(data.metrics.status_durations)
                    .sort(
                      (
                        first: [string, number],
                        second: [string, number],
                      ): number => second[1] - first[1],
                    )
                    .map(
                      ([status, seconds]: [string, number]) => {
                        const percentage: number =
                          data.metrics.lifecycle_seconds > 0
                            ? (seconds /
                                data.metrics.lifecycle_seconds) *
                              100
                            : 0;

                        return (
                          <div key={status}>
                            <div className="mb-2 flex items-center justify-between gap-4">
                              <span className="text-sm text-slate-300">
                                {formatStatus(status)}
                              </span>

                              <span className="text-xs font-medium text-slate-500">
                                {formatDuration(seconds)} ·{" "}
                                {percentage.toFixed(1)}%
                              </span>
                            </div>

                            <div className="h-2 overflow-hidden rounded-full bg-white/[0.06]">
                              <div
                                className="h-full rounded-full bg-slate-400"
                                style={{
                                  width: `${Math.min(
                                    100,
                                    percentage,
                                  )}%`,
                                }}
                              />
                            </div>
                          </div>
                        );
                      },
                    )}
                </div>
              </div>
            </section>

            <section className="rounded-2xl border border-white/10 bg-white/[0.035]">
              <div className="border-b border-white/10 px-6 py-5">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-300" />

                  <h2 className="font-semibold text-white">
                    Workflow bottlenecks
                  </h2>
                </div>

                <p className="mt-1 text-sm text-slate-500">
                  States consuming the most lifecycle time or recurring most
                  frequently.
                </p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full min-w-[850px] text-left text-sm">
                  <thead className="border-b border-white/5 text-xs uppercase tracking-[0.1em] text-slate-600">
                    <tr>
                      <th className="px-6 py-4 font-medium">Status</th>
                      <th className="px-6 py-4 font-medium">
                        Intervals
                      </th>
                      <th className="px-6 py-4 font-medium">
                        Average duration
                      </th>
                      <th className="px-6 py-4 font-medium">
                        Duration share
                      </th>
                      <th className="px-6 py-4 font-medium">
                        Recurrence
                      </th>
                      <th className="px-6 py-4 font-medium">Score</th>
                    </tr>
                  </thead>

                  <tbody className="divide-y divide-white/5">
                    {data.bottlenecks.map((bottleneck) => (
                      <tr key={bottleneck.status}>
                        <td className="px-6 py-4">
                          <span
                            className={`inline-flex rounded-lg border px-2.5 py-1 text-xs font-medium ${statusClass(
                              bottleneck.status,
                            )}`}
                          >
                            {formatStatus(bottleneck.status)}
                          </span>
                        </td>

                        <td className="px-6 py-4 text-slate-300">
                          {bottleneck.interval_count.toLocaleString()}
                        </td>

                        <td className="px-6 py-4 text-slate-300">
                          {formatDuration(
                            bottleneck.average_seconds,
                          )}
                        </td>

                        <td className="px-6 py-4 text-slate-300">
                          {(
                            bottleneck.duration_share * 100
                          ).toFixed(1)}
                          %
                        </td>

                        <td className="px-6 py-4 text-slate-300">
                          {(
                            bottleneck.recurrence_rate * 100
                          ).toFixed(0)}
                          %
                        </td>

                        <td className="px-6 py-4">
                          <span className="font-semibold text-amber-300">
                            {bottleneck.bottleneck_score.toFixed(3)}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="rounded-2xl border border-white/10 bg-white/[0.035]">
              <div className="border-b border-white/10 px-6 py-5">
                <div className="flex items-center gap-2">
                  <History className="h-4 w-4 text-slate-300" />

                  <h2 className="font-semibold text-white">
                    Event history
                  </h2>
                </div>

                <p className="mt-1 text-sm text-slate-500">
                  All {data.events.length} real workflow events returned by
                  the backend.
                </p>
              </div>

              <div className="max-h-[650px] overflow-y-auto">
                <div className="divide-y divide-white/5">
                  {data.events.map((event, index) => {
                    const previousEvent =
                      index > 0 ? data.events[index - 1] : null;

                    const changed: boolean =
                      previousEvent !== null &&
                      previousEvent.status !== event.status;

                    return (
                      <div
                        key={`${event.event_id}-${index}`}
                        className="grid gap-4 px-6 py-4 lg:grid-cols-[70px_180px_1fr_auto] lg:items-center"
                      >
                        <div className="text-xs font-medium text-slate-600">
                          #{index + 1}
                        </div>

                        <div className="text-xs text-slate-500">
                          {formatDate(event.timestamp)}
                        </div>

                        <div className="flex min-w-0 flex-wrap items-center gap-2">
                          <span
                            className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium ${statusClass(
                              event.status,
                            )}`}
                          >
                            {getStatusIcon(event.status)}
                            {formatStatus(event.status)}
                          </span>

                          {changed && previousEvent ? (
                            <>
                              <ArrowRight className="h-3.5 w-3.5 text-slate-600" />

                              <span className="text-xs text-slate-500">
                                from{" "}
                                {formatStatus(
                                  previousEvent.status,
                                )}
                              </span>
                            </>
                          ) : null}
                        </div>

                        <div className="text-right">
                          <code className="text-[11px] text-slate-600">
                            {event.event_id}
                          </code>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>

            <section className="rounded-2xl border border-white/10 bg-white/[0.035]">
              <div className="border-b border-white/10 px-6 py-5">
                <div className="flex items-center gap-2">
                  <GitBranch className="h-4 w-4 text-violet-300" />

                  <h2 className="font-semibold text-white">
                    Observed workflow path
                  </h2>
                </div>

                <p className="mt-1 text-sm text-slate-500">
                  The canonical status sequence returned by the workflow
                  intelligence endpoint.
                </p>
              </div>

              <div className="p-6">
                {workflowPath.length > 0 ? (
                  <div className="flex flex-wrap items-center gap-2">
                    {workflowPath.map(
                      (status: string, index: number) => (
                        <div
                          key={`${status}-${index}`}
                          className="flex items-center gap-2"
                        >
                          <span
                            className={`rounded-lg border px-3 py-2 text-xs font-medium ${statusClass(
                              status,
                            )}`}
                          >
                            {formatStatus(status)}
                          </span>

                          {index < workflowPath.length - 1 ? (
                            <ArrowRight className="h-3.5 w-3.5 shrink-0 text-slate-700" />
                          ) : null}
                        </div>
                      ),
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">
                    No workflow variant path was returned.
                  </p>
                )}
              </div>
            </section>

            <section className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-6">
                <div className="flex items-center gap-2">
                  <Users className="h-4 w-4 text-slate-300" />

                  <h2 className="font-semibold text-white">
                    Agent handoffs
                  </h2>
                </div>

                <p className="mt-1 text-sm text-slate-500">
                  Ownership changes recorded for this ticket.
                </p>

                <div className="mt-6">
                  {data.handoffs.length === 0 ? (
                    <div className="rounded-xl border border-white/5 bg-white/[0.025] p-5">
                      <p className="text-sm font-medium text-slate-300">
                        No handoff events recorded
                      </p>

                      <p className="mt-1 text-xs leading-5 text-slate-600">
                        The backend reports zero explicit handoff records for
                        this ticket.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {data.handoffs.map(
                        (handoff, index: number) => (
                          <div
                            key={index}
                            className="rounded-xl border border-white/5 bg-white/[0.025] p-4"
                          >
                            <pre className="overflow-x-auto text-xs text-slate-400">
                              {JSON.stringify(
                                handoff,
                                null,
                                2,
                              )}
                            </pre>
                          </div>
                        ),
                      )}
                    </div>
                  )}
                </div>
              </div>

              <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-6">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-slate-300" />

                  <h2 className="font-semibold text-white">
                    Workflow source
                  </h2>
                </div>

                <p className="mt-1 text-sm text-slate-500">
                  Provenance for this analysis.
                </p>

                <div className="mt-6 space-y-3">
                  <div className="rounded-xl border border-white/5 bg-white/[0.025] p-4">
                    <p className="text-[10px] uppercase tracking-[0.12em] text-slate-600">
                      Endpoint
                    </p>

                    <code className="mt-2 block break-all text-xs text-slate-400">
                      GET /api/v1/workflow/
                      {encodeURIComponent(data.ticket_id)}
                      /intelligence
                    </code>
                  </div>

                  <div className="rounded-xl border border-white/5 bg-white/[0.025] p-4">
                    <p className="text-[10px] uppercase tracking-[0.12em] text-slate-600">
                      Event source
                    </p>

                    <p className="mt-2 text-sm text-slate-300">
                      {data.events[0]?.source ??
                        "helpdesk_change_history"}
                    </p>
                  </div>

                  <div className="rounded-xl border border-white/5 bg-white/[0.025] p-4">
                    <p className="text-[10px] uppercase tracking-[0.12em] text-slate-600">
                      Source issue ID
                    </p>

                    <p className="mt-2 text-sm text-slate-300">
                      {String(
                        data.events[0]?.metadata?.source_issue_id ??
                          "—",
                      )}
                    </p>
                  </div>
                </div>
              </div>
            </section>
          </>
        ) : null}
      </div>
    </main>
  );
}