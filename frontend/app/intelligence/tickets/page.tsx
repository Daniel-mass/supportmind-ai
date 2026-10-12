"use client";

import { useState } from "react";
import { Activity, Clock3, GitBranch, Ticket } from "lucide-react";

import { MetricCard } from "@/components/intelligence/MetricCard";
import { StatusDurationChart } from "@/components/intelligence/StatusDurationChart";
import { useTicketIntelligence } from "@/lib/queries/intelligence";

const DEFAULT_TICKET_ID = "d1z0:186";

export default function TicketIntelligencePage() {
  const [ticketInput, setTicketInput] = useState(DEFAULT_TICKET_ID);
  const [selectedTicketId, setSelectedTicketId] =
    useState(DEFAULT_TICKET_ID);

  const { data, isLoading, isError, error } =
    useTicketIntelligence(selectedTicketId);

  function handleAnalyze() {
    const normalizedTicketId = ticketInput.trim();

    if (!normalizedTicketId) {
      return;
    }

    setSelectedTicketId(normalizedTicketId);
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      handleAnalyze();
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b bg-background/95 px-6 py-5">
        <div className="mx-auto max-w-7xl">
          <p className="text-sm font-medium text-muted-foreground">
            Intelligence
          </p>

          <div className="mt-1 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">
                Ticket Intelligence
              </h1>

              <p className="mt-1 text-sm text-muted-foreground">
                Analyze ticket lifecycle, workflow behavior, bottlenecks, and
                operational signals.
              </p>
            </div>

            <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:items-end">
              <div className="flex flex-1 flex-col gap-1.5">
                <label
                  htmlFor="ticket-id"
                  className="text-sm font-medium"
                >
                  Ticket ID
                </label>

                <input
                  id="ticket-id"
                  value={ticketInput}
                  onChange={(event) => setTicketInput(event.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="e.g. d1z0:186"
                  className="h-9 w-full rounded-md border bg-background px-3 text-sm outline-none transition focus:ring-2 focus:ring-ring sm:w-48"
                />
              </div>

              <button
                type="button"
                onClick={handleAnalyze}
                disabled={!ticketInput.trim() || isLoading}
                className="h-9 rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 disabled:pointer-events-none disabled:opacity-50"
              >
                {isLoading ? "Analyzing..." : "Analyze"}
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 p-6">
        {isLoading && (
          <div className="rounded-xl border bg-card p-8 text-sm text-muted-foreground">
            Loading intelligence for{" "}
            <span className="font-medium text-foreground">
              {selectedTicketId}
            </span>
            ...
          </div>
        )}

        {isError && (
          <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-6">
            <h2 className="font-semibold text-destructive">
              Unable to load ticket intelligence
            </h2>

            <p className="mt-2 text-sm text-muted-foreground">
              {error instanceof Error
                ? error.message
                : "An unexpected error occurred."}
            </p>

            <p className="mt-3 text-xs text-muted-foreground">
              Check that the ticket ID exists in the SupportMind dataset.
            </p>
          </div>
        )}

        {data && !isLoading && (
          <>
            {/* Ticket overview */}
            <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard
                label="Ticket"
                value={data.ticket.ticket_id}
                description={`${data.ticket.project} · Issue ${data.ticket.issue_num}`}
                icon={<Ticket className="size-4" />}
              />

              <MetricCard
                label="Lifecycle"
                value={formatDuration(data.analytics.lifecycle_seconds)}
                description="Total ticket lifecycle"
                icon={<Clock3 className="size-4" />}
              />

              <MetricCard
                label="Active Time"
                value={formatDuration(data.analytics.active_seconds)}
                description={`${formatPercentage(data.analytics.active_ratio)} of lifecycle`}
                icon={<Activity className="size-4" />}
              />

              <MetricCard
                label="Transitions"
                value={data.analytics.transition_count}
                description={`${data.analytics.handoff_count} handoffs`}
                icon={<GitBranch className="size-4" />}
              />
            </section>

            {/* Ticket metadata */}
            <section className="rounded-xl border bg-card p-6 shadow-sm">
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div>
                  <h2 className="text-base font-semibold">
                    Ticket Overview
                  </h2>

                  <p className="mt-1 text-sm text-muted-foreground">
                    Current ticket state and operational characteristics.
                  </p>
                </div>

                <div className="flex flex-wrap gap-2">
                  <StatusBadge
                    label={data.ticket.priority}
                    variant="priority"
                  />

                  <StatusBadge
                    label={data.ticket.status}
                    variant="status"
                  />

                  {data.analytics.is_high_priority && (
                    <StatusBadge
                      label="High Priority"
                      variant="danger"
                    />
                  )}

                  {data.analytics.is_waiting && (
                    <StatusBadge
                      label="Waiting"
                      variant="warning"
                    />
                  )}
                </div>
              </div>

              <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <DetailItem
                  label="Issue Type"
                  value={data.ticket.issue_type}
                />

                <DetailItem
                  label="Reporter"
                  value={data.ticket.reporter ?? "Not assigned"}
                />

                <DetailItem
                  label="Assignee"
                  value={data.ticket.assignee ?? "Not assigned"}
                />

                <DetailItem
                  label="Comments"
                  value={String(data.ticket.comments_count)}
                />
              </div>
            </section>

            {/* Status duration */}
            <StatusDurationChart
              statusDurations={data.workflow_metrics.status_durations}
            />

            {/* Lifecycle breakdown */}
            <section className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-xl border bg-card p-6 shadow-sm">
                <div>
                  <h2 className="text-base font-semibold">
                    Lifecycle Breakdown
                  </h2>

                  <p className="mt-1 text-sm text-muted-foreground">
                    Active versus waiting time across the ticket lifecycle.
                  </p>
                </div>

                <div className="mt-6 grid grid-cols-2 gap-4">
                  <BreakdownMetric
                    label="Active"
                    value={formatDuration(data.analytics.active_seconds)}
                    percentage={data.analytics.active_ratio}
                  />

                  <BreakdownMetric
                    label="Waiting"
                    value={formatDuration(data.analytics.waiting_seconds)}
                    percentage={data.analytics.waiting_ratio}
                  />
                </div>
              </div>

              <div className="rounded-xl border bg-card p-6 shadow-sm">
                <div>
                  <h2 className="text-base font-semibold">
                    Workflow Metrics
                  </h2>

                  <p className="mt-1 text-sm text-muted-foreground">
                    Structural characteristics of this ticket's workflow.
                  </p>
                </div>

                <div className="mt-6 grid grid-cols-2 gap-4">
                  <BreakdownMetric
                    label="Status Count"
                    value={String(data.workflow_metrics.unique_status_count)}
                  />

                  <BreakdownMetric
                    label="Intervals"
                    value={String(data.workflow_metrics.interval_count)}
                  />

                  <BreakdownMetric
                    label="Assignees"
                    value={String(
                      data.workflow_metrics.unique_assignee_count,
                    )}
                  />

                  <BreakdownMetric
                    label="Handoffs"
                    value={String(data.analytics.handoff_count)}
                  />
                </div>
              </div>
            </section>

            {/* Workflow timeline */}
            <section className="rounded-xl border bg-card p-6 shadow-sm">
              <div>
                <h2 className="text-base font-semibold">
                  Workflow Timeline
                </h2>

                <p className="mt-1 text-sm text-muted-foreground">
                  Chronological workflow events recorded for this ticket.
                </p>
              </div>

              <div className="mt-6 space-y-0">
                {data.workflow_events.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    No workflow events available.
                  </p>
                ) : (
                  data.workflow_events.map((event, index) => (
                    <div
                      key={event.event_id}
                      className="relative flex gap-4"
                    >
                      <div className="flex flex-col items-center">
                        <div className="mt-1 size-3 rounded-full border-2 border-primary bg-background" />

                        {index < data.workflow_events.length - 1 && (
                          <div className="w-px flex-1 bg-border" />
                        )}
                      </div>

                      <div className="pb-6">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-medium">
                            {event.status ?? event.activity}
                          </span>

                          <span className="text-xs text-muted-foreground">
                            {formatTimestamp(event.timestamp)}
                          </span>
                        </div>

                        <p className="mt-1 text-sm text-muted-foreground">
                          {event.activity}
                          {event.source
                            ? ` · ${event.source}`
                            : ""}
                        </p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

type StatusBadgeProps = {
  label: string;
  variant: "priority" | "status" | "danger" | "warning";
};

function StatusBadge({ label, variant }: StatusBadgeProps) {
  const classes = {
    priority: "border-border bg-muted text-foreground",
    status: "border-border bg-muted text-foreground",
    danger: "border-destructive/30 bg-destructive/10 text-destructive",
    warning: "border-yellow-500/30 bg-yellow-500/10 text-yellow-700",
  };

  return (
    <span
      className={`rounded-full border px-2.5 py-1 text-xs font-medium ${classes[variant]}`}
    >
      {label}
    </span>
  );
}

type DetailItemProps = {
  label: string;
  value: string;
};

function DetailItem({ label, value }: DetailItemProps) {
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </p>

      <p className="mt-1 text-sm font-medium">{value}</p>
    </div>
  );
}

type BreakdownMetricProps = {
  label: string;
  value: string;
  percentage?: number;
};

function BreakdownMetric({
  label,
  value,
  percentage,
}: BreakdownMetricProps) {
  return (
    <div className="rounded-lg border bg-muted/20 p-4">
      <p className="text-sm text-muted-foreground">{label}</p>

      <p className="mt-2 text-lg font-semibold">{value}</p>

      {percentage !== undefined && (
        <div className="mt-3">
          <div className="h-1.5 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary"
              style={{
                width: `${Math.min(Math.max(percentage * 100, 0), 100)}%`,
              }}
            />
          </div>

          <p className="mt-1 text-xs text-muted-foreground">
            {formatPercentage(percentage)}
          </p>
        </div>
      )}
    </div>
  );
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

  return remainingMinutes > 0
    ? `${hours}h ${remainingMinutes}m`
    : `${hours}h`;
}

function formatPercentage(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatTimestamp(timestamp: string) {
  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }

  return date.toLocaleString();
}