"use client";

import {
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  RefreshCw,
  Sparkles,
  TrendingUp,
} from "lucide-react";

import { useOrganizationIntelligence } from "@/lib/queries/organization";

function formatCount(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
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
  icon: typeof Sparkles;
}) {
  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-muted-foreground">{label}</p>
          <p className="mt-2 text-2xl font-semibold tracking-tight">{value}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {description}
          </p>
        </div>

        <div className="rounded-lg border bg-muted/40 p-2">
          <Icon className="h-4 w-4 text-muted-foreground" />
        </div>
      </div>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        {[1, 2, 3].map((item) => (
          <div
            key={item}
            className="h-28 animate-pulse rounded-xl border bg-muted/30"
          />
        ))}
      </div>

      <div className="h-80 animate-pulse rounded-xl border bg-muted/30" />
    </div>
  );
}

export default function EmergingIssuesPage() {
  const { data, isLoading, error, refetch, isFetching } =
    useOrganizationIntelligence();

  const emergingIssues = data?.emerging_issues ?? [];

  const topIssue = emergingIssues[0];

  return (
    <main className="space-y-8 p-6">
      {/* Header */}
      <section className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>Organization</span>
            <span>/</span>
            <span>Emerging Issues</span>
          </div>

          <div>
            <h1 className="text-3xl font-semibold tracking-tight">
              Emerging Issues
            </h1>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-muted-foreground">
              Detect issue types showing increased ticket activity in the most
              recent observation window. These signals help identify areas that
              may require operational attention.
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={() => refetch()}
          disabled={isFetching}
          className="inline-flex h-9 items-center justify-center gap-2 rounded-md border bg-background px-3 text-sm font-medium shadow-sm transition-colors hover:bg-muted disabled:pointer-events-none disabled:opacity-50"
        >
          <RefreshCw
            className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`}
          />
          {isFetching ? "Refreshing..." : "Refresh"}
        </button>
      </section>

      {/* Loading */}
      {isLoading && <LoadingState />}

      {/* Error */}
      {!isLoading && error && (
        <section className="rounded-xl border border-destructive/30 bg-destructive/5 p-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 text-destructive" />

            <div className="space-y-1">
              <h2 className="font-semibold">Unable to load emerging issues</h2>
              <p className="text-sm text-muted-foreground">
                The organization intelligence service could not be reached.
                Please try refreshing the data.
              </p>

              <button
                type="button"
                onClick={() => refetch()}
                className="mt-3 inline-flex items-center gap-2 rounded-md border bg-background px-3 py-2 text-sm font-medium hover:bg-muted"
              >
                <RefreshCw className="h-4 w-4" />
                Try again
              </button>
            </div>
          </div>
        </section>
      )}

      {/* Content */}
      {!isLoading && !error && (
        <>
          {/* Signal summary */}
          <section className="grid gap-4 md:grid-cols-3">
            <MetricCard
              label="Emerging signals"
              value={formatCount(emergingIssues.length)}
              description="Issue types currently showing increased activity"
              icon={Sparkles}
            />

            <MetricCard
              label="Top emerging issue"
              value={topIssue?.label ?? "None"}
              description={
                topIssue
                  ? `${formatCount(topIssue.count)} tickets in the recent window`
                  : "No qualifying emerging issue detected"
              }
              icon={TrendingUp}
            />

            <MetricCard
              label="Detection window"
              value="30 days"
              description="Compared with the preceding 30-day period"
              icon={BarChart3}
            />
          </section>

          {/* Detection methodology */}
          <section className="rounded-xl border bg-card p-6 shadow-sm">
            <div className="flex items-start gap-3">
              <div className="rounded-lg border bg-muted/40 p-2">
                <Sparkles className="h-4 w-4 text-muted-foreground" />
              </div>

              <div className="space-y-2">
                <h2 className="font-semibold">How signals are detected</h2>

                <p className="max-w-4xl text-sm leading-6 text-muted-foreground">
                  SupportMind compares ticket activity by issue type across two
                  equal 30-day windows. An issue becomes an emerging signal
                  when it has meaningful recent volume and more recent tickets
                  than in the preceding window.
                </p>

                <div className="flex flex-wrap gap-2 pt-1 text-xs">
                  <span className="rounded-full border bg-muted/40 px-3 py-1">
                    Recent window: 30 days
                  </span>

                  <span className="rounded-full border bg-muted/40 px-3 py-1">
                    Baseline: previous 30 days
                  </span>

                  <span className="rounded-full border bg-muted/40 px-3 py-1">
                    Minimum recent volume: 5 tickets
                  </span>
                </div>
              </div>
            </div>
          </section>

          {/* Emerging issues */}
          <section className="space-y-4">
            <div>
              <h2 className="text-xl font-semibold tracking-tight">
                Emerging issue signals
              </h2>

              <p className="mt-1 text-sm text-muted-foreground">
                Issue types with increased recent ticket activity.
              </p>
            </div>

            {emergingIssues.length === 0 ? (
              <div className="rounded-xl border border-dashed p-10 text-center">
                <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-full border bg-muted/40">
                  <TrendingUp className="h-5 w-5 text-muted-foreground" />
                </div>

                <h3 className="mt-4 font-semibold">
                  No emerging issues detected
                </h3>

                <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
                  No issue type currently meets the minimum volume and growth
                  criteria for an emerging signal.
                </p>
              </div>
            ) : (
              <div className="overflow-hidden rounded-xl border bg-card shadow-sm">
                <div className="grid grid-cols-[auto_1fr_auto] items-center gap-4 border-b bg-muted/30 px-5 py-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  <span>Rank</span>
                  <span>Issue type</span>
                  <span>Recent tickets</span>
                </div>

                <div className="divide-y">
                  {emergingIssues.map((issue, index) => (
                    <div
                      key={`${issue.label}-${index}`}
                      className="grid grid-cols-[auto_1fr_auto] items-center gap-4 px-5 py-4 transition-colors hover:bg-muted/20"
                    >
                      <div className="flex h-8 w-8 items-center justify-center rounded-full border bg-background text-sm font-semibold">
                        {index + 1}
                      </div>

                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="truncate font-medium">{issue.label}</p>

                          <ArrowUpRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                        </div>

                        <p className="mt-1 text-xs text-muted-foreground">
                          Increased activity detected
                        </p>
                      </div>

                      <div className="text-right">
                        <p className="font-semibold">
                          {formatCount(issue.count)}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          recent tickets
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>

          {/* Interpretation */}
          {emergingIssues.length > 0 && (
            <section className="rounded-xl border bg-card p-6 shadow-sm">
              <div className="flex items-start gap-3">
                <div className="rounded-lg border bg-muted/40 p-2">
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </div>

                <div>
                  <h2 className="font-semibold">Operational interpretation</h2>

                  <p className="mt-2 max-w-4xl text-sm leading-6 text-muted-foreground">
                    The signals above represent observed increases in ticket
                    activity, not predictions. They can be used as an
                    operational starting point for investigating changes in
                    customer demand, recurring issues, or support workload.
                  </p>
                </div>
              </div>
            </section>
          )}

          {/* Source */}
          <div className="flex items-center justify-between border-t pt-4 text-xs text-muted-foreground">
            <span>Source: Organization Intelligence API</span>
            <span>/api/v1/organization/intelligence</span>
          </div>
        </>
      )}
    </main>
  );
}