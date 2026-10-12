
"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CalendarDays,
  Clock3,
  RefreshCw,
  Ticket,
  TrendingUp,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type {
  TrendCategory,
} from "@/lib/api/trends";
import { useOrganizationTrends } from "@/lib/queries/organization-trends";

type RangeOption = "12" | "24" | "all";

const CHART_COLORS = [
  "#6366f1",
  "#0ea5e9",
  "#10b981",
  "#f59e0b",
  "#ec4899",
  "#8b5cf6",
  "#ef4444",
  "#14b8a6",
  "#f97316",
  "#64748b",
];

const PRIORITY_ORDER = [
  "Blocker",
  "Highest",
  "High",
  "Medium",
  "Low",
  "Lowest",
  "unknown",
];

function formatCount(value: number) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(value);
}

function formatMonth(value: string) {
  const date = new Date(`${value}-01T12:00:00`);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleDateString("en-US", {
    month: "short",
    year: "2-digit",
  });
}

function titleCase(value: string) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
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
  icon: LucideIcon;
}) {
  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-sm font-medium text-muted-foreground">
            {label}
          </p>
          <p className="mt-2 break-words text-2xl font-semibold tracking-tight">
            {value}
          </p>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">
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

function ChartCard({
  title,
  description,
  children,
  className = "",
}: {
  title: string;
  description: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`min-w-0 rounded-xl border bg-card p-5 shadow-sm sm:p-6 ${className}`}
    >
      <div className="mb-5 space-y-1">
        <h2 className="text-base font-semibold tracking-tight">
          {title}
        </h2>
        <p className="max-w-3xl text-sm leading-5 text-muted-foreground">
          {description}
        </p>
      </div>

      {children}
    </section>
  );
}

function EmptyChart({
  message,
}: {
  message: string;
}) {
  return (
    <div className="flex h-[280px] items-center justify-center rounded-lg border border-dashed bg-muted/10 px-6 text-center">
      <p className="max-w-sm text-sm leading-6 text-muted-foreground">
        {message}
      </p>
    </div>
  );
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{
    name?: string;
    value?: number | string;
    color?: string;
    dataKey?: string;
  }>;
  label?: string;
}) {
  if (!active || !payload?.length) {
    return null;
  }

  return (
    <div className="max-w-[280px] rounded-lg border bg-popover p-3 shadow-lg">
      <p className="mb-2 text-xs font-medium text-muted-foreground">
        {label ? formatMonth(label) : ""}
      </p>

      <div className="space-y-2">
        {payload
          .filter((item) => item.value !== undefined)
          .map((item, index) => (
            <div
              key={`${item.dataKey ?? item.name}-${index}`}
              className="flex items-center justify-between gap-5 text-sm"
            >
              <span className="flex min-w-0 items-center gap-2">
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{
                    backgroundColor: item.color ?? "#64748b",
                  }}
                />
                <span className="truncate">
                  {titleCase(item.name ?? item.dataKey ?? "Value")}
                </span>
              </span>

              <span className="font-semibold tabular-nums">
                {formatCount(Number(item.value ?? 0))}
              </span>
            </div>
          ))}
      </div>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[1, 2, 3, 4].map((item) => (
          <div
            key={item}
            className="h-28 animate-pulse rounded-xl border bg-muted/30"
          />
        ))}
      </div>

      <div className="h-[390px] animate-pulse rounded-xl border bg-muted/30" />

      <div className="grid gap-6 xl:grid-cols-2">
        <div className="h-[380px] animate-pulse rounded-xl border bg-muted/30" />
        <div className="h-[380px] animate-pulse rounded-xl border bg-muted/30" />
      </div>
    </div>
  );
}

function buildCategoryChartData(
  months: string[],
  trends: TrendCategory[],
  categories: string[],
) {
  const counts = new Map<string, Map<string, number>>();

  for (const item of trends) {
    if (!counts.has(item.month)) {
      counts.set(item.month, new Map());
    }

    const monthCounts = counts.get(item.month)!;
    monthCounts.set(
      item.category,
      (monthCounts.get(item.category) ?? 0) + item.count,
    );
  }

  return months.map((month) => {
    const row: Record<string, string | number> = { month };
    const monthCounts = counts.get(month);

    for (const category of categories) {
      row[category] = monthCounts?.get(category) ?? 0;
    }

    return row;
  });
}

export default function TrendsPage() {
  const {
    data,
    isLoading,
    error,
    refetch,
    isFetching,
  } = useOrganizationTrends();

  const [range, setRange] = useState<RangeOption>("all");
  const [selectedIssue, setSelectedIssue] = useState("all");

  const allMonths = data?.months ?? [];

  const months = useMemo(() => {
    if (range === "all") {
      return allMonths;
    }

    const count = Number(range);
    return allMonths.slice(-count);
  }, [allMonths, range]);

  const monthSet = useMemo(
    () => new Set(months),
    [months],
  );

  const ticketVolume = useMemo(
    () =>
      (data?.ticket_volume ?? []).filter((item) =>
        monthSet.has(item.month),
      ),
    [data?.ticket_volume, monthSet],
  );

  const filteredIssues = useMemo(
    () =>
      (data?.issue_type_trends ?? []).filter((item) =>
        monthSet.has(item.month),
      ),
    [data?.issue_type_trends, monthSet],
  );

  const filteredPriorities = useMemo(
    () =>
      (data?.priority_trends ?? []).filter((item) =>
        monthSet.has(item.month),
      ),
    [data?.priority_trends, monthSet],
  );

  const filteredStatuses = useMemo(
    () =>
      (data?.status_trends ?? []).filter((item) =>
        monthSet.has(item.month),
      ),
    [data?.status_trends, monthSet],
  );

  const issueTotals = useMemo(() => {
    const totals = new Map<string, number>();

    for (const item of filteredIssues) {
      totals.set(
        item.category,
        (totals.get(item.category) ?? 0) + item.count,
      );
    }

    return [...totals.entries()].sort(
      (a, b) => b[1] - a[1],
    );
  }, [filteredIssues]);

  const topIssueTypes = useMemo(
    () => issueTotals.slice(0, 8).map(([category]) => category),
    [issueTotals],
  );

  const activeIssue =
    selectedIssue === "all" ||
    !topIssueTypes.includes(selectedIssue)
      ? topIssueTypes[0]
      : selectedIssue;

  const visibleIssueTypes =
    selectedIssue === "all"
      ? topIssueTypes.slice(0, 5)
      : activeIssue
        ? [activeIssue]
        : [];

  const issueChartData = useMemo(
    () =>
      buildCategoryChartData(
        months,
        filteredIssues,
        visibleIssueTypes,
      ),
    [months, filteredIssues, visibleIssueTypes],
  );

  const priorityCategories = useMemo(() => {
    const available = new Set(
      filteredPriorities.map((item) => item.category),
    );

    return PRIORITY_ORDER.filter((priority) =>
      available.has(priority),
    );
  }, [filteredPriorities]);

  const priorityChartData = useMemo(
    () =>
      buildCategoryChartData(
        months,
        filteredPriorities,
        priorityCategories,
      ),
    [months, filteredPriorities, priorityCategories],
  );

  const statusTotals = useMemo(() => {
    const totals = new Map<string, number>();

    for (const item of filteredStatuses) {
      totals.set(
        item.category,
        (totals.get(item.category) ?? 0) + item.count,
      );
    }

    return [...totals.entries()].sort(
      (a, b) => b[1] - a[1],
    );
  }, [filteredStatuses]);

  const topStatuses = useMemo(
    () => statusTotals.slice(0, 6).map(([status]) => status),
    [statusTotals],
  );

  const statusChartData = useMemo(
    () =>
      buildCategoryChartData(
        months,
        filteredStatuses,
        topStatuses,
      ),
    [months, filteredStatuses, topStatuses],
  );

  const totalInRange = ticketVolume.reduce(
    (sum, item) => sum + item.count,
    0,
  );

  const busiestMonth = [...ticketVolume].sort(
    (a, b) => b.count - a.count,
  )[0];

  const averagePerMonth =
    months.length > 0
      ? Math.round(totalInRange / months.length)
      : 0;

  const rangeLabel =
    range === "all"
      ? "All available history"
      : `Latest ${range} months in dataset`;

  if (isLoading) {
    return (
      <main className="space-y-8 p-6">
        <section className="space-y-2">
          <p className="text-sm text-muted-foreground">
            Organization / Trends
          </p>
          <h1 className="text-3xl font-semibold tracking-tight">
            Trends
          </h1>
          <p className="text-sm text-muted-foreground">
            Exploring historical ticket and workflow patterns.
          </p>
        </section>
        <LoadingState />
      </main>
    );
  }

  if (error || !data) {
    return (
      <main className="space-y-6 p-6">
        <section className="space-y-2">
          <p className="text-sm text-muted-foreground">
            Organization / Trends
          </p>
          <h1 className="text-3xl font-semibold tracking-tight">
            Trends
          </h1>
        </section>

        <section className="rounded-xl border border-destructive/30 bg-destructive/5 p-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 text-destructive" />

            <div className="space-y-2">
              <h2 className="font-semibold">
                Unable to load organization trends
              </h2>
              <p className="text-sm leading-6 text-muted-foreground">
                The Trends API could not provide the historical data.
                Check that the backend is running and try again.
              </p>

              <button
                type="button"
                onClick={() => refetch()}
                className="inline-flex items-center gap-2 rounded-md border bg-background px-3 py-2 text-sm font-medium hover:bg-muted"
              >
                <RefreshCw className="h-4 w-4" />
                Try again
              </button>
            </div>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="space-y-8 p-6">
      {/* Page header */}
      <section className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>Organization</span>
            <span>/</span>
            <span>Trends</span>
          </div>

          <div>
            <h1 className="text-3xl font-semibold tracking-tight">
              Trends & Historical Patterns
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-muted-foreground">
              Explore how ticket demand, issue categories, priority
              levels, and workflow activity have changed over time.
              Use these patterns to investigate workload changes and
              operational pressure.
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={() => refetch()}
          disabled={isFetching}
          className="inline-flex h-9 shrink-0 items-center justify-center gap-2 rounded-md border bg-background px-3 text-sm font-medium shadow-sm transition-colors hover:bg-muted disabled:pointer-events-none disabled:opacity-50"
        >
          <RefreshCw
            className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`}
          />
          {isFetching ? "Refreshing..." : "Refresh"}
        </button>
      </section>

      {/* Range controls */}
      <section className="flex flex-col gap-3 rounded-xl border bg-card p-4 shadow-sm sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="rounded-lg border bg-muted/40 p-2">
            <CalendarDays className="h-4 w-4 text-muted-foreground" />
          </div>
          <div>
            <p className="text-sm font-medium">Analysis window</p>
            <p className="text-xs text-muted-foreground">
              {rangeLabel}
            </p>
          </div>
        </div>

        <select
          value={range}
          onChange={(event) =>
            setRange(event.target.value as RangeOption)
          }
          aria-label="Trend analysis time range"
          className="h-9 rounded-md border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <option value="12">Latest 12 months</option>
          <option value="24">Latest 24 months</option>
          <option value="all">All available history</option>
        </select>
      </section>

      {/* Summary metrics */}
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Tickets in dataset"
          value={formatCount(data.total_tickets)}
          description="Canonical tickets available to the analysis"
          icon={Ticket}
        />

        <MetricCard
          label="Tickets in selected window"
          value={formatCount(totalInRange)}
          description={`Across ${formatCount(months.length)} calendar months`}
          icon={BarChart3}
        />

        <MetricCard
          label="Monthly average"
          value={formatCount(averagePerMonth)}
          description="Average tickets created per displayed month"
          icon={TrendingUp}
        />

        <MetricCard
          label="Workflow events"
          value={formatCount(data.total_workflow_events)}
          description="Timestamped events across the full dataset"
          icon={Activity}
        />
      </section>

      {/* Ticket volume */}
      <ChartCard
        title="Ticket creation over time"
        description="Each point represents tickets created during that calendar month. Peaks show higher incoming volume; dips show lower recorded ticket creation. This chart describes demand, not resolution performance."
      >
        {ticketVolume.length === 0 ? (
          <EmptyChart message="No ticket creation data is available for this window." />
        ) : (
          <>
            <div className="h-[320px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={ticketVolume}
                  margin={{ top: 8, right: 12, bottom: 4, left: 0 }}
                >
                  <defs>
                    <linearGradient
                      id="ticketVolumeFill"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop
                        offset="0%"
                        stopColor="#6366f1"
                        stopOpacity={0.28}
                      />
                      <stop
                        offset="95%"
                        stopColor="#6366f1"
                        stopOpacity={0.02}
                      />
                    </linearGradient>
                  </defs>

                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                    className="stroke-border"
                  />
                  <XAxis
                    dataKey="month"
                    tickFormatter={formatMonth}
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    minTickGap={24}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tickFormatter={(value) => formatCount(Number(value))}
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={58}
                  />
                  <Tooltip
                    content={<ChartTooltip />}
                    labelFormatter={(label) => String(label)}
                  />
                  <Area
                    type="monotone"
                    dataKey="count"
                    name="Tickets created"
                    stroke="#6366f1"
                    strokeWidth={2.5}
                    fill="url(#ticketVolumeFill)"
                    activeDot={{ r: 5 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {busiestMonth && (
              <div className="mt-4 flex flex-col gap-2 rounded-lg border bg-muted/20 p-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-start gap-3">
                  <TrendingUp className="mt-0.5 h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">
                      Highest-volume month in this window
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatMonth(busiestMonth.month)} recorded the most
                      ticket creation.
                    </p>
                  </div>
                </div>

                <p className="text-lg font-semibold tabular-nums">
                  {formatCount(busiestMonth.count)} tickets
                </p>
              </div>
            )}
          </>
        )}
      </ChartCard>

      {/* Issue and priority trends */}
      <section className="grid min-w-0 gap-6 xl:grid-cols-2">
        <ChartCard
          title="Issue-type trends"
          description="Compare monthly ticket creation across the most frequent issue types in the selected window. Select one category to isolate its trend, or compare the leading five."
        >
          {topIssueTypes.length === 0 ? (
            <EmptyChart message="No issue-type trend data is available for this window." />
          ) : (
            <>
              <div className="mb-4">
                <label
                  htmlFor="issue-type-filter"
                  className="mb-1.5 block text-xs font-medium text-muted-foreground"
                >
                  Category comparison
                </label>

                <select
                  id="issue-type-filter"
                  value={selectedIssue}
                  onChange={(event) => setSelectedIssue(event.target.value)}
                  className="h-9 w-full rounded-md border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <option value="all">Compare leading 5 issue types</option>
                  {topIssueTypes.map((issue) => (
                    <option key={issue} value={issue}>
                      {titleCase(issue)}
                    </option>
                  ))}
                </select>
              </div>

              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={issueChartData}
                    margin={{ top: 8, right: 12, bottom: 4, left: 0 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      className="stroke-border"
                    />
                    <XAxis
                      dataKey="month"
                      tickFormatter={formatMonth}
                      tick={{ fontSize: 10 }}
                      tickLine={false}
                      axisLine={false}
                      minTickGap={20}
                      interval="preserveStartEnd"
                    />
                    <YAxis
                      tickFormatter={(value) => formatCount(Number(value))}
                      tick={{ fontSize: 10 }}
                      tickLine={false}
                      axisLine={false}
                      width={52}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend
                      formatter={(value) => titleCase(String(value))}
                      wrapperStyle={{ fontSize: 11, paddingTop: 12 }}
                    />

                    {visibleIssueTypes.map((issue, index) => (
                      <Line
                        key={issue}
                        type="monotone"
                        dataKey={issue}
                        name={issue}
                        stroke={CHART_COLORS[index % CHART_COLORS.length]}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <p className="mt-3 text-xs leading-5 text-muted-foreground">
                The leading categories are ranked by total ticket count in
                the selected window. Smaller categories are not shown in
                the comparison chart.
              </p>
            </>
          )}
        </ChartCard>

        <ChartCard
          title="Priority trends"
          description="Follow monthly ticket counts by priority. A sustained increase in higher-priority categories can signal growing operational pressure, but should be investigated alongside workload and resolution metrics."
        >
          {priorityCategories.length === 0 ? (
            <EmptyChart message="No priority trend data is available for this window." />
          ) : (
            <>
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={priorityChartData}
                    margin={{ top: 8, right: 12, bottom: 4, left: 0 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      className="stroke-border"
                    />
                    <XAxis
                      dataKey="month"
                      tickFormatter={formatMonth}
                      tick={{ fontSize: 10 }}
                      tickLine={false}
                      axisLine={false}
                      minTickGap={20}
                      interval="preserveStartEnd"
                    />
                    <YAxis
                      tickFormatter={(value) => formatCount(Number(value))}
                      tick={{ fontSize: 10 }}
                      tickLine={false}
                      axisLine={false}
                      width={52}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    <Legend
                      formatter={(value) => titleCase(String(value))}
                      wrapperStyle={{ fontSize: 11, paddingTop: 12 }}
                    />

                    {priorityCategories.map((priority, index) => (
                      <Line
                        key={priority}
                        type="monotone"
                        dataKey={priority}
                        name={priority}
                        stroke={CHART_COLORS[index % CHART_COLORS.length]}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="mt-3 rounded-lg border bg-muted/20 p-3">
                <p className="text-xs leading-5 text-muted-foreground">
                  Priority values reflect the canonical ticket priority
                  assigned to each ticket. The unknown category is kept
                  separate instead of being interpreted as low priority.
                </p>
              </div>
            </>
          )}
        </ChartCard>
      </section>

      {/* Workflow status activity */}
      <ChartCard
        title="Workflow status activity"
        description="Monthly counts of historical workflow events that carry a status. These are status-bearing events, not the number of unique tickets currently in each status. A ticket can contribute multiple events over its lifecycle."
      >
        {topStatuses.length === 0 ? (
          <EmptyChart message="No timestamped workflow events with status values are available for this window." />
        ) : (
          <>
            <div className="h-[320px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={statusChartData}
                  margin={{ top: 8, right: 12, bottom: 4, left: 0 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                    className="stroke-border"
                  />
                  <XAxis
                    dataKey="month"
                    tickFormatter={formatMonth}
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    minTickGap={24}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tickFormatter={(value) => formatCount(Number(value))}
                    tick={{ fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={58}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend
                    formatter={(value) => titleCase(String(value))}
                    wrapperStyle={{ fontSize: 11, paddingTop: 12 }}
                  />

                  {topStatuses.map((status, index) => (
                    <Line
                      key={status}
                      type="monotone"
                      dataKey={status}
                      name={status}
                      stroke={CHART_COLORS[index % CHART_COLORS.length]}
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 flex items-start gap-3 rounded-lg border bg-muted/20 p-4">
              <Clock3 className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
              <p className="text-xs leading-5 text-muted-foreground">
                Status activity helps reveal when workflow transitions were
                recorded. It does not measure time spent in a status; use
                Workflow Intelligence and Operational Insights for
                lifecycle and waiting-time analysis.
              </p>
            </div>
          </>
        )}
      </ChartCard>

      {/* Methodology and source */}
      <section className="rounded-xl border bg-card p-5 shadow-sm sm:p-6">
        <div className="flex items-start gap-3">
          <div className="rounded-lg border bg-muted/40 p-2">
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </div>

          <div className="min-w-0 space-y-2">
            <h2 className="font-semibold">
              How to interpret these trends
            </h2>

            <div className="grid gap-4 text-sm leading-6 text-muted-foreground md:grid-cols-3">
              <div>
                <p className="font-medium text-foreground">
                  Ticket volume
                </p>
                <p>
                  Counts tickets by their creation month. It indicates
                  recorded incoming demand, not the number of tickets
                  resolved.
                </p>
              </div>

              <div>
                <p className="font-medium text-foreground">
                  Category and priority
                </p>
                <p>
                  Group tickets by their recorded issue type and priority.
                  These charts describe the tickets created in each month,
                  not historical reclassification events.
                </p>
              </div>

              <div>
                <p className="font-medium text-foreground">
                  Workflow activity
                </p>
                <p>
                  Counts historical status-bearing events by event timestamp.
                  Multiple events may belong to the same ticket.
                </p>
              </div>
            </div>

            <div className="flex flex-col gap-1 border-t pt-4 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
              <span>Source: SupportMind Organization Trends API</span>
              <span className="break-all">
                /api/v1/organization/trends
              </span>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}