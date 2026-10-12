"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type StatusDurationChartProps = {
  statusDurations: Record<string, number>;
};

type ChartRow = {
  status: string;
  seconds: number;
};

export function StatusDurationChart({
  statusDurations,
}: StatusDurationChartProps) {
  const data: ChartRow[] = Object.entries(statusDurations).map(
    ([status, seconds]) => ({
      status,
      seconds,
    }),
  );

  if (data.length === 0) {
    return (
      <div className="flex h-80 items-center justify-center rounded-xl border bg-card">
        <p className="text-sm text-muted-foreground">
          No status duration data available.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm">
      <div className="mb-5">
        <h2 className="text-base font-semibold">Status Duration</h2>

        <p className="mt-1 text-sm text-muted-foreground">
          Time spent by the ticket in each workflow status.
        </p>
      </div>

      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{
              top: 8,
              right: 24,
              bottom: 8,
              left: 24,
            }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              horizontal={false}
            />

            <XAxis
              type="number"
              domain={[0, "dataMax"]}
              tickFormatter={formatAxisDuration}
              tick={{ fontSize: 12 }}
            />

            <YAxis
              type="category"
              dataKey="status"
              width={120}
              tick={{ fontSize: 12 }}
            />

            <Tooltip
              formatter={(value) => [
                formatDuration(Number(value)),
                "Duration",
              ]}
              labelFormatter={(label) => `Status: ${label}`}
            />

            <Bar
              dataKey="seconds"
              name="Duration"
              radius={[0, 6, 6, 0]}
              barSize={28}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function formatDuration(seconds: number) {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);

  if (minutes < 60) {
    return remainingSeconds > 0
      ? `${minutes}m ${remainingSeconds}s`
      : `${minutes}m`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;

  return remainingMinutes > 0
    ? `${hours}h ${remainingMinutes}m`
    : `${hours}h`;
}

function formatAxisDuration(seconds: number) {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }

  const minutes = Math.round(seconds / 60);

  if (minutes < 60) {
    return `${minutes}m`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;

  return remainingMinutes > 0
    ? `${hours}h ${remainingMinutes}m`
    : `${hours}h`;
}