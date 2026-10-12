
"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  RefreshCw,
  FlaskConical,
  CheckCircle2,
  AlertCircle,
  ChevronDown,
  Database,
  ChartNoAxesCombined,
  ExternalLink,
} from "lucide-react";
import { apiGet } from "@/lib/api/client";

type MlflowHealth = {
  status: string;
  tracking_uri: string;
  experiment_count_check: number;
};

type Experiment = {
  experiment_id: string;
  name: string;
  artifact_location: string;
  lifecycle_stage: string;
  tags: Record<string, string>;
};

type Run = {
  run_id: string;
  run_uuid: string;
  run_name: string | null;
  experiment_id: string;
  status: string;
  metrics: Record<string, number>;
  params: Record<string, string>;
  tags: Record<string, string>;
};

type ExperimentsResponse = {
  tracking_uri: string;
  count: number;
  experiments: Experiment[];
};

type RunsResponse = {
  tracking_uri: string;
  count: number;
  runs: Run[];
};

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toLocaleString() : String(value);
  }

  return String(value);
}

function StatusLabel({ status }: { status: string }) {
  const normalized = status.toUpperCase();
  const successful =
    normalized === "FINISHED" || normalized === "HEALTHY";

  const failed =
    normalized === "FAILED" || normalized === "UNAVAILABLE";

  return (
    <span
      className={`inline-flex rounded-full px-2.5 py-1 text-xs font-medium ${
        successful
          ? "bg-emerald-500/10 text-emerald-600"
          : failed
            ? "bg-red-500/10 text-red-600"
            : "bg-muted text-muted-foreground"
      }`}
    >
      {status || "UNKNOWN"}
    </span>
  );
}

export default function MlflowPage() {
  const [health, setHealth] = useState<MlflowHealth | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState("all");
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const [healthResult, experimentResult] = await Promise.all([
        apiGet<MlflowHealth>("/api/v1/mlflow/health"),
        apiGet<ExperimentsResponse>("/api/v1/mlflow/experiments?limit=100"),
      ]);

      setHealth(healthResult);

      const loadedExperiments = experimentResult.experiments ?? [];
      setExperiments(loadedExperiments);

      const runPath =
        selectedExperiment === "all"
          ? "/api/v1/mlflow/runs?limit=100"
          : `/api/v1/mlflow/runs?experiment_id=${encodeURIComponent(
              selectedExperiment,
            )}&limit=100`;

      const runResult = await apiGet<RunsResponse>(runPath);
      const nextRuns = runResult.runs ?? [];

      setRuns(nextRuns);

      setSelectedRun((current) =>
        current
          ? nextRuns.find((run) => run.run_id === current.run_id) ?? null
          : null,
      );
    } catch (err: unknown) {
      setHealth(null);
      setExperiments([]);
      setRuns([]);
      setSelectedRun(null);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to load MLflow data from the SupportMind API.",
      );
    } finally {
      setLoading(false);
    }
  }, [selectedExperiment]);

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard, refreshKey]);

  const finishedRuns = runs.filter(
    (run) => run.status.toUpperCase() === "FINISHED",
  ).length;

  return (
    <main className="space-y-6 p-6 md:p-8">
      <header className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <ChartNoAxesCombined className="size-4" />
            <span>Observability</span>
            <span>/</span>
            <span>MLflow</span>
          </div>

          <h1 className="mt-2 text-2xl font-semibold tracking-tight">
            MLflow Tracking
          </h1>

          <p className="mt-1 text-sm text-muted-foreground">
            Monitor experiments, model runs, parameters, and recorded metrics.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <a
            href="http://localhost:5000"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center justify-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-accent"
          >
            <ExternalLink className="size-4" />
            Open MLflow UI
          </a>

          <button
            type="button"
            onClick={() => setRefreshKey((value) => value + 1)}
            disabled={loading}
            className="inline-flex items-center justify-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-accent disabled:opacity-50"
          >
            <RefreshCw
              className={`size-4 ${loading ? "animate-spin" : ""}`}
            />
            Refresh
          </button>
        </div>
      </header>

      {error && (
        <section
          role="alert"
          className="flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/5 p-4"
        >
          <AlertCircle className="mt-0.5 size-5 shrink-0 text-red-500" />
          <div>
            <h2 className="font-medium">Unable to load MLflow data</h2>
            <p className="mt-1 break-words text-sm text-muted-foreground">
              {error}
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              Check that the SupportMind API can reach your running MLflow
              server, then refresh this page.
            </p>
          </div>
        </section>
      )}

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-xl border p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Tracking status</p>
            <Activity className="size-5 text-muted-foreground" />
          </div>

          <div className="mt-3">
            {loading && !health ? (
              <p className="text-lg text-muted-foreground">Checking…</p>
            ) : health ? (
              <StatusLabel status={health.status} />
            ) : (
              <span className="text-sm text-muted-foreground">
                Not available
              </span>
            )}
          </div>

          {health && (
            <p className="mt-2 break-all text-xs text-muted-foreground">
              {health.tracking_uri}
            </p>
          )}
        </div>

        <div className="rounded-xl border p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Experiments</p>
            <FlaskConical className="size-5 text-muted-foreground" />
          </div>

          <p className="mt-3 text-2xl font-semibold">
            {experiments.length}
          </p>

          <p className="mt-1 text-xs text-muted-foreground">
            Retrieved from MLflow
          </p>
        </div>

        <div className="rounded-xl border p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Runs loaded</p>
            <Database className="size-5 text-muted-foreground" />
          </div>

          <p className="mt-3 text-2xl font-semibold">{runs.length}</p>

          <p className="mt-1 text-xs text-muted-foreground">
            Based on the selected experiment
          </p>
        </div>

        <div className="rounded-xl border p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Finished runs</p>
            <CheckCircle2 className="size-5 text-muted-foreground" />
          </div>

          <p className="mt-3 text-2xl font-semibold">{finishedRuns}</p>

          <p className="mt-1 text-xs text-muted-foreground">
            Among the currently loaded runs
          </p>
        </div>
      </section>

      <section className="rounded-xl border">
        <div className="flex flex-col justify-between gap-3 border-b p-5 sm:flex-row sm:items-center">
          <div>
            <h2 className="font-semibold">Experiments</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Select an experiment to filter its runs.
            </p>
          </div>

          <div className="relative">
            <select
              aria-label="Filter runs by experiment"
              value={selectedExperiment}
              onChange={(event) =>
                setSelectedExperiment(event.target.value)
              }
              className="w-full appearance-none rounded-lg border bg-background py-2 pl-3 pr-9 text-sm sm:w-72"
            >
              <option value="all">All experiments</option>

              {experiments.map((experiment) => (
                <option
                  key={experiment.experiment_id}
                  value={experiment.experiment_id}
                >
                  {experiment.name} (ID: {experiment.experiment_id})
                </option>
              ))}
            </select>

            <ChevronDown className="pointer-events-none absolute right-3 top-2.5 size-4 text-muted-foreground" />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-muted/40 text-xs text-muted-foreground">
              <tr>
                <th className="px-5 py-3 font-medium">Experiment</th>
                <th className="px-5 py-3 font-medium">ID</th>
                <th className="px-5 py-3 font-medium">Lifecycle</th>
                <th className="px-5 py-3 font-medium">Artifact location</th>
              </tr>
            </thead>

            <tbody className="divide-y">
              {experiments.map((experiment) => (
                <tr
                  key={experiment.experiment_id}
                  className="hover:bg-muted/30"
                >
                  <td className="px-5 py-3 font-medium">
                    {experiment.name}
                  </td>

                  <td className="px-5 py-3 font-mono text-xs">
                    {experiment.experiment_id}
                  </td>

                  <td className="px-5 py-3">
                    {experiment.lifecycle_stage}
                  </td>

                  <td className="max-w-xs truncate px-5 py-3 text-muted-foreground">
                    {experiment.artifact_location || "—"}
                  </td>
                </tr>
              ))}

              {!loading && experiments.length === 0 && (
                <tr>
                  <td
                    colSpan={4}
                    className="px-5 py-8 text-center text-muted-foreground"
                  >
                    {error
                      ? "Experiments unavailable."
                      : "No experiments found."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-xl border">
        <div className="border-b p-5">
          <h2 className="font-semibold">Model runs</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Choose a run to inspect its logged metrics, parameters, and tags.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-muted/40 text-xs text-muted-foreground">
              <tr>
                <th className="px-5 py-3 font-medium">Run name</th>
                <th className="px-5 py-3 font-medium">Run ID</th>
                <th className="px-5 py-3 font-medium">Status</th>
                <th className="px-5 py-3 font-medium">Metrics</th>
              </tr>
            </thead>

            <tbody className="divide-y">
              {runs.map((run) => (
                <tr
                  key={run.run_id}
                  onClick={() => setSelectedRun(run)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setSelectedRun(run);
                    }
                  }}
                  role="button"
                  tabIndex={0}
                  aria-selected={selectedRun?.run_id === run.run_id}
                  className={`cursor-pointer hover:bg-muted/40 ${
                    selectedRun?.run_id === run.run_id
                      ? "bg-accent/50"
                      : ""
                  }`}
                >
                  <td className="px-5 py-3 font-medium">
                    {run.run_name || run.tags["mlflow.runName"] || "Unnamed run"}
                  </td>

                  <td className="px-5 py-3 font-mono text-xs">
                    {run.run_id}
                  </td>

                  <td className="px-5 py-3">
                    <StatusLabel status={run.status} />
                  </td>

                  <td className="px-5 py-3">
                    {Object.keys(run.metrics).length}
                  </td>
                </tr>
              ))}

              {!loading && runs.length === 0 && (
                <tr>
                  <td
                    colSpan={4}
                    className="px-5 py-8 text-center text-muted-foreground"
                  >
                    {error ? "Runs unavailable." : "No runs found."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {selectedRun && (
        <section className="space-y-5 rounded-xl border p-5">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Selected run
            </p>

            <h2 className="mt-1 break-all text-lg font-semibold">
              {selectedRun.run_name || selectedRun.run_id}
            </h2>

            <p className="mt-1 break-all font-mono text-xs text-muted-foreground">
              {selectedRun.run_id}
            </p>
          </div>

          <div>
            <h3 className="mb-3 font-medium">Logged metrics</h3>

            {Object.keys(selectedRun.metrics).length > 0 ? (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {Object.entries(selectedRun.metrics).map(([name, value]) => (
                  <div key={name} className="rounded-lg bg-muted/40 p-4">
                    <p className="break-all text-sm text-muted-foreground">
                      {name}
                    </p>

                    <p className="mt-2 break-all text-xl font-semibold">
                      {formatValue(value)}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                This run has no logged metrics.
              </p>
            )}
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            <div>
              <h3 className="mb-2 font-medium">Parameters</h3>

              <div className="overflow-x-auto rounded-lg border">
                <table className="w-full text-left text-sm">
                  <tbody className="divide-y">
                    {Object.entries(selectedRun.params).map(([key, value]) => (
                      <tr key={key}>
                        <td className="break-all px-3 py-2 text-muted-foreground">
                          {key}
                        </td>
                        <td className="break-all px-3 py-2">{value}</td>
                      </tr>
                    ))}

                    {Object.keys(selectedRun.params).length === 0 && (
                      <tr>
                        <td className="px-3 py-3 text-muted-foreground">
                          No parameters logged.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h3 className="mb-2 font-medium">Tags</h3>

              <div className="overflow-x-auto rounded-lg border">
                <table className="w-full text-left text-sm">
                  <tbody className="divide-y">
                    {Object.entries(selectedRun.tags).map(([key, value]) => (
                      <tr key={key}>
                        <td className="break-all px-3 py-2 text-muted-foreground">
                          {key}
                        </td>
                        <td className="break-all px-3 py-2">{value}</td>
                      </tr>
                    ))}

                    {Object.keys(selectedRun.tags).length === 0 && (
                      <tr>
                        <td className="px-3 py-3 text-muted-foreground">
                          No custom tags logged.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>
      )}

      {loading && (
        <p className="text-sm text-muted-foreground">
          Loading tracking data…
        </p>
      )}
    </main>
  );
}