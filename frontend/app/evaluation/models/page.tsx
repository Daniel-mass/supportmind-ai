
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ExternalLink,
  FlaskConical,
  RefreshCw,
  Search,
  XCircle,
} from "lucide-react";
import { apiGet } from "@/lib/api/client";

type Experiment = {
  experiment_id: string;
  name: string;
  artifact_location?: string;
  lifecycle_stage?: string;
  tags?: Record<string, string>;
};

type Run = {
  run_id: string;
  experiment_id: string;
  run_name?: string;
  status?: string;
  run_uuid?: string;
  metrics?: Record<string, number>;
  params?: Record<string, string>;
  tags?: Record<string, string>;
};

type ExperimentsResponse = {
  tracking_uri?: string;
  count?: number;
  experiments?: Experiment[];
};

type RunsResponse = {
  tracking_uri?: string;
  count?: number;
  runs?: Run[];
};

const MLFLOW_EXPERIMENT_URL =
  "http://127.0.0.1:5000/#/experiments/2/runs?workflowType=machine_learning&searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D";

function formatMetric(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  return Number(value.toFixed(6)).toString();
}

function statusClass(status?: string): string {
  switch (status?.toUpperCase()) {
    case "FINISHED":
      return "text-green-700 bg-green-50 dark:text-green-400 dark:bg-green-950";
    case "FAILED":
      return "text-red-700 bg-red-50 dark:text-red-400 dark:bg-red-950";
    case "RUNNING":
      return "text-blue-700 bg-blue-50 dark:text-blue-400 dark:bg-blue-950";
    default:
      return "text-muted-foreground bg-muted";
  }
}

function ErrorMessage({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-red-300 p-4 text-sm">
      <AlertCircle className="mt-0.5 size-5 shrink-0 text-red-500" />
      <div>
        <p className="font-medium">Unable to load MLflow data</p>
        <p className="mt-1 break-words text-muted-foreground">{message}</p>
      </div>
    </div>
  );
}

function MetricCards({ run }: { run: Run }) {
  const metrics = Object.entries(run.metrics ?? {});

  if (metrics.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        This run has no logged metrics available through the current API.
      </p>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {metrics.map(([name, value]) => (
        <div key={name} className="rounded-lg border p-4">
          <p className="break-words text-sm text-muted-foreground">{name}</p>
          <p className="mt-2 break-all text-2xl font-semibold tabular-nums">
            {formatMetric(value)}
          </p>
        </div>
      ))}
    </div>
  );
}

export default function ModelEvaluationPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState("2");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshingRuns, setRefreshingRuns] = useState(false);
  const [error, setError] = useState("");
  const [runsError, setRunsError] = useState("");
  const [trackingUri, setTrackingUri] = useState("http://127.0.0.1:5000");

  const loadExperiments = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const response = await apiGet<ExperimentsResponse>(
        "/api/v1/mlflow/experiments?limit=100"
      );

      const items = response.experiments ?? [];
      setExperiments(items);

      if (response.tracking_uri) {
        setTrackingUri(response.tracking_uri);
      }

      // Prefer Experiment 2, matching the MLflow page supplied.
      // If it does not exist, use the first available experiment.
      setSelectedExperiment((current) => {
        if (items.some((item) => item.experiment_id === current)) {
          return current;
        }

        if (items.some((item) => item.experiment_id === "2")) {
          return "2";
        }

        return items[0]?.experiment_id ?? "";
      });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not retrieve MLflow experiments."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const loadRuns = useCallback(async () => {
    if (!selectedExperiment) {
      setRuns([]);
      return;
    }

    setRefreshingRuns(true);
    setRunsError("");

    try {
      const response = await apiGet<RunsResponse>(
        `/api/v1/mlflow/runs?experiment_id=${encodeURIComponent(
          selectedExperiment
        )}&limit=100`
      );

      setRuns(response.runs ?? []);

      if (response.tracking_uri) {
        setTrackingUri(response.tracking_uri);
      }

      setSelectedRunId((current) =>
        (response.runs ?? []).some((run) => run.run_id === current)
          ? current
          : ""
      );
    } catch (err) {
      setRuns([]);
      setRunsError(
        err instanceof Error
          ? err.message
          : "Could not retrieve runs for this experiment."
      );
    } finally {
      setRefreshingRuns(false);
    }
  }, [selectedExperiment]);

  useEffect(() => {
    void loadExperiments();
  }, [loadExperiments]);

  useEffect(() => {
    void loadRuns();
  }, [loadRuns]);

  const filteredRuns = useMemo(() => {
    const term = search.trim().toLowerCase();

    if (!term) return runs;

    return runs.filter((run) => {
      const searchable = [
        run.run_id,
        run.run_name,
        run.status,
        ...Object.keys(run.metrics ?? {}),
        ...Object.keys(run.params ?? {}),
        ...Object.values(run.params ?? {}),
        ...Object.keys(run.tags ?? {}),
        ...Object.values(run.tags ?? {}),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      return searchable.includes(term);
    });
  }, [runs, search]);

  const selectedRun = runs.find((run) => run.run_id === selectedRunId);
  const successfulRuns = runs.filter(
    (run) => run.status?.toUpperCase() === "FINISHED"
  ).length;
  const failedRuns = runs.filter(
    (run) => run.status?.toUpperCase() === "FAILED"
  ).length;
  const experimentName =
    experiments.find(
      (experiment) => experiment.experiment_id === selectedExperiment
    )?.name ?? `Experiment ${selectedExperiment}`;

  return (
    <main className="space-y-6 p-4 md:p-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <FlaskConical className="size-6" />
            <h1 className="text-2xl font-semibold tracking-tight">
              Model Evaluation
            </h1>
          </div>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            Explore MLflow experiments, model runs, logged metrics, parameters,
            and tags from your SupportMind AI workspace.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <a
            href={MLFLOW_EXPERIMENT_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-accent"
          >
            <ExternalLink className="size-4" />
            Open MLflow Experiment
          </a>

          <button
            type="button"
            onClick={() => {
              void loadExperiments();
              void loadRuns();
            }}
            disabled={loading || refreshingRuns}
            className="inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-accent disabled:opacity-50"
          >
            <RefreshCw
              className={`size-4 ${
                loading || refreshingRuns ? "animate-spin" : ""
              }`}
            />
            Refresh
          </button>
        </div>
      </div>

      <div className="rounded-lg border p-4">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Activity className="size-4" />
          MLflow tracking server
        </div>
        <p className="mt-1 break-all font-mono text-sm">{trackingUri}</p>
      </div>

      {error && <ErrorMessage message={error} />}

      {!error && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-xl border p-5">
              <p className="text-sm text-muted-foreground">Experiments</p>
              <p className="mt-2 text-3xl font-semibold">
                {loading ? "—" : experiments.length}
              </p>
            </div>

            <div className="rounded-xl border p-5">
              <p className="text-sm text-muted-foreground">Runs loaded</p>
              <p className="mt-2 text-3xl font-semibold">
                {refreshingRuns ? "—" : runs.length}
              </p>
            </div>

            <div className="rounded-xl border p-5">
              <p className="flex items-center gap-2 text-sm text-muted-foreground">
                <CheckCircle2 className="size-4" />
                Finished runs
              </p>
              <p className="mt-2 text-3xl font-semibold">{successfulRuns}</p>
            </div>

            <div className="rounded-xl border p-5">
              <p className="flex items-center gap-2 text-sm text-muted-foreground">
                <XCircle className="size-4" />
                Failed runs
              </p>
              <p className="mt-2 text-3xl font-semibold">{failedRuns}</p>
            </div>
          </div>

          <section className="space-y-4 rounded-xl border p-4 md:p-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <div className="flex-1 space-y-2">
                <label
                  htmlFor="experiment"
                  className="text-sm font-medium"
                >
                  MLflow experiment
                </label>
                <select
                  id="experiment"
                  value={selectedExperiment}
                  onChange={(event) =>
                    setSelectedExperiment(event.target.value)
                  }
                  disabled={loading || experiments.length === 0}
                  className="w-full rounded-lg border bg-background px-3 py-2 text-sm sm:max-w-md"
                >
                  {experiments.length === 0 && (
                    <option value="">No experiments available</option>
                  )}
                  {experiments.map((experiment) => (
                    <option
                      key={experiment.experiment_id}
                      value={experiment.experiment_id}
                    >
                      {experiment.name} (ID: {experiment.experiment_id})
                    </option>
                  ))}
                </select>
              </div>

              <div className="relative w-full sm:max-w-xs">
                <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search runs, metrics, params..."
                  aria-label="Search MLflow runs"
                  className="w-full rounded-lg border bg-background py-2 pl-9 pr-3 text-sm"
                />
              </div>
            </div>

            <div>
              <h2 className="text-lg font-semibold">{experimentName}</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                {filteredRuns.length} matching runs
                {runs.length !== filteredRuns.length
                  ? ` out of ${runs.length}`
                  : ""}
              </p>
            </div>

            {runsError && <ErrorMessage message={runsError} />}

            {!runsError && (
              <div className="overflow-x-auto rounded-lg border">
                <table className="w-full min-w-[650px] text-left text-sm">
                  <thead className="bg-muted/50">
                    <tr>
                      <th className="px-4 py-3 font-medium">Run</th>
                      <th className="px-4 py-3 font-medium">Status</th>
                      <th className="px-4 py-3 font-medium">Logged metrics</th>
                      <th className="px-4 py-3 font-medium">Parameters</th>
                      <th className="px-4 py-3 font-medium">Details</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {filteredRuns.map((run) => (
                      <tr key={run.run_id} className="hover:bg-muted/30">
                        <td className="px-4 py-3">
                          <p className="font-medium">
                            {run.run_name || run.run_id.slice(0, 12)}
                          </p>
                          <p className="mt-1 font-mono text-xs text-muted-foreground">
                            {run.run_id.slice(0, 12)}…
                          </p>
                        </td>
                        <td className="px-4 py-3">
                          <span
                            className={`inline-flex rounded-full px-2 py-1 text-xs font-medium ${statusClass(
                              run.status
                            )}`}
                          >
                            {run.status || "UNKNOWN"}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          {Object.keys(run.metrics ?? {}).length}
                        </td>
                        <td className="px-4 py-3">
                          {Object.keys(run.params ?? {}).length}
                        </td>
                        <td className="px-4 py-3">
                          <button
                            type="button"
                            onClick={() => setSelectedRunId(run.run_id)}
                            className="rounded-md border px-3 py-1.5 text-xs font-medium hover:bg-accent"
                          >
                            View run
                          </button>
                        </td>
                      </tr>
                    ))}

                    {!refreshingRuns && filteredRuns.length === 0 && (
                      <tr>
                        <td
                          colSpan={5}
                          className="px-4 py-10 text-center text-muted-foreground"
                        >
                          {runs.length === 0
                            ? "No runs were returned for this experiment."
                            : "No runs match your search."}
                        </td>
                      </tr>
                    )}

                    {refreshingRuns && (
                      <tr>
                        <td
                          colSpan={5}
                          className="px-4 py-8 text-center text-muted-foreground"
                        >
                          Loading MLflow runs…
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {selectedRun && (
            <section className="space-y-5 rounded-xl border p-4 md:p-5">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <h2 className="text-lg font-semibold">
                    {selectedRun.run_name || "Run details"}
                  </h2>
                  <p className="mt-1 break-all font-mono text-xs text-muted-foreground">
                    {selectedRun.run_id}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedRunId("")}
                  className="self-start rounded-lg border px-3 py-2 text-sm hover:bg-accent"
                >
                  Close details
                </button>
              </div>

              <div>
                <h3 className="mb-3 font-medium">Logged metrics</h3>
                <MetricCards run={selectedRun} />
              </div>

              <div>
                <h3 className="mb-3 font-medium">Parameters</h3>
                {Object.keys(selectedRun.params ?? {}).length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    No parameters were logged for this run.
                  </p>
                ) : (
                  <div className="overflow-x-auto rounded-lg border">
                    <table className="w-full text-left text-sm">
                      <tbody className="divide-y">
                        {Object.entries(selectedRun.params ?? {}).map(
                          ([key, value]) => (
                            <tr key={key}>
                              <td className="px-4 py-2 font-medium">{key}</td>
                              <td className="break-all px-4 py-2 font-mono">
                                {value}
                              </td>
                            </tr>
                          )
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <div>
                <h3 className="mb-3 font-medium">Tags</h3>
                {Object.keys(selectedRun.tags ?? {}).length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    No custom tags were logged for this run.
                  </p>
                ) : (
                  <div className="overflow-x-auto rounded-lg border">
                    <table className="w-full text-left text-sm">
                      <tbody className="divide-y">
                        {Object.entries(selectedRun.tags ?? {}).map(
                          ([key, value]) => (
                            <tr key={key}>
                              <td className="px-4 py-2 font-medium">{key}</td>
                              <td className="break-all px-4 py-2">{value}</td>
                            </tr>
                          )
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <p className="text-xs text-muted-foreground">
                These are the metrics, parameters, and tags exposed by the
                current MLflow API. Evaluation-specific interpretation depends
                on what the run logged.
              </p>
            </section>
          )}
        </>
      )}
    </main>
  );
}
