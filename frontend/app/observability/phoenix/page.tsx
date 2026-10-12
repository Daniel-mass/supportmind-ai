
"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  Clock3,
  ExternalLink,
  FolderOpen,
  RefreshCw,
  Workflow,
  Layers3,
  Eye,
} from "lucide-react";
import { apiGet } from "@/lib/api/client";

type JsonRecord = Record<string, unknown>;

type PhoenixHealth = {
  status?: string;
  [key: string]: unknown;
};

type LoadResult<T> = {
  data: T | null;
  error: string | null;
};

const PHOENIX_UI_URL = "http://localhost:6006";

function asRecord(value: unknown): JsonRecord | null {
  if (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  ) {
    return value as JsonRecord;
  }

  return null;
}

function getText(
  record: JsonRecord | null,
  keys: string[],
  fallback = "—",
): string {
  if (!record) {
    return fallback;
  }

  for (const key of keys) {
    const value = record[key];

    if (
      typeof value === "string" ||
      typeof value === "number"
    ) {
      if (String(value).trim()) {
        return String(value);
      }
    }
  }

  return fallback;
}

function getArray(
  payload: unknown,
  possibleKeys: string[],
): JsonRecord[] {
  if (Array.isArray(payload)) {
    return payload
      .map(asRecord)
      .filter((item): item is JsonRecord => item !== null);
  }

  const record = asRecord(payload);

  if (!record) {
    return [];
  }

  for (const key of possibleKeys) {
    const value = record[key];

    if (Array.isArray(value)) {
      return value
        .map(asRecord)
        .filter((item): item is JsonRecord => item !== null);
    }
  }

  return [];
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function StatusLabel({ status }: { status: string }) {
  const normalized = status.toUpperCase();

  const healthy =
    normalized === "HEALTHY" ||
    normalized === "CONNECTED" ||
    normalized === "OK";

  const unhealthy =
    normalized === "UNHEALTHY" ||
    normalized === "UNAVAILABLE" ||
    normalized === "ERROR" ||
    normalized === "FAILED";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
        healthy
          ? "bg-emerald-500/10 text-emerald-600"
          : unhealthy
            ? "bg-red-500/10 text-red-600"
            : "bg-muted text-muted-foreground"
      }`}
    >
      {healthy ? (
        <CheckCircle2 className="size-3.5" />
      ) : unhealthy ? (
        <AlertCircle className="size-3.5" />
      ) : (
        <Activity className="size-3.5" />
      )}
      {status || "UNKNOWN"}
    </span>
  );
}

function DataPanel({
  title,
  description,
  items,
  emptyMessage,
}: {
  title: string;
  description: string;
  items: JsonRecord[];
  emptyMessage: string;
}) {
  return (
    <section className="overflow-hidden rounded-xl border">
      <div className="border-b p-5">
        <h2 className="font-semibold">{title}</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {description}
        </p>
      </div>

      {items.length === 0 ? (
        <div className="px-5 py-10 text-center">
          <Layers3 className="mx-auto size-8 text-muted-foreground" />
          <p className="mt-3 text-sm text-muted-foreground">
            {emptyMessage}
          </p>
        </div>
      ) : (
        <div className="divide-y">
          {items.map((item, index) => {
            const titleText = getText(
              item,
              [
                "name",
                "session_id",
                "sessionId",
                "trace_id",
                "traceId",
                "id",
                "project_name",
                "projectName",
              ],
              `Record ${index + 1}`,
            );

            const identifier = getText(
              item,
              [
                "id",
                "project_id",
                "projectId",
                "project_identifier",
                "session_id",
                "sessionId",
                "trace_id",
                "traceId",
              ],
              "",
            );

            return (
              <details key={`${identifier || titleText}-${index}`} className="group">
                <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-5 py-4 hover:bg-muted/30">
                  <div className="min-w-0">
                    <p className="break-all text-sm font-medium">
                      {titleText}
                    </p>
                    {identifier && identifier !== titleText && (
                      <p className="mt-1 break-all font-mono text-xs text-muted-foreground">
                        {identifier}
                      </p>
                    )}
                  </div>

                  <ChevronDown className="size-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-180" />
                </summary>

                <div className="border-t bg-muted/20 p-4">
                  <pre className="overflow-x-auto whitespace-pre-wrap break-words text-xs leading-5">
                    {formatValue(item)}
                  </pre>
                </div>
              </details>
            );
          })}
        </div>
      )}
    </section>
  );
}

export default function PhoenixPage() {
  const [health, setHealth] = useState<PhoenixHealth | null>(null);
  const [projects, setProjects] = useState<JsonRecord[]>([]);
  const [selectedProject, setSelectedProject] = useState("");
  const [projectDetails, setProjectDetails] = useState<JsonRecord | null>(null);
  const [traces, setTraces] = useState<JsonRecord[]>([]);
  const [sessions, setSessions] = useState<JsonRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    setError(null);
    setDetailError(null);

    try {
      const [healthResult, projectsResult] = await Promise.all([
        apiGet<PhoenixHealth>("/api/v1/phoenix/health"),
        apiGet<unknown>("/api/v1/phoenix/projects"),
      ]);

      setHealth(healthResult);

      const loadedProjects = getArray(projectsResult, [
        "projects",
        "data",
        "results",
        "items",
      ]);

      setProjects(loadedProjects);

      const preferredProject =
        loadedProjects.find((project) => {
          const identifier = getText(
            project,
            ["project_identifier", "project_id", "projectId", "id", "name"],
            "",
          );

          return identifier === selectedProject;
        }) ?? loadedProjects[0];

      const projectIdentifier = preferredProject
        ? getText(
            preferredProject,
            ["project_identifier", "project_id", "projectId", "id", "name"],
            "",
          )
        : "";

      if (!projectIdentifier) {
        setSelectedProject("");
        setProjectDetails(null);
        setTraces([]);
        setSessions([]);
        setDetailError(
          loadedProjects.length === 0
            ? "No Phoenix projects were returned by the API."
            : "Could not identify a project from the returned project data.",
        );
        return;
      }

      setSelectedProject(projectIdentifier);

      const encodedIdentifier = encodeURIComponent(projectIdentifier);

      const detailResult: LoadResult<unknown> = await (async () => {
        try {
          return {
            data: await apiGet<unknown>(
              `/api/v1/phoenix/projects/${encodedIdentifier}`,
            ),
            error: null,
          };
        } catch (err: unknown) {
          return {
            data: null,
            error:
              err instanceof Error
                ? err.message
                : "Unable to load project details.",
          };
        }
      })();

      const tracesResult: LoadResult<unknown> = await (async () => {
        try {
          return {
            data: await apiGet<unknown>(
              `/api/v1/phoenix/projects/${encodedIdentifier}/traces`,
            ),
            error: null,
          };
        } catch (err: unknown) {
          return {
            data: null,
            error:
              err instanceof Error
                ? err.message
                : "Unable to load traces.",
          };
        }
      })();

      const sessionsResult: LoadResult<unknown> = await (async () => {
        try {
          return {
            data: await apiGet<unknown>(
              `/api/v1/phoenix/projects/${encodedIdentifier}/sessions`,
            ),
            error: null,
          };
        } catch (err: unknown) {
          return {
            data: null,
            error:
              err instanceof Error
                ? err.message
                : "Unable to load sessions.",
          };
        }
      })();

      setProjectDetails(asRecord(detailResult.data));

      setTraces(
        getArray(tracesResult.data, [
          "traces",
          "data",
          "results",
          "items",
        ]),
      );

      setSessions(
        getArray(sessionsResult.data, [
          "sessions",
          "data",
          "results",
          "items",
        ]),
      );

      const detailErrors = [
        detailResult.error ? `Project details: ${detailResult.error}` : null,
        tracesResult.error ? `Traces: ${tracesResult.error}` : null,
        sessionsResult.error ? `Sessions: ${sessionsResult.error}` : null,
      ].filter((message): message is string => message !== null);

      if (detailErrors.length > 0) {
        setDetailError(detailErrors.join(" "));
      }
    } catch (err: unknown) {
      setHealth(null);
      setProjects([]);
      setProjectDetails(null);
      setTraces([]);
      setSessions([]);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to load Phoenix data from the SupportMind API.",
      );
    } finally {
      setLoading(false);
    }
  }, [selectedProject]);

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard, refreshKey]);

  const healthStatus = health
    ? getText(asRecord(health), ["status"], "UNKNOWN")
    : "UNAVAILABLE";

  const projectName =
    getText(projectDetails, ["name", "project_name", "projectName"], "") ||
    getText(
      projects.find(
        (project) =>
          getText(
            project,
            ["project_identifier", "project_id", "projectId", "id", "name"],
            "",
          ) === selectedProject,
      ) ?? null,
      ["name", "project_name", "projectName"],
      selectedProject || "No project selected",
    );

  return (
    <main className="space-y-6 p-6 md:p-8">
      <header className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Activity className="size-4" />
            <span>Observability</span>
            <span>/</span>
            <span>Phoenix</span>
          </div>

          <h1 className="mt-2 text-2xl font-semibold tracking-tight">
            Phoenix Observability
          </h1>

          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Inspect LLM application projects, traces, sessions, and
            instrumentation health from SupportMind AI.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <a
            href={PHOENIX_UI_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center justify-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium hover:bg-accent"
          >
            <ExternalLink className="size-4" />
            Open Phoenix UI
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
            <h2 className="font-medium">Unable to load Phoenix data</h2>
            <p className="mt-1 break-words text-sm text-muted-foreground">
              {error}
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              Check the SupportMind API and its connection to Phoenix, then
              refresh this page.
            </p>
          </div>
        </section>
      )}

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-xl border p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Phoenix API status</p>
            <Activity className="size-5 text-muted-foreground" />
          </div>

          <div className="mt-3">
            {loading && !health ? (
              <p className="text-sm text-muted-foreground">Checking…</p>
            ) : (
              <StatusLabel status={healthStatus} />
            )}
          </div>
        </div>

        <div className="rounded-xl border p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Projects</p>
            <FolderOpen className="size-5 text-muted-foreground" />
          </div>

          <p className="mt-3 text-2xl font-semibold">{projects.length}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Projects returned by the API
          </p>
        </div>

        <div className="rounded-xl border p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Traces</p>
            <Workflow className="size-5 text-muted-foreground" />
          </div>

          <p className="mt-3 text-2xl font-semibold">{traces.length}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Records for the selected project
          </p>
        </div>

        <div className="rounded-xl border p-5">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">Sessions</p>
            <Clock3 className="size-5 text-muted-foreground" />
          </div>

          <p className="mt-3 text-2xl font-semibold">{sessions.length}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Records for the selected project
          </p>
        </div>
      </section>

      <section className="rounded-xl border">
        <div className="flex flex-col justify-between gap-3 border-b p-5 sm:flex-row sm:items-center">
          <div>
            <h2 className="font-semibold">Project explorer</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Choose a Phoenix project to inspect its traces and sessions.
            </p>
          </div>

          <div className="relative">
            <select
              aria-label="Select Phoenix project"
              value={selectedProject}
              onChange={(event) => {
                setSelectedProject(event.target.value);
              }}
              disabled={loading || projects.length === 0}
              className="w-full appearance-none rounded-lg border bg-background py-2 pl-3 pr-9 text-sm disabled:opacity-50 sm:w-80"
            >
              {projects.length === 0 && (
                <option value="">No projects available</option>
              )}

              {projects.map((project, index) => {
                const identifier = getText(
                  project,
                  [
                    "project_identifier",
                    "project_id",
                    "projectId",
                    "id",
                    "name",
                  ],
                  "",
                );

                const name = getText(
                  project,
                  ["name", "project_name", "projectName"],
                  identifier || `Project ${index + 1}`,
                );

                return (
                  <option
                    key={identifier || `project-${index}`}
                    value={identifier}
                  >
                    {name}
                  </option>
                );
              })}
            </select>

            <ChevronDown className="pointer-events-none absolute right-3 top-2.5 size-4 text-muted-foreground" />
          </div>
        </div>

        <div className="grid gap-5 p-5 lg:grid-cols-2">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Selected project
            </p>

            <h3 className="mt-2 break-words text-lg font-semibold">
              {projectName}
            </h3>

            <p className="mt-1 break-all font-mono text-xs text-muted-foreground">
              {selectedProject || "No project identifier"}
            </p>
          </div>

          <div className="flex items-start gap-3 rounded-lg bg-muted/40 p-4">
            <Eye className="mt-0.5 size-5 shrink-0 text-muted-foreground" />

            <div>
              <p className="text-sm font-medium">Full trace inspection</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Open the Phoenix UI to inspect span timelines, trace details,
                latency, and available instrumentation views.
              </p>

              <a
                href={PHOENIX_UI_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-3 inline-flex items-center gap-2 text-sm font-medium underline underline-offset-4"
              >
                Go to Phoenix
                <ExternalLink className="size-3.5" />
              </a>
            </div>
          </div>
        </div>

        {projectDetails && (
          <details className="border-t">
            <summary className="cursor-pointer px-5 py-4 text-sm font-medium hover:bg-muted/30">
              View raw project details
            </summary>

            <pre className="overflow-x-auto whitespace-pre-wrap break-words border-t bg-muted/20 p-5 text-xs leading-5">
              {formatValue(projectDetails)}
            </pre>
          </details>
        )}
      </section>

      {detailError && (
        <section
          role="status"
          className="flex items-start gap-3 rounded-xl border p-4"
        >
          <AlertCircle className="mt-0.5 size-5 shrink-0 text-muted-foreground" />
          <div>
            <h2 className="font-medium">Some project data could not be loaded</h2>
            <p className="mt-1 break-words text-sm text-muted-foreground">
              {detailError}
            </p>
          </div>
        </section>
      )}

      <section className="grid gap-6 xl:grid-cols-2">
        <DataPanel
          title="Recent traces"
          description="Trace records returned for the selected Phoenix project. Expand a row to inspect its payload."
          items={traces}
          emptyMessage={
            loading
              ? "Loading traces…"
              : "No trace records are available for this project."
          }
        />

        <DataPanel
          title="Sessions"
          description="Session records returned for the selected Phoenix project. Expand a row to inspect its payload."
          items={sessions}
          emptyMessage={
            loading
              ? "Loading sessions…"
              : "No session records are available for this project."
          }
        />
      </section>

      {loading && (
        <p className="text-sm text-muted-foreground">
          Loading Phoenix observability data…
        </p>
      )}
    </main>
  );
}
