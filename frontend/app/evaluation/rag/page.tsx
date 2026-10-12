
"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  ClipboardCheck,
  Database,
  FileText,
  Gauge,
  Layers3,
  Loader2,
  Search,
  ShieldCheck,
  XCircle,
} from "lucide-react";

import { useDeepEvalRawResult, useDeepEvalResult } from "@/lib/queries/evaluation";

type EvidenceFilter = "all" | "cited" | "metrics" | "tickets";

const METRIC_INFO = {
  faithfulness: {
    label: "Faithfulness",
    description:
      "Checks whether the generated answer is supported by the retrieved context.",
    icon: ShieldCheck,
  },
  answer_relevancy: {
    label: "Answer Relevancy",
    description:
      "Checks whether the generated answer addresses the user's question.",
    icon: Gauge,
  },
} as const;

function formatPercent(value: number) {
  return `${(value * 100).toFixed(0)}%`;
}

function formatMetricName(value: string) {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function Panel({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-2xl border bg-card p-5 shadow-sm sm:p-6 ${className}`}
    >
      {children}
    </section>
  );
}

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
}) {
  return (
    <div>
      {eyebrow ? (
        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {eyebrow}
        </p>
      ) : null}
      <h2 className="mt-1 text-lg font-semibold tracking-tight">{title}</h2>
      {description ? (
        <p className="mt-1 text-sm leading-6 text-muted-foreground">
          {description}
        </p>
      ) : null}
    </div>
  );
}

function MetricCard({
  name,
  score,
  threshold,
  success,
  reason,
}: {
  name: string;
  score: number;
  threshold: number;
  success: boolean;
  reason?: string | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const info =
    METRIC_INFO[name as keyof typeof METRIC_INFO];
  const label = info?.label ?? formatMetricName(name);
  const Icon = info?.icon ?? Activity;
  const difference = score - threshold;

  return (
    <article className="rounded-xl border p-4 sm:p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-muted">
            <Icon className="size-5 text-muted-foreground" />
          </div>
          <div>
            <h3 className="font-semibold">{label}</h3>
            <p className="mt-1 text-xs text-muted-foreground">
              Required: {formatPercent(threshold)}
            </p>
          </div>
        </div>

        <span
          className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
            success
              ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
              : "bg-destructive/10 text-destructive"
          }`}
        >
          {success ? (
            <CheckCircle2 className="size-3.5" />
          ) : (
            <XCircle className="size-3.5" />
          )}
          {success ? "Passed" : "Failed"}
        </span>
      </div>

      <div className="mt-6 flex items-end justify-between gap-4">
        <div>
          <p className="text-3xl font-semibold tracking-tight tabular-nums">
            {formatPercent(score)}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {difference >= 0 ? "+" : ""}
            {(difference * 100).toFixed(0)} percentage points vs. threshold
          </p>
        </div>
        <div className="w-28 pb-2 sm:w-36">
          <div className="h-2 overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full rounded-full transition-all ${
                success ? "bg-emerald-500" : "bg-destructive"
              }`}
              style={{
                width: `${Math.max(0, Math.min(score * 100, 100))}%`,
              }}
            />
          </div>
          <div className="mt-2 flex justify-between text-[10px] text-muted-foreground">
            <span>0%</span>
            <span>100%</span>
          </div>
        </div>
      </div>

      <button
        type="button"
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
        className="mt-5 flex w-full items-center justify-between border-t pt-3 text-left text-sm font-medium hover:text-primary"
      >
        Evaluation explanation
        {expanded ? (
          <ChevronUp className="size-4" />
        ) : (
          <ChevronDown className="size-4" />
        )}
      </button>

      {expanded ? (
        <div className="mt-3 rounded-lg bg-muted/50 p-3 text-sm leading-6 text-muted-foreground">
          <p>
            {info?.description ??
              "Explanation returned by the evaluation report."}
          </p>
          <p className="mt-3 text-foreground">
            {reason || "No explanation was saved for this metric."}
          </p>
        </div>
      ) : null}
    </article>
  );
}

export default function RagEvaluationPage() {
  const {
    data,
    isLoading,
    isError,
    error,
    refetch,
    isFetching,
  } = useDeepEvalResult();

  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<EvidenceFilter>("all");
  const [rawOpen, setRawOpen] = useState(false);

  const {
    data: rawReport,
    isLoading: rawLoading,
    isError: rawError,
    error: rawErrorDetail,
    refetch: refetchRaw,
    isFetching: rawFetching,
  } = useDeepEvalRawResult(rawOpen);

  const citedIds = data?.grounding.cited_evidence_ids ?? [];
  const contexts = data?.retrieval.context ?? [];

  const evidence = useMemo(
    () =>
      contexts.map((content, index) => {
        const id = `E${index + 1}`;
        const isMetricContext = index >= 5;

        return {
          id,
          content,
          cited: citedIds.includes(id),
          source: isMetricContext
            ? "Organization intelligence"
            : "Sparse retrieval",
          category: isMetricContext ? "metrics" : "tickets",
        };
      }),
    [contexts, citedIds],
  );

  const filteredEvidence = useMemo(() => {
    const term = search.trim().toLowerCase();

    return evidence.filter((item) => {
      const matchesSearch =
        !term ||
        item.id.toLowerCase().includes(term) ||
        item.content.toLowerCase().includes(term) ||
        item.source.toLowerCase().includes(term);

      const matchesFilter =
        filter === "all" ||
        (filter === "cited" && item.cited) ||
        (filter === "metrics" && item.category === "metrics") ||
        (filter === "tickets" && item.category === "tickets");

      return matchesSearch && matchesFilter;
    });
  }, [evidence, filter, search]);

  if (isLoading) {
    return (
      <main className="min-h-screen bg-background">
        <div className="flex min-h-[65vh] items-center justify-center">
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <Loader2 className="size-5 animate-spin" />
            Loading DeepEval report...
          </div>
        </div>
      </main>
    );
  }

  if (isError || !data) {
    return (
      <main className="min-h-screen bg-background">
        <div className="mx-auto max-w-5xl px-6 py-10">
          <Panel>
            <div className="flex items-start gap-3">
              <XCircle className="mt-1 size-5 text-destructive" />
              <div className="min-w-0">
                <h1 className="font-semibold">
                  Unable to load the DeepEval report
                </h1>
                <p className="mt-2 break-words text-sm text-muted-foreground">
                  {error instanceof Error
                    ? error.message
                    : "The evaluation API returned an error."}
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  Check that the backend is running and that the API base URL
                  includes the correct version prefix.
                </p>
                <button
                  type="button"
                  onClick={() => void refetch()}
                  className="mt-4 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
                >
                  Retry
                </button>
              </div>
            </div>
          </Panel>
        </div>
      </main>
    );
  }

  const metrics = Object.entries(data.metrics);
  const metricCount = metrics.length;
  const passedCount = metrics.filter(([, metric]) => metric.success).length;
  const failedCount = metricCount - passedCount;
  const hasEvidenceGap =
    /no direct evidence|does not specify|not provided|insufficient evidence/i.test(
      data.output.answer,
    );

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto max-w-7xl px-5 py-8 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-5 border-b pb-7 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              <Activity className="size-4" />
              Evaluation & Quality
            </div>
            <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              RAG Evaluation Lab
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">
              Inspect answer quality, retrieved evidence, grounding and
              evaluation explanations from the latest saved DeepEval run.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <span className="inline-flex items-center gap-2 rounded-full border px-3 py-2 text-sm">
              <span className="size-2 rounded-full bg-emerald-500" />
              Report available
            </span>
            <button
              type="button"
              onClick={() => void refetch()}
              disabled={isFetching}
              className="inline-flex items-center gap-2 rounded-lg border bg-card px-4 py-2 text-sm font-medium transition hover:bg-muted disabled:opacity-60"
            >
              <Activity className={`size-4 ${isFetching ? "animate-pulse" : ""}`} />
              Refresh report
            </button>
          </div>
        </header>

        <section className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Panel>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">
                Metrics passed
              </span>
              <ClipboardCheck className="size-5 text-muted-foreground" />
            </div>
            <p className="mt-4 text-3xl font-semibold tabular-nums">
              {passedCount}/{metricCount}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {failedCount === 0
                ? "All recorded metrics meet their thresholds"
                : `${failedCount} metric(s) below threshold`}
            </p>
          </Panel>

          <Panel>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">
                Retrieved contexts
              </span>
              <Layers3 className="size-5 text-muted-foreground" />
            </div>
            <p className="mt-4 text-3xl font-semibold tabular-nums">
              {data.retrieval.context_count}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Context items recorded in this report
            </p>
          </Panel>

          <Panel>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Grounding</span>
              <ShieldCheck className="size-5 text-muted-foreground" />
            </div>
            <p className="mt-4 text-2xl font-semibold capitalize">
              {data.grounding.status}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {data.grounding.valid
                ? "The saved grounding check is valid"
                : "The saved grounding check is invalid"}
            </p>
          </Panel>

          <Panel>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Model</span>
              <Database className="size-5 text-muted-foreground" />
            </div>
            <p className="mt-4 break-words text-xl font-semibold">
              {data.model.name}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {data.model.provider} · {data.evaluation.framework}
            </p>
          </Panel>
        </section>

        <section className="mt-8">
          <SectionHeading
            eyebrow="01 / Quality metrics"
            title="Metric inspector"
            description="Open any metric to see its definition and the explanation saved by DeepEval."
          />
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            {metrics.map(([name, metric]) => (
              <MetricCard
                key={name}
                name={name}
                score={metric.score}
                threshold={metric.threshold}
                success={metric.success}
                reason={metric.reason}
              />
            ))}
          </div>
        </section>

        <section className="mt-8 grid gap-4 lg:grid-cols-[1.3fr_0.7fr]">
          <Panel>
            <SectionHeading
              eyebrow="02 / Generated response"
              title="Answer inspection"
              description="The exact answer returned by the saved evaluation."
            />
            <div className="mt-5 rounded-xl border bg-muted/20 p-4 sm:p-5">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Model output
                </span>
                <span className="rounded-md bg-muted px-2 py-1 font-mono text-xs">
                  {data.model.name}
                </span>
              </div>
              <p className="whitespace-pre-wrap break-words text-sm leading-7">
                {data.output.answer}
              </p>
            </div>
            <div className="mt-5">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Original evaluation query
              </p>
              <p className="mt-2 text-sm leading-6">{data.input.query}</p>
            </div>
          </Panel>

          <div className="space-y-4">
            <Panel>
              <div className="flex items-start gap-3">
                {data.grounding.valid ? (
                  <CheckCircle2 className="mt-0.5 size-5 text-emerald-600 dark:text-emerald-400" />
                ) : (
                  <XCircle className="mt-0.5 size-5 text-destructive" />
                )}
                <div>
                  <h3 className="font-semibold">Grounding result</h3>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {data.grounding.valid
                      ? "The report marks this answer as grounded and valid."
                      : "The report marks this answer as not validly grounded."}
                  </p>
                </div>
              </div>

              <div className="mt-5 border-t pt-4">
                <p className="text-sm text-muted-foreground">
                  Cited evidence IDs
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {citedIds.length ? (
                    citedIds.map((id) => (
                      <span
                        key={id}
                        className="rounded-lg border bg-muted/40 px-2.5 py-1.5 font-mono text-xs"
                      >
                        {id}
                      </span>
                    ))
                  ) : (
                    <span className="text-sm text-muted-foreground">
                      No cited evidence IDs recorded.
                    </span>
                  )}
                </div>
              </div>
            </Panel>

            <Panel>
              <div className="flex items-start gap-3">
                {hasEvidenceGap ? (
                  <AlertTriangle className="mt-0.5 size-5 text-amber-600 dark:text-amber-400" />
                ) : (
                  <CheckCircle2 className="mt-0.5 size-5 text-muted-foreground" />
                )}
                <div>
                  <h3 className="font-semibold">Evidence coverage</h3>
                  <p className="mt-1 text-sm leading-6 text-muted-foreground">
                    {hasEvidenceGap
                      ? "The answer explicitly identifies requested topics that are not supported by the retrieved evidence. This is a coverage gap to investigate, not proof that the metric failed."
                      : "No explicit evidence-gap phrase was detected in the saved answer. Review the retrieved contexts to assess coverage."}
                  </p>
                </div>
              </div>
            </Panel>
          </div>
        </section>

        <section className="mt-8">
          <Panel>
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <SectionHeading
                eyebrow="03 / Retrieval inspection"
                title="Evidence explorer"
                description="Search the retrieved context items and isolate cited evidence or source groups."
              />
              <div className="text-sm text-muted-foreground">
                {filteredEvidence.length} of {evidence.length} contexts
              </div>
            </div>

            <div className="mt-5 flex flex-col gap-3 lg:flex-row">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search evidence text, ID or source..."
                  className="h-10 w-full rounded-lg border bg-background pl-9 pr-4 text-sm outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/10"
                />
              </div>
              <div className="flex flex-wrap gap-2">
                {(
                  [
                    ["all", "All evidence"],
                    ["cited", "Cited only"],
                    ["metrics", "Metrics"],
                    ["tickets", "Ticket text"],
                  ] as const
                ).map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setFilter(value)}
                    aria-pressed={filter === value}
                    className={`rounded-lg border px-3 py-2 text-sm transition ${
                      filter === value
                        ? "border-primary bg-primary/10 font-medium text-foreground"
                        : "bg-background text-muted-foreground hover:bg-muted"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-5 space-y-3">
              {filteredEvidence.length === 0 ? (
                <div className="rounded-xl border border-dashed p-8 text-center">
                  <Search className="mx-auto size-6 text-muted-foreground" />
                  <p className="mt-3 font-medium">No matching evidence</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Try another search term or change the evidence filter.
                  </p>
                </div>
              ) : (
                filteredEvidence.map((item) => (
                  <article
                    key={item.id}
                    className={`rounded-xl border p-4 ${
                      item.cited
                        ? "border-emerald-500/30 bg-emerald-500/[0.03]"
                        : "bg-background"
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="rounded-md bg-muted px-2 py-1 font-mono text-xs font-semibold">
                          {item.id}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {item.source}
                        </span>
                      </div>
                      <span
                        className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                          item.cited
                            ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                            : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {item.cited ? "Cited by answer" : "Not cited"}
                      </span>
                    </div>
                    <p className="mt-3 whitespace-pre-wrap break-words text-sm leading-6">
                      {item.content}
                    </p>
                  </article>
                ))
              )}
            </div>

            <p className="mt-4 text-xs leading-5 text-muted-foreground">
              Evidence IDs are displayed in retrieved-context order (E1, E2,
              and so on). Source groups follow the saved report's context
              ordering; inspect the raw report for full source metadata.
            </p>
          </Panel>
        </section>

        <section className="mt-8">
          <Panel>
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-start gap-3">
                <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-muted">
                  <FileText className="size-5 text-muted-foreground" />
                </div>
                <div>
                  <h2 className="font-semibold">Raw evaluation report</h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Inspect the original text output saved by the evaluation
                    pipeline.
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setRawOpen((current) => !current)}
                className="inline-flex items-center justify-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition hover:bg-muted"
              >
                {rawOpen ? "Hide raw report" : "View raw report"}
                {rawOpen ? (
                  <ChevronUp className="size-4" />
                ) : (
                  <ChevronDown className="size-4" />
                )}
              </button>
            </div>

            {rawOpen ? (
              <div className="mt-5">
                {rawLoading || rawFetching ? (
                  <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
                    <Loader2 className="size-4 animate-spin" />
                    Loading raw report...
                  </div>
                ) : rawError ? (
                  <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-4">
                    <p className="text-sm text-destructive">
                      {rawErrorDetail instanceof Error
                        ? rawErrorDetail.message
                        : "Unable to load the raw report."}
                    </p>
                    <button
                      type="button"
                      onClick={() => void refetchRaw()}
                      className="mt-3 rounded-lg border px-3 py-2 text-sm hover:bg-muted"
                    >
                      Retry
                    </button>
                  </div>
                ) : rawReport?.content ? (
                  <pre className="max-h-[600px] overflow-auto whitespace-pre-wrap break-words rounded-xl border bg-muted/30 p-4 font-mono text-xs leading-6">
                    {rawReport.content}
                  </pre>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No raw report content is available.
                  </p>
                )}
              </div>
            ) : null}
          </Panel>
        </section>

        <section className="mt-8">
          <Panel>
            <SectionHeading
              eyebrow="04 / Run configuration"
              title="Evaluation metadata"
              description="Configuration values recorded with this result."
            />
            <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-xl bg-muted/40 p-4">
                <p className="text-xs text-muted-foreground">Evaluation type</p>
                <p className="mt-2 font-medium">
                  {data.evaluation.evaluation_type}
                </p>
              </div>
              <div className="rounded-xl bg-muted/40 p-4">
                <p className="text-xs text-muted-foreground">Qdrant collection</p>
                <p className="mt-2 break-all font-mono text-sm">
                  {data.metadata.qdrant_collection ?? "Not recorded"}
                </p>
              </div>
              <div className="rounded-xl bg-muted/40 p-4">
                <p className="text-xs text-muted-foreground">Retrieval top-k</p>
                <p className="mt-2 font-medium">
                  {data.metadata.top_k ?? "Not recorded"}
                </p>
              </div>
              <div className="rounded-xl bg-muted/40 p-4">
                <p className="text-xs text-muted-foreground">Reranking</p>
                <p className="mt-2 font-medium">
                  {data.metadata.reranking === undefined
                    ? "Not recorded"
                    : data.metadata.reranking
                      ? "Enabled"
                      : "Disabled"}
                </p>
              </div>
            </div>
            <div className="mt-4 rounded-xl border p-4">
              <p className="text-sm font-medium">Metrics evaluated</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {(data.metadata.metrics_evaluated ?? metrics.map(([name]) => name)).map(
                  (name) => (
                    <span
                      key={name}
                      className="rounded-md bg-muted px-2.5 py-1.5 text-xs"
                    >
                      {formatMetricName(name)}
                    </span>
                  ),
                )}
              </div>
            </div>
          </Panel>
        </section>

        <footer className="mt-8 flex flex-col gap-2 border-t pt-5 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <span>
            Data source: SupportMind backend · Saved DeepEval result
          </span>
          <span className="font-mono">
            GET /api/v1/evaluation/deepeval
          </span>
        </footer>
      </div>
    </main>
  );
}