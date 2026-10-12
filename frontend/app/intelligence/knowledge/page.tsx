
"use client";

import {
  Activity,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleCheck,
  Clipboard,
  Database,
  FileSearch,
  Info,
  Loader2,
  Network,
  Search,
  ShieldCheck,
  Sparkles,
  Timer,
  XCircle,
  Zap,
} from "lucide-react";
import { useState, type ReactNode } from "react";

import { apiPost } from "@/lib/api/client";

type Evidence = {
  evidence_id: string;
  source: string;
  evidence_type: string;
  text: string;
  source_reference?: string | null;
  score?: number | null;
  metadata: Record<string, unknown>;
};

type Grounding = {
  status: string;
  valid: boolean;
  cited_evidence_ids: string[];
  invalid_citation_ids: string[];
  unsupported_claims: unknown[];
  issues: unknown[];
  metadata: Record<string, unknown>;
};

type GraphRAGResponse = {
  query: string;
  answer: string;
  evidence: Evidence[];
  grounding: Grounding;
  provider: string;
  model_name: string;
  latency_ms?: number | null;
  metadata: Record<string, unknown>;
};

type QueryPayload = {
  query: string;
  top_k: number;
  collection_name: string;
  include_graph: boolean;
  rerank: boolean;
  rerank_top_k: number;
  include_intelligence: boolean;
  include_workflow: boolean;
  max_evidence_items: number;
  max_evidence_characters: number;
  metadata: Record<string, string>;
};

function formatLabel(value: unknown): string {
  if (typeof value !== "string" || !value) {
    return "Unavailable";
  }

  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatLatency(value?: number | null): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Unavailable";
  }

  return value >= 1000
    ? `${(value / 1000).toFixed(1)}s`
    : `${Math.round(value)}ms`;
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function Panel({
  icon: Icon,
  title,
  subtitle,
  children,
  accent = "blue",
  defaultOpen = true,
}: {
  icon: typeof BrainCircuit;
  title: string;
  subtitle: string;
  children: ReactNode;
  accent?: "blue" | "violet" | "cyan" | "amber" | "emerald";
  defaultOpen?: boolean;
}) {
  const [expanded, setExpanded] = useState(defaultOpen);

  const colors = {
    blue: "border-blue-500/20 bg-blue-500/10 text-blue-400",
    violet: "border-violet-500/20 bg-violet-500/10 text-violet-400",
    cyan: "border-cyan-500/20 bg-cyan-500/10 text-cyan-400",
    amber: "border-amber-500/20 bg-amber-500/10 text-amber-400",
    emerald: "border-emerald-500/20 bg-emerald-500/10 text-emerald-400",
  };

  return (
    <section className="overflow-hidden rounded-2xl border border-border/60 bg-card/60 backdrop-blur-xl transition-colors hover:border-border">
      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
        className="flex w-full items-center justify-between gap-3 border-b border-border/40 px-4 py-4 text-left"
      >
        <div className="flex min-w-0 items-center gap-3">
          <div className={`rounded-xl border p-2 ${colors[accent]}`}>
            <Icon className="h-4 w-4" />
          </div>
          <div className="min-w-0">
            <h2 className="text-sm font-semibold">{title}</h2>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {subtitle}
            </p>
          </div>
        </div>
        <ChevronDown
          className={`h-4 w-4 shrink-0 text-muted-foreground transition-transform ${
            expanded ? "rotate-180" : ""
          }`}
        />
      </button>

      {expanded && <div className="space-y-4 p-4">{children}</div>}
    </section>
  );
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: ReactNode;
  detail?: string;
}) {
  return (
    <div className="min-w-0 rounded-xl border border-border/50 bg-background/40 p-4">
      <p className="text-xs text-muted-foreground">{label}</p>
      <div className="mt-2 break-words text-lg font-semibold tracking-tight">
        {value}
      </div>
      {detail && (
        <p className="mt-1 text-[11px] text-muted-foreground">{detail}</p>
      )}
    </div>
  );
}

function JsonDetails({ value }: { value: unknown }) {
  const [copied, setCopied] = useState(false);

  async function copyJson() {
    try {
      await navigator.clipboard.writeText(prettyJson(value));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => void copyJson()}
          className="inline-flex items-center gap-1.5 rounded-lg border border-border/60 px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          {copied ? (
            <Check className="h-3.5 w-3.5 text-emerald-400" />
          ) : (
            <Clipboard className="h-3.5 w-3.5" />
          )}
          {copied ? "Copied" : "Copy JSON"}
        </button>
      </div>
      <pre className="max-h-[480px] overflow-auto rounded-xl border border-border/50 bg-background/70 p-4 text-[11px] leading-5 text-muted-foreground">
        {prettyJson(value)}
      </pre>
    </div>
  );
}

function EvidenceCard({ evidence }: { evidence: Evidence }) {
  const [expanded, setExpanded] = useState(false);

  const metadata = evidence.metadata ?? {};
  const sourceFile =
    typeof metadata.source_file === "string"
      ? metadata.source_file
      : undefined;

  const scores =
    metadata.retrieval_scores &&
    typeof metadata.retrieval_scores === "object"
      ? (metadata.retrieval_scores as Record<string, unknown>)
      : undefined;

  return (
    <article className="overflow-hidden rounded-xl border border-border/50 bg-background/40 transition-colors hover:border-cyan-500/30">
      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
        className="w-full p-4 text-left"
      >
        <div className="flex items-start gap-3">
          <span className="shrink-0 rounded-lg border border-cyan-500/20 bg-cyan-500/10 px-2 py-1 text-[10px] font-semibold text-cyan-400">
            {evidence.evidence_id}
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-medium">
                {sourceFile ?? formatLabel(evidence.source)}
              </span>
              <span className="rounded-full border border-border/60 px-2 py-0.5 text-[10px] text-muted-foreground">
                {formatLabel(evidence.evidence_type)}
              </span>
            </div>
            <p
              className={`mt-2 whitespace-pre-wrap text-xs leading-6 text-muted-foreground ${
                expanded ? "" : "line-clamp-3"
              }`}
            >
              {evidence.text}
            </p>
          </div>
          <ChevronDown
            className={`mt-1 h-4 w-4 shrink-0 text-muted-foreground transition-transform ${
              expanded ? "rotate-180" : ""
            }`}
          />
        </div>
      </button>

      {expanded && (
        <div className="space-y-3 border-t border-border/50 px-4 py-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Retrieval source
              </p>
              <p className="mt-1 text-xs">{formatLabel(evidence.source)}</p>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Evidence score
              </p>
              <p className="mt-1 text-xs">
                {typeof evidence.score === "number"
                  ? evidence.score.toFixed(4)
                  : "Not provided"}
              </p>
            </div>
          </div>

          {evidence.source_reference && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Source reference
              </p>
              <p className="mt-1 break-all rounded-lg bg-muted/50 p-2 font-mono text-[11px] text-muted-foreground">
                {evidence.source_reference}
              </p>
            </div>
          )}

          {scores && Object.keys(scores).length > 0 && (
            <div>
              <p className="mb-2 text-[10px] uppercase tracking-wider text-muted-foreground">
                Returned retrieval scores
              </p>
              <div className="grid gap-2 sm:grid-cols-2">
                {Object.entries(scores).map(([key, value]) => (
                  <div
                    key={key}
                    className="rounded-lg border border-border/40 bg-card/50 p-2"
                  >
                    <p className="text-[10px] text-muted-foreground">
                      {formatLabel(key)}
                    </p>
                    <p className="mt-1 break-all font-mono text-xs">
                      {typeof value === "number"
                        ? value.toFixed(4)
                        : String(value)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          <details className="rounded-lg border border-border/40 p-3">
            <summary className="cursor-pointer text-xs text-muted-foreground">
              Additional evidence metadata
            </summary>
            <div className="mt-3">
              <JsonDetails value={metadata} />
            </div>
          </details>
        </div>
      )}
    </article>
  );
}

export default function KnowledgeRAGPage() {
  const [query, setQuery] = useState(
    "What support issues are described in the indexed knowledge documents, and what troubleshooting or resolution steps are provided?",
  );
  const [collection, setCollection] = useState(
    "supportmind_retrieval_pdf_test",
  );
  const [topK, setTopK] = useState(5);
  const [maxEvidence, setMaxEvidence] = useState(10);
  const [includeGraph, setIncludeGraph] = useState(true);
  const [rerank, setRerank] = useState(true);
  const [includeIntelligence, setIncludeIntelligence] = useState(true);
  const [showTechnical, setShowTechnical] = useState(false);
  const [result, setResult] = useState<GraphRAGResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    setResult(null);
    setElapsedSeconds(0);

    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);

    const payload: QueryPayload = {
      query: query.trim(),
      top_k: topK,
      collection_name: collection.trim(),
      include_graph: includeGraph,
      rerank,
      rerank_top_k: topK,
      include_intelligence: includeIntelligence,
      include_workflow: true,
      max_evidence_items: maxEvidence,
      max_evidence_characters: 8000,
      metadata: { source: "supportmind-knowledge-workbench" },
    };

    try {
      const response = await apiPost<QueryPayload, GraphRAGResponse>(
        "/api/graph-rag/query",
        payload,
      );
      setResult(response);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "The Graph-RAG request failed.",
      );
    } finally {
      window.clearInterval(timer);
      setIsLoading(false);
    }
  }

  const groundingValid = result?.grounding?.valid === true;
  const responseMetadata = result?.metadata ?? {};
  const retrievalContextCount = responseMetadata.retrieval_context_count;
  const intelligenceEvidenceCount = responseMetadata.intelligence_evidence_count;
  const evidenceCharacterCount = responseMetadata.evidence_character_count;
  const evidenceTruncated = responseMetadata.evidence_truncated;

  return (
    <main className="relative min-h-screen overflow-hidden bg-background">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-40 top-0 h-96 w-96 rounded-full bg-cyan-500/5 blur-3xl" />
        <div className="absolute right-0 top-1/4 h-96 w-96 rounded-full bg-blue-500/5 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-80 w-80 rounded-full bg-violet-500/5 blur-3xl" />
      </div>

      <div className="relative mx-auto max-w-7xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="mb-2 flex items-center gap-2">
              <div className="rounded-lg bg-cyan-500/10 p-2 text-cyan-400">
                <Sparkles className="h-4 w-4" />
              </div>
              <span className="text-xs font-medium uppercase tracking-[0.2em] text-cyan-400">
                SupportMind AI
              </span>
            </div>
            <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              Knowledge &amp; RAG
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">
              Explore Graph-RAG answers, inspect supporting evidence, and
              verify grounding using the actual backend response.
            </p>
          </div>

          <div className="flex items-center gap-2 self-start rounded-full border border-border/60 bg-card/60 px-3 py-2">
            <Network className="h-4 w-4 text-cyan-400" />
            <span className="text-xs text-muted-foreground">
              Graph-RAG workbench
            </span>
          </div>
        </header>

        <section className="rounded-2xl border border-border/60 bg-card/60 p-5 backdrop-blur-xl sm:p-6">
          <div className="mb-4 flex items-center gap-3">
            <div className="rounded-xl border border-blue-500/20 bg-blue-500/10 p-2 text-blue-400">
              <Search className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold">Ask your knowledge base</h2>
              <p className="text-xs text-muted-foreground">
                Submit a question to the existing Graph-RAG API.
              </p>
            </div>
          </div>

          <form onSubmit={(event) => void handleSubmit(event)} className="space-y-4">
            <label className="block">
              <span className="mb-2 block text-xs font-medium text-muted-foreground">
                Your question
              </span>
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                rows={3}
                required
                disabled={isLoading}
                placeholder="Ask a question about the indexed support knowledge..."
                className="w-full resize-y rounded-xl border border-border/70 bg-background/60 px-4 py-3 text-sm leading-6 outline-none transition-colors placeholder:text-muted-foreground/70 focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/10 disabled:opacity-60"
              />
            </label>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <label className="block">
                <span className="mb-2 block text-xs text-muted-foreground">
                  Qdrant collection
                </span>
                <input
                  value={collection}
                  onChange={(event) => setCollection(event.target.value)}
                  required
                  disabled={isLoading}
                  className="w-full rounded-lg border border-border/60 bg-background/60 px-3 py-2.5 text-xs outline-none focus:border-cyan-500/50 disabled:opacity-60"
                />
              </label>

              <label className="block">
                <span className="mb-2 block text-xs text-muted-foreground">
                  Retrieval top K
                </span>
                <select
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                  disabled={isLoading}
                  className="w-full rounded-lg border border-border/60 bg-background/60 px-3 py-2.5 text-xs outline-none focus:border-cyan-500/50 disabled:opacity-60"
                >
                  {[3, 5, 8, 10].map((value) => (
                    <option key={value} value={value}>
                      {value} results
                    </option>
                  ))}
                </select>
              </label>

              <label className="block">
                <span className="mb-2 block text-xs text-muted-foreground">
                  Maximum evidence items
                </span>
                <select
                  value={maxEvidence}
                  onChange={(event) => setMaxEvidence(Number(event.target.value))}
                  disabled={isLoading}
                  className="w-full rounded-lg border border-border/60 bg-background/60 px-3 py-2.5 text-xs outline-none focus:border-cyan-500/50 disabled:opacity-60"
                >
                  {[5, 10, 15, 20].map((value) => (
                    <option key={value} value={value}>
                      {value} items
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="flex flex-wrap gap-x-5 gap-y-3">
              {[
                {
                  label: "Include graph retrieval",
                  checked: includeGraph,
                  set: setIncludeGraph,
                },
                {
                  label: "Enable reranking",
                  checked: rerank,
                  set: setRerank,
                },
                {
                  label: "Include intelligence evidence",
                  checked: includeIntelligence,
                  set: setIncludeIntelligence,
                },
              ].map((option) => (
                <label
                  key={option.label}
                  className="flex cursor-pointer items-center gap-2 text-xs text-muted-foreground"
                >
                  <input
                    type="checkbox"
                    checked={option.checked}
                    onChange={(event) => option.set(event.target.checked)}
                    disabled={isLoading}
                    className="h-3.5 w-3.5 accent-cyan-500"
                  />
                  {option.label}
                </label>
              ))}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border/50 pt-4">
              <p className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                <Info className="h-3.5 w-3.5" />
                Reranking availability depends on backend configuration.
              </p>
              <button
                type="submit"
                disabled={isLoading || !query.trim() || !collection.trim()}
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-cyan-600 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Processing · {elapsedSeconds}s
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4" />
                    Run Graph-RAG
                  </>
                )}
              </button>
            </div>
          </form>
        </section>

        {isLoading && (
          <div className="rounded-2xl border border-cyan-500/20 bg-cyan-500/5 p-5">
            <div className="flex items-start gap-3">
              <Loader2 className="mt-0.5 h-5 w-5 animate-spin text-cyan-400" />
              <div>
                <p className="text-sm font-medium">Retrieving and generating an answer</p>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  Graph-RAG can take several minutes depending on retrieval,
                  model generation, and validation. Keep this page open while
                  the request runs.
                </p>
                <p className="mt-2 flex items-center gap-1.5 text-xs text-cyan-400">
                  <Timer className="h-3.5 w-3.5" />
                  {elapsedSeconds} seconds elapsed
                </p>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-4">
            <div className="flex items-start gap-3">
              <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-400" />
              <div className="min-w-0">
                <p className="text-sm font-medium text-red-300">
                  Graph-RAG request failed
                </p>
                <p className="mt-1 break-words text-xs leading-5 text-muted-foreground">
                  {error}
                </p>
                <button
                  type="button"
                  onClick={() => setError(null)}
                  className="mt-3 rounded-lg border border-border/60 px-3 py-1.5 text-xs hover:bg-muted"
                >
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        )}

        {result && (
          <>
            <section className="overflow-hidden rounded-2xl border border-cyan-500/20 bg-card/70 shadow-lg shadow-cyan-950/5 backdrop-blur-xl">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/50 px-5 py-4">
                <div className="flex items-center gap-3">
                  <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-2 text-cyan-400">
                    <BrainCircuit className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-sm font-semibold">Graph-RAG answer</h2>
                    <p className="text-xs text-muted-foreground">
                      Generated response from the API
                    </p>
                  </div>
                </div>
                <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] ${
                  groundingValid
                    ? "border-emerald-500/20 bg-emerald-500/5 text-emerald-400"
                    : "border-amber-500/20 bg-amber-500/5 text-amber-400"
                }`}>
                  {groundingValid ? (
                    <CircleCheck className="h-3.5 w-3.5" />
                  ) : (
                    <ShieldCheck className="h-3.5 w-3.5" />
                  )}
                  {formatLabel(result.grounding?.status)}
                </span>
              </div>

              <div className="space-y-5 p-5">
                <div className="whitespace-pre-wrap text-sm leading-7 text-foreground/90">
                  {result.answer || "The API returned an empty answer."}
                </div>

                <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border/50 pt-4">
                  <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                    <span className="rounded-full border border-border/60 px-2.5 py-1">
                      {result.evidence.length} evidence items
                    </span>
                    <span className="rounded-full border border-border/60 px-2.5 py-1">
                      {formatLatency(result.latency_ms)}
                    </span>
                    <span className="rounded-full border border-border/60 px-2.5 py-1">
                      {formatLabel(result.provider)}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      void navigator.clipboard.writeText(result.answer);
                    }}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-border/60 px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                  >
                    <Clipboard className="h-3.5 w-3.5" />
                    Copy answer
                  </button>
                </div>
              </div>
            </section>

            <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <Metric
                label="Grounding validation"
                value={
                  <span className={groundingValid ? "text-emerald-400" : "text-amber-400"}>
                    {groundingValid ? "Valid" : "Review"}
                  </span>
                }
                detail={`Status: ${formatLabel(result.grounding?.status)}`}
              />
              <Metric
                label="Cited evidence"
                value={result.grounding?.cited_evidence_ids?.length ?? 0}
                detail="Citation IDs returned by validation"
              />
              <Metric
                label="Invalid citations"
                value={result.grounding?.invalid_citation_ids?.length ?? 0}
                detail="Reported by the validator"
              />
              <Metric
                label="Response latency"
                value={formatLatency(result.latency_ms)}
                detail={`${result.provider} · ${result.model_name}`}
              />
            </section>

            <Panel
              icon={ShieldCheck}
              title="Grounding & citation validation"
              subtitle="Validation signals returned by the backend"
              accent="emerald"
            >
              <div className="grid gap-3 sm:grid-cols-2">
                <Metric
                  label="Validation result"
                  value={formatLabel(result.grounding?.status)}
                  detail={result.grounding?.valid ? "Backend marked the response valid" : "Backend did not mark the response valid"}
                />
                <Metric
                  label="Cited evidence IDs"
                  value={result.grounding?.cited_evidence_ids?.length ?? 0}
                  detail={(result.grounding?.cited_evidence_ids ?? []).join(", ") || "None returned"}
                />
              </div>

              {[
                {
                  title: "Invalid citation IDs",
                  values: result.grounding?.invalid_citation_ids ?? [],
                },
                {
                  title: "Unsupported claims",
                  values: result.grounding?.unsupported_claims ?? [],
                },
                {
                  title: "Validation issues",
                  values: result.grounding?.issues ?? [],
                },
              ].map((group) => (
                <div key={group.title} className="rounded-xl border border-border/50 bg-background/40 p-3">
                  <p className="text-xs font-medium">{group.title}</p>
                  {group.values.length ? (
                    <ul className="mt-2 space-y-2">
                      {group.values.map((item, index) => (
                        <li key={`${group.title}-${index}`} className="break-words text-xs leading-5 text-muted-foreground">
                          {typeof item === "string" ? item : prettyJson(item)}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-1 text-xs text-muted-foreground">
                      None reported by the validator.
                    </p>
                  )}
                </div>
              ))}

              {result.grounding?.metadata &&
                Object.keys(result.grounding.metadata).length > 0 && (
                  <details className="rounded-xl border border-border/50 p-3">
                    <summary className="cursor-pointer text-xs text-muted-foreground">
                      Additional validation metadata
                    </summary>
                    <div className="mt-3">
                      <JsonDetails value={result.grounding.metadata} />
                    </div>
                  </details>
                )}
            </Panel>

            <Panel
              icon={FileSearch}
              title="Retrieved evidence"
              subtitle="Inspect source text, provenance, and available scores"
              accent="cyan"
            >
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <span className="rounded-full border border-cyan-500/20 bg-cyan-500/5 px-2.5 py-1 text-[11px] text-cyan-400">
                  {result.evidence.length} returned
                </span>
                {evidenceTruncated === true && (
                  <span className="rounded-full border border-amber-500/20 bg-amber-500/5 px-2.5 py-1 text-[11px] text-amber-400">
                    Evidence truncated by backend
                  </span>
                )}
                {typeof evidenceCharacterCount === "number" && (
                  <span className="rounded-full border border-border/60 px-2.5 py-1 text-[11px] text-muted-foreground">
                    {evidenceCharacterCount.toLocaleString()} characters
                  </span>
                )}
              </div>

              {result.evidence.length ? (
                <div className="space-y-2">
                  {result.evidence.map((evidence) => (
                    <EvidenceCard
                      key={evidence.evidence_id}
                      evidence={evidence}
                    />
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-border/50 bg-background/40 p-5 text-sm text-muted-foreground">
                  The API returned no evidence items for this query.
                </div>
              )}
            </Panel>

            <Panel
              icon={Activity}
              title="Retrieval diagnostics"
              subtitle="Counts and context-budget details from the response"
              accent="violet"
            >
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <Metric
                  label="Retrieval context"
                  value={typeof retrievalContextCount === "number" ? retrievalContextCount : "N/A"}
                  detail="Selected retrieval context items"
                />
                <Metric
                  label="Intelligence evidence"
                  value={typeof intelligenceEvidenceCount === "number" ? intelligenceEvidenceCount : "N/A"}
                  detail="Evidence count from the intelligence layer"
                />
                <Metric
                  label="Returned evidence"
                  value={result.evidence.length}
                  detail="After evidence fusion and budgeting"
                />
                <Metric
                  label="Evidence characters"
                  value={typeof evidenceCharacterCount === "number" ? evidenceCharacterCount.toLocaleString() : "N/A"}
                  detail={evidenceTruncated === true ? "Backend reports truncation" : "Backend truncation status shown below"}
                />
              </div>

              <div className="rounded-xl border border-border/50 bg-background/40 p-3">
                <p className="text-xs font-medium">Context budget status</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {typeof evidenceTruncated === "boolean"
                    ? evidenceTruncated
                      ? "The backend reports that evidence was truncated to fit its configured budget."
                      : "The backend reports no evidence truncation."
                    : "Truncation status was not included in the response."}
                </p>
              </div>

              <p className="text-[11px] leading-5 text-muted-foreground">
                Individual dense, BM25, graph-retrieval, and reranking candidate
                lists are not exposed in this response, so this page does not
                invent stage-level results.
              </p>
            </Panel>

            <Panel
              icon={Database}
              title="Model & runtime"
              subtitle="Provider and generation information returned by the API"
              accent="amber"
            >
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                <Metric label="Provider" value={result.provider || "Unavailable"} />
                <Metric label="Model" value={result.model_name || "Unavailable"} />
                <Metric label="End-to-end latency" value={formatLatency(result.latency_ms)} />
              </div>
            </Panel>

            <Panel
              icon={Clipboard}
              title="Technical response"
              subtitle="Full API response for debugging and inspection"
              accent="emerald"
              defaultOpen={false}
            >
              <JsonDetails value={result} />
            </Panel>

            <div className="flex items-start gap-2 px-1 pb-3 text-[11px] leading-5 text-muted-foreground">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <p>
                Metrics and validation states on this page come from the Graph-RAG
                API response. A missing field is shown as unavailable rather
                than inferred.
              </p>
            </div>
          </>
        )}

        {!result && !isLoading && !error && (
          <section className="rounded-2xl border border-dashed border-border/70 bg-card/30 px-5 py-10 text-center">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl border border-cyan-500/20 bg-cyan-500/10 text-cyan-400">
              <FileSearch className="h-5 w-5" />
            </div>
            <h2 className="mt-4 text-sm font-semibold">Ready to explore your knowledge base</h2>
            <p className="mx-auto mt-2 max-w-lg text-xs leading-6 text-muted-foreground">
              Submit a question above to see the generated answer, supporting
              evidence, grounding checks, and available retrieval diagnostics.
            </p>
          </section>
        )}
      </div>
    </main>
  );
}
