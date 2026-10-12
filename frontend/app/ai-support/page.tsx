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
  Gauge,
  GitBranch,
  Info,
  Loader2,
  MessageSquare,
  Network,
  Send,
  Sparkles,
  Timer,
  Workflow,
  XCircle,
  Zap,
} from "lucide-react";
import {
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import type {
  SupportAgentResult,
  SupportAssistResponse,
  SupportEvidence,
  SupportPrediction,
} from "@/lib/api/support";
import { useSupportAssist } from "@/lib/queries/support";

type PanelAccent =
  | "blue"
  | "violet"
  | "cyan"
  | "amber"
  | "emerald";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  data?: SupportAssistResponse;
  createdAt: number;
};

type ResponseSegment = {
  type: "text" | "bold" | "code";
  content: string;
};

type AgentDefinition = {
  key: string;
  icon: typeof BrainCircuit;
};

const AGENT_ORDER: AgentDefinition[] = [
  {
    key: "classification_agent",
    icon: BrainCircuit,
  },
  {
    key: "retrieval_agent",
    icon: FileSearch,
  },
  {
    key: "resolution_agent",
    icon: Zap,
  },
  {
    key: "escalation_agent",
    icon: GitBranch,
  },
  {
    key: "response_composer_agent",
    icon: MessageSquare,
  },
];

const WORKFLOW_STEPS = [
  "Classification",
  "Retrieval",
  "Resolution",
  "Escalation",
  "Response",
];

function cleanResponseText(text: string) {
  return text
    .replace(/&#x20;/gi, " ")
    .replace(/&#32;/gi, " ")
    .replace(/\u007f/g, "")
    .replace(/\\\*\*/g, "**")
    .replace(/\\\*/g, "*")
    .replace(/\\_/g, "_")
    .replace(/\\#/g, "#")
    .replace(/^\\---$/gm, "---")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n");
}

function parseInlineMarkdown(text: string): ResponseSegment[] {
  const segments: ResponseSegment[] = [];

  const pattern =
    /(\*\*(.+?)\*\*|`([^`]+)`)/g;

  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push({
        type: "text",
        content: text.slice(lastIndex, match.index),
      });
    }

    if (match[2]) {
      segments.push({
        type: "bold",
        content: match[2],
      });
    } else if (match[3]) {
      segments.push({
        type: "code",
        content: match[3],
      });
    }

    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < text.length) {
    segments.push({
      type: "text",
      content: text.slice(lastIndex),
    });
  }

  return segments.length > 0
    ? segments
    : [
        {
          type: "text",
          content: text,
        },
      ];
}

function InlineMarkdown({ text }: { text: string }) {
  const segments = parseInlineMarkdown(text);

  return (
    <>
      {segments.map((segment, index) => {
        if (segment.type === "bold") {
          return (
            <strong
              key={`${segment.type}-${index}`}
              className="font-semibold text-foreground"
            >
              {segment.content}
            </strong>
          );
        }

        if (segment.type === "code") {
          return (
            <code
              key={`${segment.type}-${index}`}
              className="rounded-md border border-border/60 bg-muted/70 px-1.5 py-0.5 font-mono text-[11px] text-blue-300"
            >
              {segment.content}
            </code>
          );
        }

        return (
          <span key={`${segment.type}-${index}`}>
            {segment.content}
          </span>
        );
      })}
    </>
  );
}

function FormattedResponse({ text }: { text: string }) {
  const cleanedText = cleanResponseText(text);
  const lines = cleanedText.split("\n");

  const elements: ReactNode[] = [];
  let paragraphLines: string[] = [];

  function flushParagraph() {
    if (paragraphLines.length === 0) {
      return;
    }

    const paragraph = paragraphLines.join(" ").trim();

    if (paragraph) {
      elements.push(
        <p
          key={`paragraph-${elements.length}`}
          className="text-sm leading-7 text-foreground/90"
        >
          <InlineMarkdown text={paragraph} />
        </p>,
      );
    }

    paragraphLines = [];
  }

  lines.forEach((rawLine, index) => {
    const line = rawLine.trim();

    if (!line) {
      flushParagraph();
      return;
    }

    if (/^---+$/.test(line)) {
      flushParagraph();

      elements.push(
        <div
          key={`divider-${index}`}
          className="my-4 border-t border-border/60"
        />,
      );

      return;
    }

    const headingMatch = line.match(
      /^#{1,3}\s+(.+)$/,
    );

    if (headingMatch) {
      flushParagraph();

      elements.push(
        <h4
          key={`heading-${index}`}
          className="pt-1 text-sm font-semibold text-foreground"
        >
          <InlineMarkdown text={headingMatch[1]} />
        </h4>,
      );

      return;
    }

    const numberedMatch = line.match(
      /^(\d+)\.\s+(.+)$/,
    );

    if (numberedMatch) {
      flushParagraph();

      elements.push(
        <div
          key={`numbered-${index}`}
          className="flex gap-3 rounded-xl border border-border/40 bg-background/40 px-3 py-2.5"
        >
          <span className="shrink-0 text-xs font-semibold text-blue-400">
            {numberedMatch[1]}.
          </span>

          <p className="text-sm leading-6 text-foreground/90">
            <InlineMarkdown text={numberedMatch[2]} />
          </p>
        </div>,
      );

      return;
    }

    const bulletMatch = line.match(
      /^[-*]\s+(.+)$/,
    );

    if (bulletMatch) {
      flushParagraph();

      elements.push(
        <div
          key={`bullet-${index}`}
          className="flex gap-3 rounded-xl border border-border/40 bg-background/40 px-3 py-2.5"
        >
          <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400" />

          <p className="text-sm leading-6 text-foreground/90">
            <InlineMarkdown text={bulletMatch[1]} />
          </p>
        </div>,
      );

      return;
    }

    paragraphLines.push(line);
  });

  flushParagraph();

  return (
    <div className="space-y-3">
      {elements}
    </div>
  );
}

function Panel({
  icon: Icon,
  title,
  subtitle,
  children,
  accent = "blue",
  expanded = true,
  onToggle,
}: {
  icon: typeof BrainCircuit;
  title: string;
  subtitle: string;
  children: ReactNode;
  accent?: PanelAccent;
  expanded?: boolean;
  onToggle?: () => void;
}) {
  const accents: Record<
    PanelAccent,
    {
      border: string;
      icon: string;
      glow: string;
    }
  > = {
    blue: {
      border: "border-blue-500/20",
      icon: "bg-blue-500/10 text-blue-400",
      glow: "hover:border-blue-500/35",
    },
    violet: {
      border: "border-violet-500/20",
      icon: "bg-violet-500/10 text-violet-400",
      glow: "hover:border-violet-500/35",
    },
    cyan: {
      border: "border-cyan-500/20",
      icon: "bg-cyan-500/10 text-cyan-400",
      glow: "hover:border-cyan-500/35",
    },
    amber: {
      border: "border-amber-500/20",
      icon: "bg-amber-500/10 text-amber-400",
      glow: "hover:border-amber-500/35",
    },
    emerald: {
      border: "border-emerald-500/20",
      icon: "bg-emerald-500/10 text-emerald-400",
      glow: "hover:border-emerald-500/35",
    },
  };

  const color = accents[accent];

  return (
    <div
      className={`group overflow-hidden rounded-2xl border bg-card/60 backdrop-blur-xl transition-all duration-300 ${color.border} ${color.glow}`}
    >
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between border-b border-border/50 px-4 py-3 text-left"
      >
        <div className="flex items-center gap-3">
          <div
            className={`rounded-xl p-2 ${color.icon}`}
          >
            <Icon className="h-4 w-4" />
          </div>

          <div>
            <h3 className="text-sm font-semibold">
              {title}
            </h3>

            <p className="text-xs text-muted-foreground">
              {subtitle}
            </p>
          </div>
        </div>

        <ChevronDown
          className={`h-4 w-4 text-muted-foreground transition-transform duration-300 ${
            expanded ? "rotate-180" : ""
          }`}
        />
      </button>

      {expanded && (
        <div className="p-4">
          {children}
        </div>
      )}
    </div>
  );
}

function ConfidenceBar({
  label,
  value,
  color = "bg-blue-500",
}: {
  label: string;
  value: number;
  color?: string;
}) {
  const safeValue = Math.max(
    0,
    Math.min(100, value),
  );

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="truncate text-muted-foreground">
          {label}
        </span>

        <span className="shrink-0 font-medium">
          {safeValue.toFixed(1)}%
        </span>
      </div>

      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className={`h-full rounded-full transition-all duration-1000 ${color}`}
          style={{
            width: `${safeValue}%`,
          }}
        />
      </div>
    </div>
  );
}

function AgentStep({
  icon: Icon,
  name,
  status,
  success,
  active,
}: {
  icon: typeof BrainCircuit;
  name: string;
  status: string;
  success: boolean;
  active?: boolean;
}) {
  return (
    <div
      className={`relative flex min-w-0 flex-col items-center rounded-xl border p-3 text-center transition-all ${
        active
          ? "border-blue-500/30 bg-blue-500/5"
          : success
            ? "border-border/50 bg-background/30"
            : "border-red-500/20 bg-red-500/5"
      }`}
    >
      <div
        className={`rounded-lg p-2 ${
          active
            ? "bg-blue-500/10 text-blue-400"
            : success
              ? "bg-emerald-500/10 text-emerald-400"
              : "bg-red-500/10 text-red-400"
        }`}
      >
        {active ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : success ? (
          <Icon className="h-4 w-4" />
        ) : (
          <XCircle className="h-4 w-4" />
        )}
      </div>

      <p className="mt-2 w-full truncate text-[11px] font-medium">
        {name}
      </p>

      <p className="mt-1 text-[10px] text-muted-foreground">
        {status}
      </p>

      {!active && success && (
        <CircleCheck className="absolute right-2 top-2 h-3.5 w-3.5 text-emerald-400" />
      )}
    </div>
  );
}

function formatConfidence(
  value: number | undefined,
) {
  if (
    typeof value !== "number" ||
    Number.isNaN(value)
  ) {
    return 0;
  }

  return value * 100;
}

function formatLabel(
  value: string | undefined | null,
) {
  if (!value) {
    return "Unavailable";
  }

  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) =>
      character.toUpperCase(),
    );
}

function getPrediction(
  prediction:
    | SupportPrediction
    | null
    | undefined,
  fallback = "Unavailable",
) {
  return prediction?.value
    ? formatLabel(prediction.value)
    : fallback;
}

function getAgent(
  agents: Record<string, SupportAgentResult>,
  name: string,
) {
  return agents[name];
}

function getAgentDisplayName(agentKey: string) {
  return formatLabel(
    agentKey.replace(/_agent$/, ""),
  );
}

function formatLatency(
  latency: number | undefined,
) {
  if (
    typeof latency !== "number" ||
    Number.isNaN(latency)
  ) {
    return "Unavailable";
  }

  if (latency >= 1000) {
    return `${(latency / 1000).toFixed(1)}s`;
  }

  return `${Math.round(latency)}ms`;
}

function formatSource(
  source: string | undefined,
) {
  if (!source) {
    return "Unknown source";
  }

  return source
    .replace(/^retrieval_/, "")
    .replaceAll("_", " ");
}

function EvidenceCard({
  evidence,
}: {
  evidence: SupportEvidence;
}) {
  const [expanded, setExpanded] =
    useState(false);

  const sourceFile =
    typeof evidence.provenance.source_file ===
    "string"
      ? evidence.provenance.source_file
      : undefined;

  const sourceReference =
    typeof evidence.provenance
      .source_reference === "string"
      ? evidence.provenance.source_reference
      : undefined;

  const page =
    typeof evidence.provenance.page ===
    "number"
      ? evidence.provenance.page
      : undefined;

  return (
    <div className="rounded-xl border border-border/50 bg-background/40 transition-all hover:border-cyan-500/30">
      <button
        type="button"
        onClick={() =>
          setExpanded((current) => !current)
        }
        className="w-full p-3 text-left"
      >
        <div className="flex items-start gap-3">
          <div className="shrink-0 rounded-lg bg-cyan-500/10 px-2 py-1 text-[10px] font-semibold text-cyan-400">
            {evidence.evidence_id}
          </div>

          <div className="min-w-0 flex-1">
            <p className="truncate text-xs font-medium">
              {sourceFile ??
                formatSource(evidence.source)}
            </p>

            <p
              className={`mt-1 text-[11px] leading-5 text-muted-foreground ${
                expanded
                  ? ""
                  : "line-clamp-2"
              }`}
            >
              {evidence.content}
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
        <div className="space-y-3 border-t border-border/50 px-3 pb-3 pt-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Retrieval source
              </p>

              <p className="mt-1 text-xs">
                {formatSource(evidence.source)}
              </p>
            </div>

            {page !== undefined && (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  Page
                </p>

                <p className="mt-1 text-xs">
                  {page}
                </p>
              </div>
            )}
          </div>

          {sourceReference && (
            <div>
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Provenance reference
              </p>

              <p className="mt-1 break-all rounded-lg bg-muted/50 p-2 font-mono text-[10px] text-muted-foreground">
                {sourceReference}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ProcessingIndicator({
  elapsedSeconds,
}: {
  elapsedSeconds: number;
}) {
  return (
    <div className="ml-12 rounded-2xl border border-blue-500/20 bg-blue-500/5 p-4">
      <div className="flex items-start gap-3">
        <div className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-blue-500/10 text-blue-400">
          <BrainCircuit className="h-4 w-4" />

          <span className="absolute inset-0 animate-ping rounded-xl border border-blue-400/30" />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs font-medium text-blue-300">
              SupportMind is analyzing your request
            </p>

            <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
              <Timer className="h-3 w-3" />
              {elapsedSeconds}s elapsed
            </div>
          </div>

          <p className="mt-1 text-[11px] leading-5 text-muted-foreground">
            This workflow can take time because SupportMind is running
            multiple intelligence layers. The request will remain active
            until the backend completes.
          </p>

          <div className="mt-3 grid gap-1.5 sm:grid-cols-5">
            {WORKFLOW_STEPS.map(
              (step, index) => (
                <div
                  key={step}
                  className="flex items-center gap-2 rounded-lg border border-border/40 bg-background/30 px-2 py-1.5"
                >
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-500/10 text-[9px] font-semibold text-blue-400">
                    {index + 1}
                  </span>

                  <span className="truncate text-[10px] text-muted-foreground">
                    {step}
                  </span>
                </div>
              ),
            )}
          </div>

          <div className="mt-3 flex items-center gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-400 [animation-delay:-0.3s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-400 [animation-delay:-0.15s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-400" />
          </div>
        </div>
      </div>
    </div>
  );
}

function ErrorMessage({
  error,
  onRetry,
}: {
  error: Error;
  onRetry: () => void;
}) {
  return (
    <div className="ml-12 rounded-2xl border border-red-500/20 bg-red-500/5 p-4">
      <div className="flex items-start gap-3">
        <div className="rounded-lg bg-red-500/10 p-2 text-red-400">
          <Activity className="h-4 w-4" />
        </div>

        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-red-300">
            SupportMind could not complete the request
          </p>

          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            {error.message}
          </p>

          <button
            type="button"
            onClick={onRetry}
            className="mt-3 rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-1.5 text-xs font-medium text-red-300 transition-colors hover:bg-red-500/10"
          >
            Try again
          </button>
        </div>
      </div>
    </div>
  );
}

function AssistantMessage({
  data,
}: {
  data: SupportAssistResponse;
}) {
  const intelligence = data.intelligence;
  const predictions = intelligence.predictions;
  const retrieval = intelligence.retrieval;
  const workflow = intelligence.workflow;
  const agents = intelligence.agents;

  const [expandedPanels, setExpandedPanels] =
    useState<Record<string, boolean>>({
      analysis: true,
      predictions: true,
      evidence: true,
      agents: true,
      technical: false,
    });

  const [copied, setCopied] =
    useState(false);

  const resolutionAgent = getAgent(
    agents,
    "resolution_agent",
  );

  const composerAgent = getAgent(
    agents,
    "response_composer_agent",
  );

  const resolutionResult =
    resolutionAgent?.result &&
    typeof resolutionAgent.result ===
      "object"
      ? resolutionAgent.result
      : {};

  const resolutionType =
    typeof resolutionResult.resolution_type ===
    "string"
      ? formatLabel(
          resolutionResult.resolution_type,
        )
      : "Unavailable";

  const resolutionConfidence =
    typeof resolutionResult.confidence ===
    "number"
      ? resolutionResult.confidence * 100
      : undefined;

  const requiresHumanReview =
    resolutionResult.requires_human_review ===
      true ||
    composerAgent?.result
      ?.human_review_required === true;

  const completedAgents = Object.values(
    agents,
  ).filter((agent) => agent.success).length;

  const totalAgents =
    Object.keys(agents).length;

  const successfulWorkflow =
    workflow.status === "completed";

  const completedCapabilityCount =
    workflow.completed_capabilities.length;

  const handleTogglePanel = (
    panel: string,
  ) => {
    setExpandedPanels((current) => ({
      ...current,
      [panel]: !current[panel],
    }));
  };

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(
        data.response.text,
      );

      setCopied(true);

      window.setTimeout(() => {
        setCopied(false);
      }, 1800);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-3">
        <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-blue-500/20 bg-blue-500/10 text-blue-400">
          <BrainCircuit className="h-4 w-4" />
        </div>

        <div className="min-w-0 flex-1">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <span className="text-sm font-semibold">
              SupportMind
            </span>

            <span className="rounded-full border border-blue-500/20 bg-blue-500/5 px-2 py-0.5 text-[10px] font-medium text-blue-400">
              AI RESPONSE
            </span>

            <span
              className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${
                successfulWorkflow
                  ? "border-emerald-500/20 bg-emerald-500/5 text-emerald-400"
                  : "border-amber-500/20 bg-amber-500/5 text-amber-400"
              }`}
            >
              {formatLabel(workflow.status)}
            </span>
          </div>

          <div className="rounded-2xl rounded-tl-md border border-border bg-card/80 p-5 shadow-sm backdrop-blur-xl">
            <FormattedResponse
              text={data.response.text}
            />

            {requiresHumanReview && (
              <div className="mt-5 rounded-xl border border-amber-500/20 bg-amber-500/5 p-3">
                <div className="flex items-center gap-2">
                  <Gauge className="h-4 w-4 text-amber-400" />

                  <span className="text-xs font-medium text-amber-300">
                    Human review recommended
                  </span>
                </div>

                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  SupportMind identified a resolution path that should be
                  verified by a human before final action.
                </p>
              </div>
            )}

            <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-border/50 pt-3">
              <div className="flex flex-wrap items-center gap-2 text-[10px] text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Workflow className="h-3 w-3" />
                  {completedCapabilityCount} capabilities
                </span>

                <span>•</span>

                <span>
                  {completedAgents}/{totalAgents} agents
                </span>

                <span>•</span>

                <span>
                  {formatLatency(
                    retrieval.latency_ms,
                  )}
                </span>
              </div>

              <button
                type="button"
                onClick={() => void handleCopy()}
                className="flex items-center gap-1.5 rounded-lg border border-border/60 bg-background/40 px-2.5 py-1.5 text-[10px] font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              >
                {copied ? (
                  <>
                    <Check className="h-3 w-3 text-emerald-400" />
                    Copied
                  </>
                ) : (
                  <>
                    <Clipboard className="h-3 w-3" />
                    Copy response
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="ml-12 grid gap-3">
        <Panel
          icon={BrainCircuit}
          title="AI Analysis"
          subtitle="Resolution reasoning and recommended actions"
          accent="blue"
          expanded={expandedPanels.analysis}
          onToggle={() =>
            handleTogglePanel("analysis")
          }
        >
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl bg-background/50 p-3">
              <p className="text-[11px] text-muted-foreground">
                Resolution
              </p>

              <p className="mt-1 text-sm font-medium">
                {resolutionType}
              </p>
            </div>

            <div className="rounded-xl bg-background/50 p-3">
              <p className="text-[11px] text-muted-foreground">
                Confidence
              </p>

              <p className="mt-1 text-sm font-medium">
                {resolutionConfidence !==
                undefined
                  ? `${resolutionConfidence.toFixed(
                      0,
                    )}%`
                  : "Unavailable"}
              </p>
            </div>

            <div className="rounded-xl bg-background/50 p-3">
              <p className="text-[11px] text-muted-foreground">
                Review
              </p>

              <p
                className={`mt-1 text-sm font-medium ${
                  requiresHumanReview
                    ? "text-amber-400"
                    : "text-emerald-400"
                }`}
              >
                {requiresHumanReview
                  ? "Required"
                  : "Not required"}
              </p>
            </div>
          </div>

          {typeof resolutionResult.recommendation ===
            "string" && (
            <div className="mt-3 rounded-xl border border-border/50 bg-background/40 p-3">
              <p className="text-[11px] text-muted-foreground">
                Recommendation
              </p>

              <p className="mt-1 text-xs leading-5">
                {resolutionResult.recommendation}
              </p>
            </div>
          )}

          {Array.isArray(
            resolutionResult.proposed_actions,
          ) &&
            resolutionResult.proposed_actions
              .length > 0 && (
              <div className="mt-3">
                <p className="mb-2 text-[11px] text-muted-foreground">
                  Proposed actions
                </p>

                <div className="space-y-2">
                  {resolutionResult.proposed_actions.map(
                    (action, index) => (
                      <div
                        key={`${String(
                          action,
                        )}-${index}`}
                        className="flex gap-2 rounded-lg bg-background/40 p-2.5"
                      >
                        <span className="text-[11px] font-semibold text-blue-400">
                          {index + 1}
                        </span>

                        <span className="text-xs leading-5 text-muted-foreground">
                          {String(action)}
                        </span>
                      </div>
                    ),
                  )}
                </div>
              </div>
            )}
        </Panel>

        <Panel
          icon={Activity}
          title="ML Predictions"
          subtitle="Classification and intelligence signals"
          accent="violet"
          expanded={expandedPanels.predictions}
          onToggle={() =>
            handleTogglePanel("predictions")
          }
        >
          <div className="grid gap-4 sm:grid-cols-2">
            <ConfidenceBar
              label={`Sentiment · ${getPrediction(
                predictions.sentiment,
              )}`}
              value={formatConfidence(
                predictions.sentiment
                  ?.confidence,
              )}
              color="bg-violet-500"
            />

            <ConfidenceBar
              label={`Complexity · ${getPrediction(
                predictions.complexity,
              )}`}
              value={formatConfidence(
                predictions.complexity
                  ?.confidence,
              )}
              color="bg-violet-500"
            />

            <ConfidenceBar
              label={`Category · ${getPrediction(
                predictions.category,
              )}`}
              value={formatConfidence(
                predictions.category
                  ?.confidence,
              )}
              color="bg-blue-500"
            />

            <ConfidenceBar
              label={`Topic · ${getPrediction(
                predictions.topic,
              )}`}
              value={formatConfidence(
                predictions.topic?.confidence,
              )}
              color="bg-cyan-500"
            />

            {predictions.priority && (
              <ConfidenceBar
                label={`Priority · ${getPrediction(
                  predictions.priority,
                )}`}
                value={formatConfidence(
                  predictions.priority
                    .confidence,
                )}
                color="bg-amber-500"
              />
            )}
          </div>
        </Panel>

        <Panel
          icon={FileSearch}
          title="Knowledge & RAG"
          subtitle="Retrieved evidence supporting this answer"
          accent="cyan"
          expanded={expandedPanels.evidence}
          onToggle={() =>
            handleTogglePanel("evidence")
          }
        >
          <div className="mb-4 flex flex-wrap gap-2">
            <span className="rounded-full border border-cyan-500/20 bg-cyan-500/5 px-2.5 py-1 text-[11px] text-cyan-400">
              {formatLabel(
                retrieval.strategy,
              )}
            </span>

            <span
              className={`rounded-full border px-2.5 py-1 text-[11px] ${
                retrieval.grounding_valid
                  ? "border-emerald-500/20 bg-emerald-500/5 text-emerald-400"
                  : "border-amber-500/20 bg-amber-500/5 text-amber-400"
              }`}
            >
              {formatLabel(
                retrieval.grounding_status,
              )}
            </span>

            <span className="rounded-full border border-border bg-muted/50 px-2.5 py-1 text-[11px] text-muted-foreground">
              {retrieval.evidence_count} evidence{" "}
              {retrieval.evidence_count === 1
                ? "source"
                : "sources"}
            </span>
          </div>

          <div className="space-y-2">
            {intelligence.evidence
              .length > 0 ? (
              intelligence.evidence.map(
                (evidence) => (
                  <EvidenceCard
                    key={
                      evidence.evidence_id
                    }
                    evidence={evidence}
                  />
                ),
              )
            ) : (
              <div className="rounded-xl border border-border/50 bg-background/40 p-4 text-xs text-muted-foreground">
                No evidence was returned by the
                retrieval layer.
              </div>
            )}
          </div>
        </Panel>

        <Panel
          icon={Network}
          title="Agent Workflow"
          subtitle="Multi-agent execution path"
          accent="amber"
          expanded={expandedPanels.agents}
          onToggle={() =>
            handleTogglePanel("agents")
          }
        >
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-amber-500/20 bg-amber-500/5 px-2.5 py-1 text-[11px] text-amber-400">
              {completedAgents}/{totalAgents}{" "}
              successful
            </span>

            <span className="rounded-full border border-border bg-muted/50 px-2.5 py-1 text-[11px] text-muted-foreground">
              {workflow.iteration_count}{" "}
              iterations
            </span>

            <span className="rounded-full border border-border bg-muted/50 px-2.5 py-1 text-[11px] text-muted-foreground">
              {formatLabel(
                workflow.current_step,
              )}
            </span>
          </div>

          <div className="overflow-x-auto pb-1">
            <div className="grid min-w-[700px] grid-cols-5 gap-2">
              {AGENT_ORDER.map(
                ({ key, icon }) => {
                  const agent = agents[key];

                  if (!agent) {
                    return null;
                  }

                  return (
                    <AgentStep
                      key={key}
                      icon={icon}
                      name={
                        agent.agent_name ||
                        getAgentDisplayName(
                          key,
                        )
                      }
                      status={
                        agent.success
                          ? "Completed"
                          : "Failed"
                      }
                      success={
                        agent.success
                      }
                    />
                  );
                },
              )}
            </div>
          </div>

          {workflow.errors.length > 0 && (
            <div className="mt-3 rounded-xl border border-red-500/20 bg-red-500/5 p-3">
              <p className="text-xs font-medium text-red-400">
                Workflow errors
              </p>

              <div className="mt-2 space-y-2">
                {workflow.errors.map(
                  (error, index) => (
                    <div
                      key={`${error.source}-${index}`}
                    >
                      <p className="text-[11px] font-medium">
                        {error.source}
                      </p>

                      <p className="text-[11px] leading-5 text-muted-foreground">
                        {error.message}
                      </p>
                    </div>
                  ),
                )}
              </div>
            </div>
          )}
        </Panel>

        <Panel
          icon={Database}
          title="Technical Details"
          subtitle="Runtime and infrastructure information"
          accent="emerald"
          expanded={expandedPanels.technical}
          onToggle={() =>
            handleTogglePanel("technical")
          }
        >
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {[
              [
                "Provider",
                retrieval.provider,
              ],
              [
                "Model",
                retrieval.model_name,
              ],
              [
                "Strategy",
                formatLabel(
                  retrieval.strategy,
                ),
              ],
              [
                "Evidence",
                String(
                  retrieval.evidence_count,
                ),
              ],
              [
                "Iterations",
                String(
                  workflow.iteration_count,
                ),
              ],
            ].map(([label, value]) => (
              <div
                key={label}
                className="rounded-xl border border-border/50 bg-background/40 p-3"
              >
                <p className="text-[11px] text-muted-foreground">
                  {label}
                </p>

                <p className="mt-1 truncate text-xs font-semibold">
                  {value}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-border/50 bg-background/40 p-3">
              <p className="text-[11px] text-muted-foreground">
                Grounding
              </p>

              <p
                className={`mt-1 text-xs font-semibold ${
                  retrieval.grounding_valid
                    ? "text-emerald-400"
                    : "text-amber-400"
                }`}
              >
                {retrieval.grounding_valid
                  ? "Validated"
                  : "Requires verification"}
              </p>
            </div>

            <div className="rounded-xl border border-border/50 bg-background/40 p-3">
              <p className="text-[11px] text-muted-foreground">
                Retrieval latency
              </p>

              <p className="mt-1 text-xs font-semibold">
                {formatLatency(
                  retrieval.latency_ms,
                )}
              </p>
            </div>

            <div className="rounded-xl border border-border/50 bg-background/40 p-3">
              <p className="text-[11px] text-muted-foreground">
                Workflow
              </p>

              <p className="mt-1 text-xs font-semibold">
                {formatLabel(
                  workflow.status,
                )}
              </p>
            </div>
          </div>
        </Panel>

        <div className="flex items-start gap-2 px-1 pt-1 text-[10px] leading-5 text-muted-foreground">
          <Info className="mt-0.5 h-3 w-3 shrink-0" />

          <p>
            Intelligence shown above is derived from the SupportMind
            backend response for this request. No frontend-generated
            metrics are used.
          </p>
        </div>
      </div>
    </div>
  );
}

export default function AISupportPage() {
  const [message, setMessage] =
    useState("");

  const [chatMessages, setChatMessages] =
    useState<ChatMessage[]>([]);

  const [processingStartedAt, setProcessingStartedAt] =
    useState<number | null>(null);

  const [elapsedSeconds, setElapsedSeconds] =
    useState(0);

  const supportAssist =
    useSupportAssist();

  const isProcessing =
    supportAssist.isPending;

  useEffect(() => {
    if (!isProcessing) {
      setProcessingStartedAt(null);
      setElapsedSeconds(0);
      return;
    }

    const startedAt =
      processingStartedAt ?? Date.now();

    if (processingStartedAt === null) {
      setProcessingStartedAt(startedAt);
    }

    const updateElapsed = () => {
      setElapsedSeconds(
        Math.floor(
          (Date.now() - startedAt) / 1000,
        ),
      );
    };

    updateElapsed();

    const interval = window.setInterval(
      updateElapsed,
      1000,
    );

    return () => {
      window.clearInterval(interval);
    };
  }, [
    isProcessing,
    processingStartedAt,
  ]);

  async function handleSend() {
    const trimmedMessage =
      message.trim();

    if (
      !trimmedMessage ||
      isProcessing
    ) {
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmedMessage,
      createdAt: Date.now(),
    };

    setChatMessages((current) => [
      ...current,
      userMessage,
    ]);

    setMessage("");
    setProcessingStartedAt(
      Date.now(),
    );

    try {
      const response =
        await supportAssist.mutateAsync(
          trimmedMessage,
        );

      const assistantMessage: ChatMessage =
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            response.response.text,
          data: response,
          createdAt: Date.now(),
        };

      setChatMessages((current) => [
        ...current,
        assistantMessage,
      ]);
    } catch {
      // Mutation error is displayed below.
    }
  }

  function handleRetry() {
    const lastUserMessage =
      [...chatMessages]
        .reverse()
        .find(
          (item) =>
            item.role === "user",
        );

    if (!lastUserMessage) {
      return;
    }

    setProcessingStartedAt(
      Date.now(),
    );

    void supportAssist.mutateAsync(
      lastUserMessage.content,
    ).then(
      (response) => {
        const assistantMessage: ChatMessage =
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content:
              response.response.text,
            data: response,
            createdAt: Date.now(),
          };

        setChatMessages((current) => [
          ...current,
          assistantMessage,
        ]);
      },
      () => {
        // Error state remains visible.
      },
    );
  }

  const messageCount = useMemo(
    () => chatMessages.length,
    [chatMessages.length],
  );

  return (
    <main className="relative min-h-screen overflow-hidden bg-background">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-40 top-0 h-96 w-96 rounded-full bg-blue-500/5 blur-3xl" />

        <div className="absolute right-0 top-1/4 h-96 w-96 rounded-full bg-violet-500/5 blur-3xl" />

        <div className="absolute bottom-0 left-1/3 h-80 w-80 rounded-full bg-cyan-500/5 blur-3xl" />
      </div>

      <div className="relative mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
        <header className="mb-8 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="mb-2 flex items-center gap-2">
              <div className="rounded-lg bg-blue-500/10 p-2 text-blue-400">
                <Sparkles className="h-4 w-4" />
              </div>

              <span className="text-xs font-medium uppercase tracking-[0.2em] text-blue-400">
                SupportMind AI
              </span>
            </div>

            <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              Customer Support Intelligence
            </h1>

            <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
              Ask SupportMind anything and inspect the intelligence behind
              every response.
            </p>
          </div>

          <div className="flex items-center gap-2 self-start rounded-full border border-emerald-500/20 bg-emerald-500/5 px-3 py-1.5 md:self-auto">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />

              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
            </span>

            <span className="text-xs font-medium text-emerald-400">
              Intelligence layer operational
            </span>
          </div>
        </header>

        <section className="mx-auto max-w-4xl space-y-8">
          {messageCount === 0 && (
            <div className="overflow-hidden rounded-3xl border border-border bg-card/50 p-8 text-center backdrop-blur-xl">
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl border border-blue-500/20 bg-blue-500/10 text-blue-400 shadow-lg shadow-blue-500/5">
                <BrainCircuit className="h-6 w-6" />
              </div>

              <h2 className="mt-5 text-xl font-semibold">
                SupportMind is ready
              </h2>

              <p className="mx-auto mt-2 max-w-xl text-sm leading-6 text-muted-foreground">
                Ask a customer-support question and inspect the ML
                predictions, retrieved evidence, agent workflow, evaluation
                signals, and runtime intelligence behind the answer.
              </p>

              <div className="mx-auto mt-6 grid max-w-2xl gap-2 text-left sm:grid-cols-3">
                {[
                  {
                    icon: Activity,
                    title: "ML intelligence",
                    text: "Classification, sentiment and complexity",
                  },
                  {
                    icon: FileSearch,
                    title: "RAG evidence",
                    text: "Grounded knowledge retrieved for the response",
                  },
                  {
                    icon: Network,
                    title: "Agent workflow",
                    text: "Multi-agent execution and resolution path",
                  },
                ].map(
                  ({
                    icon: Icon,
                    title,
                    text,
                  }) => (
                    <div
                      key={title}
                      className="rounded-xl border border-border/50 bg-background/30 p-3"
                    >
                      <Icon className="h-4 w-4 text-blue-400" />

                      <p className="mt-2 text-xs font-medium">
                        {title}
                      </p>

                      <p className="mt-1 text-[11px] leading-5 text-muted-foreground">
                        {text}
                      </p>
                    </div>
                  ),
                )}
              </div>
            </div>
          )}

          {chatMessages.map(
            (chatMessage) => {
              if (
                chatMessage.role ===
                "user"
              ) {
                return (
                  <div
                    key={chatMessage.id}
                    className="flex justify-end"
                  >
                    <div className="max-w-2xl rounded-2xl rounded-br-md border border-border bg-muted/50 px-5 py-4 shadow-sm">
                      <div className="mb-2 flex items-center justify-end gap-2">
                        <span className="text-[11px] font-medium text-muted-foreground">
                          You
                        </span>
                      </div>

                      <p className="whitespace-pre-wrap text-sm leading-7">
                        {
                          chatMessage.content
                        }
                      </p>
                    </div>
                  </div>
                );
              }

              if (!chatMessage.data) {
                return null;
              }

              return (
                <AssistantMessage
                  key={chatMessage.id}
                  data={chatMessage.data}
                />
              );
            },
          )}

          {isProcessing && (
            <ProcessingIndicator
              elapsedSeconds={
                elapsedSeconds
              }
            />
          )}

          {supportAssist.isError &&
            supportAssist.error && (
              <ErrorMessage
                error={
                  supportAssist.error
                }
                onRetry={
                  handleRetry
                }
              />
            )}

          <div className="sticky bottom-5 z-10">
            <div className="rounded-2xl border border-border bg-card/90 p-2 shadow-2xl shadow-black/10 backdrop-blur-xl">
              <div className="flex items-end gap-2">
                <textarea
                  value={message}
                  onChange={(event) =>
                    setMessage(
                      event.target.value,
                    )
                  }
                  onKeyDown={(event) => {
                    if (
                      event.key ===
                        "Enter" &&
                      !event.shiftKey
                    ) {
                      event.preventDefault();

                      void handleSend();
                    }
                  }}
                  placeholder="Ask SupportMind anything..."
                  rows={2}
                  disabled={
                    isProcessing
                  }
                  className="min-h-12 flex-1 resize-none border-0 bg-transparent px-3 py-2 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-60"
                />

                <button
                  type="button"
                  onClick={() =>
                    void handleSend()
                  }
                  disabled={
                    !message.trim() ||
                    isProcessing
                  }
                  className="mb-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white transition-all duration-200 hover:bg-blue-500 hover:shadow-lg hover:shadow-blue-500/20 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {isProcessing ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </button>
              </div>

              <div className="flex items-center justify-between gap-3 px-3 pb-1 pt-1">
                <span className="text-[10px] text-muted-foreground">
                  Enter to send · Shift + Enter for a new line
                </span>

                <span className="flex shrink-0 items-center gap-1 text-[10px] text-muted-foreground">
                  <Zap className="h-3 w-3" />
                  AI-assisted
                </span>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}