
import { apiGet } from "@/lib/api/client";

export type EvaluationMetric = {
  score: number;
  threshold: number;
  success: boolean;
  reason?: string | null;
};

export type DeepEvalResult = {
  evaluation: {
    framework: string;
    evaluation_type: string;
  };
  model: {
    provider: string;
    name: string;
  };
  input: {
    query: string;
  };
  output: {
    answer: string;
  };
  retrieval: {
    context_count: number;
    context: string[];
  };
  grounding: {
    status: string;
    valid: boolean;
    cited_evidence_ids: string[];
  };
  metrics: {
    faithfulness: EvaluationMetric;
    answer_relevancy: EvaluationMetric;
    [key: string]: EvaluationMetric;
  };
  metadata: {
    qdrant_collection?: string;
    model?: string;
    provider?: string;
    top_k?: number;
    reranking?: boolean;
    metrics_evaluated?: string[];
    [key: string]: unknown;
  };
};

export type EvaluationSummary = {
  status: "available" | "not_available" | string;
  framework: string | null;
  evaluation_type: string | null;
  model: DeepEvalResult["model"] | null;
  metrics: DeepEvalResult["metrics"] | null;
};

export type DeepEvalRawResult = {
  available: boolean;
  content: string | null;
  parsed_metrics?: {
    faithfulness: number | null;
    answer_relevancy: number | null;
  };
};


export function getEvaluationSummary() {
  return apiGet<EvaluationSummary>("/api/v1/evaluation/summary");
}

export function getDeepEvalResult() {
  return apiGet<DeepEvalResult>("/api/v1/evaluation/deepeval");
}

export function getDeepEvalRawResult() {
  return apiGet<DeepEvalRawResult>("/api/v1/evaluation/deepeval/raw");
}