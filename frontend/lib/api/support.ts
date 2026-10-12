import { apiPost } from "@/lib/api/client";

export type SupportAssistRequest = {
  request: string;
};

export type SupportPrediction = {
  value: string;
  confidence: number;
  source: string;
  metadata: Record<string, unknown>;
};

export type SupportPredictions = {
  category: SupportPrediction | null;
  priority: SupportPrediction | null;
  sentiment: SupportPrediction | null;
  complexity: SupportPrediction | null;
  topic: SupportPrediction | null;
};

export type SupportRetrieval = {
  strategy: string;
  evidence_count: number;
  grounding_status: string;
  grounding_valid: boolean;
  provider: string;
  model_name: string;
  latency_ms: number;
};

export type SupportEvidence = {
  evidence_id: string;
  source: string;
  content: string;
  provenance: Record<string, unknown>;
};

export type SupportAgentResult = {
  agent_name: string;
  success: boolean;
  result: Record<string, unknown>;
  evidence_ids: string[];
  error: string | null;
};

export type SupportAgents = Record<string, SupportAgentResult>;

export type SupportWorkflowError = {
  source: string;
  error_type: string;
  message: string;
  recoverable: boolean;
};

export type SupportWorkflow = {
  status: string;
  current_step: string;
  completed_capabilities: string[];
  iteration_count: number;
  errors: SupportWorkflowError[];
};

export type SupportIntelligence = {
  predictions: SupportPredictions;
  retrieval: SupportRetrieval;
  evidence: SupportEvidence[];
  agents: SupportAgents;
  workflow: SupportWorkflow;
};

export type SupportAssistResponse = {
  request: string;
  response: {
    text: string;
  };
  intelligence: SupportIntelligence;
};

export function postSupportAssist(request: string) {
  return apiPost<SupportAssistRequest, SupportAssistResponse>(
    "/api/v1/support/assist",
    { request },
  );
}