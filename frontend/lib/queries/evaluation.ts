
import { useQuery } from "@tanstack/react-query";

import {
  getDeepEvalRawResult,
  getDeepEvalResult,
  getEvaluationSummary,
} from "@/lib/api/evaluation";

export function useEvaluationSummary() {
  return useQuery({
    queryKey: ["evaluation-summary"],
    queryFn: getEvaluationSummary,
    retry: 1,
  });
}

export function useDeepEvalResult() {
  return useQuery({
    queryKey: ["deepeval-result"],
    queryFn: getDeepEvalResult,
    retry: 1,
  });
}

export function useDeepEvalRawResult(enabled = true) {
  return useQuery({
    queryKey: ["deepeval-raw-result"],
    queryFn: getDeepEvalRawResult,
    enabled,
    retry: 1,
  });
}

