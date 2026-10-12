import { useMutation } from "@tanstack/react-query";

import { postSupportAssist } from "@/lib/api/support";

export function useSupportAssist() {
  return useMutation({
    mutationFn: (request: string) => postSupportAssist(request),
  });
}