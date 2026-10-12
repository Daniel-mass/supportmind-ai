import { useQuery } from "@tanstack/react-query";

import { getOrganizationIntelligence } from "@/lib/api/organization";

export function useOrganizationIntelligence() {
  return useQuery({
    queryKey: ["organization-intelligence"],
    queryFn: getOrganizationIntelligence,
    staleTime: 30_000,
  });
}