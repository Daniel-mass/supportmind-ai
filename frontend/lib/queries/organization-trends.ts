
import { useQuery } from "@tanstack/react-query";

import { getOrganizationTrends } from "@/lib/api/trends";

export function useOrganizationTrends() {
  return useQuery({
    queryKey: ["organization-trends"],
    queryFn: getOrganizationTrends,
    staleTime: 30_000,
  });
}