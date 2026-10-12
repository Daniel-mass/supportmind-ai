import type { ReactNode } from "react";

type MetricCardProps = {
  label: string;
  value: string | number;
  description?: string;
  icon?: ReactNode;
};

export function MetricCard({
  label,
  value,
  description,
  icon,
}: MetricCardProps) {
  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-sm font-medium text-muted-foreground">
            {label}
          </p>

          <p className="mt-2 truncate text-2xl font-semibold tracking-tight">
            {value}
          </p>

          {description && (
            <p className="mt-1 text-xs text-muted-foreground">
              {description}
            </p>
          )}
        </div>

        {icon && (
          <div className="shrink-0 rounded-lg border bg-muted/40 p-2">
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}