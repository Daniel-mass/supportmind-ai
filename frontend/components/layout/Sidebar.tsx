"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  Bot,
  Brain,
  Building2,
  ChartNoAxesCombined,
  CircleGauge,
  ClipboardList,
  Headphones,
  MessageSquare,
  Network,
  Search,
  Settings2,
  ShieldCheck,
} from "lucide-react";

type NavItem = {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
};

type NavSection = {
  label: string;
  items: NavItem[];
};

const navigation: NavSection[] = [
  {
    label: "Workspace",
    items: [
      {
        label: "Overview",
        href: "/",
        icon: CircleGauge,
      },
      {
        label: "AI Support",
        href: "/ai-support",
        icon: MessageSquare,
      },
    ],
  },
  {
    label: "Intelligence",
    items: [
      {
        label: "Ticket Intelligence",
        href: "/intelligence/tickets",
        icon: Brain,
      },
      {
        label: "Knowledge & RAG",
        href: "/intelligence/knowledge",
        icon: Search,
      },
      {
        label: "Graph Intelligence",
        href: "/intelligence/graph",
        icon: Network,
      },
    ],
  },
  {
    label: "Agent Operations",
    items: [
      {
        label: "Agent Workflow",
        href: "/agents/workflow",
        icon: Activity,
      },
      {
        label: "Agent Intelligence",
        href: "/agents/intelligence",
        icon: Bot,
      },
    ],
  },
  {
    label: "Organization",
    items: [
      {
        label: "Trends",
        href: "/organization/trends",
        icon: ChartNoAxesCombined,
      },
      {
        label: "Emerging Issues",
        href: "/organization/emerging-issues",
        icon: ClipboardList,
      },
      {
        label: "Operational Insights",
        href: "/organization/insights",
        icon: Building2,
      },
    ],
  },
  {
    label: "Evaluation",
    items: [
      {
        label: "RAG Evaluation",
        href: "/evaluation/rag",
        icon: ShieldCheck,
      },
      {
        label: "Model Evaluation",
        href: "/evaluation/models",
        icon: BarChart3,
      },
    ],
  },
  {
    label: "Observability",
    items: [
      {
        label: "Phoenix",
        href: "/observability/phoenix",
        icon: Activity,
      },
      {
        label: "MLflow",
        href: "/observability/mlflow",
        icon: BarChart3,
      },
      {
        label: "System Health",
        href: "/observability/health",
        icon: Settings2,
      },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-screen w-72 shrink-0 flex-col border-r bg-background">
      <div className="flex h-16 items-center gap-3 border-b px-5">
        <div className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground">
          <Headphones className="size-5" />
        </div>

        <div className="min-w-0">
          <p className="truncate text-sm font-semibold tracking-tight">
            SupportMind AI
          </p>
          <p className="text-xs text-muted-foreground">
            Support Intelligence
          </p>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-4">
        <div className="space-y-6">
          {navigation.map((section) => (
            <div key={section.label}>
              <p className="mb-2 px-3 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                {section.label}
              </p>

              <div className="space-y-1">
                {section.items.map((item) => {
                  const Icon = item.icon;
                  const isActive =
                    pathname === item.href ||
                    (item.href !== "/" &&
                      pathname.startsWith(`${item.href}/`));

                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${
                        isActive
                          ? "bg-accent font-medium text-accent-foreground"
                          : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
                      }`}
                    >
                      <Icon className="size-4 shrink-0" />
                      <span className="truncate">{item.label}</span>
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </nav>

      <div className="border-t p-3">
        <div className="rounded-lg bg-muted/50 px-3 py-2">
          <p className="text-xs font-medium">SupportMind Engine</p>
          <p className="mt-0.5 text-[11px] text-muted-foreground">
            Intelligence platform
          </p>
        </div>
      </div>
    </aside>
  );
}