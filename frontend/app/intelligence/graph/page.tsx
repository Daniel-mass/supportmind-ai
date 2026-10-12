
"use client";

import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
  type FormEvent,
} from "react";
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  Handle,
  Position,
  MarkerType,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Activity,
  AlertCircle,
  ArrowDownRight,
  ArrowUpRight,
  Boxes,
  Check,
  ChevronDown,
  CircleDot,
  Code2,
  Database,
  ExternalLink,
  GitBranch,
  Network,
  Play,
  RefreshCw,
  Search,
  Server,
  Table2,
  Waypoints,
  X,
} from "lucide-react";
import { apiGet, apiPost } from "@/lib/api/client";

/* -------------------------------------------------------------------------- */
/* Types                                                                      */
/* -------------------------------------------------------------------------- */

type GraphNode = {
  id: string;
  labels: string[];
  properties: Record<string, unknown>;
};

type GraphEdge = {
  id: string;
  source: string;
  target: string;
  type: string;
  properties: Record<string, unknown>;
};

type GraphOverview = {
  healthy: boolean;
  database: string;
  node_count: number;
  relationship_count: number;
  node_labels: { label: string; count: number }[];
  relationship_types: { type: string; count: number }[];
};

type GraphExploreResponse = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  returned_nodes: number;
  returned_edges: number;
  limit: number;
};

type ConnectedRelationship = {
  id: string;
  type: string;
  direction: string;
  connected_node_id: string;
  connected_node_labels: string[];
  connected_node_properties: Record<string, unknown>;
  properties: Record<string, unknown>;
};

type GraphNodeDetails = {
  node: GraphNode;
  relationships: ConnectedRelationship[];
};

type SchemaNodeLabel = {
  label: string;
  node_count: number;
};

type SchemaRelationship = {
  source_labels: string[];
  relationship_type: string;
  target_labels: string[];
  relationship_count: number;
};

type GraphSchema = {
  node_labels: SchemaNodeLabel[];
  relationships: SchemaRelationship[];
};

type CypherResponse = {
  columns: string[];
  rows: Record<string, unknown>[];
  row_count: number;
  truncated: boolean;
  limit: number;
  execution_ms: number;
};

type EntityData = {
  label: string;
  title: string;
  subtitle: string;
  selected?: boolean;
};

type FlowNode = Node<EntityData, "entity">;

type ResultEntity = {
  id: string;
  label: string;
  title: string;
  subtitle: string;
};

type GraphDisplay = {
  nodes: FlowNode[];
  edges: Edge[];
};

/* -------------------------------------------------------------------------- */
/* Entity styling                                                             */
/* -------------------------------------------------------------------------- */

const LABEL_STYLES: Record<
  string,
  { border: string; background: string; text: string; dot: string }
> = {
  Ticket: {
    border: "#60a5fa",
    background: "#eff6ff",
    text: "#1d4ed8",
    dot: "#3b82f6",
  },
  Customer: {
    border: "#2dd4bf",
    background: "#f0fdfa",
    text: "#0f766e",
    dot: "#14b8a6",
  },
  Agent: {
    border: "#34d399",
    background: "#ecfdf5",
    text: "#047857",
    dot: "#10b981",
  },
  WorkflowEvent: {
    border: "#a78bfa",
    background: "#f5f3ff",
    text: "#6d28d9",
    dot: "#8b5cf6",
  },
  IssueType: {
    border: "#fb923c",
    background: "#fff7ed",
    text: "#c2410c",
    dot: "#f97316",
  },
  Priority: {
    border: "#f87171",
    background: "#fef2f2",
    text: "#b91c1c",
    dot: "#ef4444",
  },
  Status: {
    border: "#fbbf24",
    background: "#fffbeb",
    text: "#92400e",
    dot: "#f59e0b",
  },
  Team: {
    border: "#22d3ee",
    background: "#ecfeff",
    text: "#0e7490",
    dot: "#06b6d4",
  },
  Product: {
    border: "#818cf8",
    background: "#eef2ff",
    text: "#4338ca",
    dot: "#6366f1",
  },
  TicketType: {
    border: "#f472b6",
    background: "#fdf2f8",
    text: "#9d174d",
    dot: "#ec4899",
  },
  Channel: {
    border: "#a3a3a3",
    background: "#f5f5f5",
    text: "#404040",
    dot: "#737373",
  },
};

const FALLBACK_STYLE = {
  border: "#94a3b8",
  background: "#f8fafc",
  text: "#334155",
  dot: "#64748b",
};

const EXAMPLE_QUERIES = [
  {
    label: "Tickets by priority",
    description: "Inspect ticket nodes and their priority connections.",
    query:
      "MATCH (t:Ticket)-[r:HAS_PRIORITY]->(p:Priority)\nRETURN t, r, p",
  },
  {
    label: "Customer ticket history",
    description: "Explore customers and the tickets they created.",
    query:
      "MATCH (c:Customer)-[r:CREATED]->(t:Ticket)\nRETURN c, r, t",
  },
  {
    label: "Agent handoffs",
    description: "Inspect handoff links between agents.",
    query:
      "MATCH (a:Agent)-[r:HANDOFF_TO]->(b:Agent)\nRETURN a, r, b",
  },
  {
    label: "Workflow sequence",
    description: "Inspect workflow events and their next-event links.",
    query:
      "MATCH (a:WorkflowEvent)-[r:NEXT]->(b:WorkflowEvent)\nRETURN a, r, b",
  },
  {
    label: "Node label counts",
    description: "Count nodes grouped by their label.",
    query:
      "MATCH (n)\nWITH labels(n) AS node_labels, count(*) AS total\nRETURN node_labels, total\nORDER BY total DESC",
  },
];

/* -------------------------------------------------------------------------- */
/* Formatting helpers                                                         */
/* -------------------------------------------------------------------------- */

function formatValue(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "—";

  if (typeof value === "object") {
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }

  return String(value);
}

function formatNumber(value: number | undefined): string {
  return typeof value === "number" ? value.toLocaleString() : "—";
}

function getNodeTitle(node: GraphNode): string {
  const props = node.properties;

  for (const key of [
    "name",
    "ticket_id",
    "agent_id",
    "customer_id",
    "node_id",
    "title",
    "id",
  ]) {
    const value = props[key];

    if (value !== undefined && value !== null && String(value).trim()) {
      return String(value);
    }
  }

  return `${node.labels[0] ?? "Node"} · ${node.id.slice(-8)}`;
}

function getNodeSubtitle(node: GraphNode): string {
  const props = node.properties;

  for (const key of [
    "status",
    "team",
    "priority",
    "issue_type",
    "channel",
    "product",
    "subject",
  ]) {
    const value = props[key];

    if (value !== undefined && value !== null && String(value).trim()) {
      return `${key.replaceAll("_", " ")}: ${String(value)}`;
    }
  }

  return node.labels.join(" · ") || "Graph entity";
}

function getResultLabel(
  value: Record<string, unknown>,
  alias: string,
  query: string,
): string {
  const escapedAlias = alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = query.match(
    new RegExp(`\\(\\s*${escapedAlias}\\s*:\\s*(\\w+)`, "i"),
  );

  if (match?.[1]) return match[1];

  if ("ticket_id" in value || "subject" in value) return "Ticket";
  if ("agent_id" in value) return "Agent";
  if ("customer_id" in value) return "Customer";

  const nodeId = String(value.node_id ?? "");
  const prefix = nodeId.split(":")[0];

  if (prefix) {
    const known = Object.keys(LABEL_STYLES).find(
      (label) => label.toLowerCase() === prefix.toLowerCase(),
    );
    if (known) return known;
  }

  return "Result";
}

function getResultTitle(value: Record<string, unknown>): string {
  for (const key of [
    "name",
    "ticket_id",
    "agent_id",
    "customer_id",
    "title",
    "subject",
    "node_id",
    "id",
  ]) {
    if (
      value[key] !== undefined &&
      value[key] !== null &&
      String(value[key]).trim()
    ) {
      return String(value[key]);
    }
  }

  return "Graph entity";
}

function getResultSubtitle(
  value: Record<string, unknown>,
  label: string,
): string {
  for (const key of ["status", "priority", "issue_type", "channel", "subject"]) {
    if (value[key] !== undefined && value[key] !== null) {
      return `${key.replaceAll("_", " ")}: ${String(value[key])}`;
    }
  }

  return label;
}

/* -------------------------------------------------------------------------- */
/* Force-directed layout                                                      */
/*                                                                            */
/* No grid positioning: connected nodes attract one another, while all nodes  */
/* repel one another. A small deterministic starting circle keeps the layout  */
/* stable between renders.                                                    */
/* -------------------------------------------------------------------------- */

type LayoutNode = {
  id: string;
  x: number;
  y: number;
};

function forceDirectedPositions(
  ids: string[],
  links: { source: string; target: string }[],
): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const count = ids.length;

  if (count === 0) return positions;

  if (count === 1) {
    positions.set(ids[0], { x: 0, y: 0 });
    return positions;
  }

  const indexById = new Map(ids.map((id, index) => [id, index]));
  const radius = Math.max(140, Math.sqrt(count) * 88);

  const points: LayoutNode[] = ids.map((id, index) => {
    const angle = index * 2.399963229728653;
    const spread = radius * Math.sqrt((index + 0.5) / count);

    return {
      id,
      x: Math.cos(angle) * spread,
      y: Math.sin(angle) * spread,
    };
  });

  const springs = links
    .map((link) => ({
      source: indexById.get(link.source),
      target: indexById.get(link.target),
    }))
    .filter(
      (link): link is { source: number; target: number } =>
        link.source !== undefined &&
        link.target !== undefined &&
        link.source !== link.target,
    );

  const idealLength = Math.max(105, Math.min(190, 105 + count * 0.65));
  const iterations = count > 120 ? 50 : 90;
  const repulsion = 6800;
  const velocities = points.map(() => ({ x: 0, y: 0 }));

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const forces = points.map(() => ({ x: 0, y: 0 }));

    // Repulsion keeps nodes apart and prevents a rigid grid-like appearance.
    for (let i = 0; i < count; i += 1) {
      for (let j = i + 1; j < count; j += 1) {
        let dx = points[j].x - points[i].x;
        let dy = points[j].y - points[i].y;
        let distanceSquared = dx * dx + dy * dy;

        if (distanceSquared < 25) {
          dx += (i + 1) * 0.37;
          dy += (j + 1) * 0.29;
          distanceSquared = dx * dx + dy * dy;
        }

        const distance = Math.sqrt(distanceSquared);
        const force = repulsion / Math.max(distanceSquared, 25);
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;

        forces[i].x -= fx;
        forces[i].y -= fy;
        forces[j].x += fx;
        forces[j].y += fy;
      }
    }

    // Relationships act like springs, drawing connected nodes together.
    for (const spring of springs) {
      const source = points[spring.source];
      const target = points[spring.target];

      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const distance = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const displacement = distance - idealLength;
      const force = displacement * 0.018;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;

      forces[spring.source].x += fx;
      forces[spring.source].y += fy;
      forces[spring.target].x -= fx;
      forces[spring.target].y -= fy;
    }

    // Damping keeps the simulation stable and prevents excessive movement.
    const cooling = 1 - iteration / (iterations + 12);

    for (let i = 0; i < count; i += 1) {
      velocities[i].x =
        (velocities[i].x + forces[i].x * 0.025) * 0.78;
      velocities[i].y =
        (velocities[i].y + forces[i].y * 0.025) * 0.78;

      const speed = Math.hypot(velocities[i].x, velocities[i].y);
      const maxSpeed = 12 * cooling;

      if (speed > maxSpeed && speed > 0) {
        velocities[i].x = (velocities[i].x / speed) * maxSpeed;
        velocities[i].y = (velocities[i].y / speed) * maxSpeed;
      }

      points[i].x += velocities[i].x;
      points[i].y += velocities[i].y;
    }
  }

  // Centre the graph in the React Flow canvas.
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const centreX = (minX + maxX) / 2;
  const centreY = (minY + maxY) / 2;

  points.forEach((point) => {
    positions.set(point.id, {
      x: point.x - centreX,
      y: point.y - centreY,
    });
  });

  return positions;
}

/* -------------------------------------------------------------------------- */
/* Neo4j-style circular entity node                                            */
/* -------------------------------------------------------------------------- */

const EntityNode = memo(function EntityNode({
  data,
}: NodeProps<FlowNode>) {
  const style = LABEL_STYLES[data.label] ?? FALLBACK_STYLE;

  return (
    <div className="group flex w-[124px] flex-col items-center">
      <Handle
        type="target"
        position={Position.Top}
        className="!h-2 !w-2 !border-2 !border-white !bg-slate-400"
      />

      <div
        className="flex h-[82px] w-[82px] items-center justify-center rounded-full border-[3px] shadow-md transition duration-150 group-hover:scale-105 group-hover:shadow-lg"
        style={{
          borderColor: style.border,
          background: style.background,
          boxShadow: `0 3px 14px ${style.dot}20`,
        }}
        title={`${data.label}: ${data.title}`}
      >
        <div
          className="flex h-[68px] w-[68px] flex-col items-center justify-center overflow-hidden rounded-full border border-white/80 px-1 text-center"
          style={{ background: style.background }}
        >
          <span
            className="max-w-full truncate text-[9px] font-bold uppercase tracking-wide"
            style={{ color: style.text }}
          >
            {data.label}
          </span>
          <span className="mt-1 max-w-full truncate text-[10px] font-semibold text-slate-800">
            {data.title}
          </span>
        </div>
      </div>

      <span className="mt-2 max-w-[124px] truncate rounded-md border border-slate-200 bg-white/95 px-2 py-1 text-center text-[10px] text-slate-600 shadow-sm">
        {data.subtitle}
      </span>

      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-2 !w-2 !border-2 !border-white !bg-slate-400"
      />
    </div>
  );
});

const NODE_TYPES = { entity: EntityNode };

/* -------------------------------------------------------------------------- */
/* Graph construction                                                         */
/* -------------------------------------------------------------------------- */

function createFlowNodes(graphNodes: GraphNode[], graphEdges: GraphEdge[]): FlowNode[] {
  const positions = forceDirectedPositions(
    graphNodes.map((node) => node.id),
    graphEdges.map((edge) => ({
      source: edge.source,
      target: edge.target,
    })),
  );

  return graphNodes.map((node) => {
    const label = node.labels[0] ?? "Node";
    const position = positions.get(node.id) ?? { x: 0, y: 0 };

    return {
      id: node.id,
      type: "entity",
      position,
      data: {
        label,
        title: getNodeTitle(node),
        subtitle: getNodeSubtitle(node),
      },
    };
  });
}

function createFlowEdges(graphEdges: GraphEdge[]): Edge[] {
  return graphEdges.map((edge, index) => ({
    id: edge.id || `edge-${index}`,
    source: edge.source,
    target: edge.target,
    type: "default",
    label: edge.type,
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: edge.type === "HANDOFF_TO" ? "#8b5cf6" : "#94a3b8",
    },
    style: {
      stroke: edge.type === "HANDOFF_TO" ? "#8b5cf6" : "#94a3b8",
      strokeWidth: edge.type === "HANDOFF_TO" ? 2 : 1.5,
    },
    labelStyle: {
      fill: "#475569",
      fontSize: 10,
      fontWeight: 600,
    },
    labelBgStyle: { fill: "#ffffff", fillOpacity: 0.94 },
    labelBgPadding: [5, 3] as [number, number],
    labelBgBorderRadius: 4,
  }));
}

function createResultGraph(
  result: CypherResponse | null,
  query: string,
): GraphDisplay {
  if (!result) return { nodes: [], edges: [] };

  const entityMap = new Map<string, ResultEntity>();
  const relationships: {
    id: string;
    source: string;
    target: string;
    type: string;
  }[] = [];

  const addNode = (
    raw: Record<string, unknown>,
    alias: string,
  ): string | null => {
    const rawId = raw.node_id ?? raw.id;
    if (rawId === undefined || rawId === null) return null;

    const id = String(rawId);
    const label = getResultLabel(raw, alias, query);

    entityMap.set(id, {
      id,
      label,
      title: getResultTitle(raw),
      subtitle: getResultSubtitle(raw, label),
    });

    return id;
  };

  result.rows.forEach((row, rowIndex) => {
    Object.entries(row).forEach(([column, value]) => {
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        return;
      }

      const record = value as Record<string, unknown>;

      // Neo4j relationships are serialized with their endpoint identifiers.
      const startId =
        record.start_node_id ?? record.startNodeId ?? record.start;
      const endId =
        record.end_node_id ?? record.endNodeId ?? record.end;
      const relationshipType =
        record.relationship_type ?? record.type ?? record.label;

      if (
        startId !== undefined &&
        startId !== null &&
        endId !== undefined &&
        endId !== null &&
        relationshipType !== undefined
      ) {
        const source = String(startId);
        const target = String(endId);

        if (!entityMap.has(source)) {
          entityMap.set(source, {
            id: source,
            label: "Node",
            title: source,
            subtitle: "Relationship endpoint",
          });
        }

        if (!entityMap.has(target)) {
          entityMap.set(target, {
            id: target,
            label: "Node",
            title: target,
            subtitle: "Relationship endpoint",
          });
        }

        relationships.push({
          id: `result-edge-${rowIndex}-${column}-${source}-${target}`,
          source,
          target,
          type: String(relationshipType),
        });

        return;
      }

      if (
        "node_id" in record ||
        "ticket_id" in record ||
        "agent_id" in record ||
        "customer_id" in record
      ) {
        addNode(record, column);
      }
    });
  });

  const nodesWithoutPositions = Array.from(entityMap.values());
  const positions = forceDirectedPositions(
    nodesWithoutPositions.map((node) => node.id),
    relationships,
  );

  const nodes: FlowNode[] = nodesWithoutPositions.map((node) => ({
    id: node.id,
    type: "entity",
    position: positions.get(node.id) ?? { x: 0, y: 0 },
    data: {
      label: node.label,
      title: node.title,
      subtitle: node.subtitle,
    },
  }));

  const edges: Edge[] = relationships.map((relationship) => ({
    id: relationship.id,
    source: relationship.source,
    target: relationship.target,
    type: "default",
    label: relationship.type,
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: "#8b5cf6",
    },
    style: { stroke: "#8b5cf6", strokeWidth: 1.8 },
    labelStyle: { fill: "#475569", fontSize: 10, fontWeight: 600 },
    labelBgStyle: { fill: "#ffffff", fillOpacity: 0.94 },
    labelBgPadding: [5, 3] as [number, number],
    labelBgBorderRadius: 4,
  }));

  return { nodes, edges };
}

/* -------------------------------------------------------------------------- */
/* Reusable UI                                                                */
/* -------------------------------------------------------------------------- */

function StatCard({
  title,
  value,
  caption,
  icon: Icon,
}: {
  title: string;
  value: string;
  caption: string;
  icon: typeof Database;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-medium text-slate-500">{title}</p>
          <p className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">
            {value}
          </p>
          <p className="mt-1 text-xs text-slate-500">{caption}</p>
        </div>
        <div className="rounded-lg bg-slate-100 p-2.5 text-slate-600">
          <Icon size={18} />
        </div>
      </div>
    </div>
  );
}

function SectionHeading({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: typeof Database;
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
      <div className="flex items-start gap-2.5">
        <div className="mt-0.5 rounded-lg bg-slate-100 p-2 text-slate-600">
          <Icon size={16} />
        </div>
        <div>
          <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          {description && (
            <p className="mt-1 text-xs leading-5 text-slate-500">
              {description}
            </p>
          )}
        </div>
      </div>
      {action}
    </div>
  );
}

function GraphCanvas({
  graph,
  onNodeClick,
  height = "560px",
}: {
  graph: GraphDisplay;
  onNodeClick?: (nodeId: string) => void;
  height?: string;
}) {
  return (
    <div className="w-full bg-slate-50/80" style={{ height }}>
      <ReactFlow
        nodes={graph.nodes}
        edges={graph.edges}
        nodeTypes={NODE_TYPES}
        fitView
        fitViewOptions={{ padding: 0.24, maxZoom: 1 }}
        minZoom={0.08}
        maxZoom={2}
        nodesConnectable={false}
        nodesDraggable
        elementsSelectable
        onNodeClick={(_, node) => onNodeClick?.(node.id)}
        defaultEdgeOptions={{
          type: "default",
          markerEnd: { type: MarkerType.ArrowClosed },
        }}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#dbe3ed" gap={24} size={1} />
        <Controls position="bottom-left" />
        <MiniMap
          position="bottom-right"
          pannable
          zoomable
          
          nodeColor={(node) => {
           const label = node.data.label;

            if (typeof label !== "string") {
              return FALLBACK_STYLE.dot;}

            return (
              LABEL_STYLES[label as keyof typeof LABEL_STYLES]?.dot ??
              FALLBACK_STYLE.dot
  );
}}

          maskColor="rgba(241, 245, 249, 0.65)"
        />
      </ReactFlow>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Page                                                                       */
/* -------------------------------------------------------------------------- */

export default function GraphIntelligencePage() {
  const [overview, setOverview] = useState<GraphOverview | null>(null);
  const [graph, setGraph] = useState<GraphExploreResponse | null>(null);
  const [schema, setSchema] = useState<GraphSchema | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [details, setDetails] = useState<GraphNodeDetails | null>(null);

  const [labelFilter, setLabelFilter] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [limit, setLimit] = useState("100");

  const [query, setQuery] = useState(EXAMPLE_QUERIES[0].query);
  const [queryLimit, setQueryLimit] = useState("50");
  const [queryResult, setQueryResult] = useState<CypherResponse | null>(null);
  const [queryError, setQueryError] = useState("");
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryView, setQueryView] = useState<"table" | "graph">("table");

  const [showAllProperties, setShowAllProperties] = useState(false);
  const [activeTab, setActiveTab] = useState<"explorer" | "cypher">("explorer");

  const [loadingOverview, setLoadingOverview] = useState(true);
  const [loadingGraph, setLoadingGraph] = useState(true);
  const [loadingSchema, setLoadingSchema] = useState(true);
  const [loadingDetails, setLoadingDetails] = useState(false);

  const [overviewError, setOverviewError] = useState("");
  const [graphError, setGraphError] = useState("");
  const [schemaError, setSchemaError] = useState("");
  const [detailsError, setDetailsError] = useState("");

  const loadOverview = useCallback(async () => {
    setLoadingOverview(true);
    setOverviewError("");

    try {
      const result = await apiGet<GraphOverview>("/api/v1/graph/overview");
      setOverview(result);
    } catch (error) {
      setOverviewError(
        error instanceof Error
          ? error.message
          : "Unable to load graph overview.",
      );
    } finally {
      setLoadingOverview(false);
    }
  }, []);

  const loadSchema = useCallback(async () => {
    setLoadingSchema(true);
    setSchemaError("");

    try {
      const result = await apiGet<GraphSchema>("/api/v1/graph/schema");
      setSchema(result);
    } catch (error) {
      setSchema(null);
      setSchemaError(
        error instanceof Error
          ? error.message
          : "Unable to load graph schema.",
      );
    } finally {
      setLoadingSchema(false);
    }
  }, []);

  const loadGraph = useCallback(async () => {
    setLoadingGraph(true);
    setGraphError("");
    setSelectedNode(null);
    setDetails(null);
    setDetailsError("");

    const params = new URLSearchParams({ limit });
    if (labelFilter) params.set("label", labelFilter);
    if (search.trim()) params.set("search", search.trim());

    try {
      const result = await apiGet<GraphExploreResponse>(
        `/api/v1/graph/explore?${params.toString()}`,
      );
      setGraph(result);
    } catch (error) {
      setGraph(null);
      setGraphError(
        error instanceof Error
          ? error.message
          : "Unable to explore the graph.",
      );
    } finally {
      setLoadingGraph(false);
    }
  }, [labelFilter, limit, search]);

  useEffect(() => {
    void loadOverview();
    void loadSchema();
  }, [loadOverview, loadSchema]);

  useEffect(() => {
    void loadGraph();
  }, [loadGraph]);

  const loadNodeDetails = useCallback(async (node: GraphNode) => {
    setSelectedNode(node);
    setDetails(null);
    setDetailsError("");
    setLoadingDetails(true);

    try {
      const result = await apiGet<GraphNodeDetails>(
        `/api/v1/graph/nodes/${encodeURIComponent(node.id)}`,
      );
      setDetails(result);
    } catch (error) {
      setDetailsError(
        error instanceof Error
          ? error.message
          : "Unable to load node details.",
      );
    } finally {
      setLoadingDetails(false);
    }
  }, []);

  const flowGraph = useMemo(
    () => ({
      nodes: createFlowNodes(graph?.nodes ?? [], graph?.edges ?? []),
      edges: createFlowEdges(graph?.edges ?? []),
    }),
    [graph],
  );

  const queryGraph = useMemo(
    () => createResultGraph(queryResult, query),
    [queryResult, query],
  );

  const submitSearch = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSearch(searchInput.trim());
  };

  const resetFilters = () => {
    setLabelFilter("");
    setSearchInput("");
    setSearch("");
    setLimit("100");
  };

  const runQuery = async () => {
    setQueryLoading(true);
    setQueryError("");
    setQueryResult(null);

    try {
      const result = await apiPost<
        { query: string; limit: number },
        CypherResponse
      >("/api/v1/graph/cypher", {
        query: query.trim(),
        limit: Number(queryLimit),
      });

      setQueryResult(result);
      setQueryView("table");
    } catch (error) {
      setQueryError(
        error instanceof Error ? error.message : "The Cypher query failed.",
      );
    } finally {
      setQueryLoading(false);
    }
  };

  const useExample = (exampleQuery: string) => {
    setQuery(exampleQuery);
    setQueryError("");
    setQueryResult(null);
  };

  const openNeo4jBrowser = () => {
    window.open(
      "http://localhost:7474/browser/",
      "_blank",
      "noopener,noreferrer",
    );
  };

  const visibleProperties = useMemo(() => {
    const entries = Object.entries(details?.node.properties ?? {});
    return showAllProperties ? entries : entries.slice(0, 12);
  }, [details?.node.properties, showAllProperties]);

  const schemaNodeCount =
    schema?.node_labels.length ?? overview?.node_labels.length ?? 0;
  const schemaRelationshipCount =
    schema?.relationships.length ?? overview?.relationship_types.length ?? 0;

  return (
    <main className="space-y-6 p-4 md:p-6">
      <section className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
            <Network size={14} />
            Intelligence / Graph
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-900 md:text-3xl">
            Graph Intelligence
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
            Explore live entities, connected relationships, the Neo4j schema,
            and read-only Cypher results through an interactive graph canvas.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium ${
              overview?.healthy
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : overview && !overview.healthy
                  ? "border-rose-200 bg-rose-50 text-rose-700"
                  : "border-slate-200 bg-slate-50 text-slate-600"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                overview?.healthy
                  ? "bg-emerald-500"
                  : "bg-current opacity-60"
              }`}
            />
            {loadingOverview
              ? "Checking connection"
              : overview?.healthy
                ? "Neo4j connected"
                : "Connection unavailable"}
          </span>

          <button
            type="button"
            onClick={() => {
              void loadOverview();
              void loadSchema();
              void loadGraph();
            }}
            disabled={loadingOverview || loadingGraph || loadingSchema}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RefreshCw
              size={15}
              className={
                loadingOverview || loadingGraph || loadingSchema
                  ? "animate-spin"
                  : ""
              }
            />
            Refresh
          </button>

          <button
            type="button"
            onClick={openNeo4jBrowser}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3 py-2 text-sm font-medium text-white transition hover:bg-slate-700"
          >
            <ExternalLink size={14} />
            Neo4j Browser
          </button>
        </div>
      </section>

      {(overviewError || schemaError) && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
          <AlertCircle size={18} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">
              Some graph metadata could not load
            </p>
            {overviewError && (
              <p className="mt-1 break-words">{overviewError}</p>
            )}
            {schemaError && (
              <p className="mt-1 break-words">{schemaError}</p>
            )}
          </div>
        </div>
      )}

      <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          title="Total nodes"
          value={loadingOverview ? "…" : formatNumber(overview?.node_count)}
          caption="Entities stored in Neo4j"
          icon={Boxes}
        />
        <StatCard
          title="Relationships"
          value={
            loadingOverview ? "…" : formatNumber(overview?.relationship_count)
          }
          caption="Connections across the graph"
          icon={GitBranch}
        />
        <StatCard
          title="Node categories"
          value={loadingSchema ? "…" : formatNumber(schemaNodeCount)}
          caption="Labels discovered from Neo4j"
          icon={CircleDot}
        />
        <StatCard
          title="Relationship patterns"
          value={loadingSchema ? "…" : formatNumber(schemaRelationshipCount)}
          caption="Source → relationship → target"
          icon={Waypoints}
        />
      </section>

      <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 p-3">
          <button
            type="button"
            onClick={() => setActiveTab("explorer")}
            className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition ${
              activeTab === "explorer"
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            <Network size={15} />
            Relationship explorer
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("cypher")}
            className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition ${
              activeTab === "cypher"
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            <Code2 size={15} />
            Cypher workbench
          </button>
          <span className="ml-auto hidden text-[11px] text-slate-400 sm:inline">
            Read-only queries · Server-enforced result limit
          </span>
        </div>

        {activeTab === "explorer" ? (
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_320px]">
            <div className="min-w-0 border-b border-slate-200 xl:border-b-0 xl:border-r">
              <div className="border-b border-slate-200 p-4">
                <SectionHeading
                  icon={Network}
                  title="Relationship explorer"
                  description="Select a circular node to inspect its real Neo4j properties and connected relationships."
                  action={
                    <span className="inline-flex items-center gap-2 text-xs text-slate-500">
                      <Activity size={14} />
                      {loadingGraph
                        ? "Loading…"
                        : `${formatNumber(graph?.returned_nodes)} nodes · ${formatNumber(graph?.returned_edges)} edges`}
                    </span>
                  }
                />

                <form
                  onSubmit={submitSearch}
                  className="mt-4 flex flex-col gap-2 md:flex-row"
                >
                  <div className="relative min-w-0 flex-1">
                    <Search
                      size={16}
                      className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
                    />
                    <input
                      value={searchInput}
                      onChange={(event) => setSearchInput(event.target.value)}
                      placeholder="Search node properties…"
                      className="h-10 w-full rounded-lg border border-slate-200 bg-white pl-9 pr-3 text-sm text-slate-800 outline-none transition placeholder:text-slate-400 focus:border-slate-400 focus:ring-2 focus:ring-slate-100"
                    />
                  </div>

                  <select
                    value={labelFilter}
                    onChange={(event) => setLabelFilter(event.target.value)}
                    aria-label="Filter by node label"
                    className="h-10 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 outline-none focus:border-slate-400"
                  >
                    <option value="">All labels</option>
                    {(schema?.node_labels ?? overview?.node_labels ?? []).map(
                      (item) => (
                        <option value={item.label} key={item.label}>
                          {item.label}
                        </option>
                      ),
                    )}
                  </select>

                  <select
                    value={limit}
                    onChange={(event) => setLimit(event.target.value)}
                    aria-label="Maximum number of graph nodes"
                    className="h-10 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 outline-none focus:border-slate-400"
                  >
                    <option value="25">25 nodes</option>
                    <option value="50">50 nodes</option>
                    <option value="100">100 nodes</option>
                    <option value="150">150 nodes</option>
                    <option value="250">250 nodes</option>
                  </select>

                  <button
                    type="submit"
                    className="h-10 rounded-lg bg-slate-900 px-4 text-sm font-medium text-white transition hover:bg-slate-700"
                  >
                    Search
                  </button>

                  <button
                    type="button"
                    onClick={resetFilters}
                    title="Reset filters"
                    className="inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-slate-200 px-3 text-sm text-slate-600 transition hover:bg-slate-50"
                  >
                    <X size={15} />
                    Reset
                  </button>
                </form>
              </div>

              <div className="relative">
                {loadingGraph && (
                  <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/70 backdrop-blur-[1px]">
                    <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 shadow-sm">
                      <RefreshCw size={17} className="animate-spin" />
                      Loading graph data…
                    </div>
                  </div>
                )}

                {graphError ? (
                  <div className="flex h-[560px] items-center justify-center p-6">
                    <div className="max-w-md text-center">
                      <AlertCircle
                        size={28}
                        className="mx-auto text-rose-500"
                      />
                      <h3 className="mt-3 font-semibold text-slate-900">
                        Graph exploration failed
                      </h3>
                      <p className="mt-2 break-words text-sm text-slate-500">
                        {graphError}
                      </p>
                      <button
                        type="button"
                        onClick={() => void loadGraph()}
                        className="mt-4 rounded-lg bg-slate-900 px-4 py-2 text-sm text-white hover:bg-slate-700"
                      >
                        Try again
                      </button>
                    </div>
                  </div>
                ) : !loadingGraph && flowGraph.nodes.length === 0 ? (
                  <div className="flex h-[560px] items-center justify-center p-6">
                    <div className="text-center">
                      <Network
                        size={28}
                        className="mx-auto text-slate-400"
                      />
                      <h3 className="mt-3 font-semibold text-slate-900">
                        No matching nodes
                      </h3>
                      <p className="mt-1 text-sm text-slate-500">
                        Try another label or search term.
                      </p>
                    </div>
                  </div>
                ) : (
                  <GraphCanvas
                    graph={flowGraph}
                    onNodeClick={(nodeId) => {
                      const original = graph?.nodes.find(
                        (item) => item.id === nodeId,
                      );
                      if (original) void loadNodeDetails(original);
                    }}
                  />
                )}
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 bg-white px-4 py-3">
                <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                  {(schema?.node_labels ?? overview?.node_labels ?? [])
                    .slice(0, 11)
                    .map((item) => (
                      <span
                        key={item.label}
                        className="inline-flex items-center gap-1.5 text-[11px] text-slate-500"
                      >
                        <span
                          className="h-2 w-2 rounded-full"
                          style={{
                            background:
                              (LABEL_STYLES[item.label] ?? FALLBACK_STYLE).dot,
                          }}
                        />
                        {item.label}
                      </span>
                    ))}
                </div>
                <span className="text-[11px] text-slate-400">
                  Drag nodes · Scroll to zoom · Select to inspect
                </span>
              </div>
            </div>

            <aside className="min-w-0 space-y-4 p-4">
              <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
                <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3.5">
                  <div className="flex items-center gap-2">
                    <Server size={16} className="text-slate-500" />
                    <h2 className="text-sm font-semibold text-slate-900">
                      Node inspector
                    </h2>
                  </div>

                  {selectedNode && (
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedNode(null);
                        setDetails(null);
                        setDetailsError("");
                      }}
                      aria-label="Close node inspector"
                      className="rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                    >
                      <X size={16} />
                    </button>
                  )}
                </div>

                {!selectedNode ? (
                  <div className="px-5 py-10 text-center">
                    <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl bg-slate-100 text-slate-500">
                      <Network size={20} />
                    </div>
                    <p className="mt-3 text-sm font-medium text-slate-800">
                      Explore a graph entity
                    </p>
                    <p className="mx-auto mt-1 max-w-[230px] text-xs leading-5 text-slate-500">
                      Click a node to view its properties and connected
                      relationships.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4 p-4">
                    <div>
                      <span
                        className="inline-flex rounded-md px-2 py-1 text-[10px] font-semibold uppercase tracking-wider"
                        style={{
                          color:
                            LABEL_STYLES[selectedNode.labels[0] ?? ""]?.text ??
                            FALLBACK_STYLE.text,
                          background:
                            LABEL_STYLES[selectedNode.labels[0] ?? ""]
                              ?.background ?? FALLBACK_STYLE.background,
                        }}
                      >
                        {selectedNode.labels.join(" · ") || "Node"}
                      </span>

                      <h3 className="mt-2 break-words text-sm font-semibold text-slate-900">
                        {getNodeTitle(selectedNode)}
                      </h3>
                      <p className="mt-1 break-all text-[10px] leading-4 text-slate-400">
                        Element ID: {selectedNode.id}
                      </p>
                    </div>

                    {loadingDetails ? (
                      <div className="flex items-center gap-2 py-3 text-sm text-slate-500">
                        <RefreshCw size={15} className="animate-spin" />
                        Loading node details…
                      </div>
                    ) : detailsError ? (
                      <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs leading-5 text-rose-700">
                        {detailsError}
                      </div>
                    ) : details ? (
                      <>
                        <div>
                          <div className="mb-2 flex items-center justify-between">
                            <h4 className="text-xs font-semibold text-slate-800">
                              Properties
                            </h4>
                            <span className="text-[10px] text-slate-400">
                              {Object.keys(details.node.properties ?? {}).length}{" "}
                              fields
                            </span>
                          </div>

                          <div className="max-h-64 space-y-2 overflow-auto rounded-lg border border-slate-200 p-3">
                            {visibleProperties.length ? (
                              visibleProperties.map(([key, value]) => (
                                <div
                                  key={key}
                                  className="border-b border-slate-100 pb-2 last:border-0 last:pb-0"
                                >
                                  <p className="text-[10px] font-medium uppercase tracking-wide text-slate-400">
                                    {key.replaceAll("_", " ")}
                                  </p>
                                  <p className="mt-1 break-words whitespace-pre-wrap text-xs leading-5 text-slate-700">
                                    {formatValue(value)}
                                  </p>
                                </div>
                              ))
                            ) : (
                              <p className="text-xs text-slate-500">
                                No properties available.
                              </p>
                            )}
                          </div>

                          {Object.keys(details.node.properties ?? {}).length >
                            12 && (
                            <button
                              type="button"
                              onClick={() =>
                                setShowAllProperties((current) => !current)
                              }
                              className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-slate-600 hover:text-slate-900"
                            >
                              {showAllProperties ? "Show fewer" : "Show all"}
                              <ChevronDown
                                size={13}
                                className={
                                  showAllProperties ? "rotate-180" : ""
                                }
                              />
                            </button>
                          )}
                        </div>

                        <div>
                          <div className="mb-2 flex items-center justify-between">
                            <h4 className="text-xs font-semibold text-slate-800">
                              Connected relationships
                            </h4>
                            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-600">
                              {details.relationships?.length ?? 0}
                            </span>
                          </div>

                          <div className="max-h-64 space-y-2 overflow-auto">
                            {details.relationships?.length ? (
                              details.relationships.map((relationship) => (
                                <div
                                  key={relationship.id}
                                  className="rounded-lg border border-slate-200 p-3"
                                >
                                  <div className="flex items-center gap-2">
                                    <GitBranch
                                      size={13}
                                      className="shrink-0 text-violet-500"
                                    />
                                    <span className="break-words text-xs font-semibold text-slate-800">
                                      {relationship.type}
                                    </span>
                                  </div>
                                  <p className="mt-1 text-[10px] text-slate-500">
                                    {relationship.direction}
                                  </p>
                                  <p className="mt-2 break-words text-xs text-slate-700">
                                    {getNodeTitle({
                                      id: relationship.connected_node_id,
                                      labels: relationship.connected_node_labels,
                                      properties:
                                        relationship.connected_node_properties,
                                    })}
                                  </p>
                                  <p className="mt-1 text-[10px] text-slate-400">
                                    {relationship.connected_node_labels.join(
                                      " · ",
                                    )}
                                  </p>
                                </div>
                              ))
                            ) : (
                              <p className="rounded-lg bg-slate-50 p-3 text-xs text-slate-500">
                                No relationships returned for this node.
                              </p>
                            )}
                          </div>
                        </div>
                      </>
                    ) : null}
                  </div>
                )}
              </div>

              <div className="rounded-xl border border-slate-200 bg-white p-4">
                <SectionHeading
                  icon={Waypoints}
                  title="Graph schema"
                  description="Click a label to filter the explorer."
                />

                {loadingSchema ? (
                  <p className="mt-4 text-xs text-slate-500">
                    Discovering schema from Neo4j…
                  </p>
                ) : schema?.node_labels.length ? (
                  <div className="mt-3 space-y-1">
                    {schema.node_labels
                      .slice()
                      .sort((a, b) => b.node_count - a.node_count)
                      .map((item) => (
                        <button
                          key={item.label}
                          type="button"
                          onClick={() => {
                            setLabelFilter(item.label);
                            setSearch("");
                            setSearchInput("");
                            setActiveTab("explorer");
                          }}
                          className={`group flex w-full items-center justify-between gap-3 rounded-lg px-2 py-2 text-left transition ${
                            labelFilter === item.label
                              ? "bg-slate-100"
                              : "hover:bg-slate-50"
                          }`}
                        >
                          <span className="flex min-w-0 items-center gap-2">
                            <span
                              className="h-2 w-2 shrink-0 rounded-full"
                              style={{
                                background:
                                  (LABEL_STYLES[item.label] ?? FALLBACK_STYLE)
                                    .dot,
                              }}
                            />
                            <span className="truncate text-xs text-slate-700">
                              {item.label}
                            </span>
                          </span>
                          <span className="text-xs tabular-nums text-slate-500 group-hover:text-slate-900">
                            {formatNumber(item.node_count)}
                          </span>
                        </button>
                      ))}
                  </div>
                ) : (
                  <p className="mt-3 text-xs text-slate-500">
                    Schema information is unavailable.
                  </p>
                )}

                {schema?.relationships.length ? (
                  <>
                    <div className="my-3 border-t border-slate-100" />
                    <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                      Relationship patterns
                    </p>
                    <div className="max-h-64 space-y-2 overflow-auto">
                      {schema.relationships.map((relationship, index) => (
                        <div
                          key={`${relationship.relationship_type}-${index}`}
                          className="rounded-lg border border-slate-100 bg-slate-50/70 p-2.5"
                        >
                          <p className="break-words text-[11px] font-semibold text-slate-800">
                            {relationship.relationship_type}
                          </p>
                          <p className="mt-1 break-words text-[10px] leading-4 text-slate-500">
                            {relationship.source_labels.join(" / ")}{" "}
                            <ArrowUpRight
                              size={11}
                              className="inline align-middle"
                            />{" "}
                            {relationship.target_labels.join(" / ")}
                          </p>
                          <p className="mt-1 text-[10px] tabular-nums text-slate-400">
                            {formatNumber(relationship.relationship_count)}{" "}
                            relationships
                          </p>
                        </div>
                      ))}
                    </div>
                  </>
                ) : null}
              </div>

              <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex items-start gap-3">
                  <Database
                    size={17}
                    className="mt-0.5 shrink-0 text-slate-500"
                  />
                  <div>
                    <p className="text-xs font-semibold text-slate-800">
                      Data source
                    </p>
                    <p className="mt-1 text-xs leading-5 text-slate-500">
                      Neo4j · {overview?.database || "Database status unavailable"}
                    </p>
                    <p className="mt-2 text-[10px] leading-4 text-slate-400">
                      The canvas displays only the nodes and relationships
                      returned by the current exploration query.
                    </p>
                  </div>
                </div>
              </div>
            </aside>
          </div>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_300px]">
            <div className="min-w-0 border-b border-slate-200 xl:border-b-0 xl:border-r">
              <div className="border-b border-slate-200 p-4">
                <SectionHeading
                  icon={Code2}
                  title="Read-only Cypher workbench"
                  description="Run a bounded query against Neo4j and inspect the returned records."
                  action={
                    <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[10px] font-medium text-emerald-700">
                      <Check size={12} />
                      Read-only API
                    </span>
                  }
                />

                <div className="mt-4 overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
                  <div className="flex items-center justify-between border-b border-slate-800 px-3 py-2">
                    <div className="flex items-center gap-2 text-[11px] text-slate-400">
                      <Code2 size={13} />
                      Cypher query
                    </div>
                    <span className="text-[10px] text-slate-500">
                      No trailing semicolon required
                    </span>
                  </div>

                  <textarea
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    spellCheck={false}
                    aria-label="Cypher query editor"
                    className="min-h-[220px] w-full resize-y bg-slate-950 p-4 font-mono text-xs leading-6 text-emerald-100 outline-none placeholder:text-slate-600 focus:ring-2 focus:ring-inset focus:ring-slate-700"
                    placeholder="MATCH (n) RETURN n LIMIT 10"
                  />

                  <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-800 px-3 py-3">
                    <div className="flex items-center gap-2 text-[11px] text-slate-400">
                      <span>Max rows</span>
                      <select
                        value={queryLimit}
                        onChange={(event) => setQueryLimit(event.target.value)}
                        className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-xs text-slate-200 outline-none"
                      >
                        <option value="10">10</option>
                        <option value="25">25</option>
                        <option value="50">50</option>
                        <option value="100">100</option>
                        <option value="250">250</option>
                      </select>
                    </div>

                    <button
                      type="button"
                      onClick={() => void runQuery()}
                      disabled={queryLoading || !query.trim()}
                      className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 px-4 py-2 text-xs font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {queryLoading ? (
                        <RefreshCw size={14} className="animate-spin" />
                      ) : (
                        <Play size={14} />
                      )}
                      {queryLoading ? "Running query…" : "Run query"}
                    </button>
                  </div>
                </div>

                <p className="mt-2 text-[11px] leading-5 text-slate-500">
                  The backend validates queries and executes them in a
                  read-only Neo4j transaction. Writes and unsupported
                  administrative clauses are rejected server-side.
                </p>
              </div>

              {queryError && (
                <div className="m-4 flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs leading-5 text-rose-800">
                  <AlertCircle size={15} className="mt-0.5 shrink-0" />
                  <div>
                    <p className="font-semibold">Query failed</p>
                    <p className="mt-1 break-words">{queryError}</p>
                  </div>
                </div>
              )}

              {queryLoading ? (
                <div className="flex min-h-60 items-center justify-center p-8">
                  <div className="text-center">
                    <RefreshCw
                      size={22}
                      className="mx-auto animate-spin text-slate-500"
                    />
                    <p className="mt-3 text-sm font-medium text-slate-700">
                      Executing against Neo4j
                    </p>
                    <p className="mt-1 text-xs text-slate-500">
                      Waiting for the database response…
                    </p>
                  </div>
                </div>
              ) : queryResult ? (
                <div>
                  <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 px-4 py-3">
                    <span className="inline-flex items-center gap-1.5 rounded-md bg-emerald-50 px-2.5 py-1.5 text-[11px] font-medium text-emerald-700">
                      <Check size={12} />
                      Query complete
                    </span>
                    <span className="text-xs text-slate-600">
                      {formatNumber(queryResult.row_count)} rows
                    </span>
                    <span className="text-xs text-slate-400">
                      {queryResult.execution_ms.toFixed(1)} ms
                    </span>

                    {queryResult.truncated && (
                      <span className="rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-[10px] font-medium text-amber-800">
                        Results limited to {queryResult.limit} rows
                      </span>
                    )}

                    <div className="ml-auto flex items-center gap-1 rounded-lg bg-slate-100 p-1">
                      <button
                        type="button"
                        onClick={() => setQueryView("table")}
                        className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium ${
                          queryView === "table"
                            ? "bg-white text-slate-900 shadow-sm"
                            : "text-slate-500 hover:text-slate-800"
                        }`}
                      >
                        <Table2 size={13} />
                        Table
                      </button>
                      <button
                        type="button"
                        onClick={() => setQueryView("graph")}
                        className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium ${
                          queryView === "graph"
                            ? "bg-white text-slate-900 shadow-sm"
                            : "text-slate-500 hover:text-slate-800"
                        }`}
                      >
                        <Network size={13} />
                        Graph
                      </button>
                    </div>
                  </div>

                  {queryView === "table" ? (
                    <div className="max-h-[560px] overflow-auto">
                      {queryResult.rows.length && queryResult.columns.length ? (
                        <table className="w-full border-collapse text-left text-xs">
                          <thead className="sticky top-0 z-10 bg-slate-50">
                            <tr>
                              {queryResult.columns.map((column) => (
                                <th
                                  key={column}
                                  className="whitespace-nowrap border-b border-slate-200 px-3 py-3 font-semibold text-slate-600"
                                >
                                  {column}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {queryResult.rows.map((row, rowIndex) => (
                              <tr
                                key={`row-${rowIndex}`}
                                className="align-top hover:bg-slate-50"
                              >
                                {queryResult.columns.map((column) => (
                                  <td
                                    key={`${rowIndex}-${column}`}
                                    className="max-w-[360px] border-b border-slate-100 px-3 py-3"
                                  >
                                    <pre className="max-w-[360px] whitespace-pre-wrap break-words font-mono text-[11px] leading-5 text-slate-700">
                                      {formatValue(row[column])}
                                    </pre>
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <div className="p-8 text-center text-sm text-slate-500">
                          The query returned no rows.
                        </div>
                      )}
                    </div>
                  ) : queryGraph.nodes.length ? (
                    <GraphCanvas graph={queryGraph} height="520px" />
                  ) : (
                    <div className="p-8 text-center">
                      <Network
                        size={24}
                        className="mx-auto text-slate-400"
                      />
                      <p className="mt-3 text-sm font-medium text-slate-700">
                        No graph entities in these results
                      </p>
                      <p className="mx-auto mt-1 max-w-md text-xs leading-5 text-slate-500">
                        The graph view visualizes returned nodes and
                        relationships. Try the priority, customer, or agent
                        handoff examples.
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex min-h-60 items-center justify-center p-8">
                  <div className="max-w-md text-center">
                    <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-slate-100 text-slate-500">
                      <Code2 size={22} />
                    </div>
                    <p className="mt-3 text-sm font-semibold text-slate-800">
                      Ready to query the graph
                    </p>
                    <p className="mt-1 text-xs leading-5 text-slate-500">
                      Choose an example or edit the Cypher query, then run it
                      to inspect live Neo4j results here.
                    </p>
                  </div>
                </div>
              )}
            </div>

            <aside className="min-w-0 space-y-4 p-4">
              <div className="rounded-xl border border-slate-200 bg-white p-4">
                <SectionHeading
                  icon={Code2}
                  title="Query examples"
                  description="Load a real-schema query into the editor."
                />
                <div className="mt-3 space-y-2">
                  {EXAMPLE_QUERIES.map((example) => (
                    <button
                      key={example.label}
                      type="button"
                      onClick={() => useExample(example.query)}
                      className="w-full rounded-lg border border-slate-200 p-3 text-left transition hover:border-slate-300 hover:bg-slate-50"
                    >
                      <span className="flex items-center justify-between gap-2">
                        <span className="text-xs font-semibold text-slate-800">
                          {example.label}
                        </span>
                        <ArrowDownRight
                          size={14}
                          className="shrink-0 text-slate-400"
                        />
                      </span>
                      <span className="mt-1 block text-[11px] leading-5 text-slate-500">
                        {example.description}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-xl border border-slate-200 bg-white p-4">
                <SectionHeading
                  icon={Waypoints}
                  title="Available schema"
                  description="Use exact labels and relationship types in Cypher."
                />

                <div className="mt-3">
                  <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                    Node labels
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {(schema?.node_labels ?? []).map((item) => (
                      <button
                        key={item.label}
                        type="button"
                        onClick={() =>
                          useExample(
                            `MATCH (n:${item.label})\nRETURN n\nLIMIT 10`,
                          )
                        }
                        title={`Use label ${item.label} in a query`}
                        className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1 text-[10px] font-medium text-slate-600 transition hover:border-slate-400 hover:text-slate-900"
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="mt-4">
                  <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                    Relationships
                  </p>
                  <div className="max-h-72 space-y-2 overflow-auto">
                    {(schema?.relationships ?? []).map(
                      (relationship, index) => (
                        <button
                          key={`${relationship.relationship_type}-${index}`}
                          type="button"
                          onClick={() =>
                            useExample(
                              `MATCH (a:${relationship.source_labels[0] ?? ""})-[r:${relationship.relationship_type}]->(b:${relationship.target_labels[0] ?? ""})\nRETURN a, r, b\nLIMIT 25`,
                            )
                          }
                          className="w-full rounded-lg border border-slate-100 p-2.5 text-left transition hover:border-slate-300 hover:bg-slate-50"
                        >
                          <p className="break-words text-[11px] font-semibold text-slate-800">
                            {relationship.relationship_type}
                          </p>
                          <p className="mt-1 break-words text-[10px] leading-4 text-slate-500">
                            {relationship.source_labels.join(" / ")}{" "}
                            <ArrowUpRight
                              size={11}
                              className="inline align-middle"
                            />{" "}
                            {relationship.target_labels.join(" / ")}
                          </p>
                        </button>
                      ),
                    )}
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex items-start gap-3">
                  <Database
                    size={17}
                    className="mt-0.5 shrink-0 text-slate-500"
                  />
                  <div>
                    <p className="text-xs font-semibold text-slate-800">
                      Execution details
                    </p>
                    <p className="mt-1 text-xs leading-5 text-slate-500">
                      {queryResult
                        ? `${formatNumber(queryResult.row_count)} returned rows · ${queryResult.execution_ms.toFixed(1)} ms`
                        : "No query executed in this session yet."}
                    </p>
                    <p className="mt-2 text-[10px] leading-4 text-slate-400">
                      This interface is for read-only graph exploration. Use
                      Neo4j Browser for other database administration tasks.
                    </p>
                    <button
                      type="button"
                      onClick={openNeo4jBrowser}
                      className="mt-3 inline-flex items-center gap-1.5 text-xs font-medium text-slate-700 hover:text-slate-950"
                    >
                      Open Neo4j Browser <ExternalLink size={12} />
                    </button>
                  </div>
                </div>
              </div>
            </aside>
          </div>
        )}
      </section>
    </main>
  );
}
