"""Lightweight, serializable healthcare knowledge graph.

The graph represents inferred dataset semantics and relationships. It is a
context layer for quality detection, not a replacement for a clinical
terminology service or a validated ontology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass(slots=True)
class GraphNode:
    node_id: str
    node_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    relation: str
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


class HealthcareKnowledgeGraph:
    """In-memory graph with deterministic JSON serialization."""

    VERSION = "0.1"

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []

    def add_node(self, node_id: str, node_type: str, **properties: Any) -> None:
        self.nodes[node_id] = GraphNode(node_id, node_type, properties)

    def add_edge(self, source: str, target: str, relation: str, confidence: float = 1.0, **properties: Any) -> None:
        self.edges.append(GraphEdge(source, target, relation, float(confidence), properties))

    def neighbors(self, node_id: str, relation: str | None = None) -> List[GraphEdge]:
        return [
            edge for edge in self.edges
            if edge.source == node_id and (relation is None or edge.relation == relation)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_version": self.VERSION,
            "nodes": [
                {"id": n.node_id, "type": n.node_type, "properties": n.properties}
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "confidence": e.confidence,
                    "properties": e.properties,
                }
                for e in self.edges
            ],
        }


class KnowledgeGraphBuilder:
    """Construct a graph from inferred semantic context and candidate keys."""

    def build(self, semantic_context: Dict[str, Any]) -> HealthcareKnowledgeGraph:
        graph = HealthcareKnowledgeGraph()
        tables = semantic_context.get("tables", {})

        for table, table_context in tables.items():
            table_id = f"table:{table}"
            graph.add_node(table_id, "table", table=table)
            columns = table_context.get("columns", {})
            for column, context in columns.items():
                column_id = f"column:{table}.{column}"
                graph.add_node(column_id, "field", table=table, column=column, **context)
                graph.add_edge(table_id, column_id, "HAS_FIELD", context.get("confidence", 0.5))

                roles = set(context.get("roles", []))
                if "identifier" in roles:
                    graph.add_edge(column_id, f"concept:entity_identifier", "PLAYS_ROLE", context.get("confidence", 0.5))
                if "temporal" in roles:
                    graph.add_edge(column_id, f"concept:temporal_attribute", "PLAYS_ROLE", context.get("confidence", 0.5))
                if "clinical" in roles:
                    graph.add_edge(column_id, f"concept:clinical_attribute", "PLAYS_ROLE", context.get("confidence", 0.5))

            keys = table_context.get("candidate_keys", {}) or {}
            for key in keys.get("primary_keys", []) or []:
                column_id = f"column:{table}.{key}"
                if column_id in graph.nodes:
                    graph.add_edge(column_id, table_id, "CANDIDATE_PRIMARY_KEY", 0.90)
            for key in keys.get("foreign_keys", []) or []:
                column_id = f"column:{table}.{key}"
                if column_id in graph.nodes:
                    graph.add_edge(column_id, table_id, "CANDIDATE_FOREIGN_KEY", 0.80)

        self._infer_relationships(graph, tables)
        return graph

    @staticmethod
    def _infer_relationships(graph: HealthcareKnowledgeGraph, tables: Dict[str, Any]) -> None:
        """Infer conservative cross-table relationships from shared field names."""
        field_index: Dict[str, List[str]] = {}
        for table, context in tables.items():
            for column, meta in context.get("columns", {}).items():
                if "identifier" not in set(meta.get("roles", [])):
                    continue
                field_index.setdefault(column.lower(), []).append(table)

        for field, table_names in field_index.items():
            if len(table_names) < 2:
                continue
            for source in table_names:
                for target in table_names:
                    if source == target:
                        continue
                    source_col = f"column:{source}.{field}"
                    target_col = f"column:{target}.{field}"
                    if source_col in graph.nodes and target_col in graph.nodes:
                        graph.add_edge(source_col, target_col, "SHARED_IDENTIFIER_CANDIDATE", 0.65)
