"""Context enrichment and cross-table structural quality checks."""

from __future__ import annotations

from typing import Any, Dict, List

from models.detector_result import DetectorResult
from models.issue import Issue
from context.knowledge_graph import HealthcareKnowledgeGraph


class ContextEngine:
    """Attach graph-derived context to issues and detect structural relationships."""

    def enrich_issues(self, issues: List[Issue], graph: HealthcareKnowledgeGraph) -> List[Issue]:
        for issue in issues:
            table_node = f"table:{issue.table}"
            related = graph.neighbors(table_node, "HAS_FIELD")
            issue.metadata.setdefault("context", {})
            issue.metadata["context"].update({
                "graph_version": graph.VERSION,
                "table_node": table_node,
                "known_fields": len(related),
            })
            if issue.column:
                column_node = f"column:{issue.table}.{issue.column}"
                node = graph.nodes.get(column_node)
                if node:
                    issue.metadata["context"]["semantic_role"] = node.properties.get("role")
                    issue.metadata["context"]["semantic_confidence"] = node.properties.get("confidence")
        return issues

    def detect_cross_table_structure(self, semantic_context: Dict[str, Any], graph: HealthcareKnowledgeGraph) -> DetectorResult:
        """Flag ambiguous shared identifiers rather than claiming referential failure.

        Actual value-level referential integrity requires synchronized samples or
        full-table keys and is intentionally left for the next cross-table stage.
        """
        result = DetectorResult(detector_name="CrossTableContextDetector")
        for edge in graph.edges:
            if edge.relation != "SHARED_IDENTIFIER_CANDIDATE":
                continue
            source = graph.nodes.get(edge.source)
            target = graph.nodes.get(edge.target)
            if not source or not target:
                continue
            source_table = source.properties.get("table")
            target_table = target.properties.get("table")
            column = source.properties.get("column")
            result.add_issue(Issue(
                table=source_table,
                row_index=-1,
                column=column,
                issue_type="cross_table_relationship_candidate",
                severity="LOW",
                detector=result.detector_name,
                confidence=edge.confidence,
                expected_value=f"relationship candidate with {target_table}.{column}",
                metadata={
                    "relationship": edge.relation,
                    "source_table": source_table,
                    "target_table": target_table,
                    "requires_value_validation": True,
                },
            ))
        result.update_statistics(
            candidate_relationships=result.issue_count,
            validation_mode="structural_profile_only",
        )
        return result
