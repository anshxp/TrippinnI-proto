"""Cross-table referential integrity validation for synchronized table samples."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd

from models.detector_result import DetectorResult
from models.issue import Issue


class CrossTableIntegrityDetector:
    """Validate candidate FK -> PK relationships using synchronized samples.

    Relationships are supplied by the healthcare knowledge graph. The detector
    deliberately reports sample-scoped findings because the prototype does not
    materialize every MIMIC table simultaneously.
    """

    name = "CrossTableIntegrityDetector"

    def detect(
        self,
        datasets: Dict[str, pd.DataFrame],
        graph: Any,
    ) -> DetectorResult:
        result = DetectorResult(detector_name=self.name)
        relationships = self._relationship_candidates(graph)

        validated = 0
        skipped = 0

        for source_table, source_column, target_table, target_column, confidence in relationships:
            child = datasets.get(source_table)
            parent = datasets.get(target_table)

            if child is None or parent is None:
                skipped += 1
                continue
            if source_column not in child.columns or target_column not in parent.columns:
                skipped += 1
                continue

            stats = self._validate_relationship(
                child,
                source_column,
                parent,
                target_column,
            )
            stats.update({
                "child_table": source_table,
                "child_column": source_column,
                "parent_table": target_table,
                "parent_column": target_column,
                "relationship_confidence": confidence,
                "scope": "bounded_detection_samples",
            })

            result.statistics[f"{source_table}.{source_column}->{target_table}.{target_column}"] = stats
            validated += 1

            orphan_rows = int(stats["orphan_rows"])
            duplicate_parent_keys = int(stats["duplicate_parent_keys"])

            if orphan_rows or duplicate_parent_keys:
                reasons = []
                if orphan_rows:
                    reasons.append(f"{orphan_rows:,} orphan child rows")
                if duplicate_parent_keys:
                    reasons.append(f"{duplicate_parent_keys:,} duplicate parent key values")

                result.add_issue(Issue(
                    table=source_table,
                    row_index=-1,
                    column=source_column,
                    issue_type="referential",
                    severity="HIGH",
                    detector=self.name,
                    confidence=float(confidence),
                    expected_value=f"every non-null {source_column} value maps to {target_table}.{target_column}",
                    metadata={
                        "relationship": "FOREIGN_KEY_TO_PRIMARY_KEY",
                        "source_table": source_table,
                        "source_column": source_column,
                        "target_table": target_table,
                        "target_column": target_column,
                        "orphan_rows": orphan_rows,
                        "orphan_unique_values": int(stats["orphan_unique_values"]),
                        "missing_parent_records": int(stats["orphan_unique_values"]),
                        "duplicate_parent_keys": duplicate_parent_keys,
                        "child_non_null_values": int(stats["child_non_null_values"]),
                        "parent_non_null_values": int(stats["parent_non_null_values"]),
                        "referential_coverage": float(stats["referential_coverage"]),
                        "unique_referential_coverage": float(stats["unique_referential_coverage"]),
                        "validation_scope": "bounded_detection_samples",
                        "reasons": reasons,
                    },
                ))

        result.update_statistics(
            candidate_relationships=len(relationships),
            validated_relationships=validated,
            skipped_relationships=skipped,
            validation_mode="value_level_sample",
            scope="bounded_detection_samples",
        )
        return result

    @staticmethod
    def _relationship_candidates(graph: Any) -> list[tuple[str, str, str, str, float]]:
        """Return one directed FK -> PK candidate per shared identifier pair."""
        candidates: list[tuple[str, str, str, str, float]] = []
        seen: set[tuple[str, str, str, str]] = set()

        for edge in getattr(graph, "edges", []):
            if edge.relation != "SHARED_IDENTIFIER_CANDIDATE":
                continue

            source = graph.nodes.get(edge.source)
            target = graph.nodes.get(edge.target)
            if source is None or target is None:
                continue

            source_table = source.properties.get("table")
            source_column = source.properties.get("column")
            target_table = target.properties.get("table")
            target_column = target.properties.get("column")
            if not all((source_table, source_column, target_table, target_column)):
                continue

            source_primary = CrossTableIntegrityDetector._is_primary_key(graph, edge.source, source_table)
            target_primary = CrossTableIntegrityDetector._is_primary_key(graph, edge.target, target_table)

            if target_primary and not source_primary:
                child_table, child_column = source_table, source_column
                parent_table, parent_column = target_table, target_column
            elif source_primary and not target_primary:
                child_table, child_column = target_table, target_column
                parent_table, parent_column = source_table, source_column
            else:
                # Ambiguous shared identifiers are retained, but the direction
                # is deterministic. The validator will use observed uniqueness
                # to decide whether the proposed parent is plausible.
                child_table, child_column = source_table, source_column
                parent_table, parent_column = target_table, target_column

            key = (child_table, child_column, parent_table, parent_column)
            reverse = (parent_table, parent_column, child_table, child_column)
            if key in seen or reverse in seen:
                continue
            seen.add(key)
            candidates.append((
                child_table,
                child_column,
                parent_table,
                parent_column,
                float(edge.confidence),
            ))

        return candidates

    @staticmethod
    def _is_primary_key(graph: Any, column_node: str, table: str) -> bool:
        return any(
            edge.relation == "CANDIDATE_PRIMARY_KEY"
            and edge.target == f"table:{table}"
            for edge in getattr(graph, "edges", [])
            if edge.source == column_node
        )

    @staticmethod
    def _validate_relationship(
        child: pd.DataFrame,
        child_column: str,
        parent: pd.DataFrame,
        parent_column: str,
    ) -> Dict[str, Any]:
        child_values = CrossTableIntegrityDetector._canonical_values(child[child_column])
        parent_values = CrossTableIntegrityDetector._canonical_values(parent[parent_column])

        child_non_null = child_values.dropna()
        parent_non_null = parent_values.dropna()

        parent_counts = parent_non_null.value_counts(dropna=False)
        duplicate_parent_keys = int((parent_counts > 1).sum())

        parent_set = set(parent_non_null.tolist())
        orphan_mask = ~child_non_null.isin(parent_set)
        orphan_rows = int(orphan_mask.sum())
        orphan_values = child_non_null[orphan_mask].unique()

        child_unique = set(child_non_null.unique().tolist())
        parent_unique = set(parent_non_null.unique().tolist())
        matched_unique = child_unique.intersection(parent_unique)

        child_count = int(len(child_non_null))
        parent_count = int(len(parent_non_null))
        unique_child_count = int(len(child_unique))

        return {
            "child_non_null_values": child_count,
            "parent_non_null_values": parent_count,
            "child_unique_values": unique_child_count,
            "parent_unique_values": int(len(parent_unique)),
            "orphan_rows": orphan_rows,
            "orphan_unique_values": int(len(orphan_values)),
            "duplicate_parent_keys": duplicate_parent_keys,
            "matched_unique_values": int(len(matched_unique)),
            "referential_coverage": round(
                (child_count - orphan_rows) / max(child_count, 1), 6
            ),
            "unique_referential_coverage": round(
                len(matched_unique) / max(unique_child_count, 1), 6
            ),
            "parent_key_is_unique": duplicate_parent_keys == 0,
            "fk_to_pk_valid": orphan_rows == 0 and duplicate_parent_keys == 0,
        }

    @staticmethod
    def _canonical_values(series: pd.Series) -> pd.Series:
        values = series.copy()
        if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            values = values.astype("string").str.strip()
            values = values.mask(values.eq(""))
        return values.dropna()
