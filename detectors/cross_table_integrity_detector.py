"""Cross-table referential integrity validation for synchronized table samples."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from config import (
    CROSS_TABLE_ML_ENABLED,
    CROSS_TABLE_ML_MAX_CHILD_VALUES,
    CROSS_TABLE_ML_MAX_PARENT_VALUES,
    CROSS_TABLE_ML_MIN_FUZZY_SIMILARITY,
    CROSS_TABLE_ML_MIN_MARGIN,
    CROSS_TABLE_ML_MIN_PROBABILITY,
)

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
        relationships = self._relationship_candidates(graph, datasets)

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

            # Exact/canonical integrity is always authoritative. ML is only
            # allowed to resolve values that remain unmatched after that step.
            child_values = self._canonical_values(child[source_column])
            parent_values = self._canonical_values(parent[target_column])
            boundary = self._ml_boundary_matches(child_values, parent_values)
            ml_matches = boundary["matches"]

            if ml_matches:
                matched_by_ml = child_values.astype(str).isin(ml_matches.keys())
                ml_resolved_rows = int(matched_by_ml.sum())
                original_orphan_rows = int(stats["orphan_rows"])
                original_orphan_unique = int(stats["orphan_unique_values"])
                stats["ml_recovered_rows"] = ml_resolved_rows
                stats["ml_recovered_unique_values"] = len(ml_matches)
                stats["ml_candidate_count"] = int(boundary["candidate_count"])
                stats["ml_ambiguous_count"] = int(boundary["ambiguous_count"])
                stats["ml_boundary_used"] = True
                stats["ml_matches"] = ml_matches
                stats["orphan_rows"] = max(0, original_orphan_rows - ml_resolved_rows)
                stats["orphan_unique_values"] = max(
                    0, original_orphan_unique - len(ml_matches)
                )
                child_count = int(stats["child_non_null_values"])
                stats["referential_coverage"] = round(
                    (child_count - stats["orphan_rows"]) / max(child_count, 1), 6
                )
                unique_child_count = int(stats["child_unique_values"])
                stats["unique_referential_coverage"] = round(
                    (unique_child_count - stats["orphan_unique_values"])
                    / max(unique_child_count, 1),
                    6,
                )
            else:
                stats["ml_recovered_rows"] = 0
                stats["ml_recovered_unique_values"] = 0
                stats["ml_candidate_count"] = int(boundary["candidate_count"])
                stats["ml_ambiguous_count"] = int(boundary["ambiguous_count"])
                stats["ml_boundary_used"] = bool(boundary["model_used"])
                stats["ml_matches"] = {}

            stats["fk_to_pk_valid"] = (
                stats["orphan_rows"] == 0
                and stats["duplicate_parent_keys"] == 0
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
                        "ml_boundary_used": bool(stats["ml_boundary_used"]),
                        "ml_recovered_rows": int(stats["ml_recovered_rows"]),
                        "ml_recovered_unique_values": int(stats["ml_recovered_unique_values"]),
                        "ml_candidate_count": int(stats["ml_candidate_count"]),
                        "ml_ambiguous_count": int(stats["ml_ambiguous_count"]),
                        "ml_matches": stats["ml_matches"],
                        "missing_parent_records": int(stats["orphan_unique_values"]),
                        "duplicate_parent_keys": duplicate_parent_keys,
                        "child_non_null_values": int(stats["child_non_null_values"]),
                        "parent_non_null_values": int(stats["parent_non_null_values"]),
                        "referential_coverage": float(stats["referential_coverage"]),
                        "unique_referential_coverage": float(stats["unique_referential_coverage"]),
                        "fk_to_pk_valid": bool(stats["fk_to_pk_valid"]),
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
    def _relationship_candidates(
        graph: Any,
        datasets: Dict[str, pd.DataFrame],
    ) -> list[tuple[str, str, str, str, float]]:
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
                # If the graph cannot establish the direction, prefer the
                # observed one-to-many shape: the more unique column is the
                # parent candidate. If the samples are unavailable or tied,
                # keep the graph's deterministic direction.
                source_score = CrossTableIntegrityDetector._uniqueness_score(
                    datasets.get(source_table), source_column
                )
                target_score = CrossTableIntegrityDetector._uniqueness_score(
                    datasets.get(target_table), target_column
                )
                if target_score > source_score:
                    child_table, child_column = source_table, source_column
                    parent_table, parent_column = target_table, target_column
                else:
                    child_table, child_column = target_table, target_column
                    parent_table, parent_column = source_table, source_column

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
    def _uniqueness_score(
        dataframe: pd.DataFrame | None,
        column: str,
    ) -> float:
        if dataframe is None or column not in dataframe.columns:
            return 0.0
        values = CrossTableIntegrityDetector._canonical_values(dataframe[column])
        if values.empty:
            return 0.0
        return float(values.nunique(dropna=True) / len(values))

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
    def _ml_boundary_matches(
        child_values: pd.Series,
        parent_values: pd.Series,
    ) -> Dict[str, Any]:
        """Resolve bounded unmatched values with an auditable ML boundary.

        Exact/canonical equality is handled before this method. Only unresolved
        string-like values reach the ML stage. Candidate generation is blocked
        with RapidFuzz, while the classifier learns a local boundary from exact
        matches plus deterministic one-character corruption examples and
        conservative negatives. The model never mutates source data.
        """
        if not CROSS_TABLE_ML_ENABLED:
            return {"matches": {}, "candidate_count": 0, "ambiguous_count": 0, "model_used": False}

        from rapidfuzz import fuzz, process
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        child_unique = [str(v) for v in child_values.dropna().unique().tolist()]
        parent_unique = [str(v) for v in parent_values.dropna().unique().tolist()]
        if not child_unique or not parent_unique:
            return {"matches": {}, "candidate_count": 0, "ambiguous_count": 0, "model_used": False}

        parent_set = set(parent_unique)
        unresolved_children = [v for v in child_unique if v not in parent_set]
        if not unresolved_children:
            return {"matches": {}, "candidate_count": 0, "ambiguous_count": 0, "model_used": False}

        # Only string-like relationship values are eligible for approximate
        # matching. Numeric FK/PK relationships remain exact/canonical only.
        if not any(not value.isdigit() for value in unresolved_children):
            return {"matches": {}, "candidate_count": 0, "ambiguous_count": 0, "model_used": False}

        unresolved_children = unresolved_children[:CROSS_TABLE_ML_MAX_CHILD_VALUES]
        parent_pool = parent_unique[:CROSS_TABLE_ML_MAX_PARENT_VALUES]

        def features(a: str, b: str) -> list[float]:
            ratio = fuzz.ratio(a, b) / 100.0
            a_len, b_len = len(a), len(b)
            length_ratio = min(a_len, b_len) / max(a_len, b_len, 1)

            prefix = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                prefix += 1

            suffix = 0
            for x, y in zip(reversed(a), reversed(b)):
                if x != y:
                    break
                suffix += 1

            overlap = len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
            numeric_shape = float(a.isdigit() == b.isdigit())
            return [
                ratio,
                length_ratio,
                prefix / max(a_len, 1),
                suffix / max(b_len, 1),
                overlap,
                numeric_shape,
            ]

        def corrupt(value: str) -> list[str]:
            """Create deterministic, conservative typo-like positive examples."""
            if len(value) < 4:
                return []

            alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
            variants: list[str] = []
            for position, current in enumerate(value):
                replacement = next((char for char in alphabet if char != current), None)
                if replacement is None:
                    continue
                chars = list(value)
                chars[position] = replacement
                variant = "".join(chars)
                if variant != value:
                    variants.append(variant)

            # One deletion captures a common truncation boundary without
            # introducing arbitrary synthetic strings.
            for position in range(len(value)):
                variant = value[:position] + value[position + 1:]
                if len(variant) >= 3:
                    variants.append(variant)

            return variants[:40]

        training_x: list[list[float]] = []
        training_y: list[int] = []

        common = sorted(
            set(child_values.dropna().astype(str)).intersection(parent_set)
        )

        # Exact matches are positives. Synthetic one-edit variants teach the
        # classifier the intended corruption boundary instead of making it
        # learn only the identity point.
        for value in common[:200]:
            training_x.append(features(value, value))
            training_y.append(1)
            for variant in corrupt(value):
                training_x.append(features(variant, value))
                training_y.append(1)

        # Conservative negatives are relationship pairs that are clearly
        # dissimilar. We also use unrelated parent values from the bounded pool
        # so the model sees realistic non-matches.
        negative_budget = 2_000
        for child_value in unresolved_children:
            for parent_value in parent_pool:
                if child_value == parent_value:
                    continue
                score = fuzz.ratio(child_value, str(parent_value))
                if score <= 70 and negative_budget > 0:
                    training_x.append(features(child_value, str(parent_value)))
                    training_y.append(0)
                    negative_budget -= 1
                if negative_budget == 0:
                    break
            if negative_budget == 0:
                break

        # If the relationship has too few exact matches, use dissimilar
        # cross-pairs from the observed parent set as additional negatives.
        if negative_budget > 0 and len(common) > 1:
            for left in common[:200]:
                for right in common[:200]:
                    if left == right:
                        continue
                    score = fuzz.ratio(left, right)
                    if score <= 70:
                        training_x.append(features(left, right))
                        training_y.append(0)
                        negative_budget -= 1
                    if negative_budget == 0:
                        break
                if negative_budget == 0:
                    break

        if len(set(training_y)) < 2 or len(training_x) < 6:
            return {
                "matches": {},
                "candidate_count": 0,
                "ambiguous_count": 0,
                "model_used": False,
            }

        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_leaf=1,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(np.asarray(training_x), np.asarray(training_y))

        matches: Dict[str, Dict[str, Any]] = {}
        candidate_count = 0
        ambiguous_count = 0

        for child_value in unresolved_children:
            candidates = process.extract(
                child_value,
                parent_pool,
                scorer=fuzz.ratio,
                score_cutoff=CROSS_TABLE_ML_MIN_FUZZY_SIMILARITY,
                limit=3,
            )
            if not candidates:
                continue

            scored = []
            for parent_value, fuzzy_score, _ in candidates:
                candidate_count += 1
                vector = np.asarray([features(child_value, str(parent_value))])
                probability = float(model.predict_proba(vector)[0, 1])
                scored.append((probability, str(parent_value), float(fuzzy_score)))

            scored.sort(reverse=True)
            best_probability, best_parent, best_fuzzy = scored[0]
            second_probability = scored[1][0] if len(scored) > 1 else 0.0
            margin = best_probability - second_probability

            if (
                best_probability >= CROSS_TABLE_ML_MIN_PROBABILITY
                and margin >= CROSS_TABLE_ML_MIN_MARGIN
                and best_fuzzy >= CROSS_TABLE_ML_MIN_FUZZY_SIMILARITY
            ):
                matches[child_value] = {
                    "parent_value": best_parent,
                    "probability": round(best_probability, 6),
                    "margin": round(margin, 6),
                    "fuzzy_similarity": round(best_fuzzy, 3),
                    "method": "ml_boundary",
                }
            else:
                ambiguous_count += 1

        return {
            "matches": matches,
            "candidate_count": candidate_count,
            "ambiguous_count": ambiguous_count,
            "model_used": True,
        }

    @staticmethod
    def _canonical_values(series: pd.Series) -> pd.Series:
        values = series.copy()
        if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            values = values.astype("string").str.strip()
            values = values.mask(values.eq(""))
        return values.dropna()
