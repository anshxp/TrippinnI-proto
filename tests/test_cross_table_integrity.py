import pandas as pd

from context.knowledge_graph import HealthcareKnowledgeGraph
from detectors.cross_table_integrity_detector import CrossTableIntegrityDetector


def _graph_with_relationship():
    graph = HealthcareKnowledgeGraph()
    graph.add_node(
        "column:patients.subject_id",
        "field",
        table="patients",
        column="subject_id",
    )
    graph.add_node(
        "column:admissions.subject_id",
        "field",
        table="admissions",
        column="subject_id",
    )
    graph.add_node("table:patients", "table", table="patients")
    graph.add_node("table:admissions", "table", table="admissions")
    graph.add_edge(
        "column:patients.subject_id",
        "table:patients",
        "CANDIDATE_PRIMARY_KEY",
        0.90,
    )
    graph.add_edge(
        "column:admissions.subject_id",
        "column:patients.subject_id",
        "SHARED_IDENTIFIER_CANDIDATE",
        0.65,
    )
    return graph


def test_cross_table_integrity_detects_orphan_foreign_keys():
    datasets = {
        "patients": pd.DataFrame({"subject_id": [1, 2, 3]}),
        "admissions": pd.DataFrame({"subject_id": [1, 2, 999, 999]}),
    }

    result = CrossTableIntegrityDetector().detect(
        datasets,
        _graph_with_relationship(),
    )

    assert result.issue_count == 1
    issue = result.issues[0]
    assert issue.issue_type == "referential"
    assert issue.table == "admissions"
    assert issue.column == "subject_id"
    assert issue.metadata["orphan_rows"] == 2
    assert issue.metadata["orphan_unique_values"] == 1
    assert issue.metadata["missing_parent_records"] == 1
    assert issue.metadata["referential_coverage"] == 0.5
    assert issue.metadata["fk_to_pk_valid"] is False


def test_cross_table_integrity_validates_parent_key_uniqueness():
    datasets = {
        "patients": pd.DataFrame({"subject_id": [1, 1, 2]}),
        "admissions": pd.DataFrame({"subject_id": [1, 2]}),
    }

    result = CrossTableIntegrityDetector().detect(
        datasets,
        _graph_with_relationship(),
    )

    assert result.issue_count == 1
    stats = next(iter(result.statistics.values()))
    assert stats["duplicate_parent_keys"] == 1
    assert stats["orphan_rows"] == 0
    assert stats["parent_key_is_unique"] is False
    assert stats["fk_to_pk_valid"] is False


def test_cross_table_integrity_passes_valid_fk_to_pk_relationship():
    datasets = {
        "patients": pd.DataFrame({"subject_id": [1, 2, 3]}),
        "admissions": pd.DataFrame({"subject_id": [1, 2, 2, 3]}),
    }

    result = CrossTableIntegrityDetector().detect(
        datasets,
        _graph_with_relationship(),
    )

    assert result.issue_count == 0
    stats = next(iter(result.statistics.values()))
    assert stats["orphan_rows"] == 0
    assert stats["duplicate_parent_keys"] == 0
    assert stats["referential_coverage"] == 1.0
    assert stats["unique_referential_coverage"] == 1.0
    assert stats["fk_to_pk_valid"] is True


def test_cross_table_integrity_uses_ml_boundary_for_high_confidence_near_matches():
    graph = _graph_with_relationship()
    datasets = {
        "patients": pd.DataFrame({
            "subject_id": [
                "patient-001", "patient-002", "patient-003",
                "unrelated-alpha", "unrelated-beta", "unrelated-gamma",
            ]
        }),
        "admissions": pd.DataFrame({
            "subject_id": ["patient-001", "patient-002", "patient-00l"]
        }),
    }

    result = CrossTableIntegrityDetector().detect(datasets, graph)

    stats = next(iter(result.statistics.values()))
    assert stats["ml_boundary_used"] is True
    assert stats["ml_recovered_unique_values"] >= 1
    assert stats["orphan_unique_values"] == 0
    assert stats["referential_coverage"] == 1.0
    assert stats["fk_to_pk_valid"] is True
    assert "patient-00l" in stats["ml_matches"]
