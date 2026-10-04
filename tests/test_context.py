from context.context_engine import ContextEngine
from context.knowledge_graph import KnowledgeGraphBuilder
from context.semantic_context import SemanticContextBuilder


def _profile(table, columns, primary_keys=None, foreign_keys=None):
    return {
        "dataset": {"rows": 10, "columns": len(columns)},
        "columns": columns,
        "keys": {
            "primary_keys": primary_keys or [],
            "foreign_keys": foreign_keys or [],
        },
    }


def test_semantic_context_is_dataset_agnostic():
    profiles = {
        "patient_table": _profile(
            "patient_table",
            {
                "patient_id": {"semantic_type": "identifier"},
                "birth_date": {"semantic_type": "datetime"},
                "diagnosis_code": {"semantic_type": "medical_code"},
            },
            primary_keys=["patient_id"],
        )
    }
    context = SemanticContextBuilder().build(profiles)
    columns = context["tables"]["patient_table"]["columns"]

    assert "identifier" in columns["patient_id"]["roles"]
    assert "temporal" in columns["birth_date"]["roles"]
    assert "clinical" in columns["diagnosis_code"]["roles"]


def test_knowledge_graph_discovers_shared_identifier_candidates():
    profiles = {
        "patients": _profile(
            "patients",
            {"patient_id": {"semantic_type": "identifier"}},
            primary_keys=["patient_id"],
        ),
        "encounters": _profile(
            "encounters",
            {"patient_id": {"semantic_type": "identifier"}},
            foreign_keys=["patient_id"],
        ),
    }
    context = SemanticContextBuilder().build(profiles)
    graph = KnowledgeGraphBuilder().build(context)

    assert any(
        edge.relation == "SHARED_IDENTIFIER_CANDIDATE"
        for edge in graph.edges
    )


def test_cross_table_context_does_not_claim_value_level_integrity():
    profiles = {
        "patients": _profile(
            "patients",
            {"patient_id": {"semantic_type": "identifier"}},
        ),
        "encounters": _profile(
            "encounters",
            {"patient_id": {"semantic_type": "identifier"}},
        ),
    }
    context = SemanticContextBuilder().build(profiles)
    graph = KnowledgeGraphBuilder().build(context)
    result = ContextEngine().detect_cross_table_structure(context, graph)

    assert result.success
    assert result.statistics["validation_mode"] == "structural_profile_only"
    assert all(
        issue.metadata["requires_value_validation"] is True
        for issue in result.issues
    )
