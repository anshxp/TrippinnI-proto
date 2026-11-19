#!/usr/bin/env python3
"""Run local Qwen3:32B against a single TrippinnI profile bundle.

Qwen creates proposed metadata interpretation, algorithm plans and an audit SPECIFICATION.
It must not create factual runtime audit events. Those belong to the executing application.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = [
    {"id": "report_missingness", "status": "implemented_and_registered", "effect": "report_only"},
    {"id": "report_exact_duplicates", "status": "implemented_and_registered", "effect": "report_only"},
    {"id": "flag_iqr_outliers", "status": "implemented_and_registered", "effect": "flag_only"},
    {"id": "isolation_forest_flags", "status": "implemented_and_registered", "effect": "flag_only"},
    {"id": "median_imputation", "status": "implemented_and_registered", "effect": "approval_gated_mutation"},
    {"id": "mean_imputation", "status": "proposed_not_implemented", "effect": "unknown_until_implemented"},
    {"id": "mode_imputation", "status": "proposed_not_implemented", "effect": "unknown_until_implemented"},
    {"id": "knn_imputation", "status": "proposed_not_implemented", "effect": "unknown_until_implemented"},
    {"id": "mice_imputation", "status": "proposed_not_implemented", "effect": "unknown_until_implemented"},
    {"id": "missforest_imputation", "status": "proposed_not_implemented", "effect": "unknown_until_implemented"},
    {"id": "groupwise_imputation", "status": "proposed_not_implemented", "effect": "unknown_until_implemented"},
    {"id": "temporal_forward_fill", "status": "proposed_not_implemented", "effect": "unknown_until_implemented"},
]

SYSTEM_PROMPT = """You are the metadata and preprocessing-planning component for TrippinnI.
Use ONLY the supplied profiler JSON and explicit algorithm registry. Never invent profile
values, clinical facts, relationships, algorithm implementations, runtime results, timestamps,
or audit outcomes. Preserve exactness flags, summary scope, bounded samples, and candidate status.
Do not treat empirical quantiles/ranges as clinical reference ranges. Never numerically coerce
medical codes or identifiers. Never infer patient identity from fuzzy name matching.
Missingness percentage alone does not establish MCAR/MAR/MNAR. Do not impute by default.
For every algorithm, distinguish implemented/registered from proposed/unavailable.
Return valid JSON only, no markdown fences.

Generate:
1. dataset/table and column metadata, preserving source values and provenance;
2. column-level quality interpretation and uncertainty;
3. an ordered algorithm proposal with eligibility, parameters, rationale, risk, expected
   effect, approval requirement, and implementation status;
4. an audit event SPECIFICATION (event types and required fields), not fabricated events.

The registered actions are strictly limited to the registry entries marked
implemented_and_registered. Reporting/flagging actions must not be described as mutations.
Median imputation is approval-gated and must preserve/archive original missing cells.
Do not propose an executable action unless it is in the registry as implemented_and_registered.
If evidence is insufficient, return needs_human_review and explain why.
"""

def ollama_chat(url: str, model: str, messages: list[dict[str, str]], timeout: int) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["message"]["content"]


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Qwen returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Qwen response must be a JSON object")
    return value


def ask_with_retry(url: str, model: str, messages: list[dict[str, str]], timeout: int, retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return parse_json_response(ollama_chat(url, model, messages, timeout))
        except (ValueError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Qwen request failed after {retries + 1} attempt(s): {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/profiles_bundle.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen_analysis.json"))
    parser.add_argument("--model", default="qwen3:32b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    if bundle.get("bundle_type") != "trippinni_multi_table_profiler_bundle":
        parser.error("Input is not a TrippinnI profiler bundle")
    tables = bundle.get("tables")
    if not isinstance(tables, dict) or not tables:
        parser.error("Bundle contains no tables")

    registry = DEFAULT_REGISTRY
    results: dict[str, Any] = {}
    for table_name, entry in tables.items():
        user_payload = {
            "task": "Analyze this table profile. Return one JSON object for this table.",
            "table_name": table_name,
            "source_file": entry.get("source_file"),
            "source_sha256": entry.get("source_sha256"),
            "profile": entry.get("profile"),
            "algorithm_registry": registry,
            "required_output_shape": {
                "table_name": "string",
                "metadata": {"dataset": {}, "columns": []},
                "quality_interpretation": [],
                "algorithm_plan": [],
                "audit_specification": {"event_types": [], "required_common_fields": []},
                "needs_human_review": "boolean",
                "limitations": [],
            },
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        results[table_name] = ask_with_retry(args.ollama_url, args.model, messages, args.timeout, args.retries)

    # Cross-table step uses compact metadata outputs, not raw clinical rows.
    cross_payload = {
        "task": "Review cross-table consistency and candidate relationships across these table analyses.",
        "table_analyses": results,
        "constraints": [
            "Treat PK/FK and shared identifiers as candidates unless validated by explicit integrity checks.",
            "Do not assert referential integrity from matching names alone.",
            "Do not merge patient records or infer identity.",
            "Return only evidence-backed relationships and unresolved checks.",
        ],
        "required_output_shape": {
            "candidate_relationships": [],
            "cross_table_quality_risks": [],
            "recommended_integrity_checks": [],
            "limitations": [],
        },
    }
    cross_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(cross_payload, ensure_ascii=False)},
    ]
    cross_table = ask_with_retry(args.ollama_url, args.model, cross_messages, args.timeout, args.retries)

    output = {
        "pipeline_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "source_bundle": str(args.bundle),
        "source_bundle_schema_version": bundle.get("bundle_schema_version"),
        "table_analyses": results,
        "cross_table_analysis": cross_table,
        "important_boundary": (
            "This file contains proposed plans and audit specifications only. "
            "The application must emit actual runtime audit events from execution."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(args.output)
    print(f"Saved Qwen analysis for {len(results)} table(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
