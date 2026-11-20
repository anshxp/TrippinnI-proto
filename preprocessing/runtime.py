"""Deterministic, auditable preprocessing runtime; no LLM/Ollama dependency."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.registry import build_action_registry, profile_dataframe


def _select_actions(profile: dict[str, Any], specs: list[Any], *, allow_imputation: bool, profile_mode: str = "fast") -> list[dict[str, Any]]:
    """Transparent conservative rules. These are candidate decisions, not clinical labels."""
    by_name = {spec.name: spec for spec in specs}
    decisions: list[dict[str, Any]] = []
    decisions.append({
        "action": "report_missingness", "parameters": {},
        "rule_id": "DQ-MISS-001",
        "rationale": "Always profile column-level missingness before considering mutation.",
    })
    if profile_mode == "full":
        decisions.append({
            "action": "report_exact_duplicates", "parameters": {},
            "rule_id": "DQ-DUP-001",
            "rationale": "Quantify exact duplicate candidates; this action does not delete rows.",
        })
    for column, details in profile.get("columns", {}).items():
        if profile_mode != "full":
            continue
        numeric = details.get("numeric_summary")
        if numeric and details.get("distinct_non_null", 0) >= 4:
            decisions.append({
                "action": "flag_iqr_outliers", "parameters": {"column": column},
                "rule_id": "DQ-OUT-IQR-001",
                "rationale": "Numeric column has at least four distinct observed values; add a flag only, without changing clinical values.",
            })
    # Imputation is intentionally not auto-selected. It requires an explicit
    # human-approved action request and a clinically justified policy.
    return [d for d in decisions if d["action"] in by_name]


def run_dataframe(
    df: pd.DataFrame, *, file_name: str, output_dir: str | Path,
    model: str | None = None, max_iterations: int | None = None,
    allow_imputation: bool = False, context_metadata: dict | None = None,
    selected_actions: list[dict[str, Any]] | None = None,
    profile_mode: str = "fast",
) -> dict[str, Any]:
    """Profile -> deterministic allow-listed actions -> re-profile -> audit."""
    if max_iterations is not None and max_iterations < 1:
        raise ValueError("max_iterations must be positive when supplied")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    specs, handlers = build_action_registry()
    if profile_mode not in {"fast", "full"}:
        raise ValueError("profile_mode must be 'fast' or 'full'")
    working = df if profile_mode == "fast" else df.copy(deep=True)
    before_profile = profile_dataframe(working, mode=profile_mode)
    decisions = selected_actions if selected_actions is not None else _select_actions(
        before_profile, specs, allow_imputation=allow_imputation, profile_mode=profile_mode
    )
    spec_map = {s.name: s for s in specs}
    run_id = uuid.uuid4().hex
    actions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    selected_decisions = decisions if max_iterations is None else decisions[:max_iterations]
    for iteration, decision in enumerate(selected_decisions, start=1):
        name = decision.get("action")
        params = decision.get("parameters") or {}
        if name not in spec_map or name not in handlers:
            actions.append({"iteration": iteration, "action": name, "status": "blocked_unregistered"})
            continue
        if not isinstance(params, dict):
            actions.append({"iteration": iteration, "action": name, "status": "blocked_invalid_parameters"})
            continue
        spec = spec_map[name]
        if spec.requires_approval:
            actions.append({
                "iteration": iteration, "action": name, "status": "approval_required",
                "rule_id": decision.get("rule_id"),
            })
            continue
        key = (name, str(params))
        if key in seen:
            continue
        seen.add(key)
        action_id = uuid.uuid4().hex
        started = time.time()
        before = before_profile if profile_mode == "fast" else profile_dataframe(working, mode=profile_mode)
        try:
            result = handlers[name](
                working, parameters=params, archive_dir=archive,
                run_id=run_id, action_id=action_id,
            )
            status = result.get("status", "failed")
            if status == "committed":
                working = result.get("dataset", working)
            after = before_profile if profile_mode == "fast" else profile_dataframe(working, mode=profile_mode)
            record = {
                "run_id": run_id, "action_id": action_id, "iteration": iteration,
                "action": name, "module": spec.module, "status": status,
                "rule_id": decision.get("rule_id"),
                "rationale": decision.get("rationale", ""),
                "parameters": params, "summary": result.get("summary", ""),
                "changed_records": int(result.get("changed_records", 0)),
                "archived_records": int(result.get("archived_records", 0)),
                "details": result.get("details", {}),
                "before_profile": before, "after_profile": after,
                "elapsed_seconds": round(time.time() - started, 4),
                "requires_human_validation": True,
            }
            if result.get("error"):
                record["error"] = str(result["error"])
            actions.append(record)
            with (archive / f"{run_id}_action_log.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, default=str) + "\n")
            if status != "committed":
                break
        except Exception as exc:
            actions.append({
                "run_id": run_id, "action_id": action_id, "iteration": iteration,
                "action": name, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "before_profile": before, "requires_human_validation": True,
            })
            break

    output_path = root / f"{Path(file_name).stem.replace(' ', '_')}_processed.csv"
    temp_path = output_path.with_suffix(".tmp")
    working.to_csv(temp_path, index=False)
    temp_path.replace(output_path)
    final_profile = before_profile if profile_mode == "fast" else profile_dataframe(working, mode=profile_mode)
    report = {
        "run_id": run_id, "file_name": file_name,
        "decision_engine": "deterministic_rules_v1", "profile_mode": profile_mode,
        "model": None, "rows_before": len(df), "rows_after": len(working),
        "row_count_preserved": len(df) == len(working),
        "context_metadata_tables": list((context_metadata or {}).get("tables", {}).keys()),
        "before_profile": before_profile, "final_profile": final_profile,
        "actions": actions, "output_file": str(output_path),
        "archive_dir": str(archive),
        "clinical_validity": "not_established",
        "training_label_status": "candidate_requires_review",
    }
    report_path = root / f"{Path(file_name).stem}_preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_file"] = str(report_path)
    return report
