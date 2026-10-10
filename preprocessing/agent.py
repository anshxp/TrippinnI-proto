"""Guarded iterative preprocessing agent for TrippinnI.

The agent asks a local Ollama model to choose from an explicit action registry.
It never passes raw dataset rows to the model, and it does not execute arbitrary
code returned by the model. Registered actions are supplied by the caller.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import requests


@dataclass(frozen=True)
class ActionSpec:
    """An allow-listed operation available to the LLM planner."""

    name: str
    module: str  # rule, ml, or dl
    description: str
    risk: str = "low"  # low, review, or high
    requires_approval: bool = False


@dataclass
class ActionResult:
    action_id: str
    action: str
    module: str
    status: str
    summary: str
    before_profile: dict[str, Any]
    after_profile: dict[str, Any] | None = None
    changed_records: int = 0
    archived_records: int = 0
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0


class OllamaPlanner:
    """Small local planner client; default model is Qwen3 4B."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 180,
    ) -> None:
        self.model = model or os.getenv("TRIPPINNI_LLM_MODEL", "qwen3:4b")
        self.base_url = (base_url or os.getenv(
            "TRIPPINNI_OLLAMA_URL", "http://localhost:11434"
        )).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def choose_action(
        self,
        profile: dict[str, Any],
        actions: list[ActionSpec],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return one JSON action decision; never return executable code."""
        action_catalog = [asdict(item) for item in actions]
        system_prompt = (
            "You are TrippinnI's healthcare data preprocessing planner. "
            "Choose at most ONE action from the supplied allow-list per turn. "
            "Never invent action names or parameters. Preserve clinical meaning, "
            "patient/encounter links, repeated measurements, and temporal order. "
            "Do not recommend deletion, imputation, capping, or value correction "
            "unless the selected registered action explicitly supports it and "
            "its policy permits it. If evidence is insufficient, choose stop or "
            "review. Return JSON only with keys action, parameters, rationale, "
            "stop. Use action='stop' when no justified action remains."
        )
        user_payload = {
            "profile_summary": profile,
            "available_actions": action_catalog,
            "recent_action_history": history[-8:],
            "required_output_schema": {
                "action": "registered action name or stop",
                "parameters": {},
                "rationale": "brief evidence-based explanation",
                "stop": False,
            },
        }
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, default=str)},
                ],
                "options": {"temperature": 0.1, "num_ctx": 4096},
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        decision = json.loads(content)
        if not isinstance(decision, dict):
            raise ValueError("Planner response must be a JSON object.")
        return decision


class IterativePreprocessingAgent:
    """Profile -> plan -> execute one allow-listed action -> re-profile.

    profile_fn receives the current working dataset and returns compact
    metadata only. action_handlers map registered action names to callables.
    Handlers should work on a versioned copy and archive original values before
    any mutation. This class deliberately does not guess how to mutate arbitrary
    EHR tables.
    """

    def __init__(
        self,
        planner: OllamaPlanner,
        actions: list[ActionSpec],
        action_handlers: dict[str, Callable[..., dict[str, Any]]],
        profile_fn: Callable[[Any], dict[str, Any]],
        archive_dir: str | Path,
        max_iterations: int = 5,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        self.planner = planner
        self.actions = {item.name: item for item in actions}
        self.action_handlers = action_handlers
        self.profile_fn = profile_fn
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.max_iterations = max_iterations
        missing = set(self.actions) - set(self.action_handlers)
        if missing:
            raise ValueError(f"Missing handlers for registered actions: {sorted(missing)}")

    def run(self, working_dataset: Any) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        history: list[dict[str, Any]] = []
        reports: list[ActionResult] = []
        profile = self.profile_fn(working_dataset)

        for iteration in range(1, self.max_iterations + 1):
            decision = self.planner.choose_action(profile, list(self.actions.values()), history)
            action_name = decision.get("action")
            if decision.get("stop") is True or action_name == "stop":
                break
            if action_name not in self.actions:
                raise ValueError(f"Planner selected an unregistered action: {action_name!r}")

            spec = self.actions[action_name]
            if spec.requires_approval:
                history.append({
                    "iteration": iteration,
                    "action": action_name,
                    "status": "approval_required",
                    "rationale": str(decision.get("rationale", "")),
                })
                break

            started = time.time()
            action_id = uuid.uuid4().hex
            before = profile
            result_record = ActionResult(
                action_id=action_id,
                action=action_name,
                module=spec.module,
                status="running",
                summary=str(decision.get("rationale", "")),
                before_profile=before,
                started_at=started,
            )
            try:
                result = self.action_handlers[action_name](
                    working_dataset,
                    parameters=decision.get("parameters") or {},
                    archive_dir=self.archive_dir,
                    run_id=run_id,
                    action_id=action_id,
                )
                if not isinstance(result, dict) or "status" not in result:
                    raise ValueError("Action handler must return a dict with a status field.")
                result_record.status = str(result["status"])
                result_record.summary = str(result.get("summary", result_record.summary))
                result_record.changed_records = int(result.get("changed_records", 0))
                result_record.archived_records = int(result.get("archived_records", 0))
                if result_record.status == "committed":
                    if "dataset" in result:
                        working_dataset = result["dataset"]
                    profile = self.profile_fn(working_dataset)
                    result_record.after_profile = profile
                else:
                    result_record.error = result.get("error")
            except Exception as exc:
                result_record.status = "failed"
                result_record.error = f"{type(exc).__name__}: {exc}"
            finally:
                result_record.finished_at = time.time()
                reports.append(result_record)
                record = {"run_id": run_id, "iteration": iteration, **asdict(result_record)}
                log_path = self.archive_dir / f"{run_id}_action_log.jsonl"
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, default=str) + "\n")
                history.append({
                    "iteration": iteration,
                    "action": action_name,
                    "module": spec.module,
                    "status": result_record.status,
                    "summary": result_record.summary,
                    "error": result_record.error,
                })

            if result_record.status != "committed":
                break

        return {
            "run_id": run_id,
            "iterations": len(reports),
            "stopped": len(reports) < self.max_iterations,
            "final_profile": profile,
            "actions": [asdict(item) for item in reports],
            "action_log": str(self.archive_dir / f"{run_id}_action_log.jsonl"),
        }
