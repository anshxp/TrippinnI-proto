# Preprocessing

- agent.py: local Ollama planner and bounded iterative orchestration loop.
- DECISION_RULES.md: initial Rule/ML/DL catalogue, transformation guardrails,
  archive requirements, and imputation policy.

The agent uses qwen3:4b by default (TRIPPINNI_LLM_MODEL) and calls the local
Ollama chat API (TRIPPINNI_OLLAMA_URL, default http://localhost:11434).
It sends profile summaries and action metadata, not complete source tables.

The agent is not yet wired into core/orchestrator.py. Before enabling it in the
main pipeline, implement and test handlers that:
1. use the existing profiling and detection APIs,
2. write to a versioned working copy,
3. archive removed/replaced values before mutation,
4. validate postconditions and commit/rollback atomically, and
5. emit file-level reports without including raw patient data in ordinary logs.
