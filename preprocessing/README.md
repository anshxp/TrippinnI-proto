# Preprocessing agent

- `agent.py`: Ollama/Qwen planner and generic guarded loop.
- `registry.py`: allow-listed Rule and ML handlers.
- `runtime.py`: working DataFrame/chunk runner with iterative calls, per-action logs, per-file report, and processed output.
- `DECISION_RULES.md`: decision catalogue and clinical-data guardrails.

## Local setup

1. Start Ollama and confirm `ollama list` includes `qwen3:4b`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the runtime against a bounded DataFrame/chunk from your application:

```python
from preprocessing.runtime import run_dataframe

report = run_dataframe(
    df,  # a bounded pandas DataFrame, not an unbounded EHR table
    file_name="example_chunk.csv",
    output_dir="F:/TrippinnI-work/preprocessing",
    model="qwen3:4b",
    max_iterations=5,
)
print(report["report_file"])
```

The runner sends compact profile metadata and an action allow-list to the LLM,
not the source rows. Qwen proposes one registered tool at a time. Python
validates the tool name and parameters, executes the registered handler,
re-profiles the returned DataFrame, and then asks Qwen for the next action.
It writes a processed CSV, per-file JSON report, JSONL action log, and a
separate archive for replaced values.

## Initial tools

- Rule: missingness report, exact duplicate report (flag-only), IQR outlier flags.
- ML: Isolation Forest anomaly flags.
- Median imputation is excluded by default. It can only be enabled explicitly
  with `allow_imputation=True`; this is a prototype switch, not a clinical
  approval workflow.

The current handlers are conservative: duplicate and anomaly methods flag
records rather than delete them. The archive currently captures cells replaced
by median imputation; flag-only methods do not remove or overwrite source values.
Original input DataFrames are copied before processing.

## Important integration boundary

This is an opt-in runner for a bounded DataFrame or chunk. It is not yet wired
into `core/orchestrator.py` or the full chunk-streaming file writer. Do not
pass an entire large MIMIC-IV table into memory. Next, connect the runner to
the existing chunk reader and atomically write a versioned output per file,
with dataset-level rollback and postcondition tests before using it on real EHR
data. Use synthetic/de-identified test fixtures, and never commit clinical data
or archives to Git.
