# TrippinnI — Qwen3:32B profile-bundle experiment

This is an isolated experiment on branch `feature/qwen3-profile-bundle-pipeline`.
It does not change the production profiling or preprocessing path on `main`.

## Design

The profiler continues to emit one JSON report per table. Before calling the local model,
`build_profile_bundle.py` combines every report into **one JSON input file**:

`artifacts/profiles_bundle.json`

The bundle keeps each original report intact under `tables[table_name].profile`, and records
the source filename, SHA-256 digest, report version and generation timestamp. Duplicate table
names are retained with a filename-derived suffix and listed in
`duplicate_table_name_collisions`.

Qwen is then called locally through Ollama in two stages:
1. One analysis request per table: metadata interpretation, quality interpretation, algorithm
   proposal and audit-event specification.
2. One cross-table request: candidate relationships and recommended integrity checks.

The final model output is saved to `artifacts/qwen_analysis.json`.

## Windows PowerShell

Run from the repository root. Python 3.10+ is recommended.

```powershell
# 1. Put all profiler JSON reports (one per table) in this directory.
New-Item -ItemType Directory -Force .\artifacts\profiles | Out-Null

# 2. Build one combined JSON input file.
python .\experiments\qwen3_profile_pipeline\scripts\build_profile_bundle.py `
  --input-dir .\artifacts\profiles `
  --output .\artifacts\profiles_bundle.json

# 3. Ensure Ollama is running and the model is available.
ollama list
ollama run qwen3:32b

# 4. In another PowerShell window, run the pipeline.
python .\experiments\qwen3_profile_pipeline\scripts\run_qwen3_pipeline.py `
  --bundle .\artifacts\profiles_bundle.json `
  --output .\artifacts\qwen_analysis.json
```

The scripts use only the Python standard library. Ollama's local API defaults to
`http://localhost:11434`. Use `--timeout` to increase the per-request timeout.

## Profile bundle shape

```json
{
  "bundle_schema_version": "1.0.0",
  "bundle_type": "trippinni_multi_table_profiler_bundle",
  "created_at": "UTC timestamp",
  "source_count": 2,
  "table_count": 2,
  "duplicate_table_name_collisions": [],
  "tables": {
    "admissions": {
      "table_name": "admissions",
      "source_file": "admissions_profile.json",
      "source_sha256": "sha256 digest",
      "report_version": "report version",
      "generated_at": "profile timestamp",
      "profile": {}
    },
    "patients": {
      "table_name": "patients",
      "source_file": "patients_profile.json",
      "source_sha256": "sha256 digest",
      "report_version": "report version",
      "generated_at": "profile timestamp",
      "profile": {}
    }
  }
}
```

The `profile` object is the original profiler JSON, not a flattened or lossy projection.
No clinical row-level data is loaded by these scripts; only JSON profiler reports are read.

## Algorithm safety boundary

The experiment's initial allow-list reflects the prototype registry:
- `report_missingness`: reporting only.
- `report_exact_duplicates`: reporting only; no row deletion.
- `flag_iqr_outliers`: flagging only.
- `isolation_forest_flags`: flagging only.
- `median_imputation`: approval-gated; original missing cells must be archived.

Mean, mode, KNN, MICE, MissForest, group-wise and temporal imputation are marked
`proposed_not_implemented` in this experiment. They are not executable actions.
Verify this registry against the current code before promoting the experiment.

## Audit distinction

Qwen generates an **audit specification**: event types, required fields, decision rationale,
algorithm identifier/version, parameters, approvals, provenance and validation requirements.

The application must generate actual runtime audit records when work occurs, including real
timestamps, row counts, changed-cell counts, duration, success/failure and output checksums.
The model must never invent these runtime facts.

## MIMIC-IV / clinical-data safeguards

- Feed profiler summaries, not patient-level clinical rows, to the model.
- Treat primary/foreign keys and shared identifiers as candidates until integrity checks verify them.
- Preserve exactness flags and streaming summary scope; bounded samples are not full distributions.
- Do not infer MCAR/MAR/MNAR from missingness percentages alone.
- Do not treat empirical ranges as clinical reference ranges.
- Do not numerically coerce identifiers or medical codes.
- Do not allow fuzzy matching to trigger patient merges.
- Keep pre-execution classifier features separate from post-execution outcomes and audit events
  to prevent target leakage.

## Known limitations of this first experiment

- The bundle is a single JSON document. Very large bundles may exceed the model's context window;
  the runner therefore analyses tables separately, but still loads the bundle locally in Python.
- Cross-table analysis uses the model-generated per-table analyses, not raw clinical data.
- JSON mode encourages valid JSON but does not guarantee schema compliance. Review output before
  using it. This prototype currently validates JSON syntax and object shape, not every nested field.
- No algorithm is executed by this pipeline.
