# Deterministic extraction and algorithm-selection data generation

This workflow does not use Qwen, Ollama, an LLM planner, or network model calls.
A deterministic rule set chooses only allow-listed actions and records the rule
ID, rationale, parameters, profiles before/after execution, and execution status.

## First batch (10%)

```powershell
python -m preprocessing.batch_pipeline "F:/path/to/table.csv.gz" --batch-index 0 --output-root "F:/TrippinnI-work/manual_batches" --chunksize 1000
```

Second batch: use `--batch-index 1`. Each batch is a non-overlapping sequential
row window. Existing batch folders are not overwritten.

## Outputs

- Raw extracted rows: `<batch>/<batch>_raw.csv`
- Processed rows: `<batch>/preprocessing/*_processed.csv`
- Per-chunk profiles and action reports
- `audit/batch_audit.jsonl`
- `classifier_training/algorithm_selection_records.jsonl`

The default rules always profile missingness and exact duplicates, then add
IQR outlier flags for numeric columns with at least four distinct values.
These are candidate algorithm-selection examples, not ground-truth labels.
No rows or clinical values are deleted or corrected by these default actions.
Median imputation remains approval-gated and is not automatically selected.

## Limitations

Sequential row windows are not patient/encounter-aware. Do not use this sampling
method as a patient-level MIMIC-IV split: linked subject_id/hadm_id/stay_id
records can be separated. This pipeline currently accepts CSV/CSV.GZ tables,
not a coordinated multi-table relational extraction.

A successful action or a lower missingness rate does not establish clinical
validity. Human review and task-specific validation are required before turning
these candidate records into supervised classifier labels. Keep PHI and generated
patient-level artifacts outside Git.
