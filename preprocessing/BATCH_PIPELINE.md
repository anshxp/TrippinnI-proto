# Deterministic extraction and algorithm-selection data generation

This branch uses deterministic, allow-listed preprocessing rules. The batch
workflow does not invoke Qwen, Ollama, an LLM planner, or a network model API.

## Run it through app.py

Activate the project's Python 3.12 virtual environment, then run:

```powershell
python app.py --dataset-type csv --dataset-path "F:/path/to/table.csv.gz"
```

By default this:
1. Counts source data rows.
2. Physically extracts the first 10% into a new CSV under
   `outputs/manual_batches/<table>_batch_001/`.
3. Profiles and processes **all rows in that extracted CSV** in bounded-memory
   chunks (default 1,000 rows per chunk; no 5,000-row total cap).
4. Writes processed data, per-chunk profiles/action reports, a batch manifest,
   an audit JSONL, and candidate classifier-training JSONL records.

The source file is read-only and is not overwritten.

### Useful options

```powershell
# Next 10% of the same source table (non-overlapping row window)
python app.py --dataset-type csv --dataset-path "F:/path/to/table.csv.gz" --batch-index 1

# Extract 5% per batch and process in 2,000-row chunks
python app.py --dataset-type csv --dataset-path "F:/path/to/table.csv" --sample-fraction 0.05 --chunksize 2000

# Choose a separate output location
python app.py --dataset-type csv --dataset-path "F:/path/to/table.csv.gz" --output-root "F:/TrippinnI-work/manual_batches"
```

Do not reuse an existing batch output folder. The pipeline fails rather than
silently overwriting it. Use a new output root for a fresh run.

## Outputs

- Extracted source rows: `<batch>/<batch>_raw.csv`
- Processed rows: `<batch>/preprocessing/<table>_processed.csv`
- Per-chunk before/after profiles and action reports
- Batch manifest: `<batch>/batch_manifest.json`
- Run audit: `audit/batch_audit.jsonl`
- Candidate classifier records: `classifier_training/algorithm_selection_records.jsonl`

Default deterministic rules always report missingness and exact duplicate
candidates, and add IQR outlier flags for eligible numeric columns. They do not
delete rows or overwrite clinical values. Median imputation is not selected
automatically; it requires the explicit `--allow-imputation` switch, and should
only be enabled after a separate human approval and clinical policy decision.

## Important MIMIC-IV limitation

This is currently a **single-table CSV/CSV.GZ row-window extractor**, not a
relational MIMIC-IV sampler. Sequential row windows can split a patient's
records across batches and can separate `subject_id`, `hadm_id`, and
`stay_id` relationships. The manifest records this limitation. Do not treat
the resulting batches as patient-level train/test splits or as relationally
complete MIMIC-IV subsets. A coordinated patient-key sampler across all
dependent tables is a separate required implementation.

## Interpretation and safety

Candidate algorithm-selection records are not ground-truth labels. A successful
execution, a flagged outlier, or a changed missingness rate does not establish
clinical validity or data correctness. Review candidate records and validate
the result against task-specific clinical/data-governance rules before
classifier training. Keep source data, extracted PHI, and generated patient-level
artifacts outside Git.
