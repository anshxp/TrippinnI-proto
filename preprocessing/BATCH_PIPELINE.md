# Sequential batch extraction and audited preprocessing

This branch adds an explicit experimental workflow that extracts the actual rows
for each 10% batch before preprocessing. It is separate from the default
profiling orchestrator; existing profiling behavior is not replaced.

## Run one batch at a time

PowerShell:

```powershell
# First 10%
python -m preprocessing.batch_pipeline "F:/path/to/table.csv.gz" --batch-index 0 --output-root "F:/TrippinnI-work/manual_batches" --chunksize 1000 --max-iterations 2

# Second 10% (same source and output root)
python -m preprocessing.batch_pipeline "F:/path/to/table.csv.gz" --batch-index 1 --output-root "F:/TrippinnI-work/manual_batches" --chunksize 1000 --max-iterations 2

# Third 10%
python -m preprocessing.batch_pipeline "F:/path/to/table.csv.gz" --batch-index 2 --output-root "F:/TrippinnI-work/manual_batches" --chunksize 1000 --max-iterations 2
```

Batch indices are zero-based. The default fraction is 10%; the last batch may
contain fewer rows. Extraction is bounded-memory and writes a standalone raw
batch CSV. The source file is never modified. The extracted file is then passed
to the existing guarded preprocessing runtime, which processes it in smaller
chunks. The LLM receives aggregate profiles and action history, not source rows.

## Outputs

For each batch:

- `<batch>_raw.csv`: the actual extracted input rows.
- `preprocessing/*_processed.csv`: transformed batch output.
- Per-chunk JSON reports, JSONL action logs, and replacement archives.
- `batch_manifest.json`: source fingerprint, row boundaries, and artifact paths.

Across batches:

- `audit/batch_audit.jsonl`: batch-level provenance and row-preservation checks.
- `classifier_training/algorithm_selection_records.jsonl`: structured records
  containing metadata features, selected registered action, parameters,
  before/after profiles, and execution outcomes. Raw patient rows are excluded.

## Important design constraints

1. Process batches in order and use the same source file version. The source
   fingerprint is checked to prevent silently mixing dataset versions.
2. This first implementation runs each batch independently. It does **not**
   learn global imputation values, category vocabularies, scaling parameters,
   or outlier thresholds from batch 1 and reuse them automatically. Those
   transformations need fitted state and an explicit fit/transform contract
   before the workflow is suitable for consistent model training.
3. Do not treat each batch as an independent patient sample if patient or
   encounter records are relationally linked. For MIMIC-IV, preserve the
   relevant subject_id/hadm_id/stay_id grouping when defining the split; a
   simple row-offset split can divide one patient's timeline across batches.
4. Algorithm-selection records are candidate training examples, not validated
   labels. A successful execution or lower missingness does not establish
   clinical correctness. Review labels and outcomes, record human approval,
   and evaluate with patient-level splits before training a classifier.
5. Imputation remains disabled unless explicitly enabled. Start with synthetic
   or de-identified data, inspect reports and archives, and never commit clinical
   data or generated patient-level artifacts to Git.
6. Re-running an existing batch is blocked to prevent duplicate audit/training records.
   Use a new output root for a clean experiment.

## Example progression

Run batch 0, inspect its raw extraction, processed output, manifest, audit, and
action reports. Then run batch 1. Repeat until the required dataset fraction has
been covered. The JSONL training file grows incrementally, but classifier
training should wait until records have been reviewed and leakage-safe labels
have been established.
