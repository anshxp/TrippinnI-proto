# Recursive multi-table extraction and preprocessing

This branch uses deterministic, allow-listed rules. It does not invoke Qwen,
Ollama, an LLM planner, or a network model API.

## Run the entire nested dataset at once

Pass the **top-level directory** that contains the nested dataset folders.
The program recursively discovers every `.csv` and `.csv.gz` file.

```powershell
python app.py --dataset-type csv --dataset-path "C:\Users\Admin\Documents\physionet.org"
```

Do not pass a made-up example file path. Do not pass the path to one table.
Pass the directory containing the dataset's subfolders.

By default, the pipeline:
1. Recursively discovers all CSV and CSV.GZ tables.
2. Detects a MIMIC-IV `patients` table, if present.
3. Selects a reproducible 10% cohort of unique `subject_id` values (seed 42).
4. Filters tables by `subject_id`, or by linked `hadm_id` / `stay_id` values
   derived from `hosp/admissions` and `icu/icustays`.
5. Copies keyless reference/dictionary tables in full (for example, `d_items`);
   these are lookup tables, not patient records.
6. Writes every extracted table to a separate output tree.
7. Profiles and preprocesses all rows of every extracted table in bounded-memory
   chunks (default 50,000 rows per chunk; no total 5,000-row cap).
8. Writes per-table reports, audit JSONL, candidate classifier records, and one
   multi-table manifest.

The selected patient cohort is 10% of patients. The resulting row fraction will
not be exactly 10% for every table: each patient's number of admissions, ICU
stays, and events differs. This is expected and is preferable to independently
sampling 10% of each table, which would break patient-level relationships.

If no `patients` table is found, the fallback extracts the first 10% of rows
from each discovered file independently. That fallback does **not** guarantee
relational integrity.

## Output layout

Default output root: `outputs/manual_batches` (ignored by Git).

- `extracted/`: physical extracted CSV tables, preserving source subfolders
- `processed/`: preprocessed tables and per-chunk reports
- `audit/batch_audit.jsonl`: per-table extraction and processing audit
- `classifier_training/algorithm_selection_records.jsonl`: candidate records
- `multi_table_manifest.json`: selected cohort counts, per-table row counts,
  sampling method, output paths, and limitations

Use a new, empty output directory for each run. The program refuses to overwrite
an output directory that already contains files.

## Options

```powershell
# Extract a 5% patient cohort instead of 10%
python app.py --dataset-type csv --dataset-path "C:\Users\Admin\Documents\physionet.org" --sample-fraction 0.05

# Use a different deterministic patient cohort
python app.py --dataset-type csv --dataset-path "C:\Users\Admin\Documents\physionet.org" --random-seed 123

# Put generated data on another drive
python app.py --dataset-type csv --dataset-path "C:\Users\Admin\Documents\physionet.org" --output-root "F:\TrippinnI-work\mimic_10pct"
```

## Important limitations and review

This implementation uses the standard MIMIC-IV patient, admission, and ICU-stay
keys. It filters rows using the available key columns; it does not yet validate
every table-specific foreign-key constraint or guarantee completeness for
unusual/custom tables. Review the manifest and table counts before downstream
use. Tables without patient keys are copied in full as reference data.

Default rules report missingness and exact duplicate candidates and add IQR
outlier flags to eligible numeric columns. They do not delete rows or overwrite
clinical values. Imputation remains off unless explicitly enabled after human
approval.

Classifier records are candidate algorithm-selection examples, not ground-truth
labels. A successful execution or a flagged outlier does not establish clinical
validity. Validate the extracted cohort and preprocessing outputs before
classifier training. Keep source data and generated patient-level artifacts
outside Git.
