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

## Chunked CSV entry point

Run a large CSV without loading the entire source table into RAM:

```powershell
python -m preprocessing.csv_runtime path/to/input.csv --output-dir outputs/preprocessing --chunksize 50000 --model qwen3:4b
```

The source is read in bounded chunks and capped by an explicit row limit (--max-rows, default 5,000). Each chunk receives its own profile, planner decisions, action report, and archive; processed chunks are then streamed into one output CSV. The writer aligns the union of generated columns so an action-specific flag added in one chunk does not shift values in later chunks. A chunk-index JSON report records the run reports and row counts. The orchestrator uses the configured first 10% prefix for tables with known row counts and skips tables whose row count cannot be established safely.

This is still an opt-in stage and does not alter the default profiling orchestrator. Since planning and statistics are chunk-local, action choices and thresholds can differ across chunks. This implementation does not yet provide dataset-global thresholds, rollback across all chunk archives, clinical approval, or a validated EHR remediation workflow. Use synthetic/de-identified fixtures, and never commit clinical data or archives to Git.

Median imputation remains disabled by default. To explicitly enable it for an experiment, add `--allow-imputation`; this switch is not a substitute for clinical or data-owner approval.

## Orchestrator integration

The orchestrator runs preprocessing only after profiling all tables and building the semantic context/knowledge graph. For each explicitly allowlisted table, it processes only the deterministic first 10% of rows (using known MIMIC-IV v3.1 row counts or a bounded count of small files). The LLM receives aggregate profiles, schema semantics, and candidate links for the target table and directly connected tables—not source rows. The feature is disabled by default and writes only under `outputs/preprocessing/<table>/`; it does not replace source files or change profiling results.

Enable it for an explicit run in PowerShell:

```powershell
$env:TRIPPINNI_PREPROCESSING_ENABLED = "1"
$env:TRIPPINNI_PREPROCESSING_CHUNK_SIZE = "1000"
$env:TRIPPINNI_PREPROCESSING_TABLES = "hosp_patients,hosp_admissions"
$env:TRIPPINNI_PREPROCESSING_MODEL = "qwen3:8b"  # 32 GB system; use qwen3:4b on 8 GB
$env:TRIPPINNI_OLLAMA_NUM_CTX = "4096"           # use 2048 on 8 GB
$env:TRIPPINNI_PREPROCESSING_MAX_ITERATIONS = "2"
python app.py
```

To permit numeric median imputation, set `TRIPPINNI_PREPROCESSING_ALLOW_IMPUTATION=1` only after explicit data-owner review. It remains off by default. If the processed CSV and chunk report already exist, the orchestrator skips the preprocessing run; delete those generated outputs to force a rerun. A preprocessing error is logged and does not invalidate a successful profiling checkpoint.

**Performance warning:** do not allowlist the entire MIMIC-IV corpus on the first run. Start with one or two tables, a 1,000-row chunk size, and two iterations. The 10% prefix can still be large for high-volume tables, and the planner is called per chunk. This remains experimental and does not provide global cross-chunk thresholds or clinical validation.

## Hardware profiles and staged testing

Choose the model explicitly on each machine. Ollama may use system RAM and GPU memory, so the repository does not guess the model from installed RAM.

### 32 GB system

```powershell
ollama pull qwen3:8b
$env:TRIPPINNI_LLM_MODEL = "qwen3:8b"
$env:TRIPPINNI_PREPROCESSING_MODEL = "qwen3:8b"
$env:TRIPPINNI_OLLAMA_NUM_CTX = "4096"
```

### 8 GB system

```powershell
ollama pull qwen3:4b
$env:TRIPPINNI_LLM_MODEL = "qwen3:4b"
$env:TRIPPINNI_PREPROCESSING_MODEL = "qwen3:4b"
$env:TRIPPINNI_OLLAMA_NUM_CTX = "2048"
```

These are conservative starting values, not performance guarantees. On the 8 GB machine, close memory-heavy applications and reduce chunk size to 500–1000 if the machine swaps or runs out of memory.

### Test sequence

1. Confirm the model is available with `ollama list`. Then smoke-test the API by running `ollama run qwen3:4b` on the 8 GB system or `ollama run qwen3:8b` on the 32 GB system and asking it to return a small JSON object.
2. Create a tiny synthetic CSV (no patient data):

   ```powershell
   @'
   patient_id,age,heart_rate,ward
   p001,45,78,ICU
   p002,,82,ICU
   p003,52,250,WARD
   p003,52,250,WARD
   p004,38,74,WARD
   '@ | Set-Content -Encoding utf8 .\smoke_test.csv
   ```

3. Test preprocessing directly before running the full application:

   ```powershell
   python -m preprocessing.csv_runtime .\smoke_test.csv --output-dir .\outputs\smoke_test --chunksize 5 --max-rows 5 --model $env:TRIPPINNI_PREPROCESSING_MODEL --max-iterations 2
   ```

4. Check output row count and source preservation:

   ```powershell
   python -c "import pandas as pd; a=pd.read_csv('smoke_test.csv'); b=pd.read_csv('outputs/smoke_test/smoke_test_processed.csv'); print('input rows:',len(a),'output rows:',len(b)); assert len(a)==len(b)"
   ```

   Review the chunk report, per-chunk JSON reports, and action logs. Qwen may correctly choose `stop`, so a flag column is not required for a successful smoke test.

5. Only after the direct test passes, try the orchestrator with a single allowlisted table:

   ```powershell
   $env:TRIPPINNI_PREPROCESSING_ENABLED = "1"
   $env:TRIPPINNI_PREPROCESSING_TABLES = "hosp_patients"
   $env:TRIPPINNI_PREPROCESSING_CHUNK_SIZE = "1000"
   $env:TRIPPINNI_PREPROCESSING_MAX_ITERATIONS = "2"
   python app.py
   ```

   Use a table name present in the loader. Important: `app.py` still runs the main profiling pipeline across all tables; the allowlist restricts only LLM preprocessing. Use step 3 for a preprocessing-only smoke test.

### Safety and limits

- Preprocessing remains disabled by default. When enabled in the orchestrator, only table names listed in `TRIPPINNI_PREPROCESSING_TABLES` are eligible; an empty allowlist means no table is preprocessed.
- Keep median imputation disabled for the initial tests.
- Each chunk is planned independently. Thresholds and chosen actions can differ between chunks; this is not yet a globally consistent clinical remediation pipeline.
- Do not commit MIMIC-IV data, processed outputs, archives, or reports to Git.
- Successful execution and row preservation do not establish clinical validity. Review the action logs, schema changes, row counts, missingness, and flags before downstream use.
