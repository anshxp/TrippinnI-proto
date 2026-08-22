# CURRENT SYSTEM AUDIT

This audit reviews the repository as a healthcare data-quality and preprocessing platform intended to support MIMIC‑IV datasets. It documents architecture, implemented pipeline stages, MIMIC‑IV-specific support (tables/fields), all implemented data-quality rules (exactly as implemented), traceability, safety risks, security/compliance observations, missing components, tests and reproducibility, and prioritized next actions. Findings cite exact files and line ranges.

---

**1) Architecture (frontend, backend, database, scripts, APIs, execution flow)**

- **Frontend:** not found — there is no web UI or frontend app in the repository. (No files under `ui/` implementing a served frontend; `ui/streamlit_app.py` exists but is not a complete server-based front-end implementation.)
- **Backend / orchestration:** the main pipeline is implemented as `Pipeline` -> `Orchestrator` -> loader + profiler.
  - `app.py` shows the intended entrypoint that creates a `Pipeline` and runs it with `dataset_type="mimic"`, `dataset_path="data/raw/mimiciv"` ([app.py](app.py#L1-L20)).
  - `core/pipeline.py` constructs `Orchestrator` and calls `Orchestrator.initialize()` ([core/pipeline.py](core/pipeline.py#L1-L30)).
  - `core/orchestrator.py` handles loader initialization and dataset profiling ([core/orchestrator.py](core/orchestrator.py#L1-L120)). The orchestrator profiles each table using `DatasetProfilerEngine` and handles chunked profiling when a loader exposes `get_dataframe_chunks` ([core/orchestrator.py](core/orchestrator.py#L41-L60)).
- **Loaders & readers:** the loader registry includes a `MimicLoader` implementation and discovery/reader layers:
  - `loaders/registry.py` lists supported loaders and registers `MimicLoader` ([loaders/registry.py](loaders/registry.py#L1-L40)).
  - `loaders/mimic_loader.py` implements the loader that discovers files, registers them in a `DatasetCatalog`, and lazily loads or streams CSVs via `CsvReader` ([loaders/mimic_loader.py](loaders/mimic_loader.py#L12-L20), [loaders/mimic_loader.py](loaders/mimic_loader.py#L50-L60), [loaders/mimic_loader.py](loaders/mimic_loader.py#L80-L90), [loaders/mimic_loader.py](loaders/mimic_loader.py#L109-L116), [loaders/mimic_loader.py](loaders/mimic_loader.py#L153-L160)).
  - `loaders/readers/csv_reader.py` reads CSV / `.csv.gz` files and exposes `read()` and `read_chunks()` ([loaders/readers/csv_reader.py](loaders/readers/csv_reader.py#L1-L40), [loaders/readers/csv_reader.py](loaders/readers/csv_reader.py#L40-L100)).
- **Profiling & detectors:** profiling engine and detectors are implemented as separate modules.
  - `profiling/profiler.py` composes `DatasetProfiler`, `ColumnProfiler`, `MemoryProfiler`, and `KeyDetector`, and writes reports with `ReportWriter` ([profiling/profiler.py](profiling/profiler.py#L1-L40)).
  - `profiling/report_writer.py` persists profiling JSON files to `outputs/reports/profiling` ([profiling/report_writer.py](profiling/report_writer.py#L5-L18), [profiling/report_writer.py](profiling/report_writer.py#L24-L24)).
  - The quality detection orchestration is implemented in `quality/detector.py` and coordinates detectors (Missing, Duplicate, Datatype, Outlier) but is not wired automatically into the top-level `Pipeline` (`Orchestrator.initialize` only triggers profiling) ([quality/detector.py](quality/detector.py#L1-L80)).
- **Database:** no database integration or persistent DB connectors were found (no SQLAlchemy, psycopg, or other DB client code). Data is read directly from filesystem CSV/JSON files under `data/`.
- **APIs / network:** no HTTP API endpoints or server framework found. There is an LLM component that loads transformer models in-process (see `ml/llm/huggingface_llm.py`) which may download model weights from a remote repository when `from_pretrained()` is called ([ml/llm/huggingface_llm.py](ml/llm/huggingface_llm.py#L1-L30)).

---

**2) Implemented data pipeline stages**

Summary table: implemented vs partially/not implemented.

| Stage | Implemented | Files / Evidence | Notes |
|---|---:|---|---|
| Ingestion (CSV/JSON) | Implemented | `loaders/mimic_loader.py` ([loaders/mimic_loader.py](loaders/mimic_loader.py#L50-L60)), `loaders/readers/csv_reader.py` ([loaders/readers/csv_reader.py](loaders/readers/csv_reader.py#L1-L40)) | Discovery uses `RecursiveDiscovery` to find files ([loaders/discovery/recursive_discovery.py](loaders/discovery/recursive_discovery.py#L1-L40)).
| Validation (dataset-level) | Implemented (directory & format checks) | `loaders/validator/dataset_validator.py` ([loaders/validator/dataset_validator.py](loaders/validator/dataset_validator.py#L1-L40)) | Validates directory exists and that supported files exist. No content/schema validation beyond presence/format.
| Profiling (table & column) | Implemented | `profiling/profiler.py` ([profiling/profiler.py](profiling/profiler.py#L1-L40)), `profiling/column_profiler.py` ([profiling/column_profiler.py](profiling/column_profiler.py#L1-L50)), `profiling/dataset_profiler.py` ([profiling/dataset_profiler.py](profiling/dataset_profiler.py#L1-L40)) | Writes per-table JSON reports to `outputs/reports/profiling` ([profiling/report_writer.py](profiling/report_writer.py#L5-L18)).
| Duplicate detection | Implemented (exact duplicates only) | `detectors/duplicate_detector.py` ([detectors/duplicate_detector.py](detectors/duplicate_detector.py#L18-L60)) | Uses `df.duplicated(keep=False)` to flag exact duplicate rows; composite / fuzzy are not implemented (placeholders exist in comments).
| Missing-value detection | Implemented (row-by-row issue creation) | `detectors/missing_detector.py` ([detectors/missing_detector.py](detectors/missing_detector.py#L18-L80)) | For each missing cell an `Issue` object is created (table, row_index, column, severity="MEDIUM").
| Datatype validation | Implemented | `detectors/datatype_detector.py` ([detectors/datatype_detector.py](detectors/datatype_detector.py#L18-L100)) | Compares values against an `expected_dtype` taken from the profile; supports integer, float, boolean, datetime, string in `_is_valid()` ([detectors/datatype_detector.py](detectors/datatype_detector.py#L89-L114)).
| Outlier detection | Partially implemented (coordinator only) | `detectors/outlier_detector.py` ([detectors/outlier_detector.py](detectors/outlier_detector.py#L1-L120)) | Pipeline scaffolding exists and calls to methods for isolation forest / COPOD / autoencoder are placeholders returning empty DetectorResult; rule-based method returns an empty `DetectorResult` ([detectors/outlier_detector.py](detectors/outlier_detector.py#L1-L120)).
| Standardization / unit normalization | Not implemented | no preprocessing logic found in `preprocessing/*` (many files empty) | `preprocessing/*` files are present but empty (e.g., `preprocessing/missing.py`, `preprocessing/duplicates.py`, `preprocessing/outliers.py`).
| Referential integrity checks / joins | Not implemented | No code implementing multi-table referential checks (no explicit MIMIC joins). Key detection only provides candidate primary/foreign keys via heuristics ([profiling/key_detector.py](profiling/key_detector.py#L4-L40)).
| Reporting / Explainability | Implemented (basic) | `outputs/report_generator.py` ([outputs/report_generator.py](outputs/report_generator.py#L1-L80)), `explainability/llm_explainer.py` ([explainability/llm_explainer.py](explainability/llm_explainer.py#L1-L80)) | `ReportGenerator` produces a structured dict; `LLMExplainer` calls configured LLM backend to generate textual explanations (local HF model by default) — see `ml/llm/*`.

Notes: Many preprocessing modules are empty. The pipeline executes profiling by default and writes JSON reports; the quality detection pipeline exists but is not automatically invoked in the top-level `Pipeline` (profiling and quality are separate and need orchestration to run both).

---

**3) MIMIC-IV support: tables, fields, identifiers, relationships currently used**

- The repository contains a `MimicLoader` that discovers files under a provided dataset root and registers them as tables by converting path components into a single table name (e.g., `hosp/patients.csv` -> `hosp_patients`) — see `MimicLoader._table_name` ([loaders/mimic_loader.py](loaders/mimic_loader.py#L153-L160)).
- There is no explicit, hard-coded mapping of MIMIC‑IV tables or fields anywhere in the repository. Searches for canonical MIMIC identifiers (e.g., `subject_id`, `hadm_id`, `stay_id`, `icustay_id`, `admittime`, `dischtime`) returned no matches in the codebase (no occurrences found). Therefore:
  - Per-table lists of MIMIC tables are not present.
  - Per-field references to MIMIC identifiers are not present.
  - Any relationships between MIMIC tables (e.g., patient->admission->icustay) are not implemented explicitly.

Evidence and reasoning:
- Discovery pattern is generic; file paths are preserved in the `DatasetCatalog` and table names are generated from path components, not from a MIMIC schema mapping ([loaders/mimic_loader.py](loaders/mimic_loader.py#L153-L160), [loaders/catalog/dataset_catalog.py](loaders/catalog/dataset_catalog.py#L24-L37)).
- No code references to MIMIC column names or identifiers (search for `subject_id|hadm_id|stay_id|icustay_id` produced no matches).

Conclusion: MIMIC‑IV is supported at a file/CSV level (the loader can read files from a MIMIC directory structure), but there is no code that knows the semantics of MIMIC tables or fields. Any MIMIC-specific mapping, join logic, or field-level handling would need to be implemented.

---

**4) Data-quality rules: exact rules implemented, thresholds, inputs, outputs, severity, and action (flag/transform/delete)**

Below are the detectors and the exact logic they implement (citations to code). I quote implementation behavior precisely — no assumed behavior.

- MissingDetector (flags every missing cell as an Issue)
  - File: [detectors/missing_detector.py](detectors/missing_detector.py#L18-L26)
  - Rule: For each table and for each column, locate rows where `df[column].isna()` and create an `Issue` for each missing cell.
  - Issue fields set exactly as implemented:
    - `table` = table name
    - `row_index` = pandas index of the row (integer) — value comes from DataFrame index when iterating (`row_index = int(row_index)`) ([detectors/missing_detector.py](detectors/missing_detector.py#L57-L69)).
    - `column` = affected column name
    - `issue_type` = "missing"
    - `severity` = "MEDIUM"
    - `detector` = "MissingDetector"
    - `original_value` = None
    - `expected_value` = "Non-null value"
    - `confidence` = 1.0
  - Thresholds: none (every missing cell is flagged).
  - Output: `DetectorResult` containing one `Issue` per missing cell.
  - Action: flags issues only (does not transform or delete data).

- DuplicateDetector (exact duplicate rows only)
  - File: [detectors/duplicate_detector.py](detectors/duplicate_detector.py#L18-L60)
  - Rule: Identify rows where `df.duplicated(keep=False)` is True (exact duplicate rows across all columns). For each matching row index, create an `Issue` with:
    - `issue_type` = "duplicate"
    - `severity` = "HIGH"
    - `metadata` = {"method": "exact_match"}
  - Statistics returned: number of exact duplicates via `duplicate_rows.shape[0]` ([detectors/duplicate_detector.py](detectors/duplicate_detector.py#L44-L52)).
  - Thresholds: none beyond exact equality; composite/fuzzy duplicates are reported as 0 in statistics (not implemented).
  - Output: flags issues only (does not remove duplicates or transform data).

- DatatypeDetector (flags values not conforming to expected datatype)
  - File: [detectors/datatype_detector.py](detectors/datatype_detector.py#L18-L100)
  - Rule: For each table and each column where the profiling schema has an `inferred_type`, iterate rows and call `_is_valid(value, expected_dtype)`; if it returns False, create an `Issue`:
    - `issue_type` = "datatype"
    - `severity` = "HIGH"
    - `original_value` = value
    - `expected_value` = expected_dtype
    - `confidence` = 1.0
  - `_is_valid` supports exact validation for the following expected types (literal strings used in code): `integer`, `float`, `boolean`, `datetime`, `string` and uses the following checks:
    - `integer` -> `int(value)` (throws on failure)
    - `float` -> `float(value)`
    - `boolean` -> `str(value).lower()` in `{"true","false","0","1"}`
    - `datetime` -> `pd.to_datetime(value)`
    - `string` -> `str(value)`
    (See actual implementation: [detectors/datatype_detector.py](detectors/datatype_detector.py#L89-L114)).
  - Thresholds: none numeric; every non-conforming value is flagged.
  - Output: flags issues only (no transformations).

- OutlierDetector
  - File: [detectors/outlier_detector.py](detectors/outlier_detector.py#L1-L120)
  - Rule: acts as a coordinator; the following methods exist but are currently placeholders returning empty `DetectorResult` objects:
    - `_rule_based` -> returns empty `DetectorResult("RuleBased")`
    - `_isolation_forest` -> returns empty `DetectorResult("IsolationForest")`
    - `_copod` -> returns empty `DetectorResult("COPOD")`
    - `_autoencoder` -> returns empty `DetectorResult("Autoencoder")`
  - No concrete outlier detection thresholds or logic implemented; the orchestration pipeline collates counts from these subdetectors but they are no-ops ([detectors/outlier_detector.py](detectors/outlier_detector.py#L1-L120)).

- Additional detectors: `inconsistency_detector.py`, `imbalance_detector.py`, `cardinality_detector.py`, `drift_detector.py` exist but are empty files — not implemented.

All detectors flag issues only; nowhere in the detector code do I find any automatic transform or delete operation (detectors create `Issue` objects and `DetectorResult` structures; no code that mutates datasets). The `Issue` and `DetectorResult` classes define the fields used in outputs: see `models/issue.py` and `models/detector_result.py` ([models/issue.py](models/issue.py#L12-L20), [models/detector_result.py](models/detector_result.py#L20-L40)).

---

**5) Traceability: can every issue be traced to source table, source row, subject_id, hadm_id, stay_id, timestamp?**

- What is present:
  - Each `Issue` contains `table` and `row_index` and `column` and `metadata` ([models/issue.py](models/issue.py#L57-L71)).
  - `DatasetCatalog` retains the original file `Path` for each registered table via `register_table()` and `get_table_path()` ([loaders/catalog/dataset_catalog.py](loaders/catalog/dataset_catalog.py#L32-L37)).
  - `KeyDetector` can propose candidate primary / foreign keys heuristically (unique columns and columns containing keywords like "patient", "encounter", etc.) ([profiling/key_detector.py](profiling/key_detector.py#L4-L40)).

- Limitations and issues (traceability gaps):
  - `row_index` is the pandas index value present in the in-memory DataFrame at detection time. There is no universal mapping from that index back to an original file line number. When streaming chunked files, chunk indices are relative to a chunk's DataFrame and may start from 0 for each chunk — detectors that iterate a dataset dictionary expect DataFrames and use DataFrame indices directly. This means that `row_index` is not guaranteed to be the original CSV line number; additional provenance (e.g., a column for original row number or explicit offset logic) is not present.
  - None of the detectors inject or record `subject_id`, `hadm_id`, `stay_id`, or other canonical MIMIC identifiers into `Issue` objects unless those identifiers are present as columns and included in `metadata` by detector code (which the current detectors do not add). Searches for those field names in the repository return no code references.
  - There is no implemented mechanism to attach timestamps (e.g., `charttime`, `admittime`) to individual issues as provenance fields automatically. Therefore traceability to `subject_id`/`hadm_id`/`stay_id`/timestamp is only possible if those columns exist in the DataFrame and are included in the dataset passed to detectors and the user performs join/lookup using `row_index` manually.

Conclusion: table and row_index are available for tracing, and file paths are stored in the catalog; however robust clinical traceability to `subject_id`/`hadm_id`/`stay_id`/timestamp is not implemented automatically and cannot be relied upon without adding explicit provenance capture (e.g., preserving original file offsets or adding a canonical patient identifier column to every row before detection).

---

**6) Safety risks**

- Unsafe automatic transformations: none of the detectors perform automatic transformations or deletions — detectors only flag `Issue` objects. This is safer than silent automatic mutation. (See `detectors/*` code: all create issues and return `DetectorResult`.)
- Possible data leakage via explainability / LLM:
  - `explainability/llm_explainer.py` will build prompts and call `self.llm.generate()` for each issue ([explainability/llm_explainer.py](explainability/llm_explainer.py#L1-L40)). The default factory creates a `HuggingFaceLLM` which calls `AutoTokenizer.from_pretrained()` and `AutoModelForCausalLM.from_pretrained()` ([ml/llm/huggingface_llm.py](ml/llm/huggingface_llm.py#L1-L30)). If the configured model name points to a remote model repository, model weights and tokenizer may be downloaded; if an external LLM backend were added (not present), prompts could be sent off-host. The code does not redact PHI in prompts — risk of leaking PHI exists if LLM explanations are run on sensitive data.
- Broken joins / referential integrity risks:
  - No referential integrity checks or join validations are implemented; if users assume the platform enforces referential integrity across MIMIC tables, that would be incorrect.
- Temporal errors:
  - There is no temporal-consistency rule implemented (e.g., admission before discharge) — RuleValidator in `config.py` lists TODOs including date consistency but is unimplemented ([config.py](config.py#L1-L40)).
- Duplicate-counting risks:
  - Duplicate detection uses `df.duplicated(keep=False)` which marks both rows in a duplicated pair. The `duplicate_rows.shape[0]` statistic counts duplicated rows (both duplicate copies), and `result.issues` includes an `Issue` for each duplicated row index. This is logically consistent with the code but may double-count duplicates relative to some expectations (some systems count duplicate groups rather than duplicated rows). See [detectors/duplicate_detector.py](detectors/duplicate_detector.py#L44-L52).
- Incorrect clinical assumptions:
  - There is no clinical-rule engine implemented; detectors use generic heuristics (datatype checks, exact duplicates, missing values). The `RuleValidator` comments enumerate clinical checks that are TODO but not implemented — do not rely on clinical validation being present ([config.py](config.py#L1-L40)).

---

**7) Security and compliance: storage & transmission of raw data, extracts, logs, caches, exports**

- Locations where data is read from and/or persisted:
  - Raw dataset location expected by example: `data/raw/mimiciv` as shown in `app.py` ([app.py](app.py#L1-L10)).
  - `DatasetCatalog` stores table file `Path` values internally and caches DataFrames in memory (`DatasetCatalog._tables` and `_cache`) but does not write dataset copies to disk (`loaders/catalog/dataset_catalog.py`).
  - Profiling output JSON files are written to `outputs/reports/profiling` by `ReportWriter.save()` ([profiling/report_writer.py](profiling/report_writer.py#L5-L18)). These profiling files contain column-level and row-level summary metadata and may include examples/samples depending on the profiling implementation.
  - No database exports or external file export APIs were found.

- Network transmissions and LLM model download risk:
  - `ml/llm/huggingface_llm.py` calls `from_pretrained()` which will download model artifacts from a remote model hub if the model is not available locally ([ml/llm/huggingface_llm.py](ml/llm/huggingface_llm.py#L1-L30)). There is no explicit telemetry or external HTTP ingestion of dataset contents in other modules.

- Logging and caches:
  - There is a `utils/logger.py` file but it is empty (no logging configuration found). Profiling and report writing print messages to stdout (e.g., `print(f"Saved: {file}")`).
  - In-memory DataFrame cache exists in `DatasetCatalog` but there is no on-disk pickling or database storage of cached DataFrames ([loaders/catalog/dataset_catalog.py](loaders/catalog/dataset_catalog.py#L24-L37)).

- Recommendation (compliance): until explicit safeguards (redaction, encryption, access controls) are added, do not run explainability/LLM prompts on unredacted PHI and avoid leaving profiling outputs on shared file systems. Add access controls and encryption for `outputs/` and consider mandatory PHI redaction before sending data to any external model.

---

**8) Missing components: implemented / partially implemented / mocked / not implemented**

- Implemented (usable as-is):
  - CSV/JSON ingestion and discovery (`loaders/*`, `loaders/readers/csv_reader.py`, `loaders/discovery/recursive_discovery.py`).
  - Dataset-level validation for directory/format presence (`loaders/validator/dataset_validator.py`).
  - Profiling engine and column/dataset statistics (`profiling/*`), and writing profiling JSON outputs (`profiling/report_writer.py`).
  - Basic detectors: MissingDetector, DuplicateDetector (exact), DatatypeDetector (value-level checks). These detectors construct `Issue` objects but do not modify data (`detectors/missing_detector.py`, `detectors/duplicate_detector.py`, `detectors/datatype_detector.py`).
  - `quality/detector.py` orchestration of detector classes into a `QualityResult` aggregation.

- Partially implemented / scaffolded:
  - Outlier detection coordinator exists but sub-detectors are placeholders (`detectors/outlier_detector.py`).
  - `LLM` explainability stack is present (`explainability/llm_explainer.py`, `ml/llm/*`) but prompt building/parsing modules appear partial or missing (`explainability/prompt.py` is empty). Model download behavior is implemented but not necessarily intended for production use without governance.
  - `RuleValidator` is declared in `config.py` but contains TODOs only (date consistency, clinical ranges, etc.) and is not used by detectors ([config.py](config.py#L1-L40)).
  - `preprocessing/*` modules exist but are empty (not implemented): `preprocessing/missing.py`, `preprocessing/duplicates.py`, `preprocessing/outliers.py`, `preprocessing/encoding.py`, `preprocessing/executor.py`.

- Mocked / demo-only: none explicitly labeled as mock/demo; however empty modules and stubbed detector methods should be considered demo scaffolding.

- Not implemented:
  - MIMIC semantic mappings (table->schema), referential integrity checks and join logic, clinical rule sets, automatic remediation actions (transform/delete), API endpoints, persistent database storage, and test suites (most `tests/*.py` files are empty).

---

**9) Tests and reproducibility**

- Existing tests: `tests/test_detectors.py`, `tests/test_ml.py`, `tests/test_pipeline.py`, `tests/test_profiler.py` exist as files but are empty (no runnable tests found).

- Dependencies: see `requirements.txt` (not exhaustive of system-level requirements): pandas, numpy, scikit-learn, pyod, tensorflow, transformers, torch, and other packages ([requirements.txt](requirements.txt#L1-L40)).

- Environment variables: none required by code were found. `config.py` contains LLM configuration constants (e.g., `LLM_MODEL`) rather than environment-driven configuration ([config.py](config.py#L1-L20)).

- Setup commands to reproduce the pipeline locally (recommended minimal):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Notes:
- `python app.py` is the repository entrypoint used in the example and will initialize the pipeline for `dataset_path="data/raw/mimiciv"` — ensure `data/raw/mimiciv` exists and contains CSV files ([app.py](app.py#L1-L10)).
- Running `app.py` will trigger profiling and write per-table profile JSON files to `outputs/reports/profiling` ([profiling/report_writer.py](profiling/report_writer.py#L5-L18)).

---

**10) Prioritized next actions (top 10), ranked by impact × effort with concise acceptance criteria**

1. Add explicit MIMIC‑IV schema mappings and canonical field names (Impact: High, Effort: Medium)
   - Files to change: add `loaders/mimic_schema.py` or extend `MimicLoader` to map files to canonical table names and column expectations.
   - Acceptance: for a provided MIMIC dataset directory, loader returns a schema mapping listing canonical fields (e.g., `patients: ['subject_id','gender',...]`) and unit tests validate mapping for at least 3 MIMIC files.

2. Implement provenance mapping from DataFrame rows back to original source offsets and include clinical identifiers in `Issue` objects (Impact: High, Effort: Medium)
   - Add optional `__source_file__` and `__source_row__` columns on ingestion or preserve CSV row offsets when streaming.
   - Acceptance: every `Issue` saved includes `source_file`, `source_row`, and (if present) `subject_id`/`hadm_id` fields; automated test shows traceability for chunked reads.

3. Implement referential integrity and clinical rule validations (RuleValidator) (Impact: High, Effort: High)
   - Implement the TODO list in `config.py`/`RuleValidator` and create deterministic tests for date consistency and referential integrity across `patients->admissions->icustays` (or file names used).
   - Acceptance: RuleValidator returns `DetectorResult` with issues for broken references; tests exercise at least 4 clinical rules.

4. Harden LLM explainability for PHI safety (Impact: High, Effort: Medium)
   - Add automatic PHI redaction before building prompts, and a toggle to forbid external model downloads; default to local-only models.
   - Acceptance: unit tests verify redaction removes patient identifiers from prompts; explainability component cannot call remote APIs when disabled.

5. Complete outlier detection implementations (Impact: Medium, Effort: Medium)
   - Implement isolation forest, COPOD, and autoencoder detectors and integrate thresholds and configurable parameters.
   - Acceptance: outlier detectors return non-empty `DetectorResult` on synthetic tests; documented configurations exist in code or config file.

6. Fill preprocessing modules and add safe remediation actions (Impact: Medium, Effort: Medium)
   - Implement `preprocessing/missing.py` (imputation strategies), `preprocessing/duplicates.py` (de-duplication policies), and `preprocessing/encoding.py` (categorical encoding) with explicit, auditable transforms.
   - Acceptance: preprocessing produces a new dataset copy (never overwrites source) and unit tests validate transform correctness.

7. Add persistent storage option and RBAC for outputs (Impact: Medium, Effort: Medium)
   - Add configurable storage backends for reports (local, S3) and file-level access controls.
   - Acceptance: configurable `outputs` destination and doc describing access controls; integration tests for local filesystem and S3 mocks.

8. Implement comprehensive unit and integration tests (Impact: High, Effort: High)
   - Populate `tests/` with tests for loaders, profiling, detectors, and the QualityDetector orchestration.
   - Acceptance: CI runs tests with >80% coverage for changed modules.

9. Add explicit logging configuration and remove plain `print()` for production (Impact: Low, Effort: Low)
   - Implement `utils/logger.py` to centralize logging and ensure sensitive data is not logged at INFO level.
   - Acceptance: repository uses logger module for all informational messages; no raw data printed by default.

10. Document secure deployment and data handling guidelines (Impact: High, Effort: Low)
   - Add README sections describing PHI handling, redaction requirements, and recommended practices for running the pipeline on clinical data.
   - Acceptance: new `SECURITY.md` or `README.md` section with checklist; reviewer confirms content.

---

Appendix: key code locations (quick reference)

- Loader & discovery: [loaders/mimic_loader.py](loaders/mimic_loader.py#L12-L20), [loaders/discovery/recursive_discovery.py](loaders/discovery/recursive_discovery.py#L1-L40)
- Catalog / caching: [loaders/catalog/dataset_catalog.py](loaders/catalog/dataset_catalog.py#L24-L37)
- Csv reader: [loaders/readers/csv_reader.py](loaders/readers/csv_reader.py#L1-L40)
- Profiling composition: [profiling/profiler.py](profiling/profiler.py#L1-L40)
- Report writer (profiling output): [profiling/report_writer.py](profiling/report_writer.py#L5-L18)
- Key detection heuristics: [profiling/key_detector.py](profiling/key_detector.py#L4-L40)
- Detectors (Missing, Duplicate, Datatype, Outlier): [detectors/missing_detector.py](detectors/missing_detector.py#L18-L26), [detectors/duplicate_detector.py](detectors/duplicate_detector.py#L18-L60), [detectors/datatype_detector.py](detectors/datatype_detector.py#L18-L100), [detectors/outlier_detector.py](detectors/outlier_detector.py#L1-L120)
- Issue and result models: [models/issue.py](models/issue.py#L12-L20), [models/detector_result.py](models/detector_result.py#L20-L40)
- Quality orchestration: [quality/detector.py](quality/detector.py#L1-L80)

If you want, I can now:
- (A) Run the repository's pipeline on a small synthetic MIMIC subset (if you provide data) and produce sample profiling/quality outputs, or
- (B) Implement one of the prioritized next actions (pick one) and provide a patch and tests.
