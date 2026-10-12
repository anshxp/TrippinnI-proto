"""Runtime configuration for the TrippinnI prototype."""

from __future__ import annotations

import os
from pathlib import Path

import psutil

# Hardware profile: choose conservative defaults from installed system RAM.
# Explicit environment variables always override these defaults. RAM is only a
# heuristic; GPU VRAM, quantization, and other running applications also matter.
SYSTEM_RAM_GB = psutil.virtual_memory().total / (1024 ** 3)
IS_HIGH_MEMORY_SYSTEM = SYSTEM_RAM_GB >= 24
DEFAULT_LLM_MODEL = "qwen3:8b" if IS_HIGH_MEMORY_SYSTEM else "qwen3:4b"
DEFAULT_OLLAMA_NUM_CTX = 4096 if IS_HIGH_MEMORY_SYSTEM else 2048
DEFAULT_CSV_CHUNK_SIZE = 10000 if IS_HIGH_MEMORY_SYSTEM else 2000
DEFAULT_MAX_ROWS_FOR_DETECTION = 10000 if IS_HIGH_MEMORY_SYSTEM else 5000

# Local LLM configuration (Ollama).
# 32 GB machine -> Qwen3 8B; 8 GB machine -> Qwen3 4B.
# Override with TRIPPINNI_LLM_MODEL if RAM is shared or GPU limits require it.
LLM_MODEL = os.getenv("TRIPPINNI_LLM_MODEL", DEFAULT_LLM_MODEL)
OLLAMA_BASE_URL = os.getenv("TRIPPINNI_OLLAMA_URL", "http://localhost:11434")
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1

# Chunked processing configuration; lower default on the 8 GB machine.
CSV_CHUNK_SIZE = int(os.getenv("TRIPPINNI_CSV_CHUNK_SIZE", str(DEFAULT_CSV_CHUNK_SIZE)))
# Keep in-memory detection samples only when explicitly requested; reports are persisted to disk.
RETAIN_DETECTION_SAMPLES = os.getenv("TRIPPINNI_RETAIN_DETECTION_SAMPLES", "0").strip().lower() in {"1", "true", "yes"}

# LLM-guided preprocessing is opt-in and row-bounded. The orchestrator uses
# PROFILE_PREFIX_FRACTION and calls the local model for each selected chunk.
PREPROCESSING_ENABLED = os.getenv("TRIPPINNI_PREPROCESSING_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
PREPROCESSING_CHUNK_SIZE = int(os.getenv("TRIPPINNI_PREPROCESSING_CHUNK_SIZE", "1000"))
# Independent hard cap: preprocessing must not inherit the much larger profiling prefix.
PREPROCESSING_MAX_ROWS = int(os.getenv("TRIPPINNI_PREPROCESSING_MAX_ROWS", "5000"))
PREPROCESSING_MODEL = os.getenv("TRIPPINNI_PREPROCESSING_MODEL", LLM_MODEL)
PREPROCESSING_MAX_ITERATIONS = int(os.getenv("TRIPPINNI_PREPROCESSING_MAX_ITERATIONS", "3"))
PREPROCESSING_ALLOW_IMPUTATION = os.getenv("TRIPPINNI_PREPROCESSING_ALLOW_IMPUTATION", "0").strip().lower() in {"1", "true", "yes"}
# Empty allowlist means the orchestrator will not preprocess any table.
PREPROCESSING_TABLES = {name.strip() for name in os.getenv("TRIPPINNI_PREPROCESSING_TABLES", "").split(",") if name.strip()}
PREPROCESSING_INCLUDE_CONNECTED_TABLES = os.getenv("TRIPPINNI_PREPROCESSING_INCLUDE_CONNECTED_TABLES", "1").strip().lower() in {"1", "true", "yes"}
PREPROCESSING_ROW_COUNT_CHUNK_SIZE = int(os.getenv("TRIPPINNI_PREPROCESSING_ROW_COUNT_CHUNK_SIZE", "100000"))
# Conservative context sizes: 2048 for 8 GB, 4096 for 32 GB.
OLLAMA_NUM_CTX = int(os.getenv("TRIPPINNI_OLLAMA_NUM_CTX", str(DEFAULT_OLLAMA_NUM_CTX)))

# Prototype profiling configuration
# Process only the deterministic first fraction of each source table.
# This is a prefix, not a random sample: once the prefix boundary is
# reached, the reader stops and the remaining source data is never parsed.
PROFILE_PREFIX_FRACTION = 0.10

# Small unknown-row-count CSVs can be counted once before applying the prefix.
# This keeps tiny metadata tables such as provider.csv.gz exact without
# introducing a full scan for large tables.
CSV_ROW_COUNT_FALLBACK_MAX_MB = 10

# Quality detectors still receive a bounded in-memory sample.
MAX_ROWS_FOR_DETECTION = int(os.getenv("TRIPPINNI_MAX_ROWS_FOR_DETECTION", str(DEFAULT_MAX_ROWS_FOR_DETECTION)))
DETECTION_SAMPLE_SEED = 42

# Approximate duplicate detection (RapidFuzz)
# Fuzzy matching is intentionally limited to text-like, non-identifier columns.
FUZZY_DUPLICATE_ENABLED = True
FUZZY_DUPLICATE_THRESHOLD = 92.0
# Ignore very short strings because edit similarity is unstable for short text.
FUZZY_DUPLICATE_MIN_LENGTH = 4
FUZZY_DUPLICATE_MAX_VALUES_PER_COLUMN = 5_000
FUZZY_DUPLICATE_MAX_MATCHES_PER_VALUE = 3

# Cross-table ML boundary matching
CROSS_TABLE_ML_ENABLED = True
CROSS_TABLE_ML_MAX_CHILD_VALUES = 1_000
CROSS_TABLE_ML_MAX_PARENT_VALUES = 10_000
CROSS_TABLE_ML_MIN_PROBABILITY = 0.995
CROSS_TABLE_ML_MIN_MARGIN = 0.05
CROSS_TABLE_ML_MIN_FUZZY_SIMILARITY = 80.0

# Advanced EHR quality thresholds
REPRESENTATION_THRESHOLD = 0.05
FAIRNESS_MISSINGNESS_GAP = 0.10
DISTRIBUTION_DRIFT_THRESHOLD = 0.20
DISTRIBUTION_SHIFT_THRESHOLD = 0.20
ROBUSTNESS_BOOTSTRAP_RANGE = 0.20
MIN_GROUP_SIZE_FOR_BIAS = 30


# Official MIMIC-IV v3.1 row counts used to turn the 10% prefix into an
# exact row boundary without scanning the source file first.
MIMIC_V31_ROW_COUNTS = {
    "hosp_admissions": 546_028,
    "hosp_d_hcpcs": 89_208,
    "hosp_d_icd_diagnoses": 112_107,
    "hosp_d_icd_procedures": 86_423,
    "hosp_d_labitems": 1_650,
    "hosp_diagnoses_icd": 6_364_488,
    "hosp_drgcodes": 761_856,
    "hosp_emar": 42_808_593,
    "hosp_emar_detail": 87_371_064,
    "hosp_hcpcsevents": 186_074,
    "hosp_labevents": 158_374_764,
    "hosp_microbiologyevents": 3_988_224,
    "hosp_omr": 7_753_027,
    "hosp_patients": 364_627,
    "hosp_pharmacy": 17_847_567,
    "hosp_poe": 52_212_109,
    "hosp_poe_detail": 8_504_982,
    "hosp_prescriptions": 20_292_611,
    "hosp_procedures_icd": 859_655,
    "hosp_services": 593_071,
    "hosp_transfers": 2_413_581,
    "icu_icustays": 94_458,
    "icu_ingredientevents": 12_229_408,
    "icu_d_items": 4_095,
    "icu_caregiver": 454_324,
    "icu_chartevents": 432_997_491,
    "icu_datetimeevents": 9_979_761,
    "icu_inputevents": 10_953_713,
    "icu_outputevents": 5_359_395,
    "icu_procedureevents": 808_706,
}

# Local NVMe workspace (F:) for the prototype.
# Override with environment variables if the dataset/workspace lives elsewhere.
DATA_ROOT = Path(
    os.getenv(
        "TRIPPINNI_DATA_ROOT",
        r"F:\Ansh's\physionet.org\files\mimiciv\3.1",
    )
)
WORK_ROOT = Path(os.getenv("TRIPPINNI_WORK_ROOT", "F:/TrippinnI-work"))
OUTPUT_ROOT = WORK_ROOT / "outputs"
