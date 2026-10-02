"""Runtime configuration for the TrippinnI prototype."""

from __future__ import annotations

import os
from pathlib import Path

# LLM configuration
LLM_MODEL = "Qwen/Qwen3-4B-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2

# Chunked processing configuration
CSV_CHUNK_SIZE = 100_000

# Prototype profiling configuration
# Process only the deterministic first fraction of each source table.
# This is a prefix, not a random sample: once the prefix boundary is
# reached, the reader stops and the remaining source data is never parsed.
PROFILE_PREFIX_FRACTION = 0.10

# Quality detectors still receive a bounded in-memory sample.
MAX_ROWS_FOR_DETECTION = 10_000
DETECTION_SAMPLE_SEED = 42


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
    "icu_d_items": 4_095,
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
