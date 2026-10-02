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
# Stream only a fraction of each source table into the profiling pipeline.
PROFILE_SAMPLE_FRACTION = 0.10
PROFILE_SAMPLE_SEED = 42

# Quality detectors still receive a bounded in-memory sample.
MAX_ROWS_FOR_DETECTION = 10_000
DETECTION_SAMPLE_SEED = 42

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
