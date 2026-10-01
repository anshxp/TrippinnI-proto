"""Runtime configuration for the TrippinnI prototype."""

from __future__ import annotations

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
