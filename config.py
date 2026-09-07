"""Runtime configuration for the TrippinnI prototype."""

from __future__ import annotations

# LLM configuration
LLM_MODEL = "Qwen/Qwen3-4B-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2

# Chunked processing configuration
CSV_CHUNK_SIZE = 100_000
MAX_ROWS_FOR_DETECTION = 10_000
DETECTION_SAMPLE_SEED = 42
