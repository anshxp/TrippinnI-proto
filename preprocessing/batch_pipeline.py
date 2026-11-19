"""Sequential 10% extraction, preprocessing, and audit pipeline.

Each invocation processes one non-overlapping batch from a CSV/CSV.GZ source.
The extracted batch is persisted before preprocessing; the source is immutable.
Only aggregate metadata is sent to the local planner (never patient rows).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

import config
from preprocessing.csv_runtime import count_csv_rows, run_csv_chunks


def _source_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_row_window(
    source: Path, destination: Path, *, start_row: int, end_row: int,
    chunksize: int = 50_000,
) -> int:
    """Persist [start_row, end_row) to a standalone CSV using bounded memory."""
    if start_row < 0 or end_row <= start_row:
        raise ValueError("Invalid row window")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    written = 0
    source_offset = 0
    try:
        for frame in pd.read_csv(source, chunksize=chunksize, low_memory=False):
            chunk_end = source_offset + len(frame)
            left = max(start_row, source_offset)
            right = min(end_row, chunk_end)
            if left < right:
                selected = frame.iloc[left - source_offset:right - source_offset]
                selected.to_csv(temp, mode="a" if written else "w",
                                header=(written == 0), index=False)
                written += len(selected)
            source_offset = chunk_end
            if source_offset >= end_row:
                break
        if written == 0:
            raise ValueError(f"No rows extracted for [{start_row}, {end_row})")
        temp.replace(destination)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
    return written


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str) + "\n")


def run_batch(
    source_path: str | Path, *, output_root: str | Path,
    batch_index: int, batch_fraction: float = 0.10,
    extraction_chunksize: int = 50_000,
    preprocessing_chunksize: int = 1_000,
    model: str | None = None, max_iterations: int = 2,
    allow_imputation: bool = False,
) -> dict[str, Any]:
    source = Path(source_path)
    root = Path(output_root)
    if not source.is_file():
        raise FileNotFoundError(source)
    if "".join(source.suffixes[-2:]).lower() not in {".csv", ".csv.gz"}:
        raise ValueError("Only .csv and .csv.gz sources are supported")
    if batch_index < 0:
        raise ValueError("batch_index is zero-based and must be >= 0")
    if not 0 < batch_fraction <= 1:
        raise ValueError("batch_fraction must be in (0, 1]")
    if preprocessing_chunksize <= 0 or extraction_chunksize <= 0:
        raise ValueError("Chunk sizes must be positive")

    total_rows = count_csv_rows(source, chunksize=extraction_chunksize)
    if total_rows == 0:
        raise ValueError("Source contains no data rows")
    batch_rows = max(1, int(total_rows * batch_fraction))
    start_row = batch_index * batch_rows
    end_row = min(start_row + batch_rows, total_rows)
    if start_row >= total_rows:
        raise ValueError(
            f"Batch {batch_index} starts at row {start_row}, beyond {total_rows} rows"
        )

    fingerprint = _source_fingerprint(source)
    stem = source.name[:-7] if source.name.lower().endswith(".csv.gz") else source.stem
    batch_name = f"{stem}_batch_{batch_index + 1:03d}"
    batch_root = root / batch_name
    extracted = batch_root / f"{batch_name}_raw.csv"
    processed_root = batch_root / "preprocessing"
    manifest_path = batch_root / "batch_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"{batch_name} already exists. Use a new output root for a clean rerun."
        )
