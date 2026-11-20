"""Chunked CSV processing using deterministic preprocessing rules; no model dependency."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.runtime import run_dataframe


def count_csv_rows(input_path: str | Path, *, chunksize: int = 100_000) -> int:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    return sum(len(chunk) for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False))


def run_csv_chunks(
    input_path: str | Path, *, output_dir: str | Path,
    chunksize: int = 50_000, max_rows: int | None = None,
    allow_imputation: bool = False, context_metadata: dict[str, Any] | None = None,
    profile_mode: str = "fast",
) -> dict[str, Any]:
    source, destination = Path(input_path), Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if chunksize <= 0 or (max_rows is not None and max_rows <= 0):
        raise ValueError("chunksize and max_rows must be positive when supplied")
    destination.mkdir(parents=True, exist_ok=True)
    chunk_root = destination / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    stem = source.name[:-7] if source.name.lower().endswith(".csv.gz") else source.stem
    output_path = destination / f"{stem}_processed.csv"
    temp_path = destination / f".{stem}_processed.tmp"
    temp_path.unlink(missing_ok=True)

    reports, paths, columns, rows = [], [], [], 0
    try:
        for number, chunk in enumerate(pd.read_csv(source, chunksize=chunksize, nrows=max_rows, low_memory=False), start=1):
            name = f"{stem}_chunk_{number:06d}"
            report = run_dataframe(
                chunk, file_name=name, output_dir=chunk_root / name,
                allow_imputation=allow_imputation, context_metadata=context_metadata,
                profile_mode=profile_mode,
            )
            processed = Path(report["output_file"])
            header = list(pd.read_csv(processed, nrows=0).columns)
            columns.extend(c for c in header if c not in columns)
            paths.append(processed)
            rows += len(chunk)
            reports.append({
                "chunk": number, "rows": len(chunk), "run_id": report["run_id"],
                "action_count": len(report.get("actions", [])),
                "chunk_report": report["report_file"],
            })
            del chunk, report
            gc.collect()

        for i, path in enumerate(paths):
            data = pd.read_csv(path, low_memory=False).reindex(columns=columns)
            data.to_csv(temp_path, mode="a" if i else "w", header=(i == 0), index=False)
            del data
            gc.collect()
        if not paths:
            pd.read_csv(source, nrows=0).to_csv(temp_path, index=False)
        temp_path.replace(output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    summary = {
        "input_file": str(source.resolve()), "output_file": str(output_path.resolve()),
        "chunksize": chunksize, "max_rows": max_rows, "profile_mode": profile_mode,
        "scope": "all_rows" if max_rows is None else "bounded_prefix",
        "chunks_processed": len(reports), "rows_processed": rows,
        "output_columns": columns, "decision_engine": "deterministic_rules_v1",
        "allow_imputation": allow_imputation, "chunk_reports": reports,
        "note": "Chunk-local thresholds and profiles may differ. Review candidate records before classifier training.",
    }
    summary_path = destination / f"{stem}_chunk_report.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_report"] = str(summary_path.resolve())
    return summary
