"""Chunked CSV entry point for the TrippinnI preprocessing planner.

The source CSV is never modified. Each chunk is independently profiled and
processed by the bounded runtime; processed chunks are streamed to one output.
This is memory-bounded but chunk-local statistics (for example IQR) can differ
between chunks, so this is a prototype workflow rather than a clinical pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.runtime import run_dataframe


def run_csv_chunks(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    chunksize: int = 50_000,
    model: str = "qwen3:4b",
    max_iterations: int = 5,
    allow_imputation: bool = False,
) -> dict[str, Any]:
    """Process a CSV in bounded chunks and stream results to a single CSV."""
    source = Path(input_path)
    destination = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {source}")
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")
    destination.mkdir(parents=True, exist_ok=True)
    chunk_root = destination / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)

    output_path = destination / f"{source.stem}_processed.csv"
    temp_path = destination / f".{source.stem}_processed.tmp"
    index_path = destination / f"{source.stem}_chunk_report.json"
    if temp_path.exists():
        temp_path.unlink()

    chunk_reports: list[dict[str, Any]] = []
    total_rows = 0
    try:
        for chunk_number, chunk in enumerate(
            pd.read_csv(source, chunksize=chunksize, low_memory=False), start=1
        ):
            chunk_name = f"{source.stem}_chunk_{chunk_number:06d}"
            report = run_dataframe(
                chunk,
                file_name=chunk_name,
                output_dir=chunk_root / chunk_name,
                model=model,
                max_iterations=max_iterations,
                allow_imputation=allow_imputation,
            )
            processed_chunk = Path(report["output_file"])
            if not processed_chunk.is_file():
                raise RuntimeError(f"Chunk {chunk_number} did not produce an output CSV")
            output_chunk = pd.read_csv(processed_chunk, low_memory=False)
            output_chunk.to_csv(
                temp_path,
                mode="a" if chunk_number > 1 else "w",
                header=chunk_number == 1,
                index=False,
            )
            total_rows += len(chunk)
            chunk_reports.append({
                "chunk": chunk_number,
                "rows": len(chunk),
                "run_id": report["run_id"],
                "actions": report["actions"],
                "chunk_report": report["report_file"],
            })

        if not chunk_reports:
            # Preserve the schema for a header-only CSV.
            empty = pd.read_csv(source, nrows=0)
            empty.to_csv(temp_path, index=False)
        temp_path.replace(output_path)
    except Exception:
        # Do not leave a partial output masquerading as a completed dataset.
        if temp_path.exists():
            temp_path.unlink()
        raise

    summary = {
        "input_file": str(source.resolve()),
        "output_file": str(output_path.resolve()),
        "chunksize": chunksize,
        "chunks_processed": len(chunk_reports),
        "rows_processed": total_rows,
        "model": model,
        "allow_imputation": allow_imputation,
        "chunk_reports": chunk_reports,
        "note": (
            "Actions and statistics are chunk-local. Review cross-chunk consistency "
            "before using output for downstream analysis."
        ),
    }
    index_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_report"] = str(index_path.resolve())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked TrippinnI preprocessing runner")
    parser.add_argument("input_csv", help="Path to source CSV (source remains unchanged)")
    parser.add_argument("--output-dir", default="outputs/preprocessing")
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument(
        "--allow-imputation",
        action="store_true",
        help="Enable numeric median imputation for this run; use only with explicit review",
    )
    args = parser.parse_args()
    result = run_csv_chunks(
        args.input_csv,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        model=args.model,
        max_iterations=args.max_iterations,
        allow_imputation=args.allow_imputation,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
