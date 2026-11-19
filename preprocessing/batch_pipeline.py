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
    allow_imputation: bool = False, force: bool = False,
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
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source_fingerprint") == fingerprint:
            raise FileExistsError(
                f"{batch_name} already exists. Use --force to regenerate it."
            )
        raise RuntimeError("Existing batch belongs to a different source version; use a new output root.")

    extracted_count = extract_row_window(
        source, extracted, start_row=start_row, end_row=end_row,
        chunksize=extraction_chunksize,
    )
    # The persisted extracted file is the exact input to preprocessing.
    summary = run_csv_chunks(
        extracted,
        output_dir=processed_root,
        chunksize=preprocessing_chunksize,
        model=model or config.PREPROCESSING_MODEL,
        max_iterations=max_iterations,
        allow_imputation=allow_imputation,
        max_rows=extracted_count,
        context_metadata={
            "dataset": stem,
            "batch_index": batch_index,
            "source_row_start_inclusive": start_row,
            "source_row_end_exclusive": end_row,
            "batch_fraction": batch_fraction,
        },
    )

    manifest = {
        "dataset": stem,
        "source_file": str(source.resolve()),
        "source_fingerprint": fingerprint,
        "batch_index": batch_index,
        "batch_number": batch_index + 1,
        "batch_fraction": batch_fraction,
        "source_total_rows": total_rows,
        "source_row_start_inclusive": start_row,
        "source_row_end_exclusive": end_row,
        "rows_extracted": extracted_count,
        "extracted_file": str(extracted.resolve()),
        "processed_file": summary["output_file"],
        "preprocessing_report": summary["summary_report"],
        "model": model or config.PREPROCESSING_MODEL,
        "allow_imputation": allow_imputation,
        "status": "completed",
    }
    batch_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Build classifier examples only from recorded planner decisions and
    # measurable before/after profiles. Never store raw patient rows here.
    training_path = root / "classifier_training" / "algorithm_selection_records.jsonl"
    training_count = 0
    for chunk_ref in summary.get("chunk_reports", []):
        report_path = Path(chunk_ref["chunk_report"])
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for action in report.get("actions", []):
            if action.get("status") not in {"committed", "planner_failed",
                                            "unregistered_action_blocked"}:
                continue
            record = {
                "record_version": "1.0",
                "dataset": stem,
                "source_fingerprint": fingerprint,
                "batch_index": batch_index,
                "batch_row_start": start_row,
                "batch_row_end": end_row,
                "chunk_index": chunk_ref["chunk"],
                "metadata_features": action.get("before_profile", {}),
                "selected_algorithm": action.get("action"),
                "selection_status": action.get("status"),
                "algorithm_module": action.get("module"),
                "parameters": action.get("parameters", {}),
                "quality_before": action.get("before_profile", {}),
                "quality_after": action.get("after_profile"),
                "changed_records": action.get("changed_records", 0),
                "archived_records": action.get("archived_records", 0),
                "elapsed_seconds": action.get("elapsed_seconds"),
                "audit_report": str(report_path.resolve()),
            }
            _append_jsonl(training_path, record)
            training_count += 1

    audit = {
        **manifest,
        "preprocessing_chunks": summary["chunks_processed"],
        "preprocessing_rows": summary["rows_processed"],
        "training_records_appended": training_count,
        "training_records_file": str(training_path.resolve()),
        "row_count_preserved": summary["rows_processed"] == extracted_count,
    }
    _append_jsonl(root / "audit" / "batch_audit.jsonl", audit)
    if not audit["row_count_preserved"]:
        raise RuntimeError("Processed output row count differs from extracted batch")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, preprocess, and audit one sequential dataset batch"
    )
    parser.add_argument("input_csv")
    parser.add_argument("--output-root", default="outputs/manual_batches")
    parser.add_argument("--batch-index", type=int, required=True,
                        help="Zero-based batch number: 0=first 10%, 1=second 10%, etc.")
    parser.add_argument("--batch-fraction", type=float, default=0.10)
    parser.add_argument("--extract-chunksize", type=int, default=50_000)
    parser.add_argument("--chunksize", type=int, default=1_000)
    parser.add_argument("--model", default=config.PREPROCESSING_MODEL)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--allow-imputation", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_batch(
        args.input_csv, output_root=args.output_root,
        batch_index=args.batch_index, batch_fraction=args.batch_fraction,
        extraction_chunksize=args.extract_chunksize,
        preprocessing_chunksize=args.chunksize, model=args.model,
        max_iterations=args.max_iterations,
        allow_imputation=args.allow_imputation, force=args.force,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
