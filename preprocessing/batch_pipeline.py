"""Extract a sequential CSV batch, preprocess it deterministically, and emit audit/training records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.csv_runtime import count_csv_rows, run_csv_chunks


def _source_fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()


def extract_row_window(source: Path, destination: Path, *, start_row: int, end_row: int, chunksize: int = 50_000) -> int:
    if start_row < 0 or end_row <= start_row:
        raise ValueError("Invalid row window")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    written = offset = 0
    try:
        for frame in pd.read_csv(source, chunksize=chunksize, low_memory=False):
            chunk_end = offset + len(frame)
            left, right = max(start_row, offset), min(end_row, chunk_end)
            if left < right:
                frame.iloc[left-offset:right-offset].to_csv(
                    temp, mode="a" if written else "w", header=(written == 0), index=False
                )
                written += right - left
            offset = chunk_end
            if offset >= end_row:
                break
        if written != end_row - start_row:
            raise ValueError(f"Expected {end_row-start_row} rows, extracted {written}")
        temp.replace(destination)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return written


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str) + "\n")


def run_batch(
    source_path: str | Path, *, output_root: str | Path, batch_index: int,
    batch_fraction: float = 0.10, extraction_chunksize: int = 50_000,
    preprocessing_chunksize: int = 1_000, allow_imputation: bool = False,
) -> dict[str, Any]:
    source, root = Path(source_path), Path(output_root)
    if not source.is_file():
        raise FileNotFoundError(source)
    if "".join(source.suffixes[-2:]).lower() not in {".csv", ".csv.gz"}:
        raise ValueError("Only .csv and .csv.gz sources are supported")
    if batch_index < 0 or not 0 < batch_fraction <= 1:
        raise ValueError("batch_index must be >= 0 and batch_fraction in (0, 1]")
    if min(extraction_chunksize, preprocessing_chunksize) <= 0:
        raise ValueError("Chunk sizes must be positive")

    total = count_csv_rows(source, chunksize=extraction_chunksize)
    if total == 0:
        raise ValueError("Source contains no data rows")
    batch_rows = max(1, int(total * batch_fraction))
    start, end = batch_index * batch_rows, min((batch_index + 1) * batch_rows, total)
    if start >= total:
        raise ValueError(f"Batch {batch_index} starts beyond the {total} data rows")

    fingerprint = _source_fingerprint(source)
    stem = source.name[:-7] if source.name.lower().endswith(".csv.gz") else source.stem
    batch_name = f"{stem}_batch_{batch_index+1:03d}"
    batch_root = root / batch_name
    manifest_path = batch_root / "batch_manifest.json"
    if batch_root.exists():
        raise FileExistsError(f"{batch_root} already exists; choose a new output root to avoid duplicate records")
    batch_root.mkdir(parents=True)
    extracted = batch_root / f"{batch_name}_raw.csv"
    rows_extracted = extract_row_window(source, extracted, start_row=start, end_row=end, chunksize=extraction_chunksize)

    preprocessing = run_csv_chunks(
        extracted, output_dir=batch_root / "preprocessing",
        chunksize=preprocessing_chunksize, max_rows=None,
        allow_imputation=allow_imputation,
    )
    manifest = {
        "batch_name": batch_name, "source_fingerprint": fingerprint,
        "source_file_name": source.name, "source_row_count": total,
        "batch_index": batch_index, "batch_fraction": batch_fraction,
        "start_row_inclusive": start, "end_row_exclusive": end,
        "rows_extracted": rows_extracted, "rows_processed": preprocessing["rows_processed"],
        "raw_batch_file": str(extracted), "preprocessing": preprocessing,
        "sampling_method": "sequential_row_window",
        "relational_integrity_warning": "Row-window extraction does not preserve subject/encounter groups. Do not use as a patient-level MIMIC-IV split.",
        "training_label_status": "candidate_requires_review",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    audit_path = root / "audit" / "batch_audit.jsonl"
    _append_jsonl(audit_path, {
        "batch_name": batch_name, "source_fingerprint": fingerprint,
        "rows_extracted": rows_extracted, "rows_processed": preprocessing["rows_processed"],
        "row_count_preserved": rows_extracted == preprocessing["rows_processed"],
        "manifest": str(manifest_path),
    })
    training_path = root / "classifier_training" / "algorithm_selection_records.jsonl"
    for chunk in preprocessing["chunk_reports"]:
        report_file = Path(chunk["chunk_report"])
        report = json.loads(report_file.read_text(encoding="utf-8"))
        for action in report.get("actions", []):
            _append_jsonl(training_path, {
                "record_id": f"{report['run_id']}:{action.get('action_id', action.get('iteration'))}",
                "run_id": report["run_id"], "batch_name": batch_name,
                "metadata_features": action.get("before_profile", report.get("before_profile")),
                "selected_algorithm": action.get("action"),
                "algorithm_module": action.get("module"),
                "parameters": action.get("parameters", {}),
                "selection_rule_id": action.get("rule_id"),
                "selection_rationale": action.get("rationale"),
                "execution_status": action.get("status"),
                "before_profile": action.get("before_profile"),
                "after_profile": action.get("after_profile"),
                "changed_records": action.get("changed_records", 0),
                "requires_human_validation": True,
                "clinical_validity": "not_established",
                "label_status": "candidate_requires_review",
            })
    return {"manifest": str(manifest_path), "audit_file": str(audit_path), "training_records": str(training_path), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic 10% extraction and audited preprocessing")
    parser.add_argument("input_csv")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-fraction", type=float, default=0.10)
    parser.add_argument("--output-root", default="outputs/manual_batches")
    parser.add_argument("--extraction-chunksize", type=int, default=50_000)
    parser.add_argument("--chunksize", type=int, default=1_000)
    parser.add_argument("--allow-imputation", action="store_true", help="Enable only if separately approved by a human")
    args = parser.parse_args()
    result = run_batch(
        args.input_csv, output_root=args.output_root, batch_index=args.batch_index,
        batch_fraction=args.batch_fraction, extraction_chunksize=args.extraction_chunksize,
        preprocessing_chunksize=args.chunksize, allow_imputation=args.allow_imputation,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
