"""CLI entry point: extract a physical dataset batch, profile it, preprocess it, and audit it.

This branch intentionally uses deterministic rules rather than Qwen/Ollama.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocessing.batch_pipeline import run_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a physical CSV batch (10% by default), profile and preprocess "
            "every extracted row, then write audit and classifier-candidate records."
        )
    )
    parser.add_argument(
        "--dataset-type", required=True, choices=("csv",),
        help="Input type. This extraction workflow currently supports one CSV/CSV.GZ table.",
    )
    parser.add_argument(
        "--dataset-path", required=True, type=Path,
        help="Path to the source .csv or .csv.gz file. The source is never modified.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/manual_batches"),
        help="Separate output directory (default: outputs/manual_batches; ignored by Git).",
    )
    parser.add_argument(
        "--batch-index", type=int, default=0,
        help="Zero-based batch number. 0 extracts the first 10%%; 1 extracts the next 10%%.",
    )
    parser.add_argument(
        "--sample-fraction", type=float, default=0.10,
        help="Fraction of source rows per batch (default: 0.10).",
    )
    parser.add_argument(
        "--extraction-chunksize", type=int, default=50_000,
        help="Rows read per chunk while physically extracting the batch.",
    )
    parser.add_argument(
        "--chunksize", type=int, default=1_000,
        help="Rows profiled/preprocessed per in-memory chunk. All extracted rows are processed.",
    )
    parser.add_argument(
        "--allow-imputation", action="store_true",
        help="Permit the separately approval-gated median imputation action. Off by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.dataset_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(
            f"--dataset-path must point to one .csv or .csv.gz file, not a directory: {source}"
        )
    suffixes = "".join(source.suffixes[-2:]).lower()
    if suffixes not in {".csv", ".csv.gz"}:
        raise ValueError(f"Unsupported input file: {source.name}. Expected .csv or .csv.gz.")
    if source == output_root or source in output_root.parents:
        raise ValueError("Output directory must not be the source file or a parent of the source file.")

    result = run_batch(
        source,
        output_root=output_root,
        batch_index=args.batch_index,
        batch_fraction=args.sample_fraction,
        extraction_chunksize=args.extraction_chunksize,
        preprocessing_chunksize=args.chunksize,
        allow_imputation=args.allow_imputation,
    )
    print(json.dumps(result, indent=2, default=str))
    print("\\nCompleted. Check the manifest, extracted CSV, processed CSV, audit JSONL, and candidate training JSONL.")
    print("Candidate records are not ground-truth labels; review and validate them before classifier training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
