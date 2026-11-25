"""Recursive dataset extraction and deterministic preprocessing CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocessing.batch_pipeline import run_directory_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find every CSV/CSV.GZ in a dataset directory, extract a "
            "2.5% batch, preprocess every extracted table, and generate audit records."
        )
    )
    parser.add_argument(
        "--dataset-type", required=True, choices=("csv",),
        help="CSV dataset directory (recursive .csv and .csv.gz discovery).",
    )
    parser.add_argument(
        "--dataset-path", required=True, type=Path,
        help="Root directory containing your dataset's nested folders and CSV files.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/manual_batches"),
        help="Base report directory outside the source dataset and repository. Each run creates run_NNN_YYYYMMDD_HHMMSS underneath it; use the exact run folder with --resume.",
    )
    parser.add_argument("--sample-fraction", type=float, default=0.025,
                        help="Fraction of patients/rows to extract (default: 0.025).")
    parser.add_argument("--extraction-chunksize", type=int, default=5_000)
    parser.add_argument(
        "--chunksize", type=int, default=5_000,
        help="Rows per in-memory preprocessing chunk; all extracted rows are processed.",
    )
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Reproducible patient cohort seed for MIMIC-IV.")
    parser.add_argument("--profile-mode", choices=("fast", "full"), default="fast",
                        help="Fast skips exact cardinality, quantiles, and duplicate scans; full computes them.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing output directory using run_state.json.")
    parser.add_argument(
        "--allow-imputation", action="store_true",
        help="Enable approval-gated median imputation only after human approval.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.dataset_path.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(
            f"--dataset-path must be a directory containing nested CSV files: {source}"
        )
    if output == source or source in output.parents:
        raise ValueError("Output directory must be outside the source dataset directory.")

    result = run_directory_batch(
        source, output_root=output,
        batch_fraction=args.sample_fraction,
        extraction_chunksize=args.extraction_chunksize,
        preprocessing_chunksize=args.chunksize,
        allow_imputation=args.allow_imputation,
        random_seed=args.random_seed,
        resume=args.resume,
        profile_mode=args.profile_mode,
    )
    print(json.dumps(result, indent=2, default=str))
    print(f"Run status: {result['run_status']}")
    print(f"Failed tables: {result['tables_failed_count']}")
    print(f"Pipeline log: {result['log_file']}")
    print("Finished recursive extraction and preprocessing.")
    print("Read multi_table_manifest.json for per-table row counts and sampling methods.")
    print("Classifier records are candidates for review, not validated ground-truth labels.")
    return 1 if result["tables_failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
