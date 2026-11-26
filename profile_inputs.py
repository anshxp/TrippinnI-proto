"""Standalone recursive CSV profiler.

Examples:
  python profile_inputs.py --input path/to/file.csv --output-dir path/to/reports
  python profile_inputs.py --input path/to/folder --output-dir path/to/reports
  python profile_inputs.py --input folder_a folder_b file.csv --output-dir reports
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config
from profiling.profiler import DatasetProfilerEngine


SUPPORTED_SUFFIXES = (".csv", ".csv.gz")


def is_supported(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(SUPPORTED_SUFFIXES)


def collect_inputs(inputs: list[Path], recursive: bool) -> tuple[list[Path], dict[Path, Path]]:
    files: list[Path] = []
    relative_paths: dict[Path, Path] = {}
    for raw in inputs:
        source = raw.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Input path not found: {source}")
        if source.is_file():
            if not is_supported(source):
                print(f"SKIP unsupported file: {source}")
                continue
            files.append(source)
            relative_paths[source] = Path(source.parent.name) / source.name
        elif source.is_dir():
            candidates = source.rglob("*") if recursive else source.glob("*")
            for candidate in candidates:
                if candidate.is_file() and is_supported(candidate):
                    files.append(candidate.resolve())
                    # Keep the selected folder's name and its internal structure.
                    relative_paths[candidate.resolve()] = Path(source.name) / candidate.relative_to(source)
    # Stable order and deduplicate repeated paths.
    unique = sorted(set(files), key=lambda p: str(p).lower())
    return unique, relative_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile one or more CSV/CSV.GZ files or folder trees, writing JSON reports to a chosen output directory."
    )
    parser.add_argument("--input", nargs="+", type=Path, required=True,
                       help="One or more files and/or directories. Directories are scanned recursively by default.")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Destination root for profile reports. This directory is user-defined.")
    parser.add_argument("--non-recursive", action="store_true",
                       help="Only inspect files directly inside each supplied directory.")
    parser.add_argument("--chunksize", type=int, default=config.CSV_CHUNK_SIZE,
                       help="CSV chunk size for streaming profiles.")
    parser.add_argument("--sample-size", type=int, default=config.MAX_ROWS_FOR_DETECTION,
                       help="Reservoir sample size for quality/outlier screening.")
    parser.add_argument("--seed", type=int, default=config.DETECTION_SAMPLE_SEED,
                       help="Random seed for reproducible reservoir sampling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_dir.expanduser().resolve()
    files, relative_paths = collect_inputs(args.input, recursive=not args.non_recursive)
    if not files:
        print("No supported CSV or CSV.GZ files found.")
        return 2

    profiler = DatasetProfilerEngine(output_dir=output_root)
    results = []
    failed = []

    print(f"Files discovered: {len(files)}")
    print(f"Output directory: {output_root}")
    for index, source in enumerate(files, 1):
        rel = relative_paths.get(source, Path(source.name))
        report_subdir = rel.parent
        table_name = source.name[:-7] if source.name.lower().endswith(".csv.gz") else source.stem
        print(f"\n[{index}/{len(files)}] Profiling {source}")
        print(f"Report destination folder: {output_root / report_subdir}")
        try:
            chunks = pd.read_csv(source, chunksize=args.chunksize, low_memory=False)
            profile, sample = profiler.profile_chunks(
                table_name, chunks, sample_size=args.sample_size,
                sample_seed=args.seed, relative_output_path=report_subdir,
            )
            dataset = profile.get("dataset", {})
            results.append({
                "source_file": str(source), "report_file": str(output_root / report_subdir / f"{table_name}_profile.json"),
                "rows": dataset.get("rows"), "columns": dataset.get("columns"),
                "outlier_screening_sample_rows": len(sample) if sample is not None else 0,
            })
        except Exception as exc:
            failed.append({"file": str(source), "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR: {type(exc).__name__}: {exc}")

    manifest = {"input_paths": [str(p.expanduser().resolve()) for p in args.input],
                "output_dir": str(output_root), "files_discovered": len(files),
                "profiles_generated": len(results), "failures": failed, "results": results}
    manifest_path = output_root / "profile_manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print(f"Profiles generated: {len(results)} / {len(files)}")
    if failed:
        print(f"Failed files: {len(failed)} (see manifest)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
