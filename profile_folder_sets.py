"""
Standalone profiling runner for files or directory sets.

Profiles only the first --max-rows rows from each CSV/CSV.GZ file.
Extracts a capped CSV copy and outputs profile JSONs to the user-selected root.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

import config
from profiling.profiler import DatasetProfilerEngine


SUPPORTED = {".csv", ".gz"}


def is_csv(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".csv") or name.endswith(".csv.gz")


def collect_files(files: list[Path], folders: list[Path], recursive: bool) -> list[Path]:
    collected: set[Path] = set()
    for file in files:
        file = file.expanduser().resolve()
        if not file.is_file() or not is_csv(file):
            raise ValueError(f"Not a supported CSV file: {file}")
        collected.add(file)
    for folder in folders:
        folder = folder.expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(folder)
        iterator = folder.rglob("*") if recursive else folder.glob("*")
        collected.update(p.resolve() for p in iterator if p.is_file() and is_csv(p))
    return sorted(collected)


def chunks_limited(path: Path, chunk_size: int, row_limit: int) -> Iterable[pd.DataFrame]:
    rows_yielded = 0
    reader = pd.read_csv(path, chunksize=chunk_size, low_memory=False, compression="infer")
    try:
        for chunk in reader:
            if rows_yielded >= row_limit:
                break
            remaining = row_limit - rows_yielded
            chunk = chunk.iloc[:remaining]
            if chunk.empty:
                break
            rows_yielded += len(chunk)
            yield chunk
            if rows_yielded >= row_limit:
                break
    finally:
        reader.close()


def safe_table_name(path: Path, roots: list[Path]) -> str:
    # Use a relative path when possible to avoid collisions across subfolders.
    for root in roots:
        try:
            rel = path.relative_to(root)
            label = root.name + "_" + str(rel.with_suffix("")).replace("/", "_").replace("\\", "_")
            break
        except ValueError:
            continue
    else:
        label = path.stem
    label = label.replace(".csv", "").replace(".gz", "")
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{label}_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile CSV/CSV.GZ files from one or more folders; only the first N rows are read."
    )
    parser.add_argument("--folder", action="append", type=Path, default=[],
                        help="Input folder. Repeat this flag for multiple folder sets.")
    parser.add_argument("--file", action="append", type=Path, default=[],
                        help="Specific input file. Repeat for multiple files.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output root directory chosen by you.")
    parser.add_argument("--max-rows", type=int, default=20000,
                        help="Maximum initial rows read per file (default: 20000).")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Read chunk size (defaults to config.CSV_CHUNK_SIZE).")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Only scan immediate files inside each input folder.")
    args = parser.parse_args()

    if args.max_rows <= 0:
        parser.error("--max-rows must be a positive integer")
    if not args.folder and not args.file:
        parser.error("Provide at least one --folder or --file")
    chunk_size = args.chunk_size or config.CSV_CHUNK_SIZE
    if chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    folders = [p.expanduser().resolve() for p in args.folder]
    files = collect_files(args.file, folders, recursive=not args.no_recursive)
    if not files:
        print("No CSV or CSV.GZ files found.")
        return

    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Redirect the existing ReportWriter output location to this requested root.
    config.OUTPUT_ROOT = output_root
    engine = DatasetProfilerEngine()
    report_dir = output_root / "reports" / "profiling"

    print(f"Input files found: {len(files)}")
    print(f"Output directory: {output_root}")
    print(f"Per-file row cap: {args.max_rows:,}")
    print(f"Chunk size: {chunk_size:,}")

    successes, failures = 0, []
    for index, source in enumerate(files, start=1):
        table_name = safe_table_name(source, folders)
        print(f"\n[{index}/{len(files)}] Profiling: {source}")
        try:
            profile, sample = engine.profile_chunks(
                table_name,
                chunks_limited(source, chunk_size, args.max_rows),
                sample_size=config.MAX_ROWS_FOR_DETECTION,
                sample_seed=config.DETECTION_SAMPLE_SEED,
            )
            actual_rows = profile.get("dataset", {}).get("rows", 0)
            report_path = report_dir / f"{table_name}_profile.json"
            print(f"  Rows profiled: {actual_rows:,}")
            print(f"  Extracted prefix file: {extracted_path}")\n            print(f"  Profile JSON: {report_path}")
            successes += 1
        except Exception as exc:
            failures.append((source, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    print("\nBatch profiling finished.")
    print(f"Successful: {successes}/{len(files)}")
    print(f"Failed: {len(failures)}")
    if failures:
        print("Failures:")
        for source, error in failures:
            print(f"  {source}: {error}")


if __name__ == "__main__":
    main()
