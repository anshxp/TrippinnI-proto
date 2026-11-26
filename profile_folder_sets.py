"""
Standalone profiler for individual CSV/CSV.GZ files or multiple folder sets.
Reads and extracts only the first N data rows per file (20,000 by default).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

import config
from profiling.profiler import DatasetProfilerEngine


def is_csv(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".csv") or name.endswith(".csv.gz")


def collect_files(files: list[Path], folders: list[Path], recursive: bool) -> list[Path]:
    collected: set[Path] = set()
    for file in files:
        file = file.expanduser().resolve()
        if not file.is_file() or not is_csv(file):
            raise ValueError(f"Not a supported CSV or CSV.GZ file: {file}")
        collected.add(file)
    for folder in folders:
        folder = folder.expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(folder)
        iterator = folder.rglob("*") if recursive else folder.glob("*")
        collected.update(p.resolve() for p in iterator if p.is_file() and is_csv(p))
    return sorted(collected)


def extract_prefix_chunks(
    source: Path, chunk_size: int, row_limit: int, extracted_path: Path
) -> Iterable[pd.DataFrame]:
    """Yield chunks and write the same capped rows to a plain CSV extract."""
    rows_written = 0
    reader = pd.read_csv(source, chunksize=chunk_size, low_memory=False, compression="infer")
    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.unlink(missing_ok=True)
    try:
        for chunk in reader:
            if rows_written >= row_limit:
                break
            chunk = chunk.iloc[: row_limit - rows_written]
            if chunk.empty:
                break
            chunk.to_csv(
                extracted_path,
                mode="a" if rows_written else "w",
                header=(rows_written == 0),
                index=False,
            )
            rows_written += len(chunk)
            yield chunk
            if rows_written >= row_limit:
                break
    finally:
        reader.close()


def safe_table_name(source: Path, folders: list[Path]) -> str:
    for root in folders:
        try:
            relative = source.relative_to(root)
            label = root.name + "_" + str(relative.with_suffix("")).replace("/", "_").replace("\\", "_")
            break
        except ValueError:
            continue
    else:
        label = source.stem
    label = label.replace(".csv", "").replace(".gz", "")
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
    return f"{label}_{digest}"


def extracted_destination(source: Path, folders: list[Path], output_root: Path) -> Path:
    matched_root = next((root for root in folders if source.is_relative_to(root)), None)
    relative = source.relative_to(matched_root) if matched_root else Path(source.name)
    output_name = relative.name[:-7] if relative.name.lower().endswith(".csv.gz") else relative.name
    root_label = (
        matched_root.name + "_" + hashlib.sha1(str(matched_root).encode("utf-8")).hexdigest()[:6]
        if matched_root else "individual_files"
    )
    return output_root / "extracted" / root_label / relative.parent / output_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile and extract only the first N rows from CSV/CSV.GZ files."
    )
    parser.add_argument("--folder", action="append", type=Path, default=[],
                        help="Input directory; repeat to add folder sets.")
    parser.add_argument("--file", action="append", type=Path, default=[],
                        help="Specific CSV/CSV.GZ file; repeat as needed.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output root directory (chosen by the user).")
    parser.add_argument("--max-rows", type=int, default=20000,
                        help="Rows extracted and profiled per file (default 20000).")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Chunk read size (default uses project config).")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Scan only files directly inside each input folder.")
    args = parser.parse_args()

    if args.max_rows <= 0:
        parser.error("--max-rows must be greater than zero")
    if not args.folder and not args.file:
        parser.error("Supply at least one --folder or --file")
    chunk_size = args.chunk_size or config.CSV_CHUNK_SIZE
    if chunk_size <= 0:
        parser.error("--chunk-size must be greater than zero")

    folders = [p.expanduser().resolve() for p in args.folder]
    files = collect_files(args.file, folders, recursive=not args.no_recursive)
    if not files:
        print("No CSV/CSV.GZ inputs found.")
        return

    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_ROOT = output_root
    engine = DatasetProfilerEngine()
    profile_dir = output_root / "reports" / "profiling"

    print(f"Files found: {len(files)}")
    print(f"Output root: {output_root}")
    print(f"Rows per file cap: {args.max_rows:,}")
    print(f"Read chunk size: {chunk_size:,}")

    completed, failures = 0, []
    for index, source in enumerate(files, 1):
        table_name = safe_table_name(source, folders)
        extract_path = extracted_destination(source, folders, output_root)
        print(f"\n[{index}/{len(files)}] {source}")
        try:
            profile, _sample = engine.profile_chunks(
                table_name,
                extract_prefix_chunks(source, chunk_size, args.max_rows, extract_path),
                sample_size=config.MAX_ROWS_FOR_DETECTION,
                sample_seed=config.DETECTION_SAMPLE_SEED,
            )
            rows = profile.get("dataset", {}).get("rows", 0)
            report_path = profile_dir / f"{table_name}_profile.json"
            print(f"  Extracted/profiled rows: {rows:,}")
            print(f"  Extracted CSV: {extract_path}")
            print(f"  Profile JSON: {report_path}")
            completed += 1
        except Exception as exc:
            failures.append((source, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    print(f"\nCompleted {completed}/{len(files)} files; failed: {len(failures)}")
    for source, error in failures:
        print(f"  FAILED {source}: {error}")


if __name__ == "__main__":
    main()
