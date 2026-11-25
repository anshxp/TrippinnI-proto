"""Recursive multi-table extraction and deterministic preprocessing.

For MIMIC-IV, select a reproducible patient cohort and filter dependent tables
through subject_id, hadm_id, and stay_id. For non-MIMIC folders, extract a
deterministic 10% row window from every discovered CSV/CSV.GZ file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.csv_runtime import count_csv_rows, run_csv_chunks


CSV_SUFFIXES = (".csv", ".csv.gz")


def _is_csv(path: Path) -> bool:
    return path.name.lower().endswith(CSV_SUFFIXES)


def _stem(path: Path) -> str:
    return path.name[:-7] if path.name.lower().endswith(".csv.gz") else path.stem


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(
        f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    ).hexdigest()


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=str) + "\n")


def _header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _read_key_set(path: Path, key: str, *, chunksize: int) -> set[str]:
    values: set[str] = set()
    for frame in pd.read_csv(path, usecols=[key], dtype={key: "string"},
                             chunksize=chunksize, low_memory=False):
        values.update(frame[key].dropna().astype(str).tolist())
    return values


def _collect_child_keys(
    path: Path, *, key: str, selected_subjects: set[str], chunksize: int
) -> set[str]:
    keys: set[str] = set()
    for frame in pd.read_csv(
        path, usecols=["subject_id", key],
        dtype={"subject_id": "string", key: "string"},
        chunksize=chunksize, low_memory=False,
    ):
        mask = frame["subject_id"].astype("string").isin(selected_subjects)
        keys.update(frame.loc[mask, key].dropna().astype(str).tolist())
    return keys


def _write_filtered(
    source: Path, destination: Path, *, key_columns: list[str],
    selected_subjects: set[str], selected_hadm: set[str],
    selected_stays: set[str], chunksize: int, logger: logging.Logger | None = None,
    table_name: str = "",
) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.unlink(missing_ok=True)
    rows = source_rows = 0
    try:
        for frame in pd.read_csv(source, chunksize=chunksize, low_memory=False):
            source_rows += len(frame)
            mask = pd.Series(False, index=frame.index)
            if "subject_id" in key_columns:
                mask |= frame["subject_id"].astype("string").isin(selected_subjects)
            if "hadm_id" in key_columns and selected_hadm:
                mask |= frame["hadm_id"].astype("string").isin(selected_hadm)
            if "stay_id" in key_columns and selected_stays:
                mask |= frame["stay_id"].astype("string").isin(selected_stays)
            selected = frame.loc[mask] if key_columns else frame
            if not selected.empty:
                selected.to_csv(temp, mode="a" if rows else "w",
                                header=(rows == 0), index=False)
                rows += len(selected)
            if logger:
                logger.info(
                    "EXTRACTION_PROGRESS table=%s source_rows_scanned=%d rows_selected=%d",
                    table_name, source_rows, rows,
                )
        if rows == 0:
            pd.read_csv(source, nrows=0).to_csv(temp, index=False)
        temp.replace(destination)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return source_rows, rows


def _write_row_fraction(
    source: Path, destination: Path, *, fraction: float, chunksize: int
) -> tuple[int, int]:
    total = count_csv_rows(source, chunksize=chunksize)
    if total == 0:
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(source, nrows=0).to_csv(destination, index=False)
        return total, 0
    end = max(1, int(total * fraction))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.unlink(missing_ok=True)
    written = 0
    try:
        for frame in pd.read_csv(source, chunksize=chunksize, low_memory=False):
            take = frame.iloc[:max(0, end - written)]
            if not take.empty:
                take.to_csv(temp, mode="a" if written else "w",
                            header=(written == 0), index=False)
                written += len(take)
            if written >= end:
                break
        temp.replace(destination)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return total, written


def run_directory_batch(
    source_root: str | Path, *, output_root: str | Path,
    batch_fraction: float = 0.025, extraction_chunksize: int = 5_000,
    preprocessing_chunksize: int = 5_000, allow_imputation: bool = False,
    random_seed: int = 42, resume: bool = False, profile_mode: str = "fast",
) -> dict[str, Any]:
    """Extract/process every CSV recursively; use a patient cohort when MIMIC keys exist."""
    source = Path(source_root).expanduser().resolve()
    base_root = Path(output_root).expanduser().resolve()
    root = base_root
    if not source.is_dir():
        raise NotADirectoryError(f"Dataset path must be a directory: {source}")
    if base_root == source or source in base_root.parents:
        raise ValueError("Output directory must be outside the source dataset directory.")
    if not resume:
        base_root.mkdir(parents=True, exist_ok=True)
        existing_runs = [p for p in base_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
        run_numbers = [int(p.name.split("_")[1].split("-")[0]) for p in existing_runs
                       if len(p.name.split("_")) > 1 and p.name.split("_")[1].split("-")[0].isdigit()]
        next_number = max(run_numbers or [0]) + 1
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        root = base_root / f"run_{next_number:03d}_{timestamp}"
        while root.exists():
            next_number += 1
            root = base_root / f"run_{next_number:03d}_{timestamp}"
    if not 0 < batch_fraction <= 1:
        raise ValueError("batch_fraction must be in (0, 1].")
    if min(extraction_chunksize, preprocessing_chunksize) <= 0:
        raise ValueError("Chunk sizes must be positive.")
    files = sorted(p for p in source.rglob("*") if p.is_file() and _is_csv(p))
    if not files:
        raise FileNotFoundError(f"No .csv or .csv.gz files found recursively under {source}")
    state_path = root / "run_state.json"
    log_path = root / "logs" / "pipeline.log"
    if resume and root.exists() and any(root.iterdir()) and not state_path.is_file():
        raise FileExistsError(
            f"Cannot resume: output directory lacks this pipeline's run_state.json: {root}"
        )
    elif resume:
        raise FileNotFoundError(f"Cannot resume: no existing run state at {state_path}")
    root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trippinni.batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if Path(state.get("source_root", "")).resolve() != source:
            raise ValueError("Resume source directory differs from the original run.")
        logger.info("Resuming existing run at %s", root)
    else:
        state = {"version": 1, "source_root": str(source), "output_root": str(root),
                 "started_at": datetime.now(timezone.utc).isoformat(), "tables": {}}
        _save_state(state_path, state)
        logger.info("Started new run. Discovered %d CSV files.", len(files))

    # Identify the MIMIC-IV patient table. If present, sample patients once and
    # derive the encounter/stay key sets used by every dependent table.
    patient_files = [p for p in files if _stem(p).lower() == "patients"
                     and "subject_id" in _header(p)]
    mimic_mode = bool(patient_files)
    selected_subjects: set[str] = set()
    selected_hadm: set[str] = set()
    selected_stays: set[str] = set()
    patient_count = selected_patient_count = 0

    if mimic_mode:
        patient_path = patient_files[0]
        logger.info("BUILD_COHORT patient_table=%s", patient_path.relative_to(source))
        try:
            subject_values = _read_key_set(patient_path, "subject_id",
                                           chunksize=extraction_chunksize)
        except Exception:
            logger.exception("COHORT_FAILED stage=read_patient_subject_ids file=%s", patient_path)
            raise
        patient_count = len(subject_values)
        if patient_count == 0:
            raise ValueError(f"No subject_id values found in {patient_path}")
        selected_patient_count = max(1, int(patient_count * batch_fraction))
        selected_subjects = set(
            pd.Series(sorted(subject_values), dtype="string")
            .sample(n=selected_patient_count, random_state=random_seed)
            .astype(str).tolist()
        )
        admissions = [p for p in files if _stem(p).lower() == "admissions"
                      and {"subject_id", "hadm_id"}.issubset(_header(p))]
        icustays = [p for p in files if _stem(p).lower() == "icustays"
                    and {"subject_id", "stay_id"}.issubset(_header(p))]
        if admissions:
            try:
                selected_hadm = _collect_child_keys(
                    admissions[0], key="hadm_id", selected_subjects=selected_subjects,
                    chunksize=extraction_chunksize,
                )
            except Exception:
                logger.exception("COHORT_FAILED stage=read_admissions_keys file=%s", admissions[0])
                raise
        if icustays:
            try:
                selected_stays = _collect_child_keys(
                    icustays[0], key="stay_id", selected_subjects=selected_subjects,
                    chunksize=extraction_chunksize,
                )
            except Exception:
                logger.exception("COHORT_FAILED stage=read_icustays_keys file=%s", icustays[0])
                raise

    table_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    audit_path = root / "audit" / "batch_audit.jsonl"
    training_path = root / "classifier_training" / "algorithm_selection_records.jsonl"

    for source_file in files:
        relative = source_file.relative_to(source)
        relative_key = str(relative)
        fingerprint = _fingerprint(source_file)
        extracted_rel = relative.with_name(_stem(source_file) + ".csv") if relative.name.lower().endswith(".csv.gz") else relative
        extracted = root / "extracted" / extracted_rel
        processed_dir = root / "processed" / relative.parent / _stem(source_file)
        entry = state["tables"].get(relative_key, {})
        if entry.get("source_fingerprint") != fingerprint:
            entry = {"source_fingerprint": fingerprint, "extraction_status": "pending",
                     "processing_status": "pending"}
            state["tables"][relative_key] = entry

        saved_result = entry.get("table_result")
        saved_output = (Path(saved_result["preprocessing"]["output_file"])
                        if saved_result and saved_result.get("preprocessing", {}).get("output_file")
                        else None)
        if (entry.get("processing_status") == "complete" and extracted.is_file()
                and saved_result and saved_output and saved_output.is_file()):
            table_results.append(saved_result)
            logger.info("SKIP complete table=%s", relative_key)
            continue

        logger.info("START table=%s", relative_key)
        try:
            if entry.get("extraction_status") == "complete" and extracted.is_file():
                source_rows = int(entry["source_rows"])
                rows_extracted = int(entry["rows_extracted"])
                sampling_mode = entry["sampling_mode"]
                logger.info("RESUME table=%s stage=preprocessing", relative_key)
            else:
                columns = _header(source_file)
                key_columns = [key for key in ("subject_id", "hadm_id", "stay_id") if key in columns]
                if mimic_mode:
                    if "subject_id" in key_columns:
                        sampling_mode = "selected_subject_id_cohort"
                    elif "hadm_id" in key_columns and selected_hadm:
                        sampling_mode = "selected_hadm_id_cohort"
                    elif "stay_id" in key_columns and selected_stays:
                        sampling_mode = "selected_stay_id_cohort"
                    else:
                        sampling_mode = "reference_table_full_copy" if not key_columns else "key_table_no_matching_parent"
                    if sampling_mode == "reference_table_full_copy":
                        extracted.parent.mkdir(parents=True, exist_ok=True)
                        temp_extracted = extracted.with_suffix(extracted.suffix + ".tmp")
                        temp_extracted.unlink(missing_ok=True)
                        source_rows = 0
                        try:
                            for frame in pd.read_csv(source_file, chunksize=extraction_chunksize, low_memory=False):
                                frame.to_csv(temp_extracted, mode="a" if source_rows else "w",
                                             header=(source_rows == 0), index=False)
                                source_rows += len(frame)
                                logger.info(
                                    "EXTRACTION_PROGRESS table=%s source_rows_scanned=%d rows_selected=%d",
                                    relative_key, source_rows, source_rows,
                                )
                            if source_rows == 0:
                                pd.read_csv(source_file, nrows=0).to_csv(temp_extracted, index=False)
                            temp_extracted.replace(extracted)
                        except Exception:
                            temp_extracted.unlink(missing_ok=True)
                            raise
                        rows_extracted = source_rows
                    else:
                        source_rows, rows_extracted = _write_filtered(
                            source_file, extracted, key_columns=key_columns,
                            selected_subjects=selected_subjects, selected_hadm=selected_hadm,
                            selected_stays=selected_stays, chunksize=extraction_chunksize,
                            logger=logger, table_name=relative_key,
                        )
                else:
                    source_rows, rows_extracted = _write_row_fraction(
                        source_file, extracted, fraction=batch_fraction,
                        chunksize=extraction_chunksize,
                    )
                    sampling_mode = "deterministic_first_row_fraction_per_file"
                entry.update({"extraction_status": "complete", "source_rows": source_rows,
                              "rows_extracted": rows_extracted, "sampling_mode": sampling_mode,
                              "extracted_relative_path": str(extracted.relative_to(root))})
                _save_state(state_path, state)
                logger.info("EXTRACTED table=%s source_rows=%d extracted_rows=%d",
                            relative_key, source_rows, rows_extracted)

            if entry.get("processing_status") != "complete" and processed_dir.exists():
                shutil.rmtree(processed_dir)
            entry["processing_status"] = "running"
            _save_state(state_path, state)
            preprocessing = run_csv_chunks(
                extracted, output_dir=processed_dir, chunksize=preprocessing_chunksize,
                max_rows=None, allow_imputation=allow_imputation, profile_mode=profile_mode,
            )
            table_result = {
                "source_relative_path": relative_key,
                "source_fingerprint": fingerprint,
                "source_rows": source_rows,
                "extracted_relative_path": str(extracted.relative_to(root)),
                "rows_extracted": rows_extracted,
                "rows_processed": preprocessing["rows_processed"],
                "sampling_mode": sampling_mode,
                "preprocessing": preprocessing,
            }
            table_results.append(table_result)
            _append_jsonl(audit_path, {
                "source_relative_path": relative_key, "source_fingerprint": fingerprint,
                "sampling_mode": sampling_mode, "source_rows": source_rows,
                "rows_extracted": rows_extracted,
                "rows_processed": preprocessing["rows_processed"],
                "row_count_preserved": rows_extracted == preprocessing["rows_processed"],
                "clinical_validity": "not_established",
            })
            for chunk in preprocessing["chunk_reports"]:
                report = json.loads(Path(chunk["chunk_report"]).read_text(encoding="utf-8"))
                for action in report.get("actions", []):
                    _append_jsonl(training_path, {
                        "record_id": f"{report['run_id']}:{action.get('action_id', action.get('iteration'))}",
                        "run_id": report["run_id"], "source_table": relative_key,
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
            entry.update({"processing_status": "complete", "table_result": table_result,
                          "completed_at": datetime.now(timezone.utc).isoformat()})
            _save_state(state_path, state)
            logger.info("COMPLETE table=%s rows_processed=%d",
                        relative_key, preprocessing["rows_processed"])
        except Exception as exc:
            entry["processing_status"] = "failed"
            entry["last_error"] = f"{type(exc).__name__}: {exc}"
            entry["failed_at"] = datetime.now(timezone.utc).isoformat()
            _save_state(state_path, state)
            logger.exception("FAILED table=%s stage=%s", relative_key,
                             "preprocessing" if entry.get("extraction_status") == "complete" else "extraction")
            failures.append({"source_relative_path": relative_key,
                             "error": entry["last_error"]})
            continue

    manifest = {
        "source_root": str(source),
        "output_root": str(root),
        "batch_fraction": batch_fraction,
        "profile_mode": profile_mode,
        "decision_engine": "deterministic_rules_v1",
        "sampling_mode": "mimic_patient_cohort" if mimic_mode else "per_file_row_fraction",
        "random_seed": random_seed if mimic_mode else None,
        "patient_count": patient_count if mimic_mode else None,
        "selected_patient_count": selected_patient_count if mimic_mode else None,
        "source_files_discovered": len(files),
        "tables": table_results,
        "tables_processed": len(table_results),
        "tables_failed": failures,
        "tables_failed_count": len(failures),
        "run_status": "completed_with_errors" if failures else "completed",
        "resume_supported": True,
        "resume_granularity": "table_stage; a failed preprocessing table restarts from its extracted CSV",
        "log_file": str(log_path),
        "state_file": str(state_path),
        "rows_extracted_total": sum(item["rows_extracted"] for item in table_results),
        "rows_processed_total": sum(item["rows_processed"] for item in table_results),
        "training_label_status": "candidate_requires_review",
        "limitations": (
            ["Patient-level cohort selected from patients.subject_id; dependent tables filtered by subject_id/hadm_id/stay_id. "
             "Tables without these keys are copied in full as reference tables. Verify table-specific foreign keys before downstream use."]
            if mimic_mode else
            ["No patients table detected. A first-row fraction is extracted independently from each file; "
             "cross-file relational integrity is not guaranteed."]
        ),
        "audit_file": str(audit_path),
        "training_records": str(training_path),
    }
    manifest_path = root / "multi_table_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    state["run_status"] = manifest["run_status"]
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_state(state_path, state)
    logger.info("RUN FINISHED status=%s completed=%d failed=%d",
                manifest["run_status"], len(table_results), len(failures))
    file_handler.close()
    console_handler.close()
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)
    return {"manifest": str(manifest_path), **manifest}


def run_batch(
    source_path: str | Path, *, output_root: str | Path, batch_index: int,
    batch_fraction: float = 0.025, extraction_chunksize: int = 5_000,
    preprocessing_chunksize: int = 1_000, allow_imputation: bool = False,
) -> dict[str, Any]:
    """Backward-compatible single-file extraction entry point."""
    source, root = Path(source_path), Path(output_root)
    if not source.is_file() or not _is_csv(source):
        raise FileNotFoundError(f"Expected a .csv or .csv.gz file: {source}")
    if batch_index < 0 or not 0 < batch_fraction <= 1:
        raise ValueError("batch_index must be >= 0 and batch_fraction in (0, 1]")
    total = count_csv_rows(source, chunksize=extraction_chunksize)
    if total == 0:
        raise ValueError("Source contains no data rows")
    batch_rows = max(1, int(total * batch_fraction))
    start, end = batch_index * batch_rows, min((batch_index + 1) * batch_rows, total)
    if start >= total:
        raise ValueError(f"Batch {batch_index} starts beyond {total} rows")
    batch_root = root / f"{_stem(source)}_batch_{batch_index + 1:03d}"
    if batch_root.exists():
        raise FileExistsError(f"{batch_root} already exists; choose a new output root")
    batch_root.mkdir(parents=True, exist_ok=False)
    extracted = batch_root / f"{_stem(source)}_raw.csv"
    temp = extracted.with_suffix(".tmp")
    offset = written = 0
    for frame in pd.read_csv(source, chunksize=extraction_chunksize, low_memory=False):
        left, right = max(start, offset), min(end, offset + len(frame))
        if left < right:
            part = frame.iloc[left - offset:right - offset]
            part.to_csv(temp, mode="a" if written else "w",
                        header=(written == 0), index=False)
            written += len(part)
        offset += len(frame)
        if offset >= end:
            break
    if written != end - start:
        temp.unlink(missing_ok=True)
        raise ValueError(f"Expected {end-start} extracted rows, got {written}")
    temp.replace(extracted)
    preprocessing = run_csv_chunks(
        extracted, output_dir=batch_root / "preprocessing",
        chunksize=preprocessing_chunksize, max_rows=None,
        allow_imputation=allow_imputation,
    )
    manifest = {
        "source_file_name": source.name, "source_fingerprint": _fingerprint(source),
        "source_rows": total, "batch_index": batch_index,
        "batch_fraction": batch_fraction, "rows_extracted": written,
        "rows_processed": preprocessing["rows_processed"],
        "raw_batch_file": str(extracted), "preprocessing": preprocessing,
        "sampling_method": "sequential_row_window",
        "relational_integrity_warning": "Single-table row sampling does not preserve patient relationships.",
        "training_label_status": "candidate_requires_review",
    }
    manifest_path = batch_root / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {"manifest": str(manifest_path), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursive multi-table 10% extraction and audited preprocessing")
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("--batch-fraction", type=float, default=0.025)
    parser.add_argument("--output-root", default="outputs/manual_batches")
    parser.add_argument("--extraction-chunksize", type=int, default=50_000)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--allow-imputation", action="store_true")
    args = parser.parse_args()
    result = run_directory_batch(
        args.input_directory, output_root=args.output_root,
        batch_fraction=args.batch_fraction,
        extraction_chunksize=args.extraction_chunksize,
        preprocessing_chunksize=args.chunksize,
        allow_imputation=args.allow_imputation,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
