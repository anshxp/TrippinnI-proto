#!/usr/bin/env python3
"""Build one provenance-preserving JSON bundle from multiple TrippinnI profiler reports.

This utility combines profiler JSON files only. It does not load clinical rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    # Support either a single profiler report or a previously generated bundle.
    if "tables" in payload and isinstance(payload["tables"], dict):
        raise ValueError(f"{path}: input appears to be a bundle, not an individual profiler report")
    dataset = payload.get("dataset", {})
    table_name = payload.get("table_name") or (
        dataset.get("table_name") if isinstance(dataset, dict) else None
    )
    if not table_name:
        table_name = path.stem
    return {
        "table_name": str(table_name),
        "source_file": path.name,
        "source_sha256": sha256_file(path),
        "report_version": payload.get("report_version", payload.get("profile_version")),
        "generated_at": payload.get("generated_at"),
        "profile": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing profiler .json files")
    parser.add_argument("--output", type=Path, default=Path("artifacts/profiles_bundle.json"))
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    candidates = sorted(args.input_dir.rglob("*.json") if args.recursive else args.input_dir.glob("*.json"))
    if not candidates:
        parser.error(f"no .json files found in {args.input_dir}")

    tables: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, str]] = []
    for path in candidates:
        report = load_report(path)
        name = report["table_name"]
        if name in tables:
            duplicates.append({"table_name": name, "source_file": report["source_file"]})
            name = f"{name}__{Path(report['source_file']).stem}"
            report["table_name"] = name
        tables[name] = report

    bundle = {
        "bundle_schema_version": "1.0.0",
        "bundle_type": "trippinni_multi_table_profiler_bundle",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(candidates),
        "table_count": len(tables),
        "duplicate_table_name_collisions": duplicates,
        "tables": tables,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(f"Bundled {len(candidates)} profile report(s) into {args.output}")
    print(f"Tables: {', '.join(tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
