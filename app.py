"""Dataset-agnostic CLI entry point for TrippinnI."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.pipeline import Pipeline
from profiling.visualization import ReportVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a dataset, detect quality issues, build context, and generate reports."
    )
    parser.add_argument(
        "--dataset-type",
        required=True,
        help="Registered loader type (for example: csv, json, or a registered domain adapter).",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help="Path to a file or directory supported by the selected loader.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Run the pipeline without attempting report visualization.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not args.dataset_type.strip():
        raise ValueError("--dataset-type must not be empty")

    pipeline = Pipeline()
    orchestrator = pipeline.run(
        dataset_type=args.dataset_type.strip().lower(),
        dataset_path=dataset_path,
    )

    print("\\n" + "=" * 60)
    print("Dataset summary")
    print("=" * 60)
    tables = orchestrator.get_tables()
    print(f"Dataset path: {dataset_path}")
    print(f"Loader type: {args.dataset_type}")
    print(f"Discovered tables: {len(tables)}")

    for table, profile in orchestrator.get_profiles().items():
        dataset = profile.get("dataset", {})
        print(f"\\n{table}")
        print("  shape:", dataset.get("shape", (dataset.get("rows"), dataset.get("columns"))))
        print("  missing %:", dataset.get("missing_percentage", "unavailable"))

    if orchestrator.knowledge_graph is not None:
        print("\\nContext graph")
        print("  nodes:", len(orchestrator.knowledge_graph.nodes))
        print("  edges:", len(orchestrator.knowledge_graph.edges))

    print("\\nQuality detection summary")
    for table, result in orchestrator.get_quality_results().items():
        print(f"  {table}: {result.total_issues} issue(s)")
        print("    by detector:", result.detector_summary())
        print("    by severity:", result.severity_summary())

    if not args.skip_visualization:
        print("\\nVisualization")
        try:
            for name, path in ReportVisualizer().generate().items():
                print(f"  {name}: {path}")
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"  Visualization skipped: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
