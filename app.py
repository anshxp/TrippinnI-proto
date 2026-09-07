from pathlib import Path

from core.pipeline import Pipeline


DATA_ROOT = Path("data/raw")


def resolve_mimic_root() -> Path:
    """Find the MIMIC-IV version directory below data/raw."""
    direct = DATA_ROOT
    candidates = []

    for hosp_dir in direct.rglob("hosp"):
        icu_dir = hosp_dir.parent / "icu"
        if icu_dir.is_dir():
            candidates.append(hosp_dir.parent)

    unique_candidates = sorted({path.resolve() for path in candidates})

    if not unique_candidates:
        raise FileNotFoundError(
            "Could not find a MIMIC-IV dataset under data/raw. "
            "Expected a directory containing both 'hosp' and 'icu'."
        )

    if len(unique_candidates) > 1:
        paths = "\n".join(f"- {path}" for path in unique_candidates)
        raise RuntimeError(
            "Found multiple MIMIC-IV dataset roots under data/raw:\n" + paths
        )

    return unique_candidates[0]


pipeline = Pipeline()

dataset_path = resolve_mimic_root()
print(f"Using MIMIC-IV dataset root: {dataset_path}")

pipeline.run(
    dataset_type="mimic",
    dataset_path=dataset_path,
)

orchestrator = pipeline.orchestrator

print()
print("=" * 60)
print("Loaded Tables")
print("=" * 60)

for table in orchestrator.get_tables():
    print(table)

print()
print("=" * 60)
print("Profiling Summary")
print("=" * 60)

for table, profile in orchestrator.get_profiles().items():
    print()
    print(table)
    print("  shape:", profile["dataset"]["shape"])
    print("  missing %:", profile["dataset"]["missing_percentage"])

print()
print("=" * 60)
print("Quality Detection Summary")
print("=" * 60)

for table, result in orchestrator.get_quality_results().items():
    print()
    print(table)
    print("  total issues:", result.total_issues)
    print("  by detector:", result.detector_summary())
    print("  by severity:", result.severity_summary())
