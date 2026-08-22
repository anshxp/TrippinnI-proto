from core.pipeline import Pipeline

pipeline = Pipeline()

pipeline.run(
    dataset_type="mimic",
    dataset_path="data/raw/mimiciv/3.1",  # the version folder, not its parent
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