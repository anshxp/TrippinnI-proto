from core.pipeline import Pipeline

pipeline = Pipeline()

pipeline.run(
    dataset_type="synthea",
    dataset_path="data/raw/synthea"
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

print("Schema")

print("=" * 60)

schema = orchestrator.get_schema()

for table, meta in schema.items():

    print()

    print(table)

    print(meta["shape"])