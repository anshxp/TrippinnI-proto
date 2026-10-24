# TrippinnI: Dataset-Agnostic Architecture

## Design goal

TrippinnI accepts datasets through registered loader adapters. The core profiling,
quality, context, graph, confidence, reporting, and remediation interfaces must not
depend on a particular vendor, schema, table name, or clinical terminology system.
Domain-specific validation is an optional adapter/plugin, not a core assumption.

## End-to-end stages

1. **Ingestion** — a registered loader discovers files/tables, validates supported
   formats, and exposes tables through a common loader interface. Preserve the
   source; record source identity and load errors.
2. **Profiling and routing** — infer physical types, schema shape, missingness,
   cardinality, distributions, parseability, and bounded-sample metadata. Mark
   estimates as sample-scoped. Do not infer clinical/business truth from statistics.
3. **Structured preprocessing** — validate types and formats, normalize only with
   explicit policy, and treat identifiers, numeric values, categorical values,
   and temporal values as distinct processing families.
4. **Unstructured processing** — optional document/OCR/NLP adapters may produce
   candidate entities, assertions, and relations. Keep source offsets, model/version,
   confidence, and provenance. This stage is not required for structured datasets.
5. **Unified objects and context graph** — represent source fields, entities,
   relationships, temporal objects, terminology mappings, and lineage with generic
   identifiers. Semantic/domain adapters add context when available.
6. **Quality detection** — combine deterministic checks with statistical or ML
   detectors. Findings are evidence-bearing candidates, not automatic proof of error.
7. **Confidence and complexity routing** — aggregate independent evidence only when
   detector scores have compatible meanings. Route ambiguous/high-impact findings to
   review; do not interpret an uncalibrated model score as a clinical probability.
8. **Graph-assisted investigation** — use graph traversal and provenance to explain
   connected findings. Run deeper multi-hop analysis only when the graph and policy
   justify it.
9. **Resolution and feedback** — propose a typed action, validate preconditions,
   write to a separate output, record before/after evidence, and re-profile. Raw source
   data is never overwritten by default.
10. **Quality output and learning** — export machine-readable reports and human-readable
    summaries. Use reviewed outcomes as training labels only with recorded provenance,
    versioned evaluation, and leakage checks.
11. **Auditability** — retain source fingerprint, configuration/version, stage status,
    warnings, errors, model/rule versions, decisions, and transformation lineage.

## Reliability invariants

- Fail early on invalid paths, unsupported formats, malformed configuration, or missing
  required columns; report the exact stage and source.
- A stage must not be marked complete until its outputs have been written atomically
  and validated.
- Resume checkpoints must be versioned. Schema/code version changes invalidate stale
  outputs; source/config fingerprints should be added before enabling cache reuse across
  changed input files.
- Missing or unavailable evidence is represented as unknown/null, not silently as
  zero, false, or valid.
- Every approximate metric identifies its sample and estimation method.
- Detection is read-only. Remediation is a separate, opt-in, policy-gated stage.
- Transformations are idempotent where possible, reversible where feasible, and
  accompanied by before/after checks.
- Model recommendations abstain when evidence is weak or out of distribution.
- No dataset-specific field name, table name, row count, or terminology is required by
  the core pipeline. Dataset/domain rules belong in adapters and configuration.

## Current implementation mapping

The repository already provides loader abstractions, profiling, quality detector
families, confidence aggregation, semantic context, a knowledge graph, cross-table
integrity checks, reports, and optional CSV preprocessing. The generic CLI accepts
an explicit registered loader type and source path. The legacy MIMIC row-count lookup
has been disabled in the orchestrator; unknown counts should be obtained from a
loader when supported or remain unknown.

The following diagram capabilities remain adapter- or implementation-dependent and
must not be treated as complete merely because they appear in the target architecture:
OCR/document layout, clinical NER/assertion/relation extraction, terminology services,
a full temporal/identity/lexical graph suite, calibrated issue-complexity routing,
human approval workflow, reversible general-purpose remediation, and production model
monitoring/retraining.

## Run

```bash
python app.py --dataset-type csv --dataset-path /path/to/dataset
python app.py --dataset-type json --dataset-path /path/to/dataset
```

Use only loader types registered in `loaders/registry.py`. A loader's supported path
shape and file formats remain authoritative.
