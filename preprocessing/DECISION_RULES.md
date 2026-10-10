# TrippinnI preprocessing decision rules

This document defines the initial allow-list and safety contract for the
iterative local-LLM preprocessing agent. Detection and profiling do not imply
permission to mutate clinical data.

## Execution loop

1. Profile the current working version and provide only compact metadata to the
   LLM: schema, missingness summaries, distribution summaries, detected issues,
   validation results, and recent action history.
2. The LLM selects at most one registered action.
3. Validate the action name and parameters against the registry.
4. Run the action against a versioned working copy. Capture original values or
   records to the removed/replaced-data archive before mutation.
5. Re-profile and validate the result. Commit only when postconditions pass.
6. Record action, parameters, rationale, before/after profile references,
   archived-data references, status, and errors.
7. Stop when the LLM returns stop, no safe action remains, an approval gate is
   reached, or the configured iteration limit is hit.

The LLM cannot supply Python code, shell commands, imports, or arbitrary method
names. Only registered handlers may execute.

## Initial algorithm catalogue

### Rule-based methods

- Schema and datatype conformance checks: flag values that do not parse under
  the declared/inferred type; preserve original values.
- Missingness characterization: report per-column and, where feasible,
  group-wise missingness. Do not impute solely because values are missing.
- Exact duplicate detection: report exact duplicate candidates. Do not delete
  repeated clinical events automatically.
- Candidate-key and referential-integrity validation: report violations and
  preserve patient/encounter/table relationships.
- Temporal ordering checks: use explicit or high-confidence learned temporal
  constraints; flag violations rather than rewriting timestamps.
- Robust IQR outlier flags: detection only unless a separate approved policy
  authorizes a transformation.
- Unit normalization: only use explicit, validated source-to-target mappings
  with compatible dimensions. Unknown or ambiguous units require review.
- Categorical normalization: only apply explicit mappings; retain source labels
  in the archive.

### Machine-learning methods

- Isolation Forest: anomaly scoring for suitable numeric/feature representations.
- COPOD (PyOD): copula-based outlier detection for suitable tabular features.
- LOF and One-Class SVM: optional numeric anomaly signals where sample size and
  feature assumptions make them appropriate.
- Constraint inference ensemble: infer empirical temporal, identifier
  dependency, and numeric support patterns. These are dataset-derived signals,
  not authoritative clinical reference ranges.
- Missingness-pattern analysis: characterize associations in missingness before
  proposing an imputation strategy.

### Deep-learning methods

- Autoencoder reconstruction error: optional anomaly signal when data size,
  representation, and available compute justify it. It is not the default on
  low-memory machines.
- DL imputation is not enabled by default. It requires a separately evaluated
  method, explicit policy, and recorded uncertainty.

## Imputation policy

Imputation is not an unconditional default. The agent must first report
missingness, assess column semantics and the downstream objective, and select
only an explicitly registered method. Numeric median and categorical mode are
baseline candidates for controlled experiments, not universal clinical rules.
Time series and repeated measurements must not be imputed without respecting
patient/encounter grouping and temporal order. Preserve the original missing
mask and record every imputed cell, method, parameters, and output value.

## Removed/replaced-data archive

Before a row, column, or cell is removed or overwritten, archive the original
value(s) and source locator (table, row key/index, column), action ID, run ID,
timestamp, reason, and transformation parameters. Keep the archive separate
from ordinary metadata/logs; logs should contain references rather than
unnecessary raw clinical values. Protect archive files with access controls and
encryption appropriate to the dataset. Never commit EHR data or archives to Git.

## Mandatory invariants

- Never mutate the source dataset in place.
- Never delete records based only on fuzzy similarity or anomaly score.
- Never treat an empirical distribution boundary as a clinical reference range.
- Preserve identifiers, patient/encounter relationships, repeated measurements,
  and temporal meaning.
- Re-profile after each committed action.
- Failed validation means no commit; preserve enough state to roll back.
- Bound iterations and prevent repeated application of the same action without
  new evidence.
- Record provenance and before/after metrics for every committed action.
- Use synthetic/de-identified fixtures for tests; do not put clinical data in
  logs, Git, or test artifacts.

## Current implementation boundary

preprocessing/agent.py provides the local Ollama planner and bounded iterative
control loop. Application-specific profile_fn and action handlers must be
wired to the existing profiler/detectors and versioned data writer. The current
detection pipeline remains flag-only until those handlers and their tests are
integrated; this agent module does not silently turn existing detectors into
mutating transformations.
