# Module 3: auditable remediation

This module was selectively ported from the remediation branch onto the current
`main` baseline. The older branch's orchestrator was not copied because it
predates the current profiling, checkpoint/resume, and bounded-memory work.

## Status and invocation

The executor is a standalone API for an explicitly bounded in-memory dataset.
It is **not automatically run by the profiling orchestrator**. This avoids
silently mutating data during a profiling-only run.

```python
from preprocessing.executor import RemediationExecutor
from preprocessing.policy import RemediationPolicy

policy = RemediationPolicy(
    enabled=True,
    impute_numeric=True,       # median for numeric columns
    impute_categorical=True,   # explicit UNKNOWN token
    remove_exact_duplicates=False,
)
result = RemediationExecutor(policy).run(
    dataset={"table_name": bounded_dataframe},
    profiles={"table_name": profile_metadata},
)
```

The source DataFrame is copied. The returned result includes transformed
DataFrames, per-transformation audit records, and row/cell counts. Persist
audit logs and review changes before using output downstream.

## Safety defaults

All remediation is disabled by default in `config.py`. Configuration flags
are provided for later orchestration, but they do not by themselves connect the
executor to the profiling pipeline. Exact-row duplicate removal is opt-in
because identical rows can represent legitimate repeated clinical events.
Outliers are retained; statistical rarity alone is not evidence that a clinical
value is erroneous. Identifier and datetime columns are protected from generic
imputation.

## Implemented operations

- Numeric median imputation (opt-in).
- Categorical explicit `UNKNOWN` token imputation (opt-in).
- Profile-guided, loss-minimizing datatype conversion.
- Leading/trailing whitespace normalization.
- Exact whole-row duplicate removal (opt-in).
- Transformation provenance records with row, column, method, old/new value,
  detector, and UTC timestamp where applicable.
- Outlier retention and no-op categorical encoding placeholders; these do not
  alter data.

Before clinical use, validate against synthetic/de-identified fixtures and
inspect missingness patterns by table, time, and relevant subgroups. Do not
impute patient IDs, encounter IDs, linkage keys, or timestamps generically.
