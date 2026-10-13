# Missing-value algorithm inventory

This catalogue distinguishes executable handlers from methods discussed as
candidates. A method is not implemented merely because it appears in a design
document or an LLM recommendation.

## Implemented in the current preprocessing registry

| Method | Status | Effect |
|---|---|---|
| Missingness profiling | Implemented | Reports per-column missing counts and fractions; does not fill values. |
| Numeric median imputation | Implemented, approval-gated | Replaces missing values in one numeric column with its median and archives replacement metadata. Disabled unless explicitly enabled. |

The current registry does not implement mean, mode, KNN, MICE, MissForest,
group-wise imputation, temporal fill, or deep-learning imputation.

## Candidate methods to evaluate

| Method | Suitable use case | Key safeguards |
|---|---|---|
| Mean imputation | Simple baseline for roughly symmetric numeric features | Distorts variance and associations; not a universal EHR default. |
| Median imputation | Simple baseline for skewed numeric features | Does not model patient, encounter, or temporal structure. |
| Mode imputation | Categorical baseline | May hide distinctions between unknown, not measured, and not applicable. |
| Explicit missing/unknown category | Categorical fields where the source semantics justify it | Keep unknown distinct from not applicable or not collected where possible. |
| Missingness indicator | Preserve information that a value was absent alongside an imputed value | Indicator does not reconstruct the missing measurement. |
| Group-wise imputation | Values plausibly comparable within patient, encounter, ward, or cohort | Require stable keys, minimum group sizes, and fallback policy. |
| Forward fill | Repeated time-ordered measurements within a patient/encounter | Enforce time order, maximum gap, and group boundaries; avoid filling across episodes. |
| Backward fill | Limited retrospective descriptive use where future observations are permitted | Can leak future information; prohibit for prospective prediction features. |
| Time interpolation | Continuous measurements with sufficiently dense and meaningful time series | Respect irregular intervals and episode boundaries; not suitable for all clinical variables. |
| KNN imputation | Numeric or mixed data with meaningful similarity | Scale features appropriately; assess compute cost and subgroup effects. |
| Iterative imputation / MICE | Related variables with defensible conditional models | Validate assumptions; multiple imputation should preserve multiple completed datasets and uncertainty. |
| MissForest | Nonlinear relationships in mixed-type tabular data | Computationally expensive; validate distributions and downstream performance. |
| Predictive/model-based imputation | A variable has validated predictors for a defined task | Can encode bias and understate uncertainty; train/evaluate without leakage. |
| Time-series/state-space imputation | Repeated measurements with a suitable temporal model | Validate per variable and patient episode; do not assume every absent value should be reconstructed. |
| Deep-learning imputation | Large datasets with adequate compute and a validated model | Not implemented in the current registry; requires uncertainty and subgroup evaluation. |
| Leave missing / task-specific exclusion | Missingness should remain explicit or a field is not usable for a defined analysis | Assess selection bias and document why exclusion is appropriate. |

## Recommended selection sequence for healthcare data

1. Identify the field's semantic role: identifier, timestamp, categorical,
   numeric measurement, medication/order field, or derived variable.
2. Profile missingness by table, column, patient/encounter, time, and relevant
   subgroups where data volume and privacy rules permit.
3. Establish whether the absence represents not measured, unknown, not
   applicable, or a likely data-quality defect. Do not infer this solely from
   a null value.
4. Prefer no mutation until the strategy is justified. For numeric columns,
   median is only a baseline candidate; for categorical columns, mode is not
   currently implemented and should not be silently substituted.
5. For longitudinal data, enforce patient/encounter boundaries and event-time
   order. For predictive tasks, prohibit use of future observations.
6. Preserve the original missing mask, archive changes, compare before/after
   distributions and subgroup missingness, and evaluate downstream task impact.

Never impute patient IDs, encounter IDs, linkage keys, or timestamps using a
generic numeric strategy. A missing clinical measurement is not automatically
an error. These recommendations require validation on synthetic or
de-identified fixtures before any clinical-data use.
