# TrippinnI

> **Building Trustworthy Healthcare AI Through Intelligent Data Refinement**

TrippinnI is an AI-powered healthcare data quality framework designed to preprocess, validate, and refine Electronic Health Records (EHR) and Electronic Medical Records (EMR) before they are used for analytics, machine learning, or clinical decision support.

The framework combines healthcare-aware profiling, semantic context inference, a lightweight knowledge graph, deterministic validation rules, machine learning algorithms, and explainability to identify and contextualize data quality issues while producing an overall quality assessment. MIMIC-IV is used as a validation dataset for the prototype; product logic is designed to remain dataset-agnostic across EHR/EMR and other healthcare data sources.

---

## Features

- Automated healthcare dataset profiling
- Missing value detection
- Duplicate detection
  - Exact matching
  - Composite key matching
  - Fuzzy text matching (RapidFuzz) with configurable similarity threshold
  - Semantic matching (Sentence-BERT) — planned, not currently implemented
- Datatype validation
- Outlier detection
  - Robust IQR rules
  - Isolation Forest
  - COPOD (PyOD)
  - MLP autoencoder reconstruction
- Machine-learning-assisted healthcare constraint discovery
  - Isolation Forest learns temporal ordering from timestamp-pair distributions
  - Isolation Forest learns identifier dependency patterns
  - Isolation Forest learns empirical numeric support regions
  - Binary domains are inferred from observed value support
  - Every learned constraint records model provenance and confidence
- Healthcare rule validation
  - Required structural fields remain explicit schema facts
  - Temporal ordering checks use learned constraints
  - Identifier hierarchy checks use learned relational constraints
  - Identifier format/conformance checks
  - Empirical plausibility checks (not clinical reference ranges)
- Confidence aggregation
  - Weighted Voting
  - Bias–Variance
  - Dempster–Shafer
  - Meta-Classifier (Extensible)
- Overall data quality scoring
- AI-generated explanations using Hugging Face LLMs
- Comprehensive quality reporting
- Advanced EHR quality assessment
  - Bias / representativeness profiling
  - Group-wise data fairness checks
  - Interoperability/conformance signals for semantic aliases and mixed units
  - Temporal data-drift detection
  - Explicit reference-vs-current distribution-shift detection
  - Bootstrap robustness/sensitivity assessment

---

# Architecture

```
Healthcare Dataset
   │
   ▼
Module 0 — Loading
   │
   ▼
Module 1 — Healthcare Data Profiling
   │
   ▼
Semantic Context Layer
   │
   ├── Entity/field roles
   ├── Temporal context
   ├── Clinical-field hints
   ├── Candidate keys
   └── Candidate relationships
   │
   ▼
Healthcare Knowledge Graph
   │
   ├── Tables / fields
   ├── Semantic roles
   ├── Candidate identifiers
   ├── Temporal attributes
   └── Cross-table relationship candidates
   │
   ▼
Module 2 — Context-Aware Quality Detection
   │
   ├── Missingness
   ├── Duplicate / identity
   ├── Datatype / conformance
   ├── Rule validation
   ├── Outliers
   └── Cross-table structural context
   │
   ▼
Confidence Aggregation
   │
   ▼
Quality Score
   │
   ▼
Explainability
   │
   ▼
Remediation → Revalidation → AI Readiness
```

---

# Project Structure

```
TrippinnI/

├── app.py
├── config.py
├── requirements.txt

├── loaders/
├── profiling/
├── detectors/
├── features/
├── ml/
│   ├── isolation_forest.py
│   ├── copod_detector.py
│   ├── autoencoder.py
│   └── llm/
├── quality/
├── rule_engine/
├── explainability/
├── outputs/
├── preprocessing/
├── recommendation/
├── models/
├── utils/
└── data/
```

---

# Detection Pipeline

```
Dataset
   │
   ▼
Missing Detector
   │
   ▼
Duplicate Detector
   │
   ▼
Datatype Detector
   │
   ▼
Healthcare Rule Validator
   │
   ▼
Outlier Detector
   │
   ▼
Confidence Aggregation
   │
   ▼
Quality Score
   │
   ▼
LLM Explainability
   │
   ▼
Final Report
```

---

# Outlier Detection

The outlier detection module combines deterministic and machine learning approaches.

- Rule-Based Validation
- Isolation Forest
- COPOD
- Autoencoder

---

# Duplicate Detection

Duplicate detection currently separates confirmed structural duplicates from approximate text similarity.

- Exact row matching — confirmed duplicate records when every field matches
- Candidate-key uniqueness checks — repeated values in profiler-identified candidate keys
- Fuzzy text matching with RapidFuzz — probable duplicates based on configurable string similarity
- Sentence-BERT semantic matching — planned; not currently implemented

RapidFuzz matching excludes identifier columns, ignores very short strings, operates on unique normalized text values, and records the similarity score and threshold in each issue. Fuzzy findings are MEDIUM severity and should be reviewed rather than automatically deleted.

---

# Confidence Aggregation

The framework supports multiple confidence aggregation strategies.

- Weighted Voting
- Bias–Variance Combination
- Dempster–Shafer Evidence Theory
- Meta-Classifier (Extensible)

---

# Technology Stack

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-Learn
- PyOD
- TensorFlow

## NLP

- Transformers
- Sentence Transformers
- RapidFuzz

## Explainability

- Hugging Face Transformers

---

# Installation

```bash
git clone <repository_url>

cd TrippinnI

pip install -r requirements.txt
```

---

# Run

```bash
python app.py
```

---

# Current Modules

## Module 0

- Dataset Loading
- Schema Extraction
- Dataset Management

## Module 1

- Dataset Profiling
- Metadata Extraction
- Statistical Profiling

## Module 2

- Missing Detection
- Duplicate Detection
- Datatype Validation
- Outlier Detection
- Confidence Aggregation
- Quality Scoring
- LLM Explainability
- Report Generation

---

# Future Enhancements

- Additional healthcare validation rules
- Knowledge graph integration
- Real-time EHR preprocessing
- FHIR interoperability
- Dashboard for quality monitoring
- Distributed processing for large healthcare datasets

---

## License

MIT License
## Module 2 implementation status

The data-quality stage now runs five detector families: missingness, exact/candidate-key duplicates, datatype conformance, machine-learning-assisted healthcare constraint validation, and statistical/ML outlier detection. Constraint discovery is dataset-driven rather than based on MIMIC field-name allowlists. Learned temporal, relational, and numeric constraints retain model provenance and confidence.

The detector stage is flag-only: it does not impute, delete, or overwrite source data. Cross-table referential joins, remediation, revalidation, and downstream AI-readiness assessment remain separate stages because they require additional context beyond a single table/sample.


<!-- Modular ML constraint engine documentation verified against repository layout. -->


## Healthcare semantic context and knowledge graph

The prototype now builds a dataset-agnostic semantic context from profiling metadata and a lightweight knowledge graph before the final pipeline report is written.

The context layer deliberately uses conservative inference. It can identify candidate roles such as entity identifier, event time, clinical attribute, and measurement unit from observed schema/profile metadata. These are hypotheses with confidence, not hard-coded claims about a specific source system.

The knowledge graph represents:
- table and field nodes
- semantic field roles
- candidate primary/foreign-key relationships
- shared identifier relationship candidates
- temporal and clinical semantic roles
- pluggable terminology resolution interface

Detected issues are enriched with graph context so downstream reports can explain why a field is being evaluated in a particular semantic role.

Cross-table analysis now has two layers: structural relationship discovery and value-level referential-integrity validation. Candidate FK -> PK relationships are validated against the bounded synchronized detection samples, reporting orphan child rows/unique values, missing parent records, parent-key uniqueness, referential coverage, and FK -> PK validity. These findings are explicitly sample-scoped; they are not full-table guarantees.

The generated context artifact is written to outputs/reports/context/healthcare_context.json.

MIMIC-IV remains a validation dataset, not the product's schema contract. Future terminology adapters can map inferred clinical concepts to standards such as FHIR, OMOP, SNOMED CT, LOINC, RxNorm, CPT/HCPCS, or UCUM without making those standards mandatory for every dataset.


## Advanced EHR quality dimensions

The prototype now includes an AdvancedQualityDetector. It reports observable data-quality signals with explicit thresholds and metadata rather than asking an LLM to infer protected-group fairness or clinical truth.

### Bias / representativeness
Candidate demographic columns are identified conservatively from observed field names. Groups meeting the minimum sample-size threshold but falling below the configured representation threshold are flagged. This is a representation warning, not a claim that the population itself should be balanced.

### Fairness
Group-wise missingness is compared for candidate demographic attributes. A configured missingness-rate gap threshold produces a fairness warning. This currently measures data-quality disparity, not downstream model fairness.

### Interoperability / conformance
The detector identifies known semantic aliases, such as heart_rate, pulse, and hr, and detects mixed units in unit/UOM columns. These are normalization signals; they are not a complete FHIR/OMOP terminology validator.

### Data drift
When a usable time field exists, observations are split chronologically and numeric distributions are compared using standardized quantile distance; categorical distributions use total variation distance. Without a usable time field, row-order splitting is used and explicitly recorded in result metadata.

### Distribution shift
Deployment/reference comparison is available through QualityDetector.run(dataset, profile, reference_dataset=...). Numeric columns use standardized quantile distance and categorical columns use total variation distance. This requires an explicit reference dataset; the prototype does not pretend that a single dataset is a training/deployment comparison.

### Robustness
The detector performs deterministic bootstrap resampling of the evaluated sample and measures sensitivity of the overall missingness rate. The result reports bootstrap mean, standard deviation, and range. This is a data-quality stability signal, not a full causal or model-performance robustness analysis.

All thresholds are configurable in config.py. These dimensions are evaluated on the bounded detection sample when the normal MIMIC-IV pipeline is used; they should therefore be interpreted as sample-level evidence rather than exact full-table population estimates.
