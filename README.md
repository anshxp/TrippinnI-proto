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
  - Fuzzy matching (RapidFuzz)
  - Semantic matching (Sentence-BERT)
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

Duplicate detection is performed using multiple strategies.

- Exact Matching
- Composite Keys
- RapidFuzz
- Sentence-BERT

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

Cross-table analysis currently reports structural relationship candidates. It does not claim value-level referential integrity without synchronized multi-table data. That distinction is intentional and keeps the prototype honest across arbitrary healthcare datasets.

The generated context artifact is written to outputs/reports/context/healthcare_context.json.

MIMIC-IV remains a validation dataset, not the product's schema contract. Future terminology adapters can map inferred clinical concepts to standards such as FHIR, OMOP, SNOMED CT, LOINC, RxNorm, CPT/HCPCS, or UCUM without making those standards mandatory for every dataset.
