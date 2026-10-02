# TrippinnI

> **Building Trustworthy Healthcare AI Through Intelligent Data Refinement**

TrippinnI is an AI-powered healthcare data quality framework designed to preprocess, validate, and refine Electronic Health Records (EHR) and Electronic Medical Records (EMR) before they are used for analytics, machine learning, or clinical decision support.

The framework combines deterministic validation rules, machine learning algorithms, and Large Language Models (LLMs) to identify and explain data quality issues while producing an overall quality assessment.

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
Dataset
   │
   ▼
Module 0
Dataset Loading
   │
   ▼
Module 1
Dataset Profiling
   │
   ▼
Module 2
Data Quality Assessment
│
├── Missing Detection
├── Duplicate Detection
├── Datatype Validation
├── Healthcare Rule Validation
└── Outlier Detection
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
Quality Report
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
