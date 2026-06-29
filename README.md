# cleaning_automation
# AI-Powered Healthcare Data Refinery (Prototype)

## Overview

This project aims to build an **AI-powered automated healthcare data preprocessing system** capable of transforming raw Electronic Health Record (EHR) datasets into AI-ready datasets with minimal human intervention.

The current prototype demonstrates a modular preprocessing architecture where different preprocessing strategies are combined into candidate pipelines.

---

## Problem Statement

Healthcare datasets are often:

* Incomplete
* Inconsistent
* Contain missing values
* Contain outliers
* Require extensive preprocessing before Machine Learning

Data scientists spend a significant amount of time manually cleaning healthcare datasets before model development.

This project aims to automate this preprocessing workflow.

---

## Current Prototype Features

### Step 1 – Dataset Understanding

* Dataset loading
* Dataset inspection
* Data type identification
* Shape analysis

---

### Step 2 – Dataset Profiling

* Missing value analysis
* Duplicate detection
* Data type profiling
* Statistical summary generation

---

### Step 3 – Problem Detection

* Identification of preprocessing issues
* Missing column detection
* Numerical vs categorical feature separation

---

### Step 4 – Missing Value Engine

Implemented strategies:

* Mode Imputation
* Constant Imputation

Implemented using:

* SimpleImputer

---

### Step 5 – Outlier Engine

Implemented strategies:

* Keep Outliers
* IQR-based Outlier Detection

Isolation Forest implementation is under development.

---

### Step 6 – Scaling Engine

Implemented strategies:

* StandardScaler
* RobustScaler

---

### Step 7

Reserved for future Feature Engineering and Feature Selection modules.

---

### Step 8 – Pipeline Generator

Current pipeline generation:

Pipeline 1

* Mode Imputation
* Keep Outliers
* StandardScaler

Pipeline 2

* Mode Imputation
* Keep Outliers
* RobustScaler

The preprocessing system automatically executes candidate preprocessing pipelines to generate AI-ready datasets.

---

## Current Project Architecture

Raw Dataset

↓

Dataset Profiling

↓

Problem Detection

↓

Missing Value Engine

↓

Outlier Engine

↓

Scaling Engine

↓

Pipeline Generator

↓

AI-ready Dataset

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

---

## Current Status

Prototype Development (Step 8 Completed)

Upcoming modules:

* Pipeline Evaluation Engine
* Pipeline Ranking
* Automated Pipeline Selection
* Report Generation
* AI-ready Dataset Export

---

## Repository Structure

```
data/
notebooks/
outputs/
reports/
.gitignore
requirements.txt
README.md
```

---

## Future Scope

* Automatic preprocessing recommendation
* Multiple preprocessing pipeline generation
* Automated pipeline evaluation
* Intelligent pipeline selection
* AI-ready healthcare dataset generation
* Integration with Machine Learning workflows

---

## Authors

Healthcare Data Refinery Prototype

Research & Development Project


##step 9--> Dataset Feature Extraction Engine

Implemented:

- Dataset feature extraction
- Missing value percentage calculation
- Numeric & categorical feature identification
- Dataset metadata generation