"""
Feature Extractor.

Converts healthcare tables into numerical feature matrices
for machine learning algorithms.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureExtractor:
    """
    Extract numerical features for ML models.
    """

    def __init__(self) -> None:

        self.preprocessor = None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Learn preprocessing transformations.
        """

        numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        self.preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ])

        self.preprocessor.fit(df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform dataframe into feature matrix.
        """

        if self.preprocessor is None:
            raise RuntimeError(
                "FeatureExtractor must be fitted first."
            )

        return self.preprocessor.transform(df)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform.
        """

        self.fit(df)

        return self.transform(df)