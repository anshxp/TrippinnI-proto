import re
import pandas as pd


class DatatypeProfiler:
    """
    Infers semantic data types.

    This is NOT pandas dtype detection.

    This tries to understand what a column represents.
    """

    IDENTIFIER_KEYWORDS = {
        "id",
        "patient",
        "encounter",
        "organization",
        "provider",
        "payer",
        "device"
    }

    DATE_KEYWORDS = {
        "date",
        "time",
        "birth",
        "death",
        "start",
        "stop"
    }

    CODE_KEYWORDS = {
        "code",
        "icd",
        "snomed",
        "rxnorm",
        "loinc",
        "cpt"
    }

    BOOLEAN_VALUES = {
        "true",
        "false",
        "yes",
        "no",
        "y",
        "n",
        "0",
        "1"
    }

    def infer(self, column_name: str, series: pd.Series) -> str:

        name = column_name.lower()

        # -------------------------------------------------

        if self._contains_keyword(name, self.IDENTIFIER_KEYWORDS):
            return "identifier"

        # -------------------------------------------------

        if self._contains_keyword(name, self.DATE_KEYWORDS):
            return "datetime"

        # -------------------------------------------------

        if self._contains_keyword(name, self.CODE_KEYWORDS):
            return "medical_code"

        # -------------------------------------------------

        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        # -------------------------------------------------

        if self._is_boolean(series):
            return "boolean"

        # -------------------------------------------------

        if self._is_datetime(series):
            return "datetime"

        # -------------------------------------------------

        if self._is_categorical(series):
            return "categorical"

        # -------------------------------------------------

        return "text"

    def infer_streaming(
        self,
        column_name: str,
        pandas_dtype: str,
        sample: pd.Series,
        non_null_count: int,
        unique_count: int,
    ) -> str:
        """Infer a semantic type from bounded streaming state."""

        name = column_name.lower()
        if self._contains_keyword(name, self.IDENTIFIER_KEYWORDS):
            return "identifier"
        if self._contains_keyword(name, self.DATE_KEYWORDS):
            return "datetime"
        if self._contains_keyword(name, self.CODE_KEYWORDS):
            return "medical_code"
        if pd.api.types.is_numeric_dtype(pandas_dtype):
            return "numeric"
        if self._is_boolean(sample):
            return "boolean"
        if self._is_datetime(sample):
            return "datetime"
        if unique_count / max(non_null_count, 1) < 0.20:
            return "categorical"
        return "text"

    ################################################################

    def _contains_keyword(self, name, keywords):

        for keyword in keywords:
            if keyword in name:
                return True

        return False

    ################################################################

    def _is_boolean(self, series):

        values = (
            series.dropna()
                  .astype(str)
                  .str.lower()
                  .unique()
        )

        return set(values).issubset(self.BOOLEAN_VALUES)

    ################################################################

    def _is_datetime(self, series):

        sample = series.dropna()

        if len(sample) > 100:
            sample = sample.sample(100, random_state=42)

        try:

            pd.to_datetime(
                sample,
                errors="raise"
            )

            return True

        except Exception:

            return False

    ################################################################

    def _is_categorical(self, series):

        if pd.api.types.is_numeric_dtype(series):
            return False

        unique_ratio = (
            series.nunique(dropna=True)
            /
            max(len(series), 1)
        )

        return unique_ratio < 0.20
