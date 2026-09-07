import pandas as pd


class DatatypeProfiler:
    """Infer semantic types separately from physical validation types."""

    IDENTIFIER_KEYWORDS = {
        "id", "patient", "encounter", "organization", "provider", "payer", "device"
    }
    DATE_KEYWORDS = {"date", "time", "birth", "death", "start", "stop"}
    CODE_KEYWORDS = {"code", "icd", "snomed", "rxnorm", "loinc", "cpt"}
    BOOLEAN_VALUES = {"true", "false", "yes", "no", "y", "n", "0", "1"}

    def infer(self, column_name: str, series: pd.Series) -> str:
        name = column_name.lower()
        if self._contains_keyword(name, self.IDENTIFIER_KEYWORDS):
            return "identifier"
        if self._contains_keyword(name, self.DATE_KEYWORDS):
            return "datetime"
        if self._contains_keyword(name, self.CODE_KEYWORDS):
            return "medical_code"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if self._is_boolean(series):
            return "boolean"
        if self._is_datetime(series):
            return "datetime"
        if self._is_categorical(series):
            return "categorical"
        return "text"

    def infer_streaming(
        self,
        column_name: str,
        pandas_dtype: str,
        sample: pd.Series,
        non_null_count: int,
        unique_count: int,
    ) -> str:
        name = column_name.lower()
        if self._contains_keyword(name, self.IDENTIFIER_KEYWORDS):
            return "identifier"
        if self._contains_keyword(name, self.DATE_KEYWORDS):
            return "datetime"
        if self._contains_keyword(name, self.CODE_KEYWORDS):
            return "medical_code"
        if pd.api.types.is_bool_dtype(pandas_dtype):
            return "boolean"
        if pd.api.types.is_numeric_dtype(pandas_dtype):
            return "numeric"
        if self._is_boolean(sample):
            return "boolean"
        if self._is_datetime(sample):
            return "datetime"
        if unique_count / max(non_null_count, 1) < 0.20:
            return "categorical"
        return "text"

    def validation_type(self, pandas_dtype: str, semantic_type: str) -> str:
        """Return the physical type expected by value-level validation."""
        dtype = str(pandas_dtype).lower()
        if semantic_type == "datetime":
            return "datetime"
        if semantic_type == "boolean":
            return "boolean"
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        if pd.api.types.is_float_dtype(dtype):
            return "float"
        if pd.api.types.is_numeric_dtype(dtype):
            return "float"
        if semantic_type in {"identifier", "medical_code", "categorical", "text"}:
            return "string"
        return "string"

    @staticmethod
    def _contains_keyword(name, keywords):
        return any(keyword in name for keyword in keywords)

    def _is_boolean(self, series):
        values = series.dropna().astype(str).str.lower().unique()
        return len(values) > 0 and set(values).issubset(self.BOOLEAN_VALUES)

    def _is_datetime(self, series):
        sample = series.dropna()
        if len(sample) == 0:
            return False
        if len(sample) > 100:
            sample = sample.sample(100, random_state=42)
        try:
            pd.to_datetime(sample, errors="raise", format="mixed")
            return True
        except (TypeError, ValueError):
            return False

    def _is_categorical(self, series):
        if pd.api.types.is_numeric_dtype(series):
            return False
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        return unique_ratio < 0.20
