"""
Healthcare validation configuration used by the Module 2 rule validator.
"""

from __future__ import annotations

MIMIC_REQUIRED_FIELDS = {
    "hosp_admissions": ("subject_id", "hadm_id", "admittime", "dischtime"),
    "hosp_transfers": ("subject_id", "hadm_id", "transfer_id"),
    "hosp_labevents": ("labevent_id", "subject_id", "itemid"),
    "hosp_microbiologyevents": ("microevent_id", "subject_id"),
    "icu_icustays": ("subject_id", "hadm_id", "stay_id", "intime", "outtime"),
    "icu_chartevents": ("subject_id", "stay_id", "itemid", "charttime"),
    "icu_inputevents": ("subject_id", "stay_id", "itemid", "starttime", "endtime"),
    "icu_outputevents": ("subject_id", "stay_id", "itemid", "charttime"),
    "icu_procedureevents": ("subject_id", "stay_id", "itemid", "starttime", "endtime"),
}

DATE_ORDER_PAIRS = {
    "hosp_admissions": (("admittime", "dischtime"), ("edregtime", "edouttime")),
    "hosp_transfers": (("intime", "outtime"),),
    "icu_icustays": (("intime", "outtime"),),
    "icu_inputevents": (("starttime", "endtime"),),
    "icu_procedureevents": (("starttime", "endtime"),),
    "hosp_prescriptions": (("starttime", "stoptime"),),
}

# Generic ranges are deliberately limited to fields whose meaning is stable
# from the column name. MIMIC generic valuenum is not checked here because
# it represents many different clinical measurements.
NUMERIC_RANGES = {
    "anchor_age": (0.0, 120.0),
    "los": (0.0, None),
    "hospital_expire_flag": (0.0, 1.0),
}

IDENTIFIER_HIERARCHY = (
    ("hadm_id", ("subject_id",)),
    ("stay_id", ("subject_id", "hadm_id")),
)
