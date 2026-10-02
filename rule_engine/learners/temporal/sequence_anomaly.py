"""Sequence transition anomaly scoring."""
def anomaly_probability(transition_probability: float,floor: float=1e-6) -> float:
    return float(1.0-max(floor,min(1.0,transition_probability)))
