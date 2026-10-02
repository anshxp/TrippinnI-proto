"""Confidence calibration from support and learner agreement."""
def confidence(support: float,agreement: float,base: float=0.50)->float:
    support=min(1.0,max(0.0,support)); agreement=min(1.0,max(0.0,agreement))
    return min(0.995,base+0.25*support+0.25*agreement)
