"""features.py - turn one pipeline candidate (output of ALL earlier layers) into a numeric feature vector."""
import math
from typing import Dict, List

from ..fusion.common import clip01, to_float

AGENTS = ["spatial", "spectral", "feature"]
CONFLICT_LEVEL = {"LOW": 0.0, "MODERATE": 1.0, "HIGH": 2.0, "UNKNOWN": 1.0}
RELIABILITY_LEVEL = {"LOW": 0.0, "MEDIUM": 1.0, "HIGH": 2.0}

FEATURE_NAMES: List[str] = [
    # Spatial / Spectral / Feature layers
    "spatial_present", "spatial_evidence",
    "spectral_present", "spectral_evidence",
    "feature_present", "feature_evidence",
    "n_present", "mean_evidence", "max_evidence", "min_evidence", "evidence_spread",
    # Conflict layer
    "conflict", "support_count", "agreement_ratio", "conflict_level",
    # Uncertainty layer
    "uncertainty", "confidence", "reliability_level",
    # Cognitive layer (Bayesian)
    "bayes_h0", "bayes_h1", "bayes_h2", "bayes_review", "bayes_n_flags",
    # Candidate geometry
    "log_area", "aspect_ratio",
]


def candidate_to_features(c: Dict) -> List[float]:
    present, evidence = {}, {}
    for name in AGENTS:
        detected = str(c.get(f"{name}_status", "")).upper() in ("DETECTED", "EVIDENCE_PRESENT")
        present[name] = 1.0 if detected else 0.0
        evidence[name] = clip01(to_float(c.get(f"{name}_evidence"))) if detected else 0.0

    used = [evidence[n] for n in AGENTS if present[n] > 0]
    n_present = float(len(used))
    mean_e = sum(used) / len(used) if used else 0.0
    max_e = max(used) if used else 0.0
    min_e = min(used) if used else 0.0

    if "support_count" in c:
        support = to_float(c["support_count"])
    else:
        support = float(len(c.get("supporting_agents") or []))

    post = c.get("bayes_posterior") or {}
    flags = c.get("bayes_flags") or []

    bbox = c.get("bbox") or [0, 0, 1, 1]
    w, h = max(to_float(bbox[2], 1.0), 1.0), max(to_float(bbox[3], 1.0), 1.0)

    row = {
        "spatial_present": present["spatial"], "spatial_evidence": evidence["spatial"],
        "spectral_present": present["spectral"], "spectral_evidence": evidence["spectral"],
        "feature_present": present["feature"], "feature_evidence": evidence["feature"],
        "n_present": n_present, "mean_evidence": mean_e, "max_evidence": max_e,
        "min_evidence": min_e, "evidence_spread": max_e - min_e,
        "conflict": clip01(to_float(c.get("conflict"), 1.0)),
        "support_count": support,
        "agreement_ratio": clip01(support / len(AGENTS)),
        "conflict_level": CONFLICT_LEVEL.get(str(c.get("conflict_state", "UNKNOWN")).upper(), 1.0),
        "uncertainty": clip01(to_float(c.get("uncertainty"), 1.0)),
        "confidence": clip01(to_float(c.get("confidence"), 0.0)),
        "reliability_level": RELIABILITY_LEVEL.get(str(c.get("reliability", "LOW")).upper(), 0.0),
        "bayes_h0": clip01(to_float(post.get("H0"), 1 / 3)),
        "bayes_h1": clip01(to_float(post.get("H1"), 1 / 3)),
        "bayes_h2": clip01(to_float(post.get("H2"), 1 / 3)),
        "bayes_review": 1.0 if str(c.get("bayes_decision", "")).startswith("UNCERTAIN") else 0.0,
        "bayes_n_flags": float(len(flags)),
        "log_area": math.log(w * h),
        "aspect_ratio": w / h,
    }
    return [float(row[name]) for name in FEATURE_NAMES]


# --------------------------------------------------------------------------------------
# Capture-context features
#
# A candidate is much easier to judge relative to the OTHER candidates found in the same
# image: "is this the most confident spatial detection in the frame?", "is it much larger
# than the rest?". These are computed from the base features above plus the box, so existing
# data/fusion/*.csv files (which already hold those columns) work unchanged.
# --------------------------------------------------

IMAGE_W = 512.0          # MUDCAD-X frames are 512 x 512 (boxes never exceed this)
IMAGE_H = 512.0

_CONTEXT_SOURCES = ["spatial_evidence", "spectral_evidence", "feature_evidence",
                    "mean_evidence", "max_evidence", "log_area"]
CONTEXT_NAMES: List[str] = (
    [f"{s}_{kind}" for s in _CONTEXT_SOURCES for kind in ("rank", "rel_max", "rel_mean")]
    + ["n_cands", "n_spatial_in_cap", "cx", "cy", "area_frac", "touch_border",
       "sp_x_support", "sp_x_unc", "evid_minus_conf"]
)
ALL_FEATURE_NAMES: List[str] = FEATURE_NAMES + CONTEXT_NAMES


def _rank_pct(v):
    """Average rank / n (ties share a rank) - same as pandas rank(pct=True)."""
    import numpy as np
    from scipy.stats import rankdata
    return rankdata(v, method="average") / len(v)


def add_context(base, boxes):
    """
    base  : array-like (n, len(FEATURE_NAMES)) - base features of ALL candidates of ONE image
    boxes : list of n boxes [x, y, w, h] (None allowed)
    Returns an (n, len(ALL_FEATURE_NAMES)) numpy array.
    """
    import numpy as np
    base = np.asarray(base, dtype=float).reshape(-1, len(FEATURE_NAMES))
    n = len(base)
    if n == 0:
        return base.reshape(0, len(ALL_FEATURE_NAMES))
    col = {name: base[:, i] for i, name in enumerate(FEATURE_NAMES)}

    cols = []
    for s in _CONTEXT_SOURCES:
        v = col[s]
        cols += [_rank_pct(v), v - v.max(), v - v.mean()]
    cols.append(np.full(n, float(n)))
    cols.append(np.full(n, float(col["spatial_present"].sum())))

    b = np.array([[to_float(x) for x in (bx if bx is not None and len(bx) >= 4 else [0, 0, 1, 1])[:4]]
                  for bx in boxes], dtype=float).reshape(n, 4)
    x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    cols.append((x + w / 2) / IMAGE_W)
    cols.append((y + h / 2) / IMAGE_H)
    cols.append((w * h) / (IMAGE_W * IMAGE_H))
    cols.append(((x <= 1) | (y <= 1) | (x + w >= IMAGE_W - 1) | (y + h >= IMAGE_H - 1)).astype(float))
    cols.append(col["spatial_evidence"] * col["support_count"])
    cols.append(col["spatial_evidence"] * (1.0 - col["uncertainty"]))
    cols.append(col["mean_evidence"] - col["conflict"])
    return np.hstack([base, np.column_stack(cols)])


def candidates_to_matrix(candidates, use_context: bool = True):
    """Feature matrix for all candidates of ONE image (what the fusion agent sees at run time)."""
    import numpy as np
    base = np.array([candidate_to_features(c) for c in candidates], dtype=float)
    base = base.reshape(len(candidates), len(FEATURE_NAMES))
    if not use_context:
        return base
    return add_context(base, [c.get("bbox") for c in candidates])
