"""dataset.py - load data/fusion/*.csv into feature matrices (shared by train.py and evaluate.py)."""
import csv
from typing import Dict, List, Tuple

import numpy as np

from .features import FEATURE_NAMES, ALL_FEATURE_NAMES, add_context


def scene_of(capture: str) -> str:
    """Scene key of a capture path, e.g. 'MUDCAD-X/dataset/02_Summer/2022-08-17/rk1/b/1'.
    Captures of the same scene look almost identical, so cross-validation groups by scene
    (not by capture) to avoid near-duplicate frames landing on both sides of a fold."""
    parts = capture.replace("\\", "/").split("/")
    return "/".join(parts[:7])


def load_rows(paths) -> List[Dict]:
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def build_matrix(rows: List[Dict], use_context: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Feature matrix for the CSV rows; context features are computed per capture."""
    base = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows], dtype=float)
    if not use_context:
        return base, list(FEATURE_NAMES)

    out = np.zeros((len(rows), len(ALL_FEATURE_NAMES)))
    by_capture: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by_capture.setdefault(r["capture"], []).append(i)
    for idx in by_capture.values():
        boxes = [[float(rows[i]["x"]), float(rows[i]["y"]), float(rows[i]["w"]), float(rows[i]["h"])]
                 for i in idx]
        out[idx] = add_context(base[idx], boxes)
    return out, list(ALL_FEATURE_NAMES)


def load_dataset(paths, use_context: bool = True):
    """Returns (X, y, capture_ids, scene_ids, rows, feature_names)."""
    rows = load_rows(paths)
    X, names = build_matrix(rows, use_context)
    y = np.array([int(float(r["label"])) for r in rows])
    captures = np.array([r["capture"] for r in rows])
    scenes = np.array([scene_of(r["capture"]) for r in rows])
    return X, y, captures, scenes, rows, names


def row_to_candidate(r: Dict) -> Dict:
    """Rebuild the candidate dict the fusion agents read, from one CSV row (used to score the
    rule-based 'simple' / 'weighted' fusion exactly as the pipeline does)."""
    c = {"bbox": [float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])]}
    for a in ("spatial", "spectral", "feature"):
        c[f"{a}_status"] = "DETECTED" if float(r[f"{a}_present"]) > 0 else "NO_DETECTION"
        c[f"{a}_evidence"] = float(r[f"{a}_evidence"])
    c["support_count"] = float(r["support_count"])
    c["conflict"] = float(r["conflict"])
    c["uncertainty"] = float(r["uncertainty"])
    c["confidence"] = float(r["confidence"])
    c["bayes_posterior"] = {"H0": float(r["bayes_h0"]), "H1": float(r["bayes_h1"]), "H2": float(r["bayes_h2"])}
    return c
