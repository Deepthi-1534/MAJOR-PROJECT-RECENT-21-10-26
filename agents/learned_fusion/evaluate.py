"""
evaluate.py - compare fusion methods on a held-out candidate CSV (build it from the TEST split).

    python -m agents.learned_fusion.evaluate --data data/fusion/test.csv --model models/learned_fusion.joblib

Methods compared (candidate level, positive = candidate matches a ground-truth target):
  bayes_only   : P(H0) from the Bayesian cognitive layer
  weighted     : reliability-weighted fusion (agents/fusion, FUSION_METHOD = "weighted")
  simple       : the simple fusion (agents/fusion, FUSION_METHOD = "simple" - the pipeline default)
  spatial_only : the spatial agent's evidence on its own (a sanity baseline)
  learned      : trained meta-classifier

A candidate counts as detected when its score >= the method's detect threshold. Candidates sent to
review count as "not detected" in precision / recall.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..fusion.common import implausible_reason
from ..fusion.simple_fusion import SimpleFusionAgent
from .dataset import build_matrix, load_dataset, row_to_candidate
from .features import FEATURE_NAMES


def metrics(score, y, detect_t, clear_t):
    det = score >= detect_t
    clr = score <= clear_t
    rev = ~det & ~clr
    tp = int(np.sum(det & (y == 1)))
    fp = int(np.sum(det & (y == 0)))
    fn = int(np.sum(~det & (y == 1)))          # review counts as not detected here
    tn = int(np.sum(~det & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "avg_precision": round(float(average_precision_score(y, score)), 4),
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "review_rate": round(float(rev.mean()), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "missed_in_clear_band": int(np.sum(clr & (y == 1))),
    }


def learned_scores(bundle, rows, captures):
    """Score the csv with a saved model; context features are computed per capture, exactly as
    the fusion agent does at run time."""
    use_context = bundle.get("use_context", bundle["feature_names"] != FEATURE_NAMES)
    X, names = build_matrix(rows, use_context)
    if names != bundle["feature_names"]:
        raise SystemExit("Model features do not match this CSV. Retrain with the current code.")
    raw = bundle["model"].predict_proba(X)[:, 1]
    r = np.clip(raw, 1e-4, 1 - 1e-4)
    return bundle["calibrator"].predict_proba(np.log(r / (1 - r)).reshape(-1, 1))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="models/learned_fusion.joblib")
    ap.add_argument("--out", default="outputs/review1/fusion_eval.json")
    args = ap.parse_args()

    _, y, captures, _, rows, _ = load_dataset([args.data], use_context=False)
    print(f"Rows: {len(y)} | positives: {int(y.sum())} | captures: {len(set(captures))}")
    if y.sum() == 0 or y.sum() == len(y):
        raise SystemExit("Need both positive and negative candidates to evaluate.")

    bundle = joblib.load(args.model)
    learned = learned_scores(bundle, rows, captures)

    weighted = np.array([float(r["weighted_suspicion"]) for r in rows])
    bayes = np.array([float(r["bayes_h0"]) for r in rows])
    spatial = np.array([float(r["spatial_evidence"]) * float(r["spatial_present"]) for r in rows])
    simple_agent = SimpleFusionAgent()
    simple = np.array([simple_agent.score(row_to_candidate(r))[0] for r in rows])
    sc = simple_agent.simple

    # Every fusion method rejects boxes too large to be one object (agents/fusion/common.py),
    # so apply the same rule here: those candidates can never be detected.
    too_big = np.array([implausible_reason(row_to_candidate(r)) is not None for r in rows])
    print(f"Rejected by the size rule: {int(too_big.sum())} candidates "
          f"({int((too_big & (y == 1)).sum())} of them true targets)")
    bayes, weighted, simple, spatial, learned = (np.where(too_big, 0.0, s)
                                                 for s in (bayes, weighted, simple, spatial, learned))

    results = {
        "bayes_only": metrics(bayes, y, 0.6, 0.4),
        "weighted": metrics(weighted, y, 0.6, 0.4),
        "simple": metrics(simple, y, sc.detect_threshold, sc.clear_threshold),
        "spatial_only": metrics(spatial, y, 0.5, 0.25),
        "learned": metrics(learned, y, bundle["detect_threshold"], bundle["clear_threshold"]),
    }

    cols = ["roc_auc", "avg_precision", "precision", "recall", "f1", "review_rate", "missed_in_clear_band"]
    print(f"\n{'method':14s}" + "".join(f"{c:>22s}" for c in cols))
    for name, m in results.items():
        print(f"{name:14s}" + "".join(f"{m[c]:>22}" for c in cols))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
