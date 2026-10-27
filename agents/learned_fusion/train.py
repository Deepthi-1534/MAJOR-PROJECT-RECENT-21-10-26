"""
train.py - train the learned fusion (stacked meta-classifier) on candidate rows.

    python -m agents.learned_fusion.train --data data/fusion/val.csv
Options: --algo auto|lr|gb|rf|et|hgb|ens   --folds 5   --beta 1.0   --out models/learned_fusion.joblib

What it does
  * features  : the base per-candidate features PLUS capture-context features (how a candidate
                ranks against the other candidates of the same image). Use --no-context for the
                old feature set.
  * validation: cross-validation grouped by SCENE, so near-duplicate frames of one scene never sit
                in both the training and validation folds.
  * calibration: probabilities are Platt-calibrated on out-of-fold scores.
  * thresholds: "Detected" = best F-beta on the calibrated out-of-fold scores (beta > 1 favours
                recall, beta < 1 favours precision). "Not detected" = the score below which only
                --max-missed of the true targets fall.

Train on candidates from the VAL split (the detector was trained on TRAIN, so its scores on TRAIN
images are over-confident) and keep the TEST csv for evaluate.py only. Build the val csv from ALL
val captures:  python -m agents.learned_fusion.build_dataset --split val   (do not pass --every).
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import load_dataset


def _et():
    return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=42)


def _rf():
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42)


def _hgb():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, random_state=42)


def make_model(algo):
    if algo == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=5000, C=0.3))
    if algo == "gb":
        return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, random_state=42)
    if algo == "rf":
        return _rf()
    if algo == "et":
        return _et()
    if algo == "hgb":
        return _hgb()
    if algo == "ens":      # average of three different tree ensembles - the most stable choice
        return VotingClassifier([("et", _et()), ("rf", _rf()), ("hgb", _hgb())], voting="soft")
    raise ValueError(algo)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def best_fbeta_threshold(p, y, beta=1.0):
    best_t, best_f = 0.5, -1.0
    b2 = beta * beta
    for t in np.arange(0.02, 0.98, 0.01):
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        denom = (1 + b2) * tp + b2 * fn + fp
        f = (1 + b2) * tp / denom if denom else 0.0
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="models/learned_fusion.joblib")
    ap.add_argument("--algo", default="auto", choices=["auto", "lr", "gb", "rf", "et", "hgb", "ens"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="F-beta used to pick the detect threshold (2.0 = recall matters more than precision)")
    ap.add_argument("--max-missed", type=float, default=0.05,
                    help="share of true camouflage allowed below the 'not detected' threshold")
    ap.add_argument("--no-context", action="store_true", help="use only the base features")
    args = ap.parse_args()

    use_context = not args.no_context
    X, y, captures, scenes, _, names = load_dataset(args.data, use_context)
    print(f"Rows: {len(y)} | positives: {int(y.sum())} | negatives: {int((1 - y).sum())} "
          f"| captures: {len(set(captures))} | scenes: {len(set(scenes))} | features: {len(names)}")
    if y.sum() == 0 or y.sum() == len(y):
        raise SystemExit("Need both positive and negative candidates to train.")
    if y.sum() < 30:
        print(f"WARNING: only {int(y.sum())} positive examples - the model will be weak. "
              "Build the csv from ALL val captures (no --every).")

    n_splits = max(2, min(args.folds, len(set(scenes))))
    cv = list(GroupKFold(n_splits=n_splits).split(X, y, scenes))
    algos = ["lr", "hgb", "ens"] if args.algo == "auto" else [args.algo]
    results = {}
    for a in algos:
        oof = cross_val_predict(make_model(a), X, y, cv=cv, method="predict_proba")[:, 1]
        results[a] = {"oof": oof, "auc": roc_auc_score(y, oof), "ap": average_precision_score(y, oof)}
        print(f"  {a:4s}: scene-grouped CV ROC-AUC {results[a]['auc']:.3f} | average precision {results[a]['ap']:.3f}")
    algo = max(results, key=lambda k: results[k]["ap"])
    print(f"Selected: {algo}")

    # Platt calibration on out-of-fold scores
    oof = results[algo]["oof"]
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof), y)
    cal_oof = calibrator.predict_proba(logit(oof))[:, 1]
    detect_t, f = best_fbeta_threshold(cal_oof, y, args.beta)
    pos_scores = cal_oof[y == 1]
    clear_t = float(np.quantile(pos_scores, args.max_missed))
    clear_t = max(0.02, min(clear_t, detect_t - 0.05))

    pred = cal_oof >= detect_t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"Thresholds: detect >= {detect_t:.2f} | not detected <= {clear_t:.2f}")
    print(f"Out-of-fold at the detect threshold: precision {precision:.3f} | recall {recall:.3f} | F1 {f1:.3f}")

    model = make_model(algo).fit(X, y)
    if algo == "lr":
        coefs = model[-1].coef_[0]
        order = np.argsort(-np.abs(coefs))[:10]
        print("Top features (|scaled coefficient|):")
        for i in order:
            print(f"   {names[i]:28s} {coefs[i]:+.3f}")
    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        order = np.argsort(-imp)[:10]
        print("Top features (importance):")
        for i in order:
            print(f"   {names[i]:28s} {imp[i]:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "calibrator": calibrator, "algo": algo,
        "feature_names": names, "use_context": use_context,
        "detect_threshold": detect_t, "clear_threshold": clear_t,
        "cv": {"roc_auc": results[algo]["auc"], "average_precision": results[algo]["ap"],
               "precision": precision, "recall": recall, "f1": f1, "beta": args.beta,
               "grouping": "scene"},
        "n_rows": int(len(y)), "n_positive": int(y.sum()),
    }, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
