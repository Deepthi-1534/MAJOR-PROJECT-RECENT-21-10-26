"""
calibrate.py - fit the cognitive (Bayesian) layer's models from labelled fusion candidates.

    python -m agents.cognitive.calibrate --data data/fusion/val.csv --out models/cognitive_models.json

Use the VAL candidates (never the test csv). Once models/cognitive_models.json exists the cognitive
agent loads it automatically; delete the file to go back to the built-in assumptions.

IMPORTANT: the learned fusion uses the cognitive layer's output (bayes_h0 ...) as input features.
After calibrating, rebuild data/fusion/val.csv and data/fusion/test.csv and retrain the learned fusion.
"""
import argparse
import csv
import json
from pathlib import Path

from .calibration import fit_from_candidates
from .common import CALIBRATED_MODELS_PATH, models_to_json, PRESENCE_KEY, HYPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default=str(CALIBRATED_MODELS_PATH))
    args = ap.parse_args()

    rows = []
    for p in args.data:
        with open(p, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    n_pos = sum(int(float(r["label"])) for r in rows)
    print(f"Candidates: {len(rows)} | camouflaged: {n_pos} | other: {len(rows) - n_pos}")
    if n_pos < 20:
        print("WARNING: very few positives - build the csv from ALL val captures for a reliable fit.")

    models = fit_from_candidates(rows)
    print(f"\n{'agent':9s}{'hyp':5s}{'mean':>8s}{'sd':>8s}{'P(reports)':>12s}")
    for agent in ("spatial", "spectral", "feature"):
        for h in HYPS:
            g = models[agent][h]
            print(f"{agent:9s}{h:5s}{g.mu:8.3f}{g.sd:8.3f}{models[PRESENCE_KEY][agent][h]:12.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(models_to_json(models), indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
